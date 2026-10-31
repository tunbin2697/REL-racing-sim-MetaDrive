"""
Bộ điều khiển tay cầm Xbox (XInput) cho xe đua 3 trục độc lập [steer, throttle, brake]
- Joystick trái (trục X): Đánh lái (steer)
- Cò phải RT: Chân ga (throttle)
- Cò trái LT: Chân phanh (brake)
Dùng SDL GameController của pygame (chạy được trên Windows, khác XboxController của MetaDrive chỉ hỗ trợ Linux/Mac).
Hành động trả về cùng định dạng với PPO Agent, đi qua đúng cơ cấu chấp hành GT3RaceCar.process_action.
"""

import os
from typing import Optional
import numpy as np

# Panda3D giữ focus cửa sổ game -> cho phép SDL đọc tay cầm khi cửa sổ pygame không được focus
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from pygame._sdl2 import controller as sdl_controller


class XboxGamepad:
    AXIS_MAX = 32767.0

    def __init__(self, index: int = 0, steer_deadzone: float = 0.08, trigger_deadzone: float = 0.03, steer_gamma: float = 1.0):
        """
        steer_deadzone: vùng chết cần lái (loại bỏ trôi tâm joystick)
        trigger_deadzone: vùng chết cò ga/phanh (cò nhả hoàn toàn đọc ~0)
        steer_gamma: đường cong lái (1.0 tuyến tính; > 1.0 mềm hơn quanh tâm, chính xác hơn ở tốc độ cao)
        """
        self.steer_deadzone = float(steer_deadzone)
        self.trigger_deadzone = float(trigger_deadzone)
        self.steer_gamma = float(steer_gamma)
        self.pad: Optional[sdl_controller.Controller] = None

        pygame.display.init()  # event.pump yêu cầu video subsystem (không mở cửa sổ)
        sdl_controller.init()
        for i in range(pygame.joystick.get_count()):
            if sdl_controller.is_controller(i):
                if index == 0:
                    self.pad = sdl_controller.Controller(i)
                    break
                index -= 1
        if self.pad is None:
            raise RuntimeError("Không tìm thấy tay cầm Xbox. Hãy cắm dây / kết nối Bluetooth rồi chạy lại.")
        self.name = self.pad.name

    def _apply_deadzone(self, value: float, deadzone: float) -> float:
        """Vùng chết có co giãn lại để không bị nhảy bậc ngay sau vùng chết"""
        mag = abs(value)
        if mag <= deadzone:
            return 0.0
        return float(np.sign(value) * (mag - deadzone) / (1.0 - deadzone))

    def read(self) -> np.ndarray:
        """Đọc tay cầm, trả về action [steer, throttle, brake] trong không gian [-1, 1] của MetaRacingEnv"""
        pygame.event.pump()

        # MetaDrive: steer dương = rẽ trái; joystick đẩy trái cho trục X âm -> đảo dấu
        raw_x = self.pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX) / self.AXIS_MAX
        steer = -self._apply_deadzone(float(np.clip(raw_x, -1.0, 1.0)), self.steer_deadzone)
        steer = float(np.sign(steer) * abs(steer) ** self.steer_gamma)

        # Cò SDL GameController: 0 (nhả) -> 32767 (bóp hết)
        throttle = self._apply_deadzone(self.pad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT) / self.AXIS_MAX, self.trigger_deadzone)
        brake = self._apply_deadzone(self.pad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT) / self.AXIS_MAX, self.trigger_deadzone)
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        # Mã hóa giống PPO / OptimalRacingAgent: throttle, brake [0, 1] -> [-1, 1]
        return np.array([steer, 2.0 * throttle - 1.0, 2.0 * brake - 1.0], dtype=np.float32)

    def close(self):
        if self.pad is not None:
            self.pad.quit()
            self.pad = None
