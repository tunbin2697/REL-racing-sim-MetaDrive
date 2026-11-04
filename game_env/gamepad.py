"""
Xbox Gamepad Controller for Windows (pygame SDL2 XInput / DirectInput)
Hỗ trợ điều khiển mượt mà bằng tay cầm Xbox trên Windows:
- Left Stick X (Axis 0): Đánh lái mượt mà (Steering) có Deadzone.
- Right Trigger RT (Axis 5) hoặc Nút A: Tăng ga tuyến tính (Throttle [0.0, 1.0]).
- Left Trigger LT (Axis 4) hoặc Nút B/X: Lực phanh tuyến tính (Brake [0.0, 1.0]).
- Right Stick Y (Axis 3) đẩy lên / kéo xuống: Hỗ trợ ga / phanh bằng cần phụ.
- Tự động nhận diện và fallback an toàn về bàn phím nếu không có tay cầm.
"""

import os
from typing import List, Optional, Any, Tuple
import numpy as np

# Cho phép SDL đọc sự kiện khi cửa sổ Panda3D giữ focus
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class XboxGamepadController:
    """Bộ điều khiển tay cầm Xbox gọn nhẹ, độc lập cho Windows."""

    DEADZONE: float = 0.05
    STEER_EXPONENT: float = 1.5     # Đường cong phi tuyến: nhạy hơn ở vùng giữa (trước đây 1.8) để bù xe hơi understeer
    STEER_SENSITIVITY: float = 1.0   # Cho phép đánh lái full-lock (trước đây 0.85)
    INVERT_STEER: bool = True       # MetaDrive quy ước: rẽ trái là DƯƠNG (+), rẽ phải là ÂM (-)

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.joystick: Optional[Any] = None
        self.connected: bool = False
        self._rt_activated: bool = False
        self._lt_activated: bool = False
        self._init_device()

    def _init_device(self):
        if not PYGAME_AVAILABLE:
            self.connected = False
            return

        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        num_joysticks = pygame.joystick.get_count()
        if num_joysticks > self.device_id:
            try:
                self.joystick = pygame.joystick.Joystick(self.device_id)
                self.joystick.init()
                self.connected = True
                print(f">> [GAMEPAD] Đã kết nối: '{self.joystick.get_name()}' (ID: {self.device_id})")
            except Exception as e:
                print(f">> [GAMEPAD WARNING] Lỗi kết nối tay cầm: {e}")
                self.connected = False
        else:
            self.connected = False

    @classmethod
    def is_available(cls, device_id: int = 0) -> bool:
        """Kiểm tra nhanh xem có tay cầm nào đang cắm hay không."""
        if not PYGAME_AVAILABLE:
            return False
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        return pygame.joystick.get_count() > device_id

    def _normalize_trigger(self, raw_val: float, is_activated: bool) -> Tuple[float, bool]:
        """
        Pygame trên Windows: Khi mới bật, trigger có giá trị 0.0.
        Sau khi bấm lần đầu, dải giá trị chuyển thành [-1.0 (nhả), 1.0 (bấm hết)].
        """
        if not is_activated:
            if abs(raw_val) > 0.01:
                is_activated = True
                return max(0.0, float((raw_val + 1.0) / 2.0)), True
            return 0.0, False
        return max(0.0, float(np.clip((raw_val + 1.0) / 2.0, 0.0, 1.0))), True

    def process_input(self, vehicle=None) -> List[float]:
        """
        Đọc trạng thái tay cầm và trả về action chuẩn [steer, throttle, brake] in [-1, 1].
        """
        if not self.connected or self.joystick is None:
            return [0.0, -1.0, -1.0]

        try:
            pygame.event.pump()
        except Exception:
            pass

        num_axes = self.joystick.get_numaxes()
        num_buttons = self.joystick.get_numbuttons()

        # 1. Steering: Cần Analog Trái (Axis 0)
        # Trong Pygame SDL2: gạt cần sang Trái là Âm, sang Phải là Dương.
        # Trong MetaDrive: rẽ Trái là Dương (+), rẽ Phải là Âm (-).
        raw_axis = float(self.joystick.get_axis(0)) if num_axes > 0 else 0.0
        raw_steer = -raw_axis if self.INVERT_STEER else raw_axis

        if abs(raw_steer) < self.DEADZONE:
            steer = 0.0
        else:
            steer_norm = (abs(raw_steer) - self.DEADZONE) / (1.0 - self.DEADZONE)
            steer_curved = (steer_norm ** self.STEER_EXPONENT) * self.STEER_SENSITIVITY
            sign = 1.0 if raw_steer > 0 else -1.0
            steer = sign * steer_curved
        steer = float(np.clip(steer, -1.0, 1.0))

        # 2. Throttle (Ga) & Brake (Phanh) [0.0, 1.0]
        throttle = 0.0
        brake = 0.0

        if num_axes >= 6:
            raw_lt = self.joystick.get_axis(4)
            raw_rt = self.joystick.get_axis(5)
            brake, self._lt_activated = self._normalize_trigger(raw_lt, self._lt_activated)
            throttle, self._rt_activated = self._normalize_trigger(raw_rt, self._rt_activated)
        elif num_axes >= 3:
            raw_y = self.joystick.get_axis(3 if num_axes > 3 else 1)
            if abs(raw_y) > self.DEADZONE:
                if raw_y < 0:
                    throttle = min(1.0, abs(raw_y))
                else:
                    brake = min(1.0, raw_y)

        # Nút A (Ga), Nút B/X (Phanh)
        if num_buttons > 0 and self.joystick.get_button(0):
            throttle = max(throttle, 1.0)
        if num_buttons > 1 and self.joystick.get_button(1):
            brake = max(brake, 1.0)
        if num_buttons > 2 and self.joystick.get_button(2):
            brake = max(brake, 1.0)

        # Chuyển đổi [0, 1] sang [-1, 1] cho action 3 trục
        act_thr = 2.0 * float(throttle) - 1.0
        act_brk = 2.0 * float(brake) - 1.0

        return [steer, act_thr, act_brk]

    def is_respawn_pressed(self) -> bool:
        """Kiểm tra xem người chơi có bấm nút Y (Nút 3) hoặc Select/Back để hồi sinh xe không"""
        if not self.connected or self.joystick is None:
            return False
        try:
            pygame.event.pump()
            num_b = self.joystick.get_numbuttons()
            # Nút Y trên Xbox Controller thường là index 3, nút Back/View là 4 hoặc 6
            if num_b > 3 and self.joystick.get_button(3):
                return True
            if num_b > 6 and self.joystick.get_button(6):
                return True
            return False
        except Exception:
            return False

    def close(self):
        if self.joystick is not None:
            try:
                self.joystick.quit()
            except Exception:
                pass
            self.joystick = None
            self.connected = False
