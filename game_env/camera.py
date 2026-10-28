"""
RacingCameraController: Quản lý Hệ Thống Camera Mô Phỏng 3D Tự Do (360 deg Free Mouse Look Sim Racing Camera)
Áp dụng nguyên lý Single Responsibility (SRP) và Lập trình Hướng đối tượng (OOP):
- Mở khóa hoàn toàn giới hạn xoay chuột 360 độ (bỏ giới hạn góc quay và bỏ cơ chế tự động reset về tâm).
- Người dùng có thể tự do lia chuột nhìn quanh xe 360 deg, xoay ra phía sau 180 deg để xem xe bám đuôi (Ghost Car) bao lâu tùy thích mà không bị giật về giữa.
- Hỗ trợ phím tắt và thao tác tiện lợi:
  1. Lia chuột (Mouse Look): Xoay tự do 360 deg quanh xe mà không bị giới hạn góc.
  2. Chuột giữa (Middle-Click): Reset tức thì góc nhìn về chính diện (0 deg).
  3. Phím [G]: Đổi xe theo dõi (Bám sau Ghost Car <-> Bám sau Main Car).
- Tích hợp 100% với hệ thống nội suy camera gốc của MetaDrive mà không sửa đổi file thư viện.
"""

import types
from typing import Optional, Any
import numpy as np


class RacingCameraController:
    """
    Bộ điều khiển Camera 360° Free Mouse Look & Spectator Tracker.
    """
    def __init__(
        self,
        engine: Any,
        main_vehicle: Any,
        ghost_car: Optional[Any] = None,
        mouse_sensitivity: float = 3.2,
    ):
        self.engine = engine
        self.main_vehicle = main_vehicle
        self.ghost_car = ghost_car
        self.mouse_sensitivity = mouse_sensitivity
        
        # Trạng thái xe đang theo dõi: "main" hoặc "ghost"
        self.target_name = "main"
        
        self.main_camera = getattr(self.engine, "main_camera", None)
        self._orig_update_mouse_info = None
        
        # Mở khóa xoay chuột 360 độ trên main_camera
        self._setup_free_mouse_look()
        self._register_events()

    def _setup_free_mouse_look(self):
        """
        Thay thế cơ chế giới hạn góc và tự reset của MetaDrive bằng 360° Free Look
        """
        if self.main_camera is None:
            return
            
        self._orig_update_mouse_info = getattr(self.main_camera, "update_mouse_info", None)
        
        def free_update_mouse_info(cam_self):
            # Nếu không có chuột hoặc cửa sổ không active thì bỏ qua
            if not getattr(cam_self, "has_mouse", False):
                return
            mwn = getattr(cam_self.engine, "mouseWatcherNode", None)
            if mwn is None or not mwn.hasMouse():
                return

            cur_x = mwn.getMouseX()
            
            # Khởi tạo vị trí chuột mốc
            if not hasattr(cam_self, "_last_free_mouse_x") or cam_self._last_free_mouse_x is None:
                cam_self._last_free_mouse_x = cur_x
                return

            dx = cur_x - cam_self._last_free_mouse_x

            # Nếu chuột di chuyển gần mép cửa sổ (|cur_x| > 0.80), đưa con trỏ về tâm
            # để người dùng có thể tiếp tục lia chuột vô tận mà không bị chặn mép màn hình
            if abs(cur_x) > 0.80:
                cam_self.set_mouse_to_center()
                cam_self._last_free_mouse_x = 0.0
            else:
                cam_self._last_free_mouse_x = cur_x

            # Cộng dồn góc xoay theo delta lia chuột
            if abs(dx) > 0.001:
                cam_self.mouse_rotate += -dx * self.mouse_sensitivity
                # Chuẩn hóa về dải [-pi, pi]
                cam_self.mouse_rotate = (cam_self.mouse_rotate + np.pi) % (2.0 * np.pi) - np.pi

            # TUYỆT ĐỐI KHÔNG auto-reset: Giữ nguyên góc nhìn mà người dùng đã lia chuột tới!

        self.main_camera.update_mouse_info = types.MethodType(free_update_mouse_info, self.main_camera)

    def _register_events(self):
        """Đăng ký các phím bấm và nút chuột tương tác"""
        if not hasattr(self.engine, "accept"):
            return
            
        # Phím [G]: Chuyển camera bám theo Ghost Car <-> Main Car
        self.engine.accept("g", self.toggle_target)
        self.engine.accept("G", self.toggle_target)
        
        # Chuột giữa (mouse2): Reset nhanh góc nhìn về thẳng phía trước (0 deg)
        self.engine.accept("mouse2", self.reset_to_front)
        
        # Cập nhật bảng trợ giúp khi bấm [H] trên màn hình Panda3D
        self._update_onscreen_help()

    def _unregister_events(self):
        """Hủy đăng ký sự kiện khi kết thúc"""
        if not hasattr(self.engine, "ignore"):
            return
        for key in ["g", "G", "mouse2"]:
            try:
                self.engine.ignore(key)
            except Exception:
                pass

    def _update_onscreen_help(self):
        """Ghi đè HELP_MESSAGE của MetaDrive để bảng trợ giúp [H] hiển thị chính xác các phím sim racing"""
        try:
            import metadrive.engine.core.onscreen_message as osm
            osm.HELP_MESSAGE = (
                "Sim Racing Shortcuts & Controls:\n"
                "  Mouse Look: Free 360 Camera Rotation (No limits)\n"
                "  Middle-Click: Reset Camera Angle to Center (0 deg)\n"
                "  G: Switch Camera Target (Ghost Car <-> Main Car)\n"
                "  L: Toggle 3D Optimal Racing Line\n"
                "  C: Capture Screenshot (.png in project root)\n"
                "  B: Top-down View Camera\n"
                "  Q: Third-person Chase Camera\n"
                "  R: Reset Car to Start Line\n"
                "  H: Toggle this Help Overlay\n"
                "  W/A/S/D: Manual Throttle / Brake / Steer\n"
                "  Esc: Exit Simulation\n"
            )
        except Exception:
            pass

    def reset_to_front(self):
        """Reset góc xoay camera về thẳng phía trước (0 deg)"""
        if self.main_camera is not None:
            self.main_camera.mouse_rotate = 0.0
            print(">> [CAMERA] Đã reset góc nhìn về chính diện (0 deg).")

    def toggle_target(self):
        """Chuyển đổi xe theo dõi giữa Xe chính và Xe bóng ma (Ghost Car)"""
        if self.main_camera is None:
            return
            
        if self.target_name == "main":
            if self.ghost_car is not None and getattr(self.ghost_car, "active", False) and getattr(self.ghost_car, "raw_vehicle", None) is not None:
                self.target_name = "ghost"
                self.main_camera.track(self.ghost_car.raw_vehicle)
                print(">> [CAMERA] Chuyển mục tiêu sang theo dõi: [XE BÓNG MA - GHOST CAR]")
            else:
                print(">> [CAMERA] Ghost Car chưa xuất hiện hoặc chưa được kích hoạt!")
        else:
            self.target_name = "main"
            raw_main = getattr(self.main_vehicle, "raw_vehicle", self.main_vehicle)
            if raw_main is not None:
                self.main_camera.track(raw_main)
            print(">> [CAMERA] Chuyển mục tiêu sang theo dõi: [XE CHÍNH - MAIN CAR]")

    def on_reset(self, main_vehicle: Any = None, ghost_car: Optional[Any] = None):
        """Đồng bộ khi reset episode"""
        if main_vehicle is not None:
            self.main_vehicle = main_vehicle
        if ghost_car is not None:
            self.ghost_car = ghost_car
        if self.main_camera is not None:
            raw_target = getattr(self.ghost_car, "raw_vehicle", None) if self.target_name == "ghost" else getattr(self.main_vehicle, "raw_vehicle", self.main_vehicle)
            if raw_target is not None:
                self.main_camera.track(raw_target)

    def get_hud_status(self) -> str:
        """Thông tin trạng thái camera hiển thị trên Text HUD (tránh ký tự unicode lạ gây warning font)"""
        target_str = "GHOST" if self.target_name == "ghost" else "MAIN"
        cur_rot_deg = 0.0
        if self.main_camera is not None:
            cur_rot_deg = float(np.rad2deg(getattr(self.main_camera, "mouse_rotate", 0.0)))
        
        # Nhận diện hướng nhìn (dùng chữ deg thay vì ký tự unicode U+00b0)
        if abs(abs(cur_rot_deg) - 180.0) < 35.0:
            view_str = "NHIN SAU (180 deg)"
        elif abs(cur_rot_deg) < 35.0:
            view_str = "NHIN TRUOC (0 deg)"
        elif cur_rot_deg > 0:
            view_str = f"NHIN TRAI ({cur_rot_deg:+.0f} deg)"
        else:
            view_str = f"NHIN PHAI ({cur_rot_deg:+.0f} deg)"
            
        return f"[{view_str}] Target: {target_str} (Mouse look 360 | Chuot giua reset | [G] Ghost Cam)"

    def destroy(self):
        """Khôi phục lại method ban đầu và hủy đăng ký sự kiện"""
        self._unregister_events()
        if self.main_camera is not None and self._orig_update_mouse_info is not None:
            self.main_camera.update_mouse_info = self._orig_update_mouse_info
