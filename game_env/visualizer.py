"""
OOP 3D VISUALIZER MODULE (Panda3D)
Quản lý các đối tượng đồ họa trực quan 3D thời gian thực trên đường đua MetaDrive:
- Racing Line 3D: Đường đua lý thuyết tối ưu đa chiều (Heilmeier Convex QP)
- Hỗ trợ bật/tắt (Toggle) tương tác trực tiếp bằng bàn phím thông qua Panda3D Event Handling
- Hoàn toàn tuân thủ kiến trúc OOP và không sửa đổi thư viện gốc (MetaDrive / Panda3D).
"""

from typing import Optional, Any
import numpy as np

class RacingLineVisualizer:
    """
    Trực quan hóa quỹ đạo đua tối ưu 3D (Racing Line) trong không gian Panda3D của MetaDrive.
    - Vẽ đường polyline khép kín nổi bật (Neon Green/Cyan) ở cao độ z an toàn (tránh z-fighting).
    - Hỗ trợ bật/tắt (toggle) động trong thời gian thực bằng phím tắt [L].
    """
    def __init__(
        self,
        engine: Any,
        points_xy: np.ndarray,
        z: float = 0.15,
        thickness: float = 3.5,
        visible: bool = True,
    ):
        self.engine = engine
        self.points_xy = points_xy
        self.z = z
        self.thickness = thickness
        self.visible = visible
        self.node_path = None
        
        self._build_node()
        self._register_keyboard_events()

    def _build_node(self):
        if self.points_xy is None or len(self.points_xy) < 2:
            return
        try:
            from panda3d.core import LineSegs, NodePath
            lines = LineSegs("optimal_racing_line_visualizer")
            lines.setThickness(self.thickness)
            # Màu xanh neon huỳnh quang (Fluorescent Green-Cyan) RGBA
            lines.setColor(0.1, 1.0, 0.45, 0.95)

            N = len(self.points_xy)
            lines.moveTo(float(self.points_xy[0, 0]), float(self.points_xy[0, 1]), self.z)
            for i in range(1, N):
                lines.drawTo(float(self.points_xy[i, 0]), float(self.points_xy[i, 1]), self.z)
            # Khép kín vòng đua
            lines.drawTo(float(self.points_xy[0, 0]), float(self.points_xy[0, 1]), self.z)

            geom = lines.create()
            self.node_path = NodePath(geom)
            if hasattr(self.engine, "render") and self.engine.render is not None:
                self.node_path.reparentTo(self.engine.render)
                if not self.visible:
                    self.node_path.hide()
        except Exception as e:
            print(f">> [VISUALIZER] Không thể tạo 3D Racing Line: {e}")

    def _register_keyboard_events(self):
        try:
            if hasattr(self.engine, "accept"):
                self.engine.accept("l", self.toggle)
                self.engine.accept("L", self.toggle)
        except Exception as e:
            print(f">> [VISUALIZER] Không thể gán phím tắt [L]: {e}")

    def toggle(self):
        """Bật / Tắt hiển thị Racing Line trên không gian 3D"""
        if self.node_path is None:
            return
        self.visible = not self.visible
        if self.visible:
            self.node_path.show()
            try:
                print(">> [3D RACING LINE] Trang thai: BAT [ON]")
            except Exception:
                pass
        else:
            self.node_path.hide()
            try:
                print(">> [3D RACING LINE] Trang thai: TAT [OFF]")
            except Exception:
                pass

    def destroy(self):
        """Hủy node đồ họa an toàn khi kết thúc mô phỏng"""
        if self.node_path is not None:
            try:
                self.node_path.removeNode()
            except Exception:
                pass
            self.node_path = None
