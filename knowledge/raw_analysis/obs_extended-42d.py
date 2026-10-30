    def get_compact_racing_obs(
        self,
        vehicle,
        action_applied=None,
        delta_steer: float = 0.0,
        on_runoff: bool = False,
    ) -> np.ndarray:
        """
        Xây dựng vector quan sát 40 chiều (Compact Racing State) - Zero Data Leakage:
        - 16 chiều Ego Kinematics, IMU & Controls:
          + [vx/40, vy/20, yaw_rate/pi, body_slip_beta]
          + [acc_long_g, acc_lat_g, roll/0.35, pitch/0.35]
          + [prev_steer, prev_throttle, prev_brake, delta_steer]
          + [runoff_flag, lat_offset/4.5, sin_dh, cos_dh]
        - 24 chiều Lookahead Horizon dọc tim đường (6 waypoints x 4 values):
          + [dx_rel/150, dy_rel/40, sin(d_theta), kappa*50] tại các mốc [10, 25, 45, 70, 105, 150]m
        """
        if action_applied is None:
            action_applied = self.last_action
            
        pos = np.array(vehicle.position[:2], dtype=np.float32)
        heading = float(vehicle.heading_theta)
        vel = np.array(vehicle.velocity[:2], dtype=np.float32)
        speed_m_s = float(np.linalg.norm(vel))
        
        # Tìm vị trí tim đường hiện tại
        idx, current_s = self._get_track_s(pos, self.current_track_idx)
        self.current_track_idx = idx
        track_pt = self.track_xy[idx]
        track_heading = float(self.track_theta[idx])
        
        # Hệ trục xe (Ego Body Frame)
        cos_head = np.cos(heading)
        sin_head = np.sin(heading)
        fwd_vec = np.array([cos_head, sin_head], dtype=np.float32)
        lat_vec = np.array([-sin_head, cos_head], dtype=np.float32)
        
        # Vận tốc dọc và ngang
        vx = float(np.dot(vel, fwd_vec))
        vy = float(np.dot(vel, lat_vec))
        
        # Góc lệch hướng so với tiếp tuyến tim đường
        delta_heading = (heading - track_heading + np.pi) % (2.0 * np.pi) - np.pi
        cos_dh = float(np.cos(delta_heading))
        sin_dh = float(np.sin(delta_heading))
        
        # Khoảng cách lệch ngang so với tim đường
        disp_center = pos - track_pt
        track_lat_normal = np.array([-np.sin(track_heading), np.cos(track_heading)], dtype=np.float32)
        lat_offset = float(np.dot(disp_center, track_lat_normal))
        
        # Yaw rate
        yaw_rate = float(getattr(vehicle, "yaw_rate", 0.0))
        
        # Góc trượt thân xe beta (body slip angle)
        if speed_m_s > 0.5:
            body_slip = float(np.arctan2(vy, max(0.1, vx)) / (np.pi / 4.0))
        else:
            body_slip = 0.0
        body_slip = float(np.clip(body_slip, -1.5, 1.5))
        
        # Góc roll và pitch từ Panda3D origin
        roll_val = 0.0
        pitch_val = 0.0
        if hasattr(vehicle, "origin") and vehicle.origin is not None:
            try:
                hpr = vehicle.origin.get_hpr()
                pitch_val = float(np.radians(hpr[1]))
                roll_val = float(np.radians(hpr[2]))
            except Exception:
                pass
                
        # Hành động bước trước
        act_arr = np.array(self.last_action, dtype=np.float32)
        prev_steer = float(act_arr[0]) if len(act_arr) > 0 else 0.0
        prev_throttle = float(act_arr[1]) if len(act_arr) > 1 else 0.0
        prev_brake = float(act_arr[2]) if len(act_arr) > 2 else 0.0
        
        # Khoảng cách tới mép nhựa trái và phải (mặt đường rộng 9m -> mép tại +-4.5m)
        dist_left = float(np.clip((4.5 - lat_offset) / 4.5, -2.0, 2.0))
        dist_right = float(np.clip((4.5 + lat_offset) / 4.5, -2.0, 2.0))
        
        # --- 18 CHIỀU EGO KINEMATICS, IMU & CONTROLS ---
        ego_state = [
            float(np.clip(vx / 40.0, -1.0, 1.5)),               # 1. Vận tốc dọc thân xe
            float(np.clip(vy / 20.0, -1.5, 1.5)),               # 2. Vận tốc ngang thân xe
            float(np.clip(yaw_rate / 3.14159, -2.0, 2.0)),      # 3. Yaw rate quanh trục Z
            float(body_slip),                                   # 4. Góc trượt thân xe beta
            float(np.clip(self.last_acc_long_g, -2.5, 2.5)),    # 5. Gia tốc dọc IMU (G)
            float(np.clip(self.last_acc_lat_g, -2.5, 2.5)),     # 6. Gia tốc ngang IMU (G)
            float(np.clip(roll_val / 0.35, -2.0, 2.0)),         # 7. Góc nghiêng thân xe roll
            float(np.clip(pitch_val / 0.35, -2.0, 2.0)),        # 8. Góc chúc/ngóc đầu pitch
            float(np.clip(prev_steer, -1.0, 1.0)),              # 9. Vô lăng bước trước
            float(np.clip(prev_throttle, -1.0, 1.0)),           # 10. Chân ga bước trước
            float(np.clip(prev_brake, -1.0, 1.0)),              # 11. Chân phanh bước trước
            float(np.clip(delta_steer, -1.0, 1.0)),             # 12. Tốc độ xoay vô lăng
            float(1.0 if on_runoff else 0.0),                   # 13. Cờ lấn lề Runoff
            float(np.clip(lat_offset / 4.5, -2.0, 2.0)),        # 14. Độ lệch ngang so với tim đường
            float(sin_dh),                                      # 15. Sin góc lệch hướng
            float(cos_dh),                                      # 16. Cos góc lệch hướng
            dist_left,                                          # 17. Khoảng cách mép nhựa trái
            dist_right,                                         # 18. Khoảng cách mép nhựa phải
        ]
        
        # --- 24 CHIỀU LOOKAHEAD HORIZON (6 WAYPOINTS X 4) ---
        lookahead_distances = [10.0, 25.0, 45.0, 70.0, 105.0, 150.0]
        lookahead_features = []
        for dist_ahead in lookahead_distances:
            target_s = (current_s + dist_ahead) % self.circuit_length
            wpt_idx = int(np.searchsorted(self.track_s, target_s)) % self.num_track_points
            wpt_pos = self.track_xy[wpt_idx]
            wpt_theta = float(self.track_theta[wpt_idx])
            wpt_kappa = float(self.track_kappa[wpt_idx])
            
            d_vec = wpt_pos - pos
            dx_rel = float(np.dot(d_vec, fwd_vec))
            dy_rel = float(np.dot(d_vec, lat_vec))
            d_theta = (wpt_theta - heading + np.pi) % (2.0 * np.pi) - np.pi
            
            lookahead_features.extend([
                float(np.clip(dx_rel / 150.0, -1.0, 1.5)),
                float(np.clip(dy_rel / 40.0, -2.0, 2.0)),
                float(np.sin(d_theta)),
                float(np.clip(wpt_kappa * 50.0, -2.0, 2.0)),
            ])
            
        obs = np.array(ego_state + lookahead_features, dtype=np.float32)
        return obs