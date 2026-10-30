    def get_compact_racing_obs(self, vehicle, action_applied=None, on_runoff=False):
        """
        Xây dựng vector quan sát 35 chiều (Compact Racing State):
        - 10 chiều Ego Kinematics & Controls
        - 5 chiều Track Limits & Progress
        - 20 chiều Lookahead Horizon (5 waypoints x 4 values)
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
        track_curvature = float(self.track_kappa[idx])
        
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
        
        # Khoảng cách tới mép đường trái và phải (đường rộng 9m -> mép tại +-4.5m)
        dist_left = float(np.clip((4.5 - lat_offset) / 4.5, -2.0, 2.0))
        dist_right = float(np.clip((4.5 + lat_offset) / 4.5, -2.0, 2.0))
        
        # Yaw rate
        yaw_rate = float(getattr(vehicle, "yaw_rate", 0.0))
        
        # Xử lý chân ga/phanh bước trước hỗ trợ cả 2D và 3D
        if len(self.last_action) >= 3:
            t = 0.5 * (float(self.last_action[1]) + 1.0)
            b = 0.5 * (float(self.last_action[2]) + 1.0)
            last_pedal = t * (1.0 - b) - b if b > 0.05 else t
        else:
            last_pedal = float(self.last_action[1])
            
        cur_steer = float(action_applied[0])
        
        # --- 10 CHIỀU EGO KINEMATICS ---
        ego_state = [
            float(np.clip(speed_m_s / 40.0, 0.0, 1.5)),          # Vận tốc chuẩn hóa
            float(np.clip(vx / 40.0, -1.0, 1.5)),                # Vận tốc dọc
            float(np.clip(vy / 20.0, -1.5, 1.5)),                # Vận tốc trượt ngang
            float(np.clip(yaw_rate / 3.14159, -2.0, 2.0)),       # Yaw rate
            float(cur_steer),                                    # Góc lái hiện tại
            float(self.last_action[0]),                          # Góc lái bước trước
            float(np.clip(last_pedal, -1.0, 1.0)),               # Ga/phanh bước trước
            float(cos_dh),                                       # Cos góc lệch hướng
            float(sin_dh),                                       # Sin góc lệch hướng
            float(np.clip(lat_offset / 4.5, -2.0, 2.0)),         # Độ lệch ngang tim đường
        ]
        
        # --- 5 CHIỀU TRACK LIMITS & TIẾN TRÌNH ---
        progress_ratio = float(np.clip(current_s / self.circuit_length, 0.0, 1.0))
        track_info = [
            dist_left,
            dist_right,
            1.0 if on_runoff else 0.0,
            progress_ratio,
            float(np.clip(track_curvature * 50.0, -2.0, 2.0)),  # Độ cong chuẩn hóa
        ]
        
        # --- 20 CHIỀU LOOKAHEAD HORIZON (5 WAYPOINTS X 4) ---
        lookahead_distances = [10.0, 20.0, 35.0, 50.0, 70.0]
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
                float(np.clip(dx_rel / 70.0, -1.0, 1.5)),
                float(np.clip(dy_rel / 30.0, -2.0, 2.0)),
                float(np.sin(d_theta)),
                float(np.clip(wpt_kappa * 50.0, -2.0, 2.0)),
            ])
            
        obs = np.array(ego_state + track_info + lookahead_features, dtype=np.float32)
        return obs