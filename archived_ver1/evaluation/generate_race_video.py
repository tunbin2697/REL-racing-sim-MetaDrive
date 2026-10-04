"""
evaluation/generate_race_video.py: Kết xuất video MP4 tái hiện cuộc đua thực tế (kèm telemetry HUD).
Lưu video tại: plots/demo_{track_name.lower()}_race.mp4.
"""

import os
import sys
import json
import math
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def generate_video(track_name="Monza", output_filename=None):
    json_path = os.path.join(ROOT_DIR, "data", "telemetry_data.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(ROOT_DIR, "telemetry_data.json")
        
    print(f"Loading telemetry data for {track_name} from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)[track_name]

    if output_filename is None:
        plots_dir = os.path.join(ROOT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_filename = os.path.join(plots_dir, f"demo_{track_name.lower()}_race.mp4")

    track = data["track"]
    lap1 = data["laps"]["1"]
    lap3 = data["laps"]["3"]

    width, height = 1280, 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    xs = [p[0] for p in track["center"]]
    ys = [p[1] for p in track["center"]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    tw = max_x - min_x + 80
    th = max_y - min_y + 80
    scale = min((width - 320) / tw, (height - 120) / th) * 0.95
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    ox = (width - 240) / 2
    oy = height / 2

    def to_screen(gx, gy):
        sx = int(ox + (gx - cx) * scale)
        sy = int(oy - (gy - cy) * scale)
        return (sx, sy)

    center_screen = np.array([to_screen(p[0], p[1]) for p in track["center"]], dtype=np.int32)
    left_screen = np.array([to_screen(p[0], p[1]) for p in track["left"]], dtype=np.int32)
    right_screen = np.array([to_screen(p[0], p[1]) for p in track["right"]], dtype=np.int32)

    total_frames = max(len(lap1), len(lap3))
    print(f"Rendering {total_frames} frames to {output_filename}...")

    for f_idx in range(total_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (18, 15, 12)

        # Mặt đường
        for seg_i in range(len(left_screen)):
            next_i = (seg_i + 1) % len(left_screen)
            poly = np.array([
                left_screen[seg_i],
                left_screen[next_i],
                right_screen[next_i],
                right_screen[seg_i]
            ])
            cv2.fillPoly(img, [poly], (35, 30, 25))

        cv2.polylines(img, [left_screen], True, (120, 110, 100), 2, cv2.LINE_AA)
        cv2.polylines(img, [right_screen], True, (120, 110, 100), 2, cv2.LINE_AA)
        cv2.polylines(img, [center_screen], True, (55, 50, 45), 1, cv2.LINE_AA)

        if "racing_line" in track and len(track["racing_line"]) > 0:
            rl_screen = np.array([to_screen(p[0], p[1]) for p in track["racing_line"]], dtype=np.int32)
            cv2.polylines(img, [rl_screen], True, (40, 180, 220), 2, cv2.LINE_AA)

        # Ghost Car (Lap 1)
        if f_idx < len(lap1):
            g = lap1[f_idx]
            gx, gy = to_screen(g["x"], g["y"])
            cv2.circle(img, (gx, gy), 7, (40, 40, 220), -1, cv2.LINE_AA)
            cv2.circle(img, (gx, gy), 9, (80, 80, 250), 1, cv2.LINE_AA)

        # Main Car (Lap 3)
        if f_idx < len(lap3):
            c = lap3[f_idx]
            cx_scr, cy_scr = to_screen(c["x"], c["y"])
            cv2.circle(img, (cx_scr, cy_scr), 8, (220, 220, 40), -1, cv2.LINE_AA)
            cv2.circle(img, (cx_scr, cy_scr), 12, (255, 255, 100), 1, cv2.LINE_AA)

            # Telemetry HUD
            cv2.rectangle(img, (width - 240, 20), (width - 20, 220), (30, 25, 20), -1)
            cv2.rectangle(img, (width - 240, 20), (width - 20, 220), (80, 70, 60), 1)

            cv2.putText(img, f"F1 TELEMETRY: {track_name.upper()}", (width - 225, 45), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1)
            spd_kmh = c["v"] * 3.6
            cv2.putText(img, f"SPEED: {spd_kmh:5.1f} KM/H", (width - 225, 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 255, 100), 1)

            time_s = c["t"]
            cv2.putText(img, f"TIME:  {time_s:5.2f} S", (width - 225, 115), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Pedal bars
            accel = c["accel"]
            cv2.putText(img, "THROTTLE / BRAKE", (width - 225, 150), cv2.FONT_HERSHEY_DUPLEX, 0.4, (160, 160, 160), 1)
            if accel >= 0:
                cv2.rectangle(img, (width - 225, 160), (width - 225 + int(accel * 180), 175), (50, 200, 50), -1)
            else:
                cv2.rectangle(img, (width - 225, 160), (width - 225 + int(abs(accel) * 180), 175), (50, 50, 220), -1)

            cv2.putText(img, f"STEER: {c['steer']*57.3:+4.1f} DEG", (width - 225, 205), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1)

        out.write(img)

    out.release()
    print(f"[THÀNH CÔNG] Đã tạo video cuộc đua: {output_filename}")

if __name__ == "__main__":
    generate_video()
