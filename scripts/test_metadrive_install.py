"""
Script kiểm tra cài đặt MetaDrive Simulator & Panda3D Engine
Cho phép người dùng trực tiếp điều khiển xe bằng bàn phím (W, A, S, D) trong không gian 3D.

Cách chạy:
    C:\\Users\\tunbi\\miniconda3\\envs\\rel_racing\\python.exe scripts/test_metadrive_install.py
hoặc:
    conda activate rel_racing
    python scripts/test_metadrive_install.py
"""

import sys
import time
from metadrive import MetaDriveEnv

def main():
    print("=" * 60)
    print("🚗 KHỞI TẠO METADRIVE 3D SIMULATOR - KIỂM THỬ ĐỒ HOẠ & VẬT LÝ")
    print("=" * 60)
    print("Phím điều khiển:")
    print("  - [W] hoặc [Mũi tên Lên]: Tăng ga (Throttle)")
    print("  - [S] hoặc [Mũi tên Xuống]: Thắng/Lùi (Brake/Reverse)")
    print("  - [A] hoặc [Mũi tên Trái]: Đánh lái trái (Steer Left)")
    print("  - [D] hoặc [Mũi tên Phải]: Đánh lái phải (Steer Right)")
    print("  - [C]: Đổi góc nhìn Camera (Chase cam / Top-down / On-board)")
    print("  - [B]: Góc nhìn toàn cảnh từ trên cao (Bird-eye View)")
    print("  - [R]: Reset vòng đua mới")
    print("  - [H]: Bật/tắt menu trợ giúp trên màn hình")
    print("  - [Esc]: Thoát")
    print("=" * 60)

    config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=0.08,        # Có một lượng nhỏ xe cộ để kiểm tra va chạm
        start_seed=42,
        map=3,                      # Bản đồ gồm 3 khối đường (thẳng, cua cong)
        window_size=(1280, 720),
        physics_world_step_size=0.02,
        show_fps=True,
    )

    try:
        env = MetaDriveEnv(config)
        obs, info = env.reset()
        print(">> Đã khởi động Panda3D Engine thành công! Hãy click vào cửa sổ game để lái xe.")
        
        step_count = 0
        while True:
            # Khi bật manual_control=True, action truyền vào sẽ tự động được ghi đè bằng phím bấm
            obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
            step_count += 1
            
            if terminated or truncated:
                print(f">> Vòng đua kết thúc tại step {step_count} (Crash: {info.get('crash', False)}, Arrive: {info.get('arrive_dest', False)}). Đang reset...")
                obs, info = env.reset()
                step_count = 0

    except KeyboardInterrupt:
        print("\n>> Dừng mô phỏng theo yêu cầu của người dùng.")
    except Exception as e:
        print(f"\n[LỖI] Xảy ra ngoại lệ: {e}")
    finally:
        if 'env' in locals():
            env.close()
            print(">> Đã giải phóng môi trường MetaDrive an toàn.")

if __name__ == "__main__":
    main()
