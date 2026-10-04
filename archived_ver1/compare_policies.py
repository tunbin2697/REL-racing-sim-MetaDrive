"""
compare_policies.py: Entry point ở thư mục gốc để chạy benchmark so sánh các Policy đua xe F1.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import argparse
from evaluation.compare_policies import run_policy_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm đối đầu trực tiếp giữa các Policy (Centerline vs Racing Line).")
    parser.add_argument("--track", type=str, default="Silverstone", help="Tên mạch đua: Silverstone, Monza, Spa, Suzuka")
    parser.add_argument("--max_steps", type=int, default=4000, help="Số bước tối đa mỗi vòng")
    args = parser.parse_args()

    run_policy_benchmark(track_name=args.track, max_steps=args.max_steps)
