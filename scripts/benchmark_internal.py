import argparse
import subprocess
import sys
from pathlib import Path
def run_benchmark():
    parser = argparse.ArgumentParser(description="Chạy benchmark nội bộ cho nhóm 13")
    parser.add_argument("--seeds", type=int, default=10, help="Số lượng ván đấu (seeds)")
    parser.add_argument("--start-seed", type=int, default=1000, help="Seed bắt đầu")
    parser.add_argument("--total-time-sec", type=float, default=60.0, help="Ngân sách thời gian mỗi ván (giây)")
    parser.add_argument("--name", type=str, default="benchmark_vai4_60s", help="Tên file kết quả")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"{args.name}.json"

    print("=" * 60)
    print(f"BẮT ĐẦU BENCHMARK NỘI BỘ VAI 4")
    print(f"- Số ván: {args.seeds} (từ seed {args.start_seed})")
    print(f"- Ngân sách: {args.total_time_sec} giây / ván")
    print(f"- File lưu kết quả: {output_path}")
    print("=" * 60)

    cmd = [
        sys.executable,
        "scripts/benchmark_agent.py",
        "--agent", "student.agent:Agent",
        "--seeds", str(args.seeds),
        "--start-seed", str(args.start_seed),
        "--total-time-sec", str(args.total_time_sec),
        "--output", str(output_path),
    ]
    subprocess.run(cmd)
    print("\n" + "=" * 60)
    print(f"HOÀN TẤT BENCHMARK!")
    print(f"Kết quả chi tiết đã được tự động lưu tại: {output_path}")
    print("=" * 60)
if __name__ == "__main__":
    run_benchmark()