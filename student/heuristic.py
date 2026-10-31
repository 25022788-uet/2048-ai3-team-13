"""Hàm đánh giá bàn cờ — file của Vai 2 (Heuristic & Tuning).

GIAO ƯỚC CHUNG CỦA NHÓM (không tự ý đổi chữ ký hàm):
    evaluate(board) -> float
    - board: numpy 4x4, mỗi ô là 0 (trống) hoặc giá trị ô (2, 4, 8, ...).
    - Trả về số float: càng LỚN thì bàn cờ càng TỐT.
    - Không được sửa board.
    - Hàm này bị gọi hàng nghìn lần mỗi nước đi, nên phải chạy nhanh.

HIỆN TẠI: bản tạm, sao chép công thức của starter (bỏ phần merge_score vì
search.py cộng riêng). Hãy thay bằng thiết kế của nhóm.
"""

from __future__ import annotations

import numpy as np

# Trọng số để ở đầu file: khi tuning chỉ cần đổi các hằng số này.
EMPTY_WEIGHT = 250.0
LARGEST_WEIGHT = 0.1

# TODO (Vai 2): thêm các tiêu chí như monotonicity, smoothness, ô lớn ở góc,
# tiềm năng gộp ô; rồi tối ưu tốc độ (ví dụ tính trước điểm cho từng hàng).


def evaluate(board: np.ndarray) -> float:
    empty_cells = int(np.count_nonzero(board == 0))
    largest_tile = int(board.max())
    return EMPTY_WEIGHT * empty_cells + LARGEST_WEIGHT * largest_tile
