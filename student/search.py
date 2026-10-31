"""Thuật toán tìm kiếm — file của Vai 3 (Search).

GIAO ƯỚC CHUNG CỦA NHÓM (không tự ý đổi chữ ký hàm):
    best_move(board, legal_moves, deadline) -> Move
    - board: numpy 4x4 (chỉ đọc, không được sửa).
    - legal_moves: tuple các Move hợp lệ (đã có sẵn từ obs.legal_moves).
    - deadline: mốc thời gian theo time.perf_counter() mà việc tìm kiếm phải
      kết thúc trước đó.
    - PHẢI luôn trả về một phần tử của legal_moves, kể cả khi deadline đã qua.

    reset(seed) -> None: xóa cache/trạng thái giữa hai ván (agent.py gọi hàm này).

    Hai biến last_nodes và last_completed_depth được benchmark của thầy đọc để
    thống kê: hãy cập nhật chúng sau mỗi lần tìm kiếm.

HIỆN TẠI: bản tạm, nhìn trước 1 nước giống hệt starter của thầy. Hãy thay bằng
minimax + alpha-beta, rồi expectimax, iterative deepening...
"""

from __future__ import annotations

import numpy as np

from game2048.rules import move_board
from game2048.types import Move

from .heuristic import evaluate

last_nodes: int = 0
last_completed_depth: int = 0


def reset(seed: int | None = None) -> None:
    global last_nodes, last_completed_depth
    last_nodes = 0
    last_completed_depth = 0


def best_move(board: np.ndarray, legal_moves: tuple[Move, ...], deadline: float) -> Move:
    global last_nodes, last_completed_depth

    best, best_value = legal_moves[0], float("-inf")
    nodes = 0
    for move in legal_moves:
        next_board, merge_score, _changed = move_board(board, move)
        value = merge_score + evaluate(next_board)
        nodes += 1
        if value > best_value:
            best_value, best = value, move

    last_nodes = nodes
    last_completed_depth = 1
    return best
