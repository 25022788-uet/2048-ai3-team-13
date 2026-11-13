"""
Ý TƯỞNG EXPECTIMAX
    2048 có hai kiểu "nút" xen kẽ nhau:
      - Nút MAX   : đến lượt MÌNH đi -> chọn nước cho giá trị CAO NHẤT.
      - Nút CHANCE: game sinh ô ngẫu nhiên -> lấy TRUNG BÌNH có trọng số
                    (ô 2 với xác suất 0,9; ô 4 với 0,1; mỗi ô trống như nhau).
    Nút lá (hết độ sâu) thì gọi evaluate(board) để chấm điểm bàn cờ.
"""

from __future__ import annotations

import time

import numpy as np

from game2048.rules import move_board
from game2048.types import Move

from .heuristic import evaluate


MAX_DEPTH = 8


DEAD_PENALTY = -1_000_000.0


SPAWN_TILES = ((2, 0.9), (4, 0.1))

# Ở nút CHANCE sâu (thứ 2 trở đi) chỉ xét ô 2, xác suất đã chuẩn hóa: 0,9 / 0,9 = 1,0.
# Muốn tắt C2 (để so sánh) chỉ cần gán DEEP_SPAWN_TILES = SPAWN_TILES.
DEEP_SPAWN_TILES = ((2, 1.0),)


last_nodes: int = 0
last_completed_depth: int = 0


_nodes: int = 0
_deadline: float = 0.0      


class _Timeout(Exception):
    """Ném ra khi hết giờ giữa chừng để thoát khỏi đệ quy sâu; best_move sẽ bắt."""


def _tick() -> None:
    
    global _nodes
    _nodes += 1
    if (_nodes & 255) == 0 and time.perf_counter() > _deadline:
        raise _Timeout


def reset(seed: int | None = None) -> None:
  
    global last_nodes, last_completed_depth
    last_nodes = 0
    last_completed_depth = 0


def _max_value(board: np.ndarray, depth: int, level: int = 1) -> float:

    best = float("-inf")
    for move in Move:                     
        next_board, merge_score, changed = move_board(board, move)
        _tick()
        if not changed:                    
            continue
        value = merge_score + _chance_value(next_board, depth - 1, level + 1)
        if value > best:
            best = value
    if best == float("-inf"):              
        return DEAD_PENALTY
    return best


def _chance_value(board: np.ndarray, depth: int, level: int = 1) -> float:

    if depth == 0:
        return evaluate(board)

    empty_cells = np.argwhere(board == 0)  
    spawn_tiles = SPAWN_TILES if level == 1 else DEEP_SPAWN_TILES

    total = 0.0
    for row, col in empty_cells:          
        for tile, prob in spawn_tiles:
            child = board.copy()           
            child[row, col] = tile
            total += prob * _max_value(child, depth, level)
    return total / len(empty_cells)        


def _search_root(board: np.ndarray, legal_moves: tuple[Move, ...], depth: int) -> Move:

    best, best_value = legal_moves[0], float("-inf")
    for move in legal_moves:               # nút gốc: thử từng nước hợp lệ
        next_board, merge_score, _changed = move_board(board, move)
        _tick()
        value = merge_score + _chance_value(next_board, depth - 1, 1)   # nút CHANCE thứ 1
        if value > best_value:
            best_value, best = value, move
    return best


def best_move(board: np.ndarray, legal_moves: tuple[Move, ...], deadline: float) -> Move:

    global last_nodes, last_completed_depth, _nodes, _deadline
    _nodes = 0
    _deadline = deadline

    best = legal_moves[0]                 
    completed = 0
    for depth in range(1, MAX_DEPTH + 1):
        try:
            best = _search_root(board, legal_moves, depth)
        except _Timeout:
            break                         
        completed = depth

    last_nodes = _nodes
    last_completed_depth = completed
    return best
