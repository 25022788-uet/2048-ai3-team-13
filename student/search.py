"""Thuật toán tìm kiếm — file của trưởng nhóm (Search).

GIAO ƯỚC CHUNG CỦA NHÓM (không tự ý đổi chữ ký hàm):
    best_move(board, legal_moves, deadline) -> Move
    - board: numpy 4x4 (chỉ đọc, không được sửa).
    - legal_moves: tuple các Move hợp lệ (đã có sẵn từ obs.legal_moves).
    - deadline: mốc time.perf_counter() mà việc tìm kiếm phải kết thúc trước đó.
    - PHẢI luôn trả về một phần tử của legal_moves, kể cả khi deadline đã qua.

    reset(seed) -> None: xóa cache/trạng thái giữa hai ván (agent.py gọi hàm này).

    last_nodes và last_completed_depth được benchmark của thầy đọc để thống kê.

BƯỚC B: Expectimax + ITERATIVE DEEPENING (tăng dần độ sâu theo thời gian).
    Tìm độ sâu 1, rồi 2, rồi 3, ... cho đến khi hết `deadline`. Tầng đang làm dở
    khi hết giờ bị BỎ; ta dùng nước của tầng sâu nhất đã làm XONG.
    Tầng 1 không bao giờ bị ngắt, nên luôn có một nước hợp lệ để trả về.

BƯỚC C2 (hiện tại): ở các tầng CHANCE SÂU chỉ sinh ô 2 để search nhanh hơn.
    Nút CHANCE đầu tiên (ngay sau nước đi ở gốc) vẫn xét cả ô 2 lẫn ô 4.
    Từ nút CHANCE thứ hai trở đi chỉ xét ô 2 và CHUẨN HÓA xác suất 0,9 -> 1,0
    (tức tính kỳ vọng với giả định "ô mới luôn là 2"). Mỗi nút CHANCE sâu có một
    nửa số nhánh nên độ sâu 3 nhanh ~2 lần, độ sâu 4 nhanh ~4 lần.

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

# Độ sâu tối đa (số nước đi của MÌNH được nhìn trước). Thường thì hết giờ trước khi
# chạm mốc này; test dùng hằng số này để cố định độ sâu.
MAX_DEPTH = 8

# Điểm phạt khi bàn hết nước đi (thua). Rất âm để search tránh xa.
DEAD_PENALTY = -1_000_000.0

# Xác suất game sinh ô 2 / ô 4 (theo đề). Dùng ở nút CHANCE ĐẦU TIÊN.
SPAWN_TILES = ((2, 0.9), (4, 0.1))

# Ở nút CHANCE sâu (thứ 2 trở đi) chỉ xét ô 2, xác suất đã chuẩn hóa: 0,9 / 0,9 = 1,0.
# Muốn tắt C2 (để so sánh) chỉ cần gán DEEP_SPAWN_TILES = SPAWN_TILES.
DEEP_SPAWN_TILES = ((2, 1.0),)

# Hai biến benchmark của thầy đọc.
last_nodes: int = 0
last_completed_depth: int = 0

# Trạng thái trong lúc tìm. Một "nút" = một lần gọi move_board
# (vì move_board là phần tốn thời gian nhất).
_nodes: int = 0
_deadline: float = 0.0       # mốc time.perf_counter() phải dừng


class _Timeout(Exception):
    """Ném ra khi hết giờ giữa chừng để thoát khỏi đệ quy sâu; best_move sẽ bắt."""


def _tick() -> None:
    """Đếm một nút; cứ mỗi 256 nút mới nhìn đồng hồ (gọi perf_counter mỗi nút thì quá chậm).

    Tầng 1 chỉ có tối đa 4 nút (< 256) nên không bao giờ chạm mốc kiểm tra:
    tầng 1 tự động không bị ngắt, luôn có nước hợp lệ để trả về.
    """
    global _nodes
    _nodes += 1
    if (_nodes & 255) == 0 and time.perf_counter() > _deadline:
        raise _Timeout


def reset(seed: int | None = None) -> None:
    """Gọi đầu mỗi ván. Hiện chưa có cache nên chỉ đặt lại thống kê."""
    global last_nodes, last_completed_depth
    last_nodes = 0
    last_completed_depth = 0


def _max_value(board: np.ndarray, depth: int, level: int = 1) -> float:
    """Nút MAX: mình đi. Trả về giá trị của nước đi tốt nhất.

    depth = số nước đi của mình còn được phép đi (luôn >= 1 khi hàm này được gọi).
    level = số nút CHANCE đã đi qua trên đường từ gốc tới đây (để biết nút CHANCE
            kế tiếp là thứ mấy).
    """
    best = float("-inf")
    for move in Move:                      # duyệt 4 hướng UP, DOWN, LEFT, RIGHT
        next_board, merge_score, changed = move_board(board, move)
        _tick()
        if not changed:                    # nước không làm bàn đổi -> bất hợp lệ
            continue
        # Giá trị = điểm gộp ô nhận ngay + giá trị kỳ vọng của bàn sau đó.
        value = merge_score + _chance_value(next_board, depth - 1, level + 1)
        if value > best:
            best = value
    if best == float("-inf"):              # không có nước nào đi được -> thua
        return DEAD_PENALTY
    return best


def _chance_value(board: np.ndarray, depth: int, level: int = 1) -> float:
    """Nút CHANCE: game sinh ô mới. Trả về GIÁ TRỊ TRUNG BÌNH.

    depth = số nước đi của mình còn lại SAU khi ô được sinh.
    level = thứ tự của nút CHANCE này (1 = ngay sau nước đi ở gốc).
    Nếu depth == 0 thì dừng ở đây và chấm điểm luôn (nút lá).
    """
    if depth == 0:
        return evaluate(board)

    empty_cells = np.argwhere(board == 0)  # mảng các cặp [hàng, cột] của ô trống
    # C2: nút CHANCE đầu xét cả ô 2 và 4; các nút sâu hơn chỉ xét ô 2.
    spawn_tiles = SPAWN_TILES if level == 1 else DEEP_SPAWN_TILES

    total = 0.0
    for row, col in empty_cells:           # mỗi ô trống có xác suất như nhau
        for tile, prob in spawn_tiles:
            child = board.copy()           # copy để KHÔNG sửa bàn gốc
            child[row, col] = tile
            total += prob * _max_value(child, depth, level)
    return total / len(empty_cells)        # chia cho số ô trống = lấy trung bình


def _search_root(board: np.ndarray, legal_moves: tuple[Move, ...], depth: int) -> Move:
    """Một lượt Expectimax cố định `depth` từ gốc; trả về nước tốt nhất."""
    best, best_value = legal_moves[0], float("-inf")
    for move in legal_moves:               # nút gốc: thử từng nước hợp lệ
        next_board, merge_score, _changed = move_board(board, move)
        _tick()
        value = merge_score + _chance_value(next_board, depth - 1, 1)   # nút CHANCE thứ 1
        if value > best_value:
            best_value, best = value, move
    return best


def best_move(board: np.ndarray, legal_moves: tuple[Move, ...], deadline: float) -> Move:
    """Iterative deepening: thử độ sâu 1, 2, 3, ... đến khi hết `deadline` hoặc chạm MAX_DEPTH."""
    global last_nodes, last_completed_depth, _nodes, _deadline
    _nodes = 0
    _deadline = deadline

    best = legal_moves[0]                  # phương án dự phòng, luôn hợp lệ
    completed = 0
    for depth in range(1, MAX_DEPTH + 1):
        try:
            best = _search_root(board, legal_moves, depth)
        except _Timeout:
            break                          # tầng dở dang bị bỏ, giữ kết quả tầng trước
        completed = depth

    last_nodes = _nodes
    last_completed_depth = completed
    return best
