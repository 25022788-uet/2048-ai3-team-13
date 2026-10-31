"""Điểm vào của bài nộp — chỉ TRƯỞNG NHÓM (Vai 1) sửa file này.

File này chỉ NỐI các module lại với nhau:
    agent.py -> time_manager.py  (nghĩ bao lâu)
    agent.py -> search.py        (chọn nước)  -> heuristic.py (chấm điểm bàn cờ)

Mọi người làm việc trong file của mình, không cần đụng vào đây.
Cần đổi cách nối thì nhắn trưởng nhóm.
"""

from __future__ import annotations

import time
import traceback

from game2048.observation import Observation
from game2048.types import Move

from . import search, time_manager


class Agent:
    # Tên đội; hiện trong UI và file kết quả.
    name = "team-13"

    def __init__(self) -> None:
        # Số lần module bị lỗi và agent phải dùng nước dự phòng.
        # Bài hoàn chỉnh phải luôn có fallback_count == 0 (test kiểm tra điều này).
        self.fallback_count = 0
        self._warned = False

    def reset(self, seed: int | None = None) -> None:
        """Gọi đầu mỗi ván: xóa cache/trạng thái của các module."""
        self.fallback_count = 0
        self._warned = False
        search.reset(seed)
        time_manager.reset(seed)

    # Benchmark của thầy đọc hai thuộc tính này (nếu có) để thống kê tìm kiếm.
    @property
    def last_nodes(self) -> int:
        return search.last_nodes

    @property
    def last_completed_depth(self) -> int:
        return search.last_completed_depth

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        try:
            budget_ms = time_manager.allocate_ms(obs, time_limit_ms)
            deadline = time.perf_counter() + budget_ms / 1000.0
            move = search.best_move(obs.board, obs.legal_moves, deadline)
            if move not in obs.legal_moves:
                raise ValueError(f"search trả về nước không hợp lệ: {move!r}")
            return Move(move)
        except Exception:
            # Lưới an toàn: lỗi ở một module không được làm hỏng cả ván
            # (exceptions và illegal_moves phải bằng 0). In lỗi đầu tiên để
            # người viết module biết mà sửa; fallback_count sẽ tăng.
            self.fallback_count += 1
            if not self._warned:
                self._warned = True
                traceback.print_exc()
            return obs.legal_moves[0]
