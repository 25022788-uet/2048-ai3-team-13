from __future__ import annotations

import time
import traceback

from game2048.observation import Observation
from game2048.types import Move

from . import search, time_manager


class Agent:
 
    name = "team-13"

    def __init__(self) -> None:
        self.fallback_count = 0
        self._warned = False

    def reset(self, seed: int | None = None) -> None:
        """Gọi đầu mỗi ván: xóa cache/trạng thái của các module."""
        self.fallback_count = 0
        self._warned = False
        search.reset(seed)

  
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
            self.fallback_count += 1
            if not self._warned:
                self._warned = True
                traceback.print_exc()
            return obs.legal_moves[0]
