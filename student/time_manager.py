"""Quản lý thời gian — file của Vai 4 (Hiệu năng & Thời gian).

GIAO ƯỚC CHUNG CỦA NHÓM (không tự ý đổi chữ ký hàm):
    allocate_ms(obs, time_limit_ms) -> float
    - obs: Observation hiện tại. obs.time_remaining_ms là thời gian còn lại của
      CẢ VÁN (có thể là None nếu benchmark không đặt tổng thời gian).
    - time_limit_ms: giới hạn hiệu lực cho nước này mà benchmark truyền vào
      (nếu có giới hạn từng nước thì đây là giới hạn đó; nếu chỉ có tổng thời
      gian thì đây là thời gian còn lại của cả ván).
    - Trả về số mili-giây được phép nghĩ cho NƯỚC NÀY. Phải > 0 và không vượt
      quá time_limit_ms.

    reset(seed) -> None: xóa trạng thái giữa hai ván (agent.py gọi hàm này).

HIỆN TẠI: bản tạm, mỗi nước dùng tối đa 50 ms. Hãy thay bằng chiến lược chia
60 giây cho cả ván (nước dễ nghĩ ít, nước nguy hiểm nghĩ lâu).
"""

from __future__ import annotations

from game2048.observation import Observation

DEFAULT_MS = 50.0
import numpy as np

SAFETY_BUFFER_MS = 1000.0
MIN_BUDGET_MS = 3.0
MAX_BUDGET_MS = 90.0
PROB_CUTOFF = 0.001

def reset(seed: int | None = None) -> None:
    pass


def allocate_ms(obs: Observation, time_limit_ms: float) -> float:
  remaining_ms = obs.time_remaining_ms

  if remaining_ms is None:
    return float(max(1.0, min(DEFAULT_MS, time_limit_ms)))
  usable_time_ms = max(0.0, remaining_ms - SAFETY_BUFFER_MS)
  if usable_time_ms <= 0:
    return float(max(1.0, min(MIN_BUDGET_MS, time_limit_ms)))
  board = obs.board
  empty_cells = int(np.count_nonzero(board == 0))

  estimated_moves_left = max(30, empty_cells * 40 + 100)
  base_budget = usable_time_ms / estimated_moves_left
  if empty_cells >= 9:
    danger_mult = 0.35
  elif empty_cells >= 6:
    danger_mult = 0.70
  elif empty_cells >= 4:
    danger_mult = 1.30
  elif empty_cells == 3:
    danger_mult = 2.20
  elif empty_cells == 2:
    danger_mult = 3.50
  else:
    danger_mult = 5.00
  budget = base_budget * danger_mult
  budget = max(MIN_BUDGET_MS, min(budget, MAX_BUDGET_MS))  
  return max(1.0, min(budget, time_limit_ms))
