from __future__ import annotations

from game2048.observation import Observation

DEFAULT_MS = 60.0
import numpy as np

SAFETY_BUFFER_MS = 1000.0
MIN_BUDGET_MS = 3.0
MAX_BUDGET_MS = 90.0


def allocate_ms(obs: Observation, time_limit_ms: float) -> float:
  remaining_ms = obs.time_remaining_ms

  if remaining_ms is None:
    return float(max(1.0, min(DEFAULT_MS, time_limit_ms)))
  usable_time_ms = max(0.0, remaining_ms - SAFETY_BUFFER_MS)
  if usable_time_ms <= 0:
    return float(max(1.0, min(MIN_BUDGET_MS, time_limit_ms)))
  board = obs.board
  empty_cells = int(np.count_nonzero(board == 0))

  estimated_moves_left =empty_cells * 40 + 100
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
  budget = base_budget * danger_mult   # cần bao nhiêu thời gian 
  budget = max(MIN_BUDGET_MS, min(budget, MAX_BUDGET_MS))  
  return max(1.0, min(budget, time_limit_ms))
