"""Test giao ước giữa các module của nhóm (do trưởng nhóm quản lý).

Nếu test nào ở đây đỏ nghĩa là ai đó đã đổi chữ ký hàm hoặc làm hỏng giao ước.
Đừng sửa test cho xanh: hãy sửa code hoặc bàn lại giao ước với cả nhóm.
"""

import math
import time

import numpy as np

from game2048.benchmark import run_game
from game2048.engine import Game
from game2048.types import Move
from student import heuristic, search, time_manager
from student.agent import Agent


def _observation(seed: int = 0):
    return Game(seed).observe(None, None)


def test_evaluate_returns_finite_float():
    boards = [
        np.zeros((4, 4), dtype=int),
        np.array([[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 128, 256]]),
        _observation().board,
    ]
    for board in boards:
        value = heuristic.evaluate(board)
        assert isinstance(value, (int, float))
        assert math.isfinite(value)


def test_evaluate_does_not_modify_board():
    board = _observation().board.copy()
    before = board.copy()
    heuristic.evaluate(board)
    assert np.array_equal(board, before)


def test_best_move_returns_legal_move_even_after_deadline():
    obs = _observation()
    already_past = time.perf_counter() - 1.0
    move = search.best_move(obs.board, obs.legal_moves, already_past)
    assert move in obs.legal_moves


def test_allocate_ms_is_positive_and_within_limit():
    obs = _observation()
    for limit in (5.0, 50.0, 60_000.0):
        budget = time_manager.allocate_ms(obs, limit)
        assert 0 < budget <= limit


def test_agent_plays_full_game_without_fallback():
    agent = Agent()
    result = run_game(agent, seed=0, decision_budget_ms=None, total_budget_seconds=30)
    assert result["failure"] is None
    assert result["illegal_moves"] == result["exceptions"] == 0
    assert agent.fallback_count == 0


def test_choose_move_returns_move_instance():
    agent = Agent()
    agent.reset(0)
    obs = _observation()
    assert isinstance(agent.choose_move(obs, 50.0), Move)
