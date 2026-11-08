import numpy as np
from game2048.engine import Game
from student import time_manager


def test_allocate_ms_always_positive_and_under_limit():
    game = Game(seed=123)
    obs = game.observe(time_remaining_ms=60000.0, total_time_limit_ms=60000.0)

    for limit in [1.0, 5.0, 15.0, 50.0, 500.0, 60000.0]:
        budget = time_manager.allocate_ms(obs, limit)
        assert 0 < budget <= limit


def test_emergency_buffer_avoids_overtime():
    game = Game(seed=123)
    obs_low = game.observe(time_remaining_ms=300.0, total_time_limit_ms=60000.0)
    budget = time_manager.allocate_ms(obs_low, 300.0)
    assert 0 < budget <= 300.0
    assert budget <= 5.0

    
def test_crowded_board_receives_more_budget():
    game = Game(seed=123)
    obs_empty = game.observe(time_remaining_ms=40000.0, total_time_limit_ms=60000.0)
    time_empty = time_manager.allocate_ms(obs_empty, 40000.0)
    game.board = np.array([
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2, 4],
        [8, 16, 0, 0]
    ])
    obs_danger = game.observe(time_remaining_ms=40000.0, total_time_limit_ms=60000.0)
    time_danger = time_manager.allocate_ms(obs_danger, 40000.0)
    assert time_danger > time_empty