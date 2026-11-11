"""Test cho search.py (do trưởng nhóm quản lý).

Nguyên tắc: test search KHÔNG phụ thuộc vào heuristic thật (heuristic còn được
nhóm sửa liên tục), nên các test quan trọng dùng monkeypatch thay `evaluate`
bằng một hàm đơn giản, biết trước kết quả.
"""

import random
import time

import numpy as np
import pytest

from game2048.engine import Game
from game2048.rules import legal_moves
from game2048.types import Move
from student import search


def _far():
    """Một mốc deadline rất xa (còn hơn 11 ngày), tức là "không giới hạn thời gian"."""
    return time.perf_counter() + 1e6


def _board(rows):
    return np.array(rows, dtype=np.uint32)


def test_best_move_returns_legal_move_and_keeps_board(monkeypatch):
    """Nước trả về hợp lệ, không sửa board, ở cả MAX_DEPTH 1 và 2."""
    obs = Game(0).observe(None, None)
    for depth in (1, 2, 3):
        monkeypatch.setattr(search, "MAX_DEPTH", depth)
        board = obs.board.copy()
        before = board.copy()
        move = search.best_move(board, legal_moves(board), deadline=_far())
        assert move in legal_moves(board)
        assert np.array_equal(board, before)
        assert search.last_completed_depth == depth


# Hai ô 512 cạnh nhau theo hàng (gộp bằng LEFT/RIGHT) và theo cột (UP/DOWN).
_PAIR_ROW = [[512, 512, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
_PAIR_COL = [[0, 0, 0, 0], [0, 0, 0, 0], [512, 0, 0, 0], [512, 0, 0, 0]]


@pytest.mark.parametrize(
    "rows, merging_moves",
    [(_PAIR_ROW, (Move.LEFT, Move.RIGHT)), (_PAIR_COL, (Move.UP, Move.DOWN))],
)
def test_best_move_takes_the_merge_reward(monkeypatch, rows, merging_moves):
    """evaluate = 0 nên điểm chỉ đến từ việc gộp ô. Phải chọn nước gộp 512+512.

    Ta truyền legal_moves với một nước KHÔNG gộp đứng đầu, để bắt lỗi kiểu
    "luôn trả phần tử đầu tiên".
    """
    monkeypatch.setattr(search, "MAX_DEPTH", 1)
    monkeypatch.setattr(search, "evaluate", lambda b: 0.0)
    board = _board(rows)
    non_merging = [m for m in legal_moves(board) if m not in merging_moves]
    order = tuple(non_merging) + tuple(merging_moves)
    assert search.best_move(board, order, deadline=_far()) in merging_moves


def test_max_value_adds_merge_reward(monkeypatch):
    """_max_value = điểm gộp tốt nhất: gộp 512+512 được 1024 điểm."""
    monkeypatch.setattr(search, "evaluate", lambda b: 0.0)
    assert search._max_value(_board(_PAIR_ROW), 1) == 1024


def test_depth_one_has_no_lookahead(monkeypatch):
    """MAX_DEPTH = 1: mỗi nước ở gốc chỉ đúng 1 nút (giống starter); MAX_DEPTH = 2 nhiều hơn."""
    obs = Game(0).observe(None, None)
    monkeypatch.setattr(search, "MAX_DEPTH", 1)
    search.best_move(obs.board, obs.legal_moves, deadline=_far())
    assert search.last_nodes == len(obs.legal_moves)
    monkeypatch.setattr(search, "MAX_DEPTH", 2)
    search.best_move(obs.board, obs.legal_moves, deadline=_far())
    assert search.last_nodes > len(obs.legal_moves)


def test_dead_board_gets_big_negative_penalty():
    """Bàn kín và không gộp được -> rất âm (so với số cụ thể, không so với chính hằng số)."""
    board = _board([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert search._max_value(board, 1) <= -100_000


def test_chance_value_depth_zero_is_evaluate(monkeypatch):
    """depth = 0: nút CHANCE không sinh ô, chỉ chấm điểm bàn."""
    monkeypatch.setattr(search, "evaluate", lambda b: 7.0)
    board = _board([[2, 0, 0, 0]] + [[0] * 4] * 3)
    assert search._chance_value(board, 0) == 7.0


def test_chance_value_is_weighted_average(monkeypatch):
    """Giá trị kỳ vọng tính tay.

    Bàn có 2 ô trống, mọi ô còn lại khác nhau và >= 8 nên KHÔNG BAO GIỜ gộp được.
    Với evaluate = tổng các ô: sau khi sinh ô, giá trị = tổng bàn mới.
    Kỳ vọng = S + 0,9*2 + 0,1*4 = S + 2,2 với S là tổng bàn hiện tại.
    """
    monkeypatch.setattr(search, "evaluate", lambda b: float(b.sum()))
    values = [8 * 2**k for k in range(14)]            # 8, 16, ..., 65536 (14 giá trị khác nhau)
    cells = values[:]
    cells.insert(0, 0)                                 # ô trống ở (0, 0)
    cells.append(0)                                    # ô trống ở (3, 3)
    board = _board(np.array(cells).reshape(4, 4))
    assert search._chance_value(board, 1) == pytest.approx(float(board.sum()) + 2.2)


def test_reset_clears_statistics():
    search.last_nodes = 99
    search.last_completed_depth = 9
    search.reset(0)
    assert search.last_nodes == 0 and search.last_completed_depth == 0


# ---------------------------------------------------------------------------
# Bước B: deadline và iterative deepening
# ---------------------------------------------------------------------------

def _midgame_board():
    return _board([[2, 4, 0, 0], [8, 0, 2, 0], [0, 4, 0, 16], [2, 0, 0, 0]])


def test_deadline_already_passed_still_returns_legal_move():
    """Hết giờ từ trước: vẫn trả nước hợp lệ, và trả lại rất nhanh (không chạy mãi)."""
    board = _midgame_board()
    past = time.perf_counter() - 1.0
    start = time.perf_counter()
    move = search.best_move(board, legal_moves(board), past)
    assert move in legal_moves(board)
    assert search.last_completed_depth >= 1          # tầng 1 luôn xong
    assert time.perf_counter() - start < 0.1


def test_iterative_deepening_reaches_max_depth(monkeypatch):
    """Có nhiều thời gian: đi hết các tầng 1, 2, 3 và báo đúng độ sâu hoàn thành."""
    monkeypatch.setattr(search, "MAX_DEPTH", 3)
    board = _midgame_board()
    search.best_move(board, legal_moves(board), _far())
    assert search.last_completed_depth == 3


def test_search_stops_near_deadline(monkeypatch):
    """MAX_DEPTH rất lớn nhưng chỉ có 50 ms: phải dừng sớm, không chạy mãi."""
    monkeypatch.setattr(search, "MAX_DEPTH", 12)
    board = _midgame_board()
    start = time.perf_counter()
    move = search.best_move(board, legal_moves(board), start + 0.05)
    elapsed = time.perf_counter() - start
    assert move in legal_moves(board)
    assert elapsed < 0.5                      # biên rộng để máy chậm vẫn qua
    assert 1 <= search.last_completed_depth < 12


def test_unfinished_depth_is_discarded(monkeypatch):
    """Ép hết giờ giữa tầng 3 (không dùng đồng hồ thật): phải giữ kết quả tầng 2.

    Cách làm: đo tổng số nút cần để xong tầng 1+2, rồi thay _tick bằng bản
    ném _Timeout ngay khi vượt quá số đó một chút, tức là ngắt giữa tầng 3.
    """
    board = _midgame_board()
    monkeypatch.setattr(search, "MAX_DEPTH", 2)
    expected = search.best_move(board, legal_moves(board), _far())
    nodes_to_finish_depth_2 = search.last_nodes

    def tick_that_times_out_in_depth_3():
        search._nodes += 1
        if search._nodes > nodes_to_finish_depth_2 + 10:
            raise search._Timeout

    monkeypatch.setattr(search, "_tick", tick_that_times_out_in_depth_3)
    monkeypatch.setattr(search, "MAX_DEPTH", 5)
    move = search.best_move(board, legal_moves(board), _far())
    assert search.last_completed_depth == 2     # tầng 3 dở dang bị bỏ
    assert move == expected


# ---------------------------------------------------------------------------
# C2: ở nút CHANCE sâu chỉ sinh ô 2 (xác suất chuẩn hóa về 1,0)
# ---------------------------------------------------------------------------

def _board_two_empties_never_merging():
    """Bàn 2 ô trống, mọi ô còn lại khác nhau và >= 8 nên không bao giờ gộp được."""
    values = [8 * 2**k for k in range(14)]
    cells = [0] + values + [0]
    return _board(np.array(cells).reshape(4, 4))


def test_deep_chance_level_uses_only_tile_2(monkeypatch):
    """Nút CHANCE sâu (level 2): chỉ ô 2, trọng số đã chuẩn hóa -> kết quả = S + 2.

    (Nếu còn xét ô 4 sẽ ra S + 2,2; nếu quên chuẩn hóa sẽ ra 0,9 * (S + 2).)
    """
    monkeypatch.setattr(search, "evaluate", lambda b: float(b.sum()))
    board = _board_two_empties_never_merging()
    assert search._chance_value(board, 1, level=2) == pytest.approx(float(board.sum()) + 2.0)


def test_first_chance_level_still_uses_both_tiles(monkeypatch):
    """Nút CHANCE đầu (level 1) vẫn xét cả ô 2 lẫn ô 4: kết quả = S + 2,2."""
    monkeypatch.setattr(search, "evaluate", lambda b: float(b.sum()))
    board = _board_two_empties_never_merging()
    assert search._chance_value(board, 1, level=1) == pytest.approx(float(board.sum()) + 2.2)


def test_c2_reduces_node_count(monkeypatch):
    """Cùng độ sâu 3, bản C2 phải duyệt ít nút hơn hẳn bản xét đủ ô 2 và 4."""
    board = _board([[2, 4, 8, 16], [32, 64, 128, 256], [4, 8, 0, 0], [0, 0, 2, 4]])
    monkeypatch.setattr(search, "MAX_DEPTH", 3)
    search.best_move(board, legal_moves(board), _far())
    nodes_c2 = search.last_nodes
    monkeypatch.setattr(search, "DEEP_SPAWN_TILES", search.SPAWN_TILES)   # tắt C2
    search.best_move(board, legal_moves(board), _far())
    nodes_exact = search.last_nodes
    assert nodes_c2 < 0.75 * nodes_exact


def _sample_boards(count=12):
    """Bàn thật (ít ô trống) lấy từ các ván chơi ngẫu nhiên có seed cố định."""
    boards = []
    for seed in range(200):
        rng = random.Random(seed)
        game = Game(seed)
        for _ in range(150):
            obs = game.observe(None, None)
            if not obs.legal_moves:
                break
            empties = int((obs.board == 0).sum())
            if 2 <= empties <= 5 and len(obs.legal_moves) >= 2 and rng.random() < 0.05:
                boards.append(obs.board.copy())
            game.move(rng.choice(obs.legal_moves))
        if len(boards) >= count:
            break
    return boards[:count]


def test_c2_mostly_agrees_with_exact_search(monkeypatch):
    """C2 chỉ là xấp xỉ: ở độ sâu 3 phải chọn cùng nước với bản đầy đủ ở đa số bàn.

    Test này dùng heuristic thật; nếu sau này đổi heuristic mà test đỏ thì nghĩa là
    giả định "bỏ ô 4 ở tầng sâu" cần được đo lại bằng benchmark.
    """
    monkeypatch.setattr(search, "MAX_DEPTH", 3)
    boards = _sample_boards()
    assert len(boards) >= 8
    same = 0
    for board in boards:
        legal = legal_moves(board)
        monkeypatch.setattr(search, "DEEP_SPAWN_TILES", ((2, 1.0),))
        with_c2 = search.best_move(board, legal, _far())
        monkeypatch.setattr(search, "DEEP_SPAWN_TILES", search.SPAWN_TILES)
        exact = search.best_move(board, legal, _far())
        same += with_c2 == exact
    assert same >= 0.7 * len(boards)
