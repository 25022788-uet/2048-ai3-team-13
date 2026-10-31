# Quy ước làm việc của nhóm 13

## Luồng gọi
`agent.py` → `time_manager.py` (nghĩ bao lâu) và `search.py` (chọn nước) → `heuristic.py` (chấm điểm bàn cờ).

## Ai được sửa file nào
| File / thư mục | Ai được sửa |
|---|---|
| `student/agent.py`, `.github/`, `.gitignore` (chỉ thêm dòng), `TEAM.md`, `tests/test_team_interfaces.py` | Trưởng nhóm (Vai 1) |
| `student/heuristic.py`, `team/tuning/`, `tests/test_team_heuristic.py` | Vai 2 |
| `student/search.py`, `tests/test_team_search.py` | Vai 3 |
| `student/time_manager.py`, `student/fastboard.py`, `tests/test_team_perf.py` | Vai 4 |
| `team/bench/`, `docs/`, `tests/test_team_integration.py` | Vai 5 |
| Mọi thứ còn lại (`src/`, `scripts/`, test cũ của thầy, `student/examples/`, `student/__init__.py`, `README.md`, `INSTALLATION.md`, `pyproject.toml`, `requirements.txt`, `.vscode/`, `assets/`) | Không ai |

Cần đổi file của người khác: nhắn chủ file (hoặc trưởng nhóm nếu là `agent.py`), không tự sửa.

## Giao ước hàm (không tự ý đổi chữ ký)
- `heuristic.evaluate(board) -> float`
- `search.best_move(board, legal_moves, deadline) -> Move` (luôn trả nước hợp lệ, kể cả khi quá deadline)
- `search.reset(seed)`, `time_manager.reset(seed)`
- `time_manager.allocate_ms(obs, time_limit_ms) -> float`
- `search.last_nodes`, `search.last_completed_depth`: cập nhật sau mỗi lần tìm kiếm

## Quy trình
```powershell
git checkout main
git pull
git checkout -b ten-vai/viec-dang-lam
# sửa code, rồi:
python -m pytest -q
git add <file cua minh>
git commit -m "Mo ta ngan"
git push -u origin ten-vai/viec-dang-lam
```
Rồi mở Pull Request trên GitHub. Không ai push thẳng vào `main`. Merge bằng **Squash and merge**.

## Lệnh benchmark chuẩn
```powershell
python scripts/benchmark_agent.py --agent student.agent:Agent --seeds 5 --start-seed 1000 --total-time-sec 60
```
