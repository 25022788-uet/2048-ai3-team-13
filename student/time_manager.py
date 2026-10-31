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


def reset(seed: int | None = None) -> None:
    pass


def allocate_ms(obs: Observation, time_limit_ms: float) -> float:
    return max(1.0, min(DEFAULT_MS, time_limit_ms))
