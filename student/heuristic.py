"""Hàm đánh giá bàn cờ — file của Vai 2 (Heuristic & Tuning).

GIAO ƯỚC CHUNG CỦA NHÓM (không tự ý đổi chữ ký hàm):
    evaluate(board) -> float
    - board: numpy 4x4, mỗi ô là 0 (trống) hoặc giá trị ô (2, 4, 8, ...).
    - Trả về số float: càng LỚN thì bàn cờ càng TỐT.
    - Không được sửa board.
    - Hàm này bị gọi hàng nghìn lần mỗi nước đi, nên phải chạy nhanh.

HIỆN TẠI: bản tạm, sao chép công thức của starter (bỏ phần merge_score vì
search.py cộng riêng). Hãy thay bằng thiết kế của nhóm.
"""

from __future__ import annotations

import numpy as np

# Trọng số để ở đầu file: khi tuning chỉ cần đổi các hằng số này.
EMPTY_WEIGHT = 250.0
LARGEST_WEIGHT = 0.1
MONO_WEIGHT = 47.0
SMOOTH_WEIGHT = 15.0

# TODO (Vai 2): thêm các tiêu chí như monotonicity, smoothness, ô lớn ở góc,
# tiềm năng gộp ô; rồi tối ưu tốc độ (ví dụ tính trước điểm cho từng hàng).

def evaluate(board: np.ndarray) -> float:
    b= board.tolist()
    o_trong = sum(1 for row in b for i in row if i == 0)

    cap = 0   
    for i in range(4):
        for j in range(4):
            if b[i][j]==0:
                continue
            if j<3 and b[i][j] == b[i][j+1]:
                cap+=1
            if i<3 and b[i][j] == b[i+1][j]:
                cap+=1

#monotoncity
    mu=[]
    for row in b:
        hang_mu=[]
        for v in row:
            if v > 0:
                hang_mu.append(v.bit_length()-1)
            else:
                hang_mu.append(0)
        mu.append(hang_mu)

    phat = 0
    for day in mu + [list(c) for c in zip(*mu)]:
        tang = 0
        giam = 0
        for k in range(3):
            d = day[k+1] - day[k]
            if d > 0:
                tang+=d
            else:
                giam-=d
        phat += min(tang,giam)
    diem_phat_1 = phat * MONO_WEIGHT

#smoothness
    lech = 0
    for i in range(4):
        for j in range(4):
            if b[i][j] == 0:
                continue
            if j < 3 and b[i][j+1] != 0:
                lech += abs(mu[i][j+1] - mu[i][j])
            if i < 3 and b[i+1][j] != 0:
                lech += abs(mu[i+1][j] - mu[i][j])
    diem_phat_2 = lech * SMOOTH_WEIGHT

    diem = EMPTY_WEIGHT *(o_trong + cap) - diem_phat_1 - diem_phat_2

    lon_nhat = max(max(row) for row in b)
    if lon_nhat in (b[0][0], b[3][3], b[0][3], b[3][0]):
        diem+= LARGEST_WEIGHT*lon_nhat
    return float(diem)
