"""Hàm đánh giá bàn cờ — file của Vai 2 (Heuristic & Tuning).
"""

from __future__ import annotations

import numpy as np

# Trọng số 
EMPTY_WEIGHT = 50.0
LARGEST_WEIGHT = 3
MONO_WEIGHT = 200.0
SMOOTH_WEIGHT = 180.0

def evaluate(board: np.ndarray) -> float:
    b= board.tolist()
    o_trong = sum(1 for row in b for i in row if i == 0)
#tìm ô trống và cặp ô = nhau
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
