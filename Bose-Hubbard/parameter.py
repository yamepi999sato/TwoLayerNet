import numpy as np

"""
M           : サイト数
N_P         : 1サイトに入れる最大粒子数
U           : オンサイトポテンシャル
J           : ホッピング係数
MU          : 化学ポテンシャル
ETA         : 学習率

SAMPLE_N    : 1回の学習に用いるサンプルの個数
HIDDEN_N    : 隠れ層のユニット数
ITER_NUM_K  : step1の学習回数
ITER_NUM_E  : step2の学習回数

J_MAX       : ポッピング係数の最大値
MU_MAX      : 化学ポテンシャルの最大値
GRID        : 相図の刻み幅

EPS         : エネルギーの許容誤差
"""

# constants
M = 3
N_P = 3
U = 1
J = 0.3
MU = 0.5
ETA = 0.03

SAMPLE_N = 1000
HIDDEN_N = 40
ITER_NUM_K = 300
ITER_NUM_E = 700


J_MAX  = 0.5
MU_MAX = 1
GRID = 10


#EPS = 1e-6
#MAX_X = 5.0


paramter_strings = (
    f"site number M:{M}, "
    f"J:{J}, "
    f"mu:{MU}, "
    f"HiddenN:{HIDDEN_N}, " 
    f"SampleN:{SAMPLE_N}, "
    f"leraning rate Eta:{ETA}, "
    f"IterNumK:{ITER_NUM_K}, "
    f"IterNumE:{ITER_NUM_E}"
)

paramter_phase_strings = (
    f"site number M:{M}, "
    f"HiddenN:{HIDDEN_N}, " 
    f"SampleN:{SAMPLE_N}, "
    f"leraning rate Eta:{ETA}, "
    f"IterNumK:{ITER_NUM_K}, "
    f"IterNumE:{ITER_NUM_E}"
)
