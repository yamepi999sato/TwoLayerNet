import numpy as np

"""
N_P   : 1サイトに入れる最大粒子数
ETA   : 学習率
ITER  : 勾配法の反復回数の最大値
EPS   : エネルギーの許容誤差
J_MAX : ポッピングの最大値
MU_MAX: 化学ポテンシャルの最大値
GRID  : 相図の刻み幅
"""

# constants
N_P = 5
ETA = 0.1
ITER_MAX = 10000
EPS = 1e-6
J_MAX  = 0.15
MU_MAX = 1
GRID = 50
N_SEQ = np.array(range(N_P))
N_SQRT = np.sqrt(N_SEQ)

HIDDEN_N = 40
SAMPLE_N = 10000
DX = 1e-3
ETA = 0.1
ITER_NUM_I = 250
ITER_NUM_E = 250
MAX_X = 5.0

paramter_strings = (
    f"HiddenN:{HIDDEN_N}, " 
    f"SampleN:{SAMPLE_N}, "
    f"Eta:{ETA}, "
    f"IterNumI:{ITER_NUM_I}, "
    f"IterNumE:{ITER_NUM_E}"
)

M = 3