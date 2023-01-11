import numpy as np

"""
M       : サイト数
N_P     : 1サイトに入れる最大粒子数
N_tot   : 粒子数期待値
ETA     : 学習率
ITER_MAX: 勾配法の反復回数の最大値
EPS     : エネルギーの許容誤差
J_MAX   : ポッピングの最大値
MU_MAX  : 化学ポテンシャルの最大値
GRID    : 相図の刻み幅
"""

# constants
M = 3
N_P = 5
ETA = 0.01
ITER_MAX = 10000
EPS = 1e-6
J_MAX  = 0.08
MU_MAX = 0.5
GRID = 10
N_SEQ = np.array(range(N_P))
N_SQRT = np.sqrt(N_SEQ)

HIDDEN_N = 40
SAMPLE_N = 1000
DX = 1e-3
ITER_NUM_K = 150
ITER_NUM_E = 350
MAX_X = 5.0


J = 0.05
MU = 0.5
U = 1

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

