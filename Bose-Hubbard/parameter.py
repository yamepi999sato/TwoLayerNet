import numpy as np


# constants
M = 3                   # サイト数
N_P = 3                 # 1サイトに入れる最大粒子数
U = 1                   # オンサイトポテンシャル(1に固定)
J = 0.5                 # ホッピング
MU = 0.5                # 化学ポテンシャル
ETA = 0.03              # 学習率

SAMPLE_N = 1000         # 1回の学習に用いるサンプルの個数
HIDDEN_N = 40           # 隠れ層のユニット数
ITER_NUM_K = 200        # step1の学習回数
ITER_NUM_E = 300        # step2の学習回数


"""相図作成に使用"""
J_MAX  = 0.5            # ホッピングの最大値
MU_MAX = 1              # 化学ポテンシャルの最大値
GRID = 100               # 相図の刻み幅


"""Gutzwillerに使用"""
ETA_g = 0.1             # Gutzwillerの学習率
ITER_MAX = 10000        # 勾配法の反復回数の最大値
EPS = 1e-6              # エネルギーの許容誤差
N_SEQ = np.array(range(N_P))
N_SQRT = np.sqrt(N_SEQ)

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

"""相図作成に使用"""
paramter_strings_phase = (
    f"site number M:{M}, "
    f"HiddenN:{HIDDEN_N}, " 
    f"SampleN:{SAMPLE_N}, "
    f"leraning rate Eta:{ETA}, "
    f"IterNumK:{ITER_NUM_K}, "
    f"IterNumE:{ITER_NUM_E}"
)
