"""
HIDDEN_N   : 隠れ層の数
SAMPLE_N   : モンテカルロ法のサンプル数
DX         : Hの運動エネルギー計算に必要な空間差分幅
INITIAL_ETA: 学習係数の初期値
ITER_NUM_K  : step1(重なり積分最大化)の反復回数
ITER_NUM_H  : step2(エネルギー最小化)の反復回数
"""
HIDDEN_N = 100
SAMPLE_N = 1000
DX = 1e-3
ETA = 0.1
ITER_NUM_K = 250
ITER_NUM_E = 250
MAX_X = 5.0

paramter_strings = (
    f"HiddenN:{HIDDEN_N}, " 
    f"SampleN:{SAMPLE_N}, "
    f"Eta:{ETA}, "
    f"IterNumK:{ITER_NUM_K}, "
    f"IterNumE:{ITER_NUM_E}"
)