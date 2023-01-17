# 以下3行だけ山本加筆
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

""" 1次元Bose-Hubbardモデルの基底状態を機械学習で計算する
    ・step1としてTrainPsi（解析解ではない、それらしい関数）との重なり積分を最大化する
    ・step2としてエネルギー期待値を最小化する
"""
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import parameter as params
import neural_network
import optimizer
import program_check


iterData_K, iterData_E = [], []
weight = neural_network.initialize_weight(optimizer.Adam)
time_start = time.time()

GRID = params.GRID
data = None
J_list = np.linspace(0.0, params.J_MAX, GRID)
mu_list = np.linspace(0.0, params.MU_MAX, GRID)
beta_list = np.empty([GRID,GRID])
loop = 0
for yi in range(GRID):
    data = None
    for xi in reversed(range(GRID)):
        # K-maximizing (step1)
        for w in weight.values():
            w.reset_internal_params()
            
        for i in range(params.ITER_NUM_K):
            weight, K, E, beta, n_1, n_avg, p, b2 = neural_network.update(mu_list[yi], J_list[xi], weight, step=1, randomwalk=False)
            iterData_K.append((i, K, E, beta, n_1, n_avg, p, b2))
            #print(p)
            #nlist_K, psi2_K = neural_network.output_psi2(weight, L=5, N=100)

        # E-minimizing (step2)
        for w in weight.values():
            w.reset_internal_params()

        for i in range(params.ITER_NUM_K, params.ITER_NUM_K + params.ITER_NUM_E):
            weight, K, E, beta, n_1, n_avg, p, b2 = neural_network.update(mu_list[yi], J_list[xi], weight, step=2, randomwalk=False)
            iterData_E.append((i, K, E, beta, n_1, n_avg, p, b2))
            #nlist_E, psi2_E = neural_network.output_psi2(weight, L=params.MAX_X, N=100)
        
        beta_list[yi, xi] = n_avg
        print(f"#loop={loop:04} \t mu={mu_list[yi]:.4f} \t J={J_list[xi]:.4f}\t <n>={n_avg:.4f} \t beta={beta: .4f}")
        loop += 1

fig = plt.figure()
fig.suptitle(
    params.paramter_phase_strings + ", "
    f"Optimizer:{weight['w1'].__class__.__name__}, \n"
    f"ElapsedTime:{time.time()-time_start:.2f}s, "
    f"date:{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
X, Y = np.meshgrid(J_list, mu_list)
#print(X.shape)
#print(Y.shape)
#print(beta_list.shape)
plt.pcolormesh(X, Y, beta_list, shading='auto')
plt.colorbar()
plt.title("<n> (average number of particles)")
plt.xlabel('J/U')
plt.ylabel('mu/U')
plt.show()
