# 以下3行だけ山本加筆
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import parameter as params
import gutzwiller as gw
import main_neuralnet as main_n
import neural_network
import optimizer

time_start = time.time()

"""Gutzwillerで計算"""
GRID = params.GRID
GRID = 1
data = None
J_list = np.linspace(params.J, params.J +1)
mu_list = np.linspace(params.MU, params.MU +1 )
beta_list = np.empty([GRID, GRID])
E_list = np.empty([GRID, GRID])


for yi in range(GRID):
    data = None
    for xi in reversed(range(GRID)):
        data = gw.calc(mu_list[yi], J_list[xi], data) 
        beta_list[yi, xi] = data["beta"] if data["OK"] else None
        E_list[yi, xi] = data["E"] if data["OK"] else None





"""ニューラルネットワークで計算"""
mu = params.MU
J = params.J
iterData_K, iterData_E = [], []
weight = neural_network.initialize_weight(optimizer.Adam)

# K-maximizing (step1)
for i in range(params.ITER_NUM_K):
    weight, K, E, beta, nnn, n_avg, p, b2 = neural_network.update(mu, J, weight, step=1, randomwalk=False)
    if i%100==0:
        print(f"#step={i:04} \t K={K:.4f} \t H={E:.4f} \t beta={beta:.4f}")
    iterData_K.append((i, K, E, beta, nnn, n_avg, p, b2))
    #print(p)
    #nlist_K, psi2_K = neural_network.output_psi2(weight, L=5, N=100)

# E-minimizing (step2)
for w in weight.values():
    w.reset_internal_params()

for i in range(params.ITER_NUM_K, params.ITER_NUM_K + params.ITER_NUM_E):
    weight, K, E, beta, nnn, n_avg, p, b2 = neural_network.update(mu, J, weight, step=2, randomwalk=False)
    if i%100==0:
        print(f"#step={i:04} \t K={K:.4f} \t H={E:.4f} \t beta={beta:.4f}")
    iterData_E.append((i, K, E, beta, nnn, n_avg, p, b2))
is_K, Ks_K, Hs_K, beta_K, nnn_K, n_avg_K, ps_K, b2s_K = zip(*iterData_K)
is_E, Ks_E, Hs_E, beta_E, nnn_E, n_avg_E, ps_E, b2s_E = zip(*iterData_E)



"""グラフ化"""
fig = plt.figure(figsize=(20, 7))

fig.suptitle(
    "Gutwiller vs neural network\n" +  
    params.paramter_strings + ", "
    f"Optimizer:{weight['w1'].__class__.__name__}, \n"
    f"ElapsedTime:{time.time()-time_start:.2f}s, "
    f"date:{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")

# 各サイトの粒子数n_i
ax1 = fig.add_subplot(221)
ax1.plot(np.arange(1, params.M+1), nnn, label="<n>")
ax1.set_title("<n>")
ax1.set_xlabel("site")
ax1.set_ylabel("<n>")
ax1.set_xticks([1, 2, 3])
ax1.set_ylim(0, 2.5)
ax1.legend()
ax1.grid(True)

# beta
ax2 = fig.add_subplot(223)
ax2.plot(is_K, np.array(beta_K), label="step1 (K-maximizing)")
ax2.plot(is_E, np.array(beta_E), label="step2 (E-minimizing)")
ax2.set_title("beta (expectation value of annihilation operator)")
ax2.set_xlabel("iter")
ax2.set_ylabel("beta")
#ax2.set_ylim(-1, 5)
#ax2.set_yscale("log")
ax2.legend()
ax2.grid(True)

# 重なり積分(K)の収束確認(step2では厳密解と比較)
ax4 = fig.add_subplot(222)
ax4.plot(is_K, np.array(Ks_K), label="step1 (K-maximizing)")
ax4.plot(is_E, np.array(Ks_E), label="step2 (E-minimizing)")
ax4.set_title("K (overlap integral)")
ax4.set_xlabel("iter")
ax4.set_ylabel("K")
ax4.legend()
ax4.grid(True)

# エネルギー(E)の収束確認
ax3 = fig.add_subplot(224)
ax3.plot(is_K, (np.array(Hs_K)/params.M ), label="step1 (K-maximizing)")
ax3.plot(is_E, (np.array(Hs_E)/params.M ), label="step2 (E-minimizing)")
ax3.plot(np.arange(params.ITER_NUM_K + params.ITER_NUM_E), (data["E"] * np.ones(params.ITER_NUM_K + params.ITER_NUM_E)), label="Gutzwiller")
ax3.set_title("E (energy expectation value)")
ax3.set_xlabel("iter")
ax3.set_ylabel("E")
#ax3.set_yscale("log")
ax3.legend()
ax3.grid(True)

plt.subplots_adjust(hspace=0.5)
plt.show()
