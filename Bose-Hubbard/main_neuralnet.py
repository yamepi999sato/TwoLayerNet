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

# ヒストグラムの作成
#program_check.check_metropolis_sampling()

# 数値計算の本体
iterData_K, iterData_E = [], []
weight = neural_network.initialize_weight(optimizer.Adam)
time_start = time.time()


mu = params.MU
J = params.J

# K-maximizing (step1)
for i in range(params.ITER_NUM_K):
    weight, K, E, beta, nnn, n_avg, p, b2 = neural_network.update(mu, J, weight, step=1, randomwalk=False)
    if i%100==0:
        print(f"#step={i:04} \t K={K:.4f} \t H={E:.4f} \t <n>={n_avg:.4f}")
    iterData_K.append((i, K, E, beta, nnn, n_avg, p, b2))
    #print(p)
    #nlist_K, psi2_K = neural_network.output_psi2(weight, L=5, N=100)

# E-minimizing (step2)
for w in weight.values():
    w.reset_internal_params()

for i in range(params.ITER_NUM_K, params.ITER_NUM_K + params.ITER_NUM_E):
    weight, K, E, beta, nnn, n_avg, p, b2= neural_network.update(mu, J, weight, step=2, randomwalk=False)
    if i%100==0:
        print(f"#step={i:04} \t K={K:.4f} \t H={E:.4f} \t <n>={n_avg:.4f}")
    iterData_E.append((i, K, E, beta, nnn, n_avg, p, b2))
#nlist_E, psi2_E = neural_network.output_psi2(weight, L=params.MAX_X, N=100)


fig = plt.figure(figsize=(20, 8))
fig.suptitle(
    params.paramter_strings + ", "
    f"Optimizer:{weight['w1'].__class__.__name__}, \n"
    f"ElapsedTime:{time.time()-time_start:.2f}s, "
    f"date:{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
is_K, Ks_K, Hs_K, beta_K, nnn_K, n_avg_K, ps_K, b2s_K = zip(*iterData_K)
is_E, Ks_E, Hs_E, beta_E, nnn_E, n_avg_E, ps_E, b2s_E = zip(*iterData_E)



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
ax3.plot(is_K, np.array(Hs_K), label="step1 (K-maximizing)")
ax3.plot(is_E, np.array(Hs_E), label="step2 (E-minimizing)")
ax3.set_title("E (energy)")
ax3.set_xlabel("iter")
ax3.set_ylabel("E")
ax3.legend()
ax3.grid(True)

"""
# 1サイトあたりの平均の粒子数
ax2 = fig.add_subplot(223)
ax2.plot(is_K, (np.array(n_avg_K) ), label="step1 (K-maximizing)")
ax2.plot(is_E, (np.array(n_avg_E) ), label="step2 (E-minimizing)")
ax2.set_title("<n> (average number of particles)")
ax2.set_xlabel("iter")
ax2.set_ylabel("<n>")
ax2.legend()
ax2.grid(True)

# サンプリング時の確率
ax2 = fig.add_subplot(223)
ax2.plot(is_K, (np.array(ps_K) ), label="step1 (K-maximizing)")
ax2.plot(is_E, (np.array(ps_E) ), label="step2 (E-minimizing)")
ax2.set_title("psi(nlist)**2 of metropolis sampling")
ax2.set_xlabel("iter")
ax2.set_ylim(-1, 3)
#ax2.set_yscale("log")
ax2.legend()
ax2.grid(True)
"""

plt.subplots_adjust(hspace=0.5)
plt.show()

