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
    weight, K, E, beta, n_1, n_avg, p, b2 = neural_network.update(mu, J, weight, step=1, randomwalk=False)
    print(f"#step={i:04} \t K={K:.4f} \t H={E:.4f} \t p={p}")
    iterData_K.append((i, K, E, beta, n_1, n_avg, p, b2))
    #print(p)
    #nlist_K, psi2_K = neural_network.output_psi2(weight, L=5, N=100)

# E-minimizing (step2)
for w in weight.values():
    w.reset_internal_params()

for i in range(params.ITER_NUM_K, params.ITER_NUM_K + params.ITER_NUM_E):
    weight, K, E, beta, n_1, n_avg, p, b2 = neural_network.update(mu, J, weight, step=2, randomwalk=False)
    print(f"#step={i:04} \t K={K:.4f} \t H={E:.4f} \t p={p}")
    iterData_E.append((i, K, E, beta, n_1, n_avg, p, b2))
#nlist_E, psi2_E = neural_network.output_psi2(weight, L=params.MAX_X, N=100)


fig = plt.figure(figsize=(15, 5))
fig.suptitle(
    params.paramter_strings + ", "
    f"Optimizer:{weight['w1'].__class__.__name__}, "
    f"ElapsedTime:{time.time()-time_start:.2f}s, "
    f"date:{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
is_K, Ks_K, Hs_K, beta_K, ns_K, n_avg_K, ps_K, b2s_K = zip(*iterData_K)
is_E, Ks_E, Hs_E, beta_E, ns_E, n_avg_E, ps_E, b2s_E = zip(*iterData_E)

print(np.array(ps_K))

"""
# サイト1の粒子数
ax2 = fig.add_subplot(221)
ax2.plot(is_K, (np.array(ns_K) ), label="step1 (K-maximizing)")
ax2.plot(is_E, (np.array(ns_E) ), label="step2 (E-minimizing)")
ax2.set_title("n_1 (number of particles on site1)")
ax2.set_xlabel("iter")
ax2.set_ylabel("n_1")
ax2.legend()
ax2.grid(True)
"""

# 規格化用の重みw2
ax2 = fig.add_subplot(223)
ax2.plot(is_K, (np.array(b2s_K) ), label="step1 (K-maximizing)")
ax2.plot(is_E, (np.array(b2s_E) ), label="step2 (E-minimizing)")
ax2.set_title("b2")
ax2.set_xlabel("iter")
#ax2.set_ylim(-1, 5)
#ax2.set_yscale("log")
ax2.legend()
ax2.grid(True)

# サンプリング時の確率
ax2 = fig.add_subplot(221)
ax2.plot(is_K, (np.array(ps_K) ), label="step1 (K-maximizing)")
ax2.plot(is_E, (np.array(ps_E) ), label="step2 (E-minimizing)")
ax2.set_title("psi(nlist)**2 of metropolis sampling")
ax2.set_xlabel("iter")
#ax2.set_ylim(-1, 5)
ax2.set_yscale("log")
ax2.legend()
ax2.grid(True)


"""
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


# 1サイトあたりの平均の粒子数
ax2 = fig.add_subplot(223)
ax2.plot(is_K, (np.array(n_avg_K) ), label="step1 (K-maximizing)")
ax2.plot(is_E, (np.array(n_avg_E) ), label="step2 (E-minimizing)")
ax2.set_title("<n> (average number of particles)")
ax2.set_xlabel("iter")
ax2.set_ylabel("<n>")
ax2.legend()
ax2.grid(True)
"""

# 重なり積分(K)の収束確認(step2では厳密解と比較)
ax2 = fig.add_subplot(222)
ax2.plot(is_K, (np.array(Ks_K) ), label="step1 (K-maximizing)")
ax2.plot(is_E, (np.array(Ks_E) ), label="step2 (E-minimizing)")
ax2.set_title("K (overlap integral)")
ax2.set_xlabel("iter")
ax2.set_ylabel("K")
#ax2.set_yscale("log")
ax2.legend()
ax2.grid(True)

# エネルギー(E)の収束確認
ax3 = fig.add_subplot(224)
ax3.plot(is_K, (np.array(Hs_K) ), label="step1 (K-maximizing)")
ax3.plot(is_E, (np.array(Hs_E) ), label="step2 (E-minimizing)")
ax3.set_title("E (energy expectation value)")
ax3.set_xlabel("iter")
ax3.set_ylabel("E")
#ax3.set_yscale("log")
ax3.legend()
ax3.grid(True)

plt.subplots_adjust(hspace=0.5)
plt.show()


"""
# グラフ化
fig = plt.figure(figsize=(15, 5))
fig.suptitle(
    params.paramter_strings + ", "
    f"Optimizer:{weight['w1'].__class__.__name__}, "
    f"ElapsedTime:{time.time()-time_start:.2f}s, "
    f"date:{datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
psi2_Ex = neural_network.calc_exact_psi(nlist_E) ** 2
psi2_T = neural_network.calc_train_psi(nlist_E) ** 2
is_K, Ks_K, Hs_K = zip(*iterData_K)
is_E, Ks_E, Hs_E = zip(*iterData_E)

# 重なり積分(K)の収束確認(step2では厳密解と比較)
ax2 = fig.add_subplot(222)
ax2.plot(is_K, np.abs(np.array(Ks_K) - 1), label="step1 (K-maximizing)")
ax2.plot(is_E, np.abs(np.array(Ks_E) - 1), label="step2 (E-minimizing)")
ax2.set_title("Error of K (overlap integral)")
ax2.set_xlabel("iter")
ax2.set_ylabel("|K-1|")
ax2.set_yscale("log")
ax2.legend()
ax2.grid(True)

# エネルギー(E)の収束確認
ax3 = fig.add_subplot(224)
ax3.plot(is_K, np.abs(np.array(Hs_K) - 1), label="step1 (K-maximizing)")
ax3.plot(is_E, np.abs(np.array(Hs_E) - 1), label="step2 (E-minimizing)")
ax3.set_title("Error of H (energy expectation value)")
ax3.set_xlabel("iter")
ax3.set_ylabel("|E-1|")
ax3.set_yscale("log")
ax3.legend()
ax3.grid(True)

plt.subplots_adjust(hspace=0.5)
plt.show()
"""
