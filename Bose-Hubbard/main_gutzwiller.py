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
        print(f"#step={i:04}")
    iterData_K.append((i, K, E, beta, nnn, n_avg, p, b2))
    #print(p)
    #nlist_K, psi2_K = neural_network.output_psi2(weight, L=5, N=100)

# E-minimizing (step2)
for w in weight.values():
    w.reset_internal_params()

for i in range(params.ITER_NUM_K, params.ITER_NUM_K + params.ITER_NUM_E):
    weight, K, E, beta, nnn, n_avg, p, b2 = neural_network.update(mu, J, weight, step=2, randomwalk=False)
    if i%100==0:
        print(f"#step={i:04}")
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
axn = fig.add_subplot(221)
axn.plot(np.arange(1, params.M+1), nnn, label="<n>")
axn.set_title("<n>")
axn.set_xlabel("site")
axn.set_ylabel("<n>")
axn.set_xticks([1, 2, 3])
axn.set_ylim(0, 2.5)
axn.legend()
axn.grid(True)

# 重なり積分(K)の収束確認(step2では厳密解と比較)
axK = fig.add_subplot(223)
axK.plot(is_K, np.array(Ks_K), label="step1 (K-maximizing)")
axK.plot(is_E, np.array(Ks_E), label="step2 (E-minimizing)")
axK.set_title("K (overlap integral)")
axK.set_xlabel("iter")
axK.set_ylabel("K")
axK.legend()
axK.grid(True)

# エネルギー(E)の収束確認
axE = fig.add_subplot(222)
axE.plot(np.arange(params.ITER_NUM_K + params.ITER_NUM_E), (data["E"] * np.ones(params.ITER_NUM_K + params.ITER_NUM_E)), 
         label="Gutzwiller", color="forestgreen")
axE.plot(is_K, (np.array(Hs_K)/params.M ), label="step1 (K-maximizing)")
axE.plot(is_E, (np.array(Hs_E)/params.M ), label="step2 (E-minimizing)")
axE.set_title("E/U (energy)")
axE.set_xlabel("iter")
axE.set_ylabel("E/U")
axE.legend()
axE.grid(True)

# Gutzwillerとのエネルギー差
axeE = fig.add_subplot(224)
axeE.plot(is_K, np.abs((np.array(Hs_K)/params.M ) - data["E"]), label="step1 (K-maximizing)")
axeE.plot(is_E, np.abs((np.array(Hs_E)/params.M ) - data["E"]), label="step2 (E-minimizing)")
axeE.set_title("error of E/U (energy)")
axeE.set_xlabel("iter")
axeE.set_ylabel("error of E/U (|newralnetwork - Gutzwiller|)")
axeE.set_yscale("log")
axeE.legend()
axeE.grid(True)


"""
n_1 = np.arange(1, params.N_P+1)
n_2 = np.arange(1, params.N_P+1)
n_3 = np.arange(1, params.N_P+1)

# 点(x, y, z)がもつ量
#value = np.random.rand(50)
value = np.ones(params.M)
 
# figureを生成する
fig = plt.figure()
 
# axをfigureに設定する
ax = fig.add_subplot(1, 1, 1, projection='3d')
 
# カラーマップを生成
cm = plt.cm.get_cmap('RdYlBu')
 
# axに散布図を描画、戻り値にPathCollectionを得る
mappable = ax.scatter(n_1, n_2, n_3, c=value, cmap=cm)
fig.colorbar(mappable, ax=ax)
 
# 表示する
plt.show()
"""

"""
# Ψ(n_1, 1, 1)
nlist_n11 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_n11[0,n] = n
for i in range(0, params.M):
    if i != 0:
        nlist_n11[i] = 1
print(nlist_n11)

ax2 = fig.add_subplot(223)
ax2.plot(np.arange(0, params.N_P+1), neural_network.calc_psi(weight, nlist_n11).reshape(-1), label="psi(n_1, 1, 1)")
ax2.set_title("psi(n_1, 1, 1)")
ax2.set_xlabel("n_1")
ax2.set_ylabel("psi(n_1, 1, 1)")
ax2.set_xticks([0, 1, 2, 3])
ax2.legend()
ax2.grid(True)


# Ψ(n_1, 2, 2)
nlist_n22 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_n22[0,n] = n
for i in range(0, params.M):
    if i != 0:
        nlist_n22[i] = 2
print(nlist_n22)

ax2 = fig.add_subplot(222)
ax2.plot(np.arange(0, params.N_P+1), neural_network.calc_psi(weight, nlist_n22).reshape(-1), label="psi(n_1, 1, 1)")
ax2.set_title("psi(n_1, 2, 2)")
ax2.set_xlabel("n_1")
ax2.set_ylabel("psi(n_1, 2, 2)")
ax2.set_xticks([0, 1, 2, 3])
ax2.legend()
ax2.grid(True)

# (n_1, 3, 3)
nlist_n33 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_n33[0,n] = n
for i in range(0, params.M):
    if i != 0:
        nlist_n33[i] = 3
print(nlist_n33)


ax2 = fig.add_subplot(224)
ax2.plot(np.arange(0, params.N_P+1), neural_network.calc_psi(weight, nlist_n33).reshape(-1), label="psi(n_1, 1, 1)")
ax2.set_title("psi(n_1, 3, 3)")
ax2.set_xlabel("n_1")
ax2.set_ylabel("psi(n_1, 3, 3)")
ax2.set_xticks([0, 1, 2, 3])
ax2.legend()
ax2.grid(True)



# (1, n_1, 1)
nlist_1n1 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_1n1[1,n] = n
for i in range(0, params.M):
    if i != 1:
        nlist_1n1[i] = 1
print(nlist_1n1)

ax2 = fig.add_subplot(222)
ax2.plot(np.arange(0, params.N_P+1), neural_network.calc_psi(weight, nlist_1n1).reshape(-1), label="psi(n_1, 1, 1)")
ax2.set_title("psi(1, n_2, 1)")
ax2.set_xlabel("n_1")
ax2.set_ylabel("psi(1, n_2, 1)")
ax2.set_xticks([0, 1, 2, 3])
ax2.legend()
ax2.grid(True)


# (1, 1, n_3)
nlist_11n = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_11n[2,n] = n
for i in range(0, params.M):
    if i != 2:
        nlist_11n[i] = 1
print(nlist_11n)

ax2 = fig.add_subplot(224)
ax2.plot(np.arange(0, params.N_P+1), neural_network.calc_psi(weight, nlist_11n).reshape(-1), label="psi(n_1, 1, 1)")
ax2.set_title("psi(1, 1, n_3)")
ax2.set_xlabel("n_1")
ax2.set_ylabel("psi(1, 1, n_3)")
ax2.set_xticks([0, 1, 2, 3])
ax2.legend()
ax2.grid(True)
"""

"""
# beta
axb = fig.add_subplot(223)
axb.plot(is_K, np.array(beta_K), label="step1 (K-maximizing)")
axb.plot(is_E, np.array(beta_E), label="step2 (E-minimizing)")
axb.set_title("beta (expectation value of annihilation operator)")
axb.set_xlabel("iter")
axb.set_ylabel("beta")
axb.legend()
axb.grid(True)
"""

plt.subplots_adjust(hspace=0.5)
plt.show()
