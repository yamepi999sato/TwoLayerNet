import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import numpy as np
import parameter as params
import matplotlib.pyplot as plt
import neural_network

rng = np.random.default_rng()

nlist = np.array([[1, 1, 1, 1],[1, 2, 3, 4],[1, 1, 1, 1]])
n_sum = nlist.sum(0)
#print(n_sum)


def calc_train_psi(nlist):
    # step1のターゲットとなる状態
    return np.exp(-((nlist -1)**2).sum(0)/2)


def calc_psi(weight, nlist):
    fu1 = np.tanh(weight["w1"].w * nlist + weight["b1"].w)
    u2 = np.dot(weight["w2"].w.reshape(1, -1), fu1)
    return np.exp(u2 + weight["b2"].value)

def metropolis(calc_p, randomwalk=False, sample_n = params.SAMPLE_N, M = params.M):
    nlist = np.empty((M, sample_n), dtype=int)
    n_vec = np.zeros(M)
    p = calc_p(n_vec)
    assert p > 1e-10
    idn = 0
    while idn < sample_n:
        if randomwalk:
            new_n_vec = n_vec + rng.integers(-np.floor(params.N_P/2), np.floor(params.N_P/2), M, endpoint=True)
            if np.any(new_n_vec < 0) or np.any(params.N_P < new_n_vec):
                n_vec = rng.integers(0, params.N_P, M, endpoint=True)
                p = calc_p(n_vec)
                continue
        else:
            new_n_vec = rng.integers(0, params.N_P, M, endpoint=True)
        new_p = calc_p(new_n_vec)
        
        if new_p > p * rng.random():
            n_vec, p = new_n_vec, new_p
        nlist[:, idn] = n_vec
        idn += 1
    return nlist

randomwalk = False
metro = metropolis(calc_train_psi, randomwalk, sample_n = params.SAMPLE_N, M = params.M)
#print(metro.shape)
"""
x = np.arange(0, params.SAMPLE_N)
y = metro[0]
plt.title("metropolis sampling of Ψ(n)^^2, SAMPLE_N=" + str(params.SAMPLE_N) + ", N_P=" + str(params.N_P) + ", randomwalk=" + str(randomwalk))
plt.xlabel('n_1')
plt.ylabel('frequency')
plt.hist(y, bins=100, density=True, color=(1.0,0,0.0))
plt.show()
"""

"""
nlist = np.array([[1, 1, 1],[2, 2, 2]])                                       #(M, SAMPLE_N)
weight = {}
"M=2, SAMPLE_N=3, HIDDEN_N=4"
weight["w1"] = np.array([[1, 2], [4, 5],[7, 8], [10, 11]])                    # (HIDDEN_N, M)
weight["b1"] = np.array([[1, 2, 3]]  )                                        # (1, SANPLE_N)
weight["w2"] = np.array([[1],[2],[3],[4]])                                    # (HIDDEN_N, 1)
w2_Ow = np.tanh(np.dot(weight["w1"], nlist) + weight["b1"])
print(w2_Ow)                                                                    # (HIDDEN_N, SAMPLE_N)
"""

"""
def calc_psi(weight, nlist):
    fu1 = np.tanh((weight["w1"].w @ nlist) + weight["b1"].w)
    print("w1: " + str(weight["w1"].w.shape))
    print("nlist: " + str(nlist.shape))
    print(fu1.shape)
    u2 = np.dot(weight["w2"].w.reshape(1, -1), fu1)
    print("w2: " + str(weight["w2"].w.reshape(1, -1).shape))
    print("u2: " + str(u2.shape))
    return np.exp(u2 + weight["b2"].value)



nlist = np.array([[1],[1],[1]])


M=3
HIDDEN_N=params.HIDDEN_N
weight={}
weight["w1"] = (
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(HIDDEN_N, M)))
weight["w2"] = (
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(HIDDEN_N, 1))
    )
weight["b1"] = (np.zeros((HIDDEN_N, 1)))
weight["b2"] = 0


fu1 = np.tanh((weight["w1"] @ nlist) + weight["b1"])
print("w1: " + str(weight["w1"].shape))
print("nlist: " + str(nlist.shape))
print(fu1.shape)


u2 = np.dot(weight["w2"].reshape(1, -1), fu1)
print("w2: " + str(weight["w2"].reshape(1, -1).shape))
print("u2: " + str(u2.shape))
"""

M=3
SAMPLE_N=10
i=1
j=2
a = np.zeros((M, SAMPLE_N))
a[i] = -1
a[j] = 1
#print(a)

nlist=np.random.rand(M, SAMPLE_N)
psi = np.exp(- ((nlist -1)**2).sum(0)/(2 * 0.5**2) )
print(psi.shape)

