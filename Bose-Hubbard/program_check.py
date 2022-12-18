import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import numpy as np
import parameter as params
import matplotlib.pyplot as plt

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
            if (n_vec < 0) or (params.N_P < n_vec):
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

metro = metropolis(calc_train_psi, randomwalk=False, sample_n = params.SAMPLE_N, M = params.M)
print(metro.shape)

x = np.arange(0, params.SAMPLE_N)
y = metro[0]
plt.title("metropolis sampling of Ψ(n)^^2, SAMPLE_N=" + str(params.SAMPLE_N))
plt.xlabel('n_1')
plt.ylabel('frequency')
plt.hist(y, bins=100, density=True, color=(1.0,0,0.0))
plt.show()