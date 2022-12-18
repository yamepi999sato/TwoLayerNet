# 以下3行だけ山本加筆
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import numpy as np
import optimizer
import parameter as params

rng = np.random.default_rng()


def initialize_weight(opt_class):
    HIDDEN_N = params.HIDDEN_N
    M = params.M
    weight = {}
    weight["w1"] = opt_class(
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(HIDDEN_N, M))
    )
    weight["w2"] = opt_class(
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(HIDDEN_N, 1))
    )
    weight["b1"] = opt_class(np.zeros((HIDDEN_N, 1)))
    weight["b2"] = optimizer.Normalizer()
    return weight


def calc_psi(weight, nlist):
    fu1 = np.tanh(weight["w1"].w * nlist + weight["b1"].w)
    u2 = np.dot(weight["w2"].w.reshape(1, -1), fu1)
    return np.exp(u2 + weight["b2"].value)


def calc_train_psi(nlist):
    # step1のターゲットとなる状態 n=1に鋭いピークを持つガウシアン
    return np.exp(- ((nlist -1)**2).sum(0)/(2 * 0.5**2) )


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


#def update(weight, step, randomwalk):
    
    
    
        