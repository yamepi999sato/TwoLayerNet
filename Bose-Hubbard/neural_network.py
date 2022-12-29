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
    """ネットワークを使ってpsiを計算"""
    fu1 = np.tanh(weight["w1"].w * nlist + weight["b1"].w)
    u2 = np.dot(weight["w2"].w.reshape(1, -1), fu1)
    return np.exp(u2 + weight["b2"].value)


def calc_train_psi(nlist):
    """step1のターゲットとなる状態 n=1に鋭いピークを持つガウシアン"""
    return np.exp(- ((nlist -1)**2).sum(0)/(2 * 0.5**2) )


def metropolis(calc_p, randomwalk, sample_n = params.SAMPLE_N, M = params.M):
    """メトロポリス法で|psi|^2を確率分布関数にしてサンプル生成"""
    nlist = np.empty((M, sample_n), dtype=int)
    n_vec = np.zeros(M)
    p = calc_p(n_vec)
    assert p > 1e-10
    idn = 0
    while idn < sample_n:
        if randomwalk:
            """ランダムウォークの場合"""
            new_n_vec = n_vec + rng.integers(-np.floor(params.N_P/2), np.floor(params.N_P/2), M, endpoint=True)
            if np.any(n_vec < 0) or np.any(params.N_P < n_vec):
                n_vec = rng.integers(0, params.N_P, M, endpoint=True)
                p = calc_p(n_vec)
                continue
        else:
            """ランダムウォークでない場合"""
            new_n_vec = rng.integers(0, params.N_P, M, endpoint=True)
        new_p = calc_p(new_n_vec)
        
        """メトロポリステスト"""
        if new_p > p * rng.random():
            n_vec, p = new_n_vec, new_p
        nlist[:, idn] = n_vec
        idn += 1
    return nlist


def update(weight, step, randomwalk):
    nlist = metropolis(lambda nlist:calc_psi(weight, nlist).ravel**2, randomwalk=False)
    psi = calc_psi(weight, nlist)
    #DX = params.DX
    
    """前もって関数を用意"""
    def M(n_1, n_2):
        if (n_1 != 0) and (n_2 != params.N_P):
            return np.sqrt(n_1 * (n_2 + 1))
        else:
            return 0
    def tlist(i, j):
        if (0 <= i < params.N_P) and (0 <= j <params.N_P):
            a = np.zeros(M, params.SAMPLE_N)
            a[i] = -1
            a[j] = 1
            return a
        else:
            return 0
    
    """エネルギー期待値Eを計算"""
    H_vec = np.zeros(params.SAMPLE_N)
    for i in range(params.M):
        J_term = -params.J * (M(nlist[i], nlist[i+1]) * calc_psi(weight, nlist + tlist(i, i+1)) / psi + M(nlist[i+1], nlist[i]) * calc_psi(weight, nlist + tlist(i+1, i)) / psi)
        U_term = params.U/2 * nlist[i] * (nlist[i] -1)
        mu_term = - params.mu * nlist[i]
        H_vec += J_term + U_term + mu_term
    E = np.average(H_vec)
    
    "重なり積分Kを計算"
    phi = calc_train_psi(nlist)
    phipsi = phi/psi
    K = np.average(phipsi)
    
    
    
    "Owの計算(活性化関数はtanhとexp)"
    weight["w2"].Ow = np.tanh(np.dot(weight["w1"].w, nlist) + weight["b1"].w)
    weight["b1"].Ow = weight["w2"].w * (1 - weight["w2"].Ow ** 2)
    weight["w1"].Ow = weight["b1"].Ow * nlist
    if step ==1:
        
        def update_func(Ow):
            phipsiOw = phipsi * Ow
            phipsiOw_avg = np.average(phipsiOw, axis=1, keepdims=True)
            return phipsiOw_avg
        
    else:
        
        def update_func(Ow):
            Ow_avg = np.average(Ow, axis=1, keepdims=True)
            OwH = Ow * H_vec
            OwH_avg = np.average(OwH, axis=1, keepdims=True)
            return 2 * (OwH_avg - Ow_avg * E)
        
    for key, w in weight.items():
        if key=="b2":
            w.update_rough(psi)
        elif isinstance(w, optimizer.Optimizer):
            w.update_weight(update_func)
        else:
            assert False
    return weight, K, E




def output_psi2(weight, L, N):
    """グラフ用の|psi|^2を計算"""
    xlist = np.linspace(-L / 2, L / 2, N, dtype=float)
    psi2 = calc_psi(weight, xlist).ravel() ** 2
    return xlist, psi2 / (np.sum(psi2) * L / N)