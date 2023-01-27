import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import numpy as np
import optimizer
import parameter as params
rng = np.random.default_rng()


def normalize(psi):
    norm2 = (psi**2).sum()
    psi /= np.sqrt(norm2)
    return norm2

def initialize_weight(opt_class):
    HIDDEN_N = params.HIDDEN_N
    M = params.M
    weight = {}
    weight["w1"] = opt_class(
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(HIDDEN_N, M))
    )
    weight["w2"] = opt_class(
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(1, HIDDEN_N))
    )
    weight["b1"] = opt_class(np.zeros((HIDDEN_N, 1)))
    weight["b2"] = optimizer.Normalizer()
    return weight


def calc_psi(weight, nlist):
    """ネットワークを使ってpsiを計算"""
    fu1 = np.tanh(np.dot(weight["w1"].w, nlist) + weight["b1"].w)
    u2 = np.dot(weight["w2"].w, fu1)
    return np.exp(u2 + weight["b2"].value)


def calc_train_psi(nlist):
    """step1のターゲットとなる状態 n=1に鋭いピークを持つガウシアン"""
    return np.exp(- ((nlist -2.5)**2).sum(0)/(2 * 0.5**2) )


def metropolis(calc_p, randomwalk, sample_n = params.SAMPLE_N, M = params.M):
    """メトロポリス法で|psi|^2を確率分布関数にしてサンプル生成"""
    nlist = np.empty((params.M, sample_n), dtype=int)
    n_vec = np.ones((params.M, 1))
    p = calc_p(n_vec)
    assert np.all(p != np.inf)
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
        new_p = calc_p(new_n_vec.reshape(-1, 1))
        
        
        """メトロポリステスト"""
        if np.all(new_p > p * rng.random()):
            n_vec, p = new_n_vec, new_p
        nlist[:, idn] = n_vec.ravel()
        idn += 1
    #print(p)
    return nlist, p


def update(mu, J, weight, step, randomwalk):
    nlist, p = metropolis(lambda nlist:calc_psi(weight, nlist).ravel() **2, randomwalk=False)
    psi = calc_psi(weight, nlist)
    
    """前もって関数を用意"""
    def M(n_1, n_2, dtyape=int):                        
        return np.sqrt(np.where(n_2 == params.M, 0, n_2) * n_1)
    
    def tlist(i, j):
        if (0 <= i < params.M) and (0 <= j <params.M):
            a = np.zeros((params.M, params.SAMPLE_N))
            a[i] = -1
            a[j] = 1
            return a
        else:
            return 0
        
    def ilist(i):
        if (0 <= i < params.M):
            a = np.zeros((params.M, params.SAMPLE_N))
            a[i] = 1
            return a
        else:
            return 0
        
    
    """エネルギー期待値Eを計算"""
    H_vec = np.zeros(params.SAMPLE_N)
    for i in range(params.M):
        J_term = -J * (
            M(nlist[i], nlist[(i+1)%params.M]) * calc_psi(weight, nlist + tlist(i, (i+1)%params.M)) / psi 
            + M(nlist[(i+1)%params.M], nlist[i]) * calc_psi(weight, nlist + tlist((i+1)%params.M, i)) / psi
            )
        J_term = J_term.ravel()
        U_term = params.U/2 * nlist[i] * (nlist[i] -1)
        mu_term = - mu * nlist[i]
        H_vec += J_term + U_term + mu_term
    E = np.average(H_vec)

    
    "重なり積分Kを計算"
    phi = calc_train_psi(nlist)
    A1 = phi/psi
    A1_avg = np.average(A1)
    A2_avg = np.average(A1**2)
    K = A1_avg**2 / A2_avg
    

    "各サイトのる湯指数n_i"
    nnn = np.zeros(params.M)
    for i in range(params.M):
        nnn[i] = np.average(nlist[i])
    
    
    "1サイトあたりの平均の粒子数"
    n_avg = np.average(nlist)
    
    
    "生成演算子a_iの和の期待値beta"
    beta_list = np.zeros(params.SAMPLE_N)
    for i in range(params.M):
        #print("nlist[i]: " + str(nlist[i].shape))
        #print("calc_psi(weight, nlist + ilist(i)): " + str(calc_psi(weight, nlist + ilist(i)).ravel().shape))
        beta_list += np.sqrt(nlist[i]+1) * calc_psi(weight, nlist + ilist(i)).ravel() / calc_psi(weight, nlist).ravel()
    beta = np.average(beta_list / params.M)
    
    
    "Owの計算(活性化関数はtanhとexp)"
    weight["w2"].Ow = np.tanh(np.dot(weight["w1"].w, nlist) + weight["b1"].w)
    #print("W2.Ow: " + str(weight["w2"].Ow.shape))
    weight["b1"].Ow = weight["w2"].w.reshape(-1, 1) * (1 - weight["w2"].Ow ** 2)
    #print("b1.Ow: " + str(weight["b1"].Ow.shape))
    weight["w1"].Ow = weight["b1"].Ow.reshape(params.HIDDEN_N, 1, params.SAMPLE_N) * nlist.reshape(1, params.M, params.SAMPLE_N)
    if step ==1:
        
        def update_func_b1(Ow):
            #Ow_avg = np.average(Ow, axis=1, keepdims=True)
            Ow_avg = np.average(Ow, axis=1).reshape(-1, 1)
            AOw = A1 * Ow
            #AOw_avg = np.average(AOw, axis=1, keepdims=True)
            AOw_avg = np.average(AOw, axis=1).reshape(-1, 1)
            return -2 * K * (AOw_avg / A1_avg - Ow_avg)
        
        def update_func_w2(Ow):
            #Ow_avg = np.average(Ow, axis=1, keepdims=True).T
            Ow_avg = np.average(Ow, axis=1).reshape(1, -1)
            AOw = A1 * Ow
            #AOw_avg = np.average(AOw, axis=1, keepdims=True).T
            AOw_avg = np.average(AOw, axis=1).reshape(1, -1)
            return -2 * K * (AOw_avg / A1_avg - Ow_avg)
        
        def update_func_w1(Ow):
            #Ow_avg = np.average(Ow, axis=2, keepdims=False)
            Ow_avg = np.average(Ow, axis=2)
            AOw = A1 * Ow
            #AOw_avg = np.average(AOw, axis=2, keepdims=False)
            AOw_avg = np.average(AOw, axis=2)
            return -2 * K * (AOw_avg / A1_avg - Ow_avg)
        
    else:
        
        def update_func_b1(Ow):
            #Ow_avg = np.average(Ow, axis=1, keepdims=True)
            Ow_avg = np.average(Ow, axis=1).reshape(-1, 1)
            OwH = Ow * H_vec
            #OwH_avg = np.average(OwH, axis=1, keepdims=True)
            OwH_avg = np.average(OwH, axis=1).reshape(-1, 1)
            return 2 * (OwH_avg - Ow_avg * E)
        
        def update_func_w2(Ow):
            #Ow_avg = np.average(Ow, axis=1, keepdims=True).T
            Ow_avg = np.average(Ow, axis=1).reshape(1, -1)
            OwH = Ow * H_vec
            #OwH_avg = np.average(OwH, axis=1, keepdims=True).T
            OwH_avg = np.average(OwH, axis=1).reshape(1, -1)
            return 2 * (OwH_avg - Ow_avg * E)
        
        def update_func_w1(Ow):
            #Ow_avg = np.average(Ow, axis=2, keepdims=False)
            Ow_avg = np.average(Ow, axis=2)
            OwH = Ow * H_vec
            #OwH_avg = np.average(OwH, axis=2, keepdims=False)
            OwH_avg = np.average(OwH, axis=2)
            return 2 * (OwH_avg - Ow_avg * E)
        
    for key, w in weight.items():
        
        if key=="b1" and isinstance(w, optimizer.Optimizer):
            w.update_weight(update_func_b1)
        elif key=="w2" and isinstance(w, optimizer.Optimizer):
            w.update_weight(update_func_w2)
        elif key=="w1" and isinstance(w, optimizer.Optimizer):
            w.update_weight(update_func_w1)
        elif key=="b2":
            w.update_rough(psi)
        else:
            assert False
    return weight, K, E, beta, nnn, n_avg, p, weight["b2"].value



