# 以下3行だけ山本加筆
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import numpy as np
import optimizer
import parameters as params

rng = np.random.default_rng()


def initialize_weight(opt_class):
    HIDDEN_N = params.HIDDEN_N
    weight = {}
    weight["w1"] = opt_class(
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(HIDDEN_N, 1))
    )
    weight["w2"] = opt_class(
        rng.normal(scale=1 / np.sqrt(HIDDEN_N), size=(HIDDEN_N, 1))
    )
    weight["b1"] = opt_class(np.zeros((HIDDEN_N, 1)))
    weight["b2"] = optimizer.Normalizer()
    return weight


def calc_psi(weight, xlist):
    """ネットワークを使ってpsiを計算"""
    fu1 = np.tanh(weight["w1"].w * xlist + weight["b1"].w)
    u2 = np.dot(weight["w2"].w.reshape(1, -1), fu1)
    return np.exp(u2 + weight["b2"].value)


def calc_train_psi(xlist):
    """step1のターゲットとなる波動関数"""
    return np.sqrt(np.maximum(0, 1 / 2 - np.abs(xlist) / 4))


def calc_exact_psi(xlist):
    """厳密解。あくまで比較用で学習には使用しない"""
    return np.pi ** (-1 / 4) * np.exp(-(xlist**2) / 2)


def metropolis(calc_p, randomwalk=True, sample_n = params.SAMPLE_N):
    """メトロポリス法で|psi|^2を確率分布関数にしてサンプル生成"""
    xlist = np.empty(sample_n, dtype=float)
    x = 0.0
    p = calc_p(x)
    assert p > 1e-10
    idx = 0
    while idx < sample_n:
        new_x = rng.normal(scale=2.0) + (x if randomwalk else 0)
        new_p = calc_p(new_x)
        if randomwalk and np.abs(new_x)>params.MAX_X:               # new_xがparamas.MAX_Xを超えていたら、xはサンプリングし直す
            x = rng.normal(scale=2.0)
            p = calc_p(x)
            continue                                                # while内のここより下にある処理を飛ばす
        elif new_p > p * rng.random():
            x, p = new_x, new_p
        xlist[idx] = x
        idx += 1
    print(p)
    return xlist, p


def update(weight, step, randomwalk):
    """一度だけ学習
    sampleN回のモンテカルロ積分で物理量を計算し、
    重みパラメータを更新する。
    新しい重みパラメーターとグラフ用のデータを返す
    """
    xlist, p = metropolis(lambda x: calc_psi(weight, x).ravel() ** 2, randomwalk)
    psi = calc_psi(weight, xlist)

    DX = params.DX
    psi_pm = calc_psi(weight, xlist + DX) + calc_psi(weight, xlist - DX)
    phii_H_psi = 2 / (DX**2) + xlist**2 - 1 / (DX**2) * psi_pm / psi
    E = np.average(phii_H_psi)

    train_psi = calc_train_psi(xlist) if step == 1 else calc_exact_psi(xlist)
    A1 = train_psi / psi
    A1_avg = np.average(A1)
    A2_avg = np.average(A1**2)
    K = A1_avg**2 / A2_avg

    weight["w2"].Ow = np.tanh(weight["w1"].w * xlist + weight["b1"].w)
    weight["b1"].Ow = weight["w2"].w * (1 - weight["w2"].Ow ** 2)
    weight["w1"].Ow = weight["b1"].Ow * xlist
    if step == 1:

        def update_func(Ow):
            #Ow_avg = np.average(Ow, axis=1, keepdims=True)
            Ow_avg = np.average(Ow, axis=1).reshape(params.HIDDEN_N, 1)
            #AOw_avg = np.average(Ow * A1, axis=1, keepdims=True)
            AOw_avg = np.average(Ow * A1, axis=1).reshape(params.HIDDEN_N, 1)
            return -2 * K * (AOw_avg / A1_avg - Ow_avg)

    else:

        def update_func(Ow):
            #Ow_avg = np.average(Ow, axis=1, keepdims=True)
            Ow_avg = np.average(Ow, axis=1).reshape(params.HIDDEN_N, 1)
            #piHp_Ow_avg = np.average(Ow * phii_H_psi, axis=1, keepdims=True)
            piHp_Ow_avg = np.average(Ow * phii_H_psi, axis=1).reshape(params.HIDDEN_N, 1)
            return 2 * (piHp_Ow_avg - Ow_avg * E)

    for key, w in weight.items():
        if key=="b2":
            w.update_rough(psi)
        elif isinstance(w, optimizer.Optimizer):
            w.update_weight(update_func)
        else:
            assert False
    return weight, K, E, p


def output_psi2(weight, L, N):
    """グラフ用の|psi|^2を計算"""
    xlist = np.linspace(-L / 2, L / 2, N, dtype=float)
    psi2 = calc_psi(weight, xlist).ravel() ** 2
    return xlist, psi2 / (np.sum(psi2) * L / N)
