# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:18:00 2022

@author: yamep
"""
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__)))) 
from common.layers import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import sympy as sym
from sympy.plotting import plot
sym.init_printing(use_unicode=True)
from sympy import sin, cos, tan, log, exp

"""
#　数値積分
# 波動関数 N=1,2 で ∫ψ^2=1 になることを確認済み

x, y = sym.symbols("x y")                   # 変数,定数を定義
Pxy  = np.power(np.pi, -1) * exp(-(x**2+y**2))                                # 被積分関数を定義

Fx = sym.integrate(Pxy, x, y)                   # 不定積分
print(Fx)

I = sym.integrate(Pxy, (x, -np.inf, np.inf), (y, -np.inf, np.inf))        # 定積分 
print(I)
"""

x = sym.symbols("x")
psi_exact = np.power(np.pi, -1/4) * exp(-(x**2)/2)
psi_train = np.power(np.pi, -1/4) * exp(-(x**2)/3)
# 数値積分での(正しい)オーバーラップ積分 N=1, psi_train=p.power(np.pi, -1/4) * exp(-(x**2)/3) で K=0.4*sqrt(6) を確認済み
K = sym.integrate(psi_train * psi_exact, (x, -np.inf, np.inf))**2 / ( sym.integrate(psi_train**2, (x, -np.inf, np.inf)) * sym.integrate(psi_exact**2, (x, -np.inf, np.inf)) )    
#print(K)

N = 1
# メトロポリス法
i = 100000

M = int(i/10)
x = np.zeros(N)
def p(x, N):
    return ( wave_func(x, N) )**2
#sdata= np.empty((int(i/10)+1, N))
sdata= np.ones((M, N))
cnt=0
naccept = 0
for _ in range(i):
    y = x + np.random.uniform(-1,1,N)
    alpha = min(1, p(y, N)/p(x, N))
    r = np.random.uniform(0,1)
    if r > alpha:                   # 棄却
        y = x
    else:                           # 受理
        naccept += 1
        x = y
    #print(x)
    cnt += 1
    if cnt%10==0:
        sdata[int(cnt/10)-1]= x

#print(sdata.shape)
psi_metro_train = np.array(np.power(np.pi, -1/4) * np.exp(-(sdata**2)/3))
psi_metro_exact = np.array(np.power(np.pi, -1/4) * np.exp(-(sdata**2)/2))
#print(psi_metro_train.shape)
#print(psi_metro_train)
#print(psi_metro_exact.shape)
#print("提案受理回数: " + str(naccept))
#print("提案受理率: " + str(naccept/i *100) + "%")

x_array = np.arange(0, M, 1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig.subplots_adjust(right=0.5)
ax.set_title("error (y-t)/t", fontname="MS Gothic")             # タイトル
ax.set_xlabel('iter_index i')                                   # x軸ラベル  
ax.set_ylabel('error (y-t)/t')                                  # y軸ラベル
#ax.text(1.1, 0.5, condition, ha='left', va='center', transform=ax.transAxes, fontname="MS Gothic")   #表示するテキスト

ax.plot(x_array, sdata)                                # x軸,y軸に入れるリスト
plt.show()



# モンテカルロ積分
def Overlap(y, t):                          # 確認済み
    #print("y: " + str(y.shape))
    #print("t: " + str(t.shape))
    N_sample = y.shape[0]                   # xの行の数(=M)
    t_per_y = np.sum(t/y)
    t2_per_y2 = np.sum(t**2/y**2)
    K = 1/N_sample * t_per_y **2 / t2_per_y2
    return K


K_metro = Overlap(psi_metro_train, psi_metro_exact)
#print(K_metro)

error = K_metro - 0.4*np.sqrt(6)
#print("サンプル数" + str(M))
#print("誤差: " + str(error))



