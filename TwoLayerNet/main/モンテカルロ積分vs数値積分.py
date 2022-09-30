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


#　数値積分
# 確認済み

x, y = sym.symbols("x y")                   # 変数,定数を定義
fxy  = exp(-x**2/2)                                # 被積分関数を定義

Fx = sym.integrate(fxy, x)                   # 不定積分
print(Fx)

I = sym.integrate(fxy, (x, -np.inf, np.inf))        # 定積分 
print(I)







# モンテカルロ積分
def Overlap(y, t):                          # 確認済み
    #print("y: " + str(y.shape))
    #print("t: " + str(t.shape))
    N_sample = y.shape[0]                   # xの行の数(=M)
    t_per_y = np.sum(t/y)
    t2_per_y2 = np.sum(t**2/y**2)
    K = 1/N_sample * t_per_y **2 / t2_per_y2
    return K

