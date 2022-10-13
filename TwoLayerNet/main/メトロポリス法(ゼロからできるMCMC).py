# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:35:05 2022

@author: yamep
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy.plotting import plot
import time

time_sta = time.perf_counter()
niter = 100000
step_size = 0.5
x = 0
naccept = 0
x_data = np.empty(0)
sum_x = 0
expect_x = 0
expect_x_array = np.empty(0)
sum_x2 = 0
expect_x2 = 0
expect_x2_array = np.empty(0)


for iter in range(niter):
    x_data = np.append(x_data, x)
    backup_x = x
    #action_init = 0.5 * x**2
    action_init = x**2                                     # 作用
    dx = np.random.uniform(-step_size, step_size)
    
    x += dx
    #action_fin = 0.5 * x**2                                 # 作用
    action_fin = x**2 
    
    metropolis = np.random.uniform(0, 1)
    if np.exp(action_init - action_fin) > metropolis:       # 受理
        naccept += 1
    else:                                                   # 棄却
        x = backup_x
    
    sum_x += x
    expect_x = sum_x / (iter + 1)
    expect_x_array = np.append(expect_x_array, expect_x)
    sum_x2 += x**2
    expect_x2 = sum_x2 / (iter + 1)
    expect_x2_array = np.append(expect_x2_array, expect_x2)
    
#print(naccept)
#print(iter)



edge = 3                                                    # 範囲の端
step = 0.1                                                   # 間隔
x_array = np.arange(-edge, edge, step)
#y_array = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * x_array**2)
y_array = 1/np.sqrt(np.pi) * np.exp(- x_array**2)

#print(x_data)




plt.title("Metropolis sampling of Gaussian, number of iteration = " + str(niter) + " (p.55)")
#plt.title("Metropolis sampling of , number of iteration = " + str(niter) + " (p.55)")
#plt.plot(x_array,y_array,color=(0.0,0.0,0.7), label='Gaussian')
plt.hist(x_data, bins=100, density=True, color=(1.0,0,0.0), label='sample')
plt.plot(np.arange(0, niter), expect_x_array, label='<x>')
plt.plot(np.arange(0, niter), expect_x2_array, label='<x^2>')
#plt.xlabel('x')
plt.xlabel('K')
#plt.ylabel('y')
plt.legend()
#plt.ylim(-0.5, 2)
plt.grid(True)
plt.show()


time_end = time.perf_counter()
tim = time_end- time_sta
print("実行時間: " + str(tim) + " sec")
