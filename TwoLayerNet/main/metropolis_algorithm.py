# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:55:38 2022

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

N = 2
def p(x, N):
    return ( wave_func(x, N) )**2


i = 1000
M = int(i/10)
x = np.zeros(N)
#sdata= np.empty((int(i/10)+1, N))
sdata= np.ones((M, N))
cnt=0
for _ in range(i):
    y = x + np.random.uniform(-1,1,N)
    alpha = min(1, p(y, N)/p(x, N))
    r = np.random.uniform(0,1)
    if r > alpha:
        y = x
    x = y
    #print(x)
    cnt += 1
    if cnt%10==0:
        sdata[int(cnt/10)-1]= x
    

#print(sdata)
split = 100
xdata= np.zeros(split)
ydata= np.zeros(split)
t = -5.0
cnt = 0
for cnt in range(split):
    xdata[cnt] = t
    ydata[cnt] = p(np.array([t, t]), N)
    t += 10/split
#print(xdata.shape)
#print(ydata.shape)
    
#print(sdata.shape)

plt.title("Metropolis sampling of Ψ(x_1)^2")
plt.plot(xdata,ydata,color=(0.0,0.0,0.7))
plt.hist(sdata[:, 0], bins=100, density=True, color=(1.0,0,0.0))
plt.text(4, 0.5, "Hello")
plt.xlabel('x_1')
plt.ylabel('P(x_1)')
plt.legend()
#plt.ylim(-0.5, 2)
plt.grid(True)
plt.show()
"""

s = "Hello, world"

x = 0.4
y = 0.6
s = "Hello, world"

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 2, 3, 4, 5]
plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.grid()
plt.text(4, 3, "y=x")
plt.show()
"""