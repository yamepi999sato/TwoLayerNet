# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:55:38 2022

@author: yamep
"""
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__)))) 
from common.layers import *
import random
import math
import matplotlib.pyplot as plt

def p(x):
    return np.power(2/np.pi, 1/4) * wave_func(x)

N = 2
i = 10
M = int(i/10)
x = np.zeros(N)
#sdata= np.empty((int(i/10)+1, N))
sdata= np.ones((M, N))
cnt=0
for _ in range(i):
    y = x + np.random.uniform(-1,1,N)
    alpha = min(1, p(y)/p(x))
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
    ydata[cnt] = p(np.array([t, t]))
    t += 10/split
#print(xdata.shape)
#print(ydata.shape)
    
#print(sdata.shape)

plt.title("NORMAL DISTRIBUTION")
plt.plot(xdata,ydata,color=(0.0,0.0,0.7))
plt.hist(sdata[:, 0], bins=100, density=True, color=(1.0,0,0.0))
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.grid(True)
plt.show()
