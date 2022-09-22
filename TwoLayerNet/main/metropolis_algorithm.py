# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:55:38 2022

@author: yamep
"""

import random
import math
import matplotlib.pyplot as plt

def p(x):
    return 1/math.sqrt(2*math.pi)*math.exp(-x**2/2)

x = 0
sdata=[ ]
cnt=0
for _ in range(1000000):
    y = x + random.uniform(-1,1)
    alpha = min(1, p(y)/p(x))
    r = random.uniform(0,1)
    if r > alpha:
        y = x
    x = y
    cnt += 1
    if cnt%10==0:
        sdata.append(x)

xdata=[]
ydata=[]
x=-5.0
while x<5.0:
    xdata.append(x)
    ydata.append(p(x))
    x += 0.01

plt.title("NORMAL DISTRIBUTION")
plt.plot(xdata,ydata,color=(0.0,0.0,0.7))
plt.hist(sdata, bins=100, density=True, color=(1.0,0,0.0))
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.grid(True)
plt.show()