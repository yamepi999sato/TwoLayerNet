# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:01:47 2022

@author: yamep
"""
import numpy as np

y = np.array([[1.10106816],
 [1.07064206],
 [0.02032781],
 [0.77706744],
 [0.25933933]])


t = np.array([[0.66563127],
 [0.42966069],
 [0.07798092],
 [0.54939895],
 [0.42850905]])


error = np.mean((y-t)/t)
err = 0
for i in range(len(y)):
    e = (y[i]-t[i])/t[i]
    print(e)
    err += e
    
err = err/len(y)
print(err)          # 正しい誤差が表示される
print(error)        # 正しい誤差が表示される