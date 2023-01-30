import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import parameter as params
import random

time_start = time.time()
"""
#(n_1, 1, 1)
nlist_n11 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_n11[0,n] = n
for i in range(0, params.M):
    if i != 0:
        nlist_n11[i] = 1
print(nlist_n11)

# (1, n_1, 1)
nlist_1n1 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_1n1[1,n] = n
for i in range(0, params.M):
    if i != 1:
        nlist_1n1[i] = 1
print(nlist_1n1)

# (1, 1, n_3)
nlist_11n = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_11n[2,n] = n
for i in range(0, params.M):
    if i != 2:
        nlist_11n[i] = 1
print(nlist_11n)


# Ψ(n_1, 2, 2)
nlist_n22 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_n22[0,n] = n
for i in range(0, params.M):
    if i != 0:
        nlist_n22[i] = 2
print(nlist_n22)

# (n_1, 3, 3)
nlist_n33 = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist_n33[0,n] = n
for i in range(0, params.M):
    if i != 0:
        nlist_n33[i] = 3
print(nlist_n33)
"""

n_vec = np.zeros((params.M, 1))

random_ab = ["a", "b"]
random_i = np.arange(0, params.M) 
random_pm = ["+", "-"]


e = np.zeros((params.M, 1))
i = random.choice(random_i)
e[i] = 1
e_i = e

e[i] = 0
if i+1 < params.M:
    e[i+1] = 1
else:
    e[0] = 1
e_j = e
print(i)
print(e_i)
print(e_j) 
"""
if random.choice(random_ab) == "a":
    if random.choice(random_pm) == "+":
        new_n_vec = n_vec + e
    elif random.choice(random_pm) == "-":
        new_n_vec = n_vec - e
elif random.choice(random_ab) == "b":
    if random.choice(random_pm) == "+":
        new_n_vec = n_vec + e
    elif random.choice(random_pm) == "-":
        new_n_vec = n_vec - e
print(e)
print(new_n_vec)
"""
from matplotlib import pyplot as plt
import numpy as np
 
# ランダムな点を生成する(x, y, z座標)
n_1 = np.arange(0, params.N_P+1)
n_2 = np.arange(0, params.N_P+1)




n_3 = np.arange(0, params.N_P+1)

# 点(x, y, z)がもつ量
value = np.random.rand(params.N_P+1)
#value = np.ones(params.N_P)

 
# figureを生成する
fig = plt.figure()
 
# axをfigureに設定する
ax = fig.add_subplot(1, 1, 1, projection='3d')
 
# カラーマップを生成
cm = plt.cm.get_cmap('RdYlBu')
 
# axに散布図を描画、戻り値にPathCollectionを得る
mappable = ax.scatter(n_1, n_2, n_3, c=value, cmap=cm)
fig.colorbar(mappable, ax=ax)
 
# 表示する
plt.show()