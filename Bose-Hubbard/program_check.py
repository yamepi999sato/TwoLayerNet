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


# ここからループ内

ab = random.choice(random_ab)
pm = random.choice(random_pm)
e_i = np.zeros((params.M, 1))
e_j = np.zeros((params.M, 1))
i = random.choice(random_i)

print(ab)
print(pm)
print("i=" + str(i))

e_i[i] = 1
if i+1 < params.M:
    e_j[i+1] = 1
else:
    e_j[0] = 1


if ab == "a":
    if pm == "+":
        new_n_vec = n_vec + e_i
    elif pm == "-":
        new_n_vec = n_vec - e_i
elif ab == "b":
    if pm == "+":
        new_n_vec = n_vec + e_i - e_j
    elif pm == "-":
        new_n_vec = n_vec - e_i + e_j


print("new_n_vec=" + str(new_n_vec))
