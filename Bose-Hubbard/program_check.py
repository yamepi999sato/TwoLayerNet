import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import parameter as params

time_start = time.time()

nlist = np.zeros((params.M, params.N_P+1))
for n in range(params.N_P+1):
    nlist[0,n] = n
for i in range(1, params.M):
    nlist[i] = 1
print(nlist)