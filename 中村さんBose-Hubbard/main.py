# 以下3行だけ山本加筆
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import parameters as params
import gutzwiller as gw
from parameters import *

data = None
J_list = np.linspace(0.0, J_MAX, GRID)
mu_list = np.linspace(0.0, MU_MAX, GRID)
beta_list = np.empty([GRID,GRID])
for yi in range(GRID):
    data = None
    for xi in reversed(range(GRID)):
        data = gw.calc(mu_list[yi], J_list[xi], data) 
        beta_list[yi, xi] = data["beta"] if data["OK"] else None

X, Y = np.meshgrid(J_list, mu_list)
plt.pcolormesh(X, Y, beta_list)
plt.colorbar()
plt.xlabel('J/U')
plt.ylabel('mu/U')
plt.show()