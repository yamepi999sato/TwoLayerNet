# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:05:53 2022

@author: 1637460
"""

import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__)))) 
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from common.layers import *
from dataset.mnist import load_mnist
from main.two_layer_net import TwoLayerNet
import time

x = np.log(np.e)
exp = Exp()
y = exp.forward(x)
print(x)
print(y)

dout = y
dx = exp.backward(dout)


print(dx)


