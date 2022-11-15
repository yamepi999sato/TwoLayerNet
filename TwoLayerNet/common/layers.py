# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b     
        self.x = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Minus_Overlap():
    def __init__(self):
        self.y = None
        self.t = None
        self.out = None
    
    def forward(self, y, t):
        self.y = y
        self.t = t
        out = - Overlap(self.y, self.t)
        self.out = out
        return self.out

    def backward(self, dout=1):
        dx = - diff_Overlap(self.y, self.t)
        return dx

    
class Tanh:
    def __init__(self):
        self.x = None
        self.out = None
        
    def forward(self, x):
        self.x = x
        self.out = np.tanh(x)
        return self.out       
    
    def backward(self, dout):
        dx = (1- dout **2)
        return dx
        

class Exp:
    def __init__(self):
        self.x = None
        self.out = None
        
    def forward(self, x):
        self.x = x
        out = np.exp(x)
        self.out = out
        return self.out
        
    def backward(self, dout):
        dx = dout
        return dx


