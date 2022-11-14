# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        #print("Affine forward out: " + str(out) + "\n")

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        #print("Affine backward out: " + str(dx) + "\n")
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
        #print("Minus_Overlap forward out: " + str(out) + "\n")
        return self.out
        

    def backward(self, dout=1):
        dx = - diff_Overlap(self.y, self.t)
        #print("Minus_Overlap backward out: " + str(dx) + "\n")
        return dx


"""        
class numerical_minus_Overlap():
    def __init__(self):
"""        
    
class Tanh:
    def __init__(self):
        self.x = None
        self.out = None
        
    def forward(self, x):
        self.x = x
        self.out = np.tanh(x)
        #print("Tanh forward out: " + str(self.out) + "\n")
        return self.out
        
    
    def backward(self, dout):
        dx = (1- dout **2)
        #print("Tanh backward out: " + str(dx) + "\n")
        return dx
        

class Exp:
    def __init__(self):
        self.x = None
        self.out = None
        
    def forward(self, x):
        self.x = x
        out = np.exp(x)
        self.out = out
        #print("exp_out: " + str(self.out.shape))
        #print("Exp forward out: " + str(self.out) + "\n")
        return self.out
        
    
    def backward(self, dout):
        dx = dout
        #print("Exp backward out: " + str(dx) + "\n")
        return dx


