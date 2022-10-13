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

    
"""
# SoftmaxWithLossとSoftmax,Cross_Entropy_Errorの動作確認
x = np.array([[1, 2], [3, 4]])
t = np.array([5, 6])
Soft = SoftmaxWithLoss()
loss1 = Soft.forward(x, t)
#print(loss1)
"""

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
        
        
class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


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
        dx = dout * (1- self.x **2)
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


