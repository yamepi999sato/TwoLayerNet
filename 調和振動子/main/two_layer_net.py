# coding: utf-8
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 1):
        # 重みの初期化
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Tanh'] = Tanh()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Exp2'] = Exp()

        #self.lastLayer = SoftmaxWithLoss()
        self.lastLayer = Minus_Overlap()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            #print(str(layer) + ".forward out:\n" + str(x) + "\n")
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        #print(str(self.lastLayer) + ".forward out:\n" + str(y) + "\n")
        return self.lastLayer.forward(y, t)
    
    def error(self, x, t):                              # 相対誤差 (y-t)/t
        y = self.predict(x)
        error = np.mean((y-t)/t)
        #print("error")
        return error
    
    def diff(self, x, t):                               # 絶対誤差 y-t
        y = self.predict(x)
        diff = y - t
        return diff
    
    def overlap(self, y, t):
        K = Overlap(y, t)
        return K
        
    # x:入力データ, t:教師データ        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        #print(str(self.lastLayer) + ".backward out:\n" + str(dout) + "\n")
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            #print(str(layer) + ".backward out:\n" + str(dout) + "\n")

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
