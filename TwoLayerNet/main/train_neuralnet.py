# coding: utf-8
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__)))) 
import numpy as np
from common.layers import *
from dataset.mnist import load_mnist
from main.two_layer_net import TwoLayerNet

"""
# データの読み込み
# 元のコード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
"""
"""
# データの読み込み
# 一様乱数
x_train = np.random.rand(60000, 3)          # x_train.shape = (サンプル数, 粒子数)
t_train = np.random.rand(60000, 1)
x_test = np.random.rand(60000, 3)          # x_train.shape = (サンプル数, 粒子数)
t_test = np.random.rand(60000, 1)
"""

# データの読み込み
# メトロポリス法
def p(x):
    return np.power(2/np.pi, 1/4) * wave_func(x)

N = 2
i = 100
M = int(i/10)
#sdata= np.empty((int(i/10)+1, N))

x_train = np.empty((M, N))
x_test = np.empty((M, N))
print(x_train)
print("\n")

x = np.zeros(N)
cnt=0

for _ in range(i):
    y = x + np.random.uniform(-1,1,N)
    alpha = min(1, p(y)/p(x))
    r = np.random.uniform(0,1)
    if r > alpha:
        y = x
    x = y
    cnt += 1
    if cnt%10==0:
        x_train[int(cnt/10)-1] = x

cnt = 0
x = np.zeros(N)
for _ in range(i):
    y = x + np.random.uniform(-1,1,N)
    alpha = min(1, p(y)/p(x))
    r = np.random.uniform(0,1)
    if r > alpha:
        y = x
    x = y
    cnt += 1
    if cnt%10==0:
        x_test[int(cnt/10)-1]= x

t_train = wave_func(x_train).reshape(-1, 1)
t_test = wave_func(x_test).reshape(-1, 1)
print(x_train)
#print(t_train)
"""
print(x_train.shape)        # (M, N)
print(t_train.shape)        # (M, 1)
print(x_test.shape)         # (M, N)
print(t_test.shape)         # (M, 1)
"""
"""
network = TwoLayerNet(input_size=N, hidden_size=50, output_size=1)

iters_num = 2
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
#print(train_size)


train_loss_list = []
train_acc_list = []
test_acc_list = []

train_err_list =[]
train_err_list = []
#train_err_list = np.zeros(iters_num)
#test_err_list = np.zeros(iters_num)

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)   # 0からtrain_sizeまでの整数をランダムにbatch_size個抽出して1次元配列にする
    #print(batch_mask)
    x_batch = x_train[batch_mask]
    #print(x_batch.shape)                                    # (batch_size, N)
    t_batch = t_train[batch_mask]
    #print(t_batch.shape)                                    # (batch_size, 1)
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        j = int(i/ iter_per_epoch)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        train_err = network.error(x_train, t_train)
        test_err = network.error(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        #train_err_list[j] = train_error
        #test_err_list[j] = test_error
        print(train_err)
"""