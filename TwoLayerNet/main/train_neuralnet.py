# coding: utf-8
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
"""
研究室動作確認　from研究室PC
"""

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
N = 3
def p(x, N):
    return ( wave_func(x, N) )**2


i = 10000
M = int(i/10)
sdata= np.empty((M+1, N))

x_train = np.empty((M, N))
x_test = np.empty((M, N))

x = np.zeros(N)
cnt=0

for _ in range(i):
    y = x + np.random.uniform(-0.1,0.1,N)
    alpha = min(1, p(y, N)/p(x, N))
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
    alpha = min(1, p(y, N)/p(x, N))
    r = np.random.uniform(0,1)
    if r > alpha:
        y = x
    x = y
    cnt += 1
    if cnt%10==0:
        x_test[int(cnt/10)-1]= x

t_train = wave_func(x_train, N).reshape(-1, 1)
t_test = wave_func(x_test, N).reshape(-1, 1)
#print(x_train)
#print(t_train)
"""
print(x_train.shape)        # (M, N)
print(t_train.shape)        # (M, 1)
print(x_test.shape)         # (M, N)
print(t_test.shape)         # (M, 1)
"""
"""
split = 100
xdata= np.empty(split)
ydata= np.empty(split)
t = -5.0
cnt = 0
for cnt in range(split):
    xdata[cnt] = t
    ydata[cnt] = p(np.array([t, t, t]), N)
    t += 10/split
    
plt.title("Metropolis sampling of Ψ(x_1, x_2, x_3)^2, N=" + str(N) + ", M=10000" + ", x_1=x_2=x_3")
plt.plot(xdata,ydata,color=(0.0,0.0,0.7))
plt.hist(x_train[:, 0], bins=100, density=True, color=(1.0,0,0.0))
plt.xlabel('x_1=x_2=x_3')
plt.ylabel('P(x_1, x_2, x_3)=Ψ^2')
plt.legend()
#plt.ylim(-0.5, 2)
plt.grid(True)
plt.show()

"""
network = TwoLayerNet(input_size=N, hidden_size=5, output_size=1)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 5
learning_rate = 0.0001
#print(train_size)


train_loss_list = []
train_acc_list = []
test_acc_list = []

train_err_list =[]
train_err_list = []
train_y_list = []
train_err_list = []
test_err_list = []

iter_per_epoch = max(train_size / batch_size, 1)
"""
#条件を表示
print("input_size(入力層ユニット数) = N(粒子数): " + str(N) + "(個)")
print("hidden_size(隠れ層ユニット数): " + str(network.hidden_size) + "個")
print("output_size(出力層ユニット数): " + str(network.output_size) + "個")
print("train_size(全サンプル数): " + str(train_size) + "個")
print("batch_size(バッチサイズ): " + str(batch_size) + "個")
print("iter_per_epoch(1エポックの更新回数): " + str(iter_per_epoch) + "回")
print("iters_num(全更新回数): " + str(iters_num) + "回")
print("learning_rate(学習率): " + str(learning_rate))
print("\n")
print("train_err(誤差)")
"""

#条件を表示
condition = \
"input_size(入力層ユニット数) = N(粒子数): " + str(N) + "(個)\n\
hidden_size(隠れ層ユニット数): " + str(network.hidden_size) + "個\n\
output_size(出力層ユニット数): " + str(network.output_size) + "個\n\
train_size(全サンプル数): " + str(train_size) + "個\n\
batch_size(バッチサイズ): " + str(batch_size) + "個\n\
iter_per_epoch(1エポックの更新回数): " + str(iter_per_epoch) + "回\n\
iters_num(全更新回数): " + str(iters_num) + "回\n\
learning_rate(学習率): " + str(learning_rate)

print(condition)
print("train_err(誤差)")      
            
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
    #if 1:
        j = int(i/ iter_per_epoch)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        
        train_err = network.error(x_batch, t_batch)
        test_err = network.error(x_test, t_test)
        train_err_list.append(train_err)
        test_err_list.append(test_err)
        
        train_y = network.y(x_batch)
        train_y_list.append(train_y) 
        
        #print("train_y :" + str(train_y))
        #print("train_t: " + str(t_batch))
        #print((train_y-t_batch)/t_batch)                        # 正しい誤差の配列が表示される
        #print("err: " + str(np.mean(train_y-t_batch)/t_batch))  # 正しい誤差の平均が表示されない
        
        print("i=" + str(i) + ": " + str(train_err))
        #print(train_err)                                        # 正しい誤差の平均が表示される

x_array = np.arange(0, iters_num, iter_per_epoch)
        
plt.title("error (y-t)/t", fontname="MS Gothic")
plt.plot(x_array,train_err_list,color=(0.0,0.0,0.7))
#plt.hist(x_train[:, 0], bins=100, density=True, color=(1.0,0,0.0))
plt.xlabel('iter_index i')
plt.ylabel('error (y-t)/t')
plt.text(4, 3, condition, fontname="MS Gothic")
plt.legend()
#plt.ylim(-0.5, 2)
plt.grid(True)
plt.show()