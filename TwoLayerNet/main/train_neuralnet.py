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
import time

time_sta = time.perf_counter()


# データの読み込み
# メトロポリス法
N = 5                                             # 粒子数=入力層ユニット数
def p(x, N):
    return ( wave_func(x, N) )**2


M = 100                                           # 全サンプル数
i = int(M*10)
sdata= np.zeros((M+1, N))

x_train = np.zeros((M, N))
x_test = np.zeros((M, N))

x = np.zeros(N)
cnt=0



# テスト用入力データx_testの生成
cnt = 0
x = np.zeros(N)
for cnt in range(i):
    y = x + np.random.uniform(-1,1,N)               # ランダム関数
    alpha = min(1, p(y, N)/p(x, N))
    r = np.random.uniform(0,1)
    if r > alpha:
        y = x
    x = y
    cnt += 1
    if cnt%10==0:
        x_test[int(cnt/10 -1)]= x

t_train = wave_func(x_train, N).reshape(-1, 1)
t_test = wave_func(x_test, N).reshape(-1, 1)
#print("x_train=" + str(x_train))
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
network = TwoLayerNet(input_size=N, hidden_size=40, output_size=1)


train_size = x_train.shape[0]                           # 全サンプル数
batch_size = 10**2                                       # バッチサイズ
iter_per_epoch = max(train_size / batch_size, 1)        # 1エポックの更新回数
iters_num = 10**5                                       # 全更新回数
learning_rate = 0.01                                  # 学習率

train_loss_list = []
train_err_list =[]
train_err_list = []
train_y_list = []
train_err_list = []
test_err_list = []
train_overlap_list = []
test_overlap_list = []




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
print(str(condition) + "\n")
     
def p_improve(x):
    return network.predict(x)**2            

# 学習
for i in range(iters_num):
    print(i)
    
    # 訓練用入力データx_trainの生成
    for cnt in range(i):
        #print("x=" + str(x))
        y = x + np.random.uniform(-1,1,N)           # ランダム関数
        alpha = min(1, p_improve(y)/p_improve(x))
        r = np.random.uniform(0,1)
        if r > alpha:
            y = x
            x = y
            cnt += 1
        if cnt%10==0:
            x_train[int(cnt/10 -1)] = x
    
    batch_mask = np.random.choice(train_size, batch_size)   # 0からtrain_sizeまでの整数をランダムにbatch_size個抽出して1次元配列にする
    #print("batch_mask=" + str(batch_mask))
    x_batch = x_train[batch_mask]
    #print(x_batch.shape)                                    # (batch_size, N)
    #print("x_bathch " + str(x_batch))
    t_batch = t_train[batch_mask]
    #print(t_batch.shape)                                    # (batch_size, 1)
    
    # 勾配
    grad = network.gradient(x_batch, t_batch)               # ミニバッチから勾配を計算
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]       # パラメータを更新 
    
    loss = network.loss(x_batch, t_batch)                  # ミニバッチのみから損失関数を計算
    #loss = network.loss(x_train, t_train)                   # 全データから損失関数を計算
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:                             # 1エポックの更新回数に達した場合の処理
        train_err = network.error(x_batch, t_batch)
        test_err = network.error(x_test, t_test)
        train_overlap = network.overlap(x_batch, t_batch)   # オーバーラップ積分の値
        test_overlap = network.overlap(x_test, t_test)      # オーバーラップ積分の値
        diff = network.diff(x_batch, t_batch)
        print("y-t=" + str(diff))
        
        train_err_list.append(train_err)
        test_err_list.append(test_err)
        train_overlap_list.append(train_overlap)            # リストに格納
        test_overlap_list.append(test_overlap)
        
        train_y = network.predict(x_batch)
        train_y_list.append(train_y) 
        
        
        #print("i=" + str(i) + ": " + str(train_err))
        #print(train_err)                                        # 正しい誤差の平均が表示される
        
        
print(train_overlap_list)
x_array = np.arange(0, iters_num, iter_per_epoch)
#print(len(train_overlap_list))
#print(train_overlap_list)                                     # nanになってる

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig.subplots_adjust(right=0.5)
#ax.set_title("error (y-t)/t", fontname="MS Gothic") 
ax.set_title("overlap K")            # タイトル
ax.set_xlabel('iter_index i')                                   # x軸ラベル  
#ax.set_ylabel('error (y-t)/t')                                  # y軸ラベル
ax.set_ylabel('overlap K')
#ax.set_xlim(0, iters_num)
#ax.set_ylim(-10, 2)
ax.text(1.1, 0.5, condition, ha='left', va='center', transform=ax.transAxes, fontname="MS Gothic")   #表示するテキスト

#ax.plot(x_array, train_err_list)
ax.plot(x_array, train_overlap_list)                                # x軸,y軸に入れるリスト
plt.show()


"""
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
"""

time_end = time.perf_counter()
tim = time_end- time_sta
print("実行時間: " + str(tim) + " sec")
