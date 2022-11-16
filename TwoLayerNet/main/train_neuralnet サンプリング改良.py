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

# パラメータ定義
N = 5                                             # 粒子数=入力層ユニット数
M = 1000                                              # 1回の学習時に作成するサンプル数
autocorrelation_length = 10                         # 自己相関長
M_total = int(M * autocorrelation_length)           # 破棄するものも含めて1回の学習時にトータルで作成するサンプル
train_size = M                                       # 1回の学習時に作成するサンプル数
iters_num = 1000                                       # 全更新回数
learning_rate = 0.01                                  # 学習率

train_loss_list = []
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
iters_num(全更新回数): " + str(iters_num) + "回\n\
learning_rate(学習率): " + str(learning_rate)
print(str(condition) + "\n")
     

network = TwoLayerNet(input_size=N, hidden_size=40, output_size=1)           
# 学習
for iters_index in range(iters_num):
    def p(x):
        y = network.predict(x)
        return y**2
    
    x_train = np.zeros((M, N))        
    x = np.ones(N)
    metro_cnt=0

    # 訓練用入力データx_trainの生成
    for metro_cnt in range(M_total):
        #print("x=" + str(x))
        y = x + np.random.uniform(-1,1,N)           # ランダム関数
        #print(y-x)
        alpha = min(1, p(y)/p(x))
        r = np.random.uniform(0,1)
        if r > alpha:
            y = x
        x = y
        metro_cnt += 1
        if metro_cnt % 1 == 0:
            x_train[int(metro_cnt/10 -1)] = x
    
    if iters_index % 10 == 0:
        print(iters_index)
        #print(x_train)
    t_train = wave_func(x_train, N).reshape(-1, 1)

    

    # 勾配
    grad = network.gradient(x_train, t_train)               # ミニバッチから勾配を計算

    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]       # パラメータを更新 
    loss = network.loss(x_train, t_train)                  # 損失関数を計算
    train_loss_list.append(loss)

    
    # 計算結果をリストに格納
    if 1:                             # 1エポックの更新回数に達した場合の処理
        train_err = network.error(x_train, t_train)
        #test_err = network.error(x_test, t_test)
        train_overlap = network.overlap(x_train, t_train)   # オーバーラップ積分の値
        #test_overlap = network.overlap(x_test, t_test)      # オーバーラップ積分の値
        diff = network.diff(x_train, t_train)
        #print("y-t=" + str(diff))
        
        train_err_list.append(train_err)
        #test_err_list.append(test_err)
        train_overlap_list.append(train_overlap)            # リストに格納
        #test_overlap_list.append(test_overlap)
        
        train_y = network.predict(x_train)
        train_y_list.append(train_y) 
        
                
iters_array = np.arange(0, iters_num, 1)


# グラフを表示
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig.subplots_adjust(right=0.5)
#ax.set_title("error (y-t)/t", fontname="MS Gothic") 
ax.set_title("overlap K")                                       # タイトル
ax.set_xlabel('iter_index i')                                   # x軸ラベル：ループのインデックス  
#ax.set_ylabel('error (y-t)/t')                                  # y軸ラベル：相対誤差
ax.set_ylabel('overlap K')                                      # y軸ラベル：オーバーラップ積分            
ax.text(1.1, 0.5, condition, ha='left', va='center', transform=ax.transAxes, fontname="MS Gothic")   #表示するテキスト
#ax.plot(iters_array, train_err_list)                                   # x軸,y軸に入れるリスト
ax.plot(iters_array, train_overlap_list)                                # x軸,y軸に入れるリスト
plt.show()



time_end = time.perf_counter()
tim = time_end- time_sta
print("実行時間: " + str(tim) + " sec")
