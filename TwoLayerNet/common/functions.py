# coding: utf-8
import numpy as np



def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def wave_func(x, N):                           # 確認済み
    if x.ndim == 2:
        out = np.power(np.pi, -N/4) * np.exp(- np.sum((x**2)/2, axis=1))
    elif x.ndim == 1:
        out = np.power(np.pi, -N/4) * np.exp(- np.sum((x**2)/2, axis=0))
    return out

def Overlap(y, t):                          # 確認済み
    #print("y: " + str(y.shape))
    #print("t: " + str(t.shape))
    N_sample = y.shape[0]                   # xの行の数(=M)
    t_per_y = np.sum(t/y)                   # yが0だと上手く動作しない
    t2_per_y2 = np.sum(t**2/y**2)
    K = 1/N_sample * t_per_y **2 / t2_per_y2
    return K

def diff_Overlap(y, t):                     # 確認済み
    N_sample = y.shape[0]                   # xの行の数(=M)
    t_per_y = np.sum(t/y)
    t2_per_y2 = np.sum(t**2/y**2)
    diff = 2/N_sample * (t_per_y * t**2/y**3 - t2_per_y2 * t/y**2)
    return diff

x = np.array([[1],[1]])
t = np.array([[2],[3]])
print(Overlap(x, t))
#print(diff_Overlap(x, t))

