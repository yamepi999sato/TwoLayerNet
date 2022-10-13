# coding: utf-8
import numpy as np

 

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
#print(Overlap(x, t))
#print(diff_Overlap(x, t))
day = 0.9898525065970256
error = (day - 0.4*np.sqrt(6) )/(0.4*np.sqrt(6))
print(error)

