# 以下3行だけ山本加筆
import sys, os
from os.path import dirname, abspath
sys.path.append(dirname(abspath(__file__)))

from abc import ABC, abstractmethod
import numpy as np
import parameter as params


class Normalizer:
    def __init__(self):
        self.reset_internal_params()

    def reset_internal_params(self):
        self.value = 0
    
    def update_rough(self, psi):
        self.value -= 0.5*np.log(np.average(psi.ravel() ** 2)) 


class Optimizer(ABC):
    """重みパラメータwの更新"""

    def __init__(self, w):
        self.w = w
        self.eta = params.ETA
        self.Ow = np.empty((params.HIDDEN_N, np.shape(w)[0]))
        self.reset_internal_params()

    @abstractmethod
    def reset_internal_params(self):
        pass

    @abstractmethod
    def update_weight(self, update_func):
        pass


class SGD(Optimizer):
    def reset_internal_params(self):
        pass

    def update_weight(self, update_func):
        self.w -= self.eta * update_func(self.Ow)


class Momentum(Optimizer):
    def reset_internal_params(self):
        self.m = np.zeros_like(self.w)
        self.alpha = 0.9

    def update_weight(self, update_func):
        self.m *= self.alpha
        self.m -= self.eta * update_func(self.Ow)
        self.w += self.m


class AdaGrad(Optimizer):
    def reset_internal_params(self):
        self.h = np.zeros_like(self.w)

    def update_weight(self, update_func):
        uf = update_func(self.Ow)
        self.h += uf**2
        self.w -= self.eta / (np.sqrt(self.h) + 1e-7) * uf


class Adam(Optimizer):
    def reset_internal_params(self):
        self.beta1t = self.beta1 = 0.9
        self.beta2t = self.beta2 = 0.999
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)

    def update_weight(self, update_func):
        uf = update_func(self.Ow)
        self.m *= self.beta1
        self.m += (1 - self.beta1) * uf
        self.v *= self.beta2
        self.v += (1 - self.beta2) * uf**2
        self.w -= (
            self.eta * np.sqrt(1 - self.beta2t) / (1 - self.beta1t)
            * self.m / (np.sqrt(self.v) + 1e-7)
        )
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
