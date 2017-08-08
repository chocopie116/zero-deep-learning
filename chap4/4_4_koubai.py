from mpl_toolkits.mplot3d import Axes3D

import sys, os, pickle
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from common import lib as l

def numerical_gradient(f, x) :
    h = 1e-4 # 0.0001
    grad = l.np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad





def function_2(x) :
    return x[0]**2 + x[1]**2

a = numerical_gradient(function_2, l.np.array([3.0, 4.0]))
print(a)

a = numerical_gradient(function_2, l.np.array([0.0, 2.0]))
print(a)

a = numerical_gradient(function_2, l.np.array([3.0, 0.0]))
print(a)

