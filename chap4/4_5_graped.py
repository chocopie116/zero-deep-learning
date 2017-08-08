# coding: utf-8

from mpl_toolkits.mplot3d import Axes3D

import sys, os, pickle
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from common import lib as l

def _numerical_gradient_no_batch(f, x):
	h = 1e-4 # 0.0001
	grad = l.np.zeros_like(x)

	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = f(x) # f(x+h)

		x[idx] = tmp_val - h
		fxh2 = f(x) # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)

		x[idx] = tmp_val # 値を元に戻す

	return grad

def numerical_gradient(f, X):
	if X.ndim == 1:
		return _numerical_gradient_no_batch(f, X)
	else:
		grad = l.np.zeros_like(X)

		for idx, x in enumerate(X):
			grad[idx] = _numerical_gradient_no_batch(f, x)

		return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad


    return x, l.np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = l.np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

l.plt.plot( [-5, 5], [0,0], '--b')
l.plt.plot( [0,0], [-5, 5], '--b')
l.plt.plot(x_history[:,0], x_history[:,1], 'o')

l.plt.xlim(-3.5, 3.5)
l.plt.ylim(-4.5, 4.5)
l.plt.xlabel("X0")
l.plt.ylabel("X1")
l.plt.draw()
l.plt_show_alt(l.plt)
