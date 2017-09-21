from mpl_toolkits.mplot3d import Axes3D

import sys, os, pickle
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from common import lib as l

def numerical_diff(f, x) :
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2 * h)

def function_2(x) :
    return x[0]**2 + x[1]**2

# range
x = l.np.arange(-3.0, 3.0, 0.1)
y = l.np.arange(-3.0, 3.0, 0.1)

# x, yの座標をOK
X, Y = l.np.meshgrid(x, y)

# x^2 + y^2を計算する
Z = function_2(l.np.array([X, Y]))

# 描画
fig = l.plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X,Y,Z)

l.plt_show_alt(l.plt)
