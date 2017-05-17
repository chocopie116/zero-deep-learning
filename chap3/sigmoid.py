#import numpy as np
from common import lib as l

def sigmoid(x):
    return 1 / (1 + l.np.exp(-x))

print(sigmoid(l.np.array([-1.0, 1.0, 2.0])))


# graph
x = l.np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

l.plt.plot(x, y)
l.plt.ylim(-0.1, 1.1)
l.plt_show_alt(l.plt)
