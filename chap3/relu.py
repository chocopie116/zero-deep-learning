#import numpy as np
from common import lib as l

def relu(x):
    return l.np.maximum(0, x)

# graph
x = l.np.arange(-5.0, 5.0, 1)
y = relu(x)

l.plt.plot(x, y)
l.plt.ylim(-1.0, 5.0)
l.plt_show_alt(l.plt)
