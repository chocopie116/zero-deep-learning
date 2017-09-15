import sys, os
sys.path.append(os.pardir)
from common import lib as l

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = l.np.random.randn(2, 3)

    def predict(self, x):
        return l.np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

#x = l.np.array([0.6, 0.9])
#p = net.predict(x)
#print(p)
#l.np.argmax(p)

# 正解ラベル(今は3番目が正解っぽい)
t = l.np.array([0, 0, 1])
net.loss(x, t)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
