from common import lib as l

X  = l.np.array([1.0, 0.5])
W1 = l.np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = l.np.array([0.1, 0.2, 0.3])

#print(W1.shape)
#print(X.shape)
#print(B1.shape)

A1 = l.np.dot(X, W1) + B1
#print(A1)

def sigmoid(x):
    return 1 / (1 + l.np.exp(-x))

Z1 = sigmoid(A1)
print(A1)
print(Z1)




W2 = l.np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = l.np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = l.np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print(Z2) # [ 0.62624937  0.7710107 ]


def identity_function(x):
    return x

W3 = l.np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = l.np.array([0.1, 0.2])
A3 = l.np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(Y) # [ 0.31682708  0.69627909]


def init_network():
    network = {}
    network['W1'] = l.np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = l.np.array([0.1, 0.2, 0.3])
    network['W2'] = l.np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = l.np.array([0.1, 0.2])
    network['W3'] = l.np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = l.np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = l.np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = l.np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = l.np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = l.np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [ 0.31682708 0.69627909]
