from common import lib as l

def function_1(x):
    return 0.01 * x**2 + 0.1*x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

x = l.np.arange(0.0, 20.0, 0.1)
y = function_1(x)

l.plt.xlabel("x")
l.plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

l.plt.plot(x, y)
l.plt.plot(x, y2)
l.plt_show_alt(l.plt)
