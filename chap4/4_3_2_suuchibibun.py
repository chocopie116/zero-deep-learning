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

x_point = 10

# 傾きのグラフ
tf = tangent_line(function_1, x_point)
y2 = tf(x)

# Range
l.plt.xlim(0, 20)
l.plt.ylim(-1, 6)

# dashed
l.plt.plot([0, x_point], [function_1(x_point), function_1(x_point)], '--o')
l.plt.plot([x_point, x_point], [-1, function_1(x_point)], '--o')

# Drawing
l.plt.plot(x, y)
l.plt.plot(x, y2)
l.plt_show_alt(l.plt)
