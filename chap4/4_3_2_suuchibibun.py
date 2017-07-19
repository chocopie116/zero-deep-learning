from common import lib as l

def function_1(x):
    return 0.01 * x**2 + 0.1*x

x = l.np.arange(0.0, 20.0, 0.1)
y = function_1(x)

l.plt.xlabel("x")
l.plt.ylabel("f(x)")
l.plt.plot(x, y)
l.plt_show_alt(l.plt)
