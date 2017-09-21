from subprocess import *

import random, math, sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

def plt_show_alt(plt):
    plt.savefig("/tmp/output.png")
    process = Popen(["/srv/bin/imgcat", "/tmp/output.png"])
    plt.clf()


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
