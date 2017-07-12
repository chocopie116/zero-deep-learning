import numpy as np

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

print(step_function(1))

def step_function_next(x):
    y = x > 0

    return y.astype(np.int)

print(step_function_next(np.array([-1.0, 1.0, 2.0])))



