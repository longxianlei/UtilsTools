#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

"""
We add the basic activation functions in CNN, then, we compare these functions.
Sigmoid, H-Sigmoid, Swish, H-Swish, ReLU.

The often used activation functions in CNN.

"""


def sig_fun(t):
    return 1/(1+np.exp(-t))


def relu6(t):
    if t > 6:
        return 6
    elif t <= 0:
        return 0
    else:
        return t


def swish(t):
    return t*sig_fun(t)


def relu(t):
    temp = []
    for i in t:
        if i >= 0:
            temp.append(i)
        else:
            temp.append(0)
    return temp


def h_sigmoid(t):
    atemp=[]
    for i in t:

        if i+3 <= 0:
            atemp.append(0)
        elif i+3 > 6:
            atemp.append(relu6(i+3)/6)
        else:
            atemp.append(relu6(i+3)/6)
    return atemp


def h_swish(t):
    return t*h_sigmoid(t)


x = np.arange(-10, 10, 0.1)
y_1 = sig_fun(x)
y_2 = h_sigmoid(x)
y_3 = relu(x)
y_4 = swish(x)
y_5 = h_swish(x)
plt.plot(x, y_1, label='sigmoid')
plt.plot(x, y_2, label='h_sigmoid')
plt.plot(x, y_3, label='relu')
plt.plot(x, y_4, label='swish')
plt.plot(x, y_5, label='h_swish')

plt.legend()
plt.show()
