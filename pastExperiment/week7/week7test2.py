import random
import functions1
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import DataFrame
import statistics
import math
from collections import Counter

countFinal = np.array([])
countPass = np.array([])
for f in range(0,30):
    print(f)
    start = -100
    stop = 100
    m = 30
    function = functions1.CEC_functions(30)
    a = np.array([None for i in range(30)])
    for j in range(30):
        if j != m:
            a[j] = random.uniform(start, stop)

    a[m-1] = start
    b = np.array([None for i in range(30)])
    for j in range(30):
        if j != m:
            b[j] = random.uniform(start, stop)
    b[m-1] = stop

    a = a.astype(float)
    b = b.astype(float)
    alpha = np.array([x / 999 for x in range(1000)])
    X = [ al * ((a+b)/2) for al in alpha]
    X = np.array(X)
    X = X.astype(float)
    func_num = 11
    cost = np.array([function.Y(x, func_num) for x in X])
    min_cost = np.min(cost)
    index = np.argmin(min_cost)
    min_pos = X[index]

    plt.plot(alpha, cost)
    o = str(f)
    plt.savefig('LineSearch' + o + 'Benchfun' + str(func_num) + '.png')
    plt.clf()

    Y_ = cost
    minima = np.array([])
    s = 2
    count = 0
    while s == 2:
        w = np.array([])
        l2 = len(Y_)
        for i in range(0, l2):
            if i == 0 and Y_[0] < Y_[1]:
                    w = np.append(w, Y_[i])
            elif i == l2 - 1 and Y_[i] < Y_[l2 - 2]:
                    w = np.append(w, Y_[i])
            elif i != 0 and i < l2 - 1 and (Y_[i] < Y_[i - 1] and Y_[i + 1] > Y_[i]):
                w = np.append(w, Y_[i])
            elif i == 0 and len(Y_) == 1:
                w = np.append(w, Y_[i])
            elif i == 0 and len(Y_) == 0:
                w = w


        Y_ = w
        if len(Y_) == 1 or len(Y_) == 0:
            s = 1
            finaloptima = Y_

        count = count + 1

    countFinal = np.append(countFinal, finaloptima)

    countPass = np.append(countPass, count)

tablePass = Counter(countPass)
'''
print("a========")
print(a)
print("b========")
print(b)
print("c========")
print(X)
print("========")
'''
print(np.mean(countFinal))
print(countPass)
print(tablePass)