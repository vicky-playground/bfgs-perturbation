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

countLS = np.array([])
countRS = np.array([])
countPass = np.array([])
for f in range(0,1):
    dim = 30
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

    dist = np.linalg.norm(a - b)  # Euclidean distance between two vectors of a and b
    a = a.astype(float)
    b = b.astype(float)
    alpha = np.array([x / 999 for x in range(1000)])
    X = [a + al * (b - a) for al in alpha]
    X = np.array(X)
    X = X.astype(float)

    n = 99
    size = 0.5
    func_num = 11
    cost = np.array([])
    for p in range(0,len(X)):
        fsum = function.Y(X[p], func_num)
        for k in range(0,n):
            rand = list()
            for i in range(0, dim):
                tempRand = np.random.uniform(-size, size)
                rand = np.append(rand, tempRand)
            newX = X[p] + list(rand)
            fsum = fsum + function.Y(newX, func_num)

        tempSum = fsum/n+1
        cost = np.append(cost, tempSum)

    min_cost = np.min(cost)
    index = np.argmin(min_cost)
    min_pos = X[index]

    plt.plot(alpha, cost)
    o = str(f)
    plt.savefig('LineSearchAverage' + o + 'Benchfun' + str(func_num) + '.png')
    plt.clf()

    Y_ = cost
    minima = np.array([])
    s = 2
    count = 0
    holdmini = np.array([])
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

        if count == 0:
            holdmini = np.append(holdmini, w)


        Y_ = w
        if len(Y_) == 1 or len(Y_) == 0:
            s = 1
            finaloptima = Y_

        count = count + 1

    countPass = np.append(countPass, count)

    bestIndex = np.argwhere(holdmini == finaloptima)
    bestIndex = int(bestIndex)
    left = np.array([])
    right = np.array([])
    left = holdmini[0:bestIndex]
    right = holdmini[bestIndex:len(holdmini)-1]

    oddL = 0
    oddR = 0
    LS = False
    RS = False

    for e in range(0,len(left)-1):
        if left[e] >= left[e+1] and oddL < 3:
            LS = True
        elif left[e] < left[e+1] and oddL < 3:
            LS = True
            oddL = oddL + 1
        elif oddL >= 3:
            LS = False

    countLS = np.append(countLS, LS)

    for e in range(0, len(right) - 1):
        if right[e] <= right[e + 1] and oddR < 3:
            RS = True
        elif right[e] > right[e + 1] and oddR < 3:
            RS = True
            oddR = oddR + 1
        elif oddR >= 3:
            RS = False

    countRS = np.append(countRS, RS)


tablePass = Counter(countPass)
tableLS = Counter(countLS)
tableRS = Counter(countRS)
'''
print("a========")
print(a)
print("b========")
print(b)
print("c========")
print(X)
print("========")
'''
print(left)
print(finaloptima)
print(right)
print("========")
print(bestIndex)
print(len(holdmini))
print(min_cost)
print(countPass)
print(tablePass)
print("L========")
print(countLS)
print(tableLS)
print("R========")
print(countRS)
print(tableRS)
print(countfsum)
print(newX)