import random
import functions1
import numpy as np
import math
from collections import Counter

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

holdbest = np.array([])
holdallmini = np.array([])
countPass = np.array([])
for f in range(0,50):
    print(f)
    start = -35
    stop = -20
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
    X = [a + al * (b - a) for al in alpha]
    X = np.array(X)
    X = X.astype(float)
    func_num = 20
    cost = np.array([function.Y(x, func_num)-fDeltas[func_num-1] for x in X])
    min_cost = np.min(cost)
    index = np.argmin(min_cost)
    min_pos = X[index]

    Y_ = cost
    minima = np.array([])
    distance = np.array([])
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

        holdmini = np.append(holdmini, w)

        Y_ = w
        if len(Y_) == 1 or len(Y_) == 0:
            s = 1
            finaloptima = Y_

        count = count + 1
    print(holdmini)
    countPass = np.append(countPass, count)
    holdallmini = np.append(holdallmini, holdmini)
    holdbest = np.append(holdbest, finaloptima)

tablePass = Counter(cost)

print("========")
print(len(holdmini))
print(len(holdallmini))
print(len(holdbest))
print(f"result (error respect all local optima to the global optimum): {np.mean(holdallmini):.2E}")
print(f"result (error respect best solution to the global optimum): {np.mean(holdbest):.2E}")
print(countPass)
print(tablePass)