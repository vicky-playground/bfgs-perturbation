import random
import functions1
import numpy as np
import math
from collections import Counter

# Baseline of line search
countMinX = list()
countMinY = np.array([])
holdmini = np.array([])
countPass = np.array([])
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]
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
    X = [a + al * (b - a) for al in alpha]
    X = np.array(X)
    X = X.astype(float)
    func_num = 28
    cost = np.array([function.Y(x, func_num) for x in X])
    min_cost = np.min(cost)
    countMinY = np.append(countMinY, min_cost)
    index = np.argwhere(cost == min_cost)
    currentMinX = X[index]
    countMinX.append(currentMinX)


    Y_ = cost
    minima = np.array([])
    distance = np.array([])
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

        holdmini = np.append(holdmini, w)

        Y_ = w
        if len(Y_) == 1 or len(Y_) == 0:
            s = 1
            finaloptima = Y_

        count = count + 1

    countPass = np.append(countPass, count)

tablePass = Counter(countPass)

print("========")
print(len(holdmini))
print(countPass)
print("Passes:")
print(tablePass)
print("========")
minAllY = np.min(countMinY)
minYIndex = np.argwhere(countMinY == minAllY)
minX = countMinX[int(minYIndex)]
minX = np.array(minX)
print(f"Best fitness between 30 runs:{minAllY}")
print(f"Best solution between 30 runs:{minX}")
print("========")
print(f"mean fitness:{np.mean(countMinY):10.5f}")
print(f"mean error:{np.mean(countMinY)-fDeltas[func_num-1]:10.5f}")
print(f"best fitness of 30 runs:{np.min(countMinY):10.5f}")
print(f"best error of 30 runs:{np.min(countMinY) - fDeltas[func_num - 1]:10.5f}")