import random
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
import functions
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]
countfirst = np.array([])
countlast = np.array([])
countfirst1 = np.array([])
countlast1 = np.array([])
countPass = np.array([])
countnormal = np.array([])
countnormal1 = np.array([])
for f in range(0,30):
    print(f)
    d = 30
    xm = np.ones((1000, d))

    f_alpha = np.array([])

    x1 = -5.12
    xa = np.array([])
    xa = np.append(xa, x1)

    x2 = 5.12
    xb = np.array([])
    xb = np.append(xb, x2)

    xrange = 5.12
    for i in range(1, d):
        tempxa = np.random.uniform(-5.12, 5.12)
        xa = np.append(xa, tempxa)

    for i in range(1, d):
        tempxb = np.random.uniform(-5.12, 5.12)
        xb = np.append(xb, tempxb)

    alpha = np.array([])
    for a in range(0, 1000):
        alpha = np.append(alpha, a / 999)
        xc = xa + (a / 999) * (xb - xa)
        xm[a] = xc * xm[a]
        func_num = 20
        function = functions.CEC_functions(30)
        fsum = function.Y(xm[a], func_num) - fDeltas[func_num-1]
        f_alpha = np.append(f_alpha, fsum)

    minima = f_alpha
    s = 2
    count = 0
    while s == 2:
        temp = np.array([])
        l2 = len(minima)
        for c in range(0, l2):
            if c == 0 and minima[c] < minima[c + 1]:
                temp = np.append(temp, minima[c])
                countfirst1 = np.append(countfirst1, minima[c])
            elif c != 0 and c < len(minima) - 1 and minima[c] < minima[c + 1] and minima[c] < minima[c - 1]:
                temp = np.append(temp, minima[c])
                countnormal1 = np.append(countnormal1, minima[c])
            elif c != 0 and c == len(minima) - 1 and minima[c] < minima[c - 1]:
                temp = np.append(temp, minima[c])
                countlast1 = np.append(countlast1, minima[c])
            elif c == 0 and len(minima) == 1:
                temp = minima
            elif c == 0 and len(minima) == 0:
                temp = temp

        minima = temp

        if len(minima) == 1 or len(minima) == 0:
            s = 1
            finaloptima = minima

        count = count + 1

    countPass = np.append(countPass, count)


print("f========")
print(countfirst)
print("f1========")
print(countfirst1)
print("n========")
print(countnormal)
print("n1========")
print(countnormal1)
print("l========")
print(countlast)
print("l1========")
print(countlast1)
print("========")
tablePass = Counter(countPass)
print(countPass)
print(tablePass)