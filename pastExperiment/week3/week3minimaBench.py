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
countnormal = np.array([])
countnormal1 = np.array([])

countPass = np.array([])
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
        func_num = 11
        function = functions.CEC_functions(30)
        fsum = function.Y(xm[a], func_num) - fDeltas[func_num-1]
        f_alpha = np.append(f_alpha, fsum)

    l1 = len(f_alpha)
    minima = np.array([])
    for b in range(0, l1):
        if b == 0 and f_alpha[b] < f_alpha[b + 1]:
            minima = np.append(minima, f_alpha[b])
            countfirst = np.append(countfirst, f_alpha[b])
        elif b != 0 and b < len(f_alpha) - 1 and f_alpha[b] < f_alpha[b + 1] and f_alpha[b] < f_alpha[b - 1]:
            minima = np.append(minima, f_alpha[b])
            countnormal = np.append(countnormal, f_alpha[b])
        elif b != 0 and b == len(f_alpha) - 1 and f_alpha[b] < f_alpha[b - 1]:
            minima = np.append(minima, f_alpha[b])
            countlast = np.append(countlast, f_alpha[b])

    minimaPass1 = minima
    count = 1
    finaloptima = np.array([])
    s = 2

    if len(minima) > 1:
        while s == 2:
            temp = np.array([])
            l2 = len(minimaPass1)
            for c in range(0, l2):
                if c == 0 and minimaPass1[c] < minimaPass1[c + 1]:
                    temp = np.append(temp, minimaPass1[c])
                    countfirst1 = np.append(countfirst1, minimaPass1[c])
                elif c != 0 and c < len(minimaPass1) - 1 and minimaPass1[c] < minimaPass1[c + 1] and minimaPass1[c] < minimaPass1[c - 1]:
                    temp = np.append(temp, minimaPass1[c])
                    countnormal1 = np.append(countnormal1, minimaPass1[c])
                elif c != 0 and c == len(minimaPass1) - 1 and minimaPass1[c] < minimaPass1[c - 1]:
                    temp = np.append(temp, minimaPass1[c])
                    countlast1 = np.append(countlast1, minimaPass1[c])
                elif c == 0 & len(minimaPass1) == 1:
                    temp = minimaPass1

            minimaPass1 = temp

            if len(minimaPass1) == 1:
                s = 1
                finaloptima = minimaPass1

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

