import random
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
import functions1
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

countPass = np.array([])
for f in range(0,1):
    print(f)
    d = 30
    xm = np.ones((1000, d))

    f_alpha = np.array([])

    xa = np.array([])

    xb = np.array([])

    xrange = 5.12
    for i in range(1, d):
        tempxa = np.random.uniform(-5.12, 5.12)
        xa = np.append(xa, tempxa)

    for i in range(1, d):
        tempxb = np.random.uniform(-5.12, 5.12)
        xb = np.append(xb, tempxb)

    x1 = -5.12
    xa = np.append(xa, x1)

    x2 = 5.12
    xb = np.append(xb, x2)

    alpha = np.array([])
    for a in range(0, 1000):
        alpha = np.append(alpha, a / 999)
        xa = xa.astype(float)
        xb = xb.astype(float)
        xc = xa + ((a / 999) * (xb - xa))
        xc = xc.astype(float)
        xm[a] = xc * xm[a]
        func_num = 20
        function = functions1.CEC_functions(30)
        fsum = function.Y(xc, func_num)
        f_alpha = np.append(f_alpha, fsum)

    plt.plot(alpha, f_alpha)
    o = str(f)
    plt.savefig('fig11'+ o + '.png')
    plt.clf()

    l1 = len(f_alpha)
    minima = np.array([])
    for b in range(0, l1):
        if b == 0 and f_alpha[b] < f_alpha[b + 1]:
            minima = np.append(minima, f_alpha[b])
        elif b != 0 and b < len(f_alpha) - 1 and f_alpha[b] < f_alpha[b + 1] and f_alpha[b] < f_alpha[b - 1]:
            minima = np.append(minima, f_alpha[b])
        elif b != 0 and b == len(f_alpha) - 1 and f_alpha[b] < f_alpha[b - 1]:
            minima = np.append(minima, f_alpha[b])

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
                elif c != 0 and c < len(minimaPass1) - 1 and minimaPass1[c] < minimaPass1[c + 1] and minimaPass1[c] < minimaPass1[c - 1]:
                    temp = np.append(temp, minimaPass1[c])
                elif c != 0 and c == len(minimaPass1) - 1 and minimaPass1[c] < minimaPass1[c - 1]:
                    temp = np.append(temp, minimaPass1[c])
                elif c == 0 & len(minimaPass1) == 1:
                    temp = minimaPass1

            minimaPass1 = temp

            if len(minimaPass1) == 1 or len(minimaPass1) == 0:
                s = 1
                finaloptima = minimaPass1

            count = count + 1

    countPass = np.append(countPass, count)

tablePass = Counter(countPass)
print("a========")
print(xa)
print("b========")
print(xb)
print("c========")
print(xc)
print("m========")
print(xm)
print("========")
print(countPass)
print(tablePass)
