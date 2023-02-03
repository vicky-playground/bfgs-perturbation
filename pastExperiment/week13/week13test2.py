import numpy as np
import random
import functions1
from collections import Counter
import matplotlib.pyplot as plt
from cmaes import SepCMA

# line search on CMAES using median point and using best X as original point of next run
def main():
    countPass = np.array([])
    countFinal = np.array([])
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    countFinal = np.array([])

    m = 30
    start = -100  # original search space
    stop = 100
    function = functions1.CEC_functions(30)
    a = np.array([None for i in range(30)])  # set vector a
    for j in range(30):
        if j != m:
            a[j] = random.uniform(start, stop)

    a[m - 1] = start
    b = np.array([None for i in range(30)])  # set vector b
    for j in range(30):
        if j != m:
            b[j] = random.uniform(start, stop)
    b[m - 1] = stop

    func_num = 20

    a = a.astype(float)
    b = b.astype(float)
    alpha = np.array([x / 999 for x in range(1000)])
    X = [al * ((a+b)/2) for al in alpha]  # get 1000 points
    X = np.array(X)
    X = X.astype(float)

    cost = np.array([function.Y(x, func_num) for x in X])
    min_cost = np.min(cost)

    if func_num == 20:
        bestIndex = 499
        bestX = X[bestIndex]  # find location of best solution in the search space of this run
    else:
        bestIndex = np.argwhere(cost == min_cost)  # find index of best solution
        bestIndex = int(bestIndex)
        bestX = X[bestIndex]  # find location of best solution in the search space of this run

    inputX = bestX

    for f in range(0, 30):
        print(f)

        plt.plot(alpha, cost)  # plot diagram
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

        print("This run passes:" + str(count))

        countFinal = np.append(countFinal, finaloptima)

        countPass = np.append(countPass, count)

        alpha = np.array([x / 999 for x in range(1000)])
        X = [al * inputX for al in alpha]  # get 1000 points
        X = np.array(X)
        X = X.astype(float)
        cost = np.array([function.Y(x, func_num) for x in X])
        min_cost = np.min(cost)

        if func_num == 20:
            bestIndex = 499
            bestX = X[bestIndex]  # find location of best solution in the search space of this run
        else:
            bestIndex = np.argwhere(cost == min_cost)  # find index of best solution
            bestIndex = int(bestIndex)
            bestX = X[bestIndex]  # find location of best solution in the search space of this run

        inputX = bestX

    tablePass = Counter(countPass)
    print(np.mean(countFinal))
    print(countPass)
    print(tablePass)


if __name__ == "__main__":
    main()