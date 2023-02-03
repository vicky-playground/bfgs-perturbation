import numpy as np
import random
import functions1
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import DataFrame
import statistics
import math
from collections import Counter
from cmaes import SepCMA
# line search on CMAES using median point and using best X as original point of next run
def main():
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    countFinal = np.array([])

    m = 30
    start = -100  # oringinal search space
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

    func_num = 8

    a = a.astype(float)
    b = b.astype(float)
    alpha = np.array([x / 999 for x in range(1000)])
    X = a + (b - a)  # get 1000 points
    X = np.array(X)
    X = X.astype(float)

    X1 = [al * ((a+b)/2) for al in alpha]  # get 1000 points
    X1 = np.array(X1)
    X1 = X1.astype(float)

    cost = np.array([function.Y(x, func_num) for x in X1])
    min_cost = np.min(cost)
    index = np.argmin(min_cost)

    if func_num == 20:
        bestIndex = 499
        bestX = bestIndex / 999 * (stop - start) + start  # find location of best solution in the search space of this run
    else:
        bestIndex = np.argwhere(cost == min_cost)  # find index of best solution
        bestIndex = int(bestIndex)
        bestX = bestIndex / 999 * (stop - start) + start  # find location of best solution in the search space of this run

    inputX = X1[bestIndex]

    for f in range(0, 30):
        print(f)
        optimizer = SepCMA(mean=inputX, sigma=20.0)
        print(" evals    f(x)")
        print("======  ==========")

        holdValue = np.array([])

        holdX = list()

        evals = 0
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                holdX.append(x)

                value = function.Y(x, func_num)
                holdValue = np.append(holdValue, value)
                evals += 1
                solutions.append((x, value))
                if evals % 30000 == 0:
                    print(f"{evals:5d}  {value:10.5f}")

            optimizer.tell(solutions)

            if optimizer.should_stop():
                break

        minValue = np.min(holdValue)
        valueIndex = np.argwhere(holdValue == minValue)
        temp = [holdX[int(valueIndex[g])] for g in range(len(valueIndex))]
        temp = np.array(temp)
        inputX = np.mean(temp,axis=0)



        countFinal = np.append(countFinal, holdValue[len(holdValue)-1])
        if f == 29:
            print("========")
            print(len(countFinal))
            print(f"mean fitness:{np.mean(countFinal):10.5f}")
            print(f"mean error:{np.mean(countFinal)-fDeltas[func_num-1]:10.5f}")
            print(f"best fitness of 30 runs:{np.min(countFinal):10.5f}")
            print(f"best error of 30 runs:{np.min(countFinal) - fDeltas[func_num - 1]:10.5f}")


if __name__ == "__main__":
    main()