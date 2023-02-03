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

def main():
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    countFinal = np.array([])
    for f in range(0,30):
        print(f)
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

        a = a.astype(float)
        b = b.astype(float)
        alpha = np.array([x / 999 for x in range(1000)])
        X = a + (b - a)  # get 1000 points
        X = np.array(X)
        X = X.astype(float)


        optimizer = SepCMA(mean=X, sigma=80.0)
        print(" evals    f(x)")
        print("======  ==========")

        holdValue = np.array([])
        func_num = 26

        evals = 0
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()

                value = function.Y(x, func_num)
                holdValue = np.append(holdValue, value)
                evals += 1
                solutions.append((x, value))
                if evals % 30000 == 0:
                    print(f"{evals:5d}  {value:10.5f}")

            optimizer.tell(solutions)

            if optimizer.should_stop():
                break

        countFinal = np.append(countFinal, holdValue[len(holdValue)-1])
        if f == 29:
            print("========")
            print(len(countFinal))
            print(f"mean fitness:{np.mean(countFinal):10.5f}")
            print(f"mean error:{np.mean(countFinal)-fDeltas[func_num-1]:10.5f}")


if __name__ == "__main__":
    main()
