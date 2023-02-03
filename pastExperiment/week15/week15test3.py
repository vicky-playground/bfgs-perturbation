import numpy as np
import random
import functions1
from cmaes import SepCMA

# line search on CMAES
def main():
    countMinX = list()
    countMinY = np.array([])
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    countFinal = np.array([])
    for f in range(0,30):
        print(f)
        m = 30  # dimension
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

        func_num = 28

        a = a.astype(float)
        b = b.astype(float)
        alpha = np.array([x / 999 for x in range(1000)])
        X = [a + al * (b - a) for al in alpha]  # get 1000 points
        X = np.array(X)
        X = X.astype(float)

        cost = np.array([function.Y(x, func_num) for x in X]) # get returned value of function
        min_cost = np.min(cost)

        if func_num == 20:
            bestIndex = 499 # all returned values of function 20 are same, so choose median point
            bestX = X[bestIndex]  # find location of best solution in the search space of this run
        else:
            bestIndex = np.argwhere(cost == min_cost)  # find index of best solution
            bestIndex = int(bestIndex)
            bestX = X[bestIndex]  # find location of best solution in the search space of this run

        inputX = bestX

        bound = np.ones((m, 2))
        for l in range(0,30):
            bound[l,0] = -100
            bound[l,1] = 100

        optimizer = SepCMA(mean=inputX, sigma=20.0, bounds=bound)  # CMAES part
        print(" evals    f(x)")
        print("======  ==========")

        holdValue = np.array([])

        holdX = list()

        evals = 0
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()  # x CMAES found
                holdX.append(x)  # store all x

                value = function.Y(x, func_num) # value CMAES returned
                holdValue = np.append(holdValue, value)  # store all returned value
                evals += 1
                solutions.append((x, value))
                if evals % 30000 == 0: #choose 3000 or 30000 to show return value
                    print(f"{evals:5d}  {value:10.5f}")

            optimizer.tell(solutions)

            if optimizer.should_stop():
                break

        minValue = np.min(holdValue)
        countMinY = np.append(countMinY, minValue)
        valueIndex = np.argwhere(holdValue == minValue)
        temp = [holdX[int(valueIndex[g])] for g in range(len(valueIndex))]  # get multiple best solutions
        temp = np.array(temp)
        inputX = np.mean(temp, axis=0)  # get mean value of the best solutions
        inputXFinal = np.array([inputX])
        countMinX.append(inputXFinal)

        countFinal = np.append(countFinal, holdValue[len(holdValue)-1])
        if f == 29:
            minAllY = np.min(countMinY)
            minYIndex = np.argwhere(countMinY == minAllY)
            minX = countMinX[int(minYIndex)]
            minX = np.array(minX)
            print("========")
            print(minYIndex)
            print(minAllY)
            print(minX)
            print("========")
            print(len(countFinal))
            print(f"mean fitness:{np.mean(countFinal):10.5f}")
            print(f"mean error:{np.mean(countFinal)-fDeltas[func_num-1]:10.5f}")
            print(f"best fitness of 30 runs:{np.min(countFinal):10.5f}")
            print(f"best error of 30 runs:{np.min(countFinal) - fDeltas[func_num - 1]:10.5f}")


if __name__ == "__main__":
    # Praise The Omnissiah
    main()