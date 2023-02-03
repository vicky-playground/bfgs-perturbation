import random
import functions1
import numpy as np
import math
from collections import Counter

countDistance = np.array([])
countDistance1 = np.array([])
countDepth = np.array([])
countPass = np.array([])
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
    func_num = 11
    cost = np.array([function.Y(x, func_num) for x in X])
    min_cost = np.min(cost)
    gIndex = np.argwhere(cost == min_cost)
    min_pos = X[gIndex]

    Y_ = cost
    minima = np.array([])
    distance = np.array([])
    distance1 = np.array([])
    depth = np.array([])
    meanDistance = 0
    meanDistance1 = 0
    meanDepth = 0
    s = 2
    count = 0
    firstPass = 0
    holder = np.array([])
    holderIndex = np.array([])
    while s == 2:
        w = np.array([])
        index = np.array([])
        l2 = len(Y_)
        for i in range(0, l2):
            if i == 0 and Y_[0] < Y_[1]:
                    w = np.append(w, Y_[i])
                    index = np.append(index, i)
            elif i == l2 - 1 and Y_[i] < Y_[l2 - 2]:
                    w = np.append(w, Y_[i])
                    index = np.append(index, i)
            elif i != 0 and i < l2 - 1 and (Y_[i] < Y_[i - 1] and Y_[i + 1] > Y_[i]):
                w = np.append(w, Y_[i])
                index = np.append(index, i)
            elif i == 0 and len(Y_) == 1:
                w = np.append(w, Y_[i])
                index = np.append(index, i)
            elif i == 0 and len(Y_) == 0:
                w = w
                index = index

        if firstPass == 0:
            holder = w
            holderIndex = index
            firstPass = 1
            if(len(holderIndex)>1):
                for g in range(0, len(holderIndex)-1):
                    tindex = int(holderIndex[g])
                    tindex1 = int(holderIndex[g+1])
                    dist = np.linalg.norm(X[tindex] - X[tindex1])
                    distance = np.append(distance, dist)
                meanDistance = np.mean(distance) # Mean euclidean distance between two local optima
            else:
                meanDistance = 0


        Y_ = w
        if len(Y_) == 1 or len(Y_) == 0:
            s = 1
            finaloptima = Y_


        count = count + 1


    for k in range(0, len(holderIndex)):
        xindex = int(holderIndex[k])
        dist = np.linalg.norm(X[xindex] - X[gIndex])
        distance1 = np.append(distance1, dist)
    meanDistance1 = np.mean(distance1)  # Mean euclidean distance between local optima and global optima


    for k in range(0, len(holder)):
        dep = holder[k] - finaloptima
        depth = np.append(depth, dep)
    meanDepth = np.mean(depth)  # Mean depth between local optima and global optima

    countPass = np.append(countPass, count)
    countDistance = np.append(countDistance, meanDistance) # Count mean euclidean distance of each run
    countDistance1 = np.append(countDistance1, meanDistance1) # Count mean euclidean distance of each run (local optima and global optima)
    countDepth = np.append(countDepth, meanDepth)  # Count depth distance of each run (local optima and global optima)


meanDistanceMultiR = np.mean(countDistance) # Count mean euclidean distance of 30 runs
meanDistanceMultiR1 = np.mean(countDistance1) # Count mean euclidean distance of 30 runs (local optima and global optima)
meanDepthMultiR = np.mean(countDepth) # Count mean depth of 30 runs (local optima and global optima)
tablePass = Counter(countPass)

print("========")
print(countPass)
print(tablePass)
print("mean distance between two local optima of 30 runs: " + str(meanDistanceMultiR))
print("mean distance between local optima and global optima of 30 runs: " + str(meanDistanceMultiR1))
print("mean depth between local optima and global optima of 30 runs: " + str(meanDepthMultiR))
print(finaloptima)
