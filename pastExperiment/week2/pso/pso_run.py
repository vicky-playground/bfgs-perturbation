import functions
import numpy as np
from Standard_PSO import *
import pickle

runs = 5
dim = 30
bound = 100

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]


results = np.zeros(29)
for func_num in range(1, 29):
    for run in range(0, runs):
        pso = ParticleSwarmOptimizer(func_num)
        best_fit, best = pso.optimize()
        results[func_num-1] += best_fit

    results[func_num-1] = results[func_num-1] / runs
    print(f"Function {func_num}, result (error respect to the global optimum): {(results[func_num-1]-fDeltas[func_num-1]):.2E}")

np.save(f"results_pso.np", results)