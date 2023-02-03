import numpy as np
import math
from scipy.optimize import fmin_bfgs
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0,path)
from tool.benchmark2013 import functions2 as functions
import sys, scipy, numpy; print(scipy.__version__, numpy.__version__, sys.version_info)
import random

class StopOptimizingException(Exception):
    pass

def check_conv_criteria(xk):
    global result
    if np.linalg.norm(xk) < 600:
        result = xk
        raise StopOptimizingException()

if __name__ == "__main__":

    D = 30 # Dimension
    function = functions.CEC_functions(D)  # Benchmark function
    fixed_x0 = [] # record random start points     

    # 30 random start points
    for i in range(30):
        start, stop= -100,100
        x0 = np.array([None for i in range(30)])  # set vector a
        for j in range(30):
            x0[j] = random.uniform(start, stop)

        x0 = x0.astype(float) # random start point
        fixed_x0.append(x0)
    
    print(f"fixed_x0:{fixed_x0}")  

    count = 0
    # 28 CEC benchmark functions
    for f in functions.all_functions:
        count+=1       
        print(f"\n{f}")  
        """
        if f == "f3":
            for i in range(1,len(fixed_x0)+1):
                print(f"run {i}")
                fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-9, norm=float('inf'),
                epsilon=math.sqrt(math.sqrt(np.finfo(float).eps)), maxiter=None, full_output=0, disp=1,
                retall=0, callback=None)
        """
        """
        if count == (3 or 7):
            for i in range(len(fixed_x0)):
                print(f"run {i+1}. X0:{fixed_x0[i]}")
                
                try:
                   fmin_bfgs(f, fixed_x0[i], fprime=None, args=(), gtol=1e-5, norm=float('inf'),
                epsilon=math.sqrt(np.finfo(float).eps), maxiter=math.sqrt(math.sqrt(0.0000000001)), full_output=0, disp=1,
                retall=0,callback=None)
                except StopOptimizingException:
                    print(result)
        """
        
        for i in range(len(fixed_x0)):
            print(f"run {i+1}")
            fmin_bfgs(f, fixed_x0[i], fprime=None, args=(), gtol=1e-5, norm=float('inf'),
            epsilon=math.sqrt(np.finfo(float).eps), maxiter=None, full_output=0, disp=1,
            retall=0, callback=None)
        
        
            
    """
    The tolerance limit is for gtol, which is 10−5, by default. 
    """
            