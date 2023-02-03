# Using Perturbation to Optimize BFGS Algorithm

Using the same random start points for 30 dimensions to test CEC2013 benchmark functions to see if the end point perturbated by hypercubes in each iteration (replacing the end point with the point within the search space of the hypercube: the distance between the end point and the start point of each iteration *square root of 2 / 30 for each dimension) betters the standard BFGS. If the new generated end point is too close to original end point, then generate a new point until it's not too close.

## Current Result (Jan. 2023)

 ![image](https://user-images.githubusercontent.com/90204593/216679176-4b0176aa-3d62-44b5-a58f-ed992679df54.png)
 ![image](https://user-images.githubusercontent.com/90204593/216679189-3af32a3a-4fe5-438f-877d-27f460126449.png)
 ![image](https://user-images.githubusercontent.com/90204593/216679204-00bce778-40d9-4843-a3cd-95c8441b0967.png)


Next Step: Check if all the solutions generated in each evaluation is in the right search space. And increase the maximum evaluation times.
