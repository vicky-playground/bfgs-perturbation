# Using Perturbation to Optimize BFGS Algorithm

Using the same random start points for 30 dimensions to test CEC2013 benchmark functions to see if the end point perturbated by hypercubes in each iteration (replacing the end point with the point within the search space of the hypercube: the distance between the end point and the start point of each iteration *square root of 2 / 30 for each dimension) betters the standard BFGS.  <br /> 
 <br /> 
To have a fair comparison, the experiment is under these conditions for both standard BFGS and BFGS with perturbation: <br/> 
‼️ Control the random start point to be the same for both standard BFGS and BFGS with perturbation <br/>
‼️ If the new generated end point is too close to original end point, then generate a new point until it's not too close. <br /> 
‼️ Min value of the loss function with perturbation VS Min value of the loss function w/o perturbaion <br /> 
‼️ Max. evaluations: 300,000  <br /> 
‼️ Check if all the solutions generated in each evaluation are in the right search space.

## Current Result (Jan. 2023)

 ![image](<img width="503" alt="image" src="https://user-images.githubusercontent.com/90204593/217311128-79840320-3839-40da-89fa-f6b160034500.png">)
 ![image](https://user-images.githubusercontent.com/90204593/216679189-3af32a3a-4fe5-438f-877d-27f460126449.png)
 ![image](https://user-images.githubusercontent.com/90204593/216679204-00bce778-40d9-4843-a3cd-95c8441b0967.png)
