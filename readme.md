# Using Perturbation to Optimize BFGS Algorithm

Using the same random start points for 30 dimensions to test CEC2013 benchmark functions to see if the end point perturbated by hypercubes in each iteration (replacing the end point with the point of the hypercube within the search space: the distance between the end point and the start point of each iteration *square root of 2 / 30 for each dimension) betters the standard BFGS.  <br /> 
 <br /> 
To have a fair comparison, the experiment is under these conditions for both standard BFGS and BFGS with perturbation: <br/> 
‼️ Control the random start point to be the same for both standard BFGS and BFGS with perturbation <br/>
‼️ If the new generated end point is too close to original end point, then generate a new point until it's not too close. <br /> 
‼️ Min value of the loss function with perturbation VS Min value of the loss function w/o perturbaion <br /> 
‼️ Max. evaluations: 300,000  <br /> 
‼️ Check if all the solutions generated in each evaluation are in the right search space.

## Current Result (Jan. 2023)
When the perturbation bigger than 1% of bounds: 
 <img width="498" alt="image" src="https://user-images.githubusercontent.com/90204593/217411699-60c76ff9-339b-454f-b00d-ceccd69cbfc8.png">
