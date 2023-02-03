function evalValues = standardPSO(fhd, functionNumber, DIM, evalsPerDIM)
    evalValues = zeros(1,11);
    nextEval = 1;
    evalPoints = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    numEvals = 0;

    % PSO parameters -- D. Bratton and J. Kennedy, "Defining a standard for particle swarm optimization," IEEE SIS, 2007, pp. 120?27.
    popsize = 50; 
    w = 0.72984;
    c1 = 2.05 * w;
    c2 = 2.05 * w;

    % Search space parameters
%    range = 5; %BBOB2009
    range = 100; %CEC2013
    xmin = -range * ones(1,DIM); 
    xmax = range * ones(1,DIM); 
    vmin = -range * ones(1,DIM); 
    vmax = range * ones(1,DIM);

    % Random initial positions
    x = 2 * range * rand(popsize,DIM) - range; 
    % zero for initial velocity -- A. Engelbrecht, "Particle Swarm Optimization: Velocity Initialization," IEEE CEC, 2012, pp. 70-77.
    v = zeros(popsize,DIM);

    % initialize personal best positions as initial positions
    pbest = x;
    pbestCosts = feval(fhd, x', functionNumber);
    numEvals = numEvals + popsize;

    % update lbest
    [lbest] = update_lbest(pbestCosts, x, popsize);

    maxevals = DIM * evalsPerDIM; 
    maxgenerations = floor(maxevals/popsize);
    for generation = 2 : maxgenerations 
        % Update velocity 
        v = w*v + c1*rand(popsize,DIM).*(pbest-x) + c2*rand(popsize,DIM).*(lbest-x);

        % Clamp veloctiy 
        oneForViolation = v < repmat(vmin,popsize,1);  
        v = (1-oneForViolation).*v + oneForViolation.*repmat(vmin,popsize,1); 
        oneForViolation = v > repmat(vmax,popsize,1); 
        v = (1-oneForViolation).*v + oneForViolation.*repmat(vmax,popsize,1);

        % Update position 
        x = x + v;

        % Reflect-Z for particles out of bounds -- S. Helwig, J. Branke, and S. Mostaghim, "Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization," IEEE TEC: 17(2), 2013, pp. 259-271

        % reflect lower bound
        relectionAmount = repmat(xmin,popsize,1) - x;
        oneForNeedReflection = relectionAmount > zeros(popsize,DIM);
        relectionAmount = (1-oneForNeedReflection).*zeros(popsize,DIM) + oneForNeedReflection.*relectionAmount;
        % clampfirst
        x = (1-oneForNeedReflection).*x + oneForNeedReflection.*repmat(xmin,popsize,1); 
        % then reflect
        x = x + relectionAmount;

        % set velocity for reflected particles to zero
        v = (1-oneForNeedReflection).*v + oneForNeedReflection.*zeros(popsize,DIM);
        
        % reflect upper bound
        relectionAmount = repmat(xmax,popsize,1) - x;
        oneForNeedReflection = relectionAmount < zeros(popsize,DIM);
        relectionAmount = (1-oneForNeedReflection).*zeros(popsize,DIM) + oneForNeedReflection.*relectionAmount;
        % clampfirst
        x = (1-oneForNeedReflection).*x + oneForNeedReflection.*repmat(xmax,popsize,1); 
        % then reflect
        x = x + relectionAmount;

        % set velocity for reflected particles to zero
        v = (1-oneForNeedReflection).*v + oneForNeedReflection.*zeros(popsize,DIM);
        
        % Update pbest, lbest, and gbest
        newCosts = feval(fhd, x', functionNumber);
        numEvals = numEvals + popsize;

        for index = 1:popsize
            if newCosts(index) < pbestCosts(index)
                pbest(index,:) = x(index,:);
                pbestCosts(index) = newCosts(index);
            end
        end
        
        [lbest] = update_lbest(pbestCosts, pbest, popsize);
        gbestCost = min(pbestCosts);

        if numEvals >= evalPoints(nextEval) * maxevals
            evalValues(1, nextEval) = gbestCost;
            nextEval = nextEval + 1;
        end
    end
    evalValues(1, 11) = gbestCost;
end

% Function to update lbest
function [lbest] = update_lbest(costX, x, popsize)
    %particle 1 is neighbours with particle n=popsize
    sm(1, 1)= costX(1, popsize);
    sm(1, 2:3)= costX(1, 1:2);
    [cost, index] = min(sm);
    if index==1
        lbest(1, :) = x(popsize, :);
    else
        lbest(1, :) = x(index-1, :);
    end

    for i = 2:popsize-1
        sm(1, 1:3)= costX(1, i-1:i+1);
        [cost, index] = min(sm);
        lbest(i, :) = x(i+index-2, :);
    end

    % particle n=popsize is neighbours with particle 1
    sm(1, 1:2)= costX(1, popsize-1:popsize);
    sm(1, 3)= costX(1, 1);
    [cost, index] = min(sm);
    if index==3
        lbest(popsize, :) = x(1, :);
    else
        lbest(popsize, :) = x(popsize-2+index, :);
    end    
end