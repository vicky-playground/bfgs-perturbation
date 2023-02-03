%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm:    Unbiased Exploration Search hybrid with CMA-ES
%                    
% Authors:      Antonio Bolufe-Rohler and Dania Tamayo-Vera
% Description:  UES is a combination of MPS and LaF 
%               especially suited for multi-modal problems.
% Last change:  October, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fval, xmin] = UES_CMAES(FUN, DIM, totalFEs, bound)

    % MPS Parameters
    alpha = 0.1;
    gamma = 2;
    d = sqrt(DIM)*2*bound;  % Search Space Diagonal
    popsize = 500;
    
    tranPoint = 0.6;
    truncPoint = 0.0;
    maxFEsTC =  (tranPoint+truncPoint)*totalFEs;
    
    LocalFE = (1-tranPoint)*totalFEs;
    sigmaDiv = 50;
    
    
     Leaders = (bound/2)*(((unifrnd(0,1,popsize,DIM)<0.5)*2)-1);
    f_Leaders = feval(FUN, Leaders');
    FEs = popsize;
    
    Population = (bound/2)*(((unifrnd(0,1,2*popsize,DIM)<0.5)*2)-1);
    f_Pop = 1e+50*ones(2*popsize,1);

    current_m = median(f_Leaders);
    while  ( FEs < tranPoint*totalFEs) 

        new_m = median(f_Pop);
        if current_m > new_m
            current_m = new_m;
            
            [sorted, indexes] = sort(f_Pop);   
            NewLeaders = Population(indexes(1:popsize),:);
            f_NewLeaders = sorted(1:popsize)';      
            merged = [Leaders; NewLeaders];
            f_merged = [f_Leaders f_NewLeaders];
            [sorted, indexes] = sort(f_merged);
            f_Leaders = sorted(1:popsize);
            Leaders = merged(indexes(1:popsize),:); 

            f_Pop = 1e+50*ones(2*popsize,1);
        end

        [sorted, indexes] = sort(f_Pop);

        % Updating threshold   
        min_step =  max(alpha*d* ((maxFEsTC-(FEs))/maxFEsTC)^gamma, 1e-05);
        max_step = 2*min_step;         

        % Population Centroid
        centroid = repmat(sum(Population(indexes(1:popsize),:))/popsize, popsize, 1);

        % Difference Vectors
        dif = normr(centroid - Leaders);

        % Difference Vector Scaling Factor
        F = unifrnd(-max_step, max_step, popsize,1);

        % Orthogonal Vectors
        orth = normr(normrnd(0,1,popsize,DIM));
        orth = normr(orth - repmat(dot(orth',dif')',1,DIM).*dif);

        % Orthogonal Step Scaling Factor
        min_perp = sqrt(max(min_step^2-abs(F).^2,0));
        max_perp = sqrt(max(max_step^2-abs(F).^2,0));         
        FO = unifrnd(min_perp, max_perp)'; 

        % New Solutions & Clamping & Evaluation
        Population(indexes(popsize+1:2*popsize),:) =  max( min(Leaders + ...           %Population(indexes(1:popsize),:) + ...  Fixed Population
                                                                repmat(F,1,DIM).*dif + ...                  	% Difference Vector Step
                                                                repmat(FO',1,DIM).*orth, bound), -bound);   	% Orthogonal Step
        f_Pop(indexes(popsize+1:2*popsize)) = feval(FUN, Population(indexes(popsize+1:2*popsize),:)');   
        FEs = FEs + popsize;
    end    


    merged = [Leaders; NewLeaders];
    f_merged = [f_Leaders f_NewLeaders];
    [sorted, indexes] = sort(f_merged);
    f_Leaders = sorted(1:popsize);
    Leaders = merged(indexes(1:popsize),:); 
                
    for lead = 1:popsize
        if LocalFE>0
            [xmin, fmin, counteval] = LocallyOptimize('LSGO', Leaders(lead,:)', LocalFE, bound, sigmaDiv);                
            if fmin < f_Leaders(lead)
                Leaders(lead,:) = xmin';
                f_Leaders(lead) = fmin;
            end
            LocalFE  = LocalFE - counteval;
        end
    end
    
    [sorted, indexes] = sort(f_Leaders);
    fval = sorted(1);
    xmin = Leaders(indexes(1),:);    
end

function [xmin, fmin, counteval] = LocallyOptimize(FUN, x, maxFEs, bound, sigmaDiv)
    cmaesopts = {};
    cmaesopts.LBounds = -bound;
    cmaesopts.UBounds = bound;
    cmaesopts.PopSize = '(4 + floor(3 * log(N)))';
    cmaesopts.ParentNumber = 'floor(popsize / 2)';
    cmaesopts.DispFinal = 'off';
    cmaesopts.DispModulo = '0';
    cmaesopts.SaveVariables = 'off';
    cmaesopts.LogModulo = '0';
    cmaesopts.LogTime = '0';
    cmaesopts.LogPlot = 'off';
    cmaesopts.stopOnStagnation = 'off';
    cmaesopts.TolX         = '1e-20*max(insigma) % stop if x-change smaller TolX';
    cmaesopts.TolFun       = '1e-20 % stop if fun-changes smaller TolFun';
    cmaesopts.MaxFunEvals = maxFEs;    

    [xmin, fmin, counteval] = cmaes(FUN, x, ((2*bound) / sigmaDiv), cmaesopts);
end

function n = normr(m)
    [~,mc]=size(m);
    if (mc == 1)
      n = m ./ abs(m);
    else
        n=sqrt(ones./(sum((m.*m)')))'*ones(1,mc).*m;
    end
end
