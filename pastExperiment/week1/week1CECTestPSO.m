t0 = clock;

for D=[30] 

iter_max = 10000;
trials = 51;
fhd=str2func('benchmark_func');

errors = zeros(28, trials, 11);

deltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400];

for func_num = 1:28
    for j=1:trials
        [evals] = standardPSO(fhd, func_num, D, iter_max);

        disp(sprintf(['  f%d in %d-D, fbest=%.4e, elapsed time [h]: %.2f'], ...
                   func_num, D, ...
                   evals(1,11), ...
                   etime(clock, t0)/60/60));
        disp(sprintf(' '));

        errors(func_num, j, :) = evals(1,:) - deltas(func_num);
        for i = 1:11
            if errors(func_num, j, i) <= 0.00000001
                errors(func_num, j, i) = 0;
            end
        end
    end
    save(sprintf('data_standard_PSO_%d', D), 'errors');
end

end