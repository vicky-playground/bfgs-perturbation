f = @LineSearch;
tranPoint = 0.6;
totalFEs = 3e+06;
LocalFE = (1-tranPoint)*totalFEs;
sigmaDiv = 50;
bound = 100;
alpha = [];
falpha= [];

d = 30;
xm = ones(1000,30);
xa = [];
xb = [];
f_alpha = [];

x1 = -100;
x2 = 100;

range = abs(100);
for i = 1:(d-1)
tempxa = 2 * range * rand(1,1) - range;
xa = [xa, tempxa];
end
xa = [xa,x1];

for j = 1:(d-1)
tempxb = 2 * range * rand(1,1) - range;
xb = [xb, tempxb];
end
xb = [xb,x2];

for a = 0:999
    alpha = [alpha,(a/999)];
    xc = xa + (a/999)*(xb - xa);
    xm(a+1,:) = xm(a+1,:).*xc;
end 

for a = 0:999
tempx = xm(a+1,:)';
if LocalFE>0
[xmin, fmin, counteval] = cmaes('LineSearch', tempx, ((2*bound) / sigmaDiv));
alpha = [alpha,xmin'];
falpha = [falpha, fmin];
LocalFE  = LocalFE - counteval;
end
end