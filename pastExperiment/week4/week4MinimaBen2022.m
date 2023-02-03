countPass = [];
fhd=@cec22_test_func;
for f = 1:1:1
d = 20;
xm = ones(1000,d);
totalx = [];
xa = [];
xb = [];
f_alpha = [];

x1 = -100;
x2 = 100;

range = 100;
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

alpha = [];
for a = 0:999
    alpha = [alpha,(a/999)];
    xc = xa + (a/999)*(xb - xa);
    xm(a+1,:) = xm(a+1,:).*xc;
    funNumber = 12;
    eval(['load input_data/shift_data_' num2str(funNumber) '.txt']);
    eval(['O=shift_data_' num2str(funNumber) '(1,1:d);']);
    fsum = feval(fhd, xm(a+1,:)', funNumber);
    f_alpha = [f_alpha,fsum];

end 
h = figure();
plot(alpha,f_alpha,'linestyle','none','Marker','.');
saveas(h, strcat('fig',num2str(f),'.jpg'),'jpg');

minima = f_alpha;
count = 0;
finaloptima = [];
s = 2;
while s == 2
    temp = [];
    for c = 1:length(minima)
        if c == 1 & length(minima) == 1
            temp = minima;
        elseif c == 1 & minima(c) < minima(c+1)
            temp = [temp, minima(c)];
        elseif c ~= 1 & c < length(minima) & minima(c) < minima(c+1) & minima(c) < minima(c-1)
            temp = [temp, minima(c)];
        elseif c ~= 1 & c == length(minima) & minima(c) <minima(c-1)
            temp = [temp, minima(c)];
        end
    end
    
    minima = temp;
    
    if length(minima) == 1
        s = 1;
        finaloptima = minima;
    end
    
   count = count + 1;
end
countPass = [countPass, count];
end
tablePass = hist(countPass,unique(countPass)); 
tablePass
tabulate(countPass)