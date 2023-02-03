d = 30;
xm = ones(1000,d);
totalx = [];
xa = [];
xb = [];
f_alpha = [];

x1 = -5.12;
xa = [xa,x1];
x2 = 5.12;
xb = [xb,x2];

range = 5.12;
for i = 1:(d-1)
tempxa = 2 * range * rand(1,1) - range;
xa = [xa, tempxa];
end

for j = 1:(d-1)
tempxb = 2 * range * rand(1,1) - range;
xb = [xb, tempxb];
end

alpha = [];
for a = 0:999
    alpha = [alpha,(a/999)];
    xc = xa + (a/999)*(xb - xa);
    xm(a+1,:) = xm(a+1,:).*xc;
    
    fsum = 10*d + sum(xm(a+1,:).^2 - 10*cos(2*pi*xm(a+1,:)));
    f_alpha = [f_alpha,fsum];

end 

countindex = [];
minima = [];

for b = 1:length(f_alpha)
    if b == 1 & f_alpha(b) < f_alpha(b+1)
        minima = [minima, f_alpha(b)];
        countindex = [countindex, b];
    elseif b ~= 1 & b < length(f_alpha) & f_alpha(b) < f_alpha(b+1) & f_alpha(b) < f_alpha(b-1)
        minima = [minima, f_alpha(b)];
        countindex = [countindex, b];
    elseif b ~= 1 & b == length(f_alpha) & f_alpha(b) < f_alpha(b-1)
        minima = [minima, f_alpha(b)];
    end
end

minimaPass1 = minima;
count = 1;
finaloptima = [];
s = 2;
while s == 2
    temp = [];
    for c = 1:length(minimaPass1)
        if c == 1 & length(minimaPass1) == 1
            temp = minimaPass1;
        elseif c == 1 & minimaPass1(c) < minimaPass1(c+1)
            temp = [temp, minimaPass1(c)];
        elseif c ~= 1 & c < length(minimaPass1) & minimaPass1(c) < minimaPass1(c+1) & minimaPass1(c) < minimaPass1(c-1)
            temp = [temp, minimaPass1(c)];
        elseif c ~= 1 & c == length(minimaPass1) & minimaPass1(c) < minimaPass1(c-1)
            temp = [temp, minimaPass1(c)];
        end
    end
    
    minimaPass1 = temp;
    
    if length(minimaPass1) == 1
        s = 1;
        finaloptima = minimaPass1;
    end
    
   count = count + 1;
end

plot(alpha, f_alpha,'linestyle','none','Marker','.');
axis([0,1,0,700]);