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

plot(alpha, f_alpha,'linestyle','none','Marker','.');
axis([0,1,0,700]);
