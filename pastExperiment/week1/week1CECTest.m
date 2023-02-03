countfinalx = [];

for j = 1:1:30
d = 30; 
x = -3*ones(1,d); 
ran = (rand(1,d)*1 - 0.5);
rx = x + ran ;
orx = rx;
n = 0; 
size = 0.5;
frx = favgShanshan(rx(1,:),n,size,d);

loop = 0;
if(n == 0)
   loop = 250000;
elseif(n == 4)
   loop = 50000;
elseif(n == 9)
   loop = 25000;
elseif(n == 24)
    loop = 10000;
elseif(n == 49)
    loop = 5000;
elseif(n == 99)
    loop = 2500;   
elseif(n == 199)
    loop = 1250;   
elseif(n == 499)
    loop = 500;  
end

count = 0;
countv = [];
range = 1.0;
for i = 1:1:loop
    ran1 = (rand(1,d)*2*range - range);
    tx = rx + ran1;
    ftx = favgShanshan(tx(1,:),n,size,d);
    
    if(ftx < frx)
        rx = tx;
        c_distance = sqrt(sum((rx).^2));
        countv = [countv; rx];
        frx = ftx;
    end
    count = count + 1;

end

rx1 = rx;
frx1 = 10*d + sum(rx1(1,:).^2 - 10*cos(2*pi*rx1(1,:)));
range1 = 0.05;
for i = 1:1:50000
    ran1 = (rand(1,d)*2*range1 - range1);
    tx1 = rx1 + ran1;
    ftx1 = benchmark_func(tx1,16);%change function number here
    
    if(ftx1 < frx1)
        rx1 = tx1;
        c_distance1 = sqrt(sum((rx1).^2));
        countv = [countv; rx1];
        frx1 = ftx1;
    end
    count = count + 1;

end

rx1;
finalx = 10*d + sum(rx1(1,:).^2 - 10*cos(2*pi*rx1(1,:)));
countfinalx(j,1) = finalx;
end

stdf = std(countfinalx(:,1)); 
sumfinalx = sum(countfinalx(:,1));
averagefinalx = sumfinalx/30;

