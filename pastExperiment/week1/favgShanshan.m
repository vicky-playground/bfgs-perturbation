function f_average = favgShanshan(curX,n,size,d)
        new_x = curX + 2 * size * rand(n,d) - size;

        fsum = 10*d + sum(curX.^2 - 10*cos(2*pi*curX));
        
        for k = 1:n
            fsum = fsum + 10*d + sum(new_x(k,:).^2 - 10*cos(2*pi*new_x(k,:)));
        end

        f_average = fsum/(n+1);

    end
