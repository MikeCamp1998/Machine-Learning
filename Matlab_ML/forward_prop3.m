function [z,y] = forward_prop3(x, wi, wh)
    
    a = wi * [1, x]';
    z = 1 ./ (1 + exp(-(a)));
    %z = tanh(a);
    
    b = wh * [1; z];
    y = 1 ./ (1 + exp(-(b)));

end

