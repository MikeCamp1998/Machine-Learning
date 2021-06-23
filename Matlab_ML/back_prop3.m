function [wi, wh] = back_prop3(x, wi, z, wh, y, t)

    eta = 0.05;
    
    %hidden-output weight update
    deltak = y' - t;
    wh_old = wh;
    wh = wh - (eta .* deltak' * [1; z]');
    
    %input-hidden weight update
    a = wi * [1, x]';
    p = exp(-(a)) ./ ((1 + exp(-a)).^2);
    q = wh_old(:, 2:end)' * deltak';
    deltaj = p .* q;
    wi = wi - (eta .* deltaj * [1, x]);
    
end
    

    