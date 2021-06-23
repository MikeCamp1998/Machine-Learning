function [wi, wh] = mlp3(D, T, hid, iterations)
    
    in = size(D,2);
    out = size(T,2);
    step = 1000;
    count = 1;
    
    [wi0, z0, wh0, y0] = init_network3(in, hid, out);
    
    wi = wi0;
    wh = wh0;
    
    pgraph = init_progress_graph;
    
    for i = 1:iterations
        n = randi(size(D,1));
        x = D(n,:);
        t = T(n,:);
        
        [z,y] = forward_prop3(x, wi, wh);
        [wi, wh] = back_prop3(x, wi, z, wh, y, t);
        
        if (mod(i,step) == 0)
            
            count = count + 1;
            
            A = wi * [ones(1,size(D,1)); D'];
            Z = 1 ./ (1 + exp(-(A)));
        
            B = wh * [ones(1,size(Z,2)); Z];
            Y = 1 ./ (1 + exp(-(B)));
            Y = Y';
        
            mse = (1/(size(Y,1)*size(Y,2))) * sum(sum((T - Y).^2));
            pgraph = add_to_progress_graph(pgraph, count * step, mse);   
        end    
    end
end