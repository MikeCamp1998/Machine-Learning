function [wi0, z0, wh0, y0] = init_network3(in, hid, out)
    
    wi0 = rand(hid, in + 1) / 10;
    z0 = zeros(hid+1,1);
    wh0 = rand(out, hid + 1) / 10;
    y0 = zeros(out,1);

end

