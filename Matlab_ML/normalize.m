function norm_data = normalize(data)

ux = mean(data);
S = std(data);
norm_data = (data - ux) ./ S;

%Un-Vectorized Approach 

% norm_data = [];
% for i = 1:size(data,2)
%     ux = mean(data(:,i));
%     S = std(data(:,i));
%     norm_data = [norm_data, ((data(:,i) - ux) / S)];
% end

end
