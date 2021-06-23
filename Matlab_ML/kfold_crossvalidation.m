function [train,test] = kfold_crossvalidation(data, k, m)

v = ones([k,1]).*(floor(size(data,1)/k));
v = [v; size(data,1)-sum(v)];

full = mat2cell(data,v,size(data,2));
    
test = full{m,1};

full{m,1} = [];

train = cell2mat(full);l

end