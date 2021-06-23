function [w, y0] = fld(C1, C2)
    
N1 = length(C1);
N2 = length(C2);
M1 = (1/N1) * sum(C1);
M2 = (1/N2) * sum(C2);

Sb = (M2 - M1) * (M2 - M1)';
Sw1 = 0;
Sw2 = 0;

for i = 1:N1
    Sw1 = Sw1 + ((C1(i,:) - M1) * (C1(i,:) - M1)');
end

for i = 1:N2
    Sw2 = Sw2 + ((C2(i,:) - M2) * (C2(i,:) - M2)');
end

Sw = Sw1 + Sw2;

w = inv(Sw) * (M2 - M1);

proj1 = C1 * w';
proj2 = C2 * w';

alpha1 = min(proj1);
beta1 = max(proj1);

alpha2 = min(proj2);
beta2 = max(proj2);

y0 = beta1 + (alpha2-beta1).*rand(1,1);
    
end