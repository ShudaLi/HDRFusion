function [e] = log_func( x, intensity1, resample1 )

e=0;
for i=1:length(resample1)
    %idx = uint16(intensity1(i)+0.5);
    dist = x(1)*log(resample1(i))+x(2) - intensity1(i) ; 
    e = e + dist^2;
end

end