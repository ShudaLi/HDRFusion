
function [I1,I2] = search(logre, sample)

len = length(logre);
for i=1:len
    if sample - logre(i) >= 0 && sample - logre(i+1) <0
        I1 = i;
        I2 = i+1;
        break;
    end
end

end