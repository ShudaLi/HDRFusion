function [res] = sum_prec( vec )
[r,c,n] = size(vec);
lo = max([r,c,n]);
mn = min([r,c,n]);
if(mn==0)
    res=0;
    return;
end
vec = reshape(vec,1,lo);
len = length(vec)/2;
vecHalf = zeros(1,uint16(len));

while (len>1)
    for i=1:length(vecHalf)
        if(2*i<=length(vec))
            vecHalf(i)=vec(2*i-1) + vec(2*i);
        else
            vecHalf(i)=vec(2*i-1);
        end
    end
    len = length(vecHalf)/2;
    vec = vecHalf;
    vecHalf = zeros(1,uint16(len));
end

if(length(vec)==1)
    res = vec(1);
else
    res = vec(1)+vec(2);
end

end