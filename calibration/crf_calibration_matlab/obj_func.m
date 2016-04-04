function [e] = obj_func( x, intensity1, resample1, devCRF01, minStd  )

e=0;
for i=1:length(resample1)
    idx = uint16(intensity1(i)+0.5);
    if( idx < 1) 
        idx = 1;
    end
    dist = minStd(idx) - devCRF01(i) * sqrt( resample1(i)*x(1)*x(1) + x(2)*x(2) );
    if(dist >=0)
        e = e + dist;
    else
        e = e + 5*abs(dist);
    end
end

% ss = x(1);
% sc = x(2);
% e=0;
%for gl=1:rr
%    e = e + abs(devInvCRF(gl,1)*sqrt(invCRF(gl,1)*ss*ss+sc*sc) - std(gl));
%end

end