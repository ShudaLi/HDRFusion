function [counts,rows,cols]=hist_loc(mean)
    
%count histo and its locations
    [rr,cc] = size(mean);
    len = rr*cc/8;
    rows = zeros(256, len);
    cols = zeros(256, len);
    counts = zeros(256,1);
    for r=1:rr
        for c=1:cc
            gl = int16(mean(r,c)*255) +1;
            counts(gl) = counts(gl) + 1;
            if counts(gl)<=len
                rows( gl,counts(gl)) = r;
                cols( gl,counts(gl)) = c;
            else
                counts(gl)=len;
                break;
            end
        end
    end
    
end