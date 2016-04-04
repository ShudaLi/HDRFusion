function [devCRF1] = calc_dev_1( intensity, irradiance )
%y: intensity 
%x: irradiance
%dy/dx: devCRF1
%1: represents devCRF is in [0,1]
%1: or intensity will converted into [0,1]

intensity = intensity/255.;
[rr,cc] = size(intensity);
devCRF1 = zeros(rr,cc);

patch = 1;
for r=patch:rr-patch
    dy1 = sum(intensity(r+1:r+patch))/patch;
    dy0 = sum(intensity(r:(r-patch+1)))/patch;
    dx1 = sum(irradiance(r+1:r+patch))/patch;
    dx0 = sum(irradiance(r:(r-patch+1)))/patch;
    devCRF1(r) = (dy1 - dy0)/ (dx1 - dx0);
end

% tmp = patch - 1;
% while( tmp >= 2 )
%     dy1 = sum(intensity(tmp+1:2*tmp))/tmp;
%     dy0 = sum(intensity(1:tmp))/tmp;
%     dx1 = sum(irradiance(tmp+1:2*tmp))/tmp;
%     dx0 = sum(irradiance(1:tmp))/tmp;
%     devCRF1(tmp) = (dy1 - dy0)/ (dx1 - dx0);
%     tmp = tmp -1;
% end

% tmp = patch - 1;
% st = rr - patch + 1;
% while( tmp >= 2 )
%     dy1 = sum(intensity(tmp+st:2*tmp+st-1))/tmp;
%     dy0 = sum(intensity(1:tmp))/tmp;
%     dx1 = sum(irradiance(tmp+1:2*tmp))/tmp;
%     dx0 = sum(irradiance(1:tmp))/tmp;
%     devCRF1(tmp) = (dy1 - dy0)/ (dx1 - dx0);
%     tmp = tmp -1;
% end


%  dist = (0.5*1/255. + 0.3*3/255 + 0.15*5/255+ 0.05*7/255);
%  for r=4:rr-4 
%      top = 0.5*intensity(r+1,1) + 0.3*intensity(r+2,1) + 0.15*intensity(r+3,1) + 0.05*intensity(r+4,1);
%      down= 0.5*intensity(r,  1) + 0.3*intensity(r-1,1) + 0.15*intensity(r-2,1) + 0.05*intensity(r-3,1);
%      devCRF1(r) = (top - down)/ dist;
%  end


end

