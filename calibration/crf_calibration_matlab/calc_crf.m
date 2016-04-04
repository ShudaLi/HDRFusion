function [intensity,resample,ln_sample_step,ln_sample_start] = calc_crf( invCRF )
%intensity: gray level 0:255
%resample: irradiance level 0:+infinity

[res, I]=sort(invCRF(:));
len = length(res);
logres = log(res);
ln_sample_step = (logres(end) - logres(1))/len;
ln_sample_start = logres(1);

ln_resample = zeros(len,1);
resample = zeros(len,1);
intensity = zeros(len,1);
for i = 1:len
    ln_resample(i) = ln_sample_start + ln_sample_step * (i-1);
    [i1,i2] = search( logres, ln_resample(i) );
    I1 = I(i1);
    I2 = I(i2);
    resp1 = invCRF(I1,1);
    resp2 = invCRF(I2,1);
    resp = exp(ln_resample(i));
    %linear interpolation
    w1 = (resp2-resp)/(resp2-resp1);
    w2 = (resp-resp1)/(resp2-resp1);
    resample(i) = w1*resp1 + w2*resp2;
    intensity(i) = w1*(I1-1) + w2*(I2-1);
end

end