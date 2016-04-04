function [mean, std] = calc_mean_std( inten, count )
grayLvl = length( count );
imgTotal = size( inten, 1 );
meanGrayLvl = zeros( grayLvl, imgTotal );

%calc mean
meanLvl=zeros(grayLvl,1);
for lvl = 1:grayLvl
    for i = 1:imgTotal
        meanGrayLvl( lvl,i ) = sum_prec( inten(i,lvl,1:count(lvl)) );
    end
    meanLvl(lvl) = sum_prec( meanGrayLvl(lvl,:) )/count(lvl)/imgTotal;
end

%calc square difference 
stdAll=zeros(size(inten));
for lvl = 1:grayLvl
    for i = 1:imgTotal
        for b= 1:count(lvl);
            stdAll(i,lvl,b) = inten(i,lvl,b) - meanLvl(lvl);
            stdAll(i,lvl,b) = stdAll(i,lvl,b) * stdAll(i,lvl,b);
        end
    end
end

%calc sum of square
stdGrayLvl = zeros( grayLvl, imgTotal );
stdLvl=zeros(grayLvl,1);
for lvl = 1:grayLvl
    for i = 1:imgTotal
        stdGrayLvl( lvl,i ) = sum_prec( stdAll(i,lvl,1:count(lvl)) );
    end
    stdLvl(lvl) = sqrt(sum_prec( stdGrayLvl(lvl,:) )/count(lvl)/imgTotal);
end
mean = meanLvl;
std = stdLvl;

end