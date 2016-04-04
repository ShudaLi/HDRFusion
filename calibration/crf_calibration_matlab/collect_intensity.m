function [IntenR] = collect_intensity(path,idx_set,expo,imgCounter,counts_red,rows_red,cols_red)    
%collect all intensities
    IntenR = zeros(imgCounter,256,max(counts_red));
    %statistics
    for id = 1:numel(idx_set)
        i = idx_set(id);
        pathFileName = strcat(path , expo);
        pathFileName = strcat(pathFileName ,int2str(i) );
        pathFileName = strcat(pathFileName ,'.png' );
        if(exist(pathFileName))
            img=im2double(imread(pathFileName));

            %collect all intensity
            for gl =1:256
                total = counts_red(gl);
                for b=1:total
                    IntenR(id,gl,b) = img(rows_red(gl,b),cols_red(gl,b),1);
                end
            end
            disp(i);
        end
    end
end%end of function