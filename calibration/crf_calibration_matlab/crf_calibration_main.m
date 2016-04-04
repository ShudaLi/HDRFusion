%clear all;
close all;

%%calc image noise level function
path = 'C:\csxsl\src\eccv16\hdr\1406120314\';

imgPerExp = 29; % 30 frames per exposure
exposures = [3 6 12 24 48 96 192 384];
meanAllR = zeros(256,length(exposures));
stdAllR  = zeros(256,length(exposures));
meanAllG = zeros(256,length(exposures));
stdAllG  = zeros(256,length(exposures));
meanAllB = zeros(256,length(exposures));
stdAllB  = zeros(256,length(exposures));

for idx=1:numel(exposures)
    idx_set=[];
    expos = exposures(idx);
    expo = strcat(int2str(expos),'.' ); %'5.';
    %calc mean
    imgCounter = 0;
    sumimg = zeros( 480, 640, 3);
    for i=0:imgPerExp
        pathFileName = strcat(path , expo);
        pathFileName = strcat(pathFileName ,int2str(i) );
        pathFileName = strcat(pathFileName ,'.png' );
        if(exist(pathFileName))
            img = im2double(imread(pathFileName));
            sumimg = sumimg + img;
            imgCounter = imgCounter + 1;
            idx_set = [idx_set,i];
        end
    end
    meanImg = sumimg/imgCounter;
    
    %count histo and its locations
    [counts_red,rows_red,cols_red]=hist_loc(meanImg(:,:,1)); %R
    [counts_grn,rows_grn,cols_grn]=hist_loc(meanImg(:,:,2)); %G
    [counts_blu,rows_blu,cols_blu]=hist_loc(meanImg(:,:,3)); %B
    
    %collect all intensities
    IntenR = collect_intensity(path,idx_set,expo,imgCounter,counts_red,rows_red,cols_red);    
    IntenG = collect_intensity(path,idx_set,expo,imgCounter,counts_grn,rows_grn,cols_grn);    
    IntenB = collect_intensity(path,idx_set,expo,imgCounter,counts_blu,rows_blu,cols_blu);    

    %calc mean and std
    [meanAllR(:,idx), stdAllR(:,idx) ]= calc_mean_std( IntenR, counts_red );
    [meanAllG(:,idx), stdAllG(:,idx) ]= calc_mean_std( IntenG, counts_grn );
    [meanAllB(:,idx), stdAllB(:,idx) ]= calc_mean_std( IntenB, counts_blu );
    
    disp(expos);
    disp('exp done ');
end

save('all_data_2');
load('all_data_2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load parameters
%load inverse crf functions
[invCRF] = load_inv_crf();

[blu, resample_blu, ln_samp_step1, ln_samp_start1] = calc_crf(invCRF(:,1));%B
%%calc derivative of crf function
[devCRF01B] = calc_dev_1(blu, resample_blu);

figure
plot(resample_blu,blu,'b',resample_blu,devCRF01B*255,'b:');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[grn, resample_grn, ln_samp_step2, ln_samp_start2] = calc_crf(invCRF(:,2));%G
%%calc derivative of crf function
[devCRF01G] = calc_dev_1(grn, resample_grn);

figure
plot(resample_grn,grn,'b',resample_grn,devCRF01G*255,'b:');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[red, resample_red, ln_samp_step3, ln_samp_start3] = calc_crf(invCRF(:,3));%R
%%calc derivative of crf function
[devCRF01R] = calc_dev_1(red, resample_red);

figure
plot(resample_red,grn,'b',resample_red,devCRF01R*255,'b:');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%blue
[minStdB, I]= min(stdAllB,[],2);
options = optimset();
x0=[0.01, 0.01];
[xB,fval] = fminunc(@obj_func,x0,options,blu,resample_blu,devCRF01B,minStdB);

pstd = zeros(size(resample_blu));
for i=1:length(resample_blu)
    idx = uint16(blu(i)+.5);
    pstd(i) = devCRF01B(i) * sqrt( resample_blu(i)*xB(1)*xB(1) + xB(2)*xB(2) );
end
mB = max(pstd);

figure
plot(blu,pstd/mB,'b');
hold on
plot( meanAllB(:,1)*255, stdAllB(:,1)/mB,'r.','MarkerSize',10); hold on; %3
plot( meanAllB(:,2)*255, stdAllB(:,2)/mB,'m.','MarkerSize',10); hold on; %6
plot( meanAllB(:,3)*255, stdAllB(:,3)/mB,'y.','MarkerSize',10); hold on; %12
plot( meanAllB(:,4)*255, stdAllB(:,4)/mB,'r.','MarkerSize',10); hold on; %24
plot( meanAllB(:,5)*255, stdAllB(:,5)/mB,'g.','MarkerSize',10); hold on; %48
plot( meanAllB(:,6)*255, stdAllB(:,6)/mB,'b.','MarkerSize',10); hold on;%96
plot( meanAllB(:,7)*255, stdAllB(:,7)/mB,'w.','MarkerSize',10); hold on;%192
plot( meanAllB(:,8)*255, stdAllB(:,8)/mB,'k.','MarkerSize',10); hold on;%384
xlabel('blue');
xlim([0,256]);
ylim([0,5]);
legend('PCF','exposure time 3 ms','exposure time 6 ms','exposure time 12 ms','exposure time 24 ms','exposure time 48 ms','exposure time 96 ms','exposure time 192 ms','exposure time 384 ms','Location','northeast');
title('PCF and scaled standard deviation of noises for blue channel');

%%g
[minStdG, I]= min(stdAllG,[],2);
options = optimset();
x0=[0.01, 0.01];
[xG,fval] = fminunc(@obj_func,x0,options,grn,resample_grn,devCRF01G,minStdG);

pstd = zeros(size(resample_grn));
for i=1:length(resample_grn)
    idx = uint16(red(i)+.5);
    pstd(i) = devCRF01G(i) * sqrt( resample_grn(i)*xG(1)*xG(1) + xG(2)*xG(2) );
end

mG = max(pstd);

figure
plot(grn,pstd/mG,'g');
hold on
plot( meanAllG(:,1)*255, stdAllG(:,1)/mG,'r.','MarkerSize',10); hold on; %3
plot( meanAllG(:,2)*255, stdAllG(:,2)/mG,'m.','MarkerSize',10); hold on; %6
plot( meanAllG(:,3)*255, stdAllG(:,3)/mG,'y.','MarkerSize',10); hold on; %12
plot( meanAllG(:,4)*255, stdAllG(:,4)/mG,'r.','MarkerSize',10); hold on; %24
plot( meanAllG(:,5)*255, stdAllG(:,5)/mG,'g.','MarkerSize',10); hold on; %48
plot( meanAllG(:,6)*255, stdAllG(:,6)/mG,'b.','MarkerSize',10); hold on;%96
plot( meanAllG(:,7)*255, stdAllG(:,7)/mG,'w.','MarkerSize',10); hold on;%192
plot( meanAllG(:,8)*255, stdAllG(:,8)/mG,'k.','MarkerSize',10); hold on;%384
xlabel('green');
xlim([0,256]);
ylim([0,5]);
legend('PCF','exposure time 3 ms','exposure time 6 ms','exposure time 12 ms','exposure time 24 ms','exposure time 48 ms','exposure time 96 ms','exposure time 192 ms','exposure time 384 ms','Location','northeast');
title('PCF and scaled standard deviation of noises for green channel');

%red
[minStdR, I]= min(stdAllR,[],2);
options = optimset();
x0=[0.01, 0.01];
[xR,fval] = fminunc(@obj_func,x0,options,red,resample_red,devCRF01R,minStdR);

pstd = zeros(size(resample_red));
for i=1:length(resample_red)
    idx = uint16(red(i)+.5);
    pstd(i) = devCRF01R(i) * sqrt( resample_red(i)*xR(1)*xR(1) + xR(2)*xR(2) );
end

figure
plot(red,pstd/max(pstd),'r');
mR = max(pstd);
hold on
plot( meanAllR(:,1)*255, stdAllR(:,1)/mR,'r.','MarkerSize',10); hold on; %3
plot( meanAllR(:,2)*255, stdAllR(:,2)/mR,'m.','MarkerSize',10); hold on; %6
plot( meanAllR(:,3)*255, stdAllR(:,3)/mR,'y.','MarkerSize',10); hold on; %12
plot( meanAllR(:,4)*255, stdAllR(:,4)/mR,'r.','MarkerSize',10); hold on; %24
plot( meanAllR(:,5)*255, stdAllR(:,5)/mR,'g.','MarkerSize',10); hold on; %48
plot( meanAllR(:,6)*255, stdAllR(:,6)/mR,'b.','MarkerSize',10); hold on;%96
plot( meanAllR(:,7)*255, stdAllR(:,7)/mR,'w.','MarkerSize',10); hold on;%192
plot( meanAllR(:,8)*255, stdAllR(:,8)/mR,'k.','MarkerSize',10); hold on;%384
xlabel('red');
xlim([0,256]);
ylim([0,5]);
legend('PCF','exposure time 3 ms','exposure time 6 ms','exposure time 12 ms','exposure time 24 ms','exposure time 48 ms','exposure time 96 ms','exposure time 192 ms','Location','northeast');
title('PCF and scaled standard deviation of noises for red channel');


%store inverse crf functions into yml
matlab2opencv_yml(invCRF, 'inv_Cam_Response_Func', 'crf.yml');
matlab2opencv_yml([devCRF01';devCRF02';devCRF03'], 'dev_Cam_Response_Func', 'crf.yml', 'a');
matlab2opencv_yml([resample1'; resample2'; resample3'], 'radiance_Resample', 'crf.yml','a'); %x of real cam response functions
matlab2opencv_yml([intensity1'; intensity2'; intensity3'], 'intensity', 'crf.yml', 'a');%y of real cam resp func
matlab2opencv_yml([ln_samp_step1, ln_samp_start1, ln_samp_step2, ln_samp_start2, ln_samp_step3, ln_samp_start3],'ln_sample', 'crf.yml', 'a');
matlab2opencv_yml([xB,xG,xR], 'noise_Level_Func', 'crf.yml', 'a');
matlab2opencv_yml([mB,mG,mR], 'normalize_factor', 'crf.yml', 'a');
