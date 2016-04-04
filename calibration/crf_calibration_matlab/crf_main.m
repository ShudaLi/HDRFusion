%clear all;
close all;

%load inverse crf functions
[invCRF] = load_inv_crf();

%plot 3-channel of crf function
figure
plot(invCRF(:,1),[0:255],'b',invCRF(:,2),[0:255],'g',invCRF(:,3),[0:255],'r');

%%calculalate the linear approx of inverse crf of red channel
[intensity1, resample1, ln_samp_step1, ln_samp_start1] = calc_crf(invCRF(:,1));
figure
plot(invCRF(:,1),[0:255],'r',resample1,intensity1,'g')
figure
plot([0:255],log(invCRF(:,1)),'r',intensity1,log(resample1),'g');

%calculalate the inverse crf of BLUE channel
[intensity2,resample2,ln_samp_step2, ln_samp_start2] = calc_crf(invCRF(:,2));
figure
plot(invCRF(:,2),[0:255],'r',resample2,intensity2,'g')
figure
plot([0:255],log(invCRF(:,2)),'r',intensity2,log(resample2),'g');

%calculalate the inverse crf of GREEN channel
[intensity3,resample3, ln_samp_step3, ln_samp_start3] = calc_crf(invCRF(:,3));
figure
plot(invCRF(:,3),[0:255],'r',resample3,intensity3,'g')
figure
plot([0:255],log(invCRF(:,3)),'r',intensity3,log(resample3),'g');

%%calc derivative of crf function
[devCRF01] = calc_dev_1(intensity1, resample1);

figure
plot(resample1,intensity1/255,'b',resample1,devCRF01);

[devCRF02] = calc_dev_1(intensity2, resample2);

figure
plot(resample2,intensity2/255,'g',resample2,devCRF02);

[devCRF03] = calc_dev_1(intensity3, resample3);

figure
plot(resample3,intensity3/255,'r',resample1,devCRF03);


%store inverse crf functions into yml
matlab2opencv_yml(invCRF, 'inv_Cam_Response_Func', 'crf.yml');
matlab2opencv_yml([devCRF01';devCRF02';devCRF03'], 'dev_Cam_Response_Func', 'crf.yml', 'a');
matlab2opencv_yml([resample1'; resample2'; resample3'], 'radiance_Resample', 'crf.yml','a'); %x of real cam response functions
matlab2opencv_yml([intensity1'; intensity2'; intensity3'], 'intensity', 'crf.yml', 'a');%y of real cam resp func
matlab2opencv_yml([ln_samp_step1, ln_samp_start1, ln_samp_step2, ln_samp_start2, ln_samp_step3, ln_samp_start3],'ln_sample', 'crf.yml', 'a');
matlab2opencv_yml([xB,xG,xR], 'noise_Level_Func', 'crf.yml', 'a');
matlab2opencv_yml([xB,xG,xR], 'normalize_factor', 'crf.yml', 'a');


