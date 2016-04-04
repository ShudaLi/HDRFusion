# HDRFusion
A C++ implementation of our paper "HDRFusion: HDR SLAM using a low-cost auto-exposure RGB-D sensor"

For more details, please refer the project pages at https://lishuda.wordpress.com/

-------------------------------------------------------------------------------

## License

The source code is released under the MIT license. In short, you can do 
anything with the code for any purposes. For details please view the license file 
in the source codes.

-------------------------------------------------------------------------------

## Dependences

- OpenCV 3.0 (BSD license)
- Eigen (GNU free software)
- Sophus (GNU free software) 
- OpenNI 2 (Apache license)
- Boost (BSD License)
- GLEW (Free commercial use)
- CUDA

-------------------------------------------------------------------------------

## Instructions on calibration

1. compile project "multi_expo_caputurer" and "calc_crf" by following the following instructions.
	a. refer to "calibration\Windows.txt" for compilation on windows
	b. refer to "calibration\Ubuntu.txt" for compilation on ubuntu
2. execute "multi_expo_caputurer" to capture multi-exposure images: 
	a. a folder named as the serial number of the sensor will be created. The multi-exposure images are stored inside. 
	b. check the captured images manually. By default, at each exposure time, 30 frames will be captured. They will be name as "exposure_time.no.png" (3.13.png represent exposure time 3, no 13). Due to the AE setting of the Xtion RGB-D sensor, some frames may captured at inaccurate exposure. These can be identified by comparing image brightness with other images catpured at the same exposure time. Find and remove those images.
	c. copy the images into a folder, for example "/source"
3. execute project "calc_crf" to calibrate inverse CRF.
	a. update the _path variable in the main function of "calc_crf":
		_path = string("//source//");
	b. compile and execute "calc_crf".
	c. it will produce a "load_inv_crf.m" file. 
	d. copy and paste "load_inv_crf.m" into the folder of "crf_calibration_matlab"
4. execute matlab codes to calibrate crf, derivative of crf, noise level function and normalization factors
	a. use matlab to execute the "crf_calibration_main.m" in the folder of "crf_calibration_matlab"
	b. it will produce a "crf.yml" where all parameters are stored in. 
	c. copy it into the folder of "hdr_fusion//data//serial_number//"
	
-------------------------------------------------------------------------------
	
## Instructions on HDRFusion

1. compile library "rgbd" and project "hdr_fusion_main" by following the documents:
	a. refer to "hdr_fusion\Windows.txt" for compilation on windows
	b. refer to "hdr_fusion\Ubuntu.txt" for compilation on ubuntu
2. the real sample data sequences are available from https://lishuda.wordpress.com/dataset/.
	a. bear.
	b. sofa.
	c. desk.
	d. floor1.
    e. floor2.
	f. whiteboard.
	g. serial.yml.
3. set up parameters 
	a. all parameters are loaded from a .yml file at "..//hdr_fusion_main//HDRFusionControl.yml"
	b. download the sample ".oni" data and "serial.yml" file and put them under the folder "..//data//
	c. make sure the oniFile variable in "HDRFusionControl.yml" has been specified correctly.
		oniFile:  "..\\data\\bear.oni" 
4. execute "hdr_fusion_main"
	functional keys: 
	'0': align viewing postion with camera pose 
	'7': switch on/off surfaces
	'F5': switch on/off camera trajectory
	'F6': switch on/off camera frustum
	'p': pause the HDRFusion
	'l': switch on/off colour
	'T': switch on/off tone mapping operation
5. synthetic dataset are also available. The synthetic datasets are stored ".png" files. To load into HDRFusion is trivial and therefore not provided.
	a. flickering AE dataset.
	b. smooth AE dataset.