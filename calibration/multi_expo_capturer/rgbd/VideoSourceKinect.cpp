#define EXPORT
#define INFO
#define _USE_MATH_DEFINES
//gl
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//opencv
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//boost
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
//eigen
#include <Eigen/Dense>
//openni
#include <OpenNI.h>
//self
#include "GLUtil.hpp"
#include "Camera.h"
#include "Kinect.h"
#include "VideoSourceKinect.hpp"
#include <iostream>
#include <string>
#include <limits>

#ifndef CHECK_RC_
#define CHECK_RC_(rc, what)	\
	BTL_ASSERT(rc == STATUS_OK, (what) ) //std::string(xnGetStatusString(rc)
#endif

using namespace openni;
using namespace std;

namespace btl{ namespace kinect
{  

VideoSourceKinect::VideoSourceKinect (ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,const Eigen::Vector3d& eivCw_, const string& cam_param_path_ )
:_uResolution(uResolution_),_uPyrHeight(uPyrHeight_),_cam_param_path_file_name(cam_param_path_)
{
	_Cw = eivCw_;
	//definition of parameters
	_frameId = 0;
	_exposure = 0;
	_gain = 100;

	std::cout << " Done. " << std::endl;
}
VideoSourceKinect::~VideoSourceKinect()
{
	_color.stop();
	_color.destroy();
	_depth.stop();
	_depth.destroy();
	_device.close();

	openni::OpenNI::shutdown();
}

void VideoSourceKinect::init()
{
	_frameId = 0;
	//allocate
	cout<<("Allocate buffers...");
	_cvmRGB			   .create( __aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3 );
	_nTotalFrameToAccumulate = 0;

	// allocate memory for later use ( registrate the depth with rgb image
	// refreshed for every frame
	// pre-allocate cvgm to increase the speed
	_pRGBCamera.reset(new btl::image::SCamera(_cam_param_path_file_name + "RGB.yml"/*btl_knt::SCamera::CAMERA_RGB*/,_uResolution));
	
	cout << "end of VideoSourceKinect::init()" <<endl;
	return;
}

void VideoSourceKinect::initKinect()
{
	init();
	cout<<("Initialize RGBD camera...");
	//inizialization 
	Status nRetVal = openni::OpenNI::initialize();
	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());
	nRetVal = _device.open(openni::ANY_DEVICE);		

	nRetVal = _depth.create(_device, openni::SENSOR_DEPTH);
	_depth.setMirroringEnabled(false);
	nRetVal = _color.create(_device, openni::SENSOR_COLOR); 
	_color.setMirroringEnabled(false);
	_colorSensorInfo = _device.getSensorInfo(openni::SENSOR_COLOR);

	if( setVideoMode(_uResolution) == STATUS_OK )
	{
		nRetVal = _depth.start(); 
		nRetVal = _color.start(); 
	}

	if (_depth.isValid() && _color.isValid())
	{
		VideoMode depthVideoMode = _depth.getVideoMode();
		VideoMode colorVideoMode = _color.getVideoMode();

		int depthWidth = depthVideoMode.getResolutionX();
		int depthHeight = depthVideoMode.getResolutionY();
		int colorWidth = colorVideoMode.getResolutionX();
		int colorHeight = colorVideoMode.getResolutionY();

		if (depthWidth != colorWidth || depthHeight != colorHeight)
		{
			printf("Warning - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
			//return ;
		}
	}

	_streams = new VideoStream*[2];
	_streams[0] = &_depth;
	_streams[1] = &_color;

	// set as the highest resolution 0 for 480x640 

	char _serial[100];
	int size = sizeof(_serial);
	_device.getProperty(openni::DEVICE_PROPERTY_SERIAL_NUMBER, &_serial, &size);
	_serial_number = string(_serial);
	cout << _serial_number << endl;

	boost::filesystem::path dir("..//" + _serial_number);
	if (boost::filesystem::create_directory(dir))
		std::cout << "Success" << "\n";
	else
		std::cout << "Fail" << "\n";

	boost::filesystem::path dir_dep("..//" + _serial_number +"//depth//");
	if (boost::filesystem::create_directory(dir_dep))
		std::cout << "Success" << "\n";
	else
		std::cout << "Fail" << "\n";

	Mat cpuClibXYxZ0, cpuMask0;

	if (true){
		_color.getCameraSettings()->setAutoExposureEnabled(false);
		_color.getCameraSettings()->setAutoWhiteBalanceEnabled(false);
		_color.getCameraSettings()->setExposure(_exposure);
		_color.getCameraSettings()->setGain(_gain);
	}
	cout<<(" Done.");

	return;
}

bool VideoSourceKinect::getNextFrame(int* pnStatus_){

	Status nRetVal = STATUS_OK;
	unsigned int time_stamp_c;
	{
		openni::VideoStream* streams[] = { &_depth, &_color };

		int changedIndex = -1;
		//_colorFrame.release();

		while (nRetVal == STATUS_OK /*|| !_colorFrame.isValid() || !_depthFrame.isValid()*/) //if any one of the frames are not loaded properly, then loop to try to load them
		{
			nRetVal = openni::OpenNI::waitForAnyStream(streams, 2, &changedIndex, 0);
			if (nRetVal == openni::STATUS_OK)
			{
				switch (changedIndex)
				{
				case 0:
					_depth.readFrame(&_depthFrame); break;
				case 1:
					_color.getCameraSettings()->setExposure(_exposure);
					_color.readFrame(&_colorFrame);
					time_stamp_c = _colorFrame.getTimestamp();
					cout << " time stamp: " << time_stamp_c - time_stamp << endl;
					time_stamp = time_stamp_c;
					break;
				default:
					printf("Error in wait\n");
				}
			}
		}
	}

	//load color image to 
	if (!_colorFrame.isValid() || !_depthFrame.isValid()) return false;

	cv::Mat cvmRGB(__aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3, (unsigned char*)_colorFrame._getFrame()->data);
	cv::Mat cvmDep(__aDepthH[_uResolution], __aDepthW[_uResolution], CV_16UC1, (unsigned short*)_depthFrame._getFrame()->data);
	cvmRGB.copyTo(_cvmRGB);
	int avg_n = 30;
	if (_nTotalFrameToAccumulate < avg_n){
		Mat RGB;
		_cvmRGB.convertTo(RGB, CV_32SC3);
		Mat cvmBGR; cv::cvtColor(_cvmRGB, cvmBGR, CV_RGB2BGR);

		ostringstream Convert;
		Convert << "..//" << _serial_number << "//" << _color.getCameraSettings()->getExposure() << "." << _nTotalFrameToAccumulate << ".png";
		if (!imwrite(Convert.str(), cvmBGR))return false;
		{
			ostringstream Convert;
			Convert << "..//" << _serial_number << "//depth//" << _color.getCameraSettings()->getExposure() << ".depth." << _nTotalFrameToAccumulate << ".png";
			if (!imwrite(Convert.str(), cvmDep))return false;
		}
		_nTotalFrameToAccumulate++;
	}
	else if (_nTotalFrameToAccumulate == avg_n)
	{
		cout << _color.getCameraSettings()->getExposure() << endl;
		_nTotalFrameToAccumulate = 0;
		if (_exposure == 0){
			_exposure = 3;
		} else {
			_exposure <<= 1;
		}
	}
	
    return true;
}

Status VideoSourceKinect::setVideoMode(ushort uResolutionLevel_){
	_uResolution = uResolutionLevel_;
	Status nRetVal = STATUS_OK;
	
#ifdef PRINT_MODE 
	//print supported sensor format
	const openni::Array<openni::VideoMode>& color_modes = _device.getSensorInfo( openni::SENSOR_COLOR )->getSupportedVideoModes();
	cout << " Color" << endl; 
	for (int i=0; i<color_modes.getSize();i++) {
		cout<< "FPS: " << color_modes[i].getFps() << " Pixel format: " << color_modes[i].getPixelFormat() << " X resolution: " << color_modes[i].getResolutionX() << " Y resolution: " << color_modes[i].getResolutionY() << endl;
	}
	const openni::Array<openni::VideoMode>& depth_modes = _device.getSensorInfo( openni::SENSOR_DEPTH )->getSupportedVideoModes();
	cout << " Depth" << endl; 
	for (int i=0; i<depth_modes.getSize();i++) {
		cout<< "FPS: " << depth_modes[i].getFps() << " Pixel format: " << depth_modes[i].getPixelFormat() << " X resolution: " << depth_modes[i].getResolutionX() << " Y resolution: " << depth_modes[i].getResolutionY() << endl;
	}
#endif

	openni::VideoMode depthMode = _depth.getVideoMode();
	openni::VideoMode colorMode = _color.getVideoMode();

	depthMode.setFps(30);
	depthMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
	colorMode.setFps(30);
	colorMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

	switch(_uResolution){
	case 3:
		depthMode.setResolution(80,60);
		colorMode.setResolution(80,60);
		depthMode.setFps(30);
		colorMode.setFps(30);
		break;
	case 2:
		depthMode.setResolution(160,120);
		colorMode.setResolution(160,120);
		depthMode.setFps(30);
		colorMode.setFps(30);
		nRetVal = _color.setVideoMode(colorMode);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

		//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
		// Set Hole Filter
		nRetVal = _device.setDepthColorSyncEnabled(TRUE);
		break;
	case 1:
		depthMode.setResolution(320,240);
		colorMode.setResolution(320,240);
		depthMode.setFps(30);
		colorMode.setFps(30);
		nRetVal = _color.setVideoMode(colorMode);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

		//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
		// Set Hole Filter
		nRetVal = _device.setDepthColorSyncEnabled(TRUE);
		break;
	case 0:
		depthMode.setResolution(640,480);
		colorMode.setResolution(640,480);
		depthMode.setFps(30);
		colorMode.setFps(30);
		nRetVal = _color.setVideoMode(colorMode);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

		//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
		// Set Hole Filter
		nRetVal = _device.setDepthColorSyncEnabled(TRUE);
		break;
	default:
		depthMode.setResolution(640,480);
		nRetVal = _color.setVideoMode(_colorSensorInfo->getSupportedVideoModes()[10]);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

		//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
		// Set Hole Filter
		nRetVal = _device.setDepthColorSyncEnabled(FALSE);
		//colorMode.setResolution(1280,1024);
		//depthMode.setFps(15);
		//colorMode.setFps(30);
		break;
	}

	nRetVal = _depth.setVideoMode(depthMode); 
	if ( nRetVal != STATUS_OK)
	{
		printf("SimpleViewer: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
		_depth.destroy();
		return nRetVal;
	}
	
	return nRetVal;
}

} //namespace kinect
} //namespace btl
