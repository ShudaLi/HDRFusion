#define EXPORT
#define INFO
#define _USE_MATH_DEFINES
#define  NOMINMAX 
//gl
#include <GL/glew.h>
//#include <GL/freeglut.h>
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
//eigen
#include <Eigen/Core>
#include <se3.hpp>
//openni
#include <OpenNI.h>
//self
#include "Utility.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "Kinect.h"
#include "IntrinsicAnalysis.cuh"
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "CudaLib.cuh"
#include "Normal.cuh"
#include <iostream>
#include <string>
#include <limits>

#ifndef CHECK_RC_
#define CHECK_RC_(rc, what)	\
	BTL_ASSERT(rc == STATUS_OK, (what) ) //std::string(xnGetStatusString(rc)
#endif

using namespace btl::utility;
using namespace btl::device;
using namespace openni;
using namespace std;
using namespace Sophus;

namespace btl{ namespace kinect
{  

bool VideoSourceKinect::_bIsSequenceEnds = false;
VideoSourceKinect::VideoSourceKinect (ushort uResolution_, ushort uPyrHeight_, const Eigen::Vector3d& eivCw_, const string& cam_param_path_ )
:_uResolution(uResolution_),_uPyrHeight(uPyrHeight_),_cam_param_path_file_name(cam_param_path_)
{
	_Cw = eivCw_;
	//other
	//definition of parameters
	_fThresholdDepthInMeter = 0.2f;
	_fSigmaSpace = 4;
	_fSigmaDisparity = 1.f/4.f - 1.f/(4.f+_fThresholdDepthInMeter);

	_bIsSequenceEnds = false;

	_fCutOffDistance = 2.5f;
	_frameId = 0;

	_fMtr2Depth = 1000.f;
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

void VideoSourceKinect::initKinect()
{
	_frameId = 0;
	_nMode = SIMPLE_CAPTURING;
	PRINTSTR("1. Initialize RGBD camera...");
	//inizialization 
	Status nRetVal = openni::OpenNI::initialize();
	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());
	nRetVal = _device.open(openni::ANY_DEVICE);			CHECK_RC_(nRetVal, "Initialize _cContext"); 
	_pPlaybackControl = _device.getPlaybackControl();

	nRetVal = _depth.create(_device, openni::SENSOR_DEPTH); CHECK_RC_(nRetVal, "Initialize _cContext"); _depth.setMirroringEnabled(false);
	nRetVal = _color.create(_device, openni::SENSOR_COLOR); CHECK_RC_(nRetVal, "Initialize _cContext"); _color.setMirroringEnabled(false);
	_colorSensorInfo = _device.getSensorInfo(openni::SENSOR_COLOR);

	if( setVideoMode(_uResolution) == STATUS_OK )
	{
		nRetVal = _depth.start(); CHECK_RC_(nRetVal, "Create depth video stream fail");
		nRetVal = _color.start(); CHECK_RC_(nRetVal, "Create color video stream fail"); 
	}

	if (_depth.isValid() && _color.isValid())
	{
		VideoMode depthVideoMode = _depth.getVideoMode();
		VideoMode colorVideoMode = _color.getVideoMode();

		depthWidth = depthVideoMode.getResolutionX();
		depthHeight = depthVideoMode.getResolutionY();
		colorWidth = colorVideoMode.getResolutionX();
		colorHeight = colorVideoMode.getResolutionY();

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

	PRINTSTR("2. Allocate buffers...");
	_cvmRGB.create(__aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3);
	_cvmDepthFloat.create(__aDepthH[_uResolution], __aDepthW[_uResolution], CV_32FC1);

	_pRGBCamera.reset(new btl::image::SCamera(_cam_param_path_file_name + "RGB.yml"/*btl_knt::SCamera::CAMERA_RGB*/, _uResolution));
	_pDispCamera.reset(new btl::image::SCamera(_cam_param_path_file_name + "RGB.yml"/*btl_knt::SCamera::CAMERA_RGB*/, 6));

	_pCurrFrame.reset(new CKeyFrame(_pRGBCamera.get(), _uResolution, _uPyrHeight, _Cw));


	PRINTSTR(" Done.");

	return;
}
void VideoSourceKinect::initRecorder(std::string& strPath_){
	initKinect();
	_nMode = RECORDING;
	PRINTSTR("Initialize RGBD data recorder...");
	_recorder.create(strPath_.c_str());
	_recorder.attach( _depth );
	_recorder.attach( _color );
	_recorder.start();

	//store serial #number
	boost::filesystem::path p(strPath_.c_str());
	boost::filesystem::path dir = p.parent_path();
	
	string fileName = dir.string() + string("\\serial.yml");
	cv::FileStorage cFSWrite(fileName.c_str(), cv::FileStorage::WRITE);
	cFSWrite << "serial" << _serial_number;
	cFSWrite.release();

	PRINTSTR(" Done.");
}
void VideoSourceKinect::initPlayer(std::string& strPathFileName_){
	_frameId = 0;
	//_frameId = 82;
	_nMode = PLAYING_BACK;
	PRINTSTR("1. Initialize OpenNI Player...");
	//inizialization 
	Status nRetVal = openni::OpenNI::initialize();
	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());
	nRetVal = _device.open(strPathFileName_.c_str());		CHECK_RC_(nRetVal, "Open oni file");
	nRetVal = _depth.create(_device, openni::SENSOR_DEPTH); CHECK_RC_(nRetVal, "Initialize _cContext"); 
	nRetVal = _color.create(_device, openni::SENSOR_COLOR); CHECK_RC_(nRetVal, "Initialize _cContext"); 
	_pPlaybackControl = _device.getPlaybackControl();

	nRetVal = _depth.start(); CHECK_RC_(nRetVal, "Create depth video stream fail");
	nRetVal = _color.start(); CHECK_RC_(nRetVal, "Create color video stream fail"); 

	if (_depth.isValid() && _color.isValid())
	{
		VideoMode depthVideoMode = _depth.getVideoMode();
		VideoMode colorVideoMode = _color.getVideoMode();

		depthWidth = depthVideoMode.getResolutionX();
		depthHeight = depthVideoMode.getResolutionY();
		colorWidth = colorVideoMode.getResolutionX();
		colorHeight = colorVideoMode.getResolutionY();

		if (depthWidth != colorWidth || depthHeight != colorHeight)
		{
			printf("Warning - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
			PRINTSTR(" Failed.");
			//return ;
		}
	}

	_streams = new VideoStream*[2];
	_streams[0] = &_depth;
	_streams[1] = &_color;

	// set as the highest resolution 0 for 480x640 
	//register the depth generator with the image generator
	nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	_device.setDepthColorSyncEnabled(FALSE);

	const size_t last_slash_idx = strPathFileName_.rfind('//');
	if (std::string::npos != last_slash_idx)
	{
		string directory = strPathFileName_.substr(0, last_slash_idx);
		directory += string("//serial.yml");
		cout << directory << endl;
		cv::FileStorage storage(directory.c_str(), cv::FileStorage::READ);
		storage["serial"] >> _serial_number;
		storage.release();
	}

	_fCutOffDistance = 2.5;
	PRINTSTR("2. Allocate buffers...");
	_cvmRGB.create(__aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3);
	_cvmDepthFloat.create(__aDepthH[_uResolution], __aDepthW[_uResolution], CV_32FC1);

	_pRGBCamera.reset(new btl::image::SCamera(_cam_param_path_file_name + "RGB.yml"/*btl_knt::SCamera::CAMERA_RGB*/, _uResolution));
	_pDispCamera.reset(new btl::image::SCamera(_cam_param_path_file_name + "RGB.yml"/*btl_knt::SCamera::CAMERA_RGB*/, 6));

	_pCurrFrame.reset(new CKeyFrame(_pRGBCamera.get(), _uResolution, _uPyrHeight, _Cw));

	PRINTSTR(" Done.");

	return;
}//initPlayer()

bool VideoSourceKinect::getNextFrame(int* pnStatus_){

	if (_bIsSequenceEnds) { *pnStatus_ = PAUSE; _bIsSequenceEnds = false; }
	Status nRetVal = STATUS_OK;
	unsigned int time_stamp_c;
	if (_nMode == 3){ //player mode

		if (_pPlaybackControl->seek(_depth, _frameId) == openni::STATUS_OK)
		{
			// Read next frame from all streams.
			_depth.readFrame(&_depthFrame);

			// the new frameId might be different than expected (due to clipping to edges)
			_frameId = _depthFrame.getFrameIndex();
		}
		PRINT(_frameId);
		if (_pPlaybackControl->seek(_color, _frameId) == openni::STATUS_OK)
		{
			// Read next frame from all streams.
			_color.readFrame(&_colorFrame);
			// the new frameId might be different than expected (due to clipping to edges)
			_frameId = _depthFrame.getFrameIndex();
		}
		_frameId+=3;
	}
	else{
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
					_color.readFrame(&_colorFrame); 
					time_stamp_c = _colorFrame.getTimestamp();
					if (false){
						cout << _color.getCameraSettings()->getExposure() << endl;
						cout << _color.getCameraSettings()->getGain() << endl;
						cout << " time stamp: " << time_stamp_c - time_stamp << endl;
					}
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
	if (_uResolution == 1 && _colorFrame.getHeight() == 480){
		cv::Mat cvmRGB(__aRGBH[0], __aRGBW[0], CV_8UC3, (unsigned char*)_colorFrame._getFrame()->data);
		cv::Mat cvmDep(__aDepthH[0], __aDepthW[0], CV_16UC1, (unsigned short*)_depthFrame._getFrame()->data);
		cv::pyrDown(cvmRGB, _cvmRGB);
		cv::pyrDown(cvmDep, _cvmDep);
	}
	else{
		cv::Mat cvmRGB(__aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3, (unsigned char*)_colorFrame._getFrame()->data);
		cv::Mat cvmDep(__aDepthH[_uResolution], __aDepthW[_uResolution], CV_16UC1, (unsigned short*)_depthFrame._getFrame()->data);
		cvmRGB.copyTo(_cvmRGB);
		cvmDep.copyTo(_cvmDep);
	}

	_cvmDep.convertTo(_cvmDepthFloat, CV_32FC1, 0.001f);

	gpuBuildPyramidUseNICVm();

	_pCurrFrame->initRT();
		
    return true;
}

void VideoSourceKinect::gpuBuildPyramidUseNICVm( ){
	_gRGB.upload(_cvmRGB);
	_gDepth.upload(_cvmDepthFloat);

	_pCurrFrame->_gRGB.setTo(0);//clear(RGB)
	_gRGB.copyTo(_pCurrFrame->_gRGB);

	//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
	_tmp.create(_gRGB.size(), CV_32FC1);
	_tmp2.create(_gRGB.size(), CV_32FC1);
	btl::device::cvt_depth2disparity2(_gDepth, _fCutOffDistance, &_tmp);//convert depth from mm to m
	btl::device::bilateral_filtering(_tmp, _fSigmaSpace, _fSigmaDisparity, &_tmp2);
	btl::device::cvt_disparity2depth(_tmp2, &*_pCurrFrame->_agPyrDepths[0]);
	//get pts and nls
	btl::device::unproject_rgb(*_pCurrFrame->_agPyrDepths[0], _pRGBCamera->_fFx, _pRGBCamera->_fFy, _pRGBCamera->_u, _pRGBCamera->_v, 0, &*_pCurrFrame->_agPyrPts[0]);
	btl::device::compute_normals_eigen(*_pCurrFrame->_agPyrPts[0], &*_pCurrFrame->_agPyrNls[0], NULL);
	//generate black and white

	//down-sampling
	for (unsigned int i = 1; i < _uPyrHeight; i++)	{
		_tmp.setTo(0);
		cv::cuda::resize(_tmp2, _tmp, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		btl::device::bilateral_filtering(_tmp, _fSigmaSpace, _fSigmaDisparity, &_tmp2);
		btl::device::cvt_disparity2depth(_tmp2, &*_pCurrFrame->_agPyrDepths[i]);
		btl::device::unproject_rgb(*_pCurrFrame->_agPyrDepths[i], _pRGBCamera->_fFx, _pRGBCamera->_fFy, _pRGBCamera->_u, _pRGBCamera->_v, i, &*_pCurrFrame->_agPyrPts[i]);
		btl::device::compute_normals_eigen(*_pCurrFrame->_agPyrPts[i], &*_pCurrFrame->_agPyrNls[i], NULL);
	}

	for (unsigned int i = 0; i < _uPyrHeight; i++)	{
		_pCurrFrame->_agPyrPts[i]->download(*_pCurrFrame->_acvmShrPtrPyrPts[i]);
		_pCurrFrame->_agPyrNls[i]->download(*_pCurrFrame->_acvmShrPtrPyrNls[i]);
		_pCurrFrame->_agPyrReliability[i]->download(*_pCurrFrame->_acvmShrPtrPyrReliability[i]);
		//scale the depth map
		btl::device::scale_depth(i, _pRGBCamera->_fFx, _pRGBCamera->_fFy, _pRGBCamera->_u, _pRGBCamera->_v, &*_pCurrFrame->_agPyrDepths[i]);
		_pCurrFrame->_agPyrDepths[i]->download(*_pCurrFrame->_acvmShrPtrPyrDepths[i]);
	}
	return;
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
		nRetVal = _device.setDepthColorSyncEnabled(FALSE);
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
