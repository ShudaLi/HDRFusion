//display kinect depth in real-time
//#define INFO
#define TIMER
#define _USE_MATH_DEFINES
#include <GL/glew.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#ifdef __gnu_linux__
#include <sys/types.h>
#include <sys/stat.h>
#elif _WIN32
#include <direct.h>
#else 
#error "OS not supported!"
#endif
#include <iostream>
#include <string>
#include <vector>

#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

#include <Eigen/Dense>
//camera calibration from a sequence of images
#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include "GLUtil.hpp"
#include <map>
#include "Camera.h"
#include "VideoSourceKinect.hpp"

//Qt
#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"
#include "DataLive.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace qglviewer;
DataLive::DataLive()
:Data4Viewer(){
	_bRepeat = false;// repeatedly play the sequence 
	_nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
	_nMethodIdx = 0;
}

void DataLive::loadFromYml(){
	_uResolution = 0;
	_Cw = Vector3d(1,1,1);
	_cam_param_path = string("..\\data\\Xtion");
}

void DataLive::reset(){
	//for testing different oni files
	_nStatus = 9;
	_pGL.reset();
	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight) );
	_pGL->init();
	_pGL->constructVBOsPBOs();
	_pKinect.reset();

	_pKinect.reset( new btl::kinect::VideoSourceKinect(_uResolution,_uPyrHeight,true,_Cw,_cam_param_path) );
	using namespace btl::kinect;
	_pKinect->initKinect();

	int nn = 0;
	bool bIsSuccessful = false;
	while (!bIsSuccessful&&nn < 20){
		bIsSuccessful = _pKinect->getNextFrame(&_nStatus);
		nn++;
	}

	return;
}

void DataLive::updatePF(){
	//int64 A = getTickCount();
	using btl::kinect::VideoSourceKinect;

	//load data from video source and model
	if( !_pKinect->getNextFrame(&_nStatus) ) return;
	return;
}
















