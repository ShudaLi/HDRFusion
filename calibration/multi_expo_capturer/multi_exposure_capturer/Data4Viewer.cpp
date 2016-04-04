#define INFO
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
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

//camera calibration from a sequence of images
#include <Eigen/Dense>
#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include "GLUtil.hpp"
#include "Camera.h"
#include "VideoSourceKinect.hpp"

//Qt
#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace qglviewer;




Data4Viewer::Data4Viewer(){
	_uResolution = 0;
	_uPyrHeight = 3;
	_Cw = Eigen::Vector3d(0.f,0.f,0.f);
	_nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
}
Data4Viewer::~Data4Viewer()
{
}

void Data4Viewer::init()
{
	GLenum eError = glewInit(); 
	if (GLEW_OK != eError){
		cout << ("glewInit() error.") << endl;
		cout << ( glewGetErrorString(eError) ) << endl;
	}
	loadFromYml();
	reset();

	return;
}//init()


void Data4Viewer::loadFromYml(){
	return;
}

void Data4Viewer::reset(){
	return;
}

void Data4Viewer::drawGlobalView()
{
	_pKinect->_pRGBCamera->setGLProjectionMatrix(0.1f, 100.f);

	glMatrixMode(GL_MODELVIEW);
	Eigen::Affine3d tmp; tmp.setIdentity();
	Eigen::Matrix4d mMat;
	mMat.row(0) = tmp.matrix().row(0);
	mMat.row(1) = -tmp.matrix().row(1);
	mMat.row(2) = -tmp.matrix().row(2);
	mMat.row(3) = tmp.matrix().row(3);

	glLoadMatrixd(mMat.data());
	GpuMat rgb(_pKinect->_cvmRGB);
	_pKinect->_pRGBCamera->renderCameraInLocal(rgb, _pGL.get(), false, NULL, 0.2f, true); //render in model coordinate
	//cout<<("drawRGBView");
	return;
}

#define INFO
#define TIMER
