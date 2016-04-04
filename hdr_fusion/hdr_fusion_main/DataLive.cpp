//#define INFO
#define TIMER
#define _USE_MATH_DEFINES
#define  NOMINMAX 
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

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

//#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "Utility.hpp"

//camera calibration from a sequence of images
#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include <map>
#include "Camera.h"
#include "IntrinsicAnalysis.cuh"
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "TSDF.h"
#include "KinfuTracker.h"

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
	_fVolumeSize = 3.f;
	_nMode = 3;//PLAYING_BACK
	_oniFileName = std::string("x.oni"); // the openni file 
	_nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
	_bDisplayImage = false;
	_bLightOn = false;
	_bCapture = false;
	_bTrackingOnly = false;
	_bShowSurfaces = false;
	_bIsCameraPathOn = false;
	_bInitialized = false;
}

void DataLive::loadFromYml(){
	cv::FileStorage cFSRead ( "..//hdr_fusion_main//HDRFusionControl.yml", cv::FileStorage::READ );
	if (!cFSRead.isOpened()) {
		cout << "Load HDRFusionControl failed." <<endl;
		return;
	}
	cFSRead["uResolution"] >> _uResolution; 
	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["uCubicGridResolution"] >> _vXYZCubicResolution;

	cFSRead["fVolumeSize"] >> _fVolumeSize;
	cFSRead["InitialCamPos"] >> _vInitial_Cam_Pos;
	_Cw = Vector3d(_vInitial_Cam_Pos[0],_vInitial_Cam_Pos[1],_vInitial_Cam_Pos[2]);
	cFSRead["CamParamPathFileName"] >> _cam_param_path;
	cFSRead["Stage"] >> _strStage;
	cFSRead["Result_folder"] >> _result_folder; 
	//rendering
	cFSRead["bDisplayImage"] >> _bDisplayImage;
	cFSRead["bLightOn"] >> _bLightOn;
	cFSRead["nMode"] >> _nMode;//1 kinect; 2 recorder; 3 player
	cFSRead["nStatus"] >> _nStatus;

	cFSRead["bCameraPathOn"] >> _bIsCameraPathOn;
	cFSRead["oniFile"] >> _oniFileName;
	cFSRead["LoadVolume"] >> _bLoadVolume;

	cFSRead.release();
}

void DataLive::reset(){
	//for testing different oni files
	_nStatus = 9;
	//store viewing status
	if(_pGL.get()){
		_eimModelViewGL = _pGL->_ModelViewGL;
		_dZoom = _pGL->_dZoom;
		_dXAngle = _pGL->_dXAngle;
		_dYAngle = _pGL->_dYAngle;
	}
	_pGL.reset();
	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight) );

	_pGL->_bDisplayCamera = _bDisplayImage;
	_pGL->_bEnableLighting = _bLightOn;
	_pGL->_bRenderReference = _bRenderReference;
	//_pGL->clearColorDepth();
	_pGL->init();
	_pGL->constructVBOsPBOs();
	{
		//have to be after _pGL->init()
		//recover viewing status
		_pGL->_ModelViewGL = _eimModelViewGL ;
		_pGL->_dZoom = _dZoom;
		_pGL->_dXAngle = _dXAngle;
		_pGL->_dYAngle = _dYAngle;	
	}
	//reset shared pointer, notice that the relationship makes a lot of sense

	_pKinect.reset( new btl::kinect::VideoSourceKinect(_uResolution,_uPyrHeight,_Cw,_cam_param_path) );
	switch(_nMode)
	{
		using namespace btl::kinect;
	case VideoSourceKinect::SIMPLE_CAPTURING: //the simple capturing mode of the rgbd camera
		_pKinect->initKinect();
		break;
	case VideoSourceKinect::RECORDING: //record the captured sequence from the camera
		_pKinect->setDumpFileName(_oniFileName);
		_pKinect->initRecorder(_oniFileName);
		break;
	case VideoSourceKinect::PLAYING_BACK: //replay from files
		_pKinect->initPlayer(_oniFileName);
		break;
	}

	//_pDepth            .reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_Cw));
	if (true)
		_pVirtualGlobal.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(), _uResolution, _uPyrHeight, _Cw));
	else
		_pVirtualGlobal.reset(new btl::kinect::CKeyFrame(_pKinect->_pDispCamera.get(), 6, _uPyrHeight, _Cw));	
	_pVirtualCameraView.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_Cw));
	_pCameraView2.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(), _uResolution, _uPyrHeight, _Cw));

	while (!_pKinect->getNextFrame(&_nStatus)){
		;
	}
	//initialize the cubic grids
	_pGlobalMap.reset(new btl::geometry::CTsdfBlock(make_short3(_vXYZCubicResolution[0], _vXYZCubicResolution[1], _vXYZCubicResolution[2]), _fVolumeSize, 
						_pKinect->_pRGBCamera->getIntrinsics(0), SO3Group<double>(), Vector3d::Zero()));
	//initialize the tracker
	_pTracker.reset(new btl::geometry::CKinFuTracker(_pKinect->_pCurrFrame.get(), _pGlobalMap, _uResolution, _uPyrHeight));
	_pTracker->_serial_number = _pKinect->_serial_number;
	if (!_pIAVirtual){
		ostringstream directory;
		directory << "..//data//" << _pKinect->_serial_number << "//crf.yml";
		_pIAVirtual.reset(new btl::device::CIntrinsicsAnalysisMult(directory.str(), _pKinect->_pCurrFrame->_agPyrPts[0]->rows, _pKinect->_pCurrFrame->_agPyrPts[0]->cols, .25, 20));
	}

	bool bIsSuccessful = _pTracker->init(_pKinect->_pCurrFrame.get());
	if (!_strStage.compare("Tracking_n_Mapping") ){
		while (!bIsSuccessful){
			_pKinect->getNextFrame(&_nStatus);
			bIsSuccessful = _pTracker->init(_pKinect->_pCurrFrame.get());
		}
	}

	_bCapture = true;
	return;
}

void DataLive::updatePF(){
	//int64 A = getTickCount();
	using btl::kinect::VideoSourceKinect;

	if ( _bCapture ) {
		//load data from video source and model
		if( !_pKinect->getNextFrame(&_nStatus) ) return;
		//for display raw depth
		_pKinect->_pCurrFrame->_agPyrPts[2]->copyTo(_Pts);
		_pKinect->_pCurrFrame->_agPyrNls[2]->copyTo(_Nls);

		_pTracker->tracking(&*_pKinect->_pCurrFrame);
		//_bCapture = false;
	}//if( _bCapture )
	return;
}
















