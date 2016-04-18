#define INFO
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

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <se3.hpp>
#include "Utility.hpp"

//camera calibration from a sequence of images
#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include "Utility.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "IntrinsicAnalysis.cuh"
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "TSDF.h"
#include "KinfuTracker.h"
#include "piccante.hpp"

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
	_vXYZCubicResolution.push_back(512); _vXYZCubicResolution.push_back(512); _vXYZCubicResolution.push_back(512);
	_bRenderReference = true;
	_fVolumeSize = 3.f;
	_nMode = 3;//PLAYING_BACK
	_nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
	_bDisplayImage = false;
	_bLightOn = false;
	_bCapture = false;
	_bContinuous = false;
	_bTrackingOnly = false;
	_bShowSurfaces = false;
	_bShowCamera = false;
        _bIsCurrFrameOn = false;
	_nRound = 0;
	_bInitialized = false;
	_bLoadVolume = false;
	_bSave = false;
	_strStage = string("Tracking_n_Mapping");
	_bCameraFollow = true;
	_bExportRT = false;
	_bToneMapper = false;
	_prj_c_f_w.setIdentity();

	_quadratic = gluNewQuadric();                // Create A Pointer To The Quadric Object ( NEW )
	// Can also use GLU_NONE, GLU_FLAT
	gluQuadricNormals(_quadratic, GLU_SMOOTH); // Create Smooth Normals
	gluQuadricTexture(_quadratic, GL_TRUE);   // Create Texture Coords ( NEW )
}
Data4Viewer::~Data4Viewer()
{
	//_pGL->destroyVBOsPBOs();
}

void Data4Viewer::init()
{
	if( _bInitialized ) return;
	GLenum eError = glewInit(); 
	if (GLEW_OK != eError){
		PRINTSTR("glewInit() error.");
		PRINT( glewGetErrorString(eError) );
	}
	btl::gl_util::CGLUtil::initCuda();
	btl::gl_util::CGLUtil::setCudaDeviceForGLInteroperation();//initialize before using any cuda component
	loadFromYml();
	boost::filesystem::remove_all(_result_folder.c_str());
	boost::filesystem::path dir(_result_folder.c_str());
	if(boost::filesystem::create_directories(dir)) { 
		std::cout << "Success" << "\n";
	}
	reset();

	_bInitialized = true;
	_vError.clear();
	return;
}//init()


void Data4Viewer::loadFromYml(){
	return;
}

void Data4Viewer::reset(){
	return;
}

void Data4Viewer::exportRT(const Sophus::SO3Group<double>& R_, const Eigen::Vector3d& T_){

	cv::FileStorage cFSWrite("..\\..\\RT.yml", cv::FileStorage::WRITE);
	if (!cFSWrite.isOpened()) {
		cout << "Open RT.yml failed." << endl;
		return;
	}

	Eigen::Vector3d r = R_.log();
	
	cFSWrite << "r0" << r(0);
	cFSWrite << "r1" << r(1);
	cFSWrite << "r2" << r(2);
	cFSWrite << "t0" << T_(0);
	cFSWrite << "t1" << T_(1);
	cFSWrite << "t2" << T_(2);
	cFSWrite.release();
	return;
}

void Data4Viewer::importRT(Eigen::Vector3d& r_, Eigen::Vector3d& T_){

	cv::FileStorage cFSRead("..\\..\\RT.yml", cv::FileStorage::READ);
	if (!cFSRead.isOpened()) {
		cout << "Load RT.yml failed." << endl;
		return;
	}

	cFSRead["r0"] >> r_(0);
	cFSRead["r1"] >> r_(1);
	cFSRead["r2"] >> r_(2);
	cFSRead["t0"] >> T_(0);
	cFSRead["t1"] >> T_(1);
	cFSRead["t2"] >> T_(2);

	cFSRead.release();
	return;
}
void Data4Viewer::drawGlobalView()
{
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	_pKinect->_pRGBCamera->setGLProjectionMatrix(0.1f, 100.f);

	if (_bCameraFollow){
		_pTracker->getCurrentProjectionMatrix(&_prj_c_f_w);
	}

	glMatrixMode(GL_MODELVIEW);
	glMultMatrixd(_prj_c_f_w.data());//times with manipulation matrix

	if (_bShowSurfaces)	{
		_pVirtualGlobal->assignRTfromGL();
		
		if (_bExportRT){
			exportRT(_pVirtualGlobal->_R_cw, _pVirtualGlobal->_Tw);
			_bExportRT = false;
		}
		
		if (_pVirtualGlobal->_gRadiance.empty()){
			_pVirtualGlobal->_gRadiance.create(_pVirtualGlobal->_agPyrPts[0]->size(), CV_32FC3);
			_pVirtualGlobal->_gRGB.create(_pVirtualGlobal->_agPyrPts[0]->size(), CV_8UC3);
		}
		_pGlobalMap->gpuRayCastingRadiance(_pVirtualGlobal->_pRGBCamera->getIntrinsics(0), _pVirtualGlobal->_R_cw, _pVirtualGlobal->_Tw,
			&*_pVirtualGlobal->_agPyrPts[0], &*_pVirtualGlobal->_agPyrNls[0], &_pVirtualGlobal->_gRadiance ); //if capturing is on, fineCast is off
		GpuMat radiance, BGR;
		if(_bToneMapper){
			_pVirtualGlobal->_gRadiance.convertTo(radiance, CV_32FC3, 1.f / _pTracker->_avg_exposure);
			cuda::split(radiance, _pIAVirtual->_vRadianceBGR);
			_pIAVirtual->cRF(radiance, &BGR);
		} else {
			_pVirtualGlobal->_gRadiance.copyTo(radiance);
			cuda::split(radiance, _pIAVirtual->_vRadianceBGR);
			Mat cpu_rad(radiance);
			pic::Image imgPIC(_pVirtualGlobal->_agPyrPts[0]->cols, _pVirtualGlobal->_agPyrPts[0]->rows, 3);
			memcpy(imgPIC.data, cpu_rad.data, cpu_rad.rows*cpu_rad.cols * 3 * 4);
			//imgPIC.Write("..//original.hdr");
			pic::Image *imgToneMapped = pic::WardHistogramTMO(&imgPIC, NULL, 256, 100.f, 30.f);
			memcpy(cpu_rad.data, imgToneMapped->data, cpu_rad.rows*cpu_rad.cols * 3 * 4);
			radiance.upload(cpu_rad);
			radiance.convertTo(BGR, CV_8UC3, 256, .5f);
			delete imgToneMapped;
		}

		cuda::cvtColor(BGR, _pVirtualGlobal->_gRGB, CV_BGR2RGB);
		glPointSize(3.f);
		//the 3-D points have been transformed in world already
		if (_pGL->_bEnableLighting)
		{
			glEnable(GL_LIGHTING); /* glEnable(GL_TEXTURE_2D);*/
			float diffuseColor[3] = { 0.8f, 0.8f, 0.8f };
			glColor3fv(diffuseColor);
		}
		_pVirtualGlobal->renderPtsInWorld(_pGL.get(), 0); //lvl0 but _uResolution is 1
	}
	
	if (_bIsCurrFrameOn){
		_pKinect->_pCurrFrame->renderPtsInWorld(_pGL.get(), 0);
	}

	if (_bRenderReference)	{
		//_pGL->renderPatternGL(.1f,20,20);
		//_pGL->renderPatternGL(1.f,10,10);
		glDisable(GL_LIGHTING);
		glLineWidth(1.f);
		_pGlobalMap->displayBoundingBox();
		glLineWidth(2.f);
		_pGL->renderAxisGL();
	}
	if(_bShowCamera) {
		_pKinect->_pCurrFrame->copyTo( &*_pCameraView2 );
		float aColor[3] = { 0.f, 0.f, 1.f };
		Eigen::Affine3d prj_wtc; _pTracker->getCurrentProjectionMatrix(&prj_wtc);
		_pCameraView2->setRTFromPrjCfW(prj_wtc);
		_pCameraView2->renderCameraInWorld(_pGL.get(), true, aColor, _pGL->_bDisplayCamera, .2f, _pGL->_usLevel);//refine pose
	}
	
	if( _bIsCameraPathOn ){
		_pTracker->displayCameraPath();
	}
	//PRINTSTR("drawGlobalView");
	return;
}

void Data4Viewer::drawCameraView(qglviewer::Camera* pCamera_)
{
	//_pKinect->_pRGBCamera->setGLProjectionMatrix( 0.1f,100.f);
	//
	//glMatrixMode(GL_MODELVIEW);
	//Eigen::Affine3d init; init.setIdentity(); init(1, 1) = -1.f; init(2, 2) = -1.f;// rotate the default opengl camera orientation to make it facing positive z
	//glLoadMatrixd(init.data());
	//pCamera_->setFromModelViewMatrix(init.data());

	//Eigen::Affine3d trans_cw; _pTracker->getCurrentProjectionMatrix(&trans_cw);
	//_pVirtualCameraView->setRTFromPrjCfW(trans_cw);
	//_pGlobalMap->gpuRayCasting(_pVirtualCameraView->_pRGBCamera->getIntrinsics(0), _pVirtualCameraView->_R_cw, _pVirtualCameraView->_Tw,
	//	&*_pVirtualCameraView->_agPyrPts[2], &*_pVirtualCameraView->_agPyrNls[2], NULL); //if capturing is on, fineCast is off
	//bool bLightingStatus = _pGL->_bEnableLighting;
	//_pGL->_bEnableLighting = true;
	//glPointSize(3.f);
	//glColor3f(1.f, 1.f, 1.f);
	//_pVirtualCameraView->displayPointCloudInLocal(_pGL.get(), 1);
	//_pGL->_bEnableLighting = bLightingStatus;

	//PRINTSTR("drawCameraView");
	return;	
}

void Data4Viewer::drawRGBView()
{
	_pKinect->_pRGBCamera->setGLProjectionMatrix( 0.1f,100.f);

	glMatrixMode ( GL_MODELVIEW );
	Eigen::Affine3d tmp; tmp.setIdentity();
	Matrix4d mv = btl::utility::setModelViewGLfromPrj(tmp); //mv transform X_m to X_w i.e. model coordinate to world coordinate
	glLoadMatrixd( mv.data() );
	_pKinect->_pRGBCamera->renderCameraInLocal(_pKinect->_pCurrFrame->_gRGB,  _pGL.get(),false, NULL, 0.2f, true ); //render in model coordinate
	//PRINTSTR("drawRGBView");
    return;
}

void Data4Viewer::drawDepthView(qglviewer::Camera* pCamera_)
{
	_pKinect->_pRGBCamera->setGLProjectionMatrix( 0.01f,20.f);
	//set camera parameters
	Matrix4d mModelView = btl::utility::setModelViewGLfromRTCV(SO3Group<double>(), Vector3d(0, 0, 0));
	Matrix4d mTmp = mModelView.cast<double>(); 
	pCamera_->setFromModelViewMatrix( mTmp.data() );

	//display but rendering into buffers
	glPointSize(3.f);
	glColor3f(1.f, 1.f, 1.f);
	if (_Pts.empty()) return;
	_pGL->gpuMapPtResources(_Pts);
	_pGL->gpuMapNlResources(_Nls);
	glDrawArrays(GL_POINTS, 0, btl::kinect::__aDepthWxH[2]);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);// it's crucially important for program correctness, it return the buffer to opengl rendering system.

	return;
}

void Data4Viewer::setAsPrevPos()
{
	_pTracker->getPrevView(&_pGL->_ModelViewGL);
}

#define INFO
#define TIMER
