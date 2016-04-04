#define EXPORT
#define INFO
#define _USE_MATH_DEFINES
#define  NOMINMAX 
//gl
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//boost
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/special_functions/fpclassify.hpp> //isnan
#include <boost/lexical_cast.hpp>
//stl
#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <math.h>
//openncv
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "OtherUtil.hpp"
#include "Converters.hpp"
#include <se3.hpp>
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "Kinect.h"
#include "IntrinsicAnalysis.cuh"
#include "KeyFrame.h"
#include "CVUtil.hpp"
#include "Utility.hpp"
#include "CudaLib.cuh"

using namespace Eigen;
using namespace Sophus;

btl::kinect::CKeyFrame::CKeyFrame( btl::image::SCamera::tp_ptr pRGBCamera_, ushort uResolution_, ushort uPyrLevel_, const Vector3d& eivCw_/*float fCwX_, float fCwY_, float fCwZ_*/ )
:_pRGBCamera(pRGBCamera_),_uResolution(uResolution_),_uPyrHeight(uPyrLevel_),_initCw(eivCw_){
	allocate();
	initRT();
}
btl::kinect::CKeyFrame::CKeyFrame(const CKeyFrame::tp_ptr pFrame_ )
{
	_pRGBCamera = pFrame_->_pRGBCamera;
	_uResolution = pFrame_->_uResolution;
	_uPyrHeight = pFrame_->_uPyrHeight;
	_initCw = pFrame_->_initCw;
	allocate();
	pFrame_->copyTo(this);
}

btl::kinect::CKeyFrame::CKeyFrame(const CKeyFrame& Frame_)
{
	_pRGBCamera = Frame_._pRGBCamera;
	_uResolution = Frame_._uResolution;
	_uPyrHeight = Frame_._uPyrHeight;
	_initCw = Frame_._initCw;
	allocate();
	Frame_.copyTo(this);
}

btl::kinect::CKeyFrame::~CKeyFrame()
{

}
void btl::kinect::CKeyFrame::allocate(){
	namespace btl_knt = btl::kinect;
	_gRGB.create(__aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3);
	//disparity
	for(int i=0; i<_uPyrHeight; i++){
		int nRowsRGB = __aRGBH[_uResolution] >> i;
		int nColsRGB = __aRGBW[_uResolution] >> i;//__aKinectW[_uResolution]>>i;

		int nRowsDepth = __aDepthH[_uResolution]>>i;
		int nColsDepth = __aDepthW[_uResolution]>>i;

		//host
		_acvmShrPtrPyrPts[i] .reset();
		_acvmShrPtrPyrPts[i] .reset(new cv::Mat(nRowsDepth,nColsDepth,CV_32FC3));
		_acvmShrPtrPyrNls[i] .reset();
		_acvmShrPtrPyrNls[i] .reset(new cv::Mat(nRowsDepth,nColsDepth,CV_32FC3));
		_acvmShrPtrPyrReliability[i].reset();
		_acvmShrPtrPyrReliability[i].reset(new cv::Mat(nRowsDepth, nColsDepth, CV_32FC1));
		_acvmShrPtrPyrDepths[i]	 .reset();
		_acvmShrPtrPyrDepths[i]	 .reset(new cv::Mat(nRowsDepth,nColsDepth,CV_32FC1));

		_acvmShrPtrPyrRGBs[i].reset();
		_acvmShrPtrPyrRGBs[i].reset(new cv::Mat(nRowsRGB,nColsRGB,CV_8UC3));
		//device
		_agPyrPts[i] .reset();
		_agPyrPts[i] .reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC3));
		_agPyrNls[i] .reset();
		_agPyrNls[i] .reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC3));
		_agPyrReliability[i].reset();
		_agPyrReliability[i].reset(new cv::cuda::GpuMat(nRowsDepth, nColsDepth, CV_32FC1));
		_agPyrDepths[i].reset();
		_agPyrDepths[i].reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC1));

		_pry_mask[i].reset();
		_pry_mask[i].reset(new cv::cuda::GpuMat(nRowsRGB, nColsRGB, CV_8UC1));
	}

	//rendering
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 ); // 4
}

void btl::kinect::CKeyFrame::setRTw(const SO3Group<double>& Rw_, const Vector3d& Tw_){
	_R_cw = Rw_;
	_Tw = Tw_;
}

void btl::kinect::CKeyFrame::setRTFromPrjWfC(const Eigen::Affine3d& prj_wfc_){
	_R_cw = SO3Group<double>(prj_wfc_.linear().transpose());
	_Tw = _R_cw*(-prj_wfc_.translation());
}

void btl::kinect::CKeyFrame::setRTFromPrjCfW(const Eigen::Affine3d& prj_cfw_){
	_R_cw = SO3Group<double>(prj_cfw_.linear());
	_Tw = prj_cfw_.translation();
}

void btl::kinect::CKeyFrame::initRT(){
	_R_cw = SO3Group<double>();
	_Tw = _R_cw.inverse()*(-_initCw); 
}

void btl::kinect::CKeyFrame::copyRTFrom(const CKeyFrame& cFrame_ ){
	//assign rotation and translation 
	_R_cw = cFrame_._R_cw;
	_Tw = cFrame_._Tw;
}

void btl::kinect::CKeyFrame::assignRTfromGL(){
	btl::gl_util::CGLUtil::getRTFromWorld2CamCV(&_R_cw,&_Tw);
}

void btl::kinect::CKeyFrame::assignRT(const Vector3d& r_, const Vector3d& t_){
	_R_cw = SO3Group<double>::exp(r_);
	_Tw = t_;
}

void btl::kinect::CKeyFrame::copyTo( CKeyFrame* pKF_, const short sLevel_ ) const{
	//host
	if( !_acvmShrPtrPyrPts[sLevel_]->empty()) _acvmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrPts[sLevel_]);
	if( !_acvmShrPtrPyrNls[sLevel_]->empty()) _acvmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrNls[sLevel_]);
	_acvmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrRGBs[sLevel_]);
	//device
	if( !_agPyrPts[sLevel_]->empty()) _agPyrPts[sLevel_]->copyTo(*pKF_->_agPyrPts[sLevel_]);
	if (!_agPyrNls[sLevel_]->empty()) _agPyrNls[sLevel_]->copyTo(*pKF_->_agPyrNls[sLevel_]);
	if (!_agPyrReliability[sLevel_]->empty()) _agPyrReliability[sLevel_]->copyTo(*pKF_->_agPyrReliability[sLevel_]);
	if (!_pry_mask[sLevel_]->empty()) _pry_mask[sLevel_]->copyTo(*pKF_->_pry_mask[sLevel_]);
	_gRGB.copyTo(pKF_->_gRGB);
}

void btl::kinect::CKeyFrame::copyTo( CKeyFrame* pKF_ ) const{
	for(int i=0; i<_uPyrHeight; i++) {
		copyTo(pKF_,i);
	}
	_agPyrDepths[0]->copyTo(*pKF_->_agPyrDepths[0]);
	_acvmShrPtrPyrDepths[0]->copyTo(*pKF_->_acvmShrPtrPyrDepths[0]);
	//other
	pKF_->_R_cw = _R_cw;
	pKF_->_Tw = _Tw;
}

void btl::kinect::CKeyFrame::displayPointCloudInLocal(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usPyrLevel_){
	if (usPyrLevel_ >= _uPyrHeight) return;
	pGL_->gpuMapPtResources(*_agPyrPts[usPyrLevel_]);
	pGL_->gpuMapNlResources(*_agPyrNls[usPyrLevel_]);
	GpuMat rgb;  
	if (usPyrLevel_ > 0){
		cuda::resize(_gRGB, rgb, _agPyrNls[usPyrLevel_]->size());
	}
	else{
		rgb = _gRGB.clone();
	}
	if(!pGL_->_bEnableLighting) pGL_->gpuMapRGBResources(rgb);
	glDrawArrays(GL_POINTS, 0, btl::kinect::__aDepthWxH[usPyrLevel_+_uResolution] );
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	if(!pGL_->_bEnableLighting) glDisableClientState(GL_COLOR_ARRAY);
	glBindBuffer( GL_ARRAY_BUFFER, 0 );// it's crucially important for program correctness, it return the buffer to opengl rendering system.
	return;
}//gpuRenderVoxelInWorldCVGL()

void btl::kinect::CKeyFrame::renderCameraInWorld(btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, bool bRenderCamera_, const double& dPhysicalFocalLength_, const unsigned short uLevel_) {
	if (pGL_->_usPyrHeight != _uPyrHeight) return;
	glDisable(GL_LIGHTING);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	Eigen::Affine3d prj_c_f_w; prj_c_f_w.linear() = _R_cw.matrix(); prj_c_f_w.translation() = _Tw;
	Eigen::Affine3d prj_w_f_c = prj_c_f_w.inverse();
	glMultMatrixd(prj_w_f_c.data());//times with original model view matrix manipulated by mouse or keyboard

	glColor4f(1.f, 1.f, 1.f, .4f);
	glLineWidth(1);
#if USE_PBO
	//_pRGBCamera->renderCameraInLocal(*_pry_mask[pGL_->_usLevel], pGL_, bRenderCoordinate_, color_, dPhysicalFocalLength_, bRenderCamera_);//render small frame
	//_pRGBCamera->renderCameraInLocal(*_pry_mask[pGL_->_usLevel], pGL_, false, color_, 1.f, false);//render large frame
	_pRGBCamera->renderCameraInLocal(_gRGB, pGL_, bRenderCoordinate_, color_, dPhysicalFocalLength_, bRenderCamera_);//render small frame
	_pRGBCamera->renderCameraInLocal(_gRGB, pGL_, false, color_, 3.f, false);//render large frame
#else 
	if (bRenderCamera_) _pRGBCamera->loadTexture(*_acvmShrPtrPyrRGBs[uLevel_], &pGL_->_auTexture[pGL_->_usLevel]);
	_pRGBCamera->renderCameraInGLLocal(pGL_->_auTexture[pGL_->_usLevel], pGL_, bRenderCoordinate_, color_, dPhysicalFocalLength_, bRenderCamera_);
#endif	
	glPopMatrix();

	return;
}

void btl::kinect::CKeyFrame::renderPtsInWorld(btl::gl_util::CGLUtil::tp_ptr pGL_, const unsigned short uLevel_) {
	if (pGL_->_usPyrHeight != _uPyrHeight) return;
	//glDisable(GL_LIGHTING);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	Eigen::Affine3d prj_c_f_w; prj_c_f_w.linear() = _R_cw.matrix(); prj_c_f_w.translation() = _Tw;
	Eigen::Affine3d prj_w_f_c = prj_c_f_w.inverse();
	glMultMatrixd(prj_w_f_c.data());//times with original model view matrix manipulated by mouse or keyboard

	displayPointCloudInLocal(pGL_, uLevel_);
	
	glPopMatrix();

	return;
}

bool btl::kinect::CKeyFrame::isMovedwrtReferencInRadiusM(const CKeyFrame* const pRefFrame_, double dRotAngleThreshold_, double dTranslationThreshold_){
	using namespace btl::utility; //for operator <<
	//rotation angle
	cv::Mat_<double> cvmRRef,cvmRCur;
	cvmRRef << pRefFrame_->_R_cw.matrix();
	cvmRCur << _R_cw.matrix();
	cv::Mat_<double> cvmRVecRef,cvmRVecCur;
	cv::Rodrigues(cvmRRef,cvmRVecRef);
	cv::Rodrigues(cvmRCur,cvmRVecCur);
	cvmRVecCur -= cvmRVecRef;
	//get translation vector
	Vector3d eivCRef,eivCCur;
	eivCRef = pRefFrame_->_R_cw * (-pRefFrame_->_Tw);
	eivCCur =             _R_cw *             (-_Tw);
	eivCCur -= eivCRef;
	double dRot = cv::norm( cvmRVecCur, cv::NORM_L2 );
	double dTrn = eivCCur.norm();
	return ( dRot > dRotAngleThreshold_ || dTrn > dTranslationThreshold_);
}



