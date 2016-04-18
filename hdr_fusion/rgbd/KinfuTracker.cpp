//Copyright(c) 2016 Shuda Li[lishuda1980@gmail.com]
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//FOR A PARTICULAR PURPOSE AND NON - INFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
//COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define EXPORT 
#define INFO
#define _USE_MATH_DEFINES
#define  NOMINMAX 
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//stl
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#ifdef __gnu_linux__
#include <sys/types.h>
#include <sys/stat.h>
#elif _WIN32
#include <direct.h>
#else 
#error "OS not supported!"
#endif

#include <math.h>
#include <limits>
//boost
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/special_functions/fpclassify.hpp> //isnan
#include <boost/lexical_cast.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/filesystem.hpp>

//openncv
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>

#include <se3.hpp>

#include <utility>
#include <OpenNI.h>
#include "Converters.hpp"
#include "GLUtil.hpp"
#include "CVUtil.hpp"
#include "EigenUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "IntrinsicAnalysis.cuh"
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "TSDF.h"

#include "KinfuTracker.h"
#include "CudaLib.cuh" 

#include "DVOICP.cuh"
#include "DirectOrientation.cuh"
#include "ExposureEstimation.cuh"
#include "Utility.hpp"
#include "Kinect.h"
#include "pcl/vector_math.hpp"
#include "Converters.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace std;
using namespace btl::kinect;
using namespace btl::utility;
using namespace btl::image;
using namespace btl::device;
//using namespace pcl::device;
using namespace Eigen;
using namespace Sophus;

namespace btl{ namespace geometry
{

CKinFuTracker::CKinFuTracker(CKeyFrame::tp_ptr pKeyFrame_, CTsdfBlock::tp_shared_ptr pGlobalMap_, int nResolution_/*=0*/, int nPyrHeight_/*=3*/)
	:_pGlobalMap(pGlobalMap_),_nResolution(nResolution_),_nPyrHeight(nPyrHeight_)
{
	_v_T_cw_tracking.clear();

	_aMinEnergy[0] = .5f;//6.f;
	_aMinEnergy[1] = .5f;//100.f;
	_aMinEnergy[2] = .1f;//2000.f;
	_aMinEnergy[3] = 1.f;// 2000.f;
	_aMinEnergy[4] = 2000.f;//1000.f;

}

bool CKinFuTracker::init(const CKeyFrame::tp_ptr pCurFrame_){
	_nCols = pCurFrame_->_agPyrPts[0]->cols;
	_nRows = pCurFrame_->_agPyrPts[0]->rows;

	ostringstream directory;
	directory << "..//data//" << _serial_number << "//crf.yml";
	if (!_pIAPrev)
		_pIAPrev.reset(new btl::device::CIntrinsicsAnalysisMult(directory.str(), _nRows, _nCols, .25, 40));
	if (!_pIACurr)
		_pIACurr.reset(new btl::device::CIntrinsicsAnalysisMult(directory.str(), _nRows, _nCols, .25, 40));
	GpuMat bgr;
	cuda::cvtColor(pCurFrame_->_gRGB, bgr, CV_RGB2BGR);
	_pIACurr->analysis(bgr);
	constructLivePyr();

	_pRefFrame.reset();
	_pRefFrame.reset(new CKeyFrame(pCurFrame_));	//copy pKeyFrame_ to _pPrevFrameWorld
	
	//integrate the frame into the world
	GpuMat radiance;
	cuda::merge(_pIACurr->_vRadianceBGR, radiance);
	GpuMat er;
	cuda::merge(_pIACurr->_error_bgr, 3, er);
	GpuMat normal_radiance;
	cuda::merge(_pIACurr->_normalized_bgr, 3, normal_radiance);
	_pGlobalMap->gpuIntegrateDepth(*pCurFrame_->_agPyrDepths[0], *pCurFrame_->_agPyrNls[0], normal_radiance, radiance, er, 
									SE3Group<double>(pCurFrame_->_R_cw, pCurFrame_->_Tw));

	//initialize pose
	pCurFrame_->getPrjCfW(&_pose_refined_c_f_w);
	_v_T_cw_tracking.push_back(_pose_refined_c_f_w);

	//store current key point
	storeCurrFrameSynthetic(pCurFrame_);
	//storeCurrFrameReal(pCurFrame_);
	PRINTSTR("End of init");

	return true;
}

void CKinFuTracker::storeCurrFrameSynthetic(CKeyFrame::tp_ptr pCurFrame_){
	pCurFrame_->copyTo(&*_pRefFrame);
	_n_rad_origin_2_ref = _n_rad_live[2].clone();
	_pIACurr->copyTo(*_pIAPrev);

	if (_pRefFrame->_gRadiance.empty()){
		_pRefFrame->_gRadiance.create(_pRefFrame->_agPyrPts[0]->size(), CV_32FC3);
	}
	if (_pRefFrame->_gNR.empty()){
		_pRefFrame->_gNR.create(_pRefFrame->_agPyrPts[0]->size(), CV_32FC3);
	}

	_pGlobalMap->gpuRayCastingAll( _pRefFrame->_pRGBCamera->getIntrinsics(0), _pRefFrame->_R_cw, _pRefFrame->_Tw,
										&*_pRefFrame->_agPyrPts[0], &*_pRefFrame->_agPyrNls[0], &_pRefFrame->_gRadiance, &_pRefFrame->_gNR); 
	cuda::split(_pRefFrame->_gNR, _pIAPrev->_normalized_bgr);
	cuda::split(_pRefFrame->_gRadiance, _pIAPrev->_vRadianceBGR);

	constructRefePyr();
	
	pCurFrame_->getPrjCfW(&_pose_refined_c_f_w);
	_v_T_cw_tracking.push_back(_pose_refined_c_f_w);

	return;
}

void CKinFuTracker::displayCameraPath() const{
	vector<Eigen::Affine3d>::const_iterator cit = _v_T_cw_tracking.begin(); //from world to camera
	
	glDisable(GL_LIGHTING);
	glColor3f ( 0.f,0.f,1.f); 
	glLineWidth(2.f);
	glBegin(GL_LINE_STRIP);
	for (; cit != _v_T_cw_tracking.end(); cit++ )
	{
		SO3Group<double> mR = cit->linear();
		Vector3d vT = cit->translation();
		Vector3d vC = mR.inverse() *(-vT);
		vC = mR.inverse()*(-vT);
		glVertex3dv( vC.data() );
	}
	glEnd();
	return;
}

void CKinFuTracker::getPrevView( Eigen::Affine3d* pSystemPose_ ){
	int s = _v_T_cw_tracking.size();
	*pSystemPose_ = _v_T_cw_tracking[s-2];
	return;
}
void CKinFuTracker::getNextView( Eigen::Affine3d* pSystemPose_ ){
	*pSystemPose_ = _v_T_cw_tracking.front();
	PRINT(_v_T_cw_tracking.front().matrix());
	return;
}

double CKinFuTracker::dvoICPIC(const CKeyFrame::tp_ptr pRefeFrame_, CKeyFrame::tp_ptr pLiveFrame_, const short asICPIterations_[], SE3Group<double>* pT_rl_, Eigen::Vector4i* pActualIter_) const
{
	SE3Group<double> PrevT_rl = *pT_rl_;
	SE3Group<double> NewT_rl = *pT_rl_;
	//get R,T of previous 
	Matrix3d R_rl_t_tmp = PrevT_rl.so3().inverse().matrix();
	const Matd33&  devR_rl = pcl::device::device_cast<pcl::device::Matd33> (R_rl_t_tmp); //implicit inverse

	Vector3d t_rl = PrevT_rl.translation();
	const double3& devT_rl = pcl::device::device_cast<double3> (t_rl);

	//from low resolution to high
	double dCurEnergy = numeric_limits<double>::max();
	for (short sPyrLevel = pLiveFrame_->pyrHeight() - 1; sPyrLevel >= 0; sPyrLevel--){
		// for each pyramid level we have a min energy and corresponding best R t
		if (asICPIterations_[sPyrLevel] > 0){
			dCurEnergy = btl::device::dvo_icp_energy(pLiveFrame_->_pRGBCamera->getIntrinsics(sPyrLevel),
				devR_rl, devT_rl,
				*pRefeFrame_->_agPyrPts[sPyrLevel], *pRefeFrame_->_agPyrNls[sPyrLevel], _n_rad_ref[sPyrLevel],
				*pLiveFrame_->_agPyrPts[sPyrLevel], *pLiveFrame_->_agPyrNls[sPyrLevel], _n_rad_live[sPyrLevel],
				*pLiveFrame_->_agPyrDepths[sPyrLevel], _err_live[sPyrLevel], *pLiveFrame_->_pry_mask[sPyrLevel]);
			//PRINT(dMinEnergy);
		}

		SE3Group<double> MinT_rl = NewT_rl;
		double dMin = dCurEnergy;
		double dPrevEnergy = dCurEnergy;
		for (short sIter = 0; sIter < asICPIterations_[sPyrLevel]; ++sIter) {
			//get R and T
			GpuMat cvgmSumBuf = btl::device::dvo_icp(pLiveFrame_->_pRGBCamera->getIntrinsics(sPyrLevel),
				devR_rl, devT_rl,
				*pRefeFrame_->_agPyrPts[sPyrLevel], *pRefeFrame_->_agPyrNls[sPyrLevel], _n_rad_ref[sPyrLevel],
				*pLiveFrame_->_agPyrPts[sPyrLevel], *pLiveFrame_->_agPyrNls[sPyrLevel], _n_rad_live[sPyrLevel],
				*pLiveFrame_->_agPyrDepths[sPyrLevel], _err_live[sPyrLevel], *pLiveFrame_->_pry_mask[sPyrLevel]);
			Mat Buf; cvgmSumBuf.download(Buf);
			SE3Group<double> Tran_nc = btl::utility::extractRTFromBuffer<double>((double*)Buf.data);
			NewT_rl = Tran_nc * PrevT_rl;
			R_rl_t_tmp = NewT_rl.so3().inverse().matrix();
			t_rl = NewT_rl.translation();
			dCurEnergy = btl::device::dvo_icp_energy(pLiveFrame_->_pRGBCamera->getIntrinsics(sPyrLevel),
				devR_rl, devT_rl,
				*pRefeFrame_->_agPyrPts[sPyrLevel], *pRefeFrame_->_agPyrNls[sPyrLevel], _n_rad_ref[sPyrLevel],
				*pLiveFrame_->_agPyrPts[sPyrLevel], *pLiveFrame_->_agPyrNls[sPyrLevel], _n_rad_live[sPyrLevel],
				*pLiveFrame_->_agPyrDepths[sPyrLevel], _err_live[sPyrLevel], *pLiveFrame_->_pry_mask[sPyrLevel]);
			//cout << sIter << ": " << dPrevEnergy << " " << dCurEnergy << endl;
			if (dCurEnergy < dMin){
				dMin = dCurEnergy;
				MinT_rl = NewT_rl;
			}
			if (dMin / dCurEnergy > 1.125){ //diverges
				//cout << "Diverge Warning:" << endl;
				//cout <<"New "<< NewT_rl.matrix() << endl;
				//cout <<"Prev" <<PrevT_rl.matrix() << endl;
				NewT_rl = MinT_rl;
				dCurEnergy = dMin;
				break;
			}
			PrevT_rl = NewT_rl;
			if (fabs(dPrevEnergy / dCurEnergy - 1) < 1e-6f){ //converges
				//cout << "Converges" << endl;
				dCurEnergy = dMin;
				NewT_rl = MinT_rl;
				break;
			}
			dPrevEnergy = dCurEnergy;
		}//for each iteration
	}//for pyrlevel
	*pT_rl_ = NewT_rl;
	SE3Group<double> T_rw(pRefeFrame_->_R_cw, pRefeFrame_->_Tw);
	T_rw = NewT_rl.inverse()*T_rw;
	pLiveFrame_->_R_cw = T_rw.so3();
	pLiveFrame_->_Tw = T_rw.translation();

	return dCurEnergy;
}

double CKinFuTracker::directRotation(const CKeyFrame::tp_ptr pRefeFrame_, const CKeyFrame::tp_ptr pLiveFrame_, SO3Group<double>* pR_rl_)
{
	Intr sCamIntr_ = pRefeFrame_->_pRGBCamera->getIntrinsics(2);
	Matrix3d K = Matrix3d::Identity();
	//note that camera parameters are 
	K(0, 0) = sCamIntr_.fx;
	K(1, 1) = sCamIntr_.fy;
	K(0, 2) = sCamIntr_.cx;
	K(1, 2) = sCamIntr_.cy;
	SO3Group<double> CurR_rl_ = *pR_rl_;
	SO3Group<double> PrevR_rl_ = *pR_rl_;
	SO3Group<double> MinR_rl_ = *pR_rl_;
	Matrix3d R_rl_Kinv = PrevR_rl_.matrix() *K.inverse();
	Matrix3d H_rl = K * R_rl_Kinv;

	//get R,T of previous 
	Matrix3d H_rl_t = H_rl.transpose();
	Matrix3d R_rl_Kinv_t = R_rl_Kinv.transpose();
	const Matd33&  devH_rl = pcl::device::device_cast<pcl::device::Matd33> (H_rl_t);
	const Matd33&  devR_rl_Kinv = pcl::device::device_cast<pcl::device::Matd33> (R_rl_Kinv_t);
	double dMinEnergy = numeric_limits<double>::max();
	double dPrevEnergy = numeric_limits<double>::max();
	dPrevEnergy = energy_direct_radiance_rotation(sCamIntr_, devR_rl_Kinv, devH_rl, _n_rad_origin_2_ref, _n_rad_live[2], _err_live[2]);
	dMinEnergy = dPrevEnergy;
	//cout << setprecision(15) << dMinEnergy << endl;
	for (short sIter = 0; sIter < 5; ++sIter) {
		//get R and T
		GpuMat gSumBuf = btl::device::direct_rotation(sCamIntr_, devR_rl_Kinv, devH_rl, _n_rad_origin_2_ref, _n_rad_live[2], _err_live[2]);
		Mat Buf; gSumBuf.download(Buf);
		SO3Group<double> R_rl = btl::utility::extractRFromBuffer<double>((double*)Buf.data);
		//cout << Tran_nc.matrix() << endl;
		CurR_rl_ = R_rl *PrevR_rl_;
		R_rl_Kinv = CurR_rl_.matrix()*K.inverse();
		H_rl = K * R_rl_Kinv;

		H_rl_t = H_rl.transpose();
		R_rl_Kinv_t = R_rl_Kinv.transpose();
		double dCurEnergy = energy_direct_radiance_rotation(sCamIntr_, devR_rl_Kinv, devH_rl, _n_rad_origin_2_ref, _n_rad_live[2], _err_live[2]);
		//cout << sIter << ": " << dPrevEnergy << " " << dCurEnergy << endl;
		if (dCurEnergy < dMinEnergy){
			dMinEnergy = dCurEnergy;
			MinR_rl_ = CurR_rl_;
		}
		if (dMinEnergy / dCurEnergy < 0.25){ //divereges
			//cout << "Diverge Warning:" << endl;
			dCurEnergy = dMinEnergy;
			CurR_rl_ = MinR_rl_;
			break;
		}
		PrevR_rl_ = CurR_rl_;
		if (fabs(dPrevEnergy / dCurEnergy - 1) < 0.01f){ //converges
			//cout << "Converges" << endl;
			dCurEnergy = dMinEnergy;
			CurR_rl_ = MinR_rl_;
			break;
		}
		dPrevEnergy = dCurEnergy;
	}
	*pR_rl_ = CurR_rl_;
	return dMinEnergy;
}

double CKinFuTracker::icp(CKeyFrame::tp_ptr pRefeFrame_, CKeyFrame::tp_ptr pLiveFrame_){
	using namespace btl::device;
	//using the ICP to refine each hypotheses and store their alignment score
	double dICPEnergy = numeric_limits<double>::max();
	//do ICP as one of pose hypotheses
	SO3Group<double> R_rl; Vector3d T(0, 0, 0);
	//calc low for rotation estimation
	//Newcombe, R. A., Lovegrove, S. J., & Davison, A. J. (2011). DTAM : Dense Tracking and Mapping in Real-Time. In ICCV. Retrieved from http://www.youtube.com/watch?v=Df9WhgibCQA
	double E = directRotation(pRefeFrame_, pLiveFrame_, &R_rl);
	//cout << R_rl.matrix() << endl;
	Eigen::Vector4i eivIter;
	SE3Group<double> T_rl(R_rl, T);
	short asICPIterations[4] = { 4, 3, 3, 1 };
	dICPEnergy = dvoICPIC(pRefeFrame_, pLiveFrame_, asICPIterations, &T_rl, &eivIter);
	//estimate dt
	estimateExposure(pLiveFrame_, T_rl);

	return dICPEnergy;
}

void CKinFuTracker::estimateExposure(CKeyFrame::tp_ptr pCurFrame_, const SE3Group<double>& T_rl){
	SE3Group<double> T_lr = T_rl.inverse();
	//get R,T of previous 
	Matrix3d R_lr_t_tmp = T_lr.so3().inverse().matrix();
	const Matd33&  devR_lr = pcl::device::device_cast<pcl::device::Matd33> (R_lr_t_tmp); //implicit inverse

	Vector3d tt = T_lr.translation();
	const double3& devT_lr = pcl::device::device_cast<double3> (tt);

	GpuMat mask;
	if(true){
		_avg_exposure = btl::device::exposure_est2(pCurFrame_->_pRGBCamera->getIntrinsics(0), devR_lr, devT_lr,
			_pIACurr->_vRadianceBGR[0], _pIACurr->_error_bgr[0],
			_pIACurr->_vRadianceBGR[1], _pIACurr->_error_bgr[1], 
			_pIACurr->_vRadianceBGR[2], _pIACurr->_error_bgr[2], 
			*_pRefFrame->_agPyrPts[0], _pIAPrev->_vRadianceBGR[0], _pIAPrev->_vRadianceBGR[1], _pIAPrev->_vRadianceBGR[2], mask);
		//cout << "avg exposure: " << _avg_exposure << endl;
	}

	for (int ch = 0; ch < 3; ch++) {
		_pIACurr->_vRadianceBGR[ch].convertTo(_pIACurr->_vRadianceBGR[ch], CV_32FC1, _avg_exposure);
	}

	if(false){
		btl::device::remove_outlier(pCurFrame_->_pRGBCamera->getIntrinsics(0), devR_lr, devT_lr,
			_pIACurr->_vRadianceBGR[0], _pIACurr->_error_bgr[0],
			_pIACurr->_vRadianceBGR[1], _pIACurr->_error_bgr[1],
			_pIACurr->_vRadianceBGR[2], _pIACurr->_error_bgr[2],
			*_pRefFrame->_agPyrPts[0], _pIAPrev->_vRadianceBGR[0], _pIAPrev->_vRadianceBGR[1], _pIAPrev->_vRadianceBGR[2], mask);
	}

	return;
}

void CKinFuTracker::constructLivePyr(){
	//construct pyramid of normalized_radiance;
	calc_avg_min_frame( _pIACurr->_normalized_bgr[0], _pIACurr->_normalized_bgr[1], _pIACurr->_normalized_bgr[2], &(_n_rad_live[0]),
						_pIACurr->_error_bgr[0], _pIACurr->_error_bgr[1], _pIACurr->_error_bgr[2], &(_err_live[0]));

	cuda::pyrDown(_n_rad_live[0], _n_rad_live[1]);
	cuda::pyrDown(_n_rad_live[1], _n_rad_live[2]);
	cuda::pyrDown(_err_live[0], _err_live[1]);
	cuda::pyrDown(_err_live[1], _err_live[2]);

}

void CKinFuTracker::constructRefePyr(){
	//construct pyramid of normalized_radiance;
	calc_avg_frame(_pIAPrev->_normalized_bgr[0], _pIAPrev->_normalized_bgr[1], _pIAPrev->_normalized_bgr[2], &(_n_rad_ref[0]));

	cuda::pyrDown(_n_rad_ref[0], _n_rad_ref[1]);
	cuda::pyrDown(_n_rad_ref[1], _n_rad_ref[2]);
}

void CKinFuTracker::tracking(CKeyFrame::tp_ptr pCurFrame_)
{
	GpuMat bgr;
	cuda::cvtColor(pCurFrame_->_gRGB, bgr, CV_RGB2BGR);
	_pIACurr->analysis(bgr);

	constructLivePyr();

	if (icp(_pRefFrame.get(), pCurFrame_) < _aMinEnergy[_nResolution])
	{
		SE3Group<double> T_cw(pCurFrame_->_R_cw, pCurFrame_->_Tw);
		GpuMat radiance;
		cuda::merge(_pIACurr->_vRadianceBGR, radiance);
		GpuMat er;
		cuda::merge(_pIACurr->_error_bgr, 3, er);
		GpuMat normal_radiance;
		cuda::merge(_pIACurr->_normalized_bgr, 3, normal_radiance);
		_pGlobalMap->gpuIntegrateDepth(*pCurFrame_->_agPyrDepths[0], *pCurFrame_->_agPyrNls[0], normal_radiance, radiance, er, 
										SE3Group<double>(pCurFrame_->_R_cw, pCurFrame_->_Tw));
		//insert features into feature-base
		pCurFrame_->getPrjCfW(&_pose_refined_c_f_w);
		_v_T_cw_tracking.push_back(_pose_refined_c_f_w);

		storeCurrFrameSynthetic(pCurFrame_);
	}//if current frame is lost aligned

	return;
}//track

}//geometry
}//btl
