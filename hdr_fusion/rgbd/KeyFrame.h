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

#ifndef BTL_KEYFRAME
#define BTL_KEYFRAME
#define USE_PBO 1
#include "DllExportDef.h"

namespace btl { namespace kinect {

using namespace cv;
using namespace cv::cuda;
using namespace Eigen;
using namespace Sophus;

class DLL_EXPORT CKeyFrame {
	//type
public:
	typedef boost::shared_ptr< CKeyFrame > tp_shared_ptr;
	typedef boost::scoped_ptr< CKeyFrame > tp_scoped_ptr;
	typedef CKeyFrame* tp_ptr;

public:
    CKeyFrame( btl::image::SCamera::tp_ptr pRGBCamera_, ushort uResolution_, ushort uPyrLevel_, const Eigen::Vector3d& eivCw_ );
	CKeyFrame(const CKeyFrame::tp_ptr pFrame_);
	CKeyFrame(const CKeyFrame& Frame_);
	~CKeyFrame();
	
	//accumulate the relative R T to the global RT
	bool isMovedwrtReferencInRadiusM(const CKeyFrame* const pRefFrame_, double dRotAngleThreshold_, double dTranslationThreshold_);

	// set the opengl modelview matrix to align with the current view
	void getGLModelViewMatrix(Matrix4d* pModelViewGL_) const {
		*pModelViewGL_ = btl::utility::setModelViewGLfromRTCV ( _R_cw, _Tw );
		return;
	}
	Matrix4d getGLModelViewMatrix( ) const {
		return btl::utility::setModelViewGLfromRTCV ( _R_cw, _Tw );
	}

	const SO3Group<double>& getR() const {return _R_cw;}
	const Vector3d& getT() const {return _Tw;}
	void getPrjCfW(Eigen::Affine3d* ptr_proj_cfw) {
		ptr_proj_cfw->setIdentity();
		ptr_proj_cfw->translation() = _Tw;
		ptr_proj_cfw->linear() = _R_cw.matrix();
		return;
	}
	Eigen::Affine3d getPrjCfW() const {
		Eigen::Affine3d prj; prj.setIdentity();
		prj.linear() = _R_cw.matrix();
		prj.translation() = _Tw;
		return prj;
	}

	void setRTw(const SO3Group<double>& eimRotation_, const Vector3d& eivTw_);
	void setRTFromPrjWfC(const Eigen::Affine3d& prj_wfc_);
	void setRTFromPrjCfW(const Eigen::Affine3d& prj_cfw_);
	void initRT();
	void copyRTFrom(const CKeyFrame& cFrame_ );
	void assignRTfromGL();
	void assignRT(const Vector3d& r_, const Vector3d& t_);

	// render the camera location in the GL world
	void renderCameraInWorld(btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, bool bRenderCamera_, const double& dPhysicalFocalLength_, const unsigned short uLevel_);
	// render the depth in the GL world 
	void displayPointCloudInLocal(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usPyrLevel_);
	void renderPtsInWorld(btl::gl_util::CGLUtil::tp_ptr pGL_, const unsigned short uLevel_);

	// copy the content to another keyframe at 
	void copyTo( CKeyFrame* pKF_, const short sLevel_ ) const;
	void copyTo( CKeyFrame* pKF_ ) const;

	ushort pyrHeight() {return _uPyrHeight;}


private:
	//surf keyframe matching
	void allocate();
	float calcRTFromPair(const CKeyFrame& sPrevKF_, const double dDistanceThreshold_, unsigned short* pInliers_);
public:
	btl::image::SCamera::tp_ptr _pRGBCamera; //share the content of the RGBCamera with those from VideoKinectSource
	//host
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrDepths[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrNls[4]; //CV_32FC3 type
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrReliability[4]; //cpu version 
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrRGBs[4];
	//device
	boost::shared_ptr<GpuMat> _agPyrDepths[4];
	boost::shared_ptr<GpuMat> _agPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<GpuMat> _agPyrNls[4]; //CV_32FC3
	boost::shared_ptr<GpuMat> _agPyrReliability[4]; //CV_32FC1 ratio = largest / smallest eigen value
	boost::shared_ptr<GpuMat> _agPyrNormalized[4]; //CV_32FC1 ratio = largest / smallest eigen value
	GpuMat _gRGB;
	GpuMat _gRadiance;
	GpuMat _gNR;

	boost::shared_ptr<GpuMat> _pry_mask[4];
	//clusters
	//pose
	//R & T is the relative pose w.r.t. the coordinate defined in previous camera system.
	//R & T is defined using CV convention
	SO3Group<double> _R_cw;
	Vector3d _Tw; 
	Vector3d _initCw;

private:
	//for surf matching
	//host
	public:

	ushort _uPyrHeight;
	ushort _uResolution;
	
};//end of class



}//utility
}//btl

#endif
