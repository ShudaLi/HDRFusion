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

#ifndef BTL_GEOMETRY_KINFU_TRACKER
#define BTL_GEOMETRY_KINFU_TRACKER

#include "DllExportDef.h"

namespace btl{ namespace geometry
{
	using namespace std;
	using namespace cv;
	using namespace cv::cuda;
	using namespace btl::kinect;
	using namespace Eigen;
	using namespace Sophus;

	class DLL_EXPORT CKinFuTracker
	{
	public:
		//type
		typedef boost::shared_ptr<CKinFuTracker> tp_shared_ptr;
		typedef CKinFuTracker* tp_ptr;

		public:
		//both pKeyFrame_ and pGlobalMap_ must be allocated before hand
		CKinFuTracker(CKeyFrame::tp_ptr pKeyFrame_, CTsdfBlock::tp_shared_ptr pGlobalMap_, 
						int nResolution_/*=0*/, int nPyrHeight_/*=3*/);
		~CKinFuTracker(){ ; }

		virtual bool init(CKeyFrame::tp_ptr pKeyFrame_);
		virtual void tracking(CKeyFrame::tp_ptr pCurFrame_);

		void getNextView(Eigen::Affine3d* pSystemPose_);
		void getPrevView( Eigen::Affine3d* pSystemPose_ );

		void displayCameraPath() const;
		void getCurrentProjectionMatrix( Eigen::Affine3d* pProjection_ ){
			*pProjection_ = _pose_refined_c_f_w; return;
		}

		void setResultFolder( const string& result_folder_ ){
			_path_to_result = result_folder_;
		}


	protected:
		virtual void storeCurrFrameSynthetic(CKeyFrame::tp_ptr pCurFrame_);
		double icp(CKeyFrame::tp_ptr pRefeFrame_, CKeyFrame::tp_ptr pLiveFrame_);
		double dvoICPIC(const CKeyFrame::tp_ptr pRefeFrame_, CKeyFrame::tp_ptr pLiveFrame_, const short asICPIterations_[], SE3Group<double>* pT_rl_, Eigen::Vector4i* pActualIter_) const;
		double directRotation(const CKeyFrame::tp_ptr pRefeFrame_, CKeyFrame::tp_ptr pLiveFrame_, SO3Group<double>* pR_rl_);
		void estimateExposure(CKeyFrame::tp_ptr pCurFrame_, const SE3Group<double>& T_rl);
		void constructLivePyr();
		void constructRefePyr();


		//camera matrix
		pcl::device::Intr _intrinsics;
		int _nCols, _nRows;

		string _path_to_result;
		Eigen::Affine3d _pose_refined_c_f_w;//projection matrix world to cam
		vector<Eigen::Affine3d> _v_T_cw_tracking;//a vector of projection matrices, where the projection matrices transform points defined in world to camera system.
	public:
		CTsdfBlock::tp_shared_ptr _pGlobalMap; //volumetric data
		CKeyFrame::tp_scoped_ptr _pRefFrame;

		double _aMinEnergy[5];
		int _nPyrHeight;
		int _nResolution;

		GpuMat _n_rad_ref[3];
		GpuMat _n_rad_live[3];
		GpuMat _err_live[3]; 
		GpuMat _n_rad_origin_2_ref; // for rotation estimation
		GpuMat _rad_ref[3];
		GpuMat _rad_live[3];
		GpuMat _err_ref[3];

		double _avg_exposure;

		string _serial_number;
		//for image analysis
		btl::device::CIntrinsicsAnalysisMult::tp_shared_ptr _pIAPrev;
		btl::device::CIntrinsicsAnalysisMult::tp_shared_ptr _pIACurr;
	};//CKinFuTracker

}//geometry
}//btl

#endif
