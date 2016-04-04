#ifndef BTL_CUDA_EXPOSURE_HEADER
#define BTL_CUDA_EXPOSURE_HEADER
#include "DllExportDef.h"

namespace btl { namespace device {
	using namespace cv::cuda;
	using namespace pcl::device;

double DLL_EXPORT exposure_est2(const Intr& sCamIntr_,
								const Matd33& R_lr_, const double3& T_lr_,
								const GpuMat& RadianceBLive_, const GpuMat& errBLive_,
								const GpuMat& RadianceGLive_, const GpuMat& errGLive_,
								const GpuMat& RadianceRLive_, const GpuMat& errRLive_,
								const GpuMat& VMapRef_, const GpuMat& RadianceBRef_, const GpuMat& RadianceGRef_, const GpuMat& RadianceRRef_, GpuMat& mask_);

void DLL_EXPORT remove_outlier(const Intr& sCamIntr_,
								const Matd33& R_lr_, const double3& T_lr_, 
								const GpuMat& RadianceBLive_, const GpuMat& errBLive_,
								const GpuMat& RadianceGLive_, const GpuMat& errGLive_,
								const GpuMat& RadianceRLive_, const GpuMat& errRLive_,
								const GpuMat& VMapRef_, const GpuMat& RadianceBRef_, const GpuMat& RadianceGRef_, const GpuMat& RadianceRRef_, GpuMat& mask_);
}//device
}//btl

#endif