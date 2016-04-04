#ifndef BTL_CUDA_DVO_ICP_HEADER
#define BTL_CUDA_DVO_ICP_HEADER
#include "DllExportDef.h"

namespace btl { namespace device {
	using namespace cv::cuda;
	using namespace pcl::device;
GpuMat DLL_EXPORT dvo_icp(const Intr& sCamIntr_,
							const Matd33& R_rl_, const double3& T_rl_,
							const GpuMat& VMapRef_, const GpuMat& NMapRef_, const GpuMat& NIRef_,
							const GpuMat& VMapLive_, const GpuMat& NMapLive_, const GpuMat& NILive_, const GpuMat& DepthLive_, const GpuMat& ErrLive_, GpuMat& mask_);

double DLL_EXPORT dvo_icp_energy(const Intr& sCamIntr_,
							const Matd33& R_rl_, const double3& T_rl_,
							const GpuMat& VMapRef_, const GpuMat& NMapRef_, const GpuMat& NIRef_,
							const GpuMat& VMapLive_, const GpuMat& NMapLive_, const GpuMat& NILive_, const GpuMat& DepthLive_, const GpuMat& ErrLive_, GpuMat& mask_, float icp_weight = 1.f);

}//device
}//btl

#endif