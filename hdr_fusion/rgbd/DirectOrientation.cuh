#ifndef BTL_CUDA_DIRECT_RADIANCE
#define BTL_CUDA_DIRECT_RADIANCE
#include "DllExportDef.h"

namespace btl { namespace device {
	using namespace cv::cuda;
	using namespace pcl::device;

GpuMat DLL_EXPORT direct_rotation ( const Intr& intr_,
									const Matd33& R_rl_Kinv_,
									const Matd33& H_rl_,
									const GpuMat& NormalizedRadianceRef_,
									const GpuMat& NormalizedRadianceLiv_, const GpuMat& err_);

double DLL_EXPORT energy_direct_radiance_rotation(const Intr& intr_,
													const Matd33& R_rl_Kinv_,
													const Matd33& H_rl_,
													const GpuMat& NormalizedRadianceRef_,
													const GpuMat& NormalizedRadianceLiv_, const GpuMat& err_);

}//device
}//btl

#endif