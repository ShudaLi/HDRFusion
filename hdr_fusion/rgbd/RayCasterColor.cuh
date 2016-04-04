#ifndef BTL_CUDA_PCL_RAYCASTER_COLOR_HEADER
#define BTL_CUDA_PCL_RAYCASTER_COLOR_HEADER
#include "DllExportDef.h"

namespace btl{
namespace device{
	using namespace cv::cuda;
	using namespace pcl::device;
	void DLL_EXPORT cuda_ray_casting_radiance(const Intr& intr_, const Matd33& Rcf_t_, const Matd33& Rcf_, const double3& C_f_, 
								const float fTruncDistanceM_, const float& fVoxelSize_, const short3& resolution_, const float3& volume_size_,
								const GpuMat& tsdf_volume_, const GpuMat& color_volume_, GpuMat* pVMap_, GpuMat* pNMap_, GpuMat* pRadiance_);
	void DLL_EXPORT cuda_ray_casting_all(const Intr& intr_, const Matd33& Rcf_t_, const Matd33& Rcf_, const double3& C_f_,
										const float fTruncDistanceM_, const float& fVoxelSize_, const short3& resolution_, const float3& volume_size_,
										const GpuMat& tsdf_volume_, const GpuMat& nr_volume_, const GpuMat& radiance_volume_, 
										GpuMat* pVMap_, GpuMat* pNMap_, GpuMat* pRadiance_, GpuMat* pNR_);
} 
}


#endif