#ifndef BTL_CUDA_VOLUME_COLOR_HEADER
#define BTL_CUDA_VOLUME_COLOR_HEADER
#include "DllExportDef.h"

namespace btl{
namespace device{
using namespace pcl::device;
using namespace cv::cuda;

void DLL_EXPORT cuda_fuse_depth_radiance_normal(const GpuMat& scaled_depth_, const GpuMat& surface_normal_, const GpuMat& normal_radiance_, const GpuMat& radiance_, const GpuMat& err_,
												const float fVoxelSize_, const float fTruncDistanceM_,
												const Matr33d& R_cf_, const double3& Of_,
												const Intr& intr, const short3& resolution_,
												GpuMat* ptr_tsdf_volume_, GpuMat* ptr_normali_radiance_volume_, GpuMat* ptr_radiance_volume_);
}//device
}//btl
#endif