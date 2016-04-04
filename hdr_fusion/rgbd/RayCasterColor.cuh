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