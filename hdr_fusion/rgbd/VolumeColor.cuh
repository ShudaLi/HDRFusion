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