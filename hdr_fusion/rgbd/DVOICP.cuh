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