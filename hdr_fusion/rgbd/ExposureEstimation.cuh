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