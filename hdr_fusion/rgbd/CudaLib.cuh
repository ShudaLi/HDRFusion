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

#ifndef BTL_CUDA_HEADER
#define BTL_CUDA_HEADER
#include "DllExportDef.h"

namespace btl { namespace device
{
	using namespace cv::cuda;
void DLL_EXPORT cvt_depth2disparity2( const GpuMat& cvgmDepth_, float fCutOffDistance_, GpuMat* pcvgmDisparity_ );
void DLL_EXPORT cvt_disparity2depth( const GpuMat& cvgmDisparity_, GpuMat* pcvgmDepth_ );
void DLL_EXPORT unproject_ir( const GpuMat& cvgmDepth_ , 
									 const float& dFxIR_, const float& dFyIR_, const float& uIR_, const float& vIR_, 
									 GpuMat* pcvgmIRWorld_,float factor_ =1000.f );
//template void DLL_EXPORT cudaTransformIR2RGB<float>(const GpuMat& cvgmIRWorld_, const T* aR_, const T* aRT_, GpuMat* pcvgmRGBWorld_);
void DLL_EXPORT align_ir2rgb(const GpuMat& cvgmIRWorld_, const float* aR_, const float* aRT_, GpuMat* pcvgmRGBWorld_);
void DLL_EXPORT project_rgb(const GpuMat& cvgmRGBWorld_, 
	const float& dFxRGB_, const float& dFyRGB_, const float& uRGB_, const float& vRGB_, 
	GpuMat* pcvgmAligned_ );
void DLL_EXPORT bilateral_filtering(const GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, GpuMat* pcvgmDst_ );
void DLL_EXPORT pyr_down (const GpuMat& cvgmSrc_, const float& fSigmaColor_, GpuMat* pcvgmDst_);
void DLL_EXPORT unproject_rgb ( const GpuMat& cvgmDepths_, 
	const float& fFxRGB_,const float& fFyRGB_,const float& uRGB_, const float& vRGB_, unsigned int uLevel_, 
	GpuMat* pcvgmPts_ );
void DLL_EXPORT fast_normal_estimation(const GpuMat& cvgmPts_, GpuMat* pcvgmNls_ );
//get scale depth
void DLL_EXPORT scale_depth(unsigned short usPyrLevel_, const float fFx_, const float fFy_, const float u_, const float v_, GpuMat* pcvgmDepth_);
void DLL_EXPORT transform_local2world(const double* pRw_/*col major*/, const double* pTw_, GpuMat* pcvgmPts_, GpuMat* pcvgmNls_, GpuMat* pcvgmMDs_);
//resize the normal or vertex map to half of its size
void DLL_EXPORT resize_map (bool bNormalize_, const GpuMat& cvgmSrc_, GpuMat* pcvgmDst_);
void DLL_EXPORT init_idx(int nCols_, int type_, GpuMat* p_idx_);
void DLL_EXPORT sort_column_idx(GpuMat* const& p_key_, GpuMat*const & p_idx_);
void DLL_EXPORT calc_deriv(const GpuMat& rad_re_, const GpuMat& inten_, GpuMat& derivCRF_);
void DLL_EXPORT calc_avg_min_frame(const GpuMat& rad_b_, const GpuMat& rad_g_, const GpuMat& rad_r_, GpuMat* p_rad_,
	const GpuMat& err_b_, const GpuMat& err_g_, const GpuMat& err_r_, GpuMat* p_err_);
void DLL_EXPORT calc_avg_frame(const GpuMat& rad_b_, const GpuMat& rad_g_, const GpuMat& rad_r_, GpuMat* p_rad_);
}//device
}//btl
#endif
