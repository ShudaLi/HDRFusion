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
#define EXPORT

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <thrust/device_ptr.h> 
#include <thrust/sort.h> 


#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda/common.hpp>

#include "OtherUtil.hpp"
#include <math_constants.h>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include <vector>
#include <limits.h>
#include "CudaLib.cuh"

using namespace cv;
using namespace cv::cuda;

namespace btl{ namespace device		
{

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//depth to disparity
__global__ void kernelInverse(const cv::cuda::PtrStepSz<float> cvgmIn_, cv::cuda::PtrStepSz<float> cvgmOut_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	if (nX >= cvgmIn_.cols || nY >= cvgmIn_.rows) return;
	if(fabsf(cvgmIn_.ptr(nY)[nX]) > 0.01f )
		cvgmOut_.ptr(nY)[nX] = 1.f/cvgmIn_.ptr(nY)[nX];
	else
		cvgmOut_.ptr(nY)[nX] = 0;//pcl::device::numeric_limits<float>::quiet_NaN();
}//kernelInverse

void cudaDepth2Disparity( const cv::cuda::GpuMat& cvgmDepth_, cv::cuda::GpuMat* pcvgmDisparity_ ){
	//not necessary as pcvgmDisparity has been allocated in VideoSourceKinect()
	//pcvgmDisparity_->create(cvgmDepth_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmDepth_.cols, block.x), cv::cuda::device::divUp(cvgmDepth_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDepth_,*pcvgmDisparity_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}//cudaDepth2Disparity

__global__ void kernelInverse2(const cv::cuda::PtrStepSz<float> cvgmIn_, float fCutOffDistance_, cv::cuda::PtrStepSz<float> cvgmOut_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	//if (nX <2)
		//printf("nX %d, nY %d, out %f\n", nX, nY);
	if (nX >= cvgmIn_.cols || nY >= cvgmIn_.rows) return;

	if (fabsf(cvgmIn_.ptr(nY)[nX]) > 0.01f && cvgmIn_.ptr(nY)[nX] < fCutOffDistance_){
		float tmp = 1. / cvgmIn_.ptr(nY)[nX];
		cvgmOut_.ptr(nY)[nX] = tmp;
	}
	else{
		cvgmOut_.ptr(nY)[nX] = 0; //pcl::device::numeric_limits<float>::quiet_NaN();
	}
}//kernelInverse

void cvt_depth2disparity2( const cv::cuda::GpuMat& cvgmDepth_, float fCutOffDistance_, cv::cuda::GpuMat* pcvgmDisparity_ ){
	//convert the depth from mm to m
	//not necessary as pcvgmDisparity has been allocated in VideoSourceKinect()
	//pcvgmDisparity_->create(cvgmDepth_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmDepth_.cols, block.x), cv::cuda::device::divUp(cvgmDepth_.rows, block.y));
	//run kernel
	kernelInverse2<<<grid,block>>>( cvgmDepth_,fCutOffDistance_, *pcvgmDisparity_ );
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());
	return;
}//cudaDepth2Disparity


void cvt_disparity2depth( const cv::cuda::GpuMat& cvgmDisparity_, cv::cuda::GpuMat* pcvgmDepth_ ){
	pcvgmDepth_->create(cvgmDisparity_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmDisparity_.cols, block.x), cv::cuda::device::divUp(cvgmDisparity_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDisparity_,*pcvgmDepth_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//global constant used by kernelUnprojectIR() and cudaUnProjectIR()
__constant__ float _aIRCameraParameter[4];// 1/f_x, 1/f_y, u, v for IR camera; constant memory declaration
//
__global__ void kernelUnprojectIRCVmmCVm(const cv::cuda::PtrStepSz<float> cvgmDepth_, const float factor_,
	cv::cuda::PtrStepSz<float3> cvgmIRWorld_) {
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if (nX >= cvgmIRWorld_.cols || nY >= cvgmIRWorld_.rows) return;

	const float& fDepth = cvgmDepth_.ptr(nY)[nX];
	float3& temp = cvgmIRWorld_.ptr(nY)[nX];
		
	if( factor_*0.4f < fDepth && fDepth < factor_*4.f ){ //truncate, fDepth is captured from openni and always > 0
		temp.z = fDepth / factor_;//convert to meter z 5 million meter is added according to experience. as the OpenNI
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		temp.x = (nX - _aIRCameraParameter[2]) * _aIRCameraParameter[0] * temp.z;
		temp.y = (nY - _aIRCameraParameter[3]) * _aIRCameraParameter[1] * temp.z;
	}//if within 0.4m - 4m
	else{
		temp.x = temp.y = temp.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}//else

	return;
}//kernelUnprojectIRCVCV

void unproject_ir(const cv::cuda::GpuMat& cvgmDepth_ ,
						const float& fFxIR_, const float& fFyIR_, const float& uIR_, const float& vIR_, 
						cv::cuda::GpuMat* pcvgmIRWorld_, float factor_ /*=1000.f*/)
{
	//constant definition
	size_t sN = sizeof(float) * 4;
	float* const pIRCameraParameters = (float*) malloc( sN );
	pIRCameraParameters[0] = 1.f/fFxIR_;
	pIRCameraParameters[1] = 1.f/fFyIR_;
	pIRCameraParameters[2] = uIR_;
	pIRCameraParameters[3] = vIR_;
	cudaSafeCall( cudaMemcpyToSymbol(_aIRCameraParameter, pIRCameraParameters, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmDepth_.cols, block.x), cv::cuda::device::divUp(cvgmDepth_.rows, block.y));
	//run kernel
    kernelUnprojectIRCVmmCVm<<<grid,block>>>( cvgmDepth_, factor_, *pcvgmIRWorld_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());

	//release temporary pointers
	free(pIRCameraParameters);
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//global constant used by kernelUnprojectIR() and cudaTransformIR2RGB()
__constant__ float _aR[9];
__constant__ float _aRT[3];
__global__ void kernelTransformIR2RGBCVmCVm(const cv::cuda::PtrStepSz<float3> cvgmIRWorld_, cv::cuda::PtrStepSz<float3> cvgmRGBWorld_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if (nX >= cvgmRGBWorld_.cols || nY >= cvgmRGBWorld_.rows) return;

	float3& rgbWorld = cvgmRGBWorld_.ptr(nY)[nX];
	const float3& irWorld  = cvgmIRWorld_ .ptr(nY)[nX];
	if( 0.4f < irWorld.z && irWorld.z < 10.f ) {
		//_aR[0] [1] [2] //row major
		//   [3] [4] [5]
		//   [6] [7] [8]
		//_aT[0]
		//   [1]
		//   [2]
		//  pRGB_ = _aR * ( pIR_ - _aT )
		//  	  = _aR * pIR_ - _aR * _aT
		//  	  = _aR * pIR_ - _aRT
		rgbWorld.x = _aR[0] * irWorld.x + _aR[1] * irWorld.y + _aR[2] * irWorld.z - _aRT[0];
		rgbWorld.y = _aR[3] * irWorld.x + _aR[4] * irWorld.y + _aR[5] * irWorld.z - _aRT[1];
		rgbWorld.z = _aR[6] * irWorld.x + _aR[7] * irWorld.y + _aR[8] * irWorld.z - _aRT[2];
	}//if irWorld.z within 0.4m-4m
	else{
		rgbWorld.x = rgbWorld.y = rgbWorld.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}//set NaN
	return;
}//kernelTransformIR2RGB
void align_ir2rgb(const cv::cuda::GpuMat& cvgmIRWorld_, const float* aR_, const float* aRT_, cv::cuda::GpuMat* pcvgmRGBWorld_){
	cudaSafeCall( cudaMemcpyToSymbol(_aR,  aR_,  9*sizeof(float)) );
	cudaSafeCall( cudaMemcpyToSymbol(_aRT, aRT_, 3*sizeof(float)) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(pcvgmRGBWorld_->cols, block.x), cv::cuda::device::divUp(pcvgmRGBWorld_->rows, block.y));
	//run kernel
    kernelTransformIR2RGBCVmCVm<<<grid,block>>>( cvgmIRWorld_,*pcvgmRGBWorld_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}//cudaTransformIR2RGB
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//global constant used by kernelProjectRGB() and cudaProjectRGB()
__constant__ float _aRGBCameraParameter[4]; //fFxRGB_,fFyRGB_,uRGB_,vRGB_
__global__ void kernelProjectRGBCVmCVm(const cv::cuda::PtrStepSz<float3> cvgmRGBWorld_, cv::cuda::PtrStepSz<float> cvgmAligned_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	// cvgmAligned_ must be preset to zero;
	if (nX >= cvgmRGBWorld_.cols || nY >= cvgmRGBWorld_.rows) return;
	const float3& rgbWorld = cvgmRGBWorld_.ptr(nY)[nX];
	if( 0.4f < rgbWorld.z  &&  rgbWorld.z  < 10.f ){
		// get 2D image projection in RGB image of the XYZ in the world
		int nXAligned = __float2int_rn( _aRGBCameraParameter[0] * rgbWorld.x / rgbWorld.z + _aRGBCameraParameter[2] );
		int nYAligned = __float2int_rn( _aRGBCameraParameter[1] * rgbWorld.y / rgbWorld.z + _aRGBCameraParameter[3] );
		//if outside image return;
		if ( nXAligned < 0 || nXAligned >= cvgmRGBWorld_.cols || nYAligned < 0 || nYAligned >= cvgmRGBWorld_.rows )	return;
		
		float fPt = cvgmAligned_.ptr(nYAligned)[nXAligned];
		if(isnan<float>(fPt)){
			cvgmAligned_.ptr(nYAligned)[nXAligned] = rgbWorld.z;
		}//if havent been asigned
		else{
			fPt = (fPt+ rgbWorld.z)/2.f;
		}//if it does use the average 
	}//if within 0.4m-4m
	//else is not required
	//the cvgmAligned_ must be preset to NaN
	return;
}//kernelProjectRGB
void project_rgb(const cv::cuda::GpuMat& cvgmRGBWorld_, 
const float& fFxRGB_, const float& fFyRGB_, const float& uRGB_, const float& vRGB_, 
cv::cuda::GpuMat* pcvgmAligned_ ){
	pcvgmAligned_->setTo(std::numeric_limits<float>::quiet_NaN());
	//constant definition
	size_t sN = sizeof(float) * 4;
	float* const pRGBCameraParameters = (float*) malloc( sN );
	pRGBCameraParameters[0] = fFxRGB_;
	pRGBCameraParameters[1] = fFyRGB_;
	pRGBCameraParameters[2] = uRGB_;
	pRGBCameraParameters[3] = vRGB_;
	cudaSafeCall( cudaMemcpyToSymbol(_aRGBCameraParameter, pRGBCameraParameters, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmRGBWorld_.cols, block.x), cv::cuda::device::divUp(cvgmRGBWorld_.rows, block.y));
	//run kernel
    kernelProjectRGBCVmCVm<<<grid,block>>>( cvgmRGBWorld_,*pcvgmAligned_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());

	//release temporary pointers
	free(pRGBCameraParameters);
	return;
}//cudaProjectRGBCVCV()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//const float sigma_color = 30;     //in mm
//const float sigma_space = 4;     // in pixels
__constant__ float _aSigma2InvHalf[2]; //sigma_space2_inv_half,sigma_color2_inv_half

__global__ void kernelBilateral (const cv::cuda::PtrStepSz<float> src, cv::cuda::PtrStepSz<float> dst )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src.cols || y >= src.rows)  return;

    const int R = 2;//static_cast<int>(sigma_space * 1.5);
    const int D = R * 2 + 1;

    float fValueCentre = src.ptr (y)[x];
	//if fValueCentre is NaN
	if(fabs( fValueCentre ) < 0.00001f) return; 

    int tx = min (x - D/2 + D, src.cols - 1);
    int ty = min (y - D/2 + D, src.rows - 1);

    float sum1 = 0;
    float sum2 = 0;

    for (int cy = max (y - D/2, 0); cy < ty; ++cy)
    for (int cx = max (x - D/2, 0); cx < tx; ++cx){
        float  fValueNeighbour = src.ptr (cy)[cx];
		//if fValueNeighbour is NaN
		//if(fabs( fValueNeighbour - fValueCentre ) > 0.00005f) continue; 
        float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        float color2 = (fValueCentre - fValueNeighbour) * (fValueCentre - fValueNeighbour);
        float weight = __expf (-(space2 * _aSigma2InvHalf[0] + color2 * _aSigma2InvHalf[1]) );

        sum1 += fValueNeighbour * weight;
        sum2 += weight;
    }//for for each pixel in neigbbourhood

    dst.ptr (y)[x] = sum1/sum2;
	return;
}//kernelBilateral

void bilateral_filtering(const cv::cuda::GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, cv::cuda::GpuMat* pcvgmDst_ )
{
	pcvgmDst_->setTo(0);// (std::numeric_limits<float>::quiet_NaN());
	//constant definition
	size_t sN = sizeof(float) * 2;
	float* const pSigma = (float*) malloc( sN );
	pSigma[0] = 0.5f / (fSigmaSpace_ * fSigmaSpace_);
	pSigma[1] = 0.5f / (fSigmaColor_ * fSigmaColor_);
	cudaSafeCall( cudaMemcpyToSymbol(_aSigma2InvHalf, pSigma, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmSrc_.cols, block.x), cv::cuda::device::divUp(cvgmSrc_.rows, block.y));
	//run kernel
    kernelBilateral<<<grid,block>>>( cvgmSrc_,*pcvgmDst_ );
	//cudaSafeCall( cudaGetLastError () );
	//cudaSafeCall( cudaDeviceSynchronize() );

	//release temporary pointers
	free(pSigma);
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelPyrDown (const cv::cuda::PtrStepSz<float> cvgmSrc_, cv::cuda::PtrStepSz<float> cvgmDst_, float fSigmaColor_ )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cvgmDst_.cols || y >= cvgmDst_.rows) return;

    const int D = 5;

    float center = cvgmSrc_.ptr (2 * y)[2 * x];
	if( isnan<float>(center) ){//center!=center ){
		cvgmDst_.ptr (y)[x] = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//if center is NaN
    int tx = min (2 * x - D / 2 + D, cvgmSrc_.cols - 1); //ensure tx <= cvgmSrc.cols-1
    int ty = min (2 * y - D / 2 + D, cvgmSrc_.rows - 1); //ensure ty <= cvgmSrc.rows-1
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx) {
        float val = cvgmSrc_.ptr (cy)[cx];
        if (fabsf (val - center) < 3 * fSigmaColor_){//
			sum += val;
			++count;
        } //if within 3*fSigmaColor_
    }//for each pixel in the neighbourhood 5x5
    cvgmDst_.ptr (y)[x] = sum / count;
}//kernelPyrDown()

void pyr_down (const cv::cuda::GpuMat& cvgmSrc_, const float& fSigmaColor_, cv::cuda::GpuMat* pcvgmDst_)
{
	dim3 block (32, 8);
	dim3 grid (cv::cuda::device::divUp (pcvgmDst_->cols, block.x), cv::cuda::device::divUp (pcvgmDst_->rows, block.y));
	kernelPyrDown<<<grid, block>>>(cvgmSrc_, *pcvgmDst_, fSigmaColor_);
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelUnprojectRGBCVmCVm (const cv::cuda::PtrStepSz<float> cvgmDepths_, const unsigned short uScale_, cv::cuda::PtrStepSz<float3> cvgmPts_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows)  return;

	float3& pt = cvgmPts_.ptr(nY)[nX];
	const float fDepth = cvgmDepths_.ptr(nY)[nX];

	if( 0.4f < fDepth && fDepth < 10.f ){
		pt.z = fDepth;
		pt.x = ( nX*uScale_  - _aRGBCameraParameter[2] ) * _aRGBCameraParameter[0] * pt.z; //_aRGBCameraParameter[0] is 1.f/fFxRGB_
		pt.y = ( nY*uScale_  - _aRGBCameraParameter[3] ) * _aRGBCameraParameter[1] * pt.z; 
	}
	else {
		pt.x = pt.y = pt.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}
	return;
}
void unproject_rgb ( const cv::cuda::GpuMat& cvgmDepths_, 
						const float& fFxRGB_,const float& fFyRGB_,const float& uRGB_, const float& vRGB_, unsigned int uLevel_, 
						cv::cuda::GpuMat* pcvgmPts_ )
{
	unsigned short uScale = 1<< uLevel_;
	pcvgmPts_->setTo(0);
	//constant definition
	size_t sN = sizeof(float) * 4;
	float* const pRGBCameraParameters = (float*) malloc( sN );
	pRGBCameraParameters[0] = 1.f/fFxRGB_;
	pRGBCameraParameters[1] = 1.f/fFyRGB_;
	pRGBCameraParameters[2] = uRGB_;
	pRGBCameraParameters[3] = vRGB_;
	cudaSafeCall( cudaMemcpyToSymbol(_aRGBCameraParameter, pRGBCameraParameters, sN) );
	
	dim3 block (32, 8);
	dim3 grid (cv::cuda::device::divUp (pcvgmPts_->cols, block.x), cv::cuda::device::divUp (pcvgmPts_->rows, block.y));
	kernelUnprojectRGBCVmCVm<<<grid, block>>>(cvgmDepths_, uScale, *pcvgmPts_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelFastNormalEstimation (const cv::cuda::PtrStepSz<float3> cvgmPts_, cv::cuda::PtrStepSz<float3> cvgmNls_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows ) return;
	float3& fN = cvgmNls_.ptr(nY)[nX];
	if (nX == cvgmPts_.cols - 1 || nY >= cvgmPts_.rows - 1 ){
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}
	const float3& pt = cvgmPts_.ptr(nY)[nX];
	const float3& pt1= cvgmPts_.ptr(nY)[nX+1]; //right 
	const float3& pt2= cvgmPts_.ptr(nY+1)[nX]; //down

	if(isnan<float>(pt.z) ||isnan<float>(pt1.z) ||isnan<float>(pt2.z) ){
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//if input or its neighour is NaN,
	float3 v1;
	v1.x = pt1.x-pt.x;
	v1.y = pt1.y-pt.y;
	v1.z = pt1.z-pt.z;
	float3 v2;
	v2.x = pt2.x-pt.x;
	v2.y = pt2.y-pt.y;
	v2.z = pt2.z-pt.z;
	//n = v1 x v2 cross product
	float3 n;
	n.x = v1.y*v2.z - v1.z*v2.y;
	n.y = v1.z*v2.x - v1.x*v2.z;
	n.z = v1.x*v2.y - v1.y*v2.x;
	//normalization
	float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

	if( norm < 1.0e-10 ) {
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//set as NaN,
	n.x /= norm;
	n.y /= norm;
	n.z /= norm;

	if( -n.x*pt.x - n.y*pt.y - n.z*pt.z <0 ){ //this gives (0-pt).dot3( n ); 
		fN.x = -n.x;
		fN.y = -n.y;
		fN.z = -n.z;
	}//if facing away from the camera
	else{
		fN.x = n.x;
		fN.y = n.y;
		fN.z = n.z;
	}//else
	return;
}

void fast_normal_estimation(const cv::cuda::GpuMat& cvgmPts_, cv::cuda::GpuMat* pcvgmNls_ )
{
	pcvgmNls_->setTo(0);
	dim3 block (32, 8);
	dim3 grid (cv::cuda::device::divUp (cvgmPts_.cols, block.x), cv::cuda::device::divUp (cvgmPts_.rows, block.y));
	kernelFastNormalEstimation<<<grid, block>>>(cvgmPts_, *pcvgmNls_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelScaleDepthCVmCVm (cv::cuda::PtrStepSz<float> cvgmDepth_, const pcl::device::Intr sCameraIntrinsics_)
{
    int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;

    if (nX >= cvgmDepth_.cols || nY >= cvgmDepth_.rows)  return;

    float& fDepth = cvgmDepth_.ptr(nY)[nX];
    float fTanX = (nX - sCameraIntrinsics_.cx) / sCameraIntrinsics_.fx;
    float fTanY = (nY - sCameraIntrinsics_.cy) / sCameraIntrinsics_.fy;
    float fSec = sqrtf (fTanX*fTanX + fTanY*fTanY + 1);
    fDepth *= fSec; //meters
}//kernelScaleDepthCVmCVm()
//scaleDepth is to transform raw depth into scaled depth which is the distance from the 3D point to the camera centre
//     *---* 3D point
//     |  / 
//raw  | /scaled depth
//depth|/
//     * camera center
//
void scale_depth(unsigned short usPyrLevel_, const float fFx_, const float fFy_, const float u_, const float v_, cv::cuda::GpuMat* pcvgmDepth_){
	pcl::device::Intr sCameraIntrinsics(fFx_,fFy_,u_,v_);
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(pcvgmDepth_->cols, block.x), cv::cuda::device::divUp(pcvgmDepth_->rows, block.y));
	kernelScaleDepthCVmCVm<<< grid,block >>>(*pcvgmDepth_, sCameraIntrinsics(usPyrLevel_) );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ double _aRwTrans[9];//row major 
__constant__ double _aTw[3]; 
__global__ void kernelTransformLocalToWorldCVCV(cv::cuda::PtrStepSz<float3> cvgmPts_, cv::cuda::PtrStepSz<float3> cvgmNls_, cv::cuda::PtrStepSz<float3> cvgmMDs_){ 
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows)  return;
	//convert Pts
	float3& Pt = cvgmPts_.ptr(nY)[nX];
	float3 PtTmp; 
	//PtTmp = X_c - Tw
	PtTmp.x = Pt.x - _aTw[0];
	PtTmp.y = Pt.y - _aTw[1];
	PtTmp.z = Pt.z - _aTw[2];
	//Pt = RwTrans * PtTmp
	Pt.x = _aRwTrans[0]*PtTmp.x + _aRwTrans[1]*PtTmp.y + _aRwTrans[2]*PtTmp.z;
	Pt.y = _aRwTrans[3]*PtTmp.x + _aRwTrans[4]*PtTmp.y + _aRwTrans[5]*PtTmp.z;
	Pt.z = _aRwTrans[6]*PtTmp.x + _aRwTrans[7]*PtTmp.y + _aRwTrans[8]*PtTmp.z;
	{
		//convert Nls
		float3& Nl = cvgmNls_.ptr(nY)[nX];
		float3 NlTmp;
		//Nlw = RwTrans*Nlc
		NlTmp.x = _aRwTrans[0]*Nl.x + _aRwTrans[1]*Nl.y + _aRwTrans[2]*Nl.z;
		NlTmp.y = _aRwTrans[3]*Nl.x + _aRwTrans[4]*Nl.y + _aRwTrans[5]*Nl.z;
		NlTmp.z = _aRwTrans[6]*Nl.x + _aRwTrans[7]*Nl.y + _aRwTrans[8]*Nl.z;
		Nl = NlTmp;
	}

	if( cvgmMDs_.cols != 0 && cvgmMDs_.rows != 0 ){
		float3& MD = cvgmMDs_.ptr(nY)[nX];
		float3 MDTmp;
		//MDw = RwTrans*MDc
		MDTmp.x = _aRwTrans[0]*MD.x + _aRwTrans[1]*MD.y + _aRwTrans[2]*MD.z;
		MDTmp.y = _aRwTrans[3]*MD.x + _aRwTrans[4]*MD.y + _aRwTrans[5]*MD.z;
		MDTmp.z = _aRwTrans[6]*MD.x + _aRwTrans[7]*MD.y + _aRwTrans[8]*MD.z;
		MD = MDTmp;
	}
	return;
}//kernelTransformLocalToWorld()
void transform_local2world(const double* pRw_/*col major*/, const double* pTw_, cv::cuda::GpuMat* pcvgmPts_, cv::cuda::GpuMat* pcvgmNls_, cv::cuda::GpuMat* pcvgmMDs_){
	if ( pcvgmPts_->cols == 0 || pcvgmNls_->cols == 0 ) return;
	size_t sN1 = sizeof(double) * 9;
	cudaSafeCall( cudaMemcpyToSymbol(_aRwTrans, pRw_, sN1) );
	size_t sN2 = sizeof(double) * 3;
	cudaSafeCall( cudaMemcpyToSymbol(_aTw, pTw_, sN2) );
	dim3 block(32, 32);
    dim3 grid(cv::cuda::device::divUp(pcvgmPts_->cols, block.x), cv::cuda::device::divUp(pcvgmPts_->rows, block.y));
	kernelTransformLocalToWorldCVCV<<<grid,block>>>(*pcvgmPts_,*pcvgmNls_, *pcvgmMDs_);
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}//transformLocalToWorld()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool normalize>
__global__ void kernelResizeMap (const cv::cuda::PtrStepSz<float3> cvgmSrc_, cv::cuda::PtrStepSz<float3> cvgmDst_)
{
	using namespace pcl::device;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= cvgmDst_.cols || y >= cvgmDst_.rows) return;

    float3 qnan; qnan.x = qnan.y = qnan.z = pcl::device::numeric_limits<float>::quiet_NaN ();

    int xs = x * 2;
    int ys = y * 2;

    float3 x00 = cvgmSrc_.ptr (ys + 0)[xs + 0];
    float3 x01 = cvgmSrc_.ptr (ys + 0)[xs + 1];
    float3 x10 = cvgmSrc_.ptr (ys + 1)[xs + 0];
    float3 x11 = cvgmSrc_.ptr (ys + 1)[xs + 1];

    if (isnan (x00.x) || isnan (x01.x) || isnan (x10.x) || isnan (x11.x))
    {
		cvgmDst_.ptr (y)[x] = qnan;
		return;
    }
    else
    {
		float3 n;

		n = (x00 + x01 + x10 + x11) / 4;

		if (normalize)
			n = normalized<float, float3>(n);

		cvgmDst_.ptr (y)[x] = n;
    }
}//kernelResizeMap()

void resize_map (bool bNormalize_, const cv::cuda::GpuMat& cvgmSrc_, cv::cuda::GpuMat* pcvgmDst_ )
{
    int in_cols = cvgmSrc_.cols;
    int in_rows = cvgmSrc_.rows;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    pcvgmDst_->create (out_rows, out_cols,cvgmSrc_.type());

    dim3 block (32, 8);
    dim3 grid (cv::cuda::device::divUp (out_cols, block.x), cv::cuda::device::divUp (out_rows, block.y));
	if(bNormalize_)
		kernelResizeMap<true><<<grid, block>>>(cvgmSrc_, *pcvgmDst_);
	else
		kernelResizeMap<false><<<grid, block>>>(cvgmSrc_, *pcvgmDst_);
	//cudaSafeCall ( cudaGetLastError () );
    //cudaSafeCall (cudaDeviceSynchronize ());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_init_idx(PtrStepSz<float> gpu_idx_){
	//traverse each element of the matrix
	const int nC = blockDim.x * blockIdx.x + threadIdx.x;
	if (nC >= gpu_idx_.cols) return;
	gpu_idx_.ptr()[nC] = nC;
	return;
}

void init_idx(int nCols_, int type_, GpuMat* p_idx_){
	if (p_idx_->empty()) p_idx_->create(1, nCols_, type_);

	dim3 block(64, 1);
	dim3 grid(1, 1, 1);
	grid.x = cv::cuda::device::divUp(nCols_, block.x);
	kernel_init_idx <<< grid, block >>> (*p_idx_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}

void sort_column_idx(cv::cuda::GpuMat* const & p_key_, GpuMat* const & p_idx_){

	init_idx(p_key_->cols, p_key_->type(), p_idx_);

	thrust::device_ptr<float> X((float*)p_key_->data);
	thrust::device_ptr<float> V((float*)p_idx_->data);

	thrust::sort_by_key( X, X + p_key_->cols, V, thrust::greater<float>() );

	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_fill_error(const PtrStepSz<float> Error, int Idx, PtrStepSz<float> Error_field_){
	const int nX = blockDim.x * blockIdx.x + threadIdx.x;
	const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	if (nX < 0 || nX >= Error.cols || nY < 0 || nY >= Error.rows) return;
	float* pos = Error_field_.ptr(nY * Error.cols + nX);
	*(pos + Idx) = Error.ptr(nY)[nX];
	return;
}//kernel_fill_error

void cuda_fill_error(const GpuMat& Error, int Idx, GpuMat* pError_field_)
{
	assert(Idx < pError_field_->cols);
	assert(pError_field_->rows == Error.rows *Error.cols);

	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(Error.cols, block.x), cv::cuda::device::divUp(Error.rows, block.y));
	//run kernel
	kernel_fill_error << <grid, block >> >(Error, Idx, *pError_field_);
	cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void kernel_calc_deriv(PtrStepSz<T> vec_x_, PtrStepSz<T> vec_y_, PtrStepSz<T> dydx_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	if (nX >= vec_x_.cols) return;
	if (nX < 1){
		dydx_.ptr(0)[nX] = 1e-30;
		dydx_.ptr(1)[nX] = 1e-30;
		dydx_.ptr(2)[nX] = 1e-30;
	}
	else
	{
		float B;
		B = (vec_y_.ptr(0)[nX] - vec_y_.ptr(0)[nX - 1]) / (vec_x_.ptr(0)[nX] - vec_x_.ptr(0)[nX - 1]) / 255.f; dydx_.ptr(0)[nX] = B > 0 ? B : 1e-30;
		B = (vec_y_.ptr(1)[nX] - vec_y_.ptr(1)[nX - 1]) / (vec_x_.ptr(1)[nX] - vec_x_.ptr(1)[nX - 1]) / 255.f; dydx_.ptr(1)[nX] = B > 0 ? B : 1e-30;
		B = (vec_y_.ptr(2)[nX] - vec_y_.ptr(2)[nX - 1]) / (vec_x_.ptr(2)[nX] - vec_x_.ptr(2)[nX - 1]) / 255.f; dydx_.ptr(2)[nX] = B > 0 ? B : 1e-30;
	}
	return;
}

void calc_deriv(const GpuMat& rad_re_, const GpuMat& inten_, GpuMat& derivCRF_){

	derivCRF_ = inten_.clone();
	dim3 block(256, 1);
	dim3 grid(1, 1);
	kernel_calc_deriv<float> <<< grid, block >>>(rad_re_, inten_, derivCRF_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}

template <typename T>
__global__ void kernel_min_frame(PtrStepSz<T> rad_b_, PtrStepSz<T> rad_g_, PtrStepSz<T> rad_r_, PtrStepSz<T> rad_, 
	                             PtrStepSz<uchar> err_b_, PtrStepSz<uchar> err_g_, PtrStepSz<uchar> err_r_, PtrStepSz<uchar> err_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= rad_b_.cols || nY < 0 || nY >= rad_b_.rows) return;
	uchar eb = err_b_.ptr(nY)[nX];
	uchar eg = err_g_.ptr(nY)[nX];
	uchar er = err_r_.ptr(nY)[nX];
	if (eb <= eg && eb <= er){
		err_.ptr(nY)[nX] = eb;
	}
	else if (eg <= eb && eg <= er){
		err_.ptr(nY)[nX] = eg;
	}
	else if (er <= eg && er <= eg){
		err_.ptr(nY)[nX] = er;
	}
	rad_.ptr(nY)[nX] = (rad_b_.ptr(nY)[nX] * eb + rad_g_.ptr(nY)[nX] * eg + rad_r_.ptr(nY)[nX] * er) / (eb + eg + er);

	return;
}

void calc_avg_min_frame(const GpuMat& rad_b_, const GpuMat& rad_g_, const GpuMat& rad_r_, GpuMat* p_rad_,
					const GpuMat& err_b_, const GpuMat& err_g_, const GpuMat& err_r_, GpuMat* p_err_ ){
	if (p_rad_->empty()||p_err_->empty())
	{
		p_rad_->create(rad_b_.size(), CV_32FC1);
		p_err_->create(rad_b_.size(), CV_8UC1);
	}
	p_rad_->setTo(std::numeric_limits<float>::quiet_NaN());
	p_err_->setTo(0);
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(rad_b_.cols, block.x), cv::cuda::device::divUp(rad_b_.rows, block.y));

	kernel_min_frame<float> <<< grid, block >>> (rad_b_, rad_g_, rad_r_, *p_rad_, err_b_, err_g_, err_r_, *p_err_ );
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}

template <typename T>
__global__ void kernel_avg_frame(PtrStepSz<T> rad_b_, PtrStepSz<T> rad_g_, PtrStepSz<T> rad_r_, PtrStepSz<T> rad_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= rad_b_.cols || nY < 0 || nY >= rad_b_.rows) return;
	rad_.ptr(nY)[nX] = (rad_b_.ptr(nY)[nX] + rad_g_.ptr(nY)[nX] + rad_r_.ptr(nY)[nX]) / 3.f;

	return;
}

void calc_avg_frame(const GpuMat& rad_b_, const GpuMat& rad_g_, const GpuMat& rad_r_, GpuMat* p_rad_)
{
	if (p_rad_->empty())
	{
		p_rad_->create(rad_b_.size(), CV_32FC1);
	}
	p_rad_->setTo(std::numeric_limits<float>::quiet_NaN());
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(rad_b_.cols, block.x), cv::cuda::device::divUp(rad_b_.rows, block.y));

	kernel_avg_frame<float> <<< grid, block >>> (rad_b_, rad_g_, rad_r_, *p_rad_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}
}//device
}//btl
