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

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/core/cuda/utility.hpp>
#include "OtherUtil.hpp"
#include <math_constants.h>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "pcl/block.hpp"
#include <vector>
#include "DirectOrientation.cuh"


namespace btl{
	namespace device {
		typedef double float_type;
		using namespace pcl::device;
		using namespace cv::cuda;
		using namespace std;

		__constant__ Intr __sCamIntr;

		__constant__ Matd33  __R_rl_Kinv;
		__constant__ Matd33  __KR_rl;
		__constant__ Matd33  __H_rl; 
		__constant__ double3 __T_rl;
		__constant__ float __fDistThres;
		__constant__ float _fSinAngleThres;
		__constant__ float __fCosAngleThres;

		__constant__ int __nCols;
		__constant__ int __nRows;

		template< typename T3 >
		inline __host__ __device__ T3 _lerp(T3 s, T3 e, float t)
		{
			return s + (e - s)*t;
		}
		template< typename T3 >
		inline __host__ __device__ T3 _blerp(T3 c00, T3 c10, T3 c01, T3 c11, float tx, float ty)
		{
			return _lerp<T3>(_lerp<T3>(c00, c10, tx), _lerp<T3>(c01, c11, tx), ty);
		}

		struct SDeviceDirectOrientation
		{
			enum {
				CTA_SIZE_X = 32,
				CTA_SIZE_Y = 32,
				CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
			};

			enum TYPE{
				_TRUE = 1,
				_RGB_FALSE = 0,
				_V_FALSE = -1
			};

			PtrStepSz<float> _NILive;
			PtrStepSz<uchar> _ErrLive;
			PtrStepSz<float> _NIRef;
			PtrStepSz<uchar> _mask;

			mutable PtrStepSz<float_type> _cvgmBuf;
			mutable PtrStepSz<float_type> _cvgmE;
			float _weight_rgb;
			float _weight_v;
			// nX_, nY_, are the current frame pixel index
			//X_p_ : vertex in previous camera coordinate system
			//N_p_ : normal in previous camera coordinate system
			//Xc_p_ : vertex in current frame and map to previous camera coordinate system
			template <typename T3>
			inline __host__ __device__ T3 bilinearInterpolateValue(
				const PtrStepSz<T3>& array,
				float x_ind,
				float y_ind)
			{
				const int ix = floor(x_ind);
				const int iy = floor(y_ind);
				const float fx = x_ind - ix;
				const float fy = y_ind - iy;

				if (ix >= 0 && ix < __nCols - 1 &&
					iy >= 0 && iy < __nRows - 1)
				{
					///            uint32_t c00 = getpixel(src, gxi, gyi);
					const T3 c00 = array.ptr(iy)[ix + 0];

					///            uint32_t c10 = getpixel(src, gxi+1, gyi);
					const T3 c10 = array.ptr(iy)[ix + 1];

					///            uint32_t c01 = getpixel(src, gxi, gyi+1);
					const T3 c01 = array.ptr(iy + 1)[ix + 0];

					///            uint32_t c11 = getpixel(src, gxi+1, gyi+1);
					const T3 c11 = array.ptr(iy + 1)[ix + 1];

					return _blerp<T3>(c00, c10, c01, c11, fx, fy);
				}
				else
					return 0;
			}

	__device__ __forceinline__ TYPE searchForCorrespondence(int nX_, int nY_, float3& Xc_r_, float& dI_, float3* pgpK_) {
		//retrieve normal
		float3 f3Ref = __H_rl * make_float3(nX_, nY_, 1);
		f3Ref.x /= f3Ref.z;
		f3Ref.y /= f3Ref.z;
		int2 n2Ref;
		n2Ref.x = __float2int_rd(f3Ref.x + .5f);
		n2Ref.y = __float2int_rd(f3Ref.y + .5f);
		//if projected out of the frame, return false
		if (n2Ref.x <= 0 || n2Ref.y <= 0 || n2Ref.x >= __nCols - 1 || n2Ref.y >= __nRows - 1 || Xc_r_.z < 0) return _V_FALSE;

		float I_l = _NILive.ptr(nY_)[nX_]; if (I_l != I_l) return _RGB_FALSE;
		float I_r = bilinearInterpolateValue<float>(_NIRef, f3Ref.x, f3Ref.y); if (I_r != I_r) return _RGB_FALSE;
		dI_ = I_r - I_l;
		//printf("d %23.16e/t r %f \t l %f\n", dI_, I_r, I_l);
		if (fabs(dI_) > 4) return _RGB_FALSE;

		if (pgpK_){
			Xc_r_ = __R_rl_Kinv * make_float3(nX_, nY_, 1);
			//if (dI_ > 1.f) return _RGB_FALSE;
			float dx = (2 * bilinearInterpolateValue(_NIRef, f3Ref.x - 1.f, f3Ref.y) + bilinearInterpolateValue(_NIRef, f3Ref.x - 1.f, f3Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f3Ref.x - 1.f, f3Ref.y + 1.f)
				- 2 * bilinearInterpolateValue(_NIRef, f3Ref.x + 1.f, f3Ref.y) - bilinearInterpolateValue(_NIRef, f3Ref.x + 1.f, f3Ref.y - 1.f) - bilinearInterpolateValue(_NIRef, f3Ref.x + 1.f, f3Ref.y + 1.f));
			dx /= 8.f;  if (dx != dx) return _RGB_FALSE;
			float dy = (2 * bilinearInterpolateValue(_NIRef, f3Ref.x, f3Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f3Ref.x - 1.f, f3Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f3Ref.x + 1.f, f3Ref.y - 1.f)
				- 2 * bilinearInterpolateValue(_NIRef, f3Ref.x, f3Ref.y + 1.f) - bilinearInterpolateValue(_NIRef, f3Ref.x - 1.f, f3Ref.y + 1.f) - bilinearInterpolateValue(_NIRef, f3Ref.x + 1.f, f3Ref.y + 1.f));
			dy /= 8.f;  if (dy != dy) return _RGB_FALSE;// *2.5f;
			//printf("dx %e dy %e dI %f\n", dx, dy, dI_);
			if (fabs(dx) > 4.25f || fabs(dy) > 4.25f) return _RGB_FALSE;
			float A = dx * __sCamIntr.fx / Xc_r_.z;
			float B = dy * __sCamIntr.fy / Xc_r_.z;
			pgpK_->x = A;// *__sCamIntr.fx;
			pgpK_->y = B;// *__sCamIntr.fy;
			pgpK_->z = -A * Xc_r_.x / Xc_r_.z - B * Xc_r_.y / Xc_r_.z;
		}

		return _TRUE;
	}//searchForCorrespondence2() 

	__device__ __forceinline__ void calc_rotation_energy() {
		int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
		if (nX >= __nCols || nY >= __nRows) return;

		float3 Xc_r;
		float dI = 0.f;
		// read out current point in world and find its corresponding point in previous frame, which are also defined in world
		TYPE res = searchForCorrespondence(nX, nY, Xc_r, dI, NULL);
		if (_TRUE == res){
			//printf("2 nX %d nY %d PtCurr (%f %f %f)\n", nX, nY, f3PtCurr.x, f3PtCurr.y, f3PtCurr.z );
			_cvgmE.ptr(nY)[nX] = _weight_rgb*fabs(dI) *_ErrLive.ptr(nY)[nX];
			//printf("\t%f\t%f\n", fE, weight*weight*dI*dI);
			_mask.ptr(nY)[nX] = uchar(1);
		}//if correspondence found

		//_cvgmE2.ptr(nY)[nX] = weight*fabs(dI);
		return;
		return;
	}

	//32*8 threads and cols/32 * rows/8 blocks
	__device__ __forceinline__ void direct_rotation() {
		int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
		if (nX >= __nCols || nY >= __nRows) return;

		float jacob[4]; // it has to be there for all threads, otherwise, some thread will add un-initialized fE into total energy. 
		float3 Xc_r;
		float3 gpK;
		float dI = 0.f;
		// read out current point in world and find its corresponding point in previous frame, which are also defined in world
		TYPE res = searchForCorrespondence(nX, nY, Xc_r, dI, &gpK);
		if (_TRUE == res){
			float w_rgb = _weight_rgb *_ErrLive.ptr(nY)[nX];
			jacob[3] = w_rgb*dI;//order matters; energy
			jacob[0] = w_rgb*(Xc_r.y * gpK.z - Xc_r.z * gpK.y);//cross(Xw_c, Nw_p); 
			jacob[1] = w_rgb*(Xc_r.z * gpK.x - Xc_r.x * gpK.z);
			jacob[2] = w_rgb*(Xc_r.x * gpK.y - Xc_r.y * gpK.x);
		}//if correspondence found
		else{
			memset(jacob, 0, 16);//16= 4*bits(float)
		}

		int nn = nX + nY * __nCols;

		int nShift = 0;
		//int nShift2 = 0;
		for (int i = 0; i < 3; ++i){ //__nRows
#pragma unroll
			for (int j = i; j < 4; ++j){ // __nCols + b
				_cvgmBuf.ptr(nShift++)[nn] += jacob[i] * jacob[j];
				//nShift < 9 = 6 + 3, upper triangle of 3x3
			}//for
		}//for

		//printf("x %d, y %d, %f %f %f , %f %f %f , %f %f %f \n", x_l, y_l, _cvgmBuf.ptr(0)[nn], _cvgmBuf.ptr(1)[nn], _cvgmBuf.ptr(2)[nn], _cvgmBuf.ptr(3)[nn], _cvgmBuf.ptr(4)[nn], _cvgmBuf.ptr(5)[nn], _cvgmBuf.ptr(6)[nn], _cvgmBuf.ptr(7)[nn], _cvgmBuf.ptr(8)[nn]);
		return;
	}//operator()*/
};//SDeviceICPEnergyRegistration

__global__ void kernel_direct_rotation(SDeviceDirectOrientation sDO) {
	sDO.direct_rotation();
}

__global__ void kernel_energy_rotation(SDeviceDirectOrientation sDO) {
	sDO.calc_rotation_energy();
}

GpuMat direct_rotation (const Intr& intr_,
						const Matd33& R_rl_Kinv_,
						const Matd33& H_rl_,
						const GpuMat& NormalizedRadianceRef_,
						const GpuMat& NormalizedRadianceLiv_, const GpuMat& err_){

	cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &intr_, sizeof(Intr)));
	cudaSafeCall(cudaMemcpyToSymbol(__R_rl_Kinv, &R_rl_Kinv_, sizeof(Matd33)));
	cudaSafeCall(cudaMemcpyToSymbol(__H_rl, &H_rl_, sizeof(Matd33)));
	cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(NormalizedRadianceRef_.cols), sizeof(int))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(NormalizedRadianceRef_.rows), sizeof(int))); //copy host memory to constant memory on the device.
			
	GpuMat mask;
	mask.create(NormalizedRadianceRef_.size(), CV_8UC1);
	mask.setTo(0);

	SDeviceDirectOrientation directOrientation;

	directOrientation._NILive = NormalizedRadianceLiv_;
	directOrientation._ErrLive = err_;
	directOrientation._NIRef = NormalizedRadianceRef_;

	directOrientation._mask = mask;
	directOrientation._weight_rgb = .01f;

	dim3 block(SDeviceDirectOrientation::CTA_SIZE_X, SDeviceDirectOrientation::CTA_SIZE_Y);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(NormalizedRadianceRef_.cols, block.x);
	grid.y = cv::cudev::divUp(NormalizedRadianceRef_.rows, block.y);
	GpuMat cvgmBuf(9, NormalizedRadianceRef_.cols*NormalizedRadianceRef_.rows, CV_64FC1); cvgmBuf.setTo(0.);
	//the # of rows is 9, which is calculated in this way:
	// | 0  1  2  3  |
	// |    4  5  6  |
	// |       7  8  |
	// J'*J | J'*dI

	directOrientation._cvgmBuf = cvgmBuf;
	kernel_direct_rotation<<< grid, block >>>(directOrientation);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall ( cudaDeviceSynchronize() );
	GpuMat SumBuf;
	//cout << cv::Mat(cvgmBuf) << endl;
	cv::cuda::reduce(cvgmBuf, SumBuf, 1, CV_REDUCE_SUM, CV_64FC1); //1x100 where 100 is the number of ransac iterations
	//cout << cv::Mat(SumBuf) << endl;
	return SumBuf;
}


double energy_direct_radiance_rotation(const Intr& intr_,
										const Matd33& R_rl_Kinv_,
										const Matd33& H_rl_,
										const GpuMat& NormalizedRadianceRef_,
										const GpuMat& NormalizedRadianceLiv_, const GpuMat& err_){

	cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &intr_, sizeof(Intr)));
	cudaSafeCall(cudaMemcpyToSymbol(__R_rl_Kinv, &R_rl_Kinv_, sizeof(Matd33)));
	cudaSafeCall(cudaMemcpyToSymbol(__H_rl, &H_rl_, sizeof(Matd33)));
	cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(NormalizedRadianceRef_.cols), sizeof(int))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(NormalizedRadianceRef_.rows), sizeof(int))); //copy host memory to constant memory on the device.

	GpuMat mask;
	mask.create(NormalizedRadianceRef_.size(), CV_8UC1);
	mask.setTo(0);

	SDeviceDirectOrientation directOrientation;

	directOrientation._NILive = NormalizedRadianceLiv_;
	directOrientation._ErrLive = err_;
	directOrientation._NIRef = NormalizedRadianceRef_;

	directOrientation._mask = mask;
	directOrientation._weight_rgb = .01f;

	dim3 block(SDeviceDirectOrientation::CTA_SIZE_X, SDeviceDirectOrientation::CTA_SIZE_Y);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(NormalizedRadianceRef_.cols, block.x);
	grid.y = cv::cudev::divUp(NormalizedRadianceRef_.rows, block.y);

	GpuMat cvgmE(NormalizedRadianceRef_.size(), CV_64FC1); cvgmE.setTo(0.);
	directOrientation._cvgmE = cvgmE;

	kernel_energy_rotation <<< grid, block >>>(directOrientation);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	double dEnergy = sum(cvgmE)[0];
	int nPairs = sum(mask)[0];
	int total = NormalizedRadianceRef_.cols *NormalizedRadianceRef_.rows;
	//The following equations are come from equation (10) in
	//Chetverikov, D., Stepanov, D., & Krsek, P. (2005). 
	//Robust Euclidean alignment of 3D point sets: the trimmed iterative closest point algorithm. 
	//IVC, 23(3), 299¨C309. doi:10.1016/j.imavis.2004.05.007
	//dEnergy /= nPairs; 
	//float xee = float(nPairs) / float(total);
	//dEnergy /= (xee*xee*xee); 
	//cout << "energy = " << dEnergy << "\t" << nPairs << "\t" << total << "\t" << VMapLive_.cols << endl;
	return dEnergy;
}//registration()

		
}//device
}//btl