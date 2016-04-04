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
#include "DVOICP.cuh"



namespace btl{
	namespace device {
		typedef double float_type;
		using namespace pcl::device;
		using namespace cv::cuda;
		using namespace std;

		__constant__ Intr __sCamIntr;

		__constant__ Matd33  __R_rl;
		__constant__ double3 __T_rl;
		__constant__ float __fDistThres;
		__constant__ float _fSinAngleThres;
		__constant__ float __fCosAngleThres;

		__constant__ int __nCols;
		__constant__ int __nRows;

		inline __host__ __device__ float _lerp(float s, float e, float t)
		{
			return s + (e - s)*t;
		}

		inline __host__ __device__ float _blerp(float c00, float c10, float c01, float c11, float tx, float ty)
		{
			return _lerp(_lerp(c00, c10, tx), _lerp(c01, c11, tx), ty);
		}



		struct SDeviceDVOICP
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

			PtrStepSz<float> _DepthLive;

			PtrStepSz<float3> _VMapLive;
			PtrStepSz<float3> _NMapLive;
			PtrStepSz<float> _NILive;
			PtrStepSz<uchar> _ErrLive;
			PtrStepSz<uchar> _mask;
			PtrStepSz<uchar> _mask2;

			PtrStepSz<float3> _VMapRef;
			PtrStepSz<float3> _NMapRef;
			PtrStepSz<float> _NIRef;

			mutable PtrStepSz<float_type> _cvgmBuf;
			mutable PtrStepSz<float_type> _cvgmBuf2;
			mutable PtrStepSz<float_type> _cvgmE;
			mutable PtrStepSz<float_type> _cvgmE2;
			float _weight_rgb;
			float _weight_v;
			// nX_, nY_, are the current frame pixel index
			//X_p_ : vertex in previous camera coordinate system
			//N_p_ : normal in previous camera coordinate system
			//Xc_p_ : vertex in current frame and map to previous camera coordinate system

			inline __host__ __device__ float bilinearInterpolateValue(
				const PtrStepSz<float>& array,
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
					const float c00 = array.ptr(iy)[ix + 0];

					///            uint32_t c10 = getpixel(src, gxi+1, gyi);
					const float c10 = array.ptr(iy)[ix + 1];

					///            uint32_t c01 = getpixel(src, gxi, gyi+1);
					const float c01 = array.ptr(iy + 1)[ix + 0];

					///            uint32_t c11 = getpixel(src, gxi+1, gyi+1);
					const float c11 = array.ptr(iy + 1)[ix + 1];

					return _blerp(c00, c10, c01, c11, fx, fy);
				}
				else
					return NAN;
			}

			__device__ __forceinline__ TYPE searchForCorrespondence(int nX_, int nY_, float3& Xc_r_, float& dI_, float3* pgpK_) {
				//retrieve normal
				float3 X_l = _VMapLive.ptr(nY_)[nX_]; if (isnan(X_l.x) || isnan(X_l.y) || isnan(X_l.z)) return _V_FALSE; //1.retrieve vertex from current frame
				Xc_r_ = __R_rl * X_l + __T_rl; //2.transform vertex in live frame to reference frame
				//projection onto reference image
				float2 f2Ref = make_float2(Xc_r_.x * __sCamIntr.fx / Xc_r_.z + __sCamIntr.cx, Xc_r_.y * __sCamIntr.fy / Xc_r_.z + __sCamIntr.cy);
				int2 n2Ref;
				n2Ref.x = __float2int_rd(f2Ref.x + .5f);
				n2Ref.y = __float2int_rd(f2Ref.y + .5f);
				//if projected out of the frame, return false
				if (n2Ref.x <= 0 || n2Ref.y <= 0 || n2Ref.x >= __nCols - 1 || n2Ref.y >= __nRows - 1 || Xc_r_.z < 0) return _V_FALSE;

				float I_l = _NILive.ptr(nY_)[nX_]; if (I_l != I_l) return _RGB_FALSE;
				float I_r = bilinearInterpolateValue(_NIRef, f2Ref.x, f2Ref.y); if (I_r != I_r) return _RGB_FALSE;
				dI_ = I_r - I_l;
				if (fabs(dI_) > 4) return _RGB_FALSE;

				if (pgpK_){
					//if (dI_ > 1.f) return _RGB_FALSE;
					float dx = (2 * bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y) + bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y + 1.f)
						- 2 * bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y) - bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y - 1.f) - bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y + 1.f));
					dx /= 8.f;  if (dx != dx) return _RGB_FALSE;
					float dy = (2 * bilinearInterpolateValue(_NIRef, f2Ref.x, f2Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y - 1.f)
						- 2 * bilinearInterpolateValue(_NIRef, f2Ref.x, f2Ref.y + 1.f) - bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y + 1.f) - bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y + 1.f));
					dy /= 8.f;  if (dy != dy) return _RGB_FALSE;// *2.5f;
					//printf("dx %e dy %e dI %f\n", dx, dy, dI_);
					if (fabs(dx) > .25f || fabs(dy) > .25f) return _RGB_FALSE;
					float A = dx * __sCamIntr.fx / Xc_r_.z;
					float B = dy * __sCamIntr.fy / Xc_r_.z;
					pgpK_->x = A;// *__sCamIntr.fx;
					pgpK_->y = B;// *__sCamIntr.fy;
					pgpK_->z = -A * Xc_r_.x / Xc_r_.z - B * Xc_r_.y / Xc_r_.z;
				}

				return _TRUE;
			}//searchForCorrespondence2() 

			__device__ __forceinline__ TYPE searchForCorrespondence(int nX_, int nY_, float3& N_r_, float3& X_r_, float3& Xc_r_, float& dI_, float& weight_, float3* pgpK_) {
				//retrieve normal
				const float3 N_l = _NMapLive.ptr(nY_)[nX_]; if (isnan(N_l.x) || isnan(N_l.y) || isnan(N_l.z)) return _V_FALSE;
				float3 X_l = _VMapLive.ptr(nY_)[nX_]; if (isnan(X_l.x) || isnan(X_l.y) || isnan(X_l.z)) return _V_FALSE; //retrieve vertex from current frame
				Xc_r_ = __R_rl * X_l + __T_rl; //transform vertex in current frame to previous frame
				//projection onto reference image
				float2 f2Ref = make_float2(Xc_r_.x * __sCamIntr.fx / Xc_r_.z + __sCamIntr.cx, Xc_r_.y * __sCamIntr.fy / Xc_r_.z + __sCamIntr.cy);
				if (_DepthLive.rows != 0){
					weight_ /= _DepthLive.ptr(nY_)[nX_];// (_depth_curr.ptr(nY_)[nX_] * _depth_curr.ptr(nY_)[nX_]);
				}

				int2 n2Ref;
				n2Ref.x = __float2int_rd(f2Ref.x + .5f);
				n2Ref.y = __float2int_rd(f2Ref.y + .5f);
				//if projected out of the frame, return false
				if (n2Ref.x <= 0 || n2Ref.y <= 0 || n2Ref.x >= __nCols - 1 || n2Ref.y >= __nRows - 1 || Xc_r_.z < 0) return _V_FALSE;

				//retrieve corresponding reference normal
				N_r_ = _NMapRef.ptr(n2Ref.y)[n2Ref.x];  if (isnan(N_r_.x) || isnan(N_r_.y) || isnan(N_r_.z))  return _V_FALSE;
				//printf("%d\t%d (%f %f %f)\n", nX_, nY_, N_r_.x, N_r_.y, N_r_.z);
				//retrieve corresponding reference vertex
				X_r_ = _VMapRef.ptr(n2Ref.y)[n2Ref.x];  if (isnan(X_r_.x) || isnan(X_r_.y) || isnan(X_r_.x))  return _V_FALSE;
				//printf("%d\t%d X_r_ (%f %f %f) (%f %f %f)\n", nX_, nY_, X_r_.x, X_r_.y, X_r_.z, Xc_r_.x, Xc_r_.y, Xc_r_.z);
				//check distance
				float fDist = norm<float, float3>(X_r_ - Xc_r_); if (fDist != fDist) return _V_FALSE;

				_mask2.ptr(nY_)[nX_] = 1;
				//printf("%d\t%d (%f %f)\n", nX_, nY_, fDist, __fDistThres);
				if (fDist > __fDistThres + 0.027*_DepthLive.ptr(nY_)[nX_])  return _V_FALSE;
				//transform current normal to previous
				float3 Nc_p = __R_rl * N_l;
				//check normal angle
				float fCos = dot3<float, float3>(Nc_p, N_r_);
				if (fCos < __fCosAngleThres) return _V_FALSE;

				float I_r = bilinearInterpolateValue(_NIRef, f2Ref.x, f2Ref.y); if (I_r != I_r) return _RGB_FALSE;
				float I_l = _NILive.ptr(nY_)[nX_]; if (I_l != I_l) return _RGB_FALSE;
				//printf("d %23.16e/t r %f \t l %f\n", dI_, I_r, I_l);
				dI_ = I_r - I_l;
				if (fabs(dI_) > 4 || fabs(dI_) < 1e-1) return _RGB_FALSE;

				if (pgpK_){
					//if (dI_ > 1.f) return _RGB_FALSE;
					float dx = (2 * bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y) + bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y + 1.f)
						- 2 * bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y) - bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y - 1.f) - bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y + 1.f));
					dx /= 8.f;  if (dx != dx) return _RGB_FALSE;
					float dy = (2 * bilinearInterpolateValue(_NIRef, f2Ref.x, f2Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y - 1.f) + bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y - 1.f)
						- 2 * bilinearInterpolateValue(_NIRef, f2Ref.x, f2Ref.y + 1.f) - bilinearInterpolateValue(_NIRef, f2Ref.x - 1.f, f2Ref.y + 1.f) - bilinearInterpolateValue(_NIRef, f2Ref.x + 1.f, f2Ref.y + 1.f));
					dy /= 8.f;  if (dy != dy) return _RGB_FALSE;// *2.5f;
					//printf("dx %e dy %e dI %f\n", dx, dy, dI_);
					if (fabs(dx) > .25f || fabs(dy) > .25f || fabs(dx) < .01f || fabs(dy) < .01f) return _RGB_FALSE;
					float A = dx * __sCamIntr.fx / Xc_r_.z;
					float B = dy * __sCamIntr.fy / Xc_r_.z;
					pgpK_->x = A;// *__sCamIntr.fx;
					pgpK_->y = B;// *__sCamIntr.fy;
					pgpK_->z = -A * Xc_r_.x / Xc_r_.z - B * Xc_r_.y / Xc_r_.z;
				}

				return _TRUE;
			}//searchForCorrespondence2() 


			__device__ __forceinline__ void calc_energy() {
				int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
				int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
				if (nX >= __nCols || nY >= __nRows) return;

				float3 N_r, X_r, Xc_r;
				float dI = 0.f;
				// read out current point in world and find its corresponding point in previous frame, which are also defined in world
				float fE = 0.f;
				float weight_icp = 1.f;
				TYPE res = searchForCorrespondence(nX, nY, N_r, X_r, Xc_r, dI, weight_icp, NULL);
				if (_TRUE == res){
					//printf("2 nX %d nY %d PtCurr (%f %f %f)\n", nX, nY, f3PtCurr.x, f3PtCurr.y, f3PtCurr.z );
					fE = _weight_v*weight_icp*fabs(dot3<float, float3>(N_r, X_r - Xc_r)) + _weight_rgb* fabs(dI)*_ErrLive.ptr(nY)[nX]; //
					//printf("\t%f\t%f\n", fE, weight*weight*dI*dI);
					_mask.ptr(nY)[nX] = uchar(1);
				}//if correspondence found
				else if (_RGB_FALSE == res){
					fE = _weight_v*weight_icp*fabs(dot3<float, float3>(N_r, X_r - Xc_r));
					//printf("\t%f\t%f\n", fE, weight*weight*dI*dI);
					_mask.ptr(nY)[nX] = uchar(1);
				}

				_cvgmE.ptr(nY)[nX] = fE;
				//_cvgmE2.ptr(nY)[nX] = weight*fabs(dI);
				return;
			}

			//32*8 threads and cols/32 * rows/8 blocks
			__device__ __forceinline__ void align_frm() {
				int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
				int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
				if (nX >= __nCols || nY >= __nRows) return;

				float row_0[7]; // it has to be there for all threads, otherwise, some thread will add un-initialized fE into total energy. 
				float row_1[7]; // it has to be there for all threads, otherwise, some thread will add un-initialized fE into total energy. 
				//X_p : vertex in previous camera coordinate system
				float3& X_r = *(float3*)row_0;
				//N_p : normal in previous camera coordinate system
				float3& N_r = *(float3*)(row_0 + 3);
				//Xc_p : vertex in current frame and map to previous camera coordinate system
				float3 Xc_r;
				float3& gpK = *(float3*)(row_1 + 3);
				float dI = 0.f;
				float weight_icp = 1.f;
				// read out current point in world and find its corresponding point in previous frame, which are also defined in world
				TYPE res = searchForCorrespondence(nX, nY, N_r, X_r, Xc_r, dI, weight_icp, &gpK);
				if (_TRUE == res){
					weight_icp *= _weight_v;
					row_0[6] = weight_icp*dot3<float, float3>(N_r, X_r - Xc_r);
					row_0[0] = weight_icp*(Xc_r.y * N_r.z - Xc_r.z * N_r.y);//cross(Xw_c, Nw_p); 
					row_0[1] = weight_icp*(Xc_r.z * N_r.x - Xc_r.x * N_r.z);
					row_0[2] = weight_icp*(Xc_r.x * N_r.y - Xc_r.y * N_r.x);
					row_0[3] *= weight_icp;
					row_0[4] *= weight_icp;
					row_0[5] *= weight_icp;

					float w_rgb = _weight_rgb*_ErrLive.ptr(nY)[nX];
					row_1[6] = w_rgb*dI;//order matters; energy
					row_1[0] = w_rgb*(Xc_r.y * gpK.z - Xc_r.z * gpK.y);//cross(Xw_c, Nw_p); 
					row_1[1] = w_rgb*(Xc_r.z * gpK.x - Xc_r.x * gpK.z);
					row_1[2] = w_rgb*(Xc_r.x * gpK.y - Xc_r.y * gpK.x);
					row_1[3] *= w_rgb; //gpK.x*weight
					row_1[4] *= w_rgb; //gpK.y*weight
					row_1[5] *= w_rgb; //gpK.z*weight
					_mask.ptr(nY)[nX] = uchar(1);
				}//if correspondence found
				else if (_RGB_FALSE == res){
					weight_icp *= _weight_v;
					row_0[6] = weight_icp*dot3<float, float3>(N_r, X_r - Xc_r);
					row_0[0] = weight_icp*(Xc_r.y * N_r.z - Xc_r.z * N_r.y);//cross(Xw_c, Nw_p); 
					row_0[1] = weight_icp*(Xc_r.z * N_r.x - Xc_r.x * N_r.z);
					row_0[2] = weight_icp*(Xc_r.x * N_r.y - Xc_r.y * N_r.x);
					row_0[3] *= weight_icp;
					row_0[4] *= weight_icp;
					row_0[5] *= weight_icp;
					_mask2.ptr(nY)[nX] = uchar(1);
					memset(row_1, 0, 28);//28= 7*bits(float)
				}
				else{
					memset(row_0, 0, 28);//28= 7*bits(float)
					memset(row_1, 0, 28);//28= 7*bits(float)
				}

				int nShift = 0;
				//int nShift2 = 0;
				for (int i = 0; i < 6; ++i){ //__nRows
#pragma unroll
					for (int j = i; j < 7; ++j){ // __nCols + b
						_cvgmBuf.ptr(nShift++)[nX + nY * __nCols] += (row_0[i] * row_0[j] + row_1[i] * row_1[j]);
						//nShift < 27 = 21 + 6, upper triangle of 6x6
					}//for
				}//for
				return;
			}//operator()
		};//SDeviceICPEnergyRegistration

		__global__ void kernel_dvo_icp(SDeviceDVOICP sICP) {
			sICP.align_frm();
		}

		__global__ void kernel_dvo_icp_energy(SDeviceDVOICP sICP) {
			sICP.calc_energy();
		}

		GpuMat dvo_icp(const Intr& sCamIntr_,
			const Matd33& R_rl_, const double3& T_rl_,
			const GpuMat& VMapRef_, const GpuMat& NMapRef_, const GpuMat& NIRef_,
			const GpuMat& VMapLive_, const GpuMat& NMapLive_, const GpuMat& NILive_, const GpuMat& DepthLive_, const GpuMat& ErrLive_, GpuMat& mask_){
			float fDistThres_ = 0.25f; //meters works for the desktop non-stationary situation.
			float fCosAngleThres_ = 0.5f; //cos(M_PI/3)

			cudaSafeCall(cudaMemcpyToSymbol(__fDistThres, &fDistThres_, sizeof(float))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__fCosAngleThres, &fCosAngleThres_, sizeof(float))); //copy host memory to constant memory on the device.

			cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &sCamIntr_, sizeof(Intr))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__R_rl, &R_rl_, sizeof(Matd33)));
			cudaSafeCall(cudaMemcpyToSymbol(__T_rl, &T_rl_, sizeof(double3))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(VMapLive_.cols), sizeof(int))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(VMapLive_.rows), sizeof(int))); //copy host memory to constant memory on the device.

			assert(mask_.cols == VMapLive_.cols && mask_.rows == VMapLive_.rows);
			mask_.setTo(uchar(0));
			GpuMat mask2 = mask_.clone();
			mask2.setTo(0);

			SDeviceDVOICP sICP;
			sICP._DepthLive = DepthLive_;
			sICP._VMapLive = VMapLive_;
			sICP._NMapLive = NMapLive_;
			sICP._NILive = NILive_;
			sICP._ErrLive = ErrLive_;
			sICP._VMapRef = VMapRef_;
			sICP._NMapRef = NMapRef_;
			sICP._NIRef = NIRef_;
			sICP._mask = mask_;
			sICP._mask2 = mask2;
			sICP._weight_rgb = 0.01f / 32;
			sICP._weight_v = 1.0f;

			dim3 block(SDeviceDVOICP::CTA_SIZE_X, SDeviceDVOICP::CTA_SIZE_Y);
			dim3 grid(1, 1, 1);
			grid.x = cv::cudev::divUp(VMapLive_.cols, block.x);
			grid.y = cv::cudev::divUp(VMapLive_.rows, block.y);

			GpuMat cvgmBuf(27, VMapLive_.cols*VMapLive_.rows, CV_64FC1); cvgmBuf.setTo(0.);
			//GpuMat cvgmBuf2(27, VMapLive_.cols*VMapLive_.rows, CV_64FC1); cvgmBuf2.setTo(0.);
			//the # of rows is 27, which is calculated in this way:
			// | 1  2  3  4  5  6  7 |
			// |    8  9 10 11 12 13 |
			// |      14 15 16 17 18 |
			// |         19 20 21 22 |
			// |            23 24 25 |
			// |               26 27 |
			//the # of cols is equal to the # of blocks.
			sICP._cvgmBuf = cvgmBuf;
			//sICP._cvgmBuf2 = cvgmBuf2;

			kernel_dvo_icp << < grid, block >> >(sICP);
			//cudaSafeCall(cudaGetLastError());
			//cudaSafeCall(cudaDeviceSynchronize());

			//vector<GpuMat> vResults;
			GpuMat SumBuf;
			cv::cuda::reduce(cvgmBuf, SumBuf, 1, CV_REDUCE_SUM, CV_64FC1); //1x100 where 100 is the number of ransac iterations
			int nPairs = sum(mask_)[0];
			//cvgmBuf.convertTo(cvgmBuf, CV_64FC1, 1. / nPairs);
			//vResults.push_back(SumBuf);
			//GpuMat SumBuf2;
			//cv::cuda::reduce(cvgmBuf2, SumBuf2, 1, CV_REDUCE_SUM, CV_64FC1); //1x100 where 100 is the number of ransac iterations
			//vResults.push_back(SumBuf2);
			return SumBuf;
		}//registration()

		double dvo_icp_energy(const Intr& sCamIntr_,
			const Matd33& R_rl_, const double3& T_rl_,
			const GpuMat& VMapRef_, const GpuMat& NMapRef_, const GpuMat& NIRef_,
			const GpuMat& VMapLive_, const GpuMat& NMapLive_, const GpuMat& NILive_, const GpuMat& DepthLive_, const GpuMat& ErrLive_,
			GpuMat& mask_, float icp_weight){

			float fDistThres_ = 0.25f; //meters works for the desktop non-stationary situation.
			float fCosAngleThres_ = 0.5f; //cos(M_PI/3)
			cudaSafeCall(cudaMemcpyToSymbol(__fDistThres, &fDistThres_, sizeof(float))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__fCosAngleThres, &fCosAngleThres_, sizeof(float))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &sCamIntr_, sizeof(Intr))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__R_rl, &R_rl_, sizeof(Matd33)));
			cudaSafeCall(cudaMemcpyToSymbol(__T_rl, &T_rl_, sizeof(double3))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(VMapLive_.cols), sizeof(int))); //copy host memory to constant memory on the device.
			cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(VMapLive_.rows), sizeof(int))); //copy host memory to constant memory on the device.

			assert(mask_.cols == VMapLive_.cols && mask_.rows == VMapLive_.rows);
			mask_.setTo(uchar(0));

			GpuMat mask2 = mask_.clone();
			mask2.setTo(0);

			SDeviceDVOICP sICP;

			sICP._DepthLive = DepthLive_;
			sICP._VMapLive = VMapLive_;
			sICP._NMapLive = NMapLive_;
			sICP._NILive = NILive_;
			sICP._ErrLive = ErrLive_;
			sICP._VMapRef = VMapRef_;
			sICP._NMapRef = NMapRef_;
			sICP._NIRef = NIRef_;

			sICP._mask = mask_;
			sICP._mask2 = mask2;
			sICP._weight_rgb = .01 / 32;
			sICP._weight_v = 1.0;// icp_weight;

			dim3 block(SDeviceDVOICP::CTA_SIZE_X, SDeviceDVOICP::CTA_SIZE_Y);
			dim3 grid(1, 1, 1);
			grid.x = cv::cudev::divUp(VMapLive_.cols, block.x);
			grid.y = cv::cudev::divUp(VMapLive_.rows, block.y);

			GpuMat cvgmE(VMapLive_.size(), CV_64FC1); cvgmE.setTo(0.);
			sICP._cvgmE = cvgmE;

			kernel_dvo_icp_energy << < grid, block >> >(sICP);
			//cudaSafeCall(cudaGetLastError());
			//cudaSafeCall(cudaDeviceSynchronize());

			double dEnergy = sum(cvgmE)[0];
			int nPairs = sum(mask_)[0];
			int total = sum(mask2)[0];

			//The following equations are come from equation (10) in
			//Chetverikov, D., Stepanov, D., & Krsek, P. (2005). 
			//Robust Euclidean alignment of 3D point sets: the trimmed iterative closest point algorithm. 
			//IVC, 23(3), 299¨C309. doi:10.1016/j.imavis.2004.05.007
			//cout << "original = " << dEnergy << "\t";
			dEnergy /= nPairs;
			float xee = float(nPairs) / float(VMapLive_.cols*VMapLive_.rows);
			dEnergy /= (xee*xee*xee);
			//cout << "energy = " << dEnergy << "\t" << nPairs << "\t" << total << "\t" << VMapLive_.cols << endl;
			return dEnergy; // / nPairs
		}//registration()


	}//device
}//btl