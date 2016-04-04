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


namespace btl{
namespace device {
typedef double float_type;
using namespace pcl::device;
using namespace cv::cuda;
using namespace std;

__constant__ Intr __sCamIntr;

__constant__ Matd33  __R_lr;
__constant__ double3 __T_lr;
__constant__ float __fAlpha;

__constant__ int __nCols;
__constant__ int __nRows;

template<typename T>
inline __host__ __device__ float _lerp(T s, T e, float t)
{
	return s + (e - s)*t;
}

template<typename T>
inline __host__ __device__ float _blerp(T c00, T c10, T c01, T c11, float tx, float ty)
{
	return _lerp<float>(_lerp<T>(c00, c10, tx), _lerp<T>(c01, c11, tx), ty);
}


struct SDeviceExposure
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

	PtrStepSz<uchar> _errBLive;
	PtrStepSz<uchar> _errGLive;
	PtrStepSz<uchar> _errRLive;
	PtrStepSz<float> _RadianceBLive;
	PtrStepSz<float> _RadianceGLive;
	PtrStepSz<float> _RadianceRLive;
	PtrStepSz<float> _RadianceBRef;
	PtrStepSz<float> _RadianceGRef;
	PtrStepSz<float> _RadianceRRef;
	PtrStepSz<float3> _VMapRef;
	PtrStepSz<uchar> _mask;

	mutable PtrStepSz<float_type> _cvgmBuf;
	mutable PtrStepSz<float_type> _cvgmE;
	mutable PtrStepSz<float_type> _cvgmEError;

	template<typename T>
	inline __host__ __device__ float bilinearInterpolateValue(
		const PtrStepSz<T>& array,
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
			const T c00 = array.ptr(iy)[ix + 0];

			///            uint32_t c10 = getpixel(src, gxi+1, gyi);
			const T c10 = array.ptr(iy)[ix + 1];

			///            uint32_t c01 = getpixel(src, gxi, gyi+1);
			const T c01 = array.ptr(iy + 1)[ix + 0];

			///            uint32_t c11 = getpixel(src, gxi+1, gyi+1);
			const T c11 = array.ptr(iy + 1)[ix + 1];

			return _blerp<T>(c00, c10, c01, c11, fx, fy);
		}
		else
			return NAN;
	}

	__device__ __forceinline__ TYPE searchForCorrespondence(int nX_, int nY_, float3& I_r, float3& I_l, float3& err) {
		//retrieve normal
		float3 X_r = _VMapRef.ptr(nY_)[nX_]; if (isnan(X_r.x) || isnan(X_r.y) || isnan(X_r.z)) return _V_FALSE; //retrieve vertex from current frame
		float3 Xc_l_ = __R_lr * X_r + __T_lr; //transform vertex in current frame to previous frame
		//projection onto reference image
		float2 f2Liv = make_float2(Xc_l_.x * __sCamIntr.fx / Xc_l_.z + __sCamIntr.cx, Xc_l_.y * __sCamIntr.fy / Xc_l_.z + __sCamIntr.cy);
		int2 n2Liv;
		n2Liv.x = __float2int_rd(f2Liv.x + .5f);
		n2Liv.y = __float2int_rd(f2Liv.y + .5f);
		//if projected out of the frame, return false
		if (n2Liv.x <= 0 || n2Liv.y <= 0 || n2Liv.x >= __nCols - 1 || n2Liv.y >= __nRows - 1 ) return _RGB_FALSE;

		I_r.x = _RadianceBRef.ptr(nY_)[nX_]; if (I_r.x!= I_r.x) return _RGB_FALSE;
		I_l.x = bilinearInterpolateValue<float>(_RadianceBLive, f2Liv.x, f2Liv.y); if (I_l.x != I_l.x) return _RGB_FALSE;
		err.x = bilinearInterpolateValue<uchar>(_errBLive, f2Liv.x, f2Liv.y); if (err.x != err.x) return _RGB_FALSE;

		I_r.y = _RadianceGRef.ptr(nY_)[nX_]; if (I_r.y != I_r.y) return _RGB_FALSE;
		I_l.y = bilinearInterpolateValue<float>(_RadianceGLive, f2Liv.x, f2Liv.y); if (I_l.y != I_l.y) return _RGB_FALSE;
		err.y = bilinearInterpolateValue<uchar>(_errGLive, f2Liv.x, f2Liv.y); if (err.y != err.y) return _RGB_FALSE;

		I_r.z = _RadianceRRef.ptr(nY_)[nX_]; if (I_r.z != I_r.z) return _RGB_FALSE;
		I_l.z = bilinearInterpolateValue<float>(_RadianceRLive, f2Liv.x, f2Liv.y); if (I_l.z != I_l.z) return _RGB_FALSE;
		err.z = bilinearInterpolateValue<uchar>(_errRLive, f2Liv.x, f2Liv.y); if (err.z != err.z) return _RGB_FALSE;
		return _TRUE;
	}//searchForCorrespondence() 

	__device__ __forceinline__ TYPE removeOutlier(int nX_, int nY_) {
		float3 I_r, I_l;
		//retrieve normal
		float3 X_r = _VMapRef.ptr(nY_)[nX_]; if (isnan(X_r.x) || isnan(X_r.y) || isnan(X_r.z)) return _V_FALSE; //retrieve vertex from current frame
		float3 Xc_l_ = __R_lr * X_r + __T_lr; //transform vertex in current frame to previous frame
		//projection onto reference image
		float2 f2Liv = make_float2(Xc_l_.x * __sCamIntr.fx / Xc_l_.z + __sCamIntr.cx, Xc_l_.y * __sCamIntr.fy / Xc_l_.z + __sCamIntr.cy);
		int2 n2Liv;
		n2Liv.x = __float2int_rd(f2Liv.x + .5f);
		n2Liv.y = __float2int_rd(f2Liv.y + .5f);
		//if projected out of the frame, return false
		if (n2Liv.x <= 0 || n2Liv.y <= 0 || n2Liv.x >= __nCols - 1 || n2Liv.y >= __nRows - 1) return _RGB_FALSE;

		I_r.x = _RadianceBRef.ptr(nY_)[nX_]; if (I_r.x != I_r.x) return _RGB_FALSE;
		I_l.x = bilinearInterpolateValue<float>(_RadianceBLive, f2Liv.x, f2Liv.y); if (I_l.x != I_l.x) return _RGB_FALSE;

		if (fabs(I_r.x - I_l.x) > 0.4){
			_RadianceBLive.ptr(n2Liv.y)[n2Liv.x] = NAN;
			_RadianceBLive.ptr(n2Liv.y + 1)[n2Liv.x] = NAN;
			_RadianceBLive.ptr(n2Liv.y - 1)[n2Liv.x] = NAN;
			_RadianceBLive.ptr(n2Liv.y)[n2Liv.x - 1] = NAN;
			_RadianceBLive.ptr(n2Liv.y)[n2Liv.x + 1] = NAN;
			return _RGB_FALSE;
		}

		I_r.y = _RadianceGRef.ptr(nY_)[nX_]; if (I_r.y != I_r.y) return _RGB_FALSE;
		I_l.y = bilinearInterpolateValue<float>(_RadianceGLive, f2Liv.x, f2Liv.y); if (I_l.y != I_l.y) return _RGB_FALSE;

		if (fabs(I_r.y - I_l.y) > 0.4){
			_RadianceGLive.ptr(n2Liv.y)[n2Liv.x] = NAN;
			_RadianceGLive.ptr(n2Liv.y + 1)[n2Liv.x] = NAN;
			_RadianceGLive.ptr(n2Liv.y - 1)[n2Liv.x] = NAN;
			_RadianceGLive.ptr(n2Liv.y)[n2Liv.x - 1] = NAN;
			_RadianceGLive.ptr(n2Liv.y)[n2Liv.x + 1] = NAN;
			return _RGB_FALSE;
		}

		I_r.z = _RadianceRRef.ptr(nY_)[nX_]; if (I_r.z != I_r.z) return _RGB_FALSE;
		I_l.z = bilinearInterpolateValue<float>(_RadianceRLive, f2Liv.x, f2Liv.y); if (I_l.z != I_l.z) return _RGB_FALSE;

		if (fabs(I_r.z - I_l.z) > 0.4){
			_RadianceRLive.ptr(n2Liv.y)[n2Liv.x] = NAN;
			_RadianceRLive.ptr(n2Liv.y + 1)[n2Liv.x] = NAN;
			_RadianceRLive.ptr(n2Liv.y - 1)[n2Liv.x] = NAN;
			_RadianceRLive.ptr(n2Liv.y)[n2Liv.x - 1] = NAN;
			_RadianceRLive.ptr(n2Liv.y)[n2Liv.x + 1] = NAN;
			return _RGB_FALSE;
		}
		return _TRUE;
	}//searchForCorrespondence() 

	__device__ __forceinline__ void simple_expo() {
		int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
		if (nX >= __nCols || nY >= __nRows) return;

		float3 I_l, I_r, err;
		// read out current point in world and find its corresponding point in previous frame, which are also defined in world
		if (_TRUE == searchForCorrespondence(nX, nY, I_r, I_l, err)){
			int nn = nX + nY * __nCols;
			if (I_r.x > 1e-3 && I_r.y > 1e-3 && I_r.z > 1e-3){
				_cvgmE.ptr(nY)[nX] = I_r.x / I_l.x * err.x + I_r.y / I_l.y * err.y + I_r.z / I_l.z * err.z ;
				_cvgmEError.ptr(nY)[nX] = err.x + err.y + err.z;
				_mask.ptr(nY)[nX] = uchar(1);
			}
		}//if correspondence found

		return;
	}//simple_expo()

	__device__ __forceinline__ void remove_outlier() {
		int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
		if (nX >= __nCols || nY >= __nRows) return;

		// read out current point in world and find its corresponding point in previous frame, which are also defined in world
		if (_TRUE == removeOutlier(nX, nY)){
			_mask.ptr(nY)[nX] = uchar(1);
		}//if correspondence found

		return;
	}//remove_outlier()
};//SDeviceICPEnergyRegistration

__global__ void kernel_simple_expo(SDeviceExposure sICP) {
	sICP.simple_expo();
}

double exposure_est2(const Intr& sCamIntr_,
	const Matd33& R_lr_, const double3& T_lr_,
	const GpuMat& RadianceBLive_, const GpuMat& errBLive_,
	const GpuMat& RadianceGLive_, const GpuMat& errGLive_,
	const GpuMat& RadianceRLive_, const GpuMat& errRLive_,
	const GpuMat& VMapRef_, const GpuMat& RadianceBRef_, const GpuMat& RadianceGRef_, const GpuMat& RadianceRRef_, GpuMat& mask_){
	cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &sCamIntr_, sizeof(Intr))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__R_lr, &R_lr_, sizeof(Matd33)));
	cudaSafeCall(cudaMemcpyToSymbol(__T_lr, &T_lr_, sizeof(double3))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(RadianceBRef_.cols), sizeof(int))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(RadianceBRef_.rows), sizeof(int))); //copy host memory to constant memory on the device.

	mask_.create(RadianceBLive_.size(), CV_8UC1);
	mask_.setTo(uchar(0));

	SDeviceExposure sICP;

	sICP._VMapRef = VMapRef_;
	sICP._RadianceBLive = RadianceBLive_;
	sICP._RadianceGLive = RadianceGLive_;
	sICP._RadianceRLive = RadianceRLive_;
	sICP._errBLive = errBLive_;
	sICP._errGLive = errGLive_;
	sICP._errRLive = errRLive_;
	sICP._RadianceBRef = RadianceBRef_;
	sICP._RadianceGRef = RadianceGRef_;
	sICP._RadianceRRef = RadianceRRef_;

	sICP._mask = mask_;

	dim3 block(SDeviceExposure::CTA_SIZE_X, SDeviceExposure::CTA_SIZE_Y);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(RadianceBRef_.cols, block.x);
	grid.y = cv::cudev::divUp(RadianceBRef_.rows, block.y);

	GpuMat cvgmE(RadianceBRef_.size(), CV_64FC1); cvgmE.setTo(0.);
	GpuMat Error = cvgmE.clone();
	sICP._cvgmEError = Error;
	sICP._cvgmE = cvgmE;

	kernel_simple_expo << < grid, block >> >(sICP);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	//cout << cv::Mat(cvgmE) << endl;
	double dEnergy = sum(cvgmE)[0];
	double dTotalError = sum(Error)[0];
	int nPairs = sum(mask_)[0];
	dEnergy /= dTotalError;
	//dEnergy /= nPairs;
	//The following equations are come from equation (10) in
	//Chetverikov, D., Stepanov, D., & Krsek, P. (2005). 
	//Robust Euclidean alignment of 3D point sets: the trimmed iterative closest point algorithm. 
	//IVC, 23(3), 299¨C309. doi:10.1016/j.imavis.2004.05.007
	//cout << "original = " << dEnergy << "\t";
	//dEnergy /= nPairs;
	//dEnergy2 /= nPairs;
	//float xee = float(nPairs) / float(VMapLive_.cols*VMapLive_.rows);
	//dEnergy /= (xee*xee*xee); 
	//dEnergy2 /= (xee*xee*xee);
	//cout << "energy = " << dEnergy << "\t" << nPairs << "\t" << VMapRef_.cols << endl;
	return dEnergy;
}

__global__ void kernel_remove_outlier(SDeviceExposure sICP) {
	sICP.remove_outlier();
}

void remove_outlier(const Intr& sCamIntr_,
	const Matd33& R_lr_, const double3& T_lr_, 
	const GpuMat& RadianceBLive_, const GpuMat& errBLive_,
	const GpuMat& RadianceGLive_, const GpuMat& errGLive_,
	const GpuMat& RadianceRLive_, const GpuMat& errRLive_,
	const GpuMat& VMapRef_, const GpuMat& RadianceBRef_, const GpuMat& RadianceGRef_, const GpuMat& RadianceRRef_, GpuMat& mask_){
	cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &sCamIntr_, sizeof(Intr))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__R_lr, &R_lr_, sizeof(Matd33)));
	cudaSafeCall(cudaMemcpyToSymbol(__T_lr, &T_lr_, sizeof(double3))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(RadianceBRef_.cols), sizeof(int))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(RadianceBRef_.rows), sizeof(int))); //copy host memory to constant memory on the device.

	mask_.create(RadianceBLive_.size(), CV_8UC1);
	mask_.setTo(uchar(0));

	SDeviceExposure sICP;

	sICP._VMapRef = VMapRef_;
	sICP._RadianceBLive = RadianceBLive_;
	sICP._RadianceGLive = RadianceGLive_;
	sICP._RadianceRLive = RadianceRLive_;
	sICP._errBLive = errBLive_;
	sICP._errGLive = errGLive_;
	sICP._errRLive = errRLive_;
	sICP._RadianceBRef = RadianceBRef_;
	sICP._RadianceGRef = RadianceGRef_;
	sICP._RadianceRRef = RadianceRRef_;

	sICP._mask = mask_;

	dim3 block(SDeviceExposure::CTA_SIZE_X, SDeviceExposure::CTA_SIZE_Y);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(RadianceBRef_.cols, block.x);
	grid.y = cv::cudev::divUp(RadianceBRef_.rows, block.y);

	kernel_remove_outlier<< < grid, block >> >(sICP);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	//cout << cv::Mat(cvgmE) << endl;
	int nPairs = sum(mask_)[0];
	//dEnergy /= nPairs;
	//The following equations are come from equation (10) in
	//Chetverikov, D., Stepanov, D., & Krsek, P. (2005). 
	//Robust Euclidean alignment of 3D point sets: the trimmed iterative closest point algorithm. 
	//IVC, 23(3), 299¨C309. doi:10.1016/j.imavis.2004.05.007
	//cout << "original = " << dEnergy << "\t";
	//dEnergy /= nPairs;
	//dEnergy2 /= nPairs;
	//float xee = float(nPairs) / float(VMapLive_.cols*VMapLive_.rows);
	//dEnergy /= (xee*xee*xee); 
	//dEnergy2 /= (xee*xee*xee);
	cout << "inliers = " << nPairs << "\t" << VMapRef_.cols << endl;
	return;
}

}//device
}//btl