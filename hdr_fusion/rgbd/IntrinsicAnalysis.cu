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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif
#include <math_constants.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda/common.hpp>

#include <boost/shared_ptr.hpp>

#include <iostream>
#include "IntrinsicAnalysis.cuh"

using namespace cv;
using namespace std;
using namespace cv::cuda;
using namespace btl::device;

#define DIVISOR 10000
#define INV_DIV 0.0001f

inline __device__ short pack_float(float in_){
	return short(fmaxf(-32767, fminf(32768, __float2int_rz(in_* DIVISOR))));
}
inline __device__ float unpack_float(short in_){
	return __int2float_rn(in_) *INV_DIV;
}

__constant__ float __ln_sample[6]; //step, start, step, start, step, start
__constant__ float __nlf_sqr[6]; //b, g, r, x(0), x(1)
__constant__ float __normalize_factor[6]; //b, g, r, x(0), x(1)


template<typename T>
__global__ void kernel_avg(PtrStepSz<double> int_img_, PtrStepSz<T> out_, short rad_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 1 || nX >= int_img_.cols || nY < 1 || nY >= int_img_.rows) return;

	T total = 0;
	int y0 = nY - rad_ - 1; y0 = y0 < 0 ? 0 : y0;
	int x0 = nX - rad_ - 1; x0 = x0 < 0 ? 0 : x0;
	int y1 = nY + rad_; y1 = y1 >= int_img_.rows ? int_img_.rows - 1 : y1;
	int x1 = nX + rad_; x1 = x1 >= int_img_.cols ? int_img_.cols - 1 : x1;
	total += int_img_.ptr(y0)[x0];//A
	total += int_img_.ptr(y1)[x1];//D
	total -= int_img_.ptr(y0)[x1];//B
	total -= int_img_.ptr(y1)[x0];//C
	T area = (y1 - y0)*(x1 - x0);
	//printf("%f\t%f\t%f\t%d\t%d\t%f\t%d\t%d\t%f\t%d\t%d\n", total, area, total / area, nX, nY, int_img_.ptr(y0)[x0], x0, y0, int_img_.ptr(y1)[x0], x1, y1);
	out_.ptr(nY - 1)[nX - 1] = total / area;
	return;
}
template<typename T>
__global__ void kernel_std(PtrStepSz<double> int_sqr_, PtrStepSz<T> mean_, PtrStepSz<T> std_, short radius_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 1 || nX >= int_sqr_.cols || nY < 1 || nY >= int_sqr_.rows) return;

	double total = 0;
	short y0 = nY - radius_ - 1; y0 = y0 < 0 ? 0 : y0;
	short x0 = nX - radius_ - 1; x0 = x0 < 0 ? 0 : x0;
	short y1 = nY + radius_; y1 = y1 >= int_sqr_.rows ? int_sqr_.rows - 1 : y1;
	short x1 = nX + radius_; x1 = x1 >= int_sqr_.cols ? int_sqr_.cols - 1 : x1;
	total += int_sqr_.ptr(y0)[x0];//A
	total += int_sqr_.ptr(y1)[x1];//D
	total -= int_sqr_.ptr(y0)[x1];//B
	total -= int_sqr_.ptr(y1)[x0];//C
	T area = (y1 - y0)*(x1 - x0);
	T u = mean_.ptr(nY - 1)[nX - 1];
	T v = total / area - u*u;
	if (v < 0.) {
		//printf("%f %f %f %f %d %d\n", total, area, u*u, v, nY, nX);
		std_.ptr(nY - 1)[nX - 1] = NAN;
	}
	std_.ptr(nY - 1)[nX - 1] = __dsqrt_rd(v);
	return;
}

template <typename T>
inline __host__ __device__ T _lerp(T s, T e, T t)
{
	return s + (e - s)*t;
}

template <typename T>
inline __host__ __device__ T _blerp(T c00, T c10, T c01, T c11, T tx, T ty)
{
	return _lerp(_lerp(c00, c10, tx), _lerp(c01, c11, tx), ty);
}

template <typename T>
inline __host__ __device__ T normalize_with_bilinear_intr(
	const float& raw_,
	const PtrStepSz<T>& mean_,
	const PtrStepSz<T>& std_,
	T x_ind,
	T y_ind)
{
	const int ix = floor(x_ind);
	const int iy = floor(y_ind);
	const T fx = x_ind - ix;
	const T fy = y_ind - iy;

	if (ix >= 0 && ix < mean_.cols - 1 && iy >= 0 && iy < mean_.rows - 1)
	{
		const T m00 = mean_.ptr(iy)[ix + 0];
		const T m10 = mean_.ptr(iy)[ix + 1];
		const T m01 = mean_.ptr(iy + 1)[ix + 0];
		const T m11 = mean_.ptr(iy + 1)[ix + 1];

		const T s00 = std_.ptr(iy)[ix + 0];
		const T s10 = std_.ptr(iy)[ix + 1];
		const T s01 = std_.ptr(iy + 1)[ix + 0];
		const T s11 = std_.ptr(iy + 1)[ix + 1];
		T std = _blerp<T>(s00, s10, s01, s11, fx, fy);
		if (std <= T(0.)){
			//printf("%d, %d : %f \n", ix, iy, raw_ - _blerp<T>(m00, m10, m01, m11, fx, fy));
			return raw_ - _blerp<T>(m00, m10, m01, m11, fx, fy);
		}
		else{
			//printf("%d, %d : %f \n", ix, iy, (raw_ - _blerp<T>(m00, m10, m01, m11, fx, fy))/std);
			return (raw_ - _blerp<T>(m00, m10, m01, m11, fx, fy)) / std;
		}
		//printf("%f\n", tmp);
	}
	else if (ix == mean_.cols - 1 && iy < mean_.rows - 1){
		const T m00 = mean_.ptr(iy)[ix + 0];
		const T m11 = mean_.ptr(iy + 1)[ix + 0];

		const T s00 = std_.ptr(iy)[ix + 0];
		const T s11 = std_.ptr(iy + 1)[ix + 0];
		T std = _lerp(s00, s11, fy);
		if (std <= T(0.))
			return raw_ - _lerp(m00, m11, fy);
		else
			return (raw_ - _lerp(m00, m11, fy)) / std;
	}
	else if (ix < mean_.cols - 1 && iy == mean_.rows - 1){
		const T m00 = mean_.ptr(iy)[ix + 0];
		const T m11 = mean_.ptr(iy)[ix + 1];

		const T s00 = std_.ptr(iy)[ix + 0];
		const T s11 = std_.ptr(iy)[ix + 1];
		T std = _lerp(s00, s11, fx);
		if (std <= T(0.))
			return raw_ - _lerp(m00, m11, fx);
		else
			return (raw_ - _lerp(m00, m11, fx)) / std;
	}
	else if (ix == mean_.cols - 1 && iy == mean_.rows - 1){
		T std = std_.ptr(iy)[ix];
		if (std <= T(0.))
			return raw_ - mean_.ptr(iy)[ix];
		else
			return (raw_ - mean_.ptr(iy)[ix]) / std;
	}
	else
		return NAN;
}

template<typename T>
__global__ void kernel_normalize_intr(PtrStepSz<float> radiance_, PtrStepSz<T> mean_, PtrStepSz<T> std_, PtrStepSz<T> normalized_radiance_,
	float ratio_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= radiance_.cols || nY < 0 || nY >= radiance_.rows) return;
	T x = nX * T(ratio_);
	T y = nY * T(ratio_);
	//printf("%f, %f : %f \n", x, y, normalize_with_bilinear_intr<T>(in_.ptr(nY)[nX], mean_, std_, x, y));
	normalized_radiance_.ptr(nY)[nX] = (normalize_with_bilinear_intr<T>(radiance_.ptr(nY)[nX], mean_, std_, x, y)); //nr normalized radiance
	return;
}

template<typename T>
__global__ void kernel_normalize(PtrStepSz<float> in_, double mean_, double std_, PtrStepSz<short> normalized_intensity_map_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= in_.cols || nY < 0 || nY >= in_.rows) return;
	normalized_intensity_map_.ptr(nY)[nX] = pack_float((in_.ptr(nY)[nX] - mean_) / std_);
	return;
}

template <typename T>
__device__ T anti_normalize_with_bilinear_intr(
	const float& raw_,
	const PtrStepSz<T>& mean_,
	const PtrStepSz<T>& std_,
	T x_ind,
	T y_ind, bool disp = false)
{
	const int ix = floor(x_ind);
	const int iy = floor(y_ind);
	const T fx = x_ind - ix;
	const T fy = y_ind - iy;

	if (ix >= 0 && ix < mean_.cols - 1 && iy >= 0 && iy < mean_.rows - 1)
	{
		const T m00 = mean_.ptr(iy)[ix + 0];
		const T m10 = mean_.ptr(iy)[ix + 1];
		const T m01 = mean_.ptr(iy + 1)[ix + 0];
		const T m11 = mean_.ptr(iy + 1)[ix + 1];

		const T s00 = std_.ptr(iy)[ix + 0];
		const T s10 = std_.ptr(iy)[ix + 1];
		const T s01 = std_.ptr(iy + 1)[ix + 0];
		const T s11 = std_.ptr(iy + 1)[ix + 1];
		T std = _blerp<T>(s00, s10, s01, s11, fx, fy);
		if (disp){
			printf("std %f, mean %f ", std, _blerp<T>(m00, m10, m01, m11, fx, fy));
		}
		if (std <= T(0.) || std != std)
			return _blerp<T>(m00, m10, m01, m11, fx, fy);
		else
			return raw_ * std + _blerp<T>(m00, m10, m01, m11, fx, fy);
	}
	else if (ix == mean_.cols - 1 && iy < mean_.rows - 1){
		const T m00 = mean_.ptr(iy)[ix + 0];
		const T m11 = mean_.ptr(iy + 1)[ix + 0];

		const T s00 = std_.ptr(iy)[ix + 0];
		const T s11 = std_.ptr(iy + 1)[ix + 0];
		T std = _lerp(s00, s11, fy);
		if (std <= T(0.) || std != std)
			return _lerp(m00, m11, fy);
		else
			return raw_ * std + _lerp(m00, m11, fy);
	}
	else if (ix < mean_.cols - 1 && iy == mean_.rows - 1){
		const T m00 = mean_.ptr(iy)[ix + 0];
		const T m11 = mean_.ptr(iy)[ix + 1];

		const T s00 = std_.ptr(iy)[ix + 0];
		const T s11 = std_.ptr(iy)[ix + 1];
		T std = _lerp(s00, s11, fx);
		if (std <= T(0.) || std != std)
			return _lerp(m00, m11, fx);
		else
			return raw_ * std + _lerp(m00, m11, fx);
	}
	else if (ix == mean_.cols - 1 && iy == mean_.rows - 1){
		T std = std_.ptr(iy)[ix];
		if (std <= T(0.) || std != std)
			return mean_.ptr(iy)[ix];
		else
			return raw_ * std + mean_.ptr(iy)[ix];
	}
	else
		return NAN;
}

template <typename T>
__global__ void kernel_denoising(PtrStepSz<double> norm_b_, PtrStepSz<double> norm_g_, PtrStepSz<double> norm_r_,
	PtrStepSz<float> err_b_, PtrStepSz<float> err_g_, PtrStepSz<float> err_r_, PtrStepSz<T> out_b_, PtrStepSz<T> out_g_, PtrStepSz<T> out_r_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 1 || nX >= norm_b_.cols - 1 || nY < 1 || nY >= norm_b_.rows - 1) return;
	if (err_b_.ptr(nY)[nX] < 1e-3 || err_g_.ptr(nY)[nX] < 1e-3 || err_r_.ptr(nY)[nX] < 1e-3){
		{
			double wi = norm_b_.ptr(nY - 1)[nX - 1] * err_b_.ptr(nY - 1)[nX - 1] +
				norm_b_.ptr(nY - 1)[nX] * err_b_.ptr(nY - 1)[nX] +
				norm_b_.ptr(nY - 1)[nX + 1] * err_b_.ptr(nY - 1)[nX + 1] +
				norm_b_.ptr(nY)[nX - 1] * err_b_.ptr(nY)[nX - 1] +
				norm_b_.ptr(nY)[nX] * err_b_.ptr(nY)[nX] +
				norm_b_.ptr(nY)[nX + 1] * err_b_.ptr(nY)[nX + 1] +
				norm_b_.ptr(nY + 1)[nX - 1] * err_b_.ptr(nY + 1)[nX - 1] +
				norm_b_.ptr(nY + 1)[nX] * err_b_.ptr(nY + 1)[nX] +
				norm_b_.ptr(nY + 1)[nX + 1] * err_b_.ptr(nY + 1)[nX + 1];
			double W = err_b_.ptr(nY - 1)[nX - 1] + err_b_.ptr(nY - 1)[nX] + err_b_.ptr(nY - 1)[nX + 1] +
				err_b_.ptr(nY)[nX - 1] + err_b_.ptr(nY)[nX] + err_b_.ptr(nY)[nX + 1] +
				err_b_.ptr(nY + 1)[nX - 1] + err_b_.ptr(nY + 1)[nX] + err_b_.ptr(nY + 1)[nX + 1];
			out_b_.ptr(nY)[nX] = wi / W;
		}
		{
			double wi = norm_g_.ptr(nY - 1)[nX - 1] * err_g_.ptr(nY - 1)[nX - 1] +
				norm_g_.ptr(nY - 1)[nX] * err_g_.ptr(nY - 1)[nX] +
				norm_g_.ptr(nY - 1)[nX + 1] * err_g_.ptr(nY - 1)[nX + 1] +
				norm_g_.ptr(nY)[nX - 1] * err_g_.ptr(nY)[nX - 1] +
				norm_g_.ptr(nY)[nX] * err_g_.ptr(nY)[nX] +
				norm_g_.ptr(nY)[nX + 1] * err_g_.ptr(nY)[nX + 1] +
				norm_g_.ptr(nY + 1)[nX - 1] * err_g_.ptr(nY + 1)[nX - 1] +
				norm_g_.ptr(nY + 1)[nX] * err_g_.ptr(nY + 1)[nX] +
				norm_g_.ptr(nY + 1)[nX + 1] * err_g_.ptr(nY + 1)[nX + 1];
			double W = err_b_.ptr(nY - 1)[nX - 1] + err_g_.ptr(nY - 1)[nX] + err_g_.ptr(nY - 1)[nX + 1] +
				err_g_.ptr(nY)[nX - 1] + err_g_.ptr(nY)[nX] + err_g_.ptr(nY)[nX + 1] +
				err_g_.ptr(nY + 1)[nX - 1] + err_g_.ptr(nY + 1)[nX] + err_g_.ptr(nY + 1)[nX + 1];
			out_g_.ptr(nY)[nX] = wi / W;
		}
		{
			double wi = norm_r_.ptr(nY - 1)[nX - 1] * err_r_.ptr(nY - 1)[nX - 1] +
				norm_r_.ptr(nY - 1)[nX] * err_r_.ptr(nY - 1)[nX] +
				norm_r_.ptr(nY - 1)[nX + 1] * err_r_.ptr(nY - 1)[nX + 1] +
				norm_r_.ptr(nY)[nX - 1] * err_r_.ptr(nY)[nX - 1] +
				norm_r_.ptr(nY)[nX] * err_r_.ptr(nY)[nX] +
				norm_r_.ptr(nY)[nX + 1] * err_r_.ptr(nY)[nX + 1] +
				norm_r_.ptr(nY + 1)[nX - 1] * err_r_.ptr(nY + 1)[nX - 1] +
				norm_r_.ptr(nY + 1)[nX] * err_r_.ptr(nY + 1)[nX] +
				norm_r_.ptr(nY + 1)[nX + 1] * err_r_.ptr(nY + 1)[nX + 1];
			double W = err_b_.ptr(nY - 1)[nX - 1] + err_r_.ptr(nY - 1)[nX] + err_r_.ptr(nY - 1)[nX + 1] +
				err_r_.ptr(nY)[nX - 1] + err_r_.ptr(nY)[nX] + err_r_.ptr(nY)[nX + 1] +
				err_r_.ptr(nY + 1)[nX - 1] + err_r_.ptr(nY + 1)[nX] + err_r_.ptr(nY + 1)[nX + 1];
			out_r_.ptr(nY)[nX] = wi / W;
		}
	}
	else{
		out_b_.ptr(nY)[nX] = norm_b_.ptr(nY)[nX];
		out_g_.ptr(nY)[nX] = norm_g_.ptr(nY)[nX];
		out_r_.ptr(nY)[nX] = norm_r_.ptr(nY)[nX];
	}

	return;
}
template <typename T>
__global__ void kernel_anti_normalize_intr(PtrStepSz<T> normalized_radiance_, PtrStepSz<float> mean_, PtrStepSz<float> std_, PtrStepSz<float> radiance_, float ratio_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= normalized_radiance_.cols || nY < 0 || nY >= normalized_radiance_.rows) return;
	T x = nX * T(ratio_);
	T y = nY * T(ratio_);
	T nr = (normalized_radiance_.ptr(nY)[nX]); //normalized intensity
	bool disp = false;
	//if (nX == 303 && nY == 209){
	//	disp = true;
	//}
	radiance_.ptr(nY)[nX] = anti_normalize_with_bilinear_intr<float>(nr, mean_, std_, x, y, disp);
	//if (nX == 303 && nY == 209){
	//	printf("nomalized %f anti_radian %f", ni, out_.ptr(nY)[nX]);
	//}
	//printf("%f, %f : %f \n", x, y, anti_normalize_with_bilinear_intr<T>(ni, mean_, std_, x, y));
	return;
}

template <typename T>
__global__ void kernel_anti_normalize(PtrStepSz<short> in_, T mean_, T std_, PtrStepSz<float> out_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= in_.cols || nY < 0 || nY >= in_.rows) return;
	out_.ptr(nY)[nX] = unpack_float(in_.ptr(nY)[nX]) * std_ + mean_;
	return;
}

__global__ void kernel_invcrf_rgb2rad_with_ccf(PtrStepSz<uchar> inten_, PtrStepSz<float> inv_crf_, PtrStepSz<float> deriveCRF_, uchar ch_,
	PtrStepSz<float> radiance_bgr_, PtrStepSz<uchar> err_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= inten_.cols || nY < 0 || nY >= inten_.rows) return;
	uchar I = inten_.ptr(nY)[nX];
	float rad = inv_crf_.ptr()[I];
	radiance_bgr_.ptr(nY)[nX] = rad;
	float ccf = deriveCRF_.ptr()[I] * sqrtf(rad*__nlf_sqr[ch_ * 2] + __nlf_sqr[ch_ * 2 + 1]) / __normalize_factor[1];
	ccf /= 255;
	//ccf = pow(ccf,1/3);
	ccf = sqrt(ccf);
	//ccf *= ccf;
	ccf *= 255;
	err_.ptr(nY)[nX] = uchar(ccf < 255 ? ccf : 255);
	return;
}

template <typename T>
__global__ void kernel_to_intensity(PtrStepSz<T> radiance_bgr_, PtrStepSz<float> inv_crf_, PtrStepSz<uchar3> bgr_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= bgr_.cols || nY < 0 || nY >= bgr_.rows) return;
	T* ptr = radiance_bgr_.ptr(nY) + nX * 3;
	float3 radian_bgr;
	radian_bgr.x = *ptr++; //b
	radian_bgr.y = *ptr++; //g
	radian_bgr.z = *ptr; //r
	uchar3 color;
	radian_bgr.x = inv_crf_.ptr(2)[0] * pow(radian_bgr.x, inv_crf_.ptr(2)[1]) + .5f;
	color.x = uchar(radian_bgr.x > 255 ? 255 : radian_bgr.x);
	radian_bgr.y = inv_crf_.ptr(1)[0] * pow(radian_bgr.y, inv_crf_.ptr(1)[1]) + .5f;
	color.y = uchar(radian_bgr.y > 255 ? 255 : radian_bgr.y);
	radian_bgr.z = inv_crf_.ptr(0)[0] * pow(radian_bgr.z, inv_crf_.ptr(0)[1]) + .5f;
	color.z = uchar(radian_bgr.z > 255 ? 255 : radian_bgr.z);
	//printf("x %d , y %d , r %d , rad %f\n", nX, nY, color.x, radian.x);
	bgr_.ptr(nY)[nX] = color;
	return;
}


template <typename T>
__device__ uchar interpolate(T radian_, T rad1, T rad2, T int1, T int2){
	T t = (radian_ - rad1) / (rad2 - rad1);
	t = int1 * (1 - t) + int2*t + T(.5);
	return uchar(t > 255 ? 255 : t);
}

template <typename T>
__device__ float interpolate2(T radian_, T rad1, T rad2, T int1, T int2){
	T t = (radian_ - rad1) / (rad2 - rad1);
	t = int1 * (1 - t) + int2*t;
	return t;
}

template <typename T3>
__global__ void kernel_crf_rad2rgb_with_noise(PtrStepSz<T3> radiance_bgr_, PtrStepSz<float> deriveCRF_, //3x256
	PtrStepSz<float> rad_resample_, PtrStepSz<float> intensity_,
	PtrStepSz<double> ss_b_,
	PtrStepSz<double> ss_g_,
	PtrStepSz<double> ss_r_,
	PtrStepSz<double> sc_b_,
	PtrStepSz<double> sc_g_,
	PtrStepSz<double> sc_r_,
	PtrStepSz<uchar3> bgr_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;

	if (nX < 0 || nX >= bgr_.cols || nY < 0 || nY >= bgr_.rows) return;

	uchar3 color_bgr;
	T3 radian_bgr = radiance_bgr_.ptr(nY)[nX];
	//blue
	short ii = __float2int_rd((__logf(radian_bgr.x) - __ln_sample[5]) / __ln_sample[4]);
	if (ii <= 0){
		color_bgr.x = uchar(0);
	}
	else if (ii < 254){
		float rad1 = rad_resample_.ptr(2)[ii];
		float rad2 = rad_resample_.ptr(2)[ii + 1];
		float int1 = intensity_.ptr(2)[ii];
		float int2 = intensity_.ptr(2)[ii + 1];
		float ss = ss_b_.ptr(nY)[nX];
		float sc = sc_b_.ptr(nY)[nX];
		float noise = interpolate2<float>(radian_bgr.x, rad1, rad2, deriveCRF_.ptr(0)[ii], deriveCRF_.ptr(0)[ii + 1]) *
			sqrtf(radian_bgr.x*ss*ss + sc*sc);
		color_bgr.x = uchar(interpolate2<float>(radian_bgr.x, rad1, rad2, int1, int2) + noise + .5f);

		//if (nX > 300 && nX < 330 && nY > 200 && nY < 230){
		//	printf("%d, %d: noise %f grad %f ss %f sc %f ii %d\n ", nX, nY, noise, interpolate2<float>(radian_bgr.x, rad1, rad2, deriveCRF_.ptr(0)[ii], deriveCRF_.ptr(0)[ii + 1]), ss, sc, ii);
		//}
	}
	else{
		color_bgr.x = uchar(255);
	}

	//green
	ii = __float2int_rz((__logf(radian_bgr.y) - __ln_sample[3]) / __ln_sample[2]);
	if (ii <= 0){
		color_bgr.y = uchar(0);
	}
	else if (ii < 254){
		float rad1 = rad_resample_.ptr(1)[ii];
		float rad2 = rad_resample_.ptr(1)[ii + 1];
		float int1 = intensity_.ptr(1)[ii];
		float int2 = intensity_.ptr(1)[ii + 1];
		float ss = ss_g_.ptr(nY)[nX];
		float sc = sc_g_.ptr(nY)[nX];
		float noise = interpolate2<float>(radian_bgr.y, rad1, rad2, deriveCRF_.ptr(1)[ii], deriveCRF_.ptr(1)[ii + 1]) *
			sqrtf(radian_bgr.y*ss*ss + sc*sc);
		color_bgr.y = uchar(interpolate2<float>(radian_bgr.y, rad1, rad2, int1, int2) + noise + .5f);
	}
	else{
		color_bgr.y = uchar(255);
	}

	//red
	ii = __float2int_rz((__logf(radian_bgr.z) - __ln_sample[1]) / __ln_sample[0]);
	if (ii <= 0){
		color_bgr.z = uchar(0);
	}
	else if (ii < 254){
		float rad1 = rad_resample_.ptr(0)[ii];
		float rad2 = rad_resample_.ptr(0)[ii + 1];
		float int1 = intensity_.ptr(0)[ii];
		float int2 = intensity_.ptr(0)[ii + 1];
		float ss = ss_r_.ptr(nY)[nX];
		float sc = sc_r_.ptr(nY)[nX];
		float noise = interpolate2<float>(radian_bgr.z, rad1, rad2, deriveCRF_.ptr(2)[ii], deriveCRF_.ptr(2)[ii + 1]) *
			sqrtf(radian_bgr.x*ss*ss + sc*sc);
		color_bgr.z = uchar(interpolate2<float>(radian_bgr.z, rad1, rad2, int1, int2) + noise + .5f);
	}
	else{
		color_bgr.z = uchar(255);
	}

	bgr_.ptr(nY)[nX] = color_bgr;
	return;
}

template <typename T3>
__global__ void kernel_crf_rad2rgb(PtrStepSz<T3> radiance_bgr_, PtrStepSz<float> rad_resample_,
	PtrStepSz<float> intensity_, PtrStepSz<uchar3> bgr_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;

	if (nX < 0 || nX >= bgr_.cols || nY < 0 || nY >= bgr_.rows) return;

	uchar3 color_bgr;
	T3 radian_bgr = radiance_bgr_.ptr(nY)[nX];
	//blue
	short ii = __float2int_rd((__logf(radian_bgr.x) - __ln_sample[5]) / __ln_sample[4]);
	if (ii < 0){
		color_bgr.x = uchar(0);
	}
	else if (ii < 254){
		float rad1 = rad_resample_.ptr(2)[ii];
		float rad2 = rad_resample_.ptr(2)[ii + 1];
		float int1 = intensity_.ptr(2)[ii];
		float int2 = intensity_.ptr(2)[ii + 1];
		color_bgr.x = interpolate<float>(radian_bgr.x, rad1, rad2, int1, int2);
	}
	else{
		color_bgr.x = uchar(255);
	}

	//green
	ii = __float2int_rz((__logf(radian_bgr.y) - __ln_sample[3]) / __ln_sample[2]);
	if (ii < 0){
		color_bgr.y = uchar(0);
	}
	else if (ii < 254){
		float rad1 = rad_resample_.ptr(1)[ii];
		float rad2 = rad_resample_.ptr(1)[ii + 1];
		float int1 = intensity_.ptr(1)[ii];
		float int2 = intensity_.ptr(1)[ii + 1];
		color_bgr.y = interpolate<float>(radian_bgr.y, rad1, rad2, int1, int2);
	}
	else{
		color_bgr.y = uchar(255);
	}

	//red
	ii = __float2int_rz((__logf(radian_bgr.z) - __ln_sample[1]) / __ln_sample[0]);
	if (ii < 0){
		color_bgr.z = uchar(0);
	}
	else if (ii < 254){
		float rad1 = rad_resample_.ptr(0)[ii];
		float rad2 = rad_resample_.ptr(0)[ii + 1];
		float int1 = intensity_.ptr(0)[ii];
		float int2 = intensity_.ptr(0)[ii + 1];
		color_bgr.z = interpolate<float>(radian_bgr.z, rad1, rad2, int1, int2);
	}
	else{
		color_bgr.z = uchar(255);
	}

	bgr_.ptr(nY)[nX] = color_bgr;
	return;
}

template <typename T>
__global__ void kernel_est_dt(PtrStepSz<T> mean1_, PtrStepSz<T> mean2_, PtrStepSz<T> dt_) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX < 0 || nX >= mean1_.cols || nY < 0 || nY >= mean1_.rows) return;
	dt_.ptr(nY)[nX] = (mean1_.ptr(nY)[nX] / mean2_.ptr(nY)[nX]); //normalized intensity
	//printf("%f, %f : %f \n", x, y, anti_normalize_with_bilinear_intr<T>(ni, mean_, std_, x, y));
	return;
}

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
		B = (vec_y_.ptr(0)[nX] - vec_y_.ptr(0)[nX - 1]) / (vec_x_.ptr(0)[nX] - vec_x_.ptr(0)[nX - 1]); dydx_.ptr(0)[nX] = B > 0 ? B : 1e-30;
		B = (vec_y_.ptr(1)[nX] - vec_y_.ptr(1)[nX - 1]) / (vec_x_.ptr(1)[nX] - vec_x_.ptr(1)[nX - 1]); dydx_.ptr(1)[nX] = B > 0 ? B : 1e-30;
		B = (vec_y_.ptr(2)[nX] - vec_y_.ptr(2)[nX - 1]) / (vec_x_.ptr(2)[nX] - vec_x_.ptr(2)[nX - 1]); dydx_.ptr(2)[nX] = B > 0 ? B : 1e-30;
	}
	return;
}

btl::device::CIntrinsicsAnalysisSingle::CIntrinsicsAnalysisSingle(const string& pathFileName, int rows, int cols)
{
	_nRows = rows;
	_nCols = cols;
	_vRadianceBGR.push_back(GpuMat()); _vRadianceBGR[0].create(_nRows, _nCols, CV_32FC1);
	_vRadianceBGR.push_back(GpuMat()); _vRadianceBGR[1].create(_nRows, _nCols, CV_32FC1);
	_vRadianceBGR.push_back(GpuMat()); _vRadianceBGR[2].create(_nRows, _nCols, CV_32FC1);

	_normalized_bgr[0].create(_nRows, _nCols, CV_32FC1);
	_normalized_bgr[1].create(_nRows, _nCols, CV_32FC1);
	_normalized_bgr[2].create(_nRows, _nCols, CV_32FC1);

	_error_bgr[0].create(_nRows, _nCols, CV_8UC1);
	_error_bgr[1].create(_nRows, _nCols, CV_8UC1);
	_error_bgr[2].create(_nRows, _nCols, CV_8UC1);

	loadInvCRF(pathFileName);
}

btl::device::CIntrinsicsAnalysisMult::CIntrinsicsAnalysisMult(const string& pathFileName, int rows, int cols, float ratio, int win_radius)
	:CIntrinsicsAnalysisSingle(pathFileName, rows, cols){
	_ratio = ratio;
	_patch_radius = short(win_radius * _ratio + .5f);
	_patch_radius = _patch_radius < 1.f ? 1.f : _patch_radius;
	_thumb.create(_nRows*_ratio, _nCols*_ratio, CV_32FC1);
	_mean[0].create(_nRows*_ratio, _nCols*_ratio, CV_32FC1);
	_mean[1].create(_nRows*_ratio, _nCols*_ratio, CV_32FC1);
	_mean[2].create(_nRows*_ratio, _nCols*_ratio, CV_32FC1);
	_std[0].create(_nRows*_ratio, _nCols*_ratio, CV_32FC1);
	_std[1].create(_nRows*_ratio, _nCols*_ratio, CV_32FC1);
	_std[2].create(_nRows*_ratio, _nCols*_ratio, CV_32FC1);
}

void btl::device::CIntrinsicsAnalysisSingle::calcDeriv(){

	_derivCRF = _inten.clone();
	dim3 block(256, 1);
	dim3 grid(1, 1);
	kernel_calc_deriv<float> << < grid, block >> >(_rad_re, _inten, _derivCRF);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}

void btl::device::CIntrinsicsAnalysisSingle::loadInvCRF(const string& pathFileName){
	Mat iv_crf, exp_crf;

	FileStorage fs(pathFileName, FileStorage::READ);
	fs["inv_Cam_Response_Func"] >> iv_crf;
	iv_crf = iv_crf.t();//transpose
	_inv_crf.upload(iv_crf);

	Mat rad_re, inten;
	fs["radiance_Resample"] >> rad_re;
	_rad_re.upload(rad_re);
	fs["intensity"] >> inten;
	_inten.upload(inten);
	fs["ln_sample"] >> _ln_sample;
	fs["noise_Level_Func"] >> _nlf_sqr;
	for (int i = 0; i < _nlf_sqr.cols; i++)
		_nlf_sqr.ptr<float>()[i] *= _nlf_sqr.ptr<float>()[i];
	fs["normalize_factor"] >> _normalize_factor;
	calcDeriv();

	return;
}

void btl::device::CIntrinsicsAnalysisSingle::analysis(const GpuMat& Yuv_)
{
	cudaSafeCall(cudaMemcpyToSymbol(__nlf_sqr, _nlf_sqr.data, _nlf_sqr.cols * sizeof(float)));

	cuda::split(Yuv_, _vRadianceBGR);

	Mat sum0, sum2;
	cv::integral(Mat(_vRadianceBGR[0]), sum0, sum2);
	GpuMat sum_img(sum0), sum_sqr(sum2);
	assert(sum_img.type() == CV_32SC1);
	assert(sum_sqr.type() == CV_64FC1);

	_dM[0] = sum0.ptr<int>(sum0.rows - 1)[sum0.cols - 1] / double(_nRows*_nCols);
	_dS[0] = sqrt(sum2.ptr<double>(sum2.rows - 1)[sum2.cols - 1] / double(_nRows*_nCols) - _dM[0] * _dM[0]);

	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(_nCols, block.x), cv::cuda::device::divUp(_nRows, block.y));
	kernel_normalize<float> << < grid, block >> >(_vRadianceBGR[0], _dM[0], _dS[0], _normalized_bgr[0]);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

void btl::device::CIntrinsicsAnalysisMult::normalize(const GpuMat& radiance_, short ch)
{
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(radiance_.cols, block.x), cv::cuda::device::divUp(radiance_.rows, block.y));
	kernel_normalize_intr<float> << < grid, block >> >(radiance_, _mean[ch], _std[ch], _normalized_bgr[ch], _ratio);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
}

void btl::device::CIntrinsicsAnalysisMult::calc_mean_std(const GpuMat& radiance_, short ch)
{
	cuda::resize(radiance_, _thumb, Size(), _ratio, _ratio, INTER_LINEAR);
	assert(_thumb.type() == CV_32FC1);

	Mat sum0, sum2;
	cv::integral(Mat(_thumb), sum0, sum2);

	GpuMat sum_img(sum0), sum_sqr(sum2);

	assert(sum_sqr.type() == CV_64FC1);

	_dM[ch] = sum0.ptr<double>(sum0.rows - 1)[sum0.cols - 1] / double(_thumb.rows*_thumb.cols);
	_dS[ch] = sqrt(sum2.ptr<double>(sum2.rows - 1)[sum2.cols - 1] / double(_thumb.rows*_thumb.cols) - _dM[ch] * _dM[ch]);


	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(sum_img.cols, block.x), cv::cuda::device::divUp(sum_img.rows, block.y));

	//calc mean map
	kernel_avg<float> << < grid, block >> >(sum_img, _mean[ch], _patch_radius);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	//calc std map
	kernel_std<float> << < grid, block >> >(sum_sqr, _mean[ch], _std[ch], _patch_radius);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}

void btl::device::CIntrinsicsAnalysisMult::analysis(const GpuMat& bgr_)
{
	cudaSafeCall(cudaMemcpyToSymbol(__nlf_sqr, _nlf_sqr.data, _nlf_sqr.cols * sizeof(float)));
	cudaSafeCall(cudaMemcpyToSymbol(__normalize_factor, _normalize_factor.data, _normalize_factor.cols * sizeof(float)));
	if (false){
		//test the ratio between g/b and g/r
		GpuMat temp[3];
		cuda::split(bgr_, temp);
		double b = cuda::sum(temp[0])[0];
		double g = cuda::sum(temp[1])[0];
		double r = cuda::sum(temp[2])[0];
		cout << b << "\t" << g / b << endl;
		cout << g << "\t" << g / g << endl;
		cout << r << "\t" << g / r << endl;
	}

	GpuMat aBGR[3];
	cuda::split(bgr_, aBGR);
	for (int ch = 0; ch < _vRadianceBGR.size(); ch++)
	{
		dim3 block(32, 32);
		dim3 grid(cv::cuda::device::divUp(bgr_.cols, block.x), cv::cuda::device::divUp(bgr_.rows, block.y));

		//convert intensity back to radiance
		kernel_invcrf_rgb2rad_with_ccf << < grid, block >> >(aBGR[ch], _inv_crf.row(ch), _derivCRF.row(ch), ch, _vRadianceBGR[ch], _error_bgr[ch]);
		//cudaSafeCall(cudaGetLastError());
		//cudaSafeCall(cudaDeviceSynchronize());

		calc_mean_std(_vRadianceBGR[ch], ch);
		//normalize the radiance
		normalize(_vRadianceBGR[ch], ch);
	}

	if (false){
		dim3 block(32, 32);
		dim3 grid(cv::cuda::device::divUp(bgr_.cols, block.x), cv::cuda::device::divUp(bgr_.rows, block.y));

		GpuMat rad2[3];
		for (int ch = 0; ch < 3; ch++)
		{
			rad2[ch] = _vRadianceBGR[ch].clone(); rad2[ch].setTo(0);
			kernel_anti_normalize_intr<float> << < grid, block >> >(_normalized_bgr[ch], _mean[ch], _std[ch], rad2[ch], _ratio);
			//cudaSafeCall(cudaGetLastError());
			//cudaSafeCall(cudaDeviceSynchronize());

			GpuMat temp;
			cuda::absdiff(rad2[ch], _vRadianceBGR[ch], temp);
			cout << cuda::sum(temp)[0] << endl;
		}
		GpuMat radrec;
		cuda::merge(rad2, 3, radrec);
		GpuMat bgr_out; bgr_out.create(radrec.size(), CV_8UC3);
		kernel_crf_rad2rgb<float3> << < grid, block >> >(radrec, _rad_re, _inten, bgr_out);
		//cudaSafeCall(cudaGetLastError());
		//cudaSafeCall(cudaDeviceSynchronize());
		string total = "3_color_corrected.png";
		imwrite(total, Mat(bgr_out));
	}


	return;
}

void btl::device::CIntrinsicsAnalysisSingle::apply(CIntrinsicsAnalysisSingle& ia_)
{
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(_nCols, block.x), cv::cuda::device::divUp(_nRows, block.y));
	kernel_anti_normalize<float> << < grid, block >> >(_normalized_bgr[0], ia_._dM[0], ia_._dS[0], _vRadianceBGR[0]);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	ia_._vRadianceBGR[1].copyTo(_vRadianceBGR[1]);
	ia_._vRadianceBGR[2].copyTo(_vRadianceBGR[2]);
}

void btl::device::CIntrinsicsAnalysisSingle::cRF(const GpuMat& radiance_, GpuMat *pRGB_)
{
	cudaSafeCall(cudaMemcpyToSymbol(__ln_sample, _ln_sample.data, _ln_sample.cols * sizeof(float)));
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(radiance_.cols, block.x), cv::cuda::device::divUp(radiance_.rows, block.y));
	pRGB_->create(radiance_.size(), CV_8UC3);
	kernel_crf_rad2rgb<float3> << < grid, block >> >(radiance_, _rad_re, _inten, *pRGB_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

void btl::device::CIntrinsicsAnalysisSingle::cRFNoisy(const GpuMat& radiance_, GpuMat *pRGB_)
{
	float* sigma_squared = (float*)_nlf_sqr.data;
	cv::Mat mean = cv::Mat::zeros(1, 1, CV_64FC1);
	cv::Mat ss_b_(1, 1, CV_64FC1); ss_b_.setTo(sqrt(sigma_squared[0]));
	cv::Mat ss_g_(1, 1, CV_64FC1); ss_g_.setTo(sqrt(sigma_squared[2]));
	cv::Mat ss_r_(1, 1, CV_64FC1); ss_r_.setTo(sqrt(sigma_squared[4]));
	cv::Mat sc_b_(1, 1, CV_64FC1); sc_b_.setTo(sqrt(sigma_squared[1]));
	cv::Mat sc_g_(1, 1, CV_64FC1); sc_g_.setTo(sqrt(sigma_squared[3]));
	cv::Mat sc_r_(1, 1, CV_64FC1); sc_r_.setTo(sqrt(sigma_squared[5]));

	Mat sc_b_matrix(radiance_.size(), CV_64FC1);
	Mat sc_g_matrix(radiance_.size(), CV_64FC1);
	Mat sc_r_matrix(radiance_.size(), CV_64FC1);
	Mat ss_b_matrix(radiance_.size(), CV_64FC1);
	Mat ss_g_matrix(radiance_.size(), CV_64FC1);
	Mat ss_r_matrix(radiance_.size(), CV_64FC1);

	cv::randn(sc_b_matrix, mean, sc_b_);
	cv::randn(sc_g_matrix, mean, sc_g_);
	cv::randn(sc_r_matrix, mean, sc_r_);
	cv::randn(ss_b_matrix, mean, ss_b_);
	cv::randn(ss_g_matrix, mean, ss_g_);
	cv::randn(ss_r_matrix, mean, ss_r_);

	GpuMat gpu_sc_b_matrix(sc_b_matrix);
	GpuMat gpu_sc_g_matrix(sc_g_matrix);
	GpuMat gpu_sc_r_matrix(sc_r_matrix);
	GpuMat gpu_ss_b_matrix(ss_b_matrix);
	GpuMat gpu_ss_g_matrix(ss_g_matrix);
	GpuMat gpu_ss_r_matrix(ss_r_matrix);

	cudaSafeCall(cudaMemcpyToSymbol(__ln_sample, _ln_sample.data, _ln_sample.cols * sizeof(float)));
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(radiance_.cols, block.x), cv::cuda::device::divUp(radiance_.rows, block.y));
	pRGB_->create(radiance_.size(), CV_8UC3);
	kernel_crf_rad2rgb_with_noise<float3> << < grid, block >> >(radiance_, _derivCRF, _rad_re, _inten,
		gpu_sc_b_matrix,
		gpu_sc_g_matrix,
		gpu_sc_r_matrix,
		gpu_ss_b_matrix,
		gpu_ss_g_matrix,
		gpu_ss_r_matrix,
		*pRGB_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

void btl::device::CIntrinsicsAnalysisMult::apply(CIntrinsicsAnalysisMult& ia_ref_)
{
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(_nCols, block.x), cv::cuda::device::divUp(_nRows, block.y));
	//calc delta t
	GpuMat dtAll; dtAll.create(_mean[0].size(), CV_32FC1);
	GpuMat new_std;

	double dt = 0;

	if (false){
		GpuMat tmp0 = _normalized_bgr[0].clone();
		GpuMat tmp1 = _normalized_bgr[1].clone();
		GpuMat tmp2 = _normalized_bgr[2].clone();
		//denoising
		kernel_denoising<double> << < grid, block >> >(_normalized_bgr[0], _normalized_bgr[1], _normalized_bgr[2], _error_bgr[0], _error_bgr[1], _error_bgr[2], tmp0, tmp1, tmp2);
		//cudaSafeCall(cudaGetLastError());
		//cudaSafeCall(cudaDeviceSynchronize());
		tmp0.copyTo(_normalized_bgr[0]);
		tmp1.copyTo(_normalized_bgr[1]);
		tmp2.copyTo(_normalized_bgr[2]);
	}

	for (int ch = 0; ch < 3; ch++)
	{
		kernel_est_dt<float> << < grid, block >> >(ia_ref_._mean[ch], _mean[ch], dtAll); //ref dark //live bright dt < 1
		//cudaSafeCall(cudaGetLastError());
		//cudaSafeCall(cudaDeviceSynchronize());

		dt = cuda::sum(dtAll)[0] / _mean[ch].rows / _mean[ch].cols;
		_std[ch].convertTo(new_std, _std[ch].type(), dt);

		kernel_anti_normalize_intr<float> << < grid, block >> >(_normalized_bgr[ch], ia_ref_._mean[ch], new_std, _vRadianceBGR[ch], _ratio);
		//cudaSafeCall(cudaGetLastError());
		//cudaSafeCall(cudaDeviceSynchronize());
		//GpuMat temp;
		//cuda::absdiff(_vRadianceBGR[ch], ia_ref_._vRadianceBGR[ch], temp);
		//cout << Mat(temp) << endl;
		//cout << cuda::sum(temp)[0] << endl;
		if (false){
			cout << Mat(_vRadianceBGR[ch].colRange(100, 110).rowRange(100, 110)) << endl;
		}
	}

	if (false){
		cudaSafeCall(cudaMemcpyToSymbol(__ln_sample, _ln_sample.data, _ln_sample.cols * sizeof(float)));
		GpuMat rad2[3];

		GpuMat radrec;
		cuda::merge(_vRadianceBGR, radrec);
		GpuMat bgr_out; bgr_out.create(radrec.size(), CV_8UC3);
		kernel_crf_rad2rgb<float3> << < grid, block >> >(radrec, _rad_re, _inten, bgr_out);
		//cudaSafeCall(cudaGetLastError());
		//cudaSafeCall(cudaDeviceSynchronize());

		string total = "4_color_corrected.png";
		imwrite(total, Mat(bgr_out));
	}

	return;
}

void btl::device::CIntrinsicsAnalysisSingle::store(string& fileName)
{
	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(_nCols, block.x), cv::cuda::device::divUp(_nRows, block.y));
	if (true){
		GpuMat norm_ch[3];
		_normalized_bgr[0].convertTo(norm_ch[0], CV_8UC1, 128., 128.);
		_normalized_bgr[1].convertTo(norm_ch[1], CV_8UC1, 128., 128.);
		_normalized_bgr[2].convertTo(norm_ch[2], CV_8UC1, 128., 128.);

		GpuMat normalized, norm_double;
		cuda::merge(norm_ch, 3, normalized);
		string total = fileName + "_gray_normalized.png";
		imwrite(total, Mat(normalized));
	}

	{
		GpuMat nC;
		cuda::merge(_vRadianceBGR, nC);
		GpuMat rgb_out; rgb_out.create(nC.size(), CV_8UC3);
		if (false){
			kernel_to_intensity<float> << < grid, block >> >(nC, _exp_crf, rgb_out);
			//cudaSafeCall(cudaGetLastError());
			//cudaSafeCall(cudaDeviceSynchronize());
		}
		cRF(nC, &rgb_out);

		string total = fileName + "_color_corrected.png";
		imwrite(total, Mat(rgb_out));
	}

	return;
}

void btl::device::CIntrinsicsAnalysisMult::store(string& fileName)
{
	CIntrinsicsAnalysisSingle::store(fileName);

	dim3 block(32, 32);
	dim3 grid(cv::cuda::device::divUp(_nCols, block.x), cv::cuda::device::divUp(_nRows, block.y));
	if (true){
		GpuMat mean, rgb_out;;
		cuda::merge(_mean, 3, mean);
		cRF(mean, &rgb_out);
		string total = fileName + "_mean.png";
		imwrite(total, Mat(rgb_out));
	}

	if (true){
		GpuMat std, rgb_out;;
		cuda::merge(_std, 3, std);
		cRF(std, &rgb_out);
		string total = fileName + "_std.png";
		imwrite(total, Mat(rgb_out));
	}

	if (true){
		GpuMat err[3], total;
		cuda::merge(_error_bgr, 3, total);
		string pathFN = fileName + "_err.png";
		imwrite(pathFN, Mat(total));
	}

	return;
}

void btl::device::CIntrinsicsAnalysisSingle::copyTo(CIntrinsicsAnalysisSingle& ia_)
{
	for (int ch = 0; ch < 3; ch++)
	{
		_vRadianceBGR[ch].copyTo(ia_._vRadianceBGR[ch]);
		ia_._dM[ch] = _dM[ch];
		ia_._dS[ch] = _dS[ch];
		_normalized_bgr[ch].copyTo(ia_._normalized_bgr[ch]);
		_error_bgr[ch].copyTo(ia_._error_bgr[ch]);
	}

	ia_._nRows = _nRows;
	ia_._nCols = _nCols;
	return;
}

void btl::device::CIntrinsicsAnalysisMult::copyTo(CIntrinsicsAnalysisMult& ia_)
{
	CIntrinsicsAnalysisSingle::copyTo(ia_);

	for (int ch = 0; ch < 3; ch++)
	{
		_mean[ch].copyTo(ia_._mean[ch]);
		_std[ch].copyTo(ia_._std[ch]);
	}
	_thumb.copyTo(ia_._thumb);

	//ia_._area = _area;
	ia_._patch_radius = _patch_radius;
	ia_._ratio = _ratio;

	return;
}

void btl::device::CIntrinsicsAnalysisSingle::clear()
{
	for (int ch = 0; ch < 3; ch++)
	{
		_vRadianceBGR[ch].setTo(0);
		_normalized_bgr[ch].setTo(0);
		_error_bgr[ch].setTo(0);
	}
	return;
}

void btl::device::CIntrinsicsAnalysisMult::clear()
{
	CIntrinsicsAnalysisSingle::clear();
	for (int ch = 0; ch < 3; ch++)
	{
		_mean[ch].setTo(0);
		_std[ch].setTo(0);
	}
	_thumb.setTo(0.);
	return;
}