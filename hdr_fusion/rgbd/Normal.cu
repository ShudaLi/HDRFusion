#define EXPORT
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/cudaarithm.hpp>
#include <math_constants.h>
#include <opencv2/core/cuda/common.hpp>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "Normal.cuh"

namespace btl{ namespace device
{
using namespace pcl::device;
using namespace cv;
using namespace cv::cuda;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


enum
{
	kx = 5,
	ky = 5,
	STEP = 1
};

__global__ void computeNmapKernelEigen(int rows, int cols, const PtrStep<float3> vmap, PtrStep<float3> nmap)
{
	const int u = threadIdx.x + blockIdx.x * blockDim.x;
	const int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)	return;

	nmap.ptr(v)[u].x = pcl::device::numeric_limits<float>::quiet_NaN();

	float3 vt = vmap.ptr(v)[u];
	if (isnan(vt.x))
		return;

	int ty = min(v - ky / 2 + ky, rows - 1);
	int tx = min(u - kx / 2 + kx, cols - 1);

	float3 centroid = make_float3(0.f, 0.f, 0.f);
	int counter = 0;
	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v_x = vmap.ptr(cy)[cx];
			if (!isnan(v_x.x))
			{
				centroid = centroid + v_x;
				++counter;
			}
		}

	if (counter < kx * ky / 2)
		return;

	centroid *= 1.f / counter;

	float cov[] = { 0, 0, 0, 0, 0, 0 };

	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v;
			v = vmap.ptr(cy)[cx];
			if (isnan(v.x))
				continue;

			float3 d = v - centroid;

			cov[0] += d.x * d.x;               //cov (0, 0)
			cov[1] += d.x * d.y;               //cov (0, 1)
			cov[2] += d.x * d.z;               //cov (0, 2)
			cov[3] += d.y * d.y;               //cov (1, 1)
			cov[4] += d.y * d.z;               //cov (1, 2)
			cov[5] += d.z * d.z;               //cov (2, 2)
		}

	typedef Eigen33::Mat33 Mat33;
	Eigen33 eigen33(cov);

	Mat33 tmp;
	Mat33 vec_tmp;
	Mat33 evecs;
	float3 evals;
	eigen33.compute(tmp, vec_tmp, evecs, evals);

	float3 n = normalized<float,float3>(evecs[0]);

	if( dot3<float,float3>(vt,n) <=0 ) 
		nmap.ptr(v)[u] = n;
	else
		nmap.ptr(v)[u] = make_float3(-n.x,-n.y,-n.z);
	return;
}


__global__ void computeNmapKernelEigen(int rows, int cols, const PtrStep<float3> vmap, PtrStep<float3> nmap, PtrStep<float> reliability)
{
	const int u = threadIdx.x + blockIdx.x * blockDim.x;
	const int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)	return;

	nmap.ptr(v)[u].x = pcl::device::numeric_limits<float>::quiet_NaN();
	reliability.ptr(v)[u] = 0;// pcl::device::numeric_limits<float>::quiet_NaN();

	float3 vt = vmap.ptr(v)[u];
	if (isnan(vt.x))
		return;

	int ty = min(v - ky / 2 + ky, rows - 1);
	int tx = min(u - kx / 2 + kx, cols - 1);

	float3 centroid = make_float3(0.f, 0.f, 0.f);
	int counter = 0;
	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v_x = vmap.ptr(cy)[cx];
			if (!isnan(v_x.x))
			{
				centroid = centroid + v_x;
				++counter;
			}
		}

	if (counter < kx * ky / 2)
		return;

	centroid *= 1.f / counter;

	float cov[] = { 0, 0, 0, 0, 0, 0 };

	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v;
			v = vmap.ptr(cy)[cx];
			if (isnan(v.x))
				continue;

			float3 d = v - centroid;

			cov[0] += d.x * d.x;               //cov (0, 0)
			cov[1] += d.x * d.y;               //cov (0, 1)
			cov[2] += d.x * d.z;               //cov (0, 2)
			cov[3] += d.y * d.y;               //cov (1, 1)
			cov[4] += d.y * d.z;               //cov (1, 2)
			cov[5] += d.z * d.z;               //cov (2, 2)
		}

	typedef Eigen33::Mat33 Mat33;
	Eigen33 eigen33(cov);

	Mat33 tmp;
	Mat33 vec_tmp;
	Mat33 evecs;
	float3 evals;
	eigen33.compute(tmp, vec_tmp, evecs, evals);

	float3 n = normalized<float, float3>(evecs[0]);

	if (dot3<float, float3>(vt, n) <= 0)
		nmap.ptr(v)[u] = n;
	else
		nmap.ptr(v)[u] = make_float3(-n.x, -n.y, -n.z);

	evals.z = abs(evals.z);
	evals.y = abs(evals.y);
	evals.x = abs(evals.x);

	float re = (evals.y - evals.x) / (evals.x + evals.y) ;
	re = re >= 1.f ? 1.f : re;
	reliability.ptr(v)[u] = re;

	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_normals_eigen(const GpuMat& vmap, GpuMat* nmap, GpuMat* reliability /* = NULL*/)
{
	int cols = vmap.cols;
	int rows = vmap.rows;

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = cv::cuda::device::divUp(cols, block.x);
	grid.y = cv::cuda::device::divUp(rows, block.y);

	if (!reliability)
		computeNmapKernelEigen << <grid, block >> >(rows, cols, vmap, *nmap);
	else
		computeNmapKernelEigen << <grid, block >> >(rows, cols, vmap, *nmap, *reliability);

	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}



}//device
}//btl