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
#include <opencv2/core/cuda/common.hpp>

#include <iostream>

#include "pcl/vector_math.hpp"
#include "pcl/device.hpp"
#include "VolumeColor.cuh"

using namespace pcl::device;
using namespace cv::cuda;
using namespace cv;
using namespace std;

namespace btl{
	namespace device
	{
		using namespace pcl::device;
		struct FuseDepthColorParam {
			short __WEIGHT;
			short __MAX_WEIGHT;
			short __MAX_WEIGHT_SHORT;
			float __epsilon;
			float __trunc_dist;
			float __trunc_dist_inv;
			Intr __intr;
			Matr33d __R_cf;
			double3 __Of;
			float __voxel_size;
			float __inv_voxel_size;
			float __half_voxel_size;
			short3 __resolution;
			float __voxel_size_ms;
			short3 __resolution_ms;
			short3 __resolution_m_2;
			ushort __uTSDFLevelXY;
		};

		__device__ __forceinline__ float3 getYUV(short4* pos_){
			float3 YUV;
			YUV.x = *(const float*)pos_;
			if (YUV.x == YUV.x)
			{
				YUV.y = ((const uchar*)pos_)[4];
				YUV.z = ((const uchar*)pos_)[5];
			}
			else{
				YUV.x = YUV.y = YUV.z = 255;
			}
			return YUV;
		}
		__device__ __forceinline__ void setYUV(const float3& YUV_, short4* pos_){
			*(float*)pos_ = YUV_.x;
			((uchar*)pos_)[4] = uchar(YUV_.y + .5f);
			((uchar*)pos_)[5] = uchar(YUV_.z + .5f);
		}

		__constant__ FuseDepthColorParam __param;

		inline __host__ __device__ float _lerp(float s, float e, float t)
		{
			return s + (e - s)*t;
		}

		inline __host__ __device__ float _blerp(float c00, float c10, float c01, float c11, float tx, float ty)
		{
			return _lerp(_lerp(c00, c10, tx), _lerp(c01, c11, tx), ty);
		}

		inline __host__ __device__ float bilinearInterpolateValue(
			const PtrStepSz<float>& array,
			float x_ind,
			float y_ind)
		{
			const int ix = floor(x_ind);
			const int iy = floor(y_ind);
			const float fx = x_ind - ix;
			const float fy = y_ind - iy;

			if (ix >= 0 && ix < array.cols - 1 &&
				iy >= 0 && iy < array.rows - 1)
			{
				///            uint32_t c00 = getpixel(src, gxi, gyi);
				const float c00 = array.ptr(iy)[ix + 0]; if (c00 != c00) return NAN;

				///            uint32_t c10 = getpixel(src, gxi+1, gyi);
				const float c10 = array.ptr(iy)[ix + 1]; if (c10 != c10) return NAN;

				///            uint32_t c01 = getpixel(src, gxi, gyi+1);
				const float c01 = array.ptr(iy + 1)[ix + 0]; if (c01 != c01) return NAN;

				///            uint32_t c11 = getpixel(src, gxi+1, gyi+1);
				const float c11 = array.ptr(iy + 1)[ix + 1]; if (c11 != c11) return NAN;

				if (fabs(c00 - c10) < 0.15f&& fabs(c00 - c01) < 0.15f&& fabs(c00 - c11) < 0.15f){
					return _blerp(c00, c10, c01, c11, fx, fy);
				}
			}

			return NAN;
		}
		__global__ void kernel_fuse_depth_radiance_normal(const PtrStepSz<float> scaled_depth_, PtrStepSz<float3> surface_normal_, PtrStepSz<float3> normal_radiance_, PtrStepSz<float3> radiance_, PtrStepSz<uchar3> err_,
			PtrStepSz<short2> tsdf_volume_, PtrStepSz<float4> normal_radiance_volume_, PtrStepSz<float4> radiance_volume_)
		{
			short x = threadIdx.x + blockIdx.x * blockDim.x;
			short y = threadIdx.y + blockIdx.y * blockDim.y;
			//the organization of the volume is described in the following links
			//https://docs.google.com/drawings/d/1lkw9jnNsVifIc42aDCtqnMEqc53FAlJsEIk3VBMfCN0/edit
			//The x index is major, then y index, z index is the last.
			//Thus, voxel (x,y,z) is stored starting at location
			//( x + y*resolution_x + z*resolution_x*resolution_y ) * (bitpix/8)

			if (x >= __param.__resolution.x || y >= __param.__resolution.y)    return;

			float3 v_g;
			v_g.x = (x + 0.5f) * __param.__voxel_size - __param.__Of.x; // vw - Cw: voxel center in the world and camera center in the world
			v_g.y = (y + 0.5f) * __param.__voxel_size - __param.__Of.y;
			v_g.z = (0 + 0.5f) * __param.__voxel_size - __param.__Of.z;

			float v_g_part_norm = v_g.x * v_g.x + v_g.y * v_g.y; // get the distance from the voxel to the camera center

			float3 v_c;
			v_c = __param.__R_cf * v_g;
			v_c.x *= __param.__intr.fx;
			v_c.y *= __param.__intr.fy;

			float z_scaled = 0;

			float Rcurr_inv_0_z_scaled = __param.__R_cf.c[2].x * __param.__voxel_size * __param.__intr.fx;
			float Rcurr_inv_1_z_scaled = __param.__R_cf.c[2].y * __param.__voxel_size * __param.__intr.fy;

			short2* tsdf_pos = tsdf_volume_.ptr(y * __param.__resolution.x + x);
			float4* nr_pos = normal_radiance_volume_.ptr(y * __param.__resolution.x + x);
			float4* ra_pos = radiance_volume_.ptr(y * __param.__resolution.x + x);

#pragma unroll
			for (short z = 0; z < __param.__resolution.z;
				++z,
				v_g.z += __param.__voxel_size,
				z_scaled += __param.__voxel_size,
				v_c.x += Rcurr_inv_0_z_scaled,
				v_c.y += Rcurr_inv_1_z_scaled,
				tsdf_pos++,
				nr_pos++,
				ra_pos++)
			{
				float inv_z = 1.0f / (v_c.z + __param.__R_cf.c[2].z * z_scaled);
				if (inv_z < 0) continue;

				// project to current cam
				float2 fcoo = { v_c.x * inv_z, v_c.y * inv_z };
				//int2 coo = {
				//	__float2int_rd(fcoo.x + __param.__intr.cx + .5f),
				//	__float2int_rd(fcoo.y + __param.__intr.cy + .5f)
				//};
				x = __float2int_rd(fcoo.x + __param.__intr.cx + .5f);
				y = __float2int_rd(fcoo.y + __param.__intr.cy + .5f);
				if (x > 0 && y > 0 && x < scaled_depth_.cols - 1 && y < scaled_depth_.rows - 1)         //6
				{
					float /*normv*/cosa = __fsqrt_rd(fcoo.x*fcoo.x + fcoo.y*fcoo.y + __param.__intr.fx * __param.__intr.fx); //length of visual ray
					float3 nl = surface_normal_.ptr(y)[x];			if (nl.x != nl.x || nl.y != nl.y || nl.z != nl.z) continue;
					cosa = nl.x * fcoo.x / /*normv*/cosa + nl.y * fcoo.y / /*normv*/cosa + nl.z * __param.__intr.fx / /*normv*/cosa;
					/*float Dp_scaled*/inv_z = bilinearInterpolateValue(scaled_depth_, fcoo.x + __param.__intr.cx, fcoo.y + __param.__intr.cy); if (/*Dp_scaled*/inv_z != /*Dp_scaled*/inv_z) continue;
					float tsdf_curr = /*Dp_scaled*/inv_z - __fsqrt_rd(v_g.z * v_g.z + v_g_part_norm); //__fsqrt_rd sqrtf
					//float Dp_scaled = scaled_depth_.ptr(coo.y)[coo.x]; //meters
					if (/*Dp_scaled*/inv_z > 0.1f && tsdf_curr >= -__param.__trunc_dist)//&& tsdf_curr <= __param.__trunc_dist && cosa < -0.2f && cosa > -1.01f) //meters//
					{
						tsdf_curr *= __param.__trunc_dist_inv; tsdf_curr = tsdf_curr > 1.f ? 1.f : tsdf_curr; // tsdf_curr \in [-1,1]

						//read and unpack
						short weight_prev; float tsdf_prev;
						unpack_tsdf(*tsdf_pos, tsdf_prev, weight_prev);
						//linear weight
						short weight_curr = short(__param.__WEIGHT*(-cosa)*fmax(1.f, 1.f + tsdf_curr) + .5f);
						tsdf_curr *= weight_curr;
						tsdf_curr += weight_prev * tsdf_prev;
						tsdf_curr /= (weight_prev + weight_curr);

						uchar3 er = err_.ptr(y)[x];
						/*uchar min_e*/fcoo.x = __min(__min(er.x, er.y), er.z); 
						/*uchar max_e*/fcoo.y = __max(__max(er.x, er.y), er.z);
						if (/*min_e*/fcoo.x > 0.5f || /*max_e*/fcoo.y > 60) {
							float rw_c = (er.x + er.y + er.z) / 3.f;
							rw_c = rw_c >= 1 ? rw_c : 1;
							rw_c *= (-cosa);
							/*float3 nor_rad_curr*/ nl = normal_radiance_.ptr(y)[x];
							float3 rad_curr = radiance_.ptr(y)[x];
							if (rad_curr.x == rad_curr.x && rad_curr.y == rad_curr.y && rad_curr.z == rad_curr.z){
								float4 no_rad_prev = *nr_pos;
								if (no_rad_prev.x != no_rad_prev.x){
									no_rad_prev.x = /*nor_rad_curr*/ nl.x;
									no_rad_prev.y = /*nor_rad_curr*/ nl.y;
									no_rad_prev.z = /*nor_rad_curr*/ nl.z;
									no_rad_prev.w = rw_c;
									*nr_pos = no_rad_prev;
								}
								else{
									no_rad_prev.x = (no_rad_prev.x* no_rad_prev.w + /*nor_rad_curr*/ nl.x * rw_c);
									no_rad_prev.y = (no_rad_prev.y* no_rad_prev.w + /*nor_rad_curr*/ nl.y * rw_c);
									no_rad_prev.z = (no_rad_prev.z* no_rad_prev.w + /*nor_rad_curr*/ nl.z * rw_c);
									no_rad_prev.w += rw_c;

									no_rad_prev.x /= no_rad_prev.w;
									no_rad_prev.y /= no_rad_prev.w;
									no_rad_prev.z /= no_rad_prev.w;
									*nr_pos = no_rad_prev;
								}
								no_rad_prev = *ra_pos;
								if (no_rad_prev.x != no_rad_prev.x){
									no_rad_prev.x = rad_curr.x;
									no_rad_prev.y = rad_curr.y;
									no_rad_prev.z = rad_curr.z;
									no_rad_prev.w = rw_c;
									*ra_pos = no_rad_prev;
								}
								else{
									no_rad_prev.x = (no_rad_prev.x* no_rad_prev.w + rad_curr.x * rw_c);
									no_rad_prev.y = (no_rad_prev.y* no_rad_prev.w + rad_curr.y * rw_c);
									no_rad_prev.z = (no_rad_prev.z* no_rad_prev.w + rad_curr.z * rw_c);
									no_rad_prev.w += rw_c;

									no_rad_prev.x /= no_rad_prev.w;
									no_rad_prev.y /= no_rad_prev.w;
									no_rad_prev.z /= no_rad_prev.w;
									*ra_pos = no_rad_prev;
								}//else
							}//rad_curr is valid
						}//radiance is confident

						weight_curr += weight_prev; weight_curr = weight_curr > __param.__MAX_WEIGHT ? __param.__MAX_WEIGHT : weight_curr;
						pack_tsdf(tsdf_curr, weight_curr, *tsdf_pos);
					}
				}
			}       // for(int z = 0; z < VOLUME_Z; ++z)
		}      // __global__

		void cuda_fuse_depth_radiance_normal(const GpuMat& scaled_depth_, const GpuMat& surface_normal_, const GpuMat& normal_radiance_, const GpuMat& radiance_, const GpuMat& err_,
			const float fVoxelSize_, const float fTruncDistanceM_,
			const Matr33d& R_cf_, const double3& Of_,
			const Intr& intr, const short3& resolution_,
			GpuMat* ptr_tsdf_volume_, GpuMat* ptr_normali_radiance_volume_, GpuMat* ptr_radiance_volume_)
		{
			FuseDepthColorParam param;
			param.__WEIGHT = 1 << 9; //512
			param.__MAX_WEIGHT = 1 << 14; //16384
			param.__voxel_size = fVoxelSize_;
			param.__epsilon = fVoxelSize_;
			param.__intr = intr;
			param.__R_cf = R_cf_;
			param.__Of = Of_;
			param.__trunc_dist = fTruncDistanceM_;
			param.__trunc_dist_inv = 1.f / fTruncDistanceM_;
			param.__resolution = resolution_;
			cudaSafeCall(cudaMemcpyToSymbol(__param, &param, sizeof(FuseDepthColorParam))); //copy host memory to constant memory on the device.

			dim3 block(16, 16);
			dim3 grid(cv::cuda::device::divUp(resolution_.x, block.x), cv::cuda::device::divUp(resolution_.y, block.y));

			kernel_fuse_depth_radiance_normal << <grid, block >> >(scaled_depth_, surface_normal_, normal_radiance_, radiance_, err_,
				*ptr_tsdf_volume_, *ptr_normali_radiance_volume_, *ptr_radiance_volume_);
			//cudaSafeCall(cudaGetLastError());
			//cudaSafeCall(cudaDeviceSynchronize());

			return;
		}

	}//device
}//btl


