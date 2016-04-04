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
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>  

#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "RayCasterColor.cuh"
#include "Kinect.h"

#include "assert.h"
#define INFO
#include "OtherUtil.hpp"
#include <iostream> 
#include <limits>
namespace btl{ 
namespace device  {
	using namespace std;
	using namespace cv;
	using namespace cv::cuda;
	using namespace cv::cuda::device;
	using namespace pcl::device;

struct RayCasterColorParam {
	Matd33 __Rcf;
	Matd33 __Rcf_t;
	double3 __C_f; //Cw camera center
	float __time_step;
	float __cell_size;
	float __half_cell_size;
	float __inv_cell_size;
	float __half_inv_cell_size;
	int __rows;
	int __cols;
	short3 __resolution;
	short3 __resolution_m_1;
	short3 __resolution_m_2;
	Intr __intr;
	float3 __volume_size;
};

__constant__ RayCasterColorParam __param;

__device__ __forceinline__ float
getMinTime(const float3& dir)
{
	float txmin = ((dir.x > 0 ? 0.f : __param.__volume_size.x) - __param.__C_f.x ) / dir.x;
	float tymin = ((dir.y > 0 ? 0.f : __param.__volume_size.y) - __param.__C_f.y ) / dir.y;
	float tzmin = ((dir.z > 0 ? 0.f : __param.__volume_size.z) - __param.__C_f.z ) / dir.z;

	return fmax(fmax(txmin, tymin), tzmin);
}

__device__ __forceinline__ float
getMaxTime(const float3& dir)
{
	float txmax = ((dir.x > 0 ? __param.__volume_size.x : 0.f) - __param.__C_f.x ) / dir.x;
	float tymax = ((dir.y > 0 ? __param.__volume_size.y : 0.f) - __param.__C_f.y ) / dir.y;
	float tzmax = ((dir.z > 0 ? __param.__volume_size.z : 0.f) - __param.__C_f.z ) / dir.z;

	return fmin(fmin(txmax, tymax), tzmax);
}

struct RayCasterColor
{
	enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8, CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y };
	
	PtrStepSz<short2> _tsdf_volume;
	PtrStepSz<float4> _radiance_volume;
	PtrStepSz<float4> _normalized_randiance_volume;
	mutable PtrStepSz<float3> _nmap;
	mutable PtrStepSz<float3> _vmap;
	bool _no_rad; //no radiance
	mutable PtrStepSz<float3> _radiance; 
	mutable PtrStepSz<float3> _normalized_r;

    __device__ __forceinline__ int3 getVoxel (const float3& Xw_) const
    {
		// round to negative infinity
		return make_int3( __float2int_rd( Xw_.x  * __param.__inv_cell_size), 
						  __float2int_rd( Xw_.y  * __param.__inv_cell_size), 
						  __float2int_rd( Xw_.z  * __param.__inv_cell_size));
    }

    __device__ __forceinline__ float
    interpolateTrilinearyOrigin (const float3& dir, float time) const
    {
		return interpolateTrilineary (__param.__C_f + dir * time);
    }

    __device__ __forceinline__ float
    interpolateTrilineary (const float3& Xw_) const
    {
		float a = Xw_.x * __param.__inv_cell_size;
		float b = Xw_.y * __param.__inv_cell_size;
		float c = Xw_.z * __param.__inv_cell_size;

		int3 g = make_int3( __float2int_rd(a), //round down to negative infinity
							__float2int_rd(b), 
							__float2int_rd(c));//get voxel coordinate

		if (g.x<1 || g.y<1 || g.z<1 || g.x >__param.__resolution_m_2.x || g.y > __param.__resolution_m_2.y || g.z > __param.__resolution_m_2.z) return pcl::device::numeric_limits<float>::quiet_NaN();

		g.x = (Xw_.x < g.x * __param.__cell_size + __param.__half_cell_size) ? (g.x - 1.f) : g.x;
		g.y = (Xw_.y < g.y * __param.__cell_size + __param.__half_cell_size) ? (g.y - 1.f) : g.y;
		g.z = (Xw_.z < g.z * __param.__cell_size + __param.__half_cell_size) ? (g.z - 1.f) : g.z;

		a -= (g.x + 0.5f);
		b -= (g.y + 0.5f);
		c -= (g.z + 0.5f);
		int row = __param.__resolution.x * g.y + g.x;
		return  unpack_tsdf(_tsdf_volume.ptr(row)                         [g.z])     * (1 - a) * (1 - b) * (1 - c) +
				unpack_tsdf(_tsdf_volume.ptr(row + __param.__resolution.x)    [g.z])     * (1 - a) * b       * (1 - c) +
				unpack_tsdf(_tsdf_volume.ptr(row + 1)                     [g.z])     * a       * (1 - b) * (1 - c) +
				unpack_tsdf(_tsdf_volume.ptr(row + __param.__resolution.x + 1)[g.z])     * a       * b       * (1 - c) +
				unpack_tsdf(_tsdf_volume.ptr(row)                         [g.z + 1]) * (1 - a) * (1 - b) * c +
				unpack_tsdf(_tsdf_volume.ptr(row + __param.__resolution.x)    [g.z + 1]) * (1 - a) * b       * c +
				unpack_tsdf(_tsdf_volume.ptr(row + 1)                     [g.z + 1]) * a       * (1 - b) * c +
				unpack_tsdf(_tsdf_volume.ptr(row + __param.__resolution.x + 1)[g.z + 1]) * a       * b       * c;
    }

	__device__ __forceinline__ float3 getRadiance(const float4* pos_) const{
		return make_float3(pos_->x,pos_->y,pos_->z);
	}

	__device__ __forceinline__ void
		interpolateTrilinearyColor(const float3& Xw_, float3& total_radiance) const
	{
		float a = Xw_.x * __param.__inv_cell_size;
		float b = Xw_.y * __param.__inv_cell_size;
		float c = Xw_.z * __param.__inv_cell_size;

		int3 g = make_int3( __float2int_rd(a), //round down to negative infinity
							__float2int_rd(b),
							__float2int_rd(c));//get voxel coordinate

		if (g.x<1 || g.y<1 || g.z<1 || g.x >__param.__resolution_m_2.x || g.y > __param.__resolution_m_2.y || g.z > __param.__resolution_m_2.z) return;

		g.x = (Xw_.x < g.x * __param.__cell_size + __param.__half_cell_size) ? (g.x - 1.f) : g.x;
		g.y = (Xw_.y < g.y * __param.__cell_size + __param.__half_cell_size) ? (g.y - 1.f) : g.y;
		g.z = (Xw_.z < g.z * __param.__cell_size + __param.__half_cell_size) ? (g.z - 1.f) : g.z;

		a -= (g.x + 0.5f);
		b -= (g.y + 0.5f);
		c -= (g.z + 0.5f);
		int row = __param.__resolution.x * g.y + g.x;
		float total_weight = 0.f;
		total_radiance.x = total_radiance.y = total_radiance.z = 0.f;
		float weight;
		float3 nY;
		nY = getRadiance(_radiance_volume.ptr(row) + g.z);
		if (nY.x == nY.x && nY.y==nY.y && nY.z == nY.z){
			weight = (1 - a) * (1 - b) * (1 - c);
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x) + g.z);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = (1 - a) * b       * (1 - c);
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + 1) + g.z);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * (1 - b) * (1 - c);
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x + 1) + g.z);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * b       * (1 - c);
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = (1 - a) * (1 - b) * c;
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = (1 - a) * b       * c;
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + 1) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * (1 - b) * c;
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x + 1) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * b       * c;
			total_weight += weight;
			total_radiance += (nY * weight);
		}
		total_radiance /= total_weight;
		return;
	}


	__device__ __forceinline__ void ray_casting_radiance() const
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		//the organization of the volume is described in the following links
		//https://docs.google.com/drawings/d/1lkw9jnNsVifIc42aDCtqnMEqc53FAlJsEIk3VBMfCN0/edit?usp=sharing
		//The x index is major, then y index, and z index is the last.
		//Thus, voxel (x,y,z) is stored in 2-D matrix, 
		// row: y*resolution_x + x
		// col: z

		if (x >= __param.__cols || y >= __param.__rows) return;

		//float3 ray_dir = __Rcurr * get_ray_next(x, y);
		float3 ray_dir;
		//ray_dir = get_ray_next(x, y);
		ray_dir.x = (x - __param.__intr.cx) / __param.__intr.fx;
		ray_dir.y = (y - __param.__intr.cy) / __param.__intr.fy;
		ray_dir.z = 1.f;

		ray_dir = __param.__Rcf_t * ray_dir; //rotate from camera to tsdf coordinate f
		ray_dir *= rsqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z); //normalize ray_dir;

		// computer time when entry and exit volume
		float time_start_volume = fmaxf(getMinTime(ray_dir), 0.f);
		//float time_start_volume = fmax(fmax(fmax(((ray_dir.x > 0 ? 0.f : __volume_size.x) - __tcurr.x) / ray_dir.x, ((ray_dir.y > 0 ? 0.f : __volume_size.y) - __tcurr.y) / ray_dir.y), ((ray_dir.z > 0 ? 0.f : __volume_size.z) - __tcurr.z) / ray_dir.z), 0.f);
		const float time_end_volume = getMaxTime(ray_dir);

		if (time_start_volume >= time_end_volume){
			//vmap.ptr(y)[x] = nmap.ptr(y)[x] = make_float3(pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN());
			return;
		}

		int3 g = getVoxel(__param.__C_f + ray_dir * time_start_volume);
		if (!(g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __param.__resolution.x && g.y < __param.__resolution.y && g.z <__param.__resolution.z)){
			g.x = fmaxf(0, fminf(g.x, __param.__resolution_m_1.x));
			g.y = fmaxf(0, fminf(g.y, __param.__resolution_m_1.y));
			g.z = fmaxf(0, fminf(g.z, __param.__resolution_m_1.z));
		}

		float3 n;
		n.x/*tsdf*/ = unpack_tsdf(_tsdf_volume.ptr(__param.__resolution.x * g.y + g.x)[g.z]);//read tsdf at g

		//infinite loop guard
		//bool is_found = false;
		for (; time_start_volume < time_end_volume; time_start_volume += __param.__time_step)
		{
			n.y/*tsdf_prev*/ = n.x;

			g = getVoxel(__param.__C_f + ray_dir * (time_start_volume + __param.__time_step));	
			if (!(g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __param.__resolution.x && g.y < __param.__resolution.y && g.z < __param.__resolution.z))  break; //get next g
			
			n.x/*tsdf*/ = unpack_tsdf(_tsdf_volume.ptr(__param.__resolution.x * g.y + g.x)[g.z]); //read tsdf at g

			if (isnan(n.y/*tsdf_prev*/) || isnan(n.x/*tsdf*/) || n.y/*tsdf_prev*/ == n.x/*tsdf*/ || n.y/*tsdf_prev*/ < 0.f && n.x/*tsdf*/ >= 0.f)  continue;

			if (n.y/*tsdf_prev*/ >= 0.f && n.x/*tsdf*/ < 0.f)           //zero crossing
			{
				n.x/*tsdf*/ = interpolateTrilinearyOrigin(ray_dir, time_start_volume + __param.__time_step); if (isnan(n.x/*tsdf*/)) continue; //get more accurate tsdf & tsdf_prev Ftdt
				n.y/*tsdf_prev*/ = interpolateTrilinearyOrigin(ray_dir, time_start_volume);               if (isnan(n.y/*tsdf_prev*/)) continue; //Ft

				float Ts = time_start_volume - __param.__time_step * n.y/*tsdf_prev*/ / (n.x/*tsdf*/ - n.y/*tsdf_prev*/);
				float3 vertex_found = __param.__C_f + ray_dir * Ts; //(time_start_volume - __param.__time_step * n.y/*tsdf_prev*/ / (n.x/*tsdf*/ - n.y/*tsdf_prev*/));

				n.x = interpolateTrilineary(make_float3(vertex_found.x + __param.__cell_size, vertex_found.y, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x - __param.__cell_size, vertex_found.y, vertex_found.z));
				n.y = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y + __param.__cell_size, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y - __param.__cell_size, vertex_found.z));
				n.z = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z + __param.__cell_size)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z - __param.__cell_size));
				float inv_len = rsqrtf(n.x*n.x + n.y*n.y + n.z*n.z); 
				inv_len = fmaxf(inv_len + (inv_len - inv_len), 0.0); if (inv_len!=inv_len || inv_len>1e3) continue;
				n *= inv_len; if (dot3<float, float3>(n, ray_dir) >= -0.2f) continue; //exclude the points whose normal and the viewing direction are smaller than 98 degree ( 180 degree when the surface directly facing the camera )
				
				//n and vertex_found are in tsdf coordinate

				float3 v_p = _vmap.ptr(y)[x]; // in camera coordinate
				float3 n_p = _nmap.ptr(y)[x];
				if (btl::device::isnan(v_p.x) || btl::device::isnan(n_p.x)){
					if (!_no_rad)
					{
						float3 rad;
						interpolateTrilinearyColor(vertex_found, rad);
						_radiance.ptr(y)[x] = rad;
					}
					_nmap.ptr(y)[x] = __param.__Rcf*n; //transform from tsdf coordinate to camera cooridnate
					_vmap.ptr(y)[x] = __param.__Rcf*(vertex_found - __param.__C_f);
				}
				else if (!btl::device::isnan(n.x)){
					if (norm<float, float3>(v_p) > Ts) { //the second ray casting is closer
						if (!_no_rad)
						{
							float3 rad;
							interpolateTrilinearyColor(vertex_found, rad);
							_radiance.ptr(y)[x] = rad;
						}
						_nmap.ptr(y)[x] = __param.__Rcf*n;
						_vmap.ptr(y)[x] = __param.__Rcf*(vertex_found - __param.__C_f);
					}
				}

				//if (x % 73 == 1)
				//	printf("color %u, %u, %u \n", _rgb.ptr(y)[x].x, _rgb.ptr(y)[x].y, _rgb.ptr(y)[x].z);

				//is_found = true;
				break;
			}//if (tsdf_prev > 0.f && tsdf < 0.f) 
		}//for (; time_start_volume < time_exit_volume; time_start_volume += time_step)
		//if (!is_found){
		//	vmap.ptr(y)[x] = nmap.ptr(y)[x] = make_float3(pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN());
		//}
		return;
	}//ray_casting()

	__device__ __forceinline__ void
		interpolateTrilinearyAll(const float3& Xw_, float3& total_radiance_, float3& total_nr_) const
	{
		float a = Xw_.x * __param.__inv_cell_size;
		float b = Xw_.y * __param.__inv_cell_size;
		float c = Xw_.z * __param.__inv_cell_size;

		int3 g = make_int3(__float2int_rd(a), //round down to negative infinity
			__float2int_rd(b),
			__float2int_rd(c));//get voxel coordinate

		if (g.x<1 || g.y<1 || g.z<1 || g.x >__param.__resolution_m_2.x || g.y > __param.__resolution_m_2.y || g.z > __param.__resolution_m_2.z) return;

		g.x = (Xw_.x < g.x * __param.__cell_size + __param.__half_cell_size) ? (g.x - 1.f) : g.x;
		g.y = (Xw_.y < g.y * __param.__cell_size + __param.__half_cell_size) ? (g.y - 1.f) : g.y;
		g.z = (Xw_.z < g.z * __param.__cell_size + __param.__half_cell_size) ? (g.z - 1.f) : g.z;

		a -= (g.x + 0.5f);
		b -= (g.y + 0.5f);
		c -= (g.z + 0.5f);
		int row = __param.__resolution.x * g.y + g.x;
		float total_weight = 0.f;
		total_radiance_.x = total_radiance_.y = total_radiance_.z = 0.f;
		float weight;
		float3 nY;
		nY = getRadiance(_radiance_volume.ptr(row) + g.z);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = (1 - a) * (1 - b) * (1 - c);
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row) + g.z);
			total_nr_ += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x) + g.z);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = (1 - a) * b       * (1 - c);
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row + __param.__resolution.x) + g.z);
			total_nr_ += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + 1) + g.z);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * (1 - b) * (1 - c);
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row + 1) + g.z);
			total_nr_ += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x + 1) + g.z);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * b       * (1 - c);
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row + __param.__resolution.x + 1) + g.z);
			total_nr_ += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = (1 - a) * (1 - b) * c;
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row) + g.z + 1);
			total_nr_ += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = (1 - a) * b       * c;
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row + __param.__resolution.x) + g.z + 1);
			total_nr_ += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + 1) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * (1 - b) * c;
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row + 1) + g.z + 1);
			total_nr_ += (nY * weight);
		}
		nY = getRadiance(_radiance_volume.ptr(row + __param.__resolution.x + 1) + g.z + 1);
		if (nY.x == nY.x && nY.y == nY.y && nY.z == nY.z){
			weight = a       * b       * c;
			total_weight += weight;
			total_radiance_ += (nY * weight);
			nY = getRadiance(_normalized_randiance_volume.ptr(row + __param.__resolution.x + 1) + g.z + 1);
			total_nr_ += (nY * weight);
		}
		total_radiance_ /= total_weight;
		return;
	}

	__device__ __forceinline__ void ray_casting_all() const
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		//the organization of the volume is described in the following links
		//https://docs.google.com/drawings/d/1lkw9jnNsVifIc42aDCtqnMEqc53FAlJsEIk3VBMfCN0/edit?usp=sharing
		//The x index is major, then y index, and z index is the last.
		//Thus, voxel (x,y,z) is stored in 2-D matrix, 
		// row: y*resolution_x + x
		// col: z

		if (x >= __param.__cols || y >= __param.__rows) return;

		float3 ray_dir;
		ray_dir.x = (x - __param.__intr.cx) / __param.__intr.fx;
		ray_dir.y = (y - __param.__intr.cy) / __param.__intr.fy;
		ray_dir.z = 1.f;

		ray_dir = __param.__Rcf_t * ray_dir; //rotate from camera to tsdf coordinate f
		ray_dir *= rsqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z); //normalize ray_dir;

		// computer time when entry and exit volume
		float time_start_volume = fmaxf(getMinTime(ray_dir), 0.f);
		//float time_start_volume = fmax(fmax(fmax(((ray_dir.x > 0 ? 0.f : __volume_size.x) - __tcurr.x) / ray_dir.x, ((ray_dir.y > 0 ? 0.f : __volume_size.y) - __tcurr.y) / ray_dir.y), ((ray_dir.z > 0 ? 0.f : __volume_size.z) - __tcurr.z) / ray_dir.z), 0.f);
		const float time_end_volume = getMaxTime(ray_dir);

		if (time_start_volume >= time_end_volume){
			//vmap.ptr(y)[x] = nmap.ptr(y)[x] = make_float3(pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN());
			return;
		}

		int3 g = getVoxel(__param.__C_f + ray_dir * time_start_volume);
		if (!(g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __param.__resolution.x && g.y < __param.__resolution.y && g.z < __param.__resolution.z)){
			g.x = fmaxf(0, fminf(g.x, __param.__resolution_m_1.x));
			g.y = fmaxf(0, fminf(g.y, __param.__resolution_m_1.y));
			g.z = fmaxf(0, fminf(g.z, __param.__resolution_m_1.z));
		}

		float3 n;
		n.x/*tsdf*/ = unpack_tsdf(_tsdf_volume.ptr(__param.__resolution.x * g.y + g.x)[g.z]);//read tsdf at g

		//infinite loop guard
		//bool is_found = false;
		for (; time_start_volume < time_end_volume; time_start_volume += __param.__time_step)
		{
			n.y/*tsdf_prev*/ = n.x;

			g = getVoxel(__param.__C_f + ray_dir * (time_start_volume + __param.__time_step));
			if (!(g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __param.__resolution.x && g.y < __param.__resolution.y && g.z < __param.__resolution.z))  break; //get next g

			n.x/*tsdf*/ = unpack_tsdf(_tsdf_volume.ptr(__param.__resolution.x * g.y + g.x)[g.z]); //read tsdf at g

			if (isnan(n.y/*tsdf_prev*/) || isnan(n.x/*tsdf*/) || n.y/*tsdf_prev*/ == n.x/*tsdf*/ || n.y/*tsdf_prev*/ < 0.f && n.x/*tsdf*/ >= 0.f)  continue;

			if (n.y/*tsdf_prev*/ >= 0.f && n.x/*tsdf*/ < 0.f)           //zero crossing
			{
				n.x/*tsdf*/ = interpolateTrilinearyOrigin(ray_dir, time_start_volume + __param.__time_step); if (isnan(n.x/*tsdf*/)) continue; //get more accurate tsdf & tsdf_prev Ftdt
				n.y/*tsdf_prev*/ = interpolateTrilinearyOrigin(ray_dir, time_start_volume);               if (isnan(n.y/*tsdf_prev*/)) continue; //Ft

				float Ts = time_start_volume - __param.__time_step * n.y/*tsdf_prev*/ / (n.x/*tsdf*/ - n.y/*tsdf_prev*/);
				float3 vertex_found = __param.__C_f + ray_dir * Ts; //(time_start_volume - __param.__time_step * n.y/*tsdf_prev*/ / (n.x/*tsdf*/ - n.y/*tsdf_prev*/));

				n.x = interpolateTrilineary(make_float3(vertex_found.x + __param.__cell_size, vertex_found.y, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x - __param.__cell_size, vertex_found.y, vertex_found.z));
				n.y = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y + __param.__cell_size, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y - __param.__cell_size, vertex_found.z));
				n.z = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z + __param.__cell_size)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z - __param.__cell_size));
				float inv_len = rsqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
				inv_len = fmaxf(inv_len + (inv_len - inv_len), 0.0); if (inv_len != inv_len || inv_len > 1e3) continue;
				n *= inv_len; if (dot3<float, float3>(n, ray_dir) >= -0.2f) continue; //exclude the points whose normal and the viewing direction are smaller than 98 degree ( 180 degree when the surface directly facing the camera )

				//n and vertex_found are in tsdf coordinate

				float3 v_p = _vmap.ptr(y)[x]; // in camera coordinate
				float3 n_p = _nmap.ptr(y)[x];
				if (btl::device::isnan(v_p.x) || btl::device::isnan(n_p.x)){
					float3 rad, nr;
					interpolateTrilinearyAll(vertex_found, rad, nr);
					_radiance.ptr(y)[x] = rad;
					_normalized_r.ptr(y)[x] = nr;
					_nmap.ptr(y)[x] = __param.__Rcf*n; //transform from tsdf coordinate to camera cooridnate
					_vmap.ptr(y)[x] = __param.__Rcf*(vertex_found - __param.__C_f);
				}
				else if (!btl::device::isnan(n.x)){
					if (norm<float, float3>(v_p) > Ts) { //the second ray casting is closer
						float3 rad, nr;
						interpolateTrilinearyAll(vertex_found, rad, nr);
						_radiance.ptr(y)[x] = rad;
						_normalized_r.ptr(y)[x] = nr;
						_nmap.ptr(y)[x] = __param.__Rcf*n;
						_vmap.ptr(y)[x] = __param.__Rcf*(vertex_found - __param.__C_f);
					}
				}
				break;
			}//if (tsdf_prev > 0.f && tsdf < 0.f) 
		}//for (; time_start_volume < time_exit_volume; time_start_volume += time_step)
		return;
	}//ray_casting()
};//RayCaster

__global__ void kernel_ray_casting_radiance(const RayCasterColor rc) {
	rc.ray_casting_radiance();
}

//get VMap and NMap in world
void cuda_ray_casting_radiance(const Intr& intr_, const Matd33& Rcf_t_, const Matd33& Rcf_, const double3& C_f_, 
						    const float fTruncDistanceM_, const float& fVoxelSize_, const short3& resolution_, const float3& volume_size_, 
							const GpuMat& tsdf_volume_, const GpuMat& radiance_volume_, GpuMat* pVMap_, GpuMat* pNMap_, GpuMat* pRadiance_)
{
	RayCasterColorParam param;
	param.__cols = pVMap_->cols;
	param.__rows = pVMap_->rows;
	param.__cell_size = fVoxelSize_;
	param.__half_cell_size = fVoxelSize_ *0.5f;
	param.__resolution = resolution_;
	param.__resolution_m_1 = make_short3(resolution_.x - 1, resolution_.y - 1, resolution_.z - 1);
	param.__resolution_m_2 = make_short3(resolution_.x - 2, resolution_.y - 2, resolution_.z - 2);
	param.__Rcf = Rcf_; // 
	param.__Rcf_t = Rcf_t_; // 
	param.__C_f = C_f_; 
	param.__intr = intr_;
	param.__inv_cell_size = 1.f / fVoxelSize_;
	param.__volume_size = volume_size_;

	//if (!bFineCast_)
	//	param.__time_step = fVoxelSize_ * 8.f; //fTruncDistanceM_ * 0.4f; 
	//else
	param.__time_step = fVoxelSize_ * 6.f;
	cudaSafeCall(cudaMemcpyToSymbol(__param, &param, sizeof(RayCasterColorParam))); //copy host memory to constant memory on the device.

	RayCasterColor rc;
	rc._tsdf_volume = tsdf_volume_;
	rc._radiance_volume = radiance_volume_;
	rc._vmap = *pVMap_;
	rc._nmap = *pNMap_;
	if (!pRadiance_ || pRadiance_->empty())
	{
		rc._no_rad = true;
	}
	else{
		rc._no_rad = false;
		rc._radiance = *pRadiance_;
	}

	dim3 block (RayCasterColor::CTA_SIZE_X, RayCasterColor::CTA_SIZE_Y);
	dim3 grid (divUp (pVMap_->cols, block.x), divUp (pVMap_->rows, block.y));

	kernel_ray_casting_radiance<<<grid, block>>>(rc);
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());

	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_ray_casting_all(const RayCasterColor rc) {
	rc.ray_casting_all();
}

//get VMap and NMap in world
void cuda_ray_casting_all(const Intr& intr_, const Matd33& Rcf_t_, const Matd33& Rcf_, const double3& C_f_,
	const float fTruncDistanceM_, const float& fVoxelSize_, const short3& resolution_, const float3& volume_size_,
	const GpuMat& tsdf_volume_, const GpuMat& nr_volume_, const GpuMat& radiance_volume_,
	GpuMat* pVMap_, GpuMat* pNMap_, GpuMat* pRadiance_, GpuMat* pNR_)
{
	RayCasterColorParam param;
	param.__cols = pVMap_->cols;
	param.__rows = pVMap_->rows;
	param.__cell_size = fVoxelSize_;
	param.__half_cell_size = fVoxelSize_ *0.5f;
	param.__resolution = resolution_;
	param.__resolution_m_1 = make_short3(resolution_.x - 1, resolution_.y - 1, resolution_.z - 1);
	param.__resolution_m_2 = make_short3(resolution_.x - 2, resolution_.y - 2, resolution_.z - 2);
	param.__Rcf = Rcf_; // 
	param.__Rcf_t = Rcf_t_; // 
	param.__C_f = C_f_;
	param.__intr = intr_;
	param.__inv_cell_size = 1.f / fVoxelSize_;
	param.__volume_size = volume_size_;
	param.__time_step = fVoxelSize_ * 6.f;
	cudaSafeCall(cudaMemcpyToSymbol(__param, &param, sizeof(RayCasterColorParam))); //copy host memory to constant memory on the device.

	RayCasterColor rc;
	rc._tsdf_volume = tsdf_volume_;
	rc._radiance_volume = radiance_volume_;
	rc._normalized_randiance_volume = nr_volume_;
	rc._vmap = *pVMap_;
	rc._nmap = *pNMap_;
	rc._radiance = *pRadiance_;
	rc._normalized_r = *pNR_;

	dim3 block(RayCasterColor::CTA_SIZE_X, RayCasterColor::CTA_SIZE_Y);
	dim3 grid(divUp(pVMap_->cols, block.x), divUp(pVMap_->rows, block.y));

	kernel_ray_casting_all << <grid, block >> >(rc);
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());

	return;
}
}// device
}// btl
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

