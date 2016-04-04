
#ifndef PCL_GPU_KINFU_DEVICE_HPP_
#define PCL_GPU_KINFU_DEVICE_HPP_

#include "limits.hpp"
#include "vector_math.hpp"

#include "internal.h"
#include "block.hpp"

using namespace pcl::device;

namespace btl{ namespace device{

	template<typename T>
	__device__ __forceinline__ bool isnan(T t){	return t!=t;}
	template<typename T>
	__device__ __forceinline__ void outProductSelf(const T V_, T* pMRm_){
		pMRm_[0].x = V_.x*V_.x;
		pMRm_[0].y = pMRm_[1].x = V_.x*V_.y;
		pMRm_[0].z = pMRm_[2].x = V_.x*V_.z;
		pMRm_[1].y = V_.y*V_.y;
		pMRm_[1].z = pMRm_[2].y = V_.y*V_.z;
		pMRm_[2].z = V_.z*V_.z;
	}
	__device__ __forceinline__ void setIdentity(float fScalar_, float3* pMRm_){
		pMRm_[0].x = pMRm_[1].y = pMRm_[2].z = fScalar_;
		pMRm_[0].y = pMRm_[0].z = pMRm_[1].x = pMRm_[1].z = pMRm_[2].x = pMRm_[2].y = 0;
	}
	template<typename T>
	__device__ __forceinline__ T accumulate(int lvl_, T* total_){
		T nCounter;
		nCounter = atomicAdd(&total_[lvl_], 1);
		return nCounter;
	}
	template<typename T>
	__device__ __forceinline__ int decrease(int lvl_, T* total_){
		T nCounter;
		nCounter = atomicSub(&total_[lvl_], 1); //have to be this way
		return nCounter;
	}


}//device
}//btl

namespace pcl
{
namespace device
{
//////////////////////////////////////////////////////////////////////////////////////
/// for old format
//Tsdf fixed point divisor (if old format is enabled)
const int DIVISOR =  32767;     // SHRT_MAX; //30000; //

//should be multiple of 32
//enum { VOLUME_X = 512, VOLUME_Y = 512, VOLUME_Z = 512 };


const float VOLUME_SIZE = 3.0f; // in meters
 
#define INV_DIV 3.051850947599719e-5f

//constant weight
//__device__ __forceinline__ void
//pack_tsdf (float tsdf, int weight, short2& value)
//{
//    int fixedp = max (-DIVISOR, min (DIVISOR, __float2int_rz (tsdf * DIVISOR)));
//    //int fixedp = __float2int_rz(tsdf * DIVISOR);
//    value = make_short2 (fixedp, weight);
//}
//
//__device__ __forceinline__ void
//unpack_tsdf (short2 value, float& tsdf, int& weight)
//{
//    weight = value.y;
//	tsdf = __int2float_rn (value.x) / DIVISOR;   //*/ * INV_DIV;
//}
//
//__device__ __forceinline__ float
//unpack_tsdf (short2 value)
//{
//    return static_cast<float>(value.x) / DIVISOR;    //*/ * INV_DIV;
//}

//variable weight
//according to Erik, B., Kerl, C., & Kahl, F. (2013). Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions. In RSS.
__device__ __forceinline__ void
pack_tsdf(float tsdf, short weight, short2& value)
{
	int fixedp = fmaxf(-DIVISOR, fminf(DIVISOR, __float2int_rz(tsdf * DIVISOR)));
	//int fixedp = __float2int_rz(tsdf * DIVISOR);
	value = make_short2(fixedp, weight);
}

__device__ __forceinline__ void
unpack_tsdf(short2 value, float& tsdf, short& weight)
{
	weight = value.y;
	tsdf = __int2float_rn(value.x) * INV_DIV; // DIVISOR;   //*/  
}

__device__ __forceinline__ float
unpack_tsdf(short2 value)
{
	return static_cast<float>(value.x) * INV_DIV; //DIVISOR
}

//////////////////////////////////////////////////////////////////////////////////////
/// for half float
/*
__device__ __forceinline__ void
pack_tsdf (float tsdf, int weight, ushort2& value)
{
    value = make_ushort2 (__float2half_rn (tsdf), weight);
}

__device__ __forceinline__ void
unpack_tsdf (ushort2 value, float& tsdf, int& weight)
{
    tsdf = __half2float (value.x);
    weight = value.y;
}

__device__ __forceinline__ float
unpack_tsdf (ushort2 value)
{
    return __half2float (value.x);
}*/

__device__ __forceinline__ float3
operator* (const Mat33& m, const float3& vec)
{
	return make_float3(dot3<float, float3>(m.data[0], vec), dot3<float, float3>(m.data[1], vec), dot3<float, float3>(m.data[2], vec));
}

__device__ __forceinline__ float3
operator* (const Matd33& m, const float3& vec)
{
	return make_float3( float( m.data[0].x * vec.x + m.data[0].y * vec.y + m.data[0].z * vec.z),
						float( m.data[1].x * vec.x + m.data[1].y * vec.y + m.data[1].z * vec.z),
						float( m.data[2].x * vec.x + m.data[2].y * vec.y + m.data[2].z * vec.z) );
}

struct Warp
{
	enum
	{
		LOG_WARP_SIZE = 5,
		WARP_SIZE     = 1 << LOG_WARP_SIZE,
		STRIDE        = WARP_SIZE
	};

	/** \brief Returns the warp lane ID of the calling thread. */
	static __device__ __forceinline__ unsigned int 
		laneId()
	{
		unsigned int ret;
		asm("mov.u32 %0, %laneid;" : "=r"(ret) );
		return ret;
	}

	static __device__ __forceinline__ unsigned int id()
	{
		int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
		return tid >> LOG_WARP_SIZE;
	}

	static __device__ __forceinline__ 
		int laneMaskLt()
	{
#if (__CUDA_ARCH__ >= 200)
		unsigned int ret;
		asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
		return ret;
#else
		return 0xFFFFFFFF >> (32 - laneId());
#endif
	}

	static __device__ __forceinline__ int binaryExclScan(int ballot_mask)
	{
		return __popc(Warp::laneMaskLt() & ballot_mask);
	}   
};


struct Emulation
{        
	static __device__ __forceinline__ int
		warp_reduce ( volatile int *ptr , const unsigned int tid)
	{
		const unsigned int lane = tid & 31; // index of thread in warp (0..31)        

		if (lane < 16)
		{				
			int partial = ptr[tid];

			ptr[tid] = partial = partial + ptr[tid + 16];
			ptr[tid] = partial = partial + ptr[tid + 8];
			ptr[tid] = partial = partial + ptr[tid + 4];
			ptr[tid] = partial = partial + ptr[tid + 2];
			ptr[tid] = partial = partial + ptr[tid + 1];            
		}
		return ptr[tid - lane];
	}

	static __forceinline__ __device__ int 
		Ballot(int predicate, volatile int* cta_buffer)
	{
#if __CUDA_ARCH__ >= 200
		(void)cta_buffer;
		return __ballot(predicate);
#else
		int tid = pcl::device::Block::flattenedThreadId();				
		cta_buffer[tid] = predicate ? (1 << (tid & 31)) : 0;
		return warp_reduce(cta_buffer, tid);
#endif
	}

	static __forceinline__ __device__ bool
		All(int predicate, volatile int* cta_buffer)
	{
#if __CUDA_ARCH__ >= 200
		(void)cta_buffer;
		return __all(predicate);
#else
		int tid = Block::flattenedThreadId();				
		cta_buffer[tid] = predicate ? 1 : 0;
		return warp_reduce(cta_buffer, tid) == 32;
#endif
	}
};


////////////////////////////////////////////////////////////////////////////////////////
///// Prefix Scan utility

enum ScanKind { exclusive, inclusive };

template<ScanKind Kind, class T>
__device__ __forceinline__ T
scan_warp ( volatile T *ptr, const unsigned int idx = threadIdx.x )
{
    const unsigned int lane = idx & 31;       // index of thread in warp (0..31)

    if (lane >= 1)
    ptr[idx] = ptr[idx - 1] + ptr[idx];
    if (lane >= 2)
    ptr[idx] = ptr[idx - 2] + ptr[idx];
    if (lane >= 4)
    ptr[idx] = ptr[idx - 4] + ptr[idx];
    if (lane >= 8)
    ptr[idx] = ptr[idx - 8] + ptr[idx];
    if (lane >= 16)
    ptr[idx] = ptr[idx - 16] + ptr[idx];

    if (Kind == inclusive)
    return ptr[idx];
    else
    return (lane > 0) ? ptr[idx - 1] : 0;
}
}
}


__device__ __forceinline__ void computeRoots2(const float& b, const float& c, float3& roots)
{
	roots.x = 0.f;
	float d = b * b - 4.f * c;
	if (d < 0.f) // no real roots!!!! THIS SHOULD NOT HAPPEN!
		d = 0.f;

	float sd = sqrtf(d);

	roots.z = 0.5f * (b + sd);
	roots.y = 0.5f * (b - sd);
}

__device__ __forceinline__ void
computeRoots3(float c0, float c1, float c2, float3& roots)
{
	if (fabsf(c0) < numeric_limits<float>::epsilon())// one root is 0 -> quadratic equation
	{
		computeRoots2(c2, c1, roots);
	}
	else
	{
		const float s_inv3 = 1.f / 3.f;
		const float s_sqrt3 = sqrtf(3.f);
		// Construct the parameters used in classifying the roots of the equation
		// and in solving the equation for the roots in closed form.
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2*c2_over_3)*s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;

		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;

		// Compute the eigenvalues by solving for the roots of the polynomial.
		float rho = sqrtf(-a_over_3);
		float theta = atan2f(sqrtf(-q), half_b)*s_inv3;
		float cos_theta = __cosf(theta);
		float sin_theta = __sinf(theta);
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		// Sort in increasing order.
		if (roots.x >= roots.y)
			swap(roots.x, roots.y);

		if (roots.y >= roots.z)
		{
			swap(roots.y, roots.z);

			if (roots.x >= roots.y)
				swap(roots.x, roots.y);
		}
		if (roots.x <= 0) // eigenval for symetric positive semi-definite matrix can not be negative! Set it to 0
			computeRoots2(c2, c1, roots);
	}
}

struct Eigen33
{
public:
	template<int Rows>
	struct MiniMat
	{
		float3 data[Rows];
		__device__ __host__ __forceinline__ float3& operator[](int i) { return data[i]; }
		__device__ __host__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
	};
	typedef MiniMat<3> Mat33;
	typedef MiniMat<4> Mat43;


	static __forceinline__ __device__ float3
		unitOrthogonal(const float3& src)
	{
		float3 perp;
		/* Let us compute the crossed product of *this with a vector
		* that is not too close to being colinear to *this.
		*/

		/* unless the x and y coords are both close to zero, we can
		* simply take ( -y, x, 0 ) and normalize it.
		*/
		if (!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
		{
			float invnm = rsqrtf(src.x*src.x + src.y*src.y);
			perp.x = -src.y * invnm;
			perp.y = src.x * invnm;
			perp.z = 0.0f;
		}
		/* if both x and y are close to zero, then the vector is close
		* to the z-axis, so it's far from colinear to the x-axis for instance.
		* So we take the crossed product with (1,0,0) and normalize it.
		*/
		else
		{
			float invnm = rsqrtf(src.z * src.z + src.y * src.y);
			perp.x = 0.0f;
			perp.y = -src.z * invnm;
			perp.z = src.y * invnm;
		}

		return perp;
	}

	__device__ __forceinline__
		Eigen33(volatile float* mat_pkg_arg) : mat_pkg(mat_pkg_arg) {}
	__device__ __forceinline__ void
		compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals)
	{
#ifdef SCALE_NORMALIZATION
		// Scale the matrix so its entries are in [-1,1].  The scaling is applied
		// only when at least one matrix entry has magnitude larger than 1.
		float max01 = fmaxf( fabsf(mat_pkg[0]), fabsf(mat_pkg[1]) );
		float max23 = fmaxf( fabsf(mat_pkg[2]), fabsf(mat_pkg[3]) );
		float max45 = fmaxf( fabsf(mat_pkg[4]), fabsf(mat_pkg[5]) );
		float m0123 = fmaxf( max01, max23 );
		float scale = fmaxf( max45, m0123 );

		if (scale <= numeric_limits<float>::min())
			scale = 1.f;

		mat_pkg[0] /= scale;
		mat_pkg[1] /= scale;
		mat_pkg[2] /= scale;
		mat_pkg[3] /= scale;
		mat_pkg[4] /= scale;
		mat_pkg[5] /= scale;
#endif // SCALE_NORMALIZATION

		

		// The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
		// eigenvalues are the roots to this equation, all guaranteed to be
		// real-valued, because the matrix is symmetric.
		float c0 = m00() * m11() * m22()
			+ 2.f * m01() * m02() * m12()
			- m00() * m12() * m12()
			- m11() * m02() * m02()
			- m22() * m01() * m01();
		float c1 = m00() * m11() -
			m01() * m01() +
			m00() * m22() -
			m02() * m02() +
			m11() * m22() -
			m12() * m12();
		float c2 = m00() + m11() + m22();

		computeRoots3(c0, c1, c2, evals);

		if (evals.z - evals.x <= numeric_limits<float>::epsilon()) { //z == x
			evecs[0] = make_float3(1.f, 0.f, 0.f);
			evecs[1] = make_float3(0.f, 1.f, 0.f);
			evecs[2] = make_float3(0.f, 0.f, 1.f);
		}
		else if (evals.y - evals.x <= numeric_limits<float>::epsilon()) { //y == x
			// first and second equal                
			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot3<float, float3>(vec_tmp[0], vec_tmp[0]);
			float len2 = dot3<float, float3>(vec_tmp[1], vec_tmp[1]);
			float len3 = dot3<float, float3>(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				evecs[2] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				evecs[2] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				evecs[2] = vec_tmp[2] * rsqrtf(len3);
			}

			evecs[1] = unitOrthogonal(evecs[2]);
			evecs[0] = cross(evecs[1], evecs[2]);
		}
		else if (evals.z - evals.y <= numeric_limits<float>::epsilon()) //z == y
		{
			// second and third equal                                    
			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot3<float, float3>(vec_tmp[0], vec_tmp[0]);
			float len2 = dot3<float, float3>(vec_tmp[1], vec_tmp[1]);
			float len3 = dot3<float, float3>(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				evecs[0] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				evecs[0] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				evecs[0] = vec_tmp[2] * rsqrtf(len3);
			}

			evecs[1] = unitOrthogonal(evecs[0]);
			evecs[2] = cross(evecs[0], evecs[1]);
		}
		else //x != y != z
		{

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot3<float, float3>(vec_tmp[0], vec_tmp[0]);
			float len2 = dot3<float, float3>(vec_tmp[1], vec_tmp[1]);
			float len3 = dot3<float, float3>(vec_tmp[2], vec_tmp[2]);

			float mmax[3];

			unsigned int min_el = 2;
			unsigned int max_el = 2;
			if (len1 >= len2 && len1 >= len3)
			{
				mmax[2] = len1;
				evecs[2] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[2] = len2;
				evecs[2] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				mmax[2] = len3;
				evecs[2] = vec_tmp[2] * rsqrtf(len3);
			}

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			len1 = dot3<float, float3>(vec_tmp[0], vec_tmp[0]);
			len2 = dot3<float, float3>(vec_tmp[1], vec_tmp[1]);
			len3 = dot3<float, float3>(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				mmax[1] = len1;
				evecs[1] = vec_tmp[0] * rsqrtf(len1);
				min_el = len1 <= mmax[min_el] ? 1 : min_el;
				max_el = len1  > mmax[max_el] ? 1 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[1] = len2;
				evecs[1] = vec_tmp[1] * rsqrtf(len2);
				min_el = len2 <= mmax[min_el] ? 1 : min_el;
				max_el = len2  > mmax[max_el] ? 1 : max_el;
			}
			else
			{
				mmax[1] = len3;
				evecs[1] = vec_tmp[2] * rsqrtf(len3);
				min_el = len3 <= mmax[min_el] ? 1 : min_el;
				max_el = len3 >  mmax[max_el] ? 1 : max_el;
			}

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			len1 = dot3<float, float3>(vec_tmp[0], vec_tmp[0]);
			len2 = dot3<float, float3>(vec_tmp[1], vec_tmp[1]);
			len3 = dot3<float, float3>(vec_tmp[2], vec_tmp[2]);


			if (len1 >= len2 && len1 >= len3)
			{
				mmax[0] = len1;
				evecs[0] = vec_tmp[0] * rsqrtf(len1);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3  > mmax[max_el] ? 0 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[0] = len2;
				evecs[0] = vec_tmp[1] * rsqrtf(len2);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3  > mmax[max_el] ? 0 : max_el;
			}
			else
			{
				mmax[0] = len3;
				evecs[0] = vec_tmp[2] * rsqrtf(len3);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3  > mmax[max_el] ? 0 : max_el;
			}

			unsigned mid_el = 3 - min_el - max_el;
			evecs[min_el] = normalized<float, float3>(cross(evecs[(min_el + 1) % 3], evecs[(min_el + 2) % 3]));
			evecs[mid_el] = normalized<float, float3>(cross(evecs[(mid_el + 1) % 3], evecs[(mid_el + 2) % 3]));
		}
		// Rescale back to the original size.
#ifdef SCALE_NORMALIZATION
		evals *= scale;
#endif 
	}
private:
	volatile float* mat_pkg;

	__device__  __forceinline__ float m00() const { return mat_pkg[0]; }
	__device__  __forceinline__ float m01() const { return mat_pkg[1]; }
	__device__  __forceinline__ float m02() const { return mat_pkg[2]; }
	__device__  __forceinline__ float m10() const { return mat_pkg[1]; }
	__device__  __forceinline__ float m11() const { return mat_pkg[3]; }
	__device__  __forceinline__ float m12() const { return mat_pkg[4]; }
	__device__  __forceinline__ float m20() const { return mat_pkg[2]; }
	__device__  __forceinline__ float m21() const { return mat_pkg[4]; }
	__device__  __forceinline__ float m22() const { return mat_pkg[5]; }

	__device__  __forceinline__ float3 row0() const { return make_float3(m00(), m01(), m02()); }
	__device__  __forceinline__ float3 row1() const { return make_float3(m10(), m11(), m12()); }
	__device__  __forceinline__ float3 row2() const { return make_float3(m20(), m21(), m22()); }

	__device__  __forceinline__ static bool isMuchSmallerThan(float x, float y)
	{
		// copied from <eigen>/include/Eigen/src/Core/NumTraits.h
		const float prec_sqr = numeric_limits<float>::epsilon() * numeric_limits<float>::epsilon();
		return x * x <= prec_sqr * y * y;
	}
};

/*
struct Block
{
	static __device__ __forceinline__ unsigned int stride()
	{
		return blockDim.x * blockDim.y * blockDim.z;
	}

	static __device__ __forceinline__ int
		flattenedThreadId()
	{
		return threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	}

	template<int CTA_SIZE, typename T, class BinOp>
	static __device__ __forceinline__ void reduce(volatile T* buffer, BinOp op)
	{
		int tid = flattenedThreadId();
		T val = buffer[tid];

		if (CTA_SIZE >= 1024) { if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]); __syncthreads(); }
		if (CTA_SIZE >= 512) { if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]); __syncthreads(); }
		if (CTA_SIZE >= 256) { if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]); __syncthreads(); }
		if (CTA_SIZE >= 128) { if (tid <  64) buffer[tid] = val = op(val, buffer[tid + 64]); __syncthreads(); }

		if (tid < 32)
		{
			if (CTA_SIZE >= 64) { buffer[tid] = val = op(val, buffer[tid + 32]); }
			if (CTA_SIZE >= 32) { buffer[tid] = val = op(val, buffer[tid + 16]); }
			if (CTA_SIZE >= 16) { buffer[tid] = val = op(val, buffer[tid + 8]); }
			if (CTA_SIZE >= 8) { buffer[tid] = val = op(val, buffer[tid + 4]); }
			if (CTA_SIZE >= 4) { buffer[tid] = val = op(val, buffer[tid + 2]); }
			if (CTA_SIZE >= 2) { buffer[tid] = val = op(val, buffer[tid + 1]); }
		}
	}

	template<int CTA_SIZE, typename T, class BinOp>
	static __device__ __forceinline__ T reduce(volatile T* buffer, T init, BinOp op)
	{
		int tid = flattenedThreadId();
		T val = buffer[tid] = init;
		__syncthreads();

		if (CTA_SIZE >= 1024) { if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]); __syncthreads(); }
		if (CTA_SIZE >= 512) { if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]); __syncthreads(); }
		if (CTA_SIZE >= 256) { if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]); __syncthreads(); }
		if (CTA_SIZE >= 128) { if (tid <  64) buffer[tid] = val = op(val, buffer[tid + 64]); __syncthreads(); }

		if (tid < 32)
		{
			if (CTA_SIZE >= 64) { buffer[tid] = val = op(val, buffer[tid + 32]); }
			if (CTA_SIZE >= 32) { buffer[tid] = val = op(val, buffer[tid + 16]); }
			if (CTA_SIZE >= 16) { buffer[tid] = val = op(val, buffer[tid + 8]); }
			if (CTA_SIZE >= 8) { buffer[tid] = val = op(val, buffer[tid + 4]); }
			if (CTA_SIZE >= 4) { buffer[tid] = val = op(val, buffer[tid + 2]); }
			if (CTA_SIZE >= 2) { buffer[tid] = val = op(val, buffer[tid + 1]); }
		}
		__syncthreads();
		return buffer[0];
	}
};*/

#endif /* PCL_GPU_KINFU_DEVICE_HPP_ */
