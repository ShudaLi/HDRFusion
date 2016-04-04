#ifndef PCL_KINFU_INTERNAL_HPP_
#define PCL_KINFU_INTERNAL_HPP_

#include <opencv2/core/cuda/common.hpp>

namespace pcl
{
	namespace device
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		typedef float4 PointType;
		/** \brief Camera intrinsics structure
		*/
		struct Intr
		{
			float fx, fy, cx, cy;
			__device__ __host__ Intr() {}
			__device__ __host__ Intr(float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

			__device__ __host__ Intr operator()(int level_index) const
			{
				int div = 1 << level_index;
				return (Intr(fx / div, fy / div, cx / div, cy / div));
			}
		};

		/** \brief 3x3 Matrix for device code
		*/
		struct Mat33
		{
			float3 data[3];
		};

		struct Matd33
		{
			double3 data[3];
		};

		struct matrix3f{
			float3 r[3];
		};

		struct Matr33d{
			double3 c[3];
		};

		struct matrix3_cmf{
			float3 c[3];
		};

		struct matrix3_cmd{
			double3 c[3];
		};

		/** \brief Light source collection
		*/
		struct LightSource
		{
			float3 pos[1];
			int number;
		};

		template<class D, class Matx> D& device_cast(Matx& matx)
		{
			return (*reinterpret_cast<D*>(matx.data()));
		}

	}//device
}//pcl

#endif /* PCL_KINFU_INTERNAL_HPP_ */
