
#ifndef PCL_GPU_UTILS_DEVICE_VECTOR_MATH_HPP_
#define PCL_GPU_UTILS_DEVICE_VECTOR_MATH_HPP_

#include "pcl/internal.h"



namespace pcl
{
	namespace device
	{

		__device__ __host__ __forceinline__ uchar hamming_distance(uchar* table, uchar b1, uchar b2){
			return  table[b1^b2];
		}

		template <class T3>
		__device__ __host__ __forceinline__ void print_vector(const T3& v){
			printf("[%f %f %f]\n", v.x, v.y, v.z);
		}
		template <class M3>
		__device__ __host__ __forceinline__ void print_matrix(const M3& m){
			printf("[%f %f %f\n%f %f %f\n %f %f %f]\n", m.r[0].x, m.r[0].y, m.r[0].z, m.r[1].x, m.r[1].y, m.r[1].z, m.r[2].x, m.r[2].y, m.r[2].z);
		}
		template <class T>
		__device__ __host__ __forceinline__ void swap(T& a, T& b){
			T c(a); a = b; b = c;
		}
		__device__ __host__ __forceinline__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
			return make_short2(s2O1_.x + s2O2_.x, s2O1_.y + s2O2_.y);
		}
		__device__ __host__ __forceinline__ short2 operator - (const short2 s2O1_, const short2 s2O2_){ //can be called from host and device
			return make_short2(s2O1_.x - s2O2_.x, s2O1_.y - s2O2_.y);
		}
		__device__ __host__  __forceinline__ float2 operator * (const float& fO1_, const short2 s2O2_){
			return make_float2(fO1_* s2O2_.x, fO1_ * s2O2_.y);
		}
		__device__ __host__ __forceinline__ short2 operator * (const short sO1_, const short2 s2O2_){
			return make_short2(sO1_* s2O2_.x, sO1_ * s2O2_.y);
		}
		__device__ __host__ __forceinline__ float2 operator + (const float2& f2O1_, const float2& f2O2_){ //can be called from host and device
			return make_float2(f2O1_.x + f2O2_.x, f2O1_.y + f2O2_.y);
		}
		__device__ __host__ __forceinline__ float2 operator - (const float2& f2O1_, const float2& f2O2_){ //can be called from host and device
			return make_float2(f2O1_.x - f2O2_.x, f2O1_.y - f2O2_.y);
		}
		__device__ __host__  __forceinline__ double2 operator * (const double& fO1_, const double2& O2_){
			return make_double2(fO1_* O2_.x, fO1_ * O2_.y);
		}
		__device__ __host__ __forceinline__ double2 operator + (const double2& f2O1_, const double2& f2O2_){ //can be called from host and device
			return make_double2(f2O1_.x + f2O2_.x, f2O1_.y + f2O2_.y);
		}
		__device__ __host__ __forceinline__ double2 operator - (const double2 f2O1_, const double2 f2O2_){ //can be called from host and device
			return make_double2(f2O1_.x - f2O2_.x, f2O1_.y - f2O2_.y);
		}

		__device__ __host__ __forceinline__ int4 operator + (const int4 n4O1_, const int4 n4O2_){
			return make_int4(n4O1_.x + n4O2_.x, n4O1_.y + n4O2_.y, n4O1_.z + n4O2_.z, n4O1_.w + n4O2_.w);
		}
		__device__ __host__ __forceinline__ int4 operator - (const int4 n4O1_, const int4 n4O2_){
			return make_int4(n4O1_.x - n4O2_.x, n4O1_.y - n4O2_.y, n4O1_.z - n4O2_.z, n4O1_.w - n4O2_.w);
		}
		__device__ __host__ __forceinline__ float3 operator / (const float3& f3O1_, const float& fO2_){
			return make_float3(f3O1_.x / fO2_, f3O1_.y / fO2_, f3O1_.z / fO2_);
		}
		__device__ __host__ __forceinline__ float3 operator * (const float3& f3O1_, const float& fO2_){
			return make_float3(f3O1_.x * fO2_, f3O1_.y * fO2_, f3O1_.z * fO2_);
		}
		__device__ __host__ __forceinline__ void operator *= (float3& f3O1_, const float& fO2_){
			f3O1_.x *= fO2_;
			f3O1_.y *= fO2_;
			f3O1_.z *= fO2_;
			return;
		}
		__device__ __host__ __forceinline__ float3 operator * (const uchar3& O1_, const short O2_){
			return make_float3(O1_.x * O2_, O1_.y * O2_, O1_.z * O2_);
		}
		__device__ __host__ __forceinline__ float3 operator * (const uchar3& O1_, const float O2_){
			return make_float3(O1_.x * O2_, O1_.y * O2_, O1_.z * O2_);
		}
		__device__ __host__ __forceinline__ float3 operator - (const float3& f3O1_, const float3& f3O2_){
			return make_float3(f3O1_.x - f3O2_.x, f3O1_.y - f3O2_.y, f3O1_.z - f3O2_.z);
		}
		__device__ __host__ __forceinline__ float3 operator + (const float3& f3O1_, const float3& f3O2_){
			return make_float3(f3O1_.x + f3O2_.x, f3O1_.y + f3O2_.y, f3O1_.z + f3O2_.z);
		}
		__device__ __host__ __forceinline__ float3 operator + (const double3& dO1_, const float3& f3O2_){
			return make_float3(float(dO1_.x + f3O2_.x), float(dO1_.y + f3O2_.y), float(dO1_.z + f3O2_.z));
		}
		__device__ __host__ __forceinline__ float3 operator + (const float3& fO1_, const double3& dO2_){
			return make_float3(float(dO2_.x + fO1_.x), float(dO2_.y + fO1_.y), float(dO2_.z + fO1_.z));
		}
		__device__ __host__ __forceinline__ void operator += (float3& f3O1_, const float& fO2_){
			f3O1_.x += fO2_;
			f3O1_.y += fO2_;
			f3O1_.z += fO2_;
			return;
		}
		__device__ __host__ __forceinline__ void operator += (float3& f3O1_, const float3& fO2_){
			f3O1_.x += fO2_.x;
			f3O1_.y += fO2_.y;
			f3O1_.z += fO2_.z;
			return;
		}
		__device__ __host__ __forceinline__ void operator /= (float3& f3O1_, const float& fO2_){
			f3O1_.x /= fO2_;
			f3O1_.y /= fO2_;
			f3O1_.z /= fO2_;
			return;
		}
		__device__ __host__ __forceinline__ double3 operator / (const double3& f3O1_, const double& fO2_){
			return make_double3(f3O1_.x / fO2_, f3O1_.y / fO2_, f3O1_.z / fO2_);
		}
		__device__ __host__ __forceinline__ double3 operator * (const double3& f3O1_, const double& fO2_){
			return make_double3(f3O1_.x * fO2_, f3O1_.y * fO2_, f3O1_.z * fO2_);
		}
		__device__ __host__ __forceinline__ double3 operator - (const double3& f3O1_, const double3& f3O2_){
			return make_double3(f3O1_.x - f3O2_.x, f3O1_.y - f3O2_.y, f3O1_.z - f3O2_.z);
		}
		__device__ __host__ __forceinline__ float3 operator - (const double3& f3O1_, const float3& f3O2_){
			return make_float3(float(f3O1_.x - f3O2_.x),
				float(f3O1_.y - f3O2_.y),
				float(f3O1_.z - f3O2_.z));
		}
		__device__ __host__ __forceinline__ float3 operator - (const float3& f3O1_, const double3& f3O2_){
			return make_float3(float(f3O1_.x - f3O2_.x),
				float(f3O1_.y - f3O2_.y),
				float(f3O1_.z - f3O2_.z));
		}

		__device__ __host__ __forceinline__ float3 operator - (const uchar3& O1_, const uchar3& O2_){
			return make_float3(O1_.x - O2_.x, O1_.y - O2_.y, O1_.z - O2_.z);
		}

		__device__ __host__ __forceinline__ double3 operator + (const double3& f3O1_, const double3& f3O2_){
			return make_double3(f3O1_.x + f3O2_.x, f3O1_.y + f3O2_.y, f3O1_.z + f3O2_.z);
		}
		__device__ __host__ __forceinline__ double3 operator -= (const double3& f3O1_, const double3& f3O2_){
			return make_double3(f3O1_.x - f3O2_.x, f3O1_.y - f3O2_.y, f3O1_.z - f3O2_.z);
		}
		__device__ __host__ __forceinline__ double3 operator += (const double3& f3O1_, const double3& f3O2_){
			return make_double3(f3O1_.x + f3O2_.x, f3O1_.y + f3O2_.y, f3O1_.z + f3O2_.z);
		}
		__device__ __host__ __forceinline__ uchar3 operator + (const uchar3 uc3O1_, const uchar3& uc3O2_){
			return make_uchar3(uc3O1_.x + uc3O2_.x, uc3O1_.y + uc3O2_.y, uc3O1_.z + uc3O2_.z);
		}

		__device__ __host__ __forceinline__ matrix3f operator * (const matrix3f& v1, const matrix3f& v2){
			matrix3f out;
			out.r[0].x = v1.r[0].x * v2.r[0].x + v1.r[0].y * v2.r[1].x + v1.r[0].z * v2.r[2].x;
			out.r[0].y = v1.r[0].x * v2.r[0].y + v1.r[0].y * v2.r[1].y + v1.r[0].z * v2.r[2].y;
			out.r[0].z = v1.r[0].x * v2.r[0].z + v1.r[0].y * v2.r[1].z + v1.r[0].z * v2.r[2].z;

			out.r[1].x = v1.r[1].x * v2.r[0].x + v1.r[1].y * v2.r[1].x + v1.r[1].z * v2.r[2].x;
			out.r[1].y = v1.r[1].x * v2.r[0].y + v1.r[1].y * v2.r[1].y + v1.r[1].z * v2.r[2].y;
			out.r[1].z = v1.r[1].x * v2.r[0].z + v1.r[1].y * v2.r[1].z + v1.r[1].z * v2.r[2].z;

			out.r[2].x = v1.r[2].x * v2.r[0].x + v1.r[2].y * v2.r[1].x + v1.r[2].z * v2.r[2].x;
			out.r[2].y = v1.r[2].x * v2.r[0].y + v1.r[2].y * v2.r[1].y + v1.r[2].z * v2.r[2].y;
			out.r[2].z = v1.r[2].x * v2.r[0].z + v1.r[2].y * v2.r[1].z + v1.r[2].z * v2.r[2].z;
			return out;
		}
		//__device__ __host__ __forceinline__ Matr33d operator * (const Matr33d& v1, const Matr33d& v2){
		//	Matr33d out;
		//	out.r[0].x = v1.r[0].x * v2.r[0].x + v1.r[0].y * v2.r[1].x + v1.r[0].z * v2.r[2].x;
		//	out.r[0].y = v1.r[0].x * v2.r[0].y + v1.r[0].y * v2.r[1].y + v1.r[0].z * v2.r[2].y;
		//	out.r[0].z = v1.r[0].x * v2.r[0].z + v1.r[0].y * v2.r[1].z + v1.r[0].z * v2.r[2].z;

		//	out.r[1].x = v1.r[1].x * v2.r[0].x + v1.r[1].y * v2.r[1].x + v1.r[1].z * v2.r[2].x;
		//	out.r[1].y = v1.r[1].x * v2.r[0].y + v1.r[1].y * v2.r[1].y + v1.r[1].z * v2.r[2].y;
		//	out.r[1].z = v1.r[1].x * v2.r[0].z + v1.r[1].y * v2.r[1].z + v1.r[1].z * v2.r[2].z;

		//	out.r[2].x = v1.r[2].x * v2.r[0].x + v1.r[2].y * v2.r[1].x + v1.r[2].z * v2.r[2].x;
		//	out.r[2].y = v1.r[2].x * v2.r[0].y + v1.r[2].y * v2.r[1].y + v1.r[2].z * v2.r[2].y;
		//	out.r[2].z = v1.r[2].x * v2.r[0].z + v1.r[2].y * v2.r[1].z + v1.r[2].z * v2.r[2].z;
		//	return out;
		//}

		__device__ __host__ __forceinline__ float3 operator * (const matrix3_cmf& v1, const float3& v2){
			float3 out;
			out.x = v1.c[0].x * v2.x + v1.c[1].x * v2.y + v1.c[2].x * v2.z;
			out.y = v1.c[0].y * v2.x + v1.c[1].y * v2.y + v1.c[2].y * v2.z;
			out.z = v1.c[0].z * v2.x + v1.c[1].z * v2.y + v1.c[2].z * v2.z;
			return out;
		}

		__device__ __host__ __forceinline__ float3 operator * (const matrix3_cmd& v1, const float3& v2){
			float3 out;
			out.x = float(v1.c[0].x * v2.x + v1.c[1].x * v2.y + v1.c[2].x * v2.z);
			out.y = float(v1.c[0].y * v2.x + v1.c[1].y * v2.y + v1.c[2].y * v2.z);
			out.z = float(v1.c[0].z * v2.x + v1.c[1].z * v2.y + v1.c[2].z * v2.z);
			return out;
		}

		template<class M3>
		__device__ __host__ __forceinline__ M3 transpose(const M3& x){
			M3 t;
			t.r[0].x = x.r[0].x;
			t.r[1].x = x.r[0].y;
			t.r[2].x = x.r[0].z;

			t.r[0].y = x.r[1].x;
			t.r[1].y = x.r[1].y;
			t.r[2].y = x.r[1].z;

			t.r[0].z = x.r[2].x;
			t.r[1].z = x.r[2].y;
			t.r[2].z = x.r[2].z;
			return t;
		}
		template<typename T, typename T3> __device__ __host__ __forceinline__ T dot3(const T3& v1, const T3& v2)
		{
			return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
		}

		__device__ __host__ __forceinline__ float3 operator * (const matrix3f& m, const float3& x){
			float3 r;
			r.x = dot3<float, float3>(m.r[0], x);
			r.y = dot3<float, float3>(m.r[1], x);
			r.z = dot3<float, float3>(m.r[2], x);
			return r;
		}
		__device__ __host__ __forceinline__ double3 operator * (const Matr33d& v1, const double3& v2){
			double3 out;
			out.x = v1.c[0].x * v2.x + v1.c[1].x * v2.y + v1.c[2].x * v2.z;
			out.y = v1.c[0].y * v2.x + v1.c[1].y * v2.y + v1.c[2].y * v2.z;
			out.z = v1.c[0].z * v2.x + v1.c[1].z * v2.y + v1.c[2].z * v2.z;
			return out;
		}
		__device__ __host__ __forceinline__ float3 operator * (const float3& v1, const Matr33d& v2){
			float3 out;
			out.x = v1.x * v2.c[0].x + v1.y * v2.c[0].y + v1.z * v2.c[0].z;
			out.y = v1.x * v2.c[1].x + v1.y * v2.c[1].y + v1.z * v2.c[1].z;
			out.z = v1.x * v2.c[2].x + v1.y * v2.c[2].y + v1.z * v2.c[2].z;
			return out;
		}
		__device__ __host__ __forceinline__ float3 operator * (const float& v1, const float3& v2){
			float3 out;
			out.x = v1 *v2.x;
			out.y = v1 *v2.y;
			out.z = v1 *v2.z;
			return out;
		}
		__device__ __host__ __forceinline__ float3 operator * (const Matr33d& v1, const float3& v2){
			float3 out;
			out.x = v1.c[0].x * v2.x + v1.c[1].x * v2.y + v1.c[2].x * v2.z;
			out.y = v1.c[0].y * v2.x + v1.c[1].y * v2.y + v1.c[2].y * v2.z;
			out.z = v1.c[0].z * v2.x + v1.c[1].z * v2.y + v1.c[2].z * v2.z;
			return out;
		}
		template<class T, class M3>
		__device__ __host__ __forceinline__ T sum(const M3& m_){
			return m_.r[0].x + m_.r[0].y + m_.r[0].z + m_.r[1].x + m_.r[1].y + m_.r[1].z + m_.r[2].x + m_.r[2].y + m_.r[2].z;
		}

		template<typename T, typename T4> __device__ __host__ __forceinline__ T dot4(const T4& v1, const T4& v2)
		{
			return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
		}

		__device__ __host__ __forceinline__ float3 cross(const float3& v1, const float3& v2)
		{
			return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
		}

		template<class T, class T3, class T4>
		__device__ __host__ __forceinline__ T4 cvt_aa2quaternion(const T4 & aa){
			T4 q;
			T c_a = cos(aa.w / 2);
			T s_a = sin(aa.w / 2);
			q.w = c_a;
			q.x = aa.x*s_a;
			q.y = aa.y*s_a;
			q.z = aa.z*s_a;
			return q;
		}

		template<class T, class T3, class T4>
		__device__ __host__ __forceinline__ T4 cvt_aa2quaternion(T a1, const T3& axis1){
			T4 q;
			q.w = a1;
			q.x = axis1.x;
			q.y = axis1.y;
			q.z = axis1.z;
			return cvt_aa2quaternion<T, T3, T4>(q);
		}

		template<class T, class T4, class M3>
		__device__ __host__ __forceinline__ M3 quaternion_2_matrix(const T4& q){

			M3 m;

			const T tx = T(2)*q.x;
			const T ty = T(2)*q.y;
			const T tz = T(2)*q.z;
			const T twx = tx*q.w;
			const T twy = ty*q.w;
			const T twz = tz*q.w;
			const T txx = tx*q.x;
			const T txy = ty*q.x;
			const T txz = tz*q.x;
			const T tyy = ty*q.y;
			const T tyz = tz*q.y;
			const T tzz = tz*q.z;

			m.r[0].x = T(1) - (tyy + tzz);
			m.r[0].y = txy - twz;
			m.r[0].z = txz + twy;
			m.r[1].x = txy + twz;
			m.r[1].y = T(1) - (txx + tzz);
			m.r[1].z = tyz - twx;
			m.r[2].x = txz - twy;
			m.r[2].y = tyz + twx;
			m.r[2].z = T(1) - (txx + tyy);

			return m;
		}



		////////////////////////////////
		// four element vectors 


		////////////////////////////////
		// alltype binary operarators


		////////////////////////////////
		// tempalted operations vectors 

		template<typename T, typename T3> __device__ __host__ __forceinline__ float norm(const T3& val)
		{
			return sqrtf(dot3<T, T3>(val, val));
		}

		template<typename T, typename T3> __host__ __device__ __forceinline__ float inverse_norm(const T3& v)
		{
			return rsqrtf(dot3<T, T3>(v, v));
		}

		template<typename T, typename T3> __host__ __device__ __forceinline__ T3 normalized(const T3& v)
		{
			return v * inverse_norm<T, T3>(v);
		}

		template<typename T, typename T3> __host__ __device__ __forceinline__ T3 normalized_safe(const T3& v)
		{
			return (dot3<T, T3>(v, v) > 0) ? (v * rsqrtf(dot3<T, T3>(v, v))) : v;
		}
		template<typename T, typename T3> __host__ __device__ __forceinline__ T3 make_zero_3() {
			T3 out; out.x = T(0); out.y = T(0); out.z = T(0);
			return out;
		}
		template<typename T, typename T3, typename M3>
		__device__ __forceinline__ M3 make_zero_33(){
			M3 m;
			m.r[0] = make_zero_3<T, T3>();
			m.r[1] = make_zero_3<T, T3>();
			m.r[2] = make_zero_3<T, T3>();
			return m;
		}

		template<typename T, typename T3, typename T4, typename M3>
		__device__ __forceinline__ void pose_estimation(const T3& pt1_c, const T3& nl1_c, const T3& pt2_c, const T3& pt1_w, const T3& nl1_w, const T3& pt2_w, M3* R_cfw, T3* t_cfw){
			T3 c_w = pt1_w; // c_w is the origin of coordinate g w.r.t. world
			T alpha = acos(nl1_w.x); //rotation nl1_c to x axis (1,0,0)
			T3 axis; axis.x = 0; axis.y = nl1_w.z; axis.z = -nl1_w.y; //rotation axis between nl1_c to x axis (1,0,0) i.e. cross( nl1_w, x );
			axis = normalized<T, T3>(axis);

			T4 q_g_f_w = cvt_aa2quaternion<T, T3, T4>(alpha, axis);
			M3 R_g_f_w = quaternion_2_matrix<T, T4, M3>(q_g_f_w);
			//verify quaternion and rotation matrix
			//{
			//	Quaternionf qq(AngleAxisf(alpha, cvt_eigen<T, T3>(axis)));
			//	print_eigen< float >(qq);
			//	Matrix3f mm = qq.toRotationMatrix();
			//	cout << mm << endl;
			//  print<float, float4>(q_g_f_w);
			//  print<T, M3>( R_g_f_w );
			//}
			//verify rotation to make normal align with x-axis (1,0,0)
			//{
			//	T3 nl_x = R_g_f_w * nl1_w;
			//	print<T, T3>(nl_x);
			//}

			T3 c_c = pt1_c;
			T beta = acos(nl1_c.x); //rotation nl1_w to x axis (1,0,0)
			T3 axis2; axis2.x = 0; axis2.y = nl1_c.z; axis2.z = -nl1_c.y; //rotation axis between nl1_m to x axis (1,0,0) i.e. cross( nl1_w, x );
			axis2 = normalized<T, T3>(axis2);

			T4 q_gp_f_c = cvt_aa2quaternion<T, T3, T4>(beta, axis2);
			M3 R_gp_f_c = quaternion_2_matrix<T, T4, M3>(q_gp_f_c);

			//{
			//	Quaternionf qq(AngleAxisf(beta, cvt_eigen<T, T3>(axis2)));
			//	print_eigen< float >(qq);
			//	print<float, float4>(q_g_f_w);
			//	Matrix3f mm = qq.toRotationMatrix();
			//	cout << mm << endl;
			//}
			//{
			//	T3 nl_x = R_gp_f_c * nl1_c;
			//	print<T, T3>(nl_x);
			//}

			T3 pt2_g = R_g_f_w  * (pt2_w - c_w); pt2_g.x = T(0);  pt2_g = normalized<T, T3>(pt2_g);
			T3 pt2_gp = R_gp_f_c * (pt2_c - c_c); pt2_gp.x = T(0);  pt2_gp = normalized<T, T3>(pt2_gp);

			T gamma = acos(dot3<T, T3>(pt2_g, pt2_gp)); //rotate pt2_g to pt2_gp;
			T3 axis3; axis3.x = T(1); axis3.y = T(0); axis3.z = T(0);
			T4 q_gp_f_g = cvt_aa2quaternion<T, T3, T4>(gamma, axis3);
			M3 R_gp_f_g = quaternion_2_matrix<T, T4, M3>(q_gp_f_g);

			//{
			//	T3 nl; nl.x = T(1); nl.y = T(0); nl.z = T(0);
			//	print<T,T3>( R_gp_f_g * nl );
			//}
			//{
			//	cout<< norm<T,T3>( pt2_gp - R_gp_f_g * pt2_g ) << endl;
			//}
			M3 R_c_f_gp = transpose<M3>(R_gp_f_c);
			*R_cfw = R_c_f_gp * R_gp_f_g * R_g_f_w;
			//{
			//	T3 pt = *R_cfw * (pt2_w - c_w) + c_c;
			//	cout << norm<T, T3>( pt - pt2_c ) << endl;
			//}
			//{
			//	cout << norm<T, T3>(nl1_w) << endl;
			//	cout << norm<T, T3>(nl1_c) << endl;
			//	cout << norm<T, T3>(*R_cfw * nl1_w) << endl;
			//	cout << norm<T, T3>(nl1_c - *R_cfw * nl1_w) << endl;
			//}
			*t_cfw = c_c - (*R_cfw) * c_w;

			return;
		}


	}//device
}//pcl

#endif /* PCL_GPU_UTILS_DEVICE_VECTOR_MATH_HPP_ */

