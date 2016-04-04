//copied from opencv
/*

#ifndef __OPENCV_GPU_COMMON_HPP__
#define __OPENCV_GPU_COMMON_HPP__

#include <cuda_runtime.h>
#include <opencv2/cudev/ptr2d/gpumat.hpp>

#ifndef CV_PI
    #define CV_PI   3.1415926535897932384626433832795
#endif

#ifndef CV_PI_F
    #ifndef CV_PI
        #define CV_PI_F 3.14159265f
    #else
        #define CV_PI_F ((float)CV_PI)
    #endif
#endif

#if defined(__GNUC__)
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else / * defined(__CUDACC__) || defined(__MSVC__) * /
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)
#endif

namespace cv { namespace gpu 
{
    void error(const char *error_string, const char *file, const int line, const char *func);

    template <typename T> static inline bool isAligned(const T* ptr, size_t size)
    {
        return reinterpret_cast<size_t>(ptr) % size == 0;
    }

    static inline bool isAligned(size_t step, size_t size)
    {
        return step % size == 0;
    }
}}

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        cv::cuda::error(cudaGetErrorString(err), file, line, func);
}

#ifdef __CUDACC__

namespace cv { namespace gpu 
{   
	using namespace cv::cudev;
    / *__host__ __device__ __forceinline__ int divUp(int total, int grain) 
    { 
        return (total + grain - 1) / grain; 
    }* /

    namespace device 
    {
        typedef unsigned char uchar;
        typedef unsigned short ushort;
        typedef signed char schar;
        typedef unsigned int uint;

        template<typename T> inline void bindTexture(const textureReference* tex, const GpuMat_<T>& img)
        {
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
            cudaSafeCall( cudaBindTexture2D(0, tex, img.ptr(), &desc, img.cols, img.rows, img.step) );
        }
    }
}}

#endif // __CUDACC__

#endif // __OPENCV_GPU_COMMON_HPP__*/
