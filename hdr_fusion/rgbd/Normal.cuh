#ifndef NORMAL_CUDA_HEADER
#define NORMAL_CUDA_HEADER
#include "DllExportDef.h"

using namespace cv::cuda;

namespace btl{	namespace device{


void DLL_EXPORT compute_normals_eigen(const GpuMat& vmap, GpuMat* nmap, GpuMat* reliability=NULL);
	

}//device
}//btl

#endif