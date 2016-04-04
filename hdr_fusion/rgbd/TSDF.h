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

#ifndef BTL_GEOMETRY_TSDF
#define BTL_GEOMETRY_TSDF
#include "DllExportDef.h"
namespace btl{ namespace geometry
{
	using namespace cv::cuda;
	using namespace btl::kinect;
	using namespace std;
	using namespace pcl::device;
	using namespace Sophus;

	class DLL_EXPORT CTsdfBlock
	{
		//type
	public:
		typedef boost::shared_ptr<CTsdfBlock> tp_shared_ptr;

	private:
		//methods
	public:
		CTsdfBlock(const short3& TSDFResolution_, const float TSDFSize_, 
			const Intr& intr_,
			const SO3Group<double>& R_fw_, const Vector3d& Ow_);
		~CTsdfBlock();
		//helper
		//1. set inner front up left as invalid
		//2. release all gpu
		//3. clear vTotalLocal
		virtual void reset();

		void gpuIntegrateDepth(const GpuMat& scaled_depth, const GpuMat& normal_, const GpuMat& normalized_radiance_, const GpuMat& radiance_,
							   const GpuMat& err_, SE3Group<double>& T_cw);
		void gpuRayCastingRadiance(const Intr& intr, const SO3Group<double>& R_cw, const Vector3d& Tw, GpuMat* pPts, GpuMat* pNls, GpuMat* pRadiance_) const;
		void gpuRayCastingNR(const Intr& intr, const SO3Group<double>& R_cw, const Vector3d& Tw, GpuMat* pPts, GpuMat* pNls, GpuMat* pNormalRadiance_) const;
		void gpuRayCastingAll(const Intr& intr, const SO3Group<double>& R_cw, const Vector3d& Tw, GpuMat* pPts, GpuMat* pNls, GpuMat* pRadiance_, GpuMat* pNR_) const;

		void displayBoundingBox() const;
	public:
		Vector3d _Ow; //its the float absolute coordinate in world of the front up left corner of the TSDF volume
		SO3Group<double> _R_fw;

		//data
		//volume data
		//the up left front corner of the volume defines the origin of the world coordinate
		//and follows the right-hand cv-convention
		//physical size of the volume
		float3 _fTSDFSize;//in meter
		short3 _TSDFResolution; //in meter
		unsigned int _uTSDFLevelXY;
		unsigned int _uTSDFTotal;
		float _fTSDFVoxelSizeM; //in meter
		//truncated distance in meter
		//must be larger than 2*voxelsize 
		float _fTruncateDistanceM;

		GpuMat _gpu_YXxZ_tsdf_volume;
		Mat _YXxZ_tsdf_volume; //y*z,x,CV_32FC1,x-first
		GpuMat _gpu_YXxZ_normalized_radiance;
		Mat _YXxZ_radiance_volume;
		GpuMat _gpu_YXxZ_radiance;

		//for rendering 
		Intr _intr;
	};




}//geometry
}//btl
#endif

