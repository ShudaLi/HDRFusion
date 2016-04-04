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

