#define EXPORT

#define INFO
#define DEFAULT_TRIANGLES_BUFFER_SIZE 
#define _USE_MATH_DEFINES
#define  NOMINMAX 
//gl
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//boost
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
//stl
#ifdef __gnu_linux__
#include <sys/types.h>
#include <sys/stat.h>
#elif _WIN32
#include <direct.h>
#else 
#error "OS not supported!"
#endif
#include <vector>
#include <fstream>
#include <list>
#include <limits>
#include <cstring>
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda/common.hpp>
//eigen
#include <Eigen/Core>
#include <se3.hpp>

//self
#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "Utility.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "Kinect.h"
#include "IntrinsicAnalysis.cuh"
#include "KeyFrame.h"
#include "TSDF.h"
#include "CudaLib.cuh"
#include "VolumeColor.cuh"
#include "RayCasterColor.cuh"

namespace btl{ namespace geometry
{

using namespace cv;
using namespace cv::cuda;
using namespace btl::kinect;
using namespace std;
using namespace pcl::device;
using namespace Sophus;
//WD is the total extra voxels to solve the border issue.  
#define WD 32

CTsdfBlock::CTsdfBlock(const short3& TSDFResolution_, const float TSDFSize_,
						const Intr& intr_,
						const SO3Group<double>& R_fw_,
						const Vector3d& Ow_ )
						:_TSDFResolution(TSDFResolution_), _R_fw(R_fw_), _Ow(Ow_), _intr(intr_)
{
	_fTSDFSize = make_float3(TSDFSize_, TSDFSize_, TSDFSize_);
	_uTSDFLevelXY = _TSDFResolution.x*_TSDFResolution.y;
	_uTSDFTotal = _uTSDFLevelXY*_TSDFResolution.z;
	_fTSDFVoxelSizeM = TSDFSize_ / _TSDFResolution.x;
	_fTruncateDistanceM = _fTSDFVoxelSizeM*8;

	_YXxZ_tsdf_volume.create(_uTSDFLevelXY, _TSDFResolution.z, CV_16SC2);//y*x rows,z cols
	_YXxZ_radiance_volume.create(_uTSDFLevelXY, _TSDFResolution.z, CV_32FC4);//y*x rows,z cols

	_gpu_YXxZ_tsdf_volume.upload(_YXxZ_tsdf_volume);
	_gpu_YXxZ_tsdf_volume.setTo(Scalar_<short>(32767, 0));
	_gpu_YXxZ_normalized_radiance.upload(_YXxZ_radiance_volume);
	_gpu_YXxZ_normalized_radiance.setTo(Scalar_<float>(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN())); //float qnan, uchar 255, uchar 255
	_gpu_YXxZ_radiance.upload(_YXxZ_radiance_volume);
	_gpu_YXxZ_radiance.setTo(Scalar_<float>(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN())); //float qnan, uchar 255, uchar 255
	return;
}

CTsdfBlock::~CTsdfBlock(void)
{
	return;
}

void CTsdfBlock::reset(){
	//pcl::device::cuda_init_volume(&_gpu_YXxZ_tsdf_volume, _TSDFResolution);
	_gpu_YXxZ_tsdf_volume.setTo(Scalar_<short>(32767, 0));
	return;
}

void CTsdfBlock::gpuIntegrateDepth(const GpuMat& scaled_depth, const GpuMat& normal_, const GpuMat& normalized_radiance_, const GpuMat& radiance_, const GpuMat& err_,
									SE3Group<double>& T_cw ){
	
	SE3Group<double> T_cf = T_cw;
	Matrix3d R_cf = T_cf.so3().matrix();
	pcl::device::Matr33d& devR_cf = pcl::device::device_cast<pcl::device::Matr33d> (R_cf); //device cast do the inverse implicitly because eimcmRwCur is col major by default.

	Vector3d Of = T_cf.so3().inverse()*(-T_cf.translation()); //camera centre in tsdf coordinate (f)
	double3& devOf = pcl::device::device_cast<double3> (Of);

	btl::device::cuda_fuse_depth_radiance_normal(scaled_depth, normal_, normalized_radiance_, radiance_, err_,
											_fTSDFVoxelSizeM, _fTruncateDistanceM,
											devR_cf, devOf,//transform local tsdf coordinate s to camera coordinate c
											_intr, _TSDFResolution,
											&_gpu_YXxZ_tsdf_volume, &_gpu_YXxZ_normalized_radiance, &_gpu_YXxZ_radiance );
	
	return;
}

void CTsdfBlock::gpuRayCastingRadiance(const Intr& intr, const SO3Group<double>& R_cw, const Vector3d& Tw,
	GpuMat* pPts, GpuMat* pNls, GpuMat* pRadiance_) const {
	pPts->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	pNls->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	if (pRadiance_ && !pRadiance_->empty())
		pRadiance_->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));

	//get VMap and NMap in world
	SO3Group<double> R_cf = R_cw * _R_fw.inverse();
	SO3Group<double> R_cf_t = R_cf.inverse();
	Matrix3d R_cf_tmp = R_cf.matrix();
	Matrix3d R_cf_t_tmp = R_cf_t.matrix();
	pcl::device::Matd33& devR_cf_t = pcl::device::device_cast<pcl::device::Matd33> (R_cf_tmp);	//device cast do the inverse implicitly because eimcmRwCur is col major by default.
	pcl::device::Matd33& devR_cf = pcl::device::device_cast<pcl::device::Matd33> (R_cf_t_tmp);	//device cast do the inverse implicitly because eimcmRwCur is col major by default.
	//Cw = -Rw'*Tw
	Eigen::Vector3d CwCur = R_cw.inverse() * (-Tw); //camera centre of current frame in world coordinate.
	Vector3d C_f = _R_fw*(CwCur - _Ow); //camera centre in tsdf coordinate
	double3& devC_f = pcl::device::device_cast<double3> (C_f);
	btl::device::cuda_ray_casting_radiance(intr, devR_cf_t, devR_cf, devC_f,
		_fTruncateDistanceM, _fTSDFVoxelSizeM,
		_TSDFResolution, _fTSDFSize,
		_gpu_YXxZ_tsdf_volume, _gpu_YXxZ_radiance, 
		&*pPts, &*pNls,
		&*pRadiance_);
	return;
}
void CTsdfBlock::gpuRayCastingNR(const Intr& intr, const SO3Group<double>& R_cw, const Vector3d& Tw, 
	GpuMat* pPts, GpuMat* pNls, GpuMat* pRadiance_) const {
	pPts->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	pNls->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	if (pRadiance_ && !pRadiance_->empty())
		pRadiance_->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));

	//get VMap and NMap in world
	SO3Group<double> R_cf = R_cw * _R_fw.inverse();
	SO3Group<double> R_cf_t = R_cf.inverse();
	Matrix3d R_cf_tmp = R_cf.matrix();
	Matrix3d R_cf_t_tmp = R_cf_t.matrix();
	pcl::device::Matd33& devR_cf_t = pcl::device::device_cast<pcl::device::Matd33> (R_cf_tmp);	//device cast do the inverse implicitly because eimcmRwCur is col major by default.
	pcl::device::Matd33& devR_cf = pcl::device::device_cast<pcl::device::Matd33> (R_cf_t_tmp);	//device cast do the inverse implicitly because eimcmRwCur is col major by default.
	//Cw = -Rw'*Tw
	Eigen::Vector3d CwCur = R_cw.inverse() * (-Tw ); //camera centre of current frame in world coordinate.
	Vector3d C_f = _R_fw*(CwCur - _Ow); //camera centre in tsdf coordinate
	double3& devC_f = pcl::device::device_cast<double3> (C_f);
	btl::device::cuda_ray_casting_radiance(intr, devR_cf_t, devR_cf, devC_f,
											_fTruncateDistanceM, _fTSDFVoxelSizeM,
											_TSDFResolution, _fTSDFSize,
											_gpu_YXxZ_tsdf_volume, _gpu_YXxZ_normalized_radiance, 
											&*pPts, &*pNls,
											&*pRadiance_);
	return;
}

void CTsdfBlock::gpuRayCastingAll(const Intr& intr, const SO3Group<double>& R_cw, const Vector3d& Tw,
	GpuMat* pPts, GpuMat* pNls, GpuMat* pRadiance_, GpuMat* pNR_) const {
	pPts->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	pNls->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	if (pRadiance_ && !pRadiance_->empty()){
		pRadiance_->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	}
	if (pNR_ && !pNR_->empty()){
		pNR_->setTo(Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
	}
	//get VMap and NMap in world
	SO3Group<double> R_cf = R_cw * _R_fw.inverse();
	SO3Group<double> R_cf_t = R_cf.inverse();
	Matrix3d R_cf_tmp = R_cf.matrix();
	Matrix3d R_cf_t_tmp = R_cf_t.matrix();
	pcl::device::Matd33& devR_cf_t = pcl::device::device_cast<pcl::device::Matd33> (R_cf_tmp);	//device cast do the inverse implicitly because eimcmRwCur is col major by default.
	pcl::device::Matd33& devR_cf = pcl::device::device_cast<pcl::device::Matd33> (R_cf_t_tmp);	//device cast do the inverse implicitly because eimcmRwCur is col major by default.
	//Cw = -Rw'*Tw
	Eigen::Vector3d CwCur = R_cw.inverse() * (-Tw); //camera centre of current frame in world coordinate.
	Vector3d C_f = _R_fw*(CwCur - _Ow); //camera centre in tsdf coordinate
	double3& devC_f = pcl::device::device_cast<double3> (C_f);
	btl::device::cuda_ray_casting_all(intr, devR_cf_t, devR_cf, devC_f,
		_fTruncateDistanceM, _fTSDFVoxelSizeM,
		_TSDFResolution, _fTSDFSize,
		_gpu_YXxZ_tsdf_volume, _gpu_YXxZ_normalized_radiance, _gpu_YXxZ_radiance,
		&*pPts, &*pNls,
		&*pRadiance_, &*pNR_);
	return;
}

void CTsdfBlock::displayBoundingBox() const
{

	Vector3d Vs[8];

	Vs[0] = Vector3d::Zero();

	Vs[1] = Vector3d(Vs[0]);
	Vs[1][0] += _fTSDFSize.x;

	Vs[2] = Vector3d(Vs[1]);
	Vs[2][2] += _fTSDFSize.z;

	Vs[3] = Vector3d(Vs[0]);
	Vs[3][2] += _fTSDFSize.z;

	Vs[4] = Vector3d(Vs[0]);
	Vs[4][1] += _fTSDFSize.y;

	Vs[5] = Vector3d(Vs[4]);
	Vs[5][0] += _fTSDFSize.x;

	Vs[6] = Vector3d(Vs[5]);
	Vs[6][2] += _fTSDFSize.z;

	Vs[7] = Vector3d(Vs[4]);
	Vs[7][2] += _fTSDFSize.z;

	Vector3d Vw[8];
	for (int i = 0; i < 8; i++){
		Vw[i] = _R_fw.inverse()*Vs[i] + _Ow;
	}
	//top
	glBegin(GL_LINE_LOOP);
	glVertex3dv(Vw[0].data());
	glVertex3dv(Vw[1].data());
	glVertex3dv(Vw[2].data());
	glVertex3dv(Vw[3].data());
	glEnd();
	//bottom
	glBegin(GL_LINE_LOOP);
	glVertex3dv(Vw[4].data());
	glVertex3dv(Vw[5].data());
	glVertex3dv(Vw[6].data());
	glVertex3dv(Vw[7].data());
	glEnd();
	//middle
	glBegin(GL_LINES);
	glVertex3dv(Vw[0].data());
	glVertex3dv(Vw[4].data());
	glEnd();
	glBegin(GL_LINES);
	glVertex3dv(Vw[1].data());
	glVertex3dv(Vw[5].data());
	glEnd();
	glBegin(GL_LINES);
	glVertex3dv(Vw[2].data());
	glVertex3dv(Vw[6].data());
	glEnd();
	glBegin(GL_LINES);
	glVertex3dv(Vw[3].data());
	glVertex3dv(Vw[7].data());
	glEnd();
}

}//geometry
}//btl
