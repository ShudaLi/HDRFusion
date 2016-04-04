#ifndef BTL_CAMERA
#define BTL_CAMERA
#include "DllExportDef.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

namespace btl{ namespace image {

struct DLL_EXPORT SCamera
{
	//type
	typedef boost::shared_ptr<SCamera> tp_shared_ptr;
	typedef boost::scoped_ptr<SCamera> tp_scoped_ptr;
	typedef SCamera* tp_ptr;

	//constructor
	//************************************
	// Method:    SCamera
	// FullName:  btl::image::SCamera::SCamera
	// Access:    public 
	// Returns:   na
	// Qualifier: 
	// Parameter: const std::string & strCamParam_: the yml file stores the camera internal parameters
	// Parameter: ushort uResolution_: the resolution level, where 0 is the original 1 is by half, 2 is the half of haly so on
	//************************************
	SCamera(const std::string& strCamParam_,ushort uResolution_ = 0,const string& path_ = string("..\\Data\\") );//0 480x640
	~SCamera();
	//methods
	void loadTexture ( const cv::Mat& cvmImg_, GLuint* puTesture_ );
	void setGLProjectionMatrix ( const double dNear_, const double dFar_ );
	void renderCameraInLocal ( const GpuMat& gpu_img_, btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, float fPhysicalFocalLength_ = .02f, bool bRenderTexture_=true ) ;
	void renderPointOnImageLocal(const float fX_, const float fY_, const float fPhysicalFocalLength_ = 0.1f);
	void importYML(const std::string& strCamParam_);

	cv::Mat getcvmK(){
		cv::Mat cvmK = ( cv::Mat_<float>(3,3) << _fFx, 0.f , _u,
												 0.f,  _fFy, _v,
										 		 0.f,  0.f , 1.f );
		return cvmK;
	}

	//camera parameters
	ushort _uResolution;
	float _fFx, _fFy, _u, _v; 

	unsigned short _sWidth, _sHeight;
	cv::Mat _cvmDistCoeffs;
	//for undistortion
	cv::cuda::GpuMat  _cvgmMapX;
	cv::cuda::GpuMat  _cvgmMapY;
	//type
private:
	bool _bIsUndistortionOn;
	GLUquadricObj*   _quadratic;	// Storage For Our Quadratic Objects
};

}//image
}//btl
#endif
