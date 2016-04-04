#ifndef BTL_CAMERA
#define BTL_CAMERA
#include "DllExportDef.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace Sophus;

namespace btl{ namespace image {

struct DLL_EXPORT SCamera
{
	//type
	typedef boost::shared_ptr<SCamera> tp_shared_ptr;
	typedef boost::scoped_ptr<SCamera> tp_scoped_ptr;
	typedef SCamera* tp_ptr;

	SCamera(const std::string& strCamParam_,ushort uResolution_ = 0,const string& path_ = string("..\\Data\\") );//0 480x640
	~SCamera();
	//methods
	void loadTexture ( const cv::Mat& cvmImg_, GLuint* puTesture_ );
	void setGLProjectionMatrix ( const double dNear_, const double dFar_ );
	void renderCameraInLocal ( const GpuMat& gpu_img_, btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, float fPhysicalFocalLength_ = .02f, bool bRenderTexture_=true ) ;
	void importYML(const std::string& strCamParam_);

	pcl::device::Intr getIntrinsics(int lvl){
		return pcl::device::Intr(_fFx, _fFy, _u, _v)(lvl);
	}

	//camera parameters
	ushort _uResolution;
	float _fFx, _fFy, _u, _v; 
	unsigned short _sWidth, _sHeight;
	//type
private:
	GLUquadricObj*   _quadratic;	// Storage For Our Quadratic Objects
};

}//image
}//btl
#endif
