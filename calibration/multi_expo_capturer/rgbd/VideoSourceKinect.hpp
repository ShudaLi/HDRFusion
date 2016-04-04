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

#ifndef BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
#define BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT

#include "DllExportDef.h"
//#define INFO
using namespace openni;

namespace btl{
namespace kinect{

namespace btl_img = btl::image;
namespace btl_knt = btl::kinect;
using namespace Eigen;
class DLL_EXPORT VideoSourceKinect 
{
public:
	//type
	typedef boost::shared_ptr<VideoSourceKinect> tp_shared_ptr;
	//constructor
    VideoSourceKinect(ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,const Vector3d& eivCw_,const string& cam_param_path_ );
    virtual ~VideoSourceKinect();
	void initKinect();
	// 1. need to call getNextFrame()
	// 2. RGB color channel (rather than BGR as used by cv::imread())
	virtual bool getNextFrame(int* pnStatus_);
	// 0 VGA
	// 1 QVGA
	Status setVideoMode(ushort uLevel_);

protected:
	// convert the depth map/ir camera to be aligned with the rgb camera
	virtual void init();
public:
	string _serial_number;
	//parameters
	int _exposure;
	int _gain;
	unsigned int _uPyrHeight;//the height of pyramid
	ushort _uResolution;//0 640x480; 1 320x240; 2 160x120 3 80x60

	//cameras
	btl_img::SCamera::tp_scoped_ptr _pRGBCamera;
	//rgb
	cv::Mat			_cvmRGB; 
	Mat				_cvmDep;
	int _nTotalFrameToAccumulate;
protected:
	//openni
    Device _device;
    VideoStream _color;
    VideoStream _depth;
	VideoStream** _streams;//array of pointers
	
	VideoFrameRef _depthFrame;
	VideoFrameRef _colorFrame;

	const openni::SensorInfo* _depthSensorInfo;
	const openni::SensorInfo* _colorSensorInfo;
	const openni::SensorInfo* _irSensorInfo;


	// (opencv-default camera reference system convention)
	Eigen::Vector3d _Cw;
	string _cam_param_path_file_name;
	int _frameId; //for '*.oni' playback mode
	unsigned int time_stamp;

};//class VideoSourceKinect

} //namespace kinect
} //namespace btl



#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
