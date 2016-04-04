#ifndef BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
#define BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT

#include "DllExportDef.h"

//#define INFO
using namespace openni;

namespace btl{
namespace kinect{

namespace btl_img = btl::image;
namespace btl_knt = btl::kinect;
using namespace Sophus;

class DLL_EXPORT VideoSourceKinect 
{
public:
	//type
	typedef boost::shared_ptr<VideoSourceKinect> tp_shared_ptr;
	enum tp_mode { SIMPLE_CAPTURING = 1, RECORDING = 2, PLAYING_BACK = 3};
	enum tp_status { CONTINUE=01, PAUSE=02, MASK1 =07, START_RECORDING=010, STOP_RECORDING=020, CONTINUE_RECORDING=030, DUMP_RECORDING=040, MASK_RECORDER = 070 };
	enum tp_raw_data_processing_methods {BIFILTER_IN_ORIGINAL = 0,BIFILTER_IN_DISPARITY };
	//constructor
    VideoSourceKinect(ushort uResolution_, ushort uPyrHeight_, const Vector3d& eivCw_,const string& cam_param_path_ );
    virtual ~VideoSourceKinect();
	void initKinect();
	void initRecorder(std::string& strPath_);
	virtual void initPlayer(std::string& strPathFileName_);
	// 1. need to call getNextFrame() before hand
	// 2. RGB color channel (rather than BGR as used by cv::imread())
	virtual bool getNextFrame(int* pnStatus_);

	// 0 VGA
	// 1 QVGA
	Status setVideoMode(ushort uLevel_);
	void setDumpFileName( const std::string& strFileName_ ){_strDumpFileName = strFileName_;}

protected:
	virtual void gpuBuildPyramidUseNICVm();
public:
	//for undistort depth
	string _serial_number;
	//parameters
	float _fThresholdDepthInMeter; //threshold for filtering depth
	float _fSigmaSpace; //degree of blur for the bilateral filter
	float _fSigmaDisparity; 
	unsigned int _uPyrHeight;//the height of pyramid
	ushort _uResolution;//0 640x480; 1 320x240; 2 160x120 3 80x60
	float _fMtr2Depth; // 100

	//cameras
	btl_img::SCamera::tp_scoped_ptr _pRGBCamera;
	btl_img::SCamera::tp_scoped_ptr _pDispCamera;

	btl_knt::CKeyFrame::tp_scoped_ptr _pCurrFrame;
	//rgb
	cv::Mat			_cvmRGB;
	Mat				_cvmDep;

protected:
	//openni
    Device _device;
	openni::PlaybackControl* _pPlaybackControl;
    VideoStream _color;
    VideoStream _depth;
	VideoStream** _streams;//array of pointers
	Recorder _recorder;
	
	VideoFrameRef _depthFrame;
	VideoFrameRef _colorFrame;

	const openni::SensorInfo* _depthSensorInfo;
	const openni::SensorInfo* _colorSensorInfo;
	const openni::SensorInfo* _irSensorInfo;

	cv::cuda::GpuMat _gRGB;
	//depth
    cv::Mat         _cvmDepthFloat;
	cv::cuda::GpuMat _gDepth;
	boost::shared_ptr<GpuMat> _acvgmShrPtrPyr32FC1Tmp[4];
	GpuMat _tmp, _tmp2;
	// duplicated camera parameters for speed up the VideoSourceKinect::align() in . because Eigen and cv matrix class is very slow.

	//controlling flag
	static bool _bIsSequenceEnds;
	std::string _strDumpFileName;
	int _nMode; 

	float _fCutOffDistance;
	// (opencv-default camera reference system convention)
	Eigen::Vector3d _Cw;
	string _cam_param_path_file_name;
	int _frameId; //for '*.oni' playback mode
	unsigned int time_stamp;

	int depthWidth;
	int depthHeight;
	int colorWidth;
	int colorHeight;

};//class VideoSourceKinect

} //namespace kinect
} //namespace btl



#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
