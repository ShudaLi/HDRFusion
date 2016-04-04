#ifndef _DATA4VIEWER_H_
#define _DATA4VIEWER_H_

using namespace btl::gl_util;
using namespace btl::kinect;
using namespace btl::geometry;
using namespace std;
using namespace Eigen;

void convert(const Eigen::Affine3f& eiM_, Mat* pM_);
void convert(const Eigen::Matrix4f& eiM_, Mat* pM_);
void convert(const Mat& M_, Eigen::Affine3f* peiM_);
void convert(const Mat& M_, Eigen::Matrix4f* peiM_);


class Data4Viewer
{
public:
	typedef boost::shared_ptr<Data4Viewer> tp_shared_ptr;

	Data4Viewer();
	virtual ~Data4Viewer();
	virtual void init();
	virtual void loadFromYml();
	virtual void reset();
	virtual void updatePF(){;}

	virtual void drawGlobalView();
	virtual void drawCameraView(qglviewer::Camera* pCamera_);
	virtual void drawRGBView();
	virtual void drawDepthView(qglviewer::Camera* pCamera_);

	virtual void setAsPrevPos();
	virtual void switchCapturing() { _bCapture = !_bCapture; }
	virtual void switchViewPosLock() { _bViewLocked = !_bViewLocked; }
	virtual void switchShowTexts() { _bShowText = !_bShowText; }
	virtual void switchShowSurfaces() { _bShowSurfaces = !_bShowSurfaces; }
	virtual void switchPyramid() { _pGL->_usLevel = ++_pGL->_usLevel%_pGL->_usPyrHeight; }
	virtual void switchLighting() { _pGL->_bEnableLighting = !_pGL->_bEnableLighting; }
	virtual void switchImgPlane() { _pGL->_bDisplayCamera = !_pGL->_bDisplayCamera; }
	virtual void switchReferenceFrame() { _bRenderReference = !_bRenderReference; }
	virtual void switchTrackOnly() { _bTrackingOnly = !_bTrackingOnly; }
	virtual void switchCameraPath() { _bIsCameraPathOn = !_bIsCameraPathOn; }
	virtual void switchCurrentFrame() { _bIsCurrFrameOn = !_bIsCurrFrameOn; }
	virtual void switchContinuous() { _bContinuous = !_bContinuous; }
	virtual void switchShowCamera() { _bShowCamera = !_bShowCamera; }
	virtual const bool isCapturing() const { return _bCapture; }
	virtual const bool isTrackOnly() const { return _bTrackingOnly; }

	void importRT(Eigen::Vector3d& r_ , Eigen::Vector3d& T_);
	void exportRT(const Sophus::SO3Group<double>& R_, const Eigen::Vector3d& T_);
	btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
	btl::gl_util::CGLUtil::tp_shared_ptr _pGL;

	btl_img::SCamera::tp_scoped_ptr _pVirtualCamera;

	GpuMat _Pts; //for display depth view
	GpuMat _Nls; //for display depth view

	btl::kinect::CKeyFrame::tp_scoped_ptr _pVirtualGlobal;
	btl::kinect::CKeyFrame::tp_scoped_ptr _pVirtualCameraView;
	btl::kinect::CKeyFrame::tp_scoped_ptr _pCameraView2;

	btl::geometry::CTsdfBlock::tp_shared_ptr _pGlobalMap;
	GLuint _uTexture;

	std::string _strFeatureName;
	ushort _uResolution;
	ushort _uPyrHeight;
	Eigen::Vector3d _Cw; //initial camera centre in world
	vector<short> _vXYZCubicResolution;
	float _fVolumeSize;
	int _nMode;//btl::kinect::VideoSourceKinect::PLAYING_BACK
	float _fTimeLeft;
	int _nStatus;//1 restart; 2 //recording continue 3://pause 4://dump

	bool _bDisplayImage;
	bool _bLightOn;
	bool _bRenderReference;
	bool _bCapture; // controled by c
	bool _bContinuous; 
	bool _bTrackingOnly;
	bool _bViewLocked; // controlled by 2
	//bool _bShowRelocalizaitonFeatures;
	bool _bShowSurfaces;
	bool _bShowCamera;
	bool _bShowText;
	bool _bIsCameraPathOn; 
	bool _bIsCurrFrameOn; 
	bool _bLoadVolume;
	bool _bStorePoses;
	bool _bSave;
	bool _bToneMapper;

	bool _bExportRT;

	double _dZoom;
	double _dXAngle;
	double _dYAngle;
	Eigen::Affine3d _eimModelViewGL;

	vector< Vector2f > _vError;

	Eigen::Affine3d _prj_pov_f_w; 
	Eigen::Affine3d _prj_w_f_pov;

	bool _bInitialized;
	string _cam_param_path;
	vector<float> _vInitial_Cam_Pos; //intial camera position
	int _nIdx;
	float _fWeight; // the weight of similarity score and pair-wise score
public:
	short _sWidth;
	short _sHeight;
	int _nRound;
	string _strStage; // 
	Eigen::Affine3d _prj_c_f_w;
	int _nIdxPairs; 
	bool _bCameraFollow;
	String _result_folder;
	btl::geometry::CKinFuTracker::tp_shared_ptr _pTracker;

	GLUquadricObj*   _quadratic;	// Storage For Our Quadratic Objects
	btl::device::CIntrinsicsAnalysisMult::tp_shared_ptr _pIAVirtual;
};


#endif
