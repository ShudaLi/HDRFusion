#ifndef _DATA4VIEWER_H_
#define _DATA4VIEWER_H_

using namespace btl::gl_util;
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

	virtual void switchShowTexts() { _bShowText = !_bShowText; }

	btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
	btl::gl_util::CGLUtil::tp_shared_ptr _pGL;

	GLuint _uTexture;

	ushort _uResolution;
	ushort _uPyrHeight;
	Eigen::Vector3d _Cw; //initial camera centre in world
	int _nRansacIterationasTracking;
	int _nRansacIterationasRelocalisation;
	bool _bRepeat;// repeatedly play the sequence 
	int _nStatus;//1 restart; 2 //recording continue 3://pause 4://dump
	bool _bShowText;

	float _mx, _my;//mouse pointer position
	bool _bCapturePoint;
	vector<float3> _vXw; //for measuring point distance

	int _nMethodIdx;

	double _dZoom;
	double _dXAngle;
	double _dYAngle;
	Eigen::Affine3d _eimModelViewGL;

	string _cam_param_path;
	vector<float> _vInitial_Cam_Pos; //intial camera position
public:
	short _sWidth;
	short _sHeight;
};


#endif
