#ifndef _KINECT_LIVEVIEWER_QGLV_APP_H_
#define _KINECT_LIVEVIEWER_QGLV_APP_H_

using namespace btl::gl_util;
using namespace btl::kinect;
using namespace std;
using namespace Eigen;

class DataLive: public Data4Viewer
{
public:
	typedef boost::shared_ptr<DataLive> tp_shared_ptr;

	DataLive();
	virtual ~DataLive(){ ; }
	virtual void loadFromYml();
	virtual void reset();
	virtual void updatePF();
};


#endif