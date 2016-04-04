#define INFO
#define TIMER
#define _USE_MATH_DEFINES
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "Utility.hpp"

//camera calibration from a sequence of images
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <OpenNI.h>

#include "Kinect.h"
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "IntrinsicAnalysis.cuh"
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "TSDF.h"
#include "KinfuTracker.h"

#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"
#include "DataLive.h"
#include "MultiViewer.h"
#include <qapplication.h>
#include <qsplitter.h>

using namespace std;

int main(int argc, char** argv)
{
  // Read command lines arguments.
  QApplication application(argc,argv);

  // Create Splitters
  QSplitter *hSplit  = new QSplitter(Qt::Horizontal);
  QSplitter *vSplit1 = new QSplitter(Qt::Vertical,hSplit);
  QSplitter *vSplit2 = new QSplitter(Qt::Vertical,vSplit1);

  hSplit->resize(1280,720);

  DataLive::tp_shared_ptr _pData( new DataLive() );

  // Instantiate the viewers.
  Viewer global_view(string("global_view"), _pData, vSplit1, NULL);
  Viewer rgb_view(string("rgb_view"), _pData, vSplit1, &global_view);
  Viewer camera_view(string("camera_view"), _pData, vSplit1, &global_view);
  Viewer depth_view (string("depth_view"), _pData, vSplit1, &global_view);

  hSplit->addWidget(&global_view);
  hSplit->addWidget(vSplit1);
  hSplit->setStretchFactor(0, 3);
  hSplit->setStretchFactor(1, 1);
  vSplit1->addWidget(&rgb_view);
  vSplit1->addWidget(vSplit2);
  vSplit2->addWidget(&depth_view);
  vSplit2->addWidget(&camera_view);

#if QT_VERSION < 0x040000
  // Set the viewer as the application main widget.
  application.setMainWidget(&viewer);
#else
  hSplit->setWindowTitle("Kinect Multi-view");
#endif

  try{
	  // Make the viewer window visible on screen.
	  hSplit->show();
	  // Run main loop.
	  return application.exec();
  }
  catch ( btl::utility::CError& e )  {
	  if ( std::string const* mi = boost::get_error_info< btl::utility::CErrorInfo > ( e ) ) {
		  std::cerr << "Error Info: " << *mi << std::endl;
	  }
  }
  catch ( std::runtime_error& e ){
	  PRINTSTR( e.what() );
  }
}
