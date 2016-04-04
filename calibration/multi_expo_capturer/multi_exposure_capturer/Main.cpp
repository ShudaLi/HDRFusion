//display kinect depth in real-time
#define INFO
#define TIMER
#define _USE_MATH_DEFINES
#include <GL/glew.h>
//#include <cuda.h>
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

//camera calibration from a sequence of images
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
//flann
#include <OpenNI.h>

#include "Kinect.h"
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "VideoSourceKinect.hpp"

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
  hSplit->resize(640,480);
  DataLive::tp_shared_ptr _pData( new DataLive() );
  // Instantiate the viewers.
  Viewer global_view(string("global_view"), _pData, hSplit, NULL);

  hSplit->addWidget(&global_view);

#if QT_VERSION < 0x040000
  // Set the viewer as the application main widget.
  application.setMainWidget(&viewer);
#else
  hSplit->setWindowTitle("RGB-D Multi-exposure");
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
	  cout<<( e.what() );
  }
}
