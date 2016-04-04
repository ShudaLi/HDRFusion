//display kinect depth in real-time
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

#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <Eigen/Dense>
//flann
#include "QGLViewer/manipulatedCameraFrame.h"

//camera calibration from a sequence of images
#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include "VideoSourceKinect.hpp"

//Qt
#include <QResizeEvent>
#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"
#include "DataLive.h"
#include "MultiViewer.h"
#include <QCoreApplication>
using namespace qglviewer;
using namespace std;
Viewer::Viewer(string strName_, DataLive::tp_shared_ptr pData, QWidget* parent, const QGLWidget* shareWidget)
:QGLViewer(parent, shareWidget)
{
	_pData = pData;
	_strViewer = strName_;
	_bShowText = true;

	// Forbid rotation
	if(!_strViewer.compare("global_view") ){
		setAxisIsDrawn(false);
		setFPSIsDisplayed();
		setGridIsDrawn(false);
	}
	else
	{
		setAxisIsDrawn(false);
		setGridIsDrawn(false);
		WorldConstraint* constraint = new WorldConstraint();
		constraint->setRotationConstraintType(AxisPlaneConstraint::FORBIDDEN);
		constraint->setTranslationConstraintType(AxisPlaneConstraint::FORBIDDEN);
		camera()->frame()->setConstraint(constraint);
	}
}
Viewer::~Viewer()
{
}

void Viewer::draw()
{
	if(!_strViewer.compare("global_view"))	{
		_pData->updatePF();
		_pData->drawGlobalView();
		if(_bShowText){
			float aColor[4] = {0.f,1.f,1.f,1.f};	glColor4fv(aColor);
			ostringstream expo;
			expo << "exposure: " << _pData->_pKinect->_exposure << " ms";
			renderText(25, 180, QString(expo.str().c_str()), QFont("Arial", 33, QFont::Bold));
		}
	}
	return;
}

void Viewer::init()
{
	// Restore previous viewer state.
	//restoreStateFromFile();
	// Opens help window
	//help();
	startAnimation();
	//
	_pData->init();

}//init()

QString Viewer::helpString() const
{
	QString text("<h2>S i m p l e V i e w e r</h2>");
	text += "Use the mouse to move the camera around the object. ";
	text += "You can respectively revolve around, zoom and translate with the three mouse buttons. ";
	text += "Left and middle buttons pressed together rotate around the camera view direction axis<br><br>";
	text += "Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
	text += "Simply press the function key again to restore it. Several keyFrames define a ";
	text += "camera path. Paths are saved when you quit the application and restored at next start.<br><br>";
	text += "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
	text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. ";
	text += "See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>";
	text += "Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). ";
	text += "A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>";
	text += "A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. ";
	text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";
	text += "Press <b>Escape</b> to exit the viewer.";
	return text;
}

void Viewer::keyPressEvent(QKeyEvent *pEvent_)
{
	using namespace btl::kinect;
	using namespace Eigen;
	// Defines the Alt+R shortcut.
	
    if (pEvent_->key() == Qt::Key_1)
	{
		_bShowText = !_bShowText;
		updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_4 && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		_pData->switchShowTexts();
		updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_R && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		_pData->loadFromYml();
		_pData->reset();
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_Escape && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		QCoreApplication::instance()->quit();
	}
	else if (pEvent_->key() == Qt::Key_F && (pEvent_->modifiers() & Qt::ShiftModifier)){
		if (!isFullScreen()){
			toggleFullScreen();
		}
		else{
			setFullScreen(false);
			resize(1280, 480);
		}
	}
	QGLViewer::keyPressEvent(pEvent_);
}


void Viewer::mouseDoubleClickEvent(QMouseEvent* e)
{
	if ( e->button() == Qt::LeftButton )
	{
		_pData->_mx = e->localPos().x() / width() * __aRGBW[_pData->_uResolution];
		_pData->_my = e->localPos().y() / height()* __aRGBH[_pData->_uResolution];
		_pData->_bCapturePoint = true;
		updateGL();
	}

	//QGLViewer::mouseDoubleClickEvent(e);
}



