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

#define EXPORT
#define INFO
#define _USE_MATH_DEFINES
#define  NOMINMAX 
//gl
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//opencv
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <opencv2/cudaarithm.hpp>
#include <Eigen/Dense>
#include <se3.hpp>

#include "OtherUtil.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include "Camera.h"
#include <string>
#include "Kinect.h"
#include "CVUtil.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

using namespace btl::image;
using namespace Sophus;

SCamera::SCamera( const string& strCamParamPathFileName_, ushort uResolution_, const string& path_ /*= string("..\\..\\Data\\")*/ )
:/*_eType(eT_),*/ _uResolution(uResolution_)
{
	importYML(strCamParamPathFileName_);
	_quadratic = gluNewQuadric();                // Create A Pointer To The Quadric Object ( NEW )
	// Can also use GLU_NONE, GLU_FLAT
	gluQuadricNormals ( _quadratic, GLU_SMOOTH ); // Create Smooth Normals
	gluQuadricTexture ( _quadratic, GL_TRUE );   // Create Texture Coords ( NEW )
}

SCamera::~SCamera()
{
	gluDeleteQuadric(_quadratic);
}

void SCamera::importYML(const std::string& cam_param_file_name_)
{
	// create and open a character archive for output
	cv::FileStorage cFSRead ( cam_param_file_name_, cv::FileStorage::READ );

	if (!cFSRead.isOpened()) 
	{
		std::cout << "Load camera parameter failed.";
		return;
	}

	cFSRead ["_fFx"] >> _fFx;
	cFSRead ["_fFy"] >> _fFy;
	cFSRead ["_u"] >> _u;
	cFSRead ["_v"] >> _v;
	cFSRead ["_sWidth"]  >> _sWidth;
	cFSRead ["_sHeight"] >> _sHeight;
	cFSRead.release();

	if (_uResolution<6)
	{
		int div = 1 << _uResolution;
		_fFx/=div;
		_fFy/=div;
		_u/=div;
		_v/=div;

		_sWidth/=div;
		_sHeight/=div;
	}
	else if( _uResolution == 6)
	{
		_fFx *= 2;
		_fFy *= 1024.f/480.f;
		_u *= 2;
		_v *= 1024.f/480.f;
		_sWidth = 1280;
		_sHeight = 1024;
	}
	
	return;
}
void SCamera::setGLProjectionMatrix ( const double dNear_, const double dFar_ )
{
//    glutReshapeWindow( int ( dWidth ), int ( dHeight ) );

    double f = ( _fFx + _fFy ) / 2.;
    //no need to times nScaleViewport_ factor, because v/f, -(dHeight -v)/f cancel the factor off.
    double dLeft, dRight, dBottom, dTop;
    //Two assumptions:
    //1. assuming the principle point is inside the image
    //2. assuming the x axis pointing right and y axis pointing upwards
    dTop    =                _v   / f;
    dBottom = - ( _sHeight - _v ) / f;
    dLeft   =              - _u   / f;
    dRight  =   ( _sWidth  - _u ) / f;

	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity(); //use the same style as page 130, opengl programming guide
    glFrustum ( dLeft * dNear_, dRight * dNear_, dBottom * dNear_, dTop * dNear_, dNear_, dFar_ );// the corners of the near clipping plane in GL system

    return;
}
void SCamera::loadTexture ( const cv::Mat& cvmImg_, GLuint* puTexture_ )
{
	glDeleteTextures( 1, puTexture_ );
	glGenTextures ( 1, puTexture_ );
	glBindTexture ( GL_TEXTURE_2D, *puTexture_ );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture  
    // 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
    // border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
    if( 3 == cvmImg_.channels())
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, cvmImg_.cols, cvmImg_.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, cvmImg_.data ); //???????????????????
    else if( 1 == cvmImg_.channels())
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_LUMINANCE, cvmImg_.cols, cvmImg_.rows, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, cvmImg_.data );
    //glTexEnvi ( GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPEAT );
    // 2d texture, 3 colors, width, height, RGB in that order, byte data, and the data.
    //gluBuild2DMipmaps ( GL_TEXTURE_2D, GL_RGB, img.cols, img.rows,  GL_RGB, GL_UNSIGNED_BYTE, img.data );

	//if(bRenderTexture_){
		glBindTexture(GL_TEXTURE_2D, *puTexture_);

		if( 3 == cvmImg_.channels())
			glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, cvmImg_.cols,cvmImg_.rows, GL_RGB, GL_UNSIGNED_BYTE, cvmImg_.data);
		else if( 1 == cvmImg_.channels())
			glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, cvmImg_.cols,cvmImg_.rows, GL_LUMINANCE, GL_UNSIGNED_BYTE, cvmImg_.data);
	//}
    return;
}

void SCamera::renderCameraInLocal(const GpuMat& gpu_img_, btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, float fPhysicalFocalLength_ /*= .02f*/, bool bRenderTexture_/*=true*/) 
{
	GLuint uTesture_;
	if (bRenderTexture_){
		uTesture_ = pGL_->gpuMapRgb2PixelBufferObj(gpu_img_);
	}

	GLboolean bLightIsOn;
	glGetBooleanv(GL_LIGHTING, &bLightIsOn);
	if (bLightIsOn){
		glDisable(GL_LIGHTING);
	}

	const double f = (_fFx + _fFy) / 2.;

	// Draw principle point
	double dT = -_v;
	dT /= f;
	dT *= fPhysicalFocalLength_;
	double dB = -_v + _sHeight;
	dB /= f;
	dB *= fPhysicalFocalLength_;
	double dL = -_u;
	dL /= f;
	dL *= fPhysicalFocalLength_;
	double dR = -_u + _sWidth;
	dR /= f;
	dR *= fPhysicalFocalLength_;

	//draw frame
	if (bRenderTexture_)
	{
		glEnable(GL_TEXTURE_2D);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glBindTexture(GL_TEXTURE_2D, uTesture_);

		glColor4f(1.f, 1.f, 1.f, .6f); 
		glBegin(GL_QUADS);
		glTexCoord2f(0.f, 0.f);		glVertex3d(dL, dT, fPhysicalFocalLength_);
		glTexCoord2f(0.f, 1.f);		glVertex3d(dL, dB, fPhysicalFocalLength_);
		glTexCoord2f(1.f, 1.f);		glVertex3d(dR, dB, fPhysicalFocalLength_);
		glTexCoord2f(1.f, 0.f);		glVertex3d(dR, dT, fPhysicalFocalLength_);
		glEnd();
		glDisable(GL_TEXTURE_2D);
	}
	if (color_)
		glColor3fv(color_);
	else
		glColor3d(0., 0., 0.);

	glLineWidth(2.);
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0);
	glVertex3d(dL, dT, fPhysicalFocalLength_);
	glEnd();
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0);
	glVertex3d(dR, dT, fPhysicalFocalLength_);
	glEnd();
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0);
	glVertex3d(dR, dB, fPhysicalFocalLength_);
	glEnd();
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0);
	glVertex3d(dL, dB, fPhysicalFocalLength_);
	glEnd();

	glBegin(GL_LINE_LOOP);
	glVertex3d(dL, dT, fPhysicalFocalLength_);
	glVertex3d(dR, dT, fPhysicalFocalLength_);
	glVertex3d(dR, dB, fPhysicalFocalLength_);
	glVertex3d(dL, dB, fPhysicalFocalLength_);
	glEnd();

	//glPushAttrib(GL_CURRENT_BIT);

	//draw camera centre
	//glColor3d(1., 1., 0.);
	//glPointSize(5);
	//glBegin(GL_POINTS);
	//glVertex3d(0, 0, fPhysicalFocalLength_);
	//glVertex3d(0, 0, 0);
	//glEnd();

	if (bRenderCoordinate_){
		//draw principle axis
		glColor4f(0.f, 0.f, 1.f, 1.f); //z
		glLineWidth(3);
		glBegin(GL_LINES);
		glVertex3d(0, 0, 0);
		glVertex3d(0, 0, fPhysicalFocalLength_/2);
		glEnd();

		//draw x axis in camera view
		glColor4f(1.f, 0.f, 0.f, 1.f); //x
		glBegin(GL_LINES);
		glVertex3d(0, 0, 0);
		glVertex3d(fPhysicalFocalLength_/2, 0, 0);
		glEnd();

		//draw y axis in camera view
		glColor4f(0.f, 1.f, 0.f, 1.f); //y
		glBegin(GL_LINES);
		glVertex3d(0, 0, 0);
		glVertex3d(0, fPhysicalFocalLength_/2, 0);
		glEnd();

		if(true){
			glPushMatrix();
			//glScalef(2.f, 2.f, 2.f);
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);			// The Type Of Depth Test To Do
			//glShadeModel(GL_SMOOTH);		 // Enables Smooth Shading
			//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
			glEnable(GL_NORMALIZE);
			glEnable(GL_LIGHTING);
			//recursive_render(_scene, _scene->mRootNode, 0.5);
			glPopMatrix();
		}

		//glEnable(GL_BLEND);
		//glDisable(GL_DEPTH_TEST);

	}

	//glPopAttrib();

	if (bLightIsOn){
		glEnable(GL_LIGHTING);
	}

	return;
}

