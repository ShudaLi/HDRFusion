cmake_minimum_required (VERSION 2.8)
#cause a warning
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0020 NEW)
endif()

project (MultiExposureCapturer)

find_package(OpenGL REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Xml REQUIRED)

#Notice that the headers only include the headers that needs to be processed by moc.
QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})

#http://qt-project.org/forums/viewthread/30006/
#http://www.cppblog.com/mythma/archive/2013/08/12/cmake_qt5gui_find_extra_libs.html
#http://qt-project.org/quarterly/view/using_cmake_to_build_qt_projects

	include_directories ( $ENV{EIGEN_INC} )
	include_directories ( $ENV{OPENNI2_INC} )
    include_directories ( $ENV{BOOST_INC} ) 
    include_directories ( $ENV{GLEW_INC} )
	include_directories ( $ENV{CUDA_INC_PATH} )
	include_directories ( $ENV{QGLVIEWER_INC} )
	include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )

	link_directories ( $ENV{OPENNI2_LIB} )
	link_directories ( $ENV{BOOST_LIB} )
	link_directories ( $ENV{GLEW_LIB} )
	link_directories ( $ENV{QGLVIEWER_LIB} )
	link_directories ( ${PROJECT_SOURCE_DIR}/lib )

if( WIN32 )
    set(RGBDLIB optimized Rgbd debug Rgbdd)
	set(QGLVIEWER  optimized QGLViewer2  debug QGLViewerd2 )
	set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} OpenNI2
		${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_DATETIME_LIBRARY} 
		GLU32  glew32 GLU32 opengl32 ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS})
elseif(UNIX)
	set(RGBDLIB Rgbd)
	set(GLEWLIB ${GLEW_LIBRARY})
	set(GLLIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
	set(OPENNI2LIB ${OPENNI2_LIBRARY})
	set(QGLVIEWER  QGLViewer )
	set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} OpenNI2
		${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SERIALIZATION_LIBRARY} 
        ${FREEGLUTLIB} ${GLLIB} ${GLEWLIB} ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS})
endif()

add_executable(MultiExposureCapturer multi_exposure_capturer/Main.cpp  multi_exposure_capturer/MultiViewer.cpp multi_exposure_capturer/DataLive.cpp multi_exposure_capturer/Data4Viewer.cpp multi_exposure_capturer/MultiViewer.h multi_exposure_capturer/DataLive.h multi_exposure_capturer/Data4Viewer.h)
# link with the 3d party libraries
target_link_libraries( MultiExposureCapturer ${EXTRA_LIBS} )
qt5_use_modules(MultiExposureCapturer OpenGL Gui Widgets Xml)
