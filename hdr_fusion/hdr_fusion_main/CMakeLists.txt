cmake_minimum_required (VERSION 2.8)
#cause a warning
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0020 NEW)
endif()

project (HDRFusionMain)

find_package(OpenGL REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Xml REQUIRED)

#Notice that the headers only include the headers that needs to be processed by moc.
QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})

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
		${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} 
        ${GLLIB} ${GLEWLIB} ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS})
endif()

add_executable(HDRFusionMain hdr_fusion_main/Main.cpp  hdr_fusion_main/MultiViewer.cpp hdr_fusion_main/DataLive.cpp hdr_fusion_main/Data4Viewer.cpp hdr_fusion_main/MultiViewer.h hdr_fusion_main/DataLive.h hdr_fusion_main/Data4Viewer.h)
# link with the 3d party libraries
target_link_libraries( HDRFusionMain ${EXTRA_LIBS} )
qt5_use_modules(HDRFusionMain OpenGL Gui Widgets Xml)
