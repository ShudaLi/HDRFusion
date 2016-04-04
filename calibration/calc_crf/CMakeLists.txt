PROJECT( Hdr )
cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE( OpenCV REQUIRED PATHS $ENV{OPENCV_NEW} NO_DEFAULT_PATH)
include_directories($ENV{OPENCV_NEW}/include )
ADD_EXECUTABLE( Hdr Hdr.cpp )
TARGET_LINK_LIBRARIES( Hdr ${OpenCV_LIBS} )




