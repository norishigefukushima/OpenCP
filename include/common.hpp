#pragma once

#include <opencv2/opencv.hpp>

#include <opencv2/ximgproc.hpp>
#include <opencv2/xphoto.hpp>

#ifdef CP_API
#define CP_EXPORT __declspec(dllexport)
#else 
#define CP_EXPORT 
#endif


#define CV_LIB_PREFIX "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib"
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib"
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX

//#define USE_OPENCV_CUDA 1 //use
#define USE_OPENCV_CUDA 0 //not use
//#define OPENCV_LIB_WORLD 
#define OPENCV_LIB_MINIMUM

#ifdef OPENCV_LIB_WORLD
#pragma CV_LIBRARY(world)
#else

//#pragma comment(CV_LIBRARY(alphamat))
//#pragma comment(CV_LIBRARY(aruco))
//#pragma comment(CV_LIBRARY(bgsegm))
//#pragma comment(CV_LIBRARY(bioinspired))
#pragma comment(lib, CV_LIBRARY(calib3d))
#pragma comment(lib, CV_LIBRARY(ccalib))
#pragma comment(lib, CV_LIBRARY(core))
#pragma comment(lib, CV_LIBRARY(cvv))

#if USE_OPENCV_CUDA
#pragma comment(lib, CV_LIBRARY(cudaarithm))
#pragma comment(lib, CV_LIBRARY(cudabgsegm))
#pragma comment(lib, CV_LIBRARY(cudacodec))
#pragma comment(lib, CV_LIBRARY(cudafeatures2d))
#pragma comment(lib, CV_LIBRARY(cudafilters))
#pragma comment(lib, CV_LIBRARY(cudaimgproc))
#pragma comment(lib, CV_LIBRARY(cudalegacy))
#pragma comment(lib, CV_LIBRARY(cudaobjdetect))
#pragma comment(lib, CV_LIBRARY(cudaoptflow))
#pragma comment(lib, CV_LIBRARY(cudastereo))
#pragma comment(lib, CV_LIBRARY(cudawarping))
#pragma comment(lib, CV_LIBRARY(cudev))
#endif

//#pragma comment(lib, CV_LIBRARY(datasets))
//#pragma comment(lib, CV_LIBRARY(dnn))
//#pragma comment(lib, CV_LIBRARY(dnn_objdetect))
//#pragma comment(lib, CV_LIBRARY(dpm))
//#pragma comment(lib, CV_LIBRARY(face))
#pragma comment(lib, CV_LIBRARY(features2d))
#pragma comment(lib, CV_LIBRARY(flann))
//#pragma comment(lib, CV_LIBRARY(fuzzy))
//#pragma comment(lib, CV_LIBRARY(gapi))
//#pragma comment(lib, CV_LIBRARY(hal))
//#pragma comment(lib, CV_LIBRARY(hfs))
#pragma comment(lib, CV_LIBRARY(highgui))
#pragma comment(lib, CV_LIBRARY(imgcodecs))
#pragma comment(lib, CV_LIBRARY(imgproc))
//#pragma comment(lib, CV_LIBRARY(img_hash))
//#pragma comment(lib, CV_LIBRARY(img_intensity_transform))
//#pragma comment(lib, CV_LIBRARY(latentsvm))?
//#pragma comment(lib, CV_LIBRARY(line_descriptor))
//#pragma comment(lib, CV_LIBRARY(mcc))
//#pragma comment(lib, CV_LIBRARY(ml))
//#pragma comment(lib, CV_LIBRARY(objdetect))
#pragma comment(lib, CV_LIBRARY(optflow))
//#pragma comment(lib, CV_LIBRARY(phase_unwrapping))
#pragma comment(lib, CV_LIBRARY(photo))
//#pragma comment(lib, CV_LIBRARY(plot))
//#pragma comment(lib, CV_LIBRARY(quality))
//#pragma comment(lib, CV_LIBRARY(rapid))
//#pragma comment(lib, CV_LIBRARY(reg))
//#pragma comment(lib, CV_LIBRARY(rgbd))
//#pragma comment(lib, CV_LIBRARY(saliency))
//#pragma comment(lib, CV_LIBRARY(shape))
//#pragma comment(lib, CV_LIBRARY(signal))
//#pragma comment(lib, CV_LIBRARY(stereo))
//#pragma comment(lib, CV_LIBRARY(stitching))
//#pragma comment(lib, CV_LIBRARY(structured_light))
//#pragma comment(lib, CV_LIBRARY(superres))
//#pragma comment(lib, CV_LIBRARY(surface_matching))
//#pragma comment(lib, CV_LIBRARY(text))
//#pragma comment(lib, CV_LIBRARY(tracking))
//#pragma comment(lib, CV_LIBRARY(ts))
#pragma comment(lib, CV_LIBRARY(video))
#pragma comment(lib, CV_LIBRARY(videoio))
//#pragma comment(lib, CV_LIBRARY(videostab))
//#pragma comment(lib, CV_LIBRARY(viz))
//#pragma comment(lib, CV_LIBRARY(xfeatures2d))
#pragma comment(lib, CV_LIBRARY(ximgproc))
//#pragma comment(lib, CV_LIBRARY(xobjdetect))
#pragma comment(lib, CV_LIBRARY(xphoto))
#endif //OPENCV_LIB_WORLD

namespace cp
{
	// key define
#ifndef VK_ESCAPE
#define VK_ESCAPE 0x1B
#endif // VK_ESCAPE

//color define
#define COLOR_WHITE cv::Scalar(255,255,255)
#define COLOR_GRAY10 cv::Scalar(10,10,10)
#define COLOR_GRAY20 cv::Scalar(20,20,20)
#define COLOR_GRAY30 cv::Scalar(10,30,30)
#define COLOR_GRAY40 cv::Scalar(40,40,40)
#define COLOR_GRAY50 cv::Scalar(50,50,50)
#define COLOR_GRAY60 cv::Scalar(60,60,60)
#define COLOR_GRAY70 cv::Scalar(70,70,70)
#define COLOR_GRAY80 cv::Scalar(80,80,80)
#define COLOR_GRAY90 cv::Scalar(90,90,90)
#define COLOR_GRAY100 cv::Scalar(100,100,100)
#define COLOR_GRAY110 cv::Scalar(101,110,110)
#define COLOR_GRAY120 cv::Scalar(120,120,120)
#define COLOR_GRAY130 cv::Scalar(130,130,140)
#define COLOR_GRAY140 cv::Scalar(140,140,140)
#define COLOR_GRAY150 cv::Scalar(150,150,150)
#define COLOR_GRAY160 cv::Scalar(160,160,160)
#define COLOR_GRAY170 cv::Scalar(170,170,170)
#define COLOR_GRAY180 cv::Scalar(180,180,180)
#define COLOR_GRAY190 cv::Scalar(190,190,190)
#define COLOR_GRAY200 cv::Scalar(200,200,200)
#define COLOR_GRAY210 cv::Scalar(210,210,210)
#define COLOR_GRAY220 cv::Scalar(220,220,220)
#define COLOR_GRAY230 cv::Scalar(230,230,230)
#define COLOR_GRAY240 cv::Scalar(240,240,240)
#define COLOR_GRAY250 cv::Scalar(250,250,250)
#define COLOR_BLACK cv::Scalar(0,0,0)

#define COLOR_RED cv::Scalar(0,0,255)
#define COLOR_GREEN cv::Scalar(0,255,0)
#define COLOR_BLUE cv::Scalar(255,0,0)
#define COLOR_ORANGE cv::Scalar(0,100,255)
#define COLOR_YELLOW cv::Scalar(0,255,255)
#define COLOR_MAGENDA cv::Scalar(255,0,255)
#define COLOR_CYAN cv::Scalar(255,255,0)

	enum
	{
		FILTER_DEFAULT = 0,
		FILTER_CIRCLE,
		FILTER_RECTANGLE,
		FILTER_SEPARABLE,
		FILTER_SLOWEST,// for just comparison.
	};

	enum SeparableMethod
	{
		DUAL_KERNEL_HV = 0,
		DUAL_KERNEL_VH,
		DUAL_KERNEL_HVVH,
		DUAL_KERNEL_CROSS,
		DUAL_KERNEL_CROSSCROSS,
	};
}