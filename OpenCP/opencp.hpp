#pragma once

#include <opencv2/opencv.hpp>
using namespace cv;

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)


#ifdef _DEBUG
#pragma comment(lib, "opencv_viz"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_videostab"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_video"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ts"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_superres"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_stitching"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ocl"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_objdetect"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_nonfree"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_ml"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_legacy"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_flann"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_features2d"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_calib3d"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_viz"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_videostab"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_video"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ts"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_superres"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_stitching"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ocl"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_objdetect"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_nonfree"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_ml"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_legacy"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
#pragma comment(lib, "opencv_gpu"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_flann"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_features2d"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_calib3d"CV_VERSION_NUMBER".lib")
#endif

//Draw fuction

//merge two images into one image with some options.
void patchBlendImage(Mat& src1, Mat& src2, Mat& dest, Scalar linecolor=CV_RGB(0,0,0), int linewidth = 2, int direction = 0);

// utility functions
void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest);
void alphaBlend(const Mat& src1, const Mat& src2, double alpha, Mat& dest);
void guiAlphaBlend(const Mat& src1, const Mat& src2);
void showMatInfo(InputArray src_, string name);

enum
{
	TIME_AUTO=0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};
class CalcTime
{
	int64 pre;
	string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode();
	vector<string> lap_mes;
public:
	
	void start();
	void setMode(int mode);
	void setMessage(string src);
	void restart();
	double getTime();
	void show();
	void show(string message);
	void lap(string message);
	void init(string message, int mode, bool isShow);

	CalcTime(string message, int mode=TIME_AUTO, bool isShow=true);
	CalcTime();

	~CalcTime();
};

//image processing
//convert a BGR color image into a continued one channel data: ex BGRBGRBGR... -> BBBB...(image size), GGGG....(image size), RRRR....(image size).
void cvtColorBGR2PLANE(const Mat& src, Mat& dest);

//convert a BGR color image into a skipped one channel data: ex BGRBGRBGR... -> BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),...
void splitBGRLineInterleave( const Mat& src, Mat& dest);

//slic
void SLIC(const Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);
void drawSLIC(const Mat& src, Mat& segment, Mat& dest, bool isLine=true, Scalar line_color=Scalar(0,0,255));
void SLICBase(Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);

//bilateral filters

enum
{
	BILATERAL_DEFAULT = 0,
	BILATERAL_CIRCLE = 0,
	BILATERAL_RECT ,
	BILATERAL_SEPARABLE,
	BILATERAL_ORDER2,//underconstruction
	BILATERAL_ORDER2_SEPARABLE//underconstruction
};

void bilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_DEFAULT, int borderType=cv::BORDER_REPLICATE);

//just for comparison
void bilateralFilterSlowest(const Mat& src, Mat& dest, int d, double sigma_color, double sigma_space);

