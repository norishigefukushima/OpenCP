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
void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest);
void alphaBlend(const Mat& src1, const Mat& src2, double alpha, Mat& dest);
void guiAlphaBlend(const Mat& src1, const Mat& src2);

//sse utils
void memcpy_float_sse(float* dest, float* src, const int size);

// utility functions
void showMatInfo(InputArray src_, string name="Mat");

class ConsoleImage
{
private:
	int count;
	string windowName;
	std::vector<std::string> strings;
	bool isLineNumber;
public:
	void setIsLineNumber(bool isLine = true);
	bool getIsLineNumber();
	cv::Mat show;

	void init(Size size, string wname);
	ConsoleImage();
	ConsoleImage(cv::Size size, string wname = "console");
	~ConsoleImage();

	void printData();
	void clear();

	void operator()(string src);
	void operator()(const char *format, ...);
	void operator()(cv::Scalar color, const char *format, ...);

	void flush(bool isClear=true);
};



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

class Stat 
{
public:
	Vector<double> data;
	int num_data;
	Stat();
	~Stat();
	double getMin();
	double getMax();
	double getMean();
	double getStd();
	double getMedian();

	void push_back(double val);

	void clear();
	void show();
};

void addNoise(Mat&src, Mat& dest, double sigma, double solt_papper_rate=0.0);

//image processing

//bit convert
void cvt32f8u(const Mat& src, Mat& dest);
void cvt8u32f(const Mat& src, Mat& dest, const float amp);
void cvt8u32f(const Mat& src, Mat& dest);

//convert a BGR color image into a continued one channel data: ex BGRBGRBGR... -> BBBB...(image size), GGGG....(image size), RRRR....(image size).
void cvtColorBGR2PLANE(const Mat& src, Mat& dest);
void cvtColorPLANE2BGR(const Mat& src, Mat& dest);

void cvtColorBGRA2BGR(const Mat& src, Mat& dest);
void cvtColorBGR2BGRA(const Mat& src, Mat& dest, const uchar alpha=255);


//convert a BGR color image into a skipped one channel data: ex BGRBGRBGR... -> BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),...
void splitBGRLineInterleave( const Mat& src, Mat& dest);

//split by number of grid
void mergeFromGrid(Vector<Mat>& src, Size beforeSize, Mat& dest, Size grid, int borderRadius);
void splitToGrid(const Mat& src, Vector<Mat>& dest, Size grid, int borderRadius);

//slic
void SLIC(const Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);
void drawSLIC(const Mat& src, Mat& segment, Mat& dest, bool isLine=true, Scalar line_color=Scalar(0,0,255));
void SLICBase(Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);

//bilateral filters


enum
{
	BILATERAL_ORDER2,//underconstruction
	BILATERAL_ORDER2_SEPARABLE//underconstruction
};

enum
{
	FILTER_DEFAULT = 0,
	FILTER_CIRCLE,
	FILTER_RECTANGLE,
	FILTER_SEPARABLE,
	FILTER_SLOWEST,// for just comparison.
};

void bilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=FILTER_DEFAULT, int borderType=cv::BORDER_REPLICATE);
void jointBilateralFilter(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=FILTER_DEFAULT, int borderType=cv::BORDER_REPLICATE);

void weightedBilateralFilter(const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method, int borderType=cv::BORDER_REPLICATE);
void weightedJointBilateralFilter(const Mat& src, Mat& weightMap,const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method, int borderType);

void guidedFilter(const Mat& src,  Mat& dest, const int radius,const float eps);
void guidedFilter(const Mat& src, const Mat& guidance, Mat& dest, const int radius,const float eps);

void guidedFilterMultiCore(const Mat& src, Mat& dest, int r,float eps, int numcore=0);
void guidedFilterMultiCore(const Mat& src, const Mat& guide, Mat& dest, int r,float eps,int numcore=0);


typedef enum
{
	DTF_RF=0,//Recursive Filtering
	DTF_NC=1,//Normalized Convolution
	DTF_IC=1,//Interpolated Convolution

}DTF_METHOD;

void domainTransformFilter(cv::Mat& img, cv::Mat& out, double sigma_s, double sigma_r, int maxiter, int method=DTF_RF);

void recursiveBilateralFilter(Mat& src, Mat& dest, float sigma_range, float sigma_spatial, int method=0);
class RecursiveBilateralFilter
{
private:
	Mat bgra;

	Mat texture;//texture is joint signal
	Mat destf; 
	Mat temp; 
	Mat tempw;

	Size size;
public:
	void setColorLUTGaussian(float* lut, float sigma);
	void setColorLUTLaplacian(float* lut, float sigma);
	void init(Size size_);
	RecursiveBilateralFilter(Size size);
	RecursiveBilateralFilter();
	~RecursiveBilateralFilter();
	void operator()(const Mat& src, Mat& dest, float sigma_range, float sigma_spatial);
	void operator()(const Mat& src, const Mat& guide, Mat& dest, float sigma_range, float sigma_spatial);
};


void binalyWeightedRangeFilter(const Mat& src, Mat& dst, Size kernelSize, float threshold, int method=FILTER_DEFAULT, int borderType=cv::BORDER_REPLICATE);
void jointBinalyWeightedRangeFilter(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, float threshold, int method=FILTER_DEFAULT, int borderType=cv::BORDER_REPLICATE);

void nonLocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma=-1.0, int method=FILTER_DEFAULT);

void iterativeBackProjectionDeblurGaussian(const Mat& src, Mat& dest, const Size ksize, const double sigma, const double lambda, const int iteration);
void iterativeBackProjectionDeblurBilateral(const Mat& src, Mat& dest, const Size ksize, const double sigma_color, const double sigma_space, const double lambda, const int iteration);

enum
{
	PROCESS_LAB=0,
	PROCESS_BGR
};

void detailEnhancementBilateral(Mat& src, Mat& dest, int d, float sigma_color, float sigma_space, float boost, int color=PROCESS_LAB);