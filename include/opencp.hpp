#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <opencv2/ximgproc.hpp>
#ifdef CP_API
#define CP_EXPORT __declspec(dllexport)
#else 
#define CP_EXPORT 
#endif

#define CV_LIB_PREFIX comment(lib, "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib")
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib")
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX

#pragma CV_LIBRARY(aruco)
#pragma CV_LIBRARY(bgsegm)
#pragma CV_LIBRARY(bioinspired)
#pragma CV_LIBRARY(calib3d)
#pragma CV_LIBRARY(ccalib)
#pragma CV_LIBRARY(core)

#pragma CV_LIBRARY(cudaarithm)
#pragma CV_LIBRARY(cudabgsegm)
#pragma CV_LIBRARY(cudacodec)
#pragma CV_LIBRARY(cudafeatures2d)
#pragma CV_LIBRARY(cudafilters)
#pragma CV_LIBRARY(cudaimgproc)
#pragma CV_LIBRARY(cudalegacy)
#pragma CV_LIBRARY(cudaobjdetect)
#pragma CV_LIBRARY(cudaoptflow)
#pragma CV_LIBRARY(cudastereo)
#pragma CV_LIBRARY(cudawarping)
#pragma CV_LIBRARY(cudev)


#pragma CV_LIBRARY(datasets)
#pragma CV_LIBRARY(dnn)
#pragma CV_LIBRARY(dpm)
#pragma CV_LIBRARY(face)
#pragma CV_LIBRARY(features2d)
#pragma CV_LIBRARY(flann)
#pragma CV_LIBRARY(hal)
#pragma CV_LIBRARY(highgui)
#pragma CV_LIBRARY(imgcodecs)
#pragma CV_LIBRARY(imgproc)

//#pragma CV_LIBRARY(latentsvm)
#pragma CV_LIBRARY(line_descriptor)
#pragma CV_LIBRARY(ml)
#pragma CV_LIBRARY(objdetect)
#pragma CV_LIBRARY(optflow)
#pragma CV_LIBRARY(photo)
#pragma CV_LIBRARY(reg)
#pragma CV_LIBRARY(rgbd)
#pragma CV_LIBRARY(saliency)
#pragma CV_LIBRARY(shape)
#pragma CV_LIBRARY(stitching)
#pragma CV_LIBRARY(structured_light)
#pragma CV_LIBRARY(superres)
#pragma CV_LIBRARY(surface_matching)
#pragma CV_LIBRARY(text)
#pragma CV_LIBRARY(tracking)
#pragma CV_LIBRARY(ts)
#pragma CV_LIBRARY(video)
#pragma CV_LIBRARY(videoio)
#pragma CV_LIBRARY(videostab)
#pragma CV_LIBRARY(viz)
#pragma CV_LIBRARY(xfeatures2d)
#pragma CV_LIBRARY(ximgproc)
#pragma CV_LIBRARY(xobjdetect)
#pragma CV_LIBRARY(xphoto)

//FFTW
#pragma comment(lib, "libfftw3-3.lib")
#pragma comment(lib, "libfftw3f-3.lib")
#pragma comment(lib, "libfftw3l-3.lib")

namespace cp
{
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//Utility Functions, Drawing Fuction
	//============================================================================================================================================================

#ifndef VK_ESCAPE
#define VK_ESCAPE 0x1B
#endif // VK_ESCAPE

	
	CP_EXPORT void fitPlaneCrossProduct(std::vector<cv::Point3f>& src, cv::Point3f& dest);
	CP_EXPORT void fitPlanePCA(cv::InputArray src, cv::Point3f& dest);
	CP_EXPORT void fitPlaneRANSAC(std::vector<cv::Point3f>& src, cv::Point3f& dest, int numofsample, float threshold, int refineIter = 0);

	CP_EXPORT void drawHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, bool isGrid = true);
	CP_EXPORT void drawAccumulateHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, bool isGrid = true);
	CP_EXPORT void drawHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, bool isGrid = true);
	CP_EXPORT void drawAccumulateHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, bool isGrid = true);

	CP_EXPORT void addBoxMask(cv::Mat& mask, int boundx, int boundy);
	CP_EXPORT cv::Mat createBoxMask(cv::Size size, int boundx, int boundy);
	CP_EXPORT void setBoxMask(cv::Mat& mask, int boundx, int boundy);
	CP_EXPORT void diffshow(std::string wname, cv::InputArray src, cv::InputArray ref, const double scale = 1.0);
	//merge two images into one image with some options.
	CP_EXPORT void patchBlendImage(cv::Mat& src1, cv::Mat& src2, cv::Mat& dest, cv::Scalar linecolor = CV_RGB(0, 0, 0), int linewidth = 2, int direction = 0);
	CP_EXPORT void alphaBlend(const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& alpha, cv::Mat& dest);
	CP_EXPORT void alphaBlend(cv::InputArray src1, cv::InputArray src2, const double alpha, cv::OutputArray dest);
	CP_EXPORT void guiAlphaBlend(cv::InputArray src1, cv::InputArray src2, bool isShowImageStats = false);
	CP_EXPORT void guiZoom(cv::InputArray src, cv::OutputArray dest = cv::noArray());
	CP_EXPORT void guiContrast(cv::InputArray src);

	CP_EXPORT void guiFilterSpeckle(cv::InputArray src);
	CP_EXPORT void guiVideoShow(std::string wname);

	//sse utils
	CP_EXPORT void memcpy_float_sse(float* dest, float* src, const int size);
	CP_EXPORT void setDepthMaxValue(cv::InputOutputArray src);

	CP_EXPORT void guiShift(cv::InputArray centerimg, cv::InputArray leftimg, cv::InputArray rightimg, int max_move, std::string window_name = "Shift");
	CP_EXPORT void guiShift(cv::InputArray fiximg, cv::InputArray moveimg, const int max_move = 200, std::string window_name = "Shift");

	CP_EXPORT void warpShiftH(cv::InputArray src, cv::OutputArray dest, const int shiftH);
	CP_EXPORT void warpShift(cv::InputArray src, cv::OutputArray dest, int shiftx, int shifty = 0, int borderType = -1);
	CP_EXPORT void warpShiftSubpix(cv::InputArray  src, cv::OutputArray dest, double shiftx, double shifty = 0, const int inter_method = cv::INTER_LANCZOS4);

	CP_EXPORT void imshowFFT(std::string wname, cv::InputArray src);
	CP_EXPORT void imshowScale(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);
	CP_EXPORT void imshowNormalize(std::string wname, cv::InputArray src);

	CP_EXPORT void showMatInfo(cv::InputArray src, std::string name = "Mat");

	CP_EXPORT void guiCompareDiff(const cv::Mat& before, const cv::Mat& after, const cv::Mat& ref);
	CP_EXPORT void guiAbsDiffCompareNE(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareEQ(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareLE(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareGE(const cv::Mat& src1, const cv::Mat& src2);

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

	CP_EXPORT void drawGrid(cv::InputOutputArray src, cv::Point crossCenter, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawPlus(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawAsterisk(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawTimes(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_typee = 8, int shift = 0);
	CP_EXPORT void triangleinv(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);
	CP_EXPORT void triangle(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);

	class CP_EXPORT ConsoleImage
	{
	private:
		int count;
		std::string windowName;
		std::vector<std::string> strings;
		bool isLineNumber;
	public:
		void setIsLineNumber(bool isLine = true);
		bool getIsLineNumber();
		cv::Mat show;

		void init(cv::Size size, std::string wname);
		ConsoleImage();
		ConsoleImage(cv::Size size, std::string wname = "console");
		~ConsoleImage();

		void printData();
		void clear();

		void operator()(std::string src);
		void operator()(const char *format, ...);
		void operator()(cv::Scalar color, const char *format, ...);

		void flush(bool isClear = true);
	};

	enum
	{
		TIME_AUTO = 0,
		TIME_NSEC,
		TIME_MSEC,
		TIME_SEC,
		TIME_MIN,
		TIME_HOUR,
		TIME_DAY
	};
	class CP_EXPORT CalcTime
	{
		int64 pre;
		std::string mes;

		int timeMode;

		double cTime;
		bool _isShow;

		int autoMode;
		int autoTimeMode();
		std::vector<std::string> lap_mes;
	public:

		void start();
		void setMode(int mode);
		void setMessage(std::string& src);
		void restart();
		double getTime();
		void show();
		void show(std::string message);
		void lap(std::string message);
		void init(std::string message, int mode, bool isShow);

		CalcTime(std::string message, int mode = TIME_AUTO, bool isShow = true);
		CalcTime(char* message, int mode = TIME_AUTO, bool isShow = true);
		CalcTime();

		~CalcTime();
	};

	class CP_EXPORT DestinationTimePrediction
	{
	public:
		int destCount;
		int pCount;
		int64 startTime;

		int64 firstprediction;

		int64 prestamp;
		int64 prestamp_for_prediction;

		void init(int DestinationCount);
		DestinationTimePrediction();
		DestinationTimePrediction(int DestinationCount);
		int autoTimeMode(double cTime);
		void tick2Time(double tick, std::string mes);
		int64 getTime(std::string mes);
		~DestinationTimePrediction();
		void predict();
		double predict(int presentCount, int interval = 500);
	};

	class CP_EXPORT Stat
	{
	public:
		std::vector<double> data;
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

	class CP_EXPORT CSV
	{
		FILE* fp;
		bool isTop;
		long fileSize;
		std::string filename;
	public:
		std::vector<double> argMin;
		std::vector<double> argMax;
		std::vector<std::vector<double>> data;
		std::vector<bool> filter;
		int width;
		void findMinMax(int result_index, bool isUseFilter, double minValue, double maxValue);
		void initFilter();
		void filterClear();
		void makeFilter(int index, double val, double emax = 0.00000001);
		void readHeader();
		void readData();

		void init(std::string name, bool isWrite, bool isClear);
		CSV();
		CSV(std::string name, bool isWrite = true, bool isClear = true);

		~CSV();
		void write(std::string v);
		void write(double v);
		void end();
	};

	enum
	{
		PLOT_ARG_MAX = 1,
		PLOT_ARG_MIN = -1
	};
	class CP_EXPORT Plot2D
	{
		std::vector<std::vector<double>> data;
		cv::Mat graphBase;
		int w;
		int h;
		void createPlot();
		void setMinMaxX(double minv, double maxv, int count);
		void setMinMaxY(double minv, double maxv, int count);
	public:
		cv::Mat show;
		cv::Mat graph;
		cv::Size size;
		double minx;
		double maxx;
		int countx;

		double miny;
		double maxy;
		int county;

		Plot2D(cv::Size graph_size, double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);

		void setMinMax(double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);
		void add(int x, int y, double val);
		void writeGraph(bool isColor, int arg_min_max, double minvalue = 0, double maxvalue = 0, bool isMinMaxSet = false);
		void setLabel(std::string namex, std::string namey);
		//void plot(CSV& result, vector<ExperimentalParameters>& parameters);
	};

	class CP_EXPORT gnuplot
	{
		FILE* fp;
	public:
		gnuplot(std::string gnuplotpath);
		void cmd(std::string name);
		~gnuplot();
	};

	class CP_EXPORT Plot
	{
	protected:
		struct PlotInfo
		{
			std::vector<cv::Point2d> data;
			cv::Scalar color;
			int symbolType;
			int lineType;
			int thickness;

			std::string keyname;
		};
		std::vector<PlotInfo> pinfo;

		std::string xlabel;
		std::string ylabel;

		int data_max;

		cv::Scalar background_color;

		cv::Size plotsize;
		cv::Point origin;

		double xmin;
		double xmax;
		double ymin;
		double ymax;
		double xmax_no_margin;
		double xmin_no_margin;
		double ymax_no_margin;
		double ymin_no_margin;

		void init();
		void point2val(cv::Point pt, double* valx, double* valy);

		bool isZeroCross;
		bool isXYMAXMIN;
		bool isXYCenter;

		bool isPosition;
		cv::Scalar getPseudoColor(uchar val);
		cv::Mat plotImage;
		cv::Mat keyImage;
	public:
		enum
		{
			SYMBOL_NOPOINT = 0,
			SYMBOL_PLUS,
			SYMBOL_TIMES,
			SYMBOL_ASTERRISK,
			SYMBOL_CIRCLE,
			SYMBOL_RECTANGLE,
			SYMBOL_CIRCLE_FILL,
			SYMBOL_RECTANGLE_FILL,
			SYMBOL_TRIANGLE,
			SYMBOL_TRIANGLE_FILL,
			SYMBOL_TRIANGLE_INV,
			SYMBOL_TRIANGLE_INV_FILL,
		};

		enum
		{
			LINE_NONE,
			LINE_LINEAR,
			LINE_H2V,
			LINE_V2H
		};
		cv::Mat render;
		cv::Mat graphImage;

		Plot(cv::Size window_size = cv::Size(1024, 768));
		~Plot();

		void setXYOriginZERO();
		void setXOriginZERO();
		void setYOriginZERO();

		void recomputeXYMAXMIN(bool isCenter = false, double marginrate = 0.9);
		void setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_);
		void setPlotImageSize(cv::Size s);
		void setXYMinMax(double xmin_, double xmax_, double ymin_, double ymax_);
		void setXMinMax(double xmin_, double xmax_);
		void setYMinMax(double ymin_, double ymax_);
		void setBackGoundColor(cv::Scalar cl);

		void makeBB(bool isFont);

		void setPlot(int plotnum, cv::Scalar color = COLOR_RED, int symboltype = SYMBOL_PLUS, int linetype = LINE_LINEAR, int thickness = 1);
		void setPlotThickness(int plotnum, int thickness_);
		void setPlotColor(int plotnum, cv::Scalar color);
		void setPlotSymbol(int plotnum, int symboltype);
		void setPlotLineType(int plotnum, int linetype);
		void setPlotKeyName(int plotnum, std::string name);

		void setLinetypeALL(int linetype);

		void plotPoint(cv::Point2d = cv::Point2d(0.0, 0.0), cv::Scalar color = COLOR_BLACK, int thickness_ = 1, int linetype = LINE_LINEAR);
		void plotGrid(int level);
		void plotData(int gridlevel = 0, int isKey = 0);

		void plot(std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");

		void makeKey(int num);

		void save(std::string name);

		void push_back(std::vector<cv::Point> point, int plotIndex = 0);
		void push_back(std::vector<cv::Point2d> point, int plotIndex = 0);
		void push_back(double x, double y, int plotIndex = 0);

		void erase(int sampleIndex, int plotIndex = 0);
		void insert(cv::Point2d v, int sampleIndex, int plotIndex = 0);
		void insert(cv::Point v, int sampleIndex, int plotIndex = 0);
		void insert(double x, double y, int sampleIndex, int plotIndex = 0);

		void clear(int datanum = -1);

		void swapPlot(int plotIndex1, int plotIndex2);
	};

	void plotGraph(cv::Mat& render, std::vector<cv::Point2d>& data, double xmin, double xmax, double ymin, double ymax,
		cv::Scalar color = COLOR_RED, int lt = Plot::SYMBOL_PLUS, int isLine = Plot::LINE_LINEAR, int thickness = 1, int ps = 4);

	//simd functions
	void print_m128(__m128d src);
	void print_m128(__m128 src);
	void print_m128i_char(__m128i src);
	void print_m128i_uchar(__m128i src);
	void print_m128i_short(__m128i src);
	void print_m128i_ushort(__m128i src);
	void print_m128i_int(__m128i src);
	void print_m128i_uint(__m128i src);

	//============================================================================================================================================================
	//Image processing Functions imageprocessing
	//============================================================================================================================================================

	CP_EXPORT cv::Mat imreadPPMX(std::string filename);
	CP_EXPORT void readYUVGray(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void readYUV2BGR(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void writeYUVBGR(std::string fname, cv::InputArray src);
	CP_EXPORT void writeYUVGray(std::string fname, cv::InputArray src);
	CP_EXPORT void writeYUV(cv::Mat& InputArray, std::string name, int mode = 1);

	class CP_EXPORT YUVReader
	{
		FILE* fp;
		int framemax;
		char* buff;
		bool isloop;

		int yuvSize;

	public:
		int width;
		int height;
		int imageSize;
		int imageCSize;
		int frameCount;

		void init(std::string name, cv::Size size, int frame_max);
		YUVReader(std::string name, cv::Size size, int frame_max);
		YUVReader();
		~YUVReader();

		void readNext(cv::Mat& dest);
		bool read(cv::Mat& dest, int frame);
	};

	CP_EXPORT double YPSNR(cv::InputArray src1, cv::InputArray src2);
	CP_EXPORT double calcBadPixel(const cv::Mat& src, const cv::Mat& ref, int threshold);
	CP_EXPORT double SSIM(cv::Mat& src, cv::Mat& ref, double sigma = 1.5);
	CP_EXPORT double calcTV(cv::Mat& src);
	CP_EXPORT double calcEntropy(cv::InputArray src, cv::InputArray mask = cv::noArray());

	CP_EXPORT void addNoise(cv::InputArray src, cv::OutputArray dest, double sigma, double solt_papper_ratio = 0.0);

	enum
	{
		IQM_PSNR = 0,
		IQM_MSE,
		IQM_MSAD,
		IQM_DELTA,
		IQM_SSIM,
		IQM_SSIM_FAST,
		IQM_SSIM_MODIFY,
		IQM_SSIM_FASTMODIFY,
		IQM_CWSSIM,
		IQM_CWSSIM_FAST,
		IQM_MSSSIM,
		IQM_MSSSIM_FAST
	};
	CP_EXPORT double calcImageQualityMetric(cv::InputArray src, cv::InputArray target, const int metric = IQM_PSNR, const int boundingBox = 0);

	CP_EXPORT double PSNR64F(cv::InputArray src1, cv::InputArray src2);
	CP_EXPORT double MSE(cv::InputArray src1, cv::InputArray src2);
	CP_EXPORT double MSE(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask);

	enum
	{
		PIXEL_DIFF_DIRECTION_H = 0,
		PIXEL_DIFF_DIRECTION_V,
		PIXEL_DIFF_DIRECTION_HV,
		PIXEL_DIFF_DIRECTION_HH,
		PIXEL_DIFF_DIRECTION_VV,
		PIXEL_DIFF_DIRECTION_HHVV,
		PIXEL_DIFF_DIRECTION_HMIN,
		PIXEL_DIFF_DIRECTION_HMAX,
		PIXEL_DIFF_DIRECTION_VMIN,
		PIXEL_DIFF_DIRECTION_VMAX,
		PIXEL_DIFF_DIRECTION_HVMIN,
		PIXEL_DIFF_DIRECTION_HVMAX
	};
	CP_EXPORT void pixelDiffABS(cv::Mat& src, cv::Mat& dest, int direction = PIXEL_DIFF_DIRECTION_H);
	CP_EXPORT void pixelDiffThresh(cv::Mat& src, cv::Mat& dest, double thresh, int direction = PIXEL_DIFF_DIRECTION_H);

	CP_EXPORT void eraseBoundary(const cv::Mat& src, cv::Mat& dest, int step, int border = cv::BORDER_REPLICATE);


	//arithmetics
	CP_EXPORT void pow_fmath(const float a, const cv::Mat&  src, cv::Mat& dest);
	CP_EXPORT void pow_fmath(const cv::Mat& src, const float a, cv::Mat& dest);
	CP_EXPORT void pow_fmath(const cv::Mat& src1, const cv::Mat&  src2, cv::Mat& dest);
	CP_EXPORT void compareRange(cv::InputArray src, cv::OutputArray destMask, const double validMin, const double validMax);

	//bit convert
	CP_EXPORT void cvt32f8u(const cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void cvt8u32f(const cv::Mat& src, cv::Mat& dest, const float amp);
	CP_EXPORT void cvt8u32f(const cv::Mat& src, cv::Mat& dest);

	//convert a BGR color image into a continued one channel data: ex BGRBGRBGR... -> BBBB...(image size), GGGG....(image size), RRRR....(image size).
	//colorconvert 
	CP_EXPORT void cvtColorBGR2PLANE(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cvtColorPLANE2BGR(cv::InputArray src, cv::OutputArray dest);

	CP_EXPORT void cvtColorBGRA2BGR(const cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void cvtColorBGRA32f2BGR8u(const cv::Mat& src, cv::Mat& dest);

	CP_EXPORT void cvtColorBGR2BGRA(const cv::Mat& src, cv::Mat& dest, const uchar alpha = 255);
	CP_EXPORT void cvtColorBGR8u2BGRA32f(const cv::Mat& src, cv::Mat& dest, const float alpha = 255.f);

	CP_EXPORT void cvtColorOPP2BGR(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cvtColorBGR2OPP(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cvtColorMatrix(cv::InputArray src, cv::InputOutputArray dest, cv::InputArray C);

	CP_EXPORT void cvtRAWVector2BGR(std::vector<float>& src, cv::OutputArray dest, cv::Size size);
	CP_EXPORT void cvtBGR2RawVector(cv::InputArray src, std::vector<float>& dest);

	//color correction colorcorrection whilebalance
	CP_EXPORT void findColorMatrixAvgStdDev(cv::InputArray ref_image, cv::InputArray target_image, cv::OutputArray colorMatrix, const double validMin, const double validMax);

	//convert a BGR color image into a skipped one channel data: ex BGRBGRBGR... -> BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),...
	CP_EXPORT void splitBGRLineInterleave(cv::InputArray src, cv::OutputArray dest);

	CP_EXPORT void mergeHorizon(const std::vector<cv::Mat>& src, cv::Mat& dest);
	CP_EXPORT void splitHorizon(const cv::Mat& src, std::vector<cv::Mat>& dest, int num);
	//split by number of grid
	CP_EXPORT void mergeFromGrid(std::vector<cv::Mat>& src, cv::Size beforeSize, cv::Mat& dest, cv::Size grid, int borderRadius);
	CP_EXPORT void splitToGrid(const cv::Mat& src, std::vector<cv::Mat>& dest, cv::Size grid, int borderRadius);

	//slic
	CP_EXPORT void SLICSegment2Vector3D(cv::InputArray segment, cv::InputArray signal, std::vector<std::vector<cv::Point3f>>& segmentPoint);
	CP_EXPORT void SLICSegment2Vector3D(cv::InputArray segment, cv::InputArray signal, std::vector<std::vector<cv::Point3i>>& segmentPoint);
	CP_EXPORT void SLICVector2Segment(std::vector<std::vector<cv::Point>>& segmentPoint, cv::Size outputImageSize, cv::OutputArray segment);
	CP_EXPORT void SLICVector3D2Signal(std::vector<std::vector<cv::Point3f>>& segmentPoint, cv::Size outputImageSize, cv::OutputArray signal);
	CP_EXPORT void SLICSegment2Vector(cv::InputArray segment, std::vector<std::vector<cv::Point>>& segmentPoint);
	CP_EXPORT void SLIC(cv::InputArray src, cv::OutputArray segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);
	CP_EXPORT void drawSLIC(cv::InputArray src, cv::InputArray segment, cv::OutputArray dst, bool isMean = true, bool isLine = true, cv::Scalar line_color = cv::Scalar(0, 0, 255));
	CP_EXPORT void SLICBase(cv::Mat& src, cv::Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);//not optimized code for test


	//============================================================================================================================================================
	//Filtering Functions
	//============================================================================================================================================================

	CP_EXPORT void blurRemoveMinMax(const cv::Mat& src, cv::Mat& dest, const int r);
	//MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE
	CP_EXPORT void maxFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, int shape = cv::MORPH_RECT);
	CP_EXPORT void maxFilter(cv::InputArray src, cv::OutputArray dest, int radius);
	//MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE
	CP_EXPORT void minFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, int shape = cv::MORPH_RECT);
	CP_EXPORT void minFilter(cv::InputArray src, cv::OutputArray dest, int radius);

	enum
	{
		FILTER_DEFAULT = 0,
		FILTER_CIRCLE,
		FILTER_RECTANGLE,
		FILTER_SEPARABLE,
		FILTER_SLOWEST,// for just comparison.
	};

	class CP_EXPORT PostFilterSet
	{
		cv::Mat buff, bufff;
	public:
		PostFilterSet();
		~PostFilterSet();
		void filterDisp8U2Depth32F(cv::Mat& src, cv::Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method = FILTER_DEFAULT);
		void filterDisp8U2Depth16U(cv::Mat& src, cv::Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method = FILTER_DEFAULT);
		void filterDisp8U2Disp32F(cv::Mat& src, cv::Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method = FILTER_DEFAULT);
		void operator()(cv::Mat& src, cv::Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, int brange_th, int brange_method = FILTER_DEFAULT);
	};
	enum
	{
		GAUSSIAN_FILTER_DCT,
		GAUSSIAN_FILTER_FIR,
		GAUSSIAN_FILTER_BOX,
		GAUSSIAN_FILTER_EBOX,
		GAUSSIAN_FILTER_SII,
		GAUSSIAN_FILTER_AM,
		GAUSSIAN_FILTER_AM2,
		GAUSSIAN_FILTER_DERICHE,
		GAUSSIAN_FILTER_VYV,
		GAUSSIAN_FILTER_SR,
	};
	CP_EXPORT void GaussianFilter(cv::InputArray src, cv::OutputArray dest, const double sigma_space, const int filter_method, const int K = 0, const double tol = 1.0e-6);
	CP_EXPORT void GaussianFilterwithMask(const cv::Mat src, cv::Mat& dest, int r, float sigma, int method, cv::Mat& mask);//slowest

	CP_EXPORT void weightedGaussianFilter(cv::Mat& src, cv::Mat& weight, cv::Mat& dest, cv::Size ksize, float sigma, int border_type = cv::BORDER_REPLICATE);

	CP_EXPORT void jointNearestFilter(cv::InputArray src, cv::InputArray guide, cv::Size ksize, cv::OutputArray dest);
	CP_EXPORT void jointNearestFilterBase(cv::InputArray src, cv::InputArray guide, cv::Size ksize, cv::OutputArray dest);

	CP_EXPORT void wiener2(cv::Mat&src, cv::Mat& dest, int szWindowX, int szWindowY);
	CP_EXPORT void coherenceEnhancingShockFilter(cv::InputArray src, cv::OutputArray dest, const int sigma, const int str_sigma, const double blend, const int iter);

	//bilateral filters
	enum
	{
		BILATERAL_ORDER2,//underconstruction
		BILATERAL_ORDER2_SEPARABLE//underconstruction
	};

	CP_EXPORT void bilateralFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void dualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void dualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_guide_color, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void jointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, cv::Size ksize, double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, int d, double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void bilateralWeightMap(cv::InputArray src_, cv::OutputArray dst_, cv::Size kernelSize, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void dualBilateralWeightMap(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void dualBilateralWeightMapXOR(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	enum SeparableMethod
	{
		DUAL_KERNEL_HV = 0,
		DUAL_KERNEL_VH,
		DUAL_KERNEL_HVVH,
		DUAL_KERNEL_CROSS,
		DUAL_KERNEL_CROSSCROSS,
	};
	CP_EXPORT void separableBilateralFilter(const cv::Mat& src, cv::Mat& dst, cv::Size kernelSize, double sigma_color, double sigma_space, double alpha, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableJointBilateralFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, cv::Size kernelSize, double sigma_color, double sigma_space, double alpha, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableNonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma = -1.0, double alpha = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableNonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma = -1.0, double alpha = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableDualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int sp_kernel_type = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableDualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int sp_kernel_type = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableJointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, cv::Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableJointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void bilateralFilterL2(cv::InputArray src, cv::OutputArray dest, int radius, double sigma_color, double sigma_space, int borderType = cv::BORDER_REPLICATE);
	

	CP_EXPORT void weightedBilateralFilter(cv::InputArray src, cv::InputArray weight, cv::OutputArray dst, int D, double sigma_color, double sigma_space, int method, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedBilateralFilter(cv::InputArray src, cv::InputArray weight, cv::OutputArray dst, cv::Size kernelSize, double sigma_color, double sigma_space, int method, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void weightedJointBilateralFilter(cv::InputArray src, cv::InputArray weightMap, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedJointBilateralFilter(cv::InputArray src, cv::InputArray weightMap, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void guidedFilter(const cv::Mat& src, cv::Mat& dest, const int radius, const float eps);
	CP_EXPORT void guidedFilter(const cv::Mat& src, const cv::Mat& guidance, cv::Mat& dest, const int radius, const float eps);

	CP_EXPORT void guidedFilterMultiCore(const cv::Mat& src, cv::Mat& dest, int r, float eps, int numcore = 0);
	CP_EXPORT void guidedFilterMultiCore(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, int r, float eps, int numcore = 0);

	CP_EXPORT void L0Smoothing(cv::Mat &im8uc3, cv::Mat& dest, float lambda = 0.02f, float kappa = 2.f);

	class CP_EXPORT RealtimeO1BilateralFilter
	{
	protected:
	
		std::vector<cv::Mat> bgrid;//for presubsampling

		std::vector<cv::Mat> sub_range;
		std::vector<cv::Mat> normalize_sub_range;

		std::vector<uchar> bin2num;
		std::vector<uchar> idx;
		std::vector<float> a;

		int num_bin;
		int bin_depth;
		void createBin(cv::Size imsize, int num_bin, int channles);
		void disposeBin(int number_of_bin);

		double sigma_color;
		float CV_DECL_ALIGNED(16) color_weight_32F[256 * 3];
		double CV_DECL_ALIGNED(16) color_weight_64F[256 * 3];
		void setColorLUT(double sigma_color, int channlels);

		int normType;
		template <typename T, typename S>
		void splatting(const T* s, S* su, S* sd, const uchar* j, const uchar v, const int imageSize, const int channels);
		template <typename T, typename S>
		void splattingColor(const T* s, S* su, S* sd, const uchar* j, const uchar* v, const int imageSize, const int channels, const int type);

		double sigma_space;
		int radius;
		int filterK;
		int filter_type;
		virtual void blurring(const cv::Mat& src, cv::Mat& dest);

		template <typename T, typename S>
		void bodySaveMemorySize_(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest);
		template <typename T, typename S>
		void body_(const cv::Mat& src, const cv::Mat& joint, cv::Mat& dest);

		void body(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, bool save_memorySize);
	public:
		RealtimeO1BilateralFilter();
		~RealtimeO1BilateralFilter();
		void showBinIndex();//show bin index for debug
		void setBinDepth(int depth = CV_32F);
		enum
		{
			L1SQR,//norm for OpenCV's native Bilateral filter
			L1,
			L2
		};
		
		void setColorNorm(int norm=L1SQR);
		int downsampleSizeSplatting;
		int downsampleSizeBlurring;
		int downsampleMethod;
		int upsampleMethod;
		bool isSaveMemory;

		enum
		{
			FIR_SEPARABLE,
			IIR_AM,
			IIR_SR,
			IIR_Deriche,
			IIR_YVY,
		};

		void gaussIIR(cv::InputArray src, cv::OutputArray dest, float sigma_color, float sigma_space, int num_bin, int method, int K);
		void gaussIIR(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, float sigma_color, float sigma_space, int num_bin, int method, int K);
		void gaussFIR(cv::InputArray src, cv::OutputArray dest, int r, float sigma_color, float sigma_space, int num_bin);
		void gaussFIR(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, int r, float sigma_color, float sigma_space, int num_bin);
	};

	enum
	{
		NO_WEIGHT = 0,
		GAUSSIAN,
		BILATERAL
	};
	CP_EXPORT void weightedMedianFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedMedianFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedMedianFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor1, double sigmaColor2, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedModeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedModeFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedModeFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor1, double sigmaColor2, double sigmaSpace, int metric, int method);

	typedef enum
	{
		DTF_L1 = 1,
		DTF_L2 = 2
	}DTF_NORM;

	typedef enum
	{
		DTF_RF = 0,//Recursive Filtering
		DTF_NC = 1,//Normalized Convolution
		DTF_IC = 2,//Interpolated Convolution

	}DTF_METHOD;

	typedef enum
	{
		DTF_BGRA_SSE = 0,
		DTF_BGRA_SSE_PARALLEL,
		DTF_SLOWEST
	}DTF_IMPLEMENTATION;


	CP_EXPORT void domainTransformFilter(cv::InputArray srcImage, cv::OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm = DTF_L1, const int convolutionType = DTF_RF, const int implementation = DTF_SLOWEST);
	CP_EXPORT void domainTransformFilter(cv::InputArray srcImage, cv::InputArray guideImage, cv::OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm = DTF_L1, const int convolutionType = DTF_RF, const int implementation = DTF_SLOWEST);

	CP_EXPORT void recursiveBilateralFilter(cv::Mat& src, cv::Mat& dest, float sigma_range, float sigma_spatial, int method = 0);
	class CP_EXPORT RecursiveBilateralFilter
	{
	private:
		cv::Mat bgra;

		cv::Mat texture;//texture is joint signal
		cv::Mat destf;
		cv::Mat temp;
		cv::Mat tempw;

		cv::Size size;
	public:
		void setColorLUTGaussian(float* lut, float sigma);
		void setColorLUTLaplacian(float* lut, float sigma);
		void init(cv::Size size_);
		RecursiveBilateralFilter(cv::Size size);
		RecursiveBilateralFilter();
		~RecursiveBilateralFilter();
		void operator()(const cv::Mat& src, cv::Mat& dest, float sigma_range, float sigma_spatial);
		void operator()(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, float sigma_range, float sigma_spatial);
	};


	CP_EXPORT void binalyWeightedRangeFilter(const cv::Mat& src, cv::Mat& dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void binalyWeightedRangeFilter(const cv::Mat& src, cv::Mat& dst, int D, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBinalyWeightedRangeFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBinalyWeightedRangeFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, int D, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void centerReplacedBinalyWeightedRangeFilter(const cv::Mat& src, const cv::Mat& center, cv::Mat& dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void nonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma = -1.0, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void nonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma = -1.0, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void epsillonFilter(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, int borderType=cv::BORDER_REPLICATE);

	CP_EXPORT void jointNonLocalMeansFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointNonLocalMeansFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedJointNonLocalMeansFilter(cv::Mat& src, cv::Mat& weightMap, cv::Mat& guide, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma);

	CP_EXPORT void iterativeBackProjectionDeblurGaussian(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double sigma, const double lambda, const int iteration);
	CP_EXPORT void iterativeBackProjectionDeblurBilateral(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double sigma_color, const double sigma_space, const double lambda, const int iteration);

	CP_EXPORT void bilateralFilterPermutohedralLattice(cv::Mat& src, cv::Mat& dest, float sigma_space, float sigma_color);

	class CP_EXPORT CrossBasedLocalFilter
	{
		int minSearch;
		struct cross
		{
			uchar hp;
			uchar hm;
			float divh;
			uchar vp;
			uchar vm;
			float divv;
		};
		cv::Size size;
		int r;
		int thresh;
		cross* crossdata;
		template <class T>
		void orthogonalIntegralImageFilterF_(cv::Mat& src, cv::Mat& dest);
		template <class T>
		void orthogonalIntegralImageFilterF_(cv::Mat& src, cv::Mat& weight, cv::Mat& dest);
		template <class T>
		void orthogonalIntegralImageFilterI_(cv::Mat& src, cv::Mat& dest);
		template <class T>
		void orthogonalIntegralImageFilterI_(cv::Mat& src, cv::Mat& weight, cv::Mat& dest);

	public:
		enum
		{
			CROSS_BASED_LOCAL_FILTER_ARM_BASIC = 0,
			CROSS_BASED_LOCAL_FILTER_ARM_SAMELENGTH,
			CROSS_BASED_LOCAL_FILTER_ARM_SMOOTH_SAMELANGTH
		};
		void setMinSearch(int val);
		cv::Mat areaMap;
		~CrossBasedLocalFilter();
		CrossBasedLocalFilter();
		CrossBasedLocalFilter(cv::Mat& guide, const int r_, const int thresh_);

		void getCrossAreaCountMap(cv::Mat& dest, int type = CV_8U);

		void makeKernel(cv::Mat& guide, const int r, int thresh, int method = CrossBasedLocalFilter::CROSS_BASED_LOCAL_FILTER_ARM_BASIC);
		void makeKernel(cv::Mat& guide, const int r, int thresh, double smoothingrate, int method = CrossBasedLocalFilter::CROSS_BASED_LOCAL_FILTER_ARM_BASIC);
		void visualizeKernel(cv::Mat& dest, cv::Point& pt);

		void operator()(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, int thresh, int iteration = 1);
		void operator()(cv::Mat& src, cv::Mat& dest);
		void operator()(cv::Mat& src, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, const int r, int thresh, int iteration = 1);
		void operator()(cv::Mat& src, cv::Mat& weight, cv::Mat& dest);
	};

	class CP_EXPORT CrossBasedLocalMultipointFilter
	{
		void crossBasedLocalMultipointFilterSrc1Guidance1_(cv::Mat& src, cv::Mat& joint, cv::Mat& dest, const int radius, const float eps);
		void crossBasedLocalMultipointFilterSrc1Guidance3SSE_(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const int thresh, const float eps);
		void crossBasedLocalMultipointFilterSrc1Guidance3_(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const float eps);
		void crossBasedLocalMultipointFilterSrc1Guidance1SSE_(cv::Mat& src, cv::Mat& joint, cv::Mat& dest, const int radius, const int thresh, const float eps);

	public:
		CrossBasedLocalFilter clf;

		void operator()(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const int thresh, const float eps, bool initCLF = true);
	};
	void  crossBasedLocalMultipointFilter(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const int thresh, const float eps);

	enum
	{
		PROCESS_LAB = 0,
		PROCESS_BGR
	};

#define VOLUME_TYPE CV_32F
	//cost volume filtering
	class CP_EXPORT CostVolumeRefinement
	{
	public:

		enum
		{
			L1_NORM = 1,
			L2_NORM = 2,
			EXP = 3
		};
		enum
		{
			COST_VOLUME_BOX = 0,
			COST_VOLUME_GAUSSIAN,
			COST_VOLUME_MEDIAN,
			COST_VOLUME_BILATERAL,
			COST_VOLUME_BILATERAL_SP,
			COST_VOLUME_GUIDED,
			COST_VOLUME_CROSS_BASED_ADAPTIVE_BOX
		};
		enum
		{
			SUBPIXEL_NONE = 0,
			SUBPIXEL_QUAD,
			SUBPIXEL_LINEAR
		};
		//L1: min(abs(d-D(p)),data_trunc) or L2: //min((d-D(p))^2,data_trunc)
		void buildCostVolume(cv::Mat& disp, cv::Mat& mask, int data_trunc, int metric);
		void buildWeightedCostVolume(cv::Mat& disp, cv::Mat& weight, int data_trunc, int metric);
		void buildCostVolume(cv::Mat& disp, int dtrunc, int metric);


		int minDisparity;
		int numDisparity;
		int sub_method;
		std::vector<cv::Mat> dsv;
		std::vector<cv::Mat> dsv2;
		CostVolumeRefinement(int disparitymin, int disparity_range);
		void wta(cv::Mat& dest);
		void subpixelInterpolation(cv::Mat& dest, int method);

		//void crossBasedAdaptiveboxRefinement(cv::Mat& disp, cv::Mat& guide,cv::Mat& dest, int data_trunc, int metric, int r, int thresh,int iter=1);
		void medianRefinement(cv::Mat& disp, cv::Mat& dest, int data_trunc, int metric, int r, int iter = 1);

		void boxRefinement(cv::Mat& disp, cv::Mat& dest, int data_trunc, int metric, int r, int iter = 1);
		void weightedBoxRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& dest, int data_trunc, int metric, int r, int iter = 1);

		void gaussianRefinement(cv::Mat& disp, cv::Mat& dest, int data_trunc, int metric, int r, double sigma, int iter = 1);
		void weightedGaussianRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& dest, int data_trunc, int metric, int r, double sigma, int iter = 1);

		void jointBilateralRefinement(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void jointBilateralRefinement2(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);
		void jointBilateralRefinementSP(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);
		void jointBilateralRefinementSP2(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void weightedJointBilateralRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void weightedJointBilateralRefinementSP(cv::Mat& disp, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void guidedRefinement(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double eps, int iter = 1);
		void weightedGuidedRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double eps, int iter = 1);
	};

	CP_EXPORT void detailEnhancementBilateral(cv::Mat& src, cv::Mat& dest, int d, float sigma_color, float sigma_space, float boost, int color = PROCESS_LAB);

	class CP_EXPORT DenoiseDXTShrinkage
	{
	private:
		enum
		{
			DenoiseDCT = 0,
			DenoiseDHT = 1,
			DenoiseDWT = 2//not supported
		};

		int basis;
		cv::Size patch_size;
		cv::Size size;
		cv::Mat buff;
		cv::Mat sum;

		cv::Mat im;

		int channel;
		void body(float *src0, float* dest0, float *src1, float* dest1, float *src2, float* dest2, float Th);

		void body(float *src, float* dest, float Th);
		void body(float *src, float* dest, float* wmap, float Th);

		void bodyTest(float *src, float* dest, float Th);

		void body(float *src, float* dest, float Th, int dr);

		void div(float* inplace0, float* inplace1, float* inplace2, float* w0, float* w1, float* w2, const int size1);
		void div(float* inplace0, float* inplace1, float* inplace2, const int patch_area, const int size1);

		void div(float* inplace0, float* w0, const int size1);
		void div(float* inplace0, const int patch_area, const int size1);

		void decorrelateColorForward(float* src, float* dest, int width, int height);
		void decorrelateColorInvert(float* src, float* dest, int width, int height);

	public:
		bool isSSE;
		void cvtColorOrder32F_BGR2BBBBGGGGRRRR(const cv::Mat& src, cv::Mat& dest);
		void cvtColorOrder32F_BBBBGGGGRRRR2BGR(const cv::Mat& src, cv::Mat& dest);

		void init(cv::Size size_, int color_, cv::Size patch_size_);
		DenoiseDXTShrinkage(cv::Size size, int color, cv::Size patch_size_ = cv::Size(8, 8));
		DenoiseDXTShrinkage();
		void operator()(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0);

		void shearable(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0, int direct = 0);
		void weighted(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0);

		void test(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8));
	};

	class CP_EXPORT HazeRemove
	{
	public:
		cv::Size size;
		cv::Mat dark;
		std::vector<cv::Mat> minvalue;
		cv::Mat tmap;
		cv::Scalar A;

		void darkChannel(cv::Mat& src, int r);
		void getAtmosphericLight(cv::Mat& srcImage, double topPercent = 0.1);
		void getTransmissionMap(float omega = 0.95f);
		void removeHaze(cv::Mat& src, cv::Mat& trans, cv::Scalar v, cv::Mat& dest, float clip = 0.3f);
		HazeRemove();
		~HazeRemove();

		void getAtmosphericLightImage(cv::Mat& dest);
		void showTransmissionMap(cv::Mat& dest, bool isPseudoColor = false);
		void showDarkChannel(cv::Mat& dest, bool isPseudoColor = false);
		void operator() (cv::Mat& src, cv::Mat& dest, int r_dark, double toprate, int r_joint, double e_joint);
		void gui(cv::Mat& src, std::string wname = "hazeRemove");
	};

	//============================================================================================================================================================
	//stereo 3D viewsynthesis Functions 
	//============================================================================================================================================================

	CP_EXPORT double calcBadPixel(cv::InputArray groundtruth, cv::InputArray disparityImage, cv::InputArray mask, double th, double amp);
	CP_EXPORT double calcBadPixel(cv::InputArray groundtruth, cv::InputArray disparityImage, cv::InputArray mask, double th, double amp, cv::OutputArray outErr);
	class CP_EXPORT StereoEval
	{
		void threshmap_init();
	public:
		bool isInit;
		std::string message;
		cv::Mat state_all;
		cv::Mat state_nonocc;
		cv::Mat state_disc;
		cv::Mat ground_truth;
		cv::Mat mask_all;
		cv::Mat all_th;
		cv::Mat mask_nonocc;
		cv::Mat nonocc_th;
		cv::Mat mask_disc;
		cv::Mat disc_th;
		double amp;
		double all;
		double nonocc;
		double disc;

		double allMSE;
		double nonoccMSE;
		double discMSE;

		void init(cv::Mat& groundtruth, cv::Mat& maskNonocc, cv::Mat& maskAll, cv::Mat& maskDisc, double amp);

		StereoEval();
		StereoEval(std::string groundtruthPath, std::string maskNonoccPath, std::string maskAllPath, std::string maskDiscPath, double amp);
		StereoEval(cv::Mat& groundtruth, cv::Mat& maskNonocc, cv::Mat& maskAll, cv::Mat& maskDisc, double amp);
		~StereoEval(){ ; }

		void getBadPixel(cv::Mat& src, double threshold = 1.0, bool isPrint = true);
		void getMSE(cv::Mat& src, bool isPrint = true);
		virtual void operator() (cv::Mat& src, double threshold = 1.0, bool isPrint = true, int disparity_scale = 1);
		void compare(cv::Mat& before, cv::Mat& after, double threshold = 1.0, bool isPrint = true);
	};

	class CP_EXPORT Calibrator
	{
	private:
		std::vector<std::vector<cv::Point3f>> objectPoints;
		std::vector<std::vector<cv::Point2f>> imagePoints;
		std::vector<cv::Point3f> chessboard3D;

		void generatechessboard3D();
		void initRemap();

	public:
		cv::Point2f getImagePoint(const int number_of_chess, const int index = -1);
		cv::Size imageSize;
		cv::Mat intrinsic;
		cv::Mat distortion;

		cv::Size patternSize;
		float lengthofchess;
		int numofchessboards;
		double rep_error;

		std::vector<cv::Mat> rt;
		std::vector<cv::Mat> tv;
		int flag;

		cv::Mat mapu, mapv;

		void init(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_);
		Calibrator(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_);
		Calibrator();
		~Calibrator();

		void setIntrinsic(double focal_length);
		void solvePnP(const int number_of_chess, cv::Mat& r, cv::Mat& t);
		void readParameter(char* name);
		void writeParameter(char* name);
		bool findChess(cv::Mat& im, cv::Mat& dest);
		void pushImagePoint(std::vector<cv::Point2f> point);
		void pushObjectPoint(std::vector<cv::Point3f> point);
		void undistort(cv::Mat& src, cv::Mat& dest);
		void printParameters();
		double operator()();//calibrate camera
	};

	class CP_EXPORT MultiCameraCalibrator
	{
	private:
		std::vector<std::vector<cv::Point3f>> objectPoints;
		std::vector<std::vector<std::vector<cv::Point2f>>> imagePoints;
		std::vector<cv::Point3f> chessboard3D;
		std::vector<cv::Mat> reR;
		std::vector<cv::Mat> reT;
		cv::Mat E;
		cv::Mat F;
		cv::Mat Q;
		cv::Mat intrinsicRect;

		double reprojectionerr;

		void generatechessboard3D();
		void initRemap();

	public:
		int flag;
		int numofcamera;
		cv::Size imageSize;
		std::vector<cv::Mat> intrinsic;
		std::vector<cv::Mat> distortion;

		cv::Size patternSize;
		float lengthofchess;
		int numofchessboards;

		std::vector<cv::Mat> R;
		std::vector<cv::Mat> P;
		std::vector<cv::Mat> mapu;
		std::vector<cv::Mat> mapv;

		void readParameter(char* name);
		void writeParameter(char* name);

		void init(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_, int numofcamera_);
		MultiCameraCalibrator(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_, int numofcamera_);

		MultiCameraCalibrator();
		~MultiCameraCalibrator();

		//MultiCameraCalibrator cloneParameters();
		bool findChess(std::vector<cv::Mat>& im, std::vector <cv::Mat>& dest = std::vector<cv::Mat>(0));

		void printParameters();

		double getRectificationErrorBetween(int a, int b);
		double getRectificationErrorDisparity();
		double getRectificationErrorDisparityBetween(int ref1, int ref2);
		double getRectificationError();

		//Calibration
		void operator ()(bool isFixIntrinsic = false, int refCamera1 = 0, int refCamera2 = 0);
		void rectifyImageRemap(cv::Mat& src, cv::Mat& dest, int numofcamera);
	};

	CP_EXPORT float rectifyMultiCollinear(
		const std::vector<cv::Mat>& cameraMatrix,
		const std::vector<cv::Mat>& distCoeffs,
		const int anchorView1,
		const int anchorView2,
		const std::vector<std::vector<std::vector<cv::Point2f>> >& anchorpt,
		cv::Size imageSize, const std::vector<cv::Mat>& relativeR, const std::vector<cv::Mat>& relativeT,
		std::vector<cv::Mat>& R, std::vector<cv::Mat>& P, cv::Mat& Q,
		double alpha, cv::Size newImgSize,
		cv::Rect* anchorROI1, cv::Rect* anchorROI2, int flags);

	void lookat(const cv::Point3d& from, const cv::Point3d& to, cv::Mat& destR);
	void eular2rot(double pitch, double roll, double yaw, cv::Mat& dest);

	void rotPitch(cv::Mat& src, cv::Mat& dest, const double pitch);
	void rotYaw(cv::Mat& src, cv::Mat& dest, const double yaw);

	void stereoInterlace(cv::Mat& lim, cv::Mat& rim, cv::Mat& dest, int d, int left_right_swap);
	void stereoAnaglyph(cv::Mat& lim, cv::Mat& rim, cv::Mat& dest, int shift);

	void disp16S2depth16U(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	void disp8U2depth32F(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	void disp16S2depth32F(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	void depth32F2disp8U(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	void depth16U2disp8U(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);

	class CP_EXPORT dispRefinement
	{
	private:

	public:

		int r;
		int th;
		int iter_ex;
		int th_r;
		int r_flip;
		int iter;
		int iter_g;
		int r_g;
		int eps_g;
		int th_FB;

		dispRefinement();
		void boundaryDetect(cv::Mat& src, cv::Mat& guid, cv::Mat& dest, cv::Mat& mask);
		void dispRefine(cv::Mat& src, cv::Mat& guid, cv::Mat& guid_mask, cv::Mat& alpha);
		void operator()(cv::Mat& src, cv::Mat& guid, cv::Mat& dest);
	};

	class CP_EXPORT mattingMethod
	{
	private:
		cv::Mat trimap;
		cv::Mat trimask;
		cv::Mat f;
		cv::Mat b;
		cv::Mat a;

	public:

		int r;
		int iter;
		int iter_g;
		int r_g;
		int eps_g;
		int th_FB;
		int r_Wgauss;
		int sigma_Wgauss;
		int th;

		mattingMethod();
		void boundaryDetect(cv::Mat& disp);
		void getAmap(cv::Mat& img);
		void getFBimg(cv::Mat& img);
		void operator()(cv::Mat& img, cv::Mat& disp, cv::Mat& alpha, cv::Mat& Fimg, cv::Mat& Bimg);
	};

	CP_EXPORT void boundaryReconstructionFilter(cv::InputArray src, cv::OutputArray dest, cv::Size ksize, const float frec = 1.f, const float color = 1.f, const float space = 1.f);

	CP_EXPORT void correctDisparityBoundaryFillOcc(cv::Mat& src, cv::Mat& refimg, const int r, cv::Mat& dest);
	CP_EXPORT void correctDisparityBoundary(cv::Mat& src, cv::Mat& refimg, const int r, const int edgeth, cv::Mat& dest, const int secondr = 0, const int minedge = 0);

	class CP_EXPORT StereoBMSimple
	{
		cv::Mat bufferGray;
		cv::Mat bufferGray1;
		cv::Mat bufferGray2;
		cv::Mat bufferGray3;
		cv::Mat bufferGray4;
		cv::Mat bufferGray5;
		void shiftImage(cv::Mat& src, cv::Mat& dest, const int shift);

		std::vector<cv::Mat> target;
		std::vector<cv::Mat> refference;
		cv::Mat specklebuffer;
	public:
		int border;


		int sobelAlpha;
		int prefSize;
		int prefParam;
		int prefParam2;
		int preFilterCap;

		int uniquenessRatio;
		int SADWindowSize;
		int SADWindowSizeH;

		int numberOfDisparities;
		int minDisparity;
		int error_truncate;
		int disp12diff;

		int speckleWindowSize;
		int speckleRange;
		double eps;
		std::vector<cv::Mat> DSI;

		bool isProcessLBorder;
		bool isMinCostFilter;
		bool isBoxSubpix;
		int subboxRange;
		int subboxWindowR;
		cv::Mat minCostMap;

		bool isBT;
		int P1;
		int P2;

		cv::Mat costMap;
		cv::Mat weightMap;

		StereoBMSimple(int blockSize, int minDisp, int disparityRange);
		void StereoBMSimple::imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::Mat& output, int mindis, int range);

		void prefilter(cv::Mat& src1, cv::Mat& src2);
		void preFilter(cv::Mat& src, cv::Mat& dest, int param);

		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::OutputArray output = cv::noArray());

		//void getMatchingCostSADandSobel(vector<Mat>& target, vector<Mat>& refference, const int d,Mat& dest);
		void textureAlpha(cv::Mat& src, cv::Mat& dest, const int th1, const int th2, const int r);
		void getMatchingCostSADAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);
		void getMatchingCostSAD(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, const int d, cv::Mat& dest);
		void halfPixel(cv::Mat& src, cv::Mat& srcp, cv::Mat& srcm);
		void getMatchingCostBT(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, const int d, cv::Mat& dest);
		void getMatchingCostBTAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);

		void getOptScanline();
		void getMatchingCost(const int d, cv::Mat& dest);
		void getCostAggregationBM(cv::Mat& src, cv::Mat& dest, int d);
		void getCostAggregation(cv::Mat& src, cv::Mat& dest, cv::InputArray joint = cv::noArray());
		void getWTA(std::vector<cv::Mat>& dsi, cv::Mat& dest);

		void refineFromCost(cv::Mat& src, cv::Mat& dest);
		void getWeightUniqness(cv::Mat& disp);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, StereoEval& eval);

		//post filter
		void uniquenessFilter(cv::Mat& costMap, cv::Mat& dest);
		enum
		{
			SUBPIXEL_NONE = 0,
			SUBPIXEL_QUAD,
			SUBPIXEL_LINEAR
		};
		int subpixMethod;
		void subpixelInterpolation(cv::Mat& dest, int method);

		void fastLRCheck(cv::Mat& costMap, cv::Mat& dest);
		void fastLRCheck(cv::Mat& dest);
		void minCostFilter(cv::Mat& costMap, cv::Mat& dest);
	};

	//StereoBM from OpenCV2.4.9 version
	class CP_EXPORT StereoBM2
	{
	public:
		enum {
			PREFILTER_NORMALIZED_RESPONSE = 0, PREFILTER_XSOBEL = 1,
			BASIC_PRESET = 0, FISH_EYE_PRESET = 1, NARROW_PRESET = 2
		};

		//! the default constructor
		StereoBM2();
		//! the full constructor taking the camera-specific preset, number of disparities and the SAD window size
		StereoBM2(int preset, int ndisparities = 0, int SADWindowSize = 21);
		//! the method that reinitializes the state. The previous content is destroyed
		void init(int preset, int ndisparities = 0, int SADWindowSize = 21);

		//! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
		void operator()(cv::InputArray left, cv::InputArray right, cv::OutputArray disparity, int disptype = CV_16S);

		//! pointer to the underlying CvStereoBMState
		cv::Ptr<CvStereoBMState> state;
	};

	class CP_EXPORT StereoBMEx
	{
		StereoBM2 bm;
		void parameterUpdate();
		void prefilter(cv::Mat& sl, cv::Mat& sr);
	public:
		double prefilterAlpha;
		int preFilterType; // =CV_STEREO_BM_NORMALIZED_RESPONSE now
		int preFilterSize; // averaging window size: ~5x5..21x21
		int preFilterCap; // the output of pre-filtering is clipped by [-preFilterCap,preFilterCap]
		// correspondence using Sum of Absolute Difference (SAD)
		int SADWindowSize; // ~5x5..21x21
		int minDisparity;  // minimum disparity (can be negative)
		int numberOfDisparities; // maximum disparity - minimum disparity (> 0)

		// post-filtering
		int textureThreshold;  // the disparity is only computed for pixels
		// with textured enough neighborhood
		int uniquenessRatio;   // accept the computed disparity d* only if
		// SAD(d) >= SAD(d*)*(1 + uniquenessRatio/100.)
		// for any d != d*+/-1 within the search range.
		int speckleWindowSize; // disparity variation window
		int speckleRange; // acceptable range of variation in window

		int trySmallerWindows; // if 1, the results may be more accurate,
		// at the expense of slower processing 
		int disp12MaxDiff;

		int lr_thresh;
		int isOcclusion;
		int medianKernel;

		StereoBMEx(int preset, int ndisparities = 0, int SADWindowSize_ = 21);
		void showPostFilter();
		void showPreFilter();
		void setPreFilter(int preFilterType_, int preFilterSize_, int preFilterCap_);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, cv::Mat& dispR, int bd);

		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, StereoEval& eval);
	};

	class CP_EXPORT StereoDP
	{
		void shiftImage(cv::Mat& src, cv::Mat& dest, const int shift);
	public:
		int minDisparity;
		int disparityRange;

		int isOcclusion;
		int medianKernel;

		double param1;
		double param2;
		double param3;
		double param4;
		double param5;

		StereoDP(int minDisparity_, int disparityRange_);

		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd = 0);

		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, StereoEval& eval);
	};

	class CP_EXPORT StereoSGBM2
	{
	public:
		enum { DISP_SHIFT = 4, DISP_SCALE = (1 << DISP_SHIFT) };

		//! the default constructor
		StereoSGBM2();

		//! the full constructor taking all the necessary algorithm parameters
		StereoSGBM2(int minDisparity, int numDisparities, cv::Size SADWindowSize,
			int P1 = 0, int P2 = 0, int disp12MaxDiff = 0,
			int preFilterCap = 0, int uniquenessRatio = 0,
			int speckleWindowSize = 0, int speckleRange = 0,
			bool fullDP = false, double _costAlpha = 1.0, int _ad_max = 31, int _subpixel_r = 4, int _subpixel_th = 32);
		//! the destructor
		virtual ~StereoSGBM2();

		//! the stereo correspondence operator that computes disparity map for the specified rectified stereo pair
		virtual void operator()(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp_l, cv::Mat& disp_r);
		virtual void operator()(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp_l);
		void test(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp_l, cv::Point& pt, cv::Mat& gt);
		int minDisparity;
		int numberOfDisparities;
		cv::Size SADWindowSize;
		int preFilterCap;
		int uniquenessRatio;
		int P1, P2;
		int speckleWindowSize;
		int speckleRange;
		int disp12MaxDiff;
		bool fullDP;
		int subpixel_r;
		int subpixel_th;
		int ad_max;
		double costAlpha;

	protected:
		cv::Mat buffer;
	};

	class CP_EXPORT StereoSGBMEx
	{
		StereoSGBM2 sgbm;
	public:
		int minDisparity;
		int numberOfDisparities;
		cv::Size SADWindowSize;
		int P1;
		int P2;
		int disp12MaxDiff;
		int preFilterCap;
		int uniquenessRatio;
		int speckleWindowSize;
		int speckleRange;
		bool fullDP;

		double costAlpha;
		int ad_max;

		int cross_check_threshold;
		int subpixel_r;
		int subpixel_th;
		int isOcclusion;
		int isStreakingRemove;
		int medianKernel;

		StereoSGBMEx(int minDisparity_, int numDisparities_, cv::Size SADWindowSize_,
			int P1_ = 0, int P2_ = 0, int disp12MaxDiff_ = 0,
			int preFilterCap_ = 0, int uniquenessRatio_ = 0,
			int speckleWindowSize_ = 0, int speckleRange_ = 0,
			bool fullDP_ = true);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, cv::Mat& dispR, int bd, int lr_thresh);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, cv::Mat& dispR, cv::InputArray ref = cv::noArray());
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, StereoEval& eval);

		void test(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd, cv::Point& pt, cv::Mat& gt);
	};

	/*
	for legacy code
	class XCV_EXPORTS StereoGCEx
	{
	CvStereoGCState* gc;
	public:
	int maxIters;
	int numDisparities;
	int minDisparity;
	int interactionRadius;
	float K;
	float lambda;
	float lambda1;
	float lambda2;
	int occlusionCost;

	int isOcclusion;
	int medianKernel;

	StereoGCEx(int numDisparities_, int maxIteration=1);
	~StereoGCEx();

	void operator()(Mat& leftim, Mat& rightim, Mat& dispL, Mat& dispR, int bd);
	void operator()(Mat& leftim, Mat& rightim, Mat& dispL, int bd);
	void check(Mat& leftim, Mat& rightim, Mat& dispL, Mat& dispR);
	void check(Mat& leftim, Mat& rightim, Mat& dispL);
	};
	*/

	CP_EXPORT void reprojectXYZ(cv::InputArray depth, cv::OutputArray xyz, cv::InputArray intrinsic, cv::InputArray distortion);

	class CP_EXPORT PointCloudShow
	{
	private:
		std::string wname;
		cv::Point pt;
		bool isInit;
		int x;
		int y;
		int z;
		int pitch;
		int yaw;

		int loolatx;
		int loolaty;

		int renderOpt;
		int viewSW;

		int br;
		int bth;

		int maxr;
		bool isDrawLine;
		bool isWrite;
		bool isLookat;
		cv::Point3d look;

		void depth2XYZ(cv::Mat& srcDepth, float focal);
		void depth2XYZ(cv::Mat& srcDepth, cv::InputArray srcK, cv::InputArray srcDist);

	public:
		cv::Mat xyz;

		PointCloudShow();
		void disparity2XYZ(cv::Mat& srcDisparity, float disp_amp, float focal, float baseline);

		void loop(cv::Mat& image, cv::Mat& srcDisparity, float disp_amp, float focal, float baseline, int loopcount);

		void loop_depth(cv::Mat& image, cv::Mat& srcDepth, float focal, int loopcount);
		void loop_depth(cv::Mat& image, cv::Mat& srcDepth, cv::Mat& image2, cv::Mat& srcDepth2, float focal, cv::Mat& R, cv::Mat& t, cv::Mat& k, int loopcount);

		void PointCloudShow::warp_xyz(cv::Mat& dest, cv::Mat& image, cv::InputArray xyz, cv::InputArray R, cv::InputArray t, cv::InputArray k);
		void warp_depth(cv::Mat& dest, cv::Mat& image, cv::Mat& srcDepth, float focal, cv::InputArray R, cv::InputArray t, cv::InputArray k);
		void warp_depth(cv::Mat& dest, cv::Mat& image, cv::Mat& srcDepth, cv::InputArray srcK, cv::InputArray srcDist, cv::InputArray R, cv::InputArray t, cv::InputArray destK, cv::InputArray destDist);
		void warp_disparity(cv::Mat& dest, cv::Mat& image, cv::Mat& srcDisparity, float disp_amp, float focal, float baseline, cv::Mat& R, cv::Mat& t, cv::Mat& k);
	};

	class CP_EXPORT StereoViewSynthesis
	{

	private:
		void depthfilter(cv::Mat& depth, cv::Mat& depth2, cv::Mat& mask2, int viewstep, double disp_amp);
		template <class T>
		void analyzeSynthesizedViewDetail_(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp, cv::Mat& srcsynth, cv::Mat& ref);
		template <class T>
		void viewsynth(const cv::Mat& srcL, const cv::Mat& srcR, const cv::Mat& dispL, const cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);
		template <class T>
		void makeMask_(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp);
		template <class T>
		void viewsynthSingle(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);

	public:
		//warping parameters
		enum
		{
			WAPR_IMG_INV = 0,//Mori et al.
			WAPR_IMG_FWD_SUB_INV, //Zenger et al.
		};
		int warpMethod;

		int warpInterpolationMethod;//Nearest, Linear or Cubic
		bool warpSputtering;
		int large_jump;

		//warped depth filtering parameters
		enum
		{
			DEPTH_FILTER_SPECKLE = 0,
			DEPTH_FILTER_MEDIAN,
			DEPTH_FILTER_MEDIAN_ERODE,
			DEPTH_FILTER_CRACK,
			DEPTH_FILTER_MEDIAN_BILATERAL,
			DEPTH_FILTER_NONE
		};
		int depthfiltermode;
		int warpedMedianKernel;

		int warpedSpeckesWindow;
		int warpedSpeckesRange;

		int bilateral_r;
		float bilateral_sigma_space;
		float bilateral_sigma_color;

		//blending parameter

		int blendMethod;
		double blend_z_thresh;

		//post filtering parameters
		enum
		{
			POST_GAUSSIAN_FILL = 0,
			POST_FILL,
			POST_NONE
		};
		int postFilterMethod;

		enum
		{
			FILL_OCCLUSION_LINE = 0,
			FILL_OCCLUSION_REFLECT = 1,
			FILL_OCCLUSION_STRETCH = -1,
			FILL_OCCLUSION_HV = 2,
			FILL_OCCLUSION_INPAINT_NS = 3, // OpenCV Navier-Stokes algorithm
			FILL_OCCLUSION_INPAINT_TELEA = 4, // OpenCV A. Telea algorithm
		};
		int inpaintMethod;

		double inpaintr;//parameter for opencv inpaint 
		int canny_t1;
		int canny_t2;

		cv::Size occBlurSize;

		cv::Size boundaryKernelSize;
		double boundarySigma;
		double boundaryGaussianRatio;

		//preset
		enum
		{
			PRESET_FASTEST = 0,
			PRESET_SLOWEST,
		};

		StereoViewSynthesis();
		StereoViewSynthesis(int preset);
		void init(int preset);

		void operator()(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
		void operator()(const cv::Mat& srcL, const cv::Mat& srcR, const cv::Mat& dispL, const cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);

		cv::Mat diskMask;
		cv::Mat allMask;//all mask
		cv::Mat boundaryMask;//disparity boundary
		cv::Mat nonOcclusionMask;
		cv::Mat occlusionMask;//half and full occlusion
		cv::Mat fullOcclusionMask;//full occlusion
		cv::Mat nonFullOcclusionMask; //bar of full occlusion
		cv::Mat halfOcclusionMask;//left and right half ooclusion

		void viewsynthSingleAlphaMap(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);
		void alphaSynth(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
		void noFilter(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
		void analyzeSynthesizedViewDetail(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp, cv::Mat& srcsynth, cv::Mat& ref);
		void analyzeSynthesizedView(cv::Mat& srcsynth, cv::Mat& ref);
		void makeMask(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp);
		void makeMask(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp, cv::Mat& srcsynth, cv::Mat& ref);

		void check(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, cv::Mat& ref);
		void check(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, cv::Mat& ref);
		void preview(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, int invalidvalue, double disp_amp);
		void preview(cv::Mat& src, cv::Mat& disp, int invalidvalue, double disp_amp);
	};

	class CP_EXPORT OpticalFlowBM
	{
		cv::Mat buffSpeckle;
	public:
		std::vector<cv::Mat>cost;
		std::vector<cv::Mat>ocost;
		OpticalFlowBM();
		void cncheck(cv::Mat& srcx, cv::Mat& srcy, cv::Mat& destx, cv::Mat& desty, int thresh, int invalid);
		void operator()(cv::Mat& curr, cv::Mat& next, cv::Mat& dstx, cv::Mat& dsty, cv::Size ksize, int minx, int maxx, int miny, int maxy, int bd = 30);
	};
	CP_EXPORT void drawOpticalFlow(const cv::Mat_<cv::Point2f>& flow, cv::Mat& dst, float maxmotion = -1);
	CP_EXPORT void mergeFlow(cv::Mat& flow, cv::Mat& xflow, cv::Mat& yflow);

	class CP_EXPORT DepthMapSubpixelRefinment
	{
		cv::Mat pslice;
		cv::Mat cslice;
		cv::Mat mslice;
		double calcReprojectionError(const cv::Mat& leftim, const cv::Mat& rightim, const cv::Mat& leftdisp, const cv::Mat& rightdisp, int disp_amp, bool left2right = true);

		template <class S, class T>
		void getDisparitySubPixel_Integer(cv::Mat& src, cv::Mat& dest, int disp_amp);
		void bluidCostSlice(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dest, int metric, int truncate);
	public:
		DepthMapSubpixelRefinment();
		void operator()(const cv::Mat& leftim, const cv::Mat& rightim, const cv::Mat& leftdisp, const cv::Mat& rightdisp, int disp_amp, cv::Mat& leftdest, cv::Mat& rightdest);
		void naive(const cv::Mat& leftim, const cv::Mat& rightim, const cv::Mat& leftdisp, const cv::Mat& rightdisp, int disp_amp, cv::Mat& leftdest, cv::Mat& rightdest);
	};

	//depth map hole filling
	enum
	{
		FILL_DISPARITY = 0,
		FILL_DEPTH = 1
	};
	CP_EXPORT void fillOcclusion(cv::InputOutputArray src, int invalidvalue = 0, int method = FILL_DISPARITY);// for disparity map
	CP_EXPORT void jointColorDepthFillOcclusion(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, const cv::Size ksize, double threshold);

	//remove Streaking Noise in stereo DP matching and hole filling function
	CP_EXPORT void removeStreakingNoise(cv::Mat& src, cv::Mat& dest, int th);
	CP_EXPORT void removeStreakingNoiseV(cv::Mat& src, cv::Mat& dest, int th);

	enum
	{
		LR_CHECK_DISPARITY_BOTH = 0,
		LR_CHECK_DISPARITY_ONLY_L,
		LR_CHECK_DISPARITY_ONLY_R
	};
	CP_EXPORT void fastLRCheckDisparity(cv::Mat& disp, const double disp12diff, double amp);
	CP_EXPORT void LRCheckDisparity(cv::Mat& left_disp, cv::Mat& right_disp, int disparity_max, const int disp12diff = 0, double invalidvalue = 0, const int amp = 1, const int mode = LR_CHECK_DISPARITY_BOTH);
	CP_EXPORT void LRCheckDisparityAdd(cv::Mat& left_disp, cv::Mat& right_disp, const int disp12diff = 0, const int amp = 1);

	enum
	{
		DISPARITY_COLOR_GRAY = 0,
		DISPARITY_COLOR_GRAY_OCC,
		DISPARITY_COLOR_PSEUDO
	};
	CP_EXPORT void cvtDisparityColor(cv::Mat& src, cv::Mat& dest, int minDisparity, int numDisparities, int option = DISPARITY_COLOR_GRAY, int amp = 16);
	CP_EXPORT void imshowDisparity(std::string name, cv::Mat& src, int option = DISPARITY_COLOR_GRAY, int minDisparity = 0, int numDisparities = 0, int amp = 1);

	CP_EXPORT void createDisparityALLMask(cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void createDisparityNonOcclusionMask(cv::Mat& src, double amp, double thresh, cv::Mat& dest);

	CP_EXPORT void dispalityFitPlane(cv::InputArray disparity, cv::InputArray image, cv::OutputArray dest, int slicRegionSize, float slicRegularization, float slicMinRegionRatio, int slicMaxIteration, int ransacNumofSample, float ransacThreshold);
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//under construction
	/////////////////////////////////////////////////////////////////////////////////////////////////

	CP_EXPORT void nnUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void linearUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void setUpsampleMask(cv::InputArray src, cv::OutputArray dst);

	CP_EXPORT void jointBilateralUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c, double sigma_s);
	CP_EXPORT void jointBilateralNNUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c, double sigma_s);
	CP_EXPORT void jointBilateralLinearUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c);

	CP_EXPORT void noiseAwareFilterDepthUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c, double sigma_d, double sigma_s, double eps, double tau);
	/////////////////////////////////////////////////////////////////////////////////////////////////


}//end namespace


//template for new files
/*



#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{



}




*/