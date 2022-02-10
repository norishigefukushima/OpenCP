#pragma once

#include "common.hpp"
#include "plot.hpp"

namespace cp
{
	//split multichannel image, scaling ax+b, cast to 8U, and then imshow the selected channel with trackbar
	CP_EXPORT void imshowSplitScale(std::string wname, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);
	//split multichannel image and then imshow the selected channel with trackbar.
	CP_EXPORT void imshowSplit(std::string wname, cv::InputArray src);

	//normalize image and then cast to 8U and imshow. NORM_INF(32) scale 0-max
	CP_EXPORT void imshowNormalize(std::string wname, cv::InputArray src, const int norm_type = cv::NORM_MINMAX);

	//scaling ax+b, cast to 8U, and then imshow
	CP_EXPORT void imshowScale(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);

	//scaling a|x|+b, cast to 8U, and then imshow
	CP_EXPORT void imshowScaleAbs(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);

	//resize image, cast 8U (optional), and then imshow 
	CP_EXPORT void imshowResize(std::string name, cv::InputArray src, const cv::Size dsize, const double fx = 0.0, const double fy = 0.0, const int interpolation = cv::INTER_NEAREST, bool isCast8U = true);

	//3 times count down
	CP_EXPORT void imshowCountDown(std::string wname, cv::InputArray src, const int waitTime = 1000, cv::Scalar color = cv::Scalar::all(0), const int pointSize = 128, std::string fontName = "Consolas");

	class CP_EXPORT StackImage
	{
		std::vector<cv::Mat> stack;
		std::string wname;
		int num_stack = 0;
		int stack_max = 0;
	public:
		StackImage(std::string window_name = "image stack");
		void setWindowName(std::string window_name);
		void overwrite(cv::Mat& src);
		void push(cv::Mat& src);
		void show();
		void show(cv::Mat& src);
	};

	enum DRAW_SIGNAL_CHANNEL
	{
		B,
		G,
		R,
		Y
	};
	CP_EXPORT void drawSignalX(cv::Mat& src1, cv::Mat& src2, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size outputImageSize, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = cp::Plot::LINEAR);// color 0:B, 1:G, 2:R, 3:Y
	CP_EXPORT void drawSignalX(cv::InputArrayOfArrays src, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size outputImageSize, int analysisLineHeight, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = cp::Plot::LINEAR);// color 0:B, 1:G, 2:R, 3:Y

	CP_EXPORT void drawSignalY(cv::Mat& src1, cv::Mat& src2, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = cp::Plot::LINEAR);// color 0:B, 1:G, 2:R, 3:Y
	CP_EXPORT void drawSignalY(cv::Mat& src, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = cp::Plot::LINEAR);// color 0:B, 1:G, 2:R, 3:Y
	CP_EXPORT void drawSignalY(std::vector<cv::Mat>& src, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = cp::Plot::LINEAR);// color 0:B, 1:G, 2:R, 3:Y

	CP_EXPORT void guiAnalysisImage(cv::InputArray src);
	CP_EXPORT void guiAnalysisCompare(cv::Mat& src1, cv::Mat& src2);
	CP_EXPORT void imshowAnalysis(std::string winname, cv::Mat& src);
	CP_EXPORT void imshowAnalysis(std::string winname, std::vector<cv::Mat>& s);
	CP_EXPORT void imshowAnalysisCompare(std::string winname, cv::Mat& src1, cv::Mat& src2);
}