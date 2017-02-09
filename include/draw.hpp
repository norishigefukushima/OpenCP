#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void triangle(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);
	CP_EXPORT void triangleinv(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);
	CP_EXPORT void drawPlus(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawTimes(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_typee = 8, int shift = 0);
	CP_EXPORT void drawGrid(cv::InputOutputArray src, cv::Point crossCenter, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawAsterisk(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void eraseBoundary(const cv::Mat& src, cv::Mat& dest, int step, int border = cv::BORDER_REPLICATE);
	CP_EXPORT void imshowNormalize(std::string wname, cv::InputArray src);
	CP_EXPORT void imshowScale(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);
	CP_EXPORT void patchBlendImage(cv::Mat& src1, cv::Mat& src2, cv::Mat& dest, cv::Scalar linecolor = CV_RGB(0, 0, 0), int linewidth = 2, int direction = 0);
}