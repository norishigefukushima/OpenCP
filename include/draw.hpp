#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void triangle(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void triangleinv(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void diamond(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void pentagon(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void drawPlus(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawTimes(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_typee = 8, int shift = 0);
	CP_EXPORT void drawAsterisk(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawGrid(cv::InputOutputArray src, cv::Point crossCenter = cv::Point(0, 0), cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);//when crossCenter = Point(0, 0), draw grid line crossing center point
	CP_EXPORT void drawGridMulti(cv::InputOutputArray src, cv::Size division = cv::Size(2, 2), cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);
	
	//copyMakeBorder without boundary expansion: erase step pixels and then copyMakeBorder with step
	CP_EXPORT void eraseBoundary(const cv::Mat& src, cv::Mat& dest, const int step, const int border = cv::BORDER_DEFAULT);
}