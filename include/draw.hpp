#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void triangle(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color, int thickness = 1);
	CP_EXPORT void triangleinv(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color, int thickness = 1);
	CP_EXPORT void diamond(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color, int thickness = 1);
	CP_EXPORT void pentagon(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color, int thickness = 1);
	CP_EXPORT void drawPlus(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawTimes(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color, int thickness = 1, int line_typee = 8, int shift = 0);
	CP_EXPORT void drawGrid(cv::InputOutputArray src, cv::Point crossCenter, cv::Scalar color, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawAsterisk(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color, int thickness = 1, int line_type = 8, int shift = 0);

	CP_EXPORT void eraseBoundary(const cv::Mat& src, cv::Mat& dest, int step, int border = cv::BORDER_REPLICATE);
}