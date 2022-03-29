#pragma once

#include "common.hpp"

namespace cp
{
	//convert like clone() method;
	CP_EXPORT cv::Mat convert(cv::InputArray src, const int depth, const double alpha = 1.0, const double beta = 0.0);
	//a(x-b)+b
	CP_EXPORT cv::Mat cenvertCentering(cv::InputArray src, int depth, double a = 1.0, double b = 127.5);

	//convert with gray color conversion like clone() method;
	CP_EXPORT cv::Mat convertGray(cv::InputArray& src, const int depth, const double alpha = 1.0, const double beta = 0.0);

	//copyMakeBorder with normal parameters
	CP_EXPORT cv::Mat border(cv::Mat& src, const int top, const int bottom, const int left, const int right, const int borderType = cv::BORDER_DEFAULT);
	//copyMakeBorder with one parameter
	CP_EXPORT cv::Mat border(cv::Mat& src, const int r, const int borderType = cv::BORDER_DEFAULT);
}