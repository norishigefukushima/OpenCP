#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void GaussianFilterwithMask(const cv::Mat src, cv::Mat& dest, int r, float sigma, int method, cv::Mat& mask);//slowest
	CP_EXPORT void weightedGaussianFilter(cv::Mat& src, cv::Mat& weight, cv::Mat& dest, cv::Size ksize, float sigma, int border_type = cv::BORDER_REPLICATE);
}