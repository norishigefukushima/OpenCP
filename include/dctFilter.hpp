#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void deblurDCTTihkonov(cv::Mat& src, cv::Mat& dest, const float sigma, const float eps);
	CP_EXPORT void deblurDCTWiener(cv::Mat& src, cv::Mat& dest, const float sigma, const float eps);
}