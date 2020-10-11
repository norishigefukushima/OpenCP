#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void cvt32f8u(const cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void cvt8u32f(const cv::Mat& src, cv::Mat& dest, const float amp);
	CP_EXPORT void cvt8u32f(const cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void cvt32F16F(cv::Mat& srcdst);
}