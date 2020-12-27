#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void cvt32F8U(const cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void cvt64F8U(const cv::Mat& src, cv::Mat& dest);

	CP_EXPORT void cvt8U32F(const cv::Mat& src, cv::Mat& dest, const float amp);
	CP_EXPORT void cvt8U32F(const cv::Mat& src, cv::Mat& dest);
	
	CP_EXPORT void cvt32F16F(cv::Mat& srcdst);
}