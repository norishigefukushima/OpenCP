#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void pow_fmath(const float a, const cv::Mat&  src, cv::Mat& dest);
	CP_EXPORT void pow_fmath(const cv::Mat& src, const float a, cv::Mat& dest);
	CP_EXPORT void pow_fmath(const cv::Mat& src1, const cv::Mat&  src2, cv::Mat& dest);
	CP_EXPORT void compareRange(cv::InputArray src, cv::OutputArray destMask, const double validMin, const double validMax);
	CP_EXPORT void setTypeMaxValue(cv::InputOutputArray src);
	CP_EXPORT void setTypeMinValue(cv::InputOutputArray src);
}