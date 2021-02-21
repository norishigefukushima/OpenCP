#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void highDimensionalGaussianFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const cv::Size ksize, const float sigma_range, const float sigma_space, const int border = cv::BORDER_DEFAULT);
}