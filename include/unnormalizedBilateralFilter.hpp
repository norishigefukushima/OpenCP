#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void unnormalizedBilateralFilter(cv::Mat& src, cv::Mat& dest, const int r, const float sigma_range, const float sigma_space, const bool isEnhance, int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void unnormalizedBilateralFilterCenterBlur(cv::Mat& src, cv::Mat& dest, const int r, const float sigma_range, const float sigma_space, const float sigma_space_center, const bool isEnhance, int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void unnormalizedBilateralFilterMulti(cv::Mat& src, cv::Mat& dest, const int r, const float sigma_range, const float sigma_space, const int level, const bool isEnhance, int borderType = cv::BORDER_DEFAULT);
}