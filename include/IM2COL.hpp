#pragma once
#include "common.hpp"

namespace cp
{
	
	CP_EXPORT void IM2COL(const cv::Mat& src, cv::Mat& dst, const int neighborhood_r, const int border = cv::BORDER_DEFAULT);
	CP_EXPORT void IM2COL(const std::vector<cv::Mat>& src, cv::Mat& dst, const int neighborhood_r, const int border = cv::BORDER_DEFAULT);

}