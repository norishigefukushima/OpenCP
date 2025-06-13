#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void DCCI(const cv::Mat& srcImg, cv::Mat& dstImg, const float threshold = 1.15);
}