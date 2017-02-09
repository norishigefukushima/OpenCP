#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void stereoInterlace(cv::Mat& lim, cv::Mat& rim, cv::Mat& dest, int d, int left_right_swap);
	CP_EXPORT void stereoAnaglyph(cv::Mat& lim, cv::Mat& rim, cv::Mat& dest, int shift);
}