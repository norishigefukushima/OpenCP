#pragma once
#include "common.hpp"
#include "inlineMathFunctions.hpp"

namespace cp
{
	CP_EXPORT cv::Mat createWindowFunction(const int r, const double sigma, int depth, const int window_type, const bool isSeparable);
}