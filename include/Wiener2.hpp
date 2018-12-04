#pragma once

#include "common.hpp"

namespace cp
{
#if CV_MAJOR_VERSION <= 3
	CP_EXPORT void wiener2(cv::Mat&src, cv::Mat& dest, int szWindowX, int szWindowY);
#endif
}