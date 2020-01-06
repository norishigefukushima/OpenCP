#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT int countDenormalizedNumber(const cv::Mat& src);
	CP_EXPORT double countDenormalizedNumberRatio(const cv::Mat& src);
}