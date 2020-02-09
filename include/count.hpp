#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT int countNaN(cv::InputArray src);
	CP_EXPORT int countInf(cv::InputArray src);
	CP_EXPORT int countDenormalizedNumber(cv::InputArray src);
	CP_EXPORT double countDenormalizedNumberRatio(cv::InputArray src);
}