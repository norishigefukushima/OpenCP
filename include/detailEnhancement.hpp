#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		PROCESS_LAB = 0,
		PROCESS_BGR
	};
	CP_EXPORT void detailEnhancementBilateral(cv::Mat& src, cv::Mat& dest, int d, float sigma_color, float sigma_space, float boost, int color = PROCESS_LAB);
}