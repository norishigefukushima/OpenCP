#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void coherenceEnhancingShockFilter(cv::InputArray src, cv::OutputArray dest, const int sigma, const int str_sigma, const double blend, const int iter);
}