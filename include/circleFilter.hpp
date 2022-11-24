#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void circleFilter(cv::InputArray src, cv::OutputArray dest, const int r, const int borderType = cv::BorderTypes::BORDER_DEFAULT);
}