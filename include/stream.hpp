#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void streamCopy(cv::InputArray src, cv::OutputArray dst);
	CP_EXPORT void streamConvertTo8U(cv::InputArray src, cv::OutputArray dst);
	CP_EXPORT void streamSet(cv::InputArray src, const float val, const bool isParallel, const int unroll = 1);
}