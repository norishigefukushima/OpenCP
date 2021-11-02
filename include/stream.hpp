#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void streamCopy(cv::InputArray src, cv::OutputArray dst);
	CP_EXPORT void streamConvertTo8U(cv::InputArray src, cv::OutputArray dst);
}