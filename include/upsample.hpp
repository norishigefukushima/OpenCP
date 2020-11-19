#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT void nnUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void linearUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cubicUpsample(cv::InputArray src, cv::OutputArray dest, const double a = -1.0);
	CP_EXPORT void setUpsampleMask(cv::InputArray src, cv::OutputArray dst);
}
