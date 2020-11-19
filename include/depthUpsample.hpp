#pragma once
#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT void noiseAwareFilterDepthUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, const double sigma_c, const double sigma_d, const double sigma_s, const double eps, const double tau);
}