#pragma once

#include "common.hpp"

namespace cp
{
	//src + boost(src-g*src)
	CP_EXPORT void detailEnhancementBox(cv::InputArray src, cv::OutputArray dest, const int r, const float boost = 1.f);
	//src + boost(src-g*src)
	CP_EXPORT void detailEnhancementGauss(cv::InputArray src, cv::OutputArray dest, const int r, const float sigma_space, const float boost = 1.f);
	//src + boost(src-g*src)
	CP_EXPORT void detailEnhancementBilateral(cv::InputArray src, cv::OutputArray dest, const int r, const float sigma_color, const float sigma_space, const float boost = 1.f);
	//src + boost(src-g*src)
	CP_EXPORT void detailEnhancementGuided(cv::InputArray src, cv::OutputArray dest, const int r, const float eps, const float boost = 1.f);

}