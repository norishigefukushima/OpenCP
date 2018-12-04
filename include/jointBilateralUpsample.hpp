#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void nnUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void linearUpsample(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cubicUpsample(cv::InputArray src, cv::OutputArray dest, const double a=-1.0);
	CP_EXPORT void setUpsampleMask(cv::InputArray src, cv::OutputArray dst);
	CP_EXPORT void noiseAwareFilterDepthUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c, double sigma_d, double sigma_s, double eps, double tau);

	CP_EXPORT void jointBilateralUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c, double sigma_s);
	CP_EXPORT void jointBilateralNNUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c, double sigma_s);
	CP_EXPORT void jointBilateralLinearUpsample(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, double sigma_c);
}