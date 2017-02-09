#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		BILATERAL_ORDER2,//underconstruction
		BILATERAL_ORDER2_SEPARABLE//underconstruction
	};

	CP_EXPORT void bilateralFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void bilateralWeightMap(cv::InputArray src_, cv::OutputArray dst_, cv::Size kernelSize, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableBilateralFilter(const cv::Mat& src, cv::Mat& dst, cv::Size kernelSize, double sigma_color, double sigma_space, double alpha, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedBilateralFilter(cv::InputArray src, cv::InputArray weight, cv::OutputArray dst, int D, double sigma_color, double sigma_space, int method, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedBilateralFilter(cv::InputArray src, cv::InputArray weight, cv::OutputArray dst, cv::Size kernelSize, double sigma_color, double sigma_space, int method, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void bilateralFilterL2(cv::InputArray src, cv::OutputArray dest, int radius, double sigma_color, double sigma_space, int borderType = cv::BORDER_REPLICATE);
}