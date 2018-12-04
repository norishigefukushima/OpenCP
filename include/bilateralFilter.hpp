#pragma once

#include "common.hpp"

namespace cp
{
	//In RANGE_NORM_L1, range norm is mesured in L1, and this is the same as OpenCV. This implementation can save LUT size, but is not accurate implementation of bilateral filter's paper.
	//In RANGE_NORM_L2 range norm is mesured in L2, This implementation is accurate implementation of bilateral filter's paper.
	enum
	{
		RANGE_NORM_L1 = 1,//exp(-0.5*sigma^2*(|r|+|g|+|b|)^2)
		RANGE_NORM_L2 = 2,//exp(-0.5*sigma^2*(sqrt(r^2+g^2+b^2))^2)

	};

	CP_EXPORT void bilateralFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void bilateralWeightMap(cv::InputArray src_, cv::OutputArray dst_, cv::Size kernelSize, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableBilateralFilter(const cv::Mat& src, cv::Mat& dst, cv::Size kernelSize, double sigma_color, double sigma_space, double alpha, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedBilateralFilter(cv::InputArray src, cv::InputArray weight, cv::OutputArray dst, int D, double sigma_color, double sigma_space, int method, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedBilateralFilter(cv::InputArray src, cv::InputArray weight, cv::OutputArray dst, cv::Size kernelSize, double sigma_color, double sigma_space, int method, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void bilateralFilterL2(cv::InputArray src, cv::OutputArray dest, int radius, double sigma_color, double sigma_space, int borderType = cv::BORDER_REPLICATE);
}