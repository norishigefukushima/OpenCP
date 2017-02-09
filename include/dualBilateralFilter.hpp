#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void dualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void dualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_guide_color, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void dualBilateralWeightMap(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void dualBilateralWeightMapXOR(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void dualBilateralWeightMapBase(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, int d, double sigma_color, double sigma_guide_color, double sigma_space, int borderType, bool isLaplace);
	CP_EXPORT void separableDualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int sp_kernel_type = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableDualBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int sp_kernel_type = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
}