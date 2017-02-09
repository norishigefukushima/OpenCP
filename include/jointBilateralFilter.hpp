#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void jointBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedJointBilateralFilter(cv::InputArray src, cv::InputArray weightMap, cv::InputArray guide, cv::OutputArray dest, cv::Size kernelSize, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedJointBilateralFilter(cv::InputArray src, cv::InputArray weightMap, cv::InputArray guide, cv::OutputArray dest, int D, double sigma_color, double sigma_space, int kernelType = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
}