#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void jointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, cv::Size ksize, double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, int d, double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int kernel_type = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableJointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, cv::Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableJointDualBilateralFilter(const cv::Mat& src, const cv::Mat& guide1, const cv::Mat& guide2, cv::Mat& dst, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1 = 1.0, double alpha2 = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
}