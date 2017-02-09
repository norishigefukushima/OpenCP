#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void iterativeBackProjectionDeblurGaussian(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double sigma, const double lambda, const int iteration);
	CP_EXPORT void iterativeBackProjectionDeblurBilateral(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double sigma_color, const double sigma_space, const double lambda, const int iteration);
}