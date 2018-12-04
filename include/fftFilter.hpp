#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void GaussianFilterFFT(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double sigma, const int depth = CV_64F);

	CP_EXPORT void wienerDeconvolutionGauss(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double sigma, const double eps, const int depth = CV_64F);
	CP_EXPORT void wienerDeconvolution(const cv::Mat& src, cv::Mat& dest, const cv::Mat & kernel, double mu);
}