#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void LucyRichardsonGauss(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const int iteration);
	CP_EXPORT void LucyRichardsonGaussTikhonov(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float beta, const int iteration);

	CP_EXPORT void iterativeBackProjectionDeblurGaussianTV(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const float th, const int iteration);

	CP_EXPORT void iterativeBackProjectionDeblurGaussian(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration);
	CP_EXPORT void iterativeBackProjectionDeblurGaussian(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration, cv::Mat& init);

	CP_EXPORT void iterativeBackProjectionDeblurBoxGaussian(const cv::Mat& src, cv::Mat& dest, const int d, const cv::Size ksize, const float backprojection_sigma, const float lambda, const int iteration);
	CP_EXPORT void iterativeBackProjectionDeblurBoxGaussian(const cv::Mat& src, cv::Mat& dest, const int d, const cv::Size ksize, const float backprojection_sigma, const float lambda, const int iteration, cv::Mat& init);

	CP_EXPORT void iterativeBackProjectionDeblurPascal3x3Gaussian(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float backprojection_sigma, const float lambda, const int iteration, cv::Mat& init);
	CP_EXPORT void iterativeBackProjectionDeblurPascal3x3Gaussian(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float backprojection_sigma, const float lambda, const int iteration);

	CP_EXPORT void iterativeBackProjectionDeblurGaussianFast(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration);

	CP_EXPORT void iterativeBackProjectionDeblurBilateral(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma_space, const float backprojection_sigma_color, const float lambda, const int iteration);
	CP_EXPORT void iterativeBackProjectionDeblurBilateral(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const float sigma, const float backprojection_sigma_space, const float backprojection_sigma_color, const float lambda, const int iteration, cv::Mat& init);

	CP_EXPORT void iterativeBackProjectionDeblurGuidedImageFilter(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double eps, const double sigma_space, const double lambda, const int iteration);

	CP_EXPORT void guiIBP(cv::Mat& src, cv::Mat& ref, std::string wname = "IBP");
}