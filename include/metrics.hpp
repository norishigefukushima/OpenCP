#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT double PSNR64F(cv::InputArray src1, cv::InputArray src2);
	CP_EXPORT double MSE(cv::InputArray src1, cv::InputArray src2);
	CP_EXPORT double MSE(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask);
	CP_EXPORT double YPSNR(cv::InputArray src1, cv::InputArray src2);
	CP_EXPORT double calcBadPixel(const cv::Mat& src, const cv::Mat& ref, int threshold);
	CP_EXPORT double SSIM(cv::Mat& src, cv::Mat& ref, double sigma = 1.5);
	CP_EXPORT double calcTV(cv::Mat& src);
	CP_EXPORT double calcEntropy(cv::InputArray src, cv::InputArray mask = cv::noArray());
}