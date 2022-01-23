#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void separableNonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma = -1.0, double alpha = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableNonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma = -1.0, double alpha = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);

	//weight[i] = (-pow(abs(i/sigma), powexp) / powexp), powexp=2: Gaussian, powexp=infinity: Box
	CP_EXPORT void nonLocalMeansFilterL1PatchDistance(const cv::Mat& src, cv::Mat& dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int method = FILTER_DEFAULT, const int borderType = cv::BORDER_REPLICATE);
	//weight[i] = (-pow(abs(i/sigma), powexp) / powexp), powexp=2: Gaussian, powexp=infinity: Box
	CP_EXPORT void nonLocalMeansFilterL1PatchDistance(const cv::Mat& src, cv::Mat& dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int method = FILTER_DEFAULT, const int borderType = cv::BORDER_REPLICATE);
	//weight[i] = (-pow(abs(i/sigma), powexp) / powexp), powexp=2: Gaussian, powexp=infinity: Box
	CP_EXPORT void nonLocalMeansFilter(const cv::Mat& src, cv::Mat& dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int borderType = cv::BORDER_REPLICATE);
	//weight[i] = (-pow(abs(i/sigma), powexp) / powexp), powexp=2: Gaussian, powexp=infinity: Box
	CP_EXPORT void nonLocalMeansFilter(const cv::Mat& src, cv::Mat& dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int borderType = cv::BORDER_REPLICATE);
	//powexp=infinity in nonLocalMeansFilter
	CP_EXPORT void epsillonFilterL1PatchDistance(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, int borderType = cv::BORDER_REPLICATE);
}