#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void separableNonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma = -1.0, double alpha = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void separableNonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma = -1.0, double alpha = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void nonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma = -1.0, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void nonLocalMeansFilter(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma = -1.0, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void epsillonFilter(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, int borderType = cv::BORDER_REPLICATE);
}