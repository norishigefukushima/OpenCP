#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void guidedFilter(const cv::Mat& src, cv::Mat& dest, const int radius, const float eps);
	CP_EXPORT void guidedFilter(const cv::Mat& src, const cv::Mat& guidance, cv::Mat& dest, const int radius, const float eps);
	CP_EXPORT void guidedFilterMultiCore(const cv::Mat& src, cv::Mat& dest, int r, float eps, int numcore = 0);
	CP_EXPORT void guidedFilterMultiCore(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, int r, float eps, int numcore = 0);
}