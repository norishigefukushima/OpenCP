#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void binalyWeightedRangeFilter(cv::InputArray src, cv::OutputArray dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void binalyWeightedRangeFilter(cv::InputArray src, cv::OutputArray dst, int D, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBinalyWeightedRangeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBinalyWeightedRangeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, int D, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void centerReplacedBinalyWeightedRangeFilter(const cv::Mat& src, const cv::Mat& center, cv::Mat& dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
}