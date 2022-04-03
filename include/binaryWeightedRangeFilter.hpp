#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void binaryWeightedRangeFilter(cv::InputArray src, cv::OutputArray dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void binaryWeightedRangeFilter(cv::InputArray src, cv::OutputArray dst, int D, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBinaryWeightedRangeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointBinaryWeightedRangeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, int D, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);

	CP_EXPORT void centerReplacedBinaryWeightedRangeFilter(const cv::Mat& src, const cv::Mat& center, cv::Mat& dst, cv::Size kernelSize, float threshold, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
}