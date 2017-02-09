#pragma once

#include "common.hpp"

namespace cp
{
	//split by number of grid
	CP_EXPORT void mergeFromGrid(std::vector<cv::Mat>& src, cv::Size beforeSize, cv::Mat& dest, cv::Size grid, int borderRadius);
	CP_EXPORT void splitToGrid(const cv::Mat& src, std::vector<cv::Mat>& dest, cv::Size grid, int borderRadius);

	CP_EXPORT void mergeHorizon(const std::vector<cv::Mat>& src, cv::Mat& dest);
	CP_EXPORT void splitHorizon(const cv::Mat& src, std::vector<cv::Mat>& dest, int num);
}
