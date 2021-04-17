#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void drawHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true, const int normalize_value = 0);
	CP_EXPORT void drawHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true, const int normalize_value = 0);

	CP_EXPORT void drawAccumulateHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true);
	CP_EXPORT void drawAccumulateHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true);
	
	CP_EXPORT void guiLocalDiffHistogram(cv::Mat& src, bool isWait = true, std::string wname = "local histogram");
}
