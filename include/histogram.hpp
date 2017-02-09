#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void drawHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, bool isGrid = true);
	CP_EXPORT void drawAccumulateHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, bool isGrid = true);
	CP_EXPORT void drawAccumulateHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, bool isGrid = true);
	CP_EXPORT void drawHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, bool isGrid = true);
}
