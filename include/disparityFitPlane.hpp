#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void disparityFitPlane(cv::InputArray disparity, cv::InputArray image, cv::OutputArray dest, int slicRegionSize, float slicRegularization, float slicMinRegionRatio, int slicMaxIteration, int ransacNumofSample, float ransacThreshold);
}