#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void jointNearestFilter(cv::InputArray src, cv::InputArray guide, cv::Size ksize, cv::OutputArray dest);
	CP_EXPORT void jointNearestFilterBase(cv::InputArray src, cv::InputArray guide, cv::Size ksize, cv::OutputArray dest);
}