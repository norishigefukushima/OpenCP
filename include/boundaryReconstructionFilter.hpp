#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void boundaryReconstructionFilter(cv::InputArray src, cv::OutputArray dest, cv::Size ksize, const float frec = 1.f, const float color = 1.f, const float space = 1.f);
}