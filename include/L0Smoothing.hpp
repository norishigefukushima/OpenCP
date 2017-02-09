#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void L0Smoothing(cv::Mat &im8uc3, cv::Mat& dest, float lambda = 0.02f, float kappa = 2.f);
}