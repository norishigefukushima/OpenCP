#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void addNoise(cv::InputArray src, cv::OutputArray dest, const double sigma, const double solt_papper_ratio = 0.0, const int seed=0);
}