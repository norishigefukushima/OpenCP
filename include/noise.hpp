#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void addNoise(cv::InputArray src, cv::OutputArray dest, const double sigma, const double solt_papper_ratio = 0.0, const uint64 seed = 0);
	CP_EXPORT void addJPEGNoise(cv::InputArray src, cv::OutputArray dest, const int quality);
}