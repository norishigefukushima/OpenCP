#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void addNoise(cv::InputArray src, cv::OutputArray dest, double sigma, double solt_papper_ratio = 0.0);
}