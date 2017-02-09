#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void bilateralFilterPermutohedralLattice(cv::Mat& src, cv::Mat& dest, float sigma_space, float sigma_color);
}