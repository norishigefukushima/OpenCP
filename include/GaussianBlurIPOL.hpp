#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		GAUSSIAN_FILTER_DCT,
		GAUSSIAN_FILTER_FIR,
		GAUSSIAN_FILTER_BOX,
		GAUSSIAN_FILTER_EBOX,
		GAUSSIAN_FILTER_SII,
		GAUSSIAN_FILTER_AM,
		GAUSSIAN_FILTER_AM2,
		GAUSSIAN_FILTER_DERICHE,
		GAUSSIAN_FILTER_VYV,
		GAUSSIAN_FILTER_SR,
	};
	CP_EXPORT void GaussianFilter(cv::InputArray src, cv::OutputArray dest, const double sigma_space, const int filter_method, const int K = 0, const double tol = 1.0e-6);
}