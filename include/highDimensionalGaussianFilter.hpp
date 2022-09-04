#pragma once

#include "common.hpp"

namespace cp
{
	enum class HDGFSchedule
	{
		COMPUTE,
		LUT_SQRT
	};
	CP_EXPORT void highDimensionalGaussianFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border = cv::BORDER_DEFAULT, HDGFSchedule schedule = HDGFSchedule::COMPUTE);
	CP_EXPORT void highDimensionalGaussianFilter(cv::InputArray src, cv::InputArray guide, cv::InputArray center,cv::OutputArray dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border = cv::BORDER_DEFAULT, HDGFSchedule schedule = HDGFSchedule::COMPUTE);

}