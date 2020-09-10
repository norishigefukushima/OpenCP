#pragma once

#include "common.hpp"

namespace cp
{
	
	//D. Min, J. Lu, and M. N. Do. "Depth video enhancement based on weighted mode filtering." IEEE Transactions on Image Processing 21(3), pp. 1176-1190, 2011.
	
	enum class WHF_HISTOGRAM_WEIGHT_FUNCTION
	{
		IMPULSE,
		LINEAR,
		QUADRIC,
		GAUSSIAN,//original paper

		SIZE
	};
	CP_EXPORT std::string getWHFHistogramWeightName(const WHF_HISTOGRAM_WEIGHT_FUNCTION method);

	enum WHF_OPERATION
	{
		BOX_MODE,
		GAUSSIAN_MODE,
		BILATERAL_MODE,//original paper
		BOX_MEDIAN,
		GAUSSIAN_MEDIAN,
		BILATERAL_MEDIAN,

		SIZE
	};
	CP_EXPORT std::string getWHFOperationName(const WHF_OPERATION method);
	
	CP_EXPORT void weightedHistogramFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
	CP_EXPORT void weightedWeightedHistogramFilter(cv::InputArray src, cv::InputArray weight, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
	
	CP_EXPORT void weightedModeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
	CP_EXPORT void weightedWeightedModeFilter(cv::InputArray src, cv::InputArray weight, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
	CP_EXPORT void weightedMedianFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
	CP_EXPORT void weightedWeightedMedianFilter(cv::InputArray src, cv::InputArray weight, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
}