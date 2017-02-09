#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		NO_WEIGHT = 0,
		GAUSSIAN,
		BILATERAL
	};

	CP_EXPORT void weightedMedianFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedMedianFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedMedianFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor1, double sigmaColor2, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedModeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedModeFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedModeFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::OutputArray dst, int r, int truncate, double sigmaColor1, double sigmaColor2, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedModeFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::InputArray mask, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedMedianFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::InputArray mask, cv::OutputArray dst, int r, int truncate, double sigmaColor, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedModeFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::InputArray mask, cv::OutputArray dst, int r, int truncate, double sigmaColor1, double sigmaColor2, double sigmaSpace, int metric, int method);
	CP_EXPORT void weightedweightedMedianFilter(cv::InputArray src, cv::InputArray wmap, cv::InputArray guide, cv::InputArray mask, cv::OutputArray dst, int r, int truncate, double sigmaColor1, double sigmaColor2, double sigmaSpace, int metric, int method);
}