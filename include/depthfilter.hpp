#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		FILL_DISPARITY = 0,
		FILL_DEPTH = 1
	};
	//method FILL_DISPARITY(0),FILL_DEPTH(1)
	CP_EXPORT void fillOcclusion(cv::InputOutputArray src, int invalidvalue = 0, int method = FILL_DISPARITY);// for disparity map
	CP_EXPORT void jointColorDepthFillOcclusion(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, const cv::Size ksize, double threshold);

	//remove Streaking Noise in stereo DP matching and hole filling function
	CP_EXPORT void removeStreakingNoise(cv::Mat& src, cv::Mat& dest, int th);
	CP_EXPORT void removeStreakingNoiseV(cv::Mat& src, cv::Mat& dest, int th);

	enum
	{
		LR_CHECK_DISPARITY_BOTH = 0,
		LR_CHECK_DISPARITY_ONLY_L,
		LR_CHECK_DISPARITY_ONLY_R
	};
	CP_EXPORT void fastLRCheckDisparity(cv::Mat& disp, const double disp12diff, double amp);
	CP_EXPORT void LRCheckDisparity(cv::Mat& left_disp, cv::Mat& right_disp, int disparity_max, const int disp12diff = 0, double invalidvalue = 0, const int amp = 1, const int mode = LR_CHECK_DISPARITY_BOTH);
	CP_EXPORT void LRCheckDisparityAdd(cv::Mat& left_disp, cv::Mat& right_disp, const int disp12diff = 0, const int amp = 1);

	enum class DISPARITY_COLOR
	{
		GRAY = 0,
		GRAY_OCC,
		COLOR_PSEUDO
	};
	CP_EXPORT void cvtDisparityColor(const cv::Mat& src, cv::Mat& dest, const int minDisparity, const int numDisparities, const DISPARITY_COLOR option = DISPARITY_COLOR::GRAY, const int amp = 16);
	CP_EXPORT void imshowDisparity(const std::string name, const cv::Mat& src, const DISPARITY_COLOR option = DISPARITY_COLOR::GRAY, const int minDisparity = 0, const int numDisparities = 0, const int amp = 1);


}