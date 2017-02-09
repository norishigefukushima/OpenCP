#pragma once

#include "common.hpp"
#include "depthEval.hpp"
#include "StereoSGM2.hpp"

namespace cp
{
	class CP_EXPORT StereoSGBMEx
	{
		StereoSGBM2 sgbm;
	public:
		int minDisparity;
		int numberOfDisparities;
		cv::Size SADWindowSize;
		int P1;
		int P2;
		int disp12MaxDiff;
		int preFilterCap;
		int uniquenessRatio;
		int speckleWindowSize;
		int speckleRange;
		bool fullDP;

		double costAlpha;
		int ad_max;

		int cross_check_threshold;
		int subpixel_r;
		int subpixel_th;
		int isOcclusion;
		int isStreakingRemove;
		int medianKernel;

		StereoSGBMEx(int minDisparity_, int numDisparities_, cv::Size SADWindowSize_,
			int P1_ = 0, int P2_ = 0, int disp12MaxDiff_ = 0,
			int preFilterCap_ = 0, int uniquenessRatio_ = 0,
			int speckleWindowSize_ = 0, int speckleRange_ = 0,
			bool fullDP_ = true);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, cv::Mat& dispR, int bd, int lr_thresh);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, cv::Mat& dispR, cv::InputArray ref = cv::noArray());
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, StereoEval& eval);

		void test(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd, cv::Point& pt, cv::Mat& gt);
	};
}