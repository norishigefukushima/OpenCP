#pragma once

#include "common.hpp"
#include "stereoEval.hpp"

namespace cp
{
#if CV_MAJOR_VERSION <= 3
	class CP_EXPORT StereoDP
	{
		void shiftImage(cv::Mat& src, cv::Mat& dest, const int shift);
	public:
		int minDisparity;
		int disparityRange;

		int isOcclusion;
		int medianKernel;

		double param1;
		double param2;
		double param3;
		double param4;
		double param5;

		StereoDP(int minDisparity_, int disparityRange_);

		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd = 0);

		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, StereoEval& eval);
	};
#endif
}