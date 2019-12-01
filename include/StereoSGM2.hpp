#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT StereoSGBM2
	{
	public:
		enum { DISP_SHIFT = 4, DISP_SCALE = (1 << DISP_SHIFT) };

		//! the default constructor
		StereoSGBM2();

		//! the full constructor taking all the necessary algorithm parameters
		StereoSGBM2(int minDisparity, int numDisparities, cv::Size SADWindowSize,
			int P1 = 0, int P2 = 0, int disp12MaxDiff = 0,
			int preFilterCap = 0, int uniquenessRatio = 0,
			int speckleWindowSize = 0, int speckleRange = 0,
			bool fullDP = false, double _costAlpha = 1.0, int _ad_max = 31, int _subpixel_r = 4, int _subpixel_th = 32);

		//! the stereo correspondence operator that computes disparity map for the specified rectified stereo pair
		void operator()(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp_l, cv::Mat& disp_r);
		void operator()(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp_l);
		void test(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp_l, cv::Point& pt, cv::Mat& gt);

		int minDisparity;
		int numberOfDisparities;
		cv::Size SADWindowSize;
		int preFilterCap;
		int uniquenessRatio;
		int P1, P2;
		int speckleWindowSize;
		int speckleRange;
		int disp12MaxDiff;
		bool fullDP;
		int subpixel_r;
		int subpixel_th;
		int ad_max;
		double costAlpha;

		//protected:
		cv::Mat buffer;
	};
}