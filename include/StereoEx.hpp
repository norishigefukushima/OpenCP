#pragma once

#include "common.hpp"
#include "stereoEval.hpp"
#include "StereoSGM2.hpp"

namespace cp
{
	class CP_EXPORT StereoBMEx
	{
		cv::Ptr<cv::StereoBM>  bm;

		void parameterUpdate();
		void prefilter(cv::Mat& sl, cv::Mat& sr);
	public:
		double prefilterAlpha;
		int preFilterType; // =CV_STEREO_BM_NORMALIZED_RESPONSE now
		int preFilterSize; // averaging window size: ~5x5..21x21
		int preFilterCap; // the output of pre-filtering is clipped by [-preFilterCap,preFilterCap]
		// correspondence using Sum of Absolute Difference (SAD)
		int SADWindowSize; // ~5x5..21x21
		int minDisparity;  // minimum disparity (can be negative)
		int numberOfDisparities; // maximum disparity - minimum disparity (> 0)

		// post-filtering
		int textureThreshold;  // the disparity is only computed for pixels
		// with textured enough neighborhood
		int uniquenessRatio;   // accept the computed disparity d* only if
		// SAD(d) >= SAD(d*)*(1 + uniquenessRatio/100.)
		// for any d != d*+/-1 within the search range.
		int speckleWindowSize; // disparity variation window
		int speckleRange; // acceptable range of variation in window

		int trySmallerWindows; // if 1, the results may be more accurate,
		// at the expense of slower processing 
		int disp12MaxDiff;

		int lr_thresh;
		int isOcclusion;
		int medianKernel;

		StereoBMEx(int preset, const int ndisparities = 0, const int disparity_min=0, const int SADWindowSize_ = 21);
		void showPostFilter();
		void showPreFilter();
		void setPreFilter(int preFilterType_, int preFilterSize_, int preFilterCap_);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, cv::Mat& dispR, int bd);

		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, StereoEval& eval);
	};

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