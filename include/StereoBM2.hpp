#pragma once

#include "common.hpp"
#include "depthEval.hpp"

namespace cp
{
	class CP_EXPORT StereoBM2
	{
	public:
		enum {
			PREFILTER_NORMALIZED_RESPONSE = 0, PREFILTER_XSOBEL = 1,
			BASIC_PRESET = 0, FISH_EYE_PRESET = 1, NARROW_PRESET = 2
		};

		//! the default constructor
		StereoBM2();
		//! the full constructor taking the camera-specific preset, number of disparities and the SAD window size
		StereoBM2(int preset, int ndisparities = 0, int SADWindowSize = 21);
		//! the method that reinitializes the state. The previous content is destroyed
		void init(int preset, int ndisparities = 0, int SADWindowSize = 21);

		//! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
		void operator()(cv::InputArray left, cv::InputArray right, cv::OutputArray disparity, int disptype = CV_16S);

		//! pointer to the underlying CvStereoBMState
		cv::Ptr<CvStereoBMState> state;
	};

	class CP_EXPORT StereoBMEx
	{
		StereoBM2 bm;
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

		StereoBMEx(int preset, int ndisparities = 0, int SADWindowSize_ = 21);
		void showPostFilter();
		void showPreFilter();
		void setPreFilter(int preFilterType_, int preFilterSize_, int preFilterCap_);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, int bd);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dispL, cv::Mat& dispR, int bd);

		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& disp, StereoEval& eval);
	};
}