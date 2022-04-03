#pragma once

#include "common.hpp"

namespace cp
{
	//compute min/max disparity value from disparity map (d_min=floor(min(d)/amp), d_max=ceil(max(d)/amp))
	CP_EXPORT void computeDisparityMinMax(cv::InputArray disparity, const int amp, int& min_d, int& max_d);
	CP_EXPORT double calcBadPixel(cv::InputArray groundtruth, cv::InputArray disparityImage, cv::InputArray mask, double th, double amp);
	CP_EXPORT double calcBadPixel(cv::InputArray groundtruth, cv::InputArray disparityImage, cv::InputArray mask, double th, double amp, cv::OutputArray outErr);
	CP_EXPORT void createDisparityALLMask(cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void createDisparityNonOcclusionMask(cv::Mat& src, double amp, double thresh, cv::Mat& dest);

	class CP_EXPORT StereoEval
	{
		void threshmap_init();
		bool skip_disc = false;
	public:
		int ignoreLeftBoundary;
		int boundingBox;
		bool isInit = false;
		std::string message;
		cv::Mat state_all;
		cv::Mat state_nonocc;
		cv::Mat state_disc;
		cv::Mat ground_truth;
		cv::Mat mask_all;
		cv::Mat all_th;
		cv::Mat mask_nonocc;
		cv::Mat nonocc_th;
		cv::Mat mask_disc;
		cv::Mat disc_th;
		double amp;
		double all;
		double nonocc;
		double disc;

		double allMSE;
		double nonoccMSE;
		double discMSE;

		void init(cv::Mat& groundtruth, cv::Mat& maskNonocc, cv::Mat& maskAll, cv::Mat& maskDisc, double amp);
		void init(cv::Mat& groundtruth, const double amp, const int ignoreLeftBoundary, const int boundingBox);

		StereoEval();
		StereoEval(std::string groundtruthPath, std::string maskNonoccPath, std::string maskAllPath, std::string maskDiscPath, double amp);
		StereoEval(cv::Mat& groundtruth, cv::Mat& maskNonocc, cv::Mat& maskAll, cv::Mat& maskDisc, double amp);
		StereoEval(cv::Mat& groundtruth, const double amp, const int ignoreLeftBoundary = 0, const int boundingBox = 0);

		std::string getBadPixel(cv::Mat& src, double threshold = 1.0, bool isPrint = true);
		std::string getMSE(cv::Mat& src, const int disparity_scale = 1, const bool isPrint = true);
		std::string operator() (cv::InputArray src, const double threshold = 1.0, const int disparity_scale = 1, const bool isPrint = true);
		void compare(cv::Mat& before, cv::Mat& after, double threshold = 1.0, bool isPrint = true);
	};
}