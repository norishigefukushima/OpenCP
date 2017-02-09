#pragma once

#include "common.hpp"

namespace cp
{
#define VOLUME_TYPE CV_32F

	//cost volume filtering
	class CP_EXPORT CostVolumeRefinement
	{
	public:

		enum
		{
			L1_NORM = 1,
			L2_NORM = 2,
			EXP = 3
		};
		enum
		{
			COST_VOLUME_BOX = 0,
			COST_VOLUME_GAUSSIAN,
			COST_VOLUME_MEDIAN,
			COST_VOLUME_BILATERAL,
			COST_VOLUME_BILATERAL_SP,
			COST_VOLUME_GUIDED,
			COST_VOLUME_CROSS_BASED_ADAPTIVE_BOX
		};
		enum
		{
			SUBPIXEL_NONE = 0,
			SUBPIXEL_QUAD,
			SUBPIXEL_LINEAR
		};
		//L1: min(abs(d-D(p)),data_trunc) or L2: //min((d-D(p))^2,data_trunc)
		void buildCostVolume(cv::Mat& disp, cv::Mat& mask, int data_trunc, int metric);
		void buildWeightedCostVolume(cv::Mat& disp, cv::Mat& weight, int data_trunc, int metric);
		void buildCostVolume(cv::Mat& disp, int dtrunc, int metric);


		int minDisparity;
		int numDisparity;
		int sub_method;
		std::vector<cv::Mat> dsv;
		std::vector<cv::Mat> dsv2;
		CostVolumeRefinement(int disparitymin, int disparity_range);
		void wta(cv::Mat& dest);
		void subpixelInterpolation(cv::Mat& dest, int method);

		//void crossBasedAdaptiveboxRefinement(cv::Mat& disp, cv::Mat& guide,cv::Mat& dest, int data_trunc, int metric, int r, int thresh,int iter=1);
		void medianRefinement(cv::Mat& disp, cv::Mat& dest, int data_trunc, int metric, int r, int iter = 1);

		void boxRefinement(cv::Mat& disp, cv::Mat& dest, int data_trunc, int metric, int r, int iter = 1);
		void weightedBoxRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& dest, int data_trunc, int metric, int r, int iter = 1);

		void gaussianRefinement(cv::Mat& disp, cv::Mat& dest, int data_trunc, int metric, int r, double sigma, int iter = 1);
		void weightedGaussianRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& dest, int data_trunc, int metric, int r, double sigma, int iter = 1);

		void jointBilateralRefinement(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void jointBilateralRefinement2(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);
		void jointBilateralRefinementSP(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);
		void jointBilateralRefinementSP2(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void weightedJointBilateralRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void weightedJointBilateralRefinementSP(cv::Mat& disp, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter = 1);

		void guidedRefinement(cv::Mat& disp, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double eps, int iter = 1);
		void weightedGuidedRefinement(cv::Mat& disp, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, int data_trunc, int metric, int r, double eps, int iter = 1);
	};
}