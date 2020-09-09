#pragma once

#include "common.hpp"

namespace cp
{

	class WeightedHistogram
	{
	private:
		float* histbuff;
		int histbuffsize;
	public:
		int histMin;
		int histMax;
		int histSize;
		enum
		{
			L0_NORM = 0,
			L1_NORM,
			L2_NORM,
			EXP
		};

		enum
		{
			NO_WEIGHT_MODE = 0,
			GAUSSIAN_MODE,
			BILATERAL_MODE,
			NO_WEIGHT_MEDIAN,
			GAUSSIAN_MEDIAN,
			BILATERAL_MEDIAN
		};

		int truncate;
		float* hist;

		WeightedHistogram(int truncate_val, int mode_ = WeightedHistogram::MAX, const int max_val = 256);
		~WeightedHistogram();

		void clear();
		void add(float addval, int bin, int metric = 0);
		void addWithRange(float addval, int bin, int metric = 0);

		enum
		{
			MAX = 0,
			MEDIAN
		};
		int mode;
		int returnVal();

		int returnMax();
		int returnMedian();
		int returnMaxwithRange();
		int returnMedianwithRange();
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