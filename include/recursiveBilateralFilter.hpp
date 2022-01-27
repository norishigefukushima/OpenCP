#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void recursiveBilateralFilter(cv::Mat& src, cv::Mat& dest, float sigma_range, float sigma_spatial, int method = 0);

	class CP_EXPORT RecursiveBilateralFilter
	{
	private:
		cv::Mat bgra; //src signal of BGRA
		cv::Mat texture; //texture is joint signal of BGRA
		cv::Mat destf;
		cv::Mat temp;
		cv::Mat tempw;

		cv::Size size;
	public:
		void setColorLUTGaussian(float* lut, float sigma);
		void setColorLUTLaplacian(float* lut, float sigma);
		void alloc(cv::Size size_);
		RecursiveBilateralFilter(cv::Size size);
		RecursiveBilateralFilter();
		~RecursiveBilateralFilter();
		void operator()(const cv::Mat& src, cv::Mat& dest, float sigma_range, float sigma_spatial);
		void operator()(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, float sigma_range, float sigma_spatial);
	};
}