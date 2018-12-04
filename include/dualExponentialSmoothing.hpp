#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		VECTOR_WITHOUT = 0,
		VECTOR_AVX = 1
	};

	CP_EXPORT float sigma2LaplacianSmootihngAlpha(const float sigma, float p);

	CP_EXPORT void LaplacianSmoothingIIRFilter(cv::Mat& src, cv::Mat& dest, const double sigma_, int opt=VECTOR_AVX);
	CP_EXPORT void LaplacianSmoothingFIRFilter(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt = VECTOR_AVX);
}