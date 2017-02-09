#pragma once

#include "common.hpp"

namespace cp
{
	typedef enum
	{
		DTF_L1 = 1,
		DTF_L2 = 2
	}DTF_NORM;

	typedef enum
	{
		DTF_RF = 0,//Recursive Filtering
		DTF_NC = 1,//Normalized Convolution
		DTF_IC = 2,//Interpolated Convolution

	}DTF_METHOD;

	typedef enum
	{
		DTF_BGRA_SSE = 0,
		DTF_BGRA_SSE_PARALLEL,
		DTF_SLOWEST
	}DTF_IMPLEMENTATION;

	CP_EXPORT void domainTransformFilter(cv::InputArray srcImage, cv::OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm = DTF_L1, const int convolutionType = DTF_RF, const int implementation = DTF_SLOWEST);
	CP_EXPORT void domainTransformFilter(cv::InputArray srcImage, cv::InputArray guideImage, cv::OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm = DTF_L1, const int convolutionType = DTF_RF, const int implementation = DTF_SLOWEST);
}