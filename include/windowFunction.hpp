#pragma once
#include "common.hpp"
#include "inlineMathFunctions.hpp"

namespace cp
{
	/*
	enum
	{
	GAUSSIAN_WINDOW,
	EXP_L1_WINDOW,//dual exponential, Laplacian distribution
	EXP_L2_WINDOW, //Gaussian window
	EXP_L3_WINDOW,
	EXP_L4_WINDOW,
	EXP_L5_WINDOW,
	EXP_L6_WINDOW,
	EXP_L7_WINDOW,
	EXP_L8_WINDOW,
	EXP_L9_WINDOW,
	EXP_L10_WINDOW,
	EXP_L20_WINDOW,
	EXP_L40_WINDOW,
	EXP_L80_WINDOW,
	EXP_L160_WINDOW,

	BOX_WINDOW,
	BARTLETT_WINDOW,//triangle window
	WELCH_WINDOW,
	PARZEN_WINDOW,//peachwise approximation of Gaussian with cubic
	DIVSQRT_WINDOW,

	HANN_WINDOW,
	HAMMING_WINDOW,
	BLACKMAN_WINDOW,
	NUTTALL_WINDOW,
	AKAIKE_WINDOW,
	FLATTOP_WINDOW,

	WINDOW_TYPE_SIZE
	};
	*/
	CP_EXPORT cv::Mat createWindowFunction(const int r, const double sigma, int depth, const int window_type, const bool isSeparable);
}