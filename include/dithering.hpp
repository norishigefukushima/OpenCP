#pragma once

#include "common.hpp"

namespace cp
{

	enum DITHER_METHOD
	{
		OSTROMOUKHOW,
		FLOYD_STEINBERG,
		SIERRA2,
		SIERRA3,
		JARVIS,
		STUCKI,
		BURKES,
		STEAVENSON,
		RANDOM_DIFFUSION,

		DITHERING_NUMBER_OF_METHODS,
	};

	enum DITHER_SCANORDER
	{
		FORWARD,
		MEANDERING,
		IN2OUT, //for kernel sampling
		OUT2IN, //for kernel sampling
		FOURDIRECTION, //for kernel sampling
		FOURDIRECTIONIN2OUT,

		DITHERING_NUMBER_OF_ORDER,
	};

	CP_EXPORT std::string getDitheringOrderName(const int method);
	CP_EXPORT std::string getDitheringMethodName(const int method);
	CP_EXPORT int ditherDestruction(cv::Mat& src, cv::Mat& dest, const int dithering_method, int process_order = OUT2IN);
	CP_EXPORT int dither(const cv::Mat& src, cv::Mat& dest, const int dithering_method, int process_order = MEANDERING);

	CP_EXPORT int ditheringFloydSteinberg(cv::Mat& remap, cv::Mat& dest, int process_order);

	//visualize dithering order for debug
	CP_EXPORT void ditheringOrderViz(cv::Mat& src, int process_order);
}