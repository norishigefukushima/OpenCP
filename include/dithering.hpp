#pragma once

#include "common.hpp"

namespace cp
{

	enum DITHER_METHOD
	{
		OSTROMOUKHOW,
		FLOYD_STEINBERG, //Robert W. Floyd and Louis Steinberg, An Adaptive 		Algorithm for Spatial Grayscale.Proceedings of the Society		for Information Display 17 (2) 75 - 77, 1976 
		FAN, //(Floyd-Steinberg derivative) Zhigang Fan, A simple modification of error-diffusion weights. In the Proceedings of SPIE'92.
		SIERRA2,
		SIERRA3,
		JARVIS,//J. F. Jarvis, C. N. Judice and W. H. Ninke, A Survey of 		Techniques for the Display of Continuous Tone Pictures on 		Bi - level Displays.Computer Graphics and Image Processing, 5 13 - 40, 1976.
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