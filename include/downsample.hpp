#pragma once
#include "common.hpp"

namespace cp
{
	enum class Downsample
	{
		INTER_NEAREST,
		INTER_LINEAR,
		INTER_CUBIC,
		INTER_AREA,
		INTER_LANCZOS4,
		CP_NEAREST,
		CP_LINEAR,
		CP_CUBIC,
		CP_AREA,
		CP_LANCZOS,
		CP_GAUSS,
		CP_GAUSS_FAST,

		SIZE
	};

	//parameter:	0->use default parameter
	//CP_CUBIC:		alpha(1.5)
	//CP_LANCZOS:	order(4)
	//CP_GAUSS:		sigma_clip(3.0)
	//CP_GAUSS_FAST:sigma_clip(3.0)
	//r = (scale/2)*radius_ratio
	//Gauss: sigma = r/parameter
	CP_EXPORT void downsample(cv::InputArray src, cv::OutputArray dest, const int scale, const Downsample downsample_method, const double parameter = 0.0, double radius_ratio = 1.0);
	// add last element for cp::downsample
	CP_EXPORT void downsamplePlus1(cv::InputArray src, cv::OutputArray dest, const int scale, const Downsample downsample_method, const double parameter = 0.0, double radius_ratio = 1.0);
	CP_EXPORT std::string getDowsamplingMethod(const Downsample method);
}