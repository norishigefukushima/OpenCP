#pragma once

#include "common.hpp"

namespace cp
{
	//x- a*gauss(x-b)(x-b)
	CP_EXPORT void contrastSToneExp(cv::InputArray src, cv::OutputArray dest, const double sigma = 30.0, const double a = 1.0, const double b = 127.5);
	// Gamma correction
	CP_EXPORT void contrastGamma(cv::InputArray src, cv::OutputArray dest, const double gamma);
	//quantization for posterization
	CP_EXPORT void quantization(cv::InputArray src, cv::OutputArray dest, const int num_levels);

	CP_EXPORT cv::Mat guiContrast(cv::InputArray src, std::string wname = "contrast");
}
