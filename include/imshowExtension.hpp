#pragma once

#include "common.hpp"

namespace cp
{
	//normalize image and then cast to 8U and imshow. NORM_INF scale 0-max
	CP_EXPORT void imshowNormalize(std::string wname, cv::InputArray src, const int norm_type = cv::NORM_MINMAX);

	//scaling ax+b and then cast to 8U and imshow
	CP_EXPORT void imshowScale(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);

	//resize image, cast 8U (optional), and then imshow 
	CP_EXPORT void imshowResize(std::string name, cv::InputArray src, const cv::Size dsize, const double fx = 0.0, const double fy = 0.0, const int interpolation = cv::INTER_NEAREST, bool isCast8U = true);

	//3 times count down
	CP_EXPORT void imshowCountDown(std::string wname, cv::InputArray src, const int waitTime = 1000, cv::Scalar color = cv::Scalar::all(0), const int pointSize = 128, std::string fontName = "Consolas");
}