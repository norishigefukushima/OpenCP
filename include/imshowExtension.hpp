#pragma once

#include "common.hpp"

namespace cp
{
	//normalize image and then cast to 8U and imshow
	CP_EXPORT void imshowNormalize(std::string wname, cv::InputArray src);
	//scaling ax+b and then cast to 8U and imshow
	CP_EXPORT void imshowScale(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);
	//3 times count down
	CP_EXPORT void imshowCountDown(std::string wname, cv::InputArray src, const int waitTime = 1000, cv::Scalar color = cv::Scalar::all(0), const int pointSize = 128, std::string fontName = "Consolas");
}