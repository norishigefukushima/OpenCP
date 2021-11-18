#pragma once

#include "common.hpp"
namespace cp
{
	CP_EXPORT void pixelization(cv::InputArray src, cv::OutputArray dest, const cv::Size pixelSize, const cv::Scalar color = cv::Scalar::all(255), const int thichness = 0);
	CP_EXPORT void guiPixelization(std::string wname, cv::Mat& src);
}
