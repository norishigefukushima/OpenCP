#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void blurRemoveMinMax(const cv::Mat& src, cv::Mat& dest, const int r);
	//MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE
	CP_EXPORT void maxFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, int shape = cv::MORPH_RECT);
	CP_EXPORT void maxFilter(cv::InputArray src, cv::OutputArray dest, int radius);
	//MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE
	CP_EXPORT void minFilter(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, int shape = cv::MORPH_RECT);
	CP_EXPORT void minFilter(cv::InputArray src, cv::OutputArray dest, int radius);
}