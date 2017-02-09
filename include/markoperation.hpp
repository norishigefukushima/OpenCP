#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT void addBoxMask(cv::Mat& mask, int boundx, int boundy);
	CP_EXPORT cv::Mat createBoxMask(cv::Size size, int boundx, int boundy);
	CP_EXPORT void setBoxMask(cv::Mat& mask, int boundx, int boundy);
}