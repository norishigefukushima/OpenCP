#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT void addBoxMask(cv::Mat& mask, const int boundx, const int boundy);//overwrite box mask
	CP_EXPORT cv::Mat createBoxMask(const cv::Size size, const int boundx, const int boundy);//create box mask
	CP_EXPORT cv::Mat createBoxMask(const cv::Size size, const int top, const int bottom, const int left, const int right);//create box mask with 4 parameter
	CP_EXPORT void setBoxMask(cv::Mat& mask, const int boundx, const int boundy);//clear mask and then set box mask
}