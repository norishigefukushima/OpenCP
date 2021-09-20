#pragma once

#include "common.hpp"
#include <fpplus/fpplus.h>

namespace cp
{
	CP_EXPORT void convertToDD(const cv::Mat& src, doubledouble* dest);
	CP_EXPORT void convertDDTo(const doubledouble* src, cv::Mat& dest);
	CP_EXPORT void convertDDTo(const doubledouble* src, const cv::Size size, cv::Mat& dest, const int depth);

}