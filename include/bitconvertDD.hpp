#pragma once

#include "common.hpp"
#include <fpplus/fpplus.h>

namespace cp
{
	CP_EXPORT void cvtMattoDD(const cv::Mat& src, doubledouble* dest);
	CP_EXPORT void cvtDDtoMat(const doubledouble* src, cv::Mat& dest);
	CP_EXPORT void cvtDDtoMat(const doubledouble* src, const cv::Size size, cv::Mat& dest, const int depth);

}