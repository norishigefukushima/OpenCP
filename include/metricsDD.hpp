#pragma once

#include "common.hpp"
#include <fpplus/fpplus.h>

namespace cp
{
	CP_EXPORT double getPSNR_DD(doubledouble* src1, doubledouble* src2, const int size);
	CP_EXPORT double getPSNR_DD(doubledouble* src1, cv::Mat& src2);
	CP_EXPORT double getPSNR_DD(cv::Mat& src1, doubledouble* src2);
	CP_EXPORT double getPSNR_DD(cv::Mat& src1, cv::Mat& src2);//src1 and/or src2 must be CV_64FC2
}