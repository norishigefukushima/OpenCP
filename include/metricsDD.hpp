#pragma once

#include "common.hpp"
#include <fpplus/fpplus.h>

namespace cp
{
	CP_EXPORT double PSNR_DD(const doubledouble* src1, const doubledouble* src2, const int size);
	CP_EXPORT double PSNR_DD(const doubledouble* src1, cv::InputArray src2);
}