#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void memcpy_float_sse(float* dest, float* src, const int size);
}