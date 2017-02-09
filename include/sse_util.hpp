#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void memcpy_float_sse(float* dest, float* src, const int size);
	void print_m128(__m128d src);
	void print_m128(__m128 src);
	void print_m128i_char(__m128i src);
	void print_m128i_uchar(__m128i src);
	void print_m128i_short(__m128i src);
	void print_m128i_ushort(__m128i src);
	void print_m128i_int(__m128i src);
	void print_m128i_uint(__m128i src);
}