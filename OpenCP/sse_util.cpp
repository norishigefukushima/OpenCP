#include "opencp.hpp"
#include <stdio.h>

namespace cp
{

	void memcpy_float_sse(float* dest, float* src, const int size)
	{
		int x = 0;
		for (x = 0; x <= size - 4; x += 4)
		{
			_mm_storeu_ps(dest + x, _mm_loadu_ps(src + x));
		}
		for (; x < size; x++)
		{
			dest[x] = src[x];
		}
	}

	//broadcast
	/*
	__m128 xxxx = _mm_shuffle_ps(first, first, 0x00); // _MM_SHUFFLE(0, 0, 0, 0)
	__m128 yyyy = _mm_shuffle_ps(first, first, 0x55); // _MM_SHUFFLE(1, 1, 1, 1)
	__m128 zzzz = _mm_shuffle_ps(first, first, 0xAA); // _MM_SHUFFLE(2, 2, 2, 2)
	__m128 wwww = _mm_shuffle_ps(first, first, 0xFF); // _MM_SHUFFLE(3, 3, 3, 3)
	*/


	void print_m128(__m128d src)
	{
		printf_s("%5.3f %5.3f\n",
			src.m128d_f64[0], src.m128d_f64[1]);
	}

	void print_m128(__m128 src)
	{
		printf_s("%5.3f %5.3f %5.3f %5.3f\n",
			src.m128_f32[0], src.m128_f32[1],
			src.m128_f32[2], src.m128_f32[3]);
	}

	void print_m128i_char(__m128i src)
	{
		for (int i = 0; i < 16; i++)
		{
			printf_s("%3d ", src.m128i_i8[i]);
		}
		printf_s("\n");
	}

	void print_m128i_uchar(__m128i src)
	{
		for (int i = 0; i < 16; i++)
		{
			printf_s("%3d ", src.m128i_u8[i]);
		}
		printf_s("\n");
	}

	void print_m128i_short(__m128i src)
	{
		for (int i = 0; i < 8; i++)
		{
			printf_s("%3d ", src.m128i_i16[i]);
		}
		printf_s("\n");
	}

	void print_m128i_ushort(__m128i src)
	{
		for (int i = 0; i < 8; i++)
		{
			printf_s("%3d ", src.m128i_u16[i]);
		}
		printf_s("\n");
	}

	void print_m128i_int(__m128i src)
	{
		for (int i = 0; i < 4; i++)
		{
			printf_s("%3d ", src.m128i_i32[i]);
		}
		printf_s("\n");
	}

	void print_m128i_uint(__m128i src)
	{
		for (int i = 0; i < 4; i++)
		{
			printf_s("%3d ", src.m128i_u32[i]);
		}
		printf_s("\n");
	}
}