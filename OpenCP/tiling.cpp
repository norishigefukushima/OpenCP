#include "tiling.hpp"
#include "../include/inlineSIMDFunctions.hpp"
#include "debugcp.hpp"
#include "../include/onelineCVFunctions.hpp"
#ifdef _OPENMP_LLVM_RUNTIME
#include <omp_llvm.h>
#else
#include <omp.h>
#endif
using namespace std;
using namespace cv;

namespace cp
{
#pragma region cropTile
#pragma region cropTile div
	static void cropTile8U_Replicate(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tileSizeXInternal = src.cols / div_size.width;
		const int tileSizeYInternal = src.rows / div_size.height;
		const int tileSizeXExternal = tileSizeXInternal + leftb + rightb;
		const int tileSizeYExternal = tileSizeYInternal + topb + bottomb;

		if (dest.size() != Size(tileSizeXExternal, tileSizeYExternal)) dest.create(Size(tileSizeXExternal, tileSizeYExternal), CV_8U);

		const int firstXInternalGrid = tileSizeXInternal * idx.x;
		const int left = max(0, leftb - firstXInternalGrid);
		const int sleft = leftb - left;
		const int right = max(0, tileSizeXExternal + firstXInternalGrid - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int memcpySizeX = tileSizeXExternal - left - right;

		const int firstYInternalGrid = tileSizeYInternal * idx.y;
		const int top = max(0, topb - firstYInternalGrid);
		const int stop = topb - top;
		const int bottom = max(0, tileSizeYExternal + firstYInternalGrid - src.rows - topb);
		const int memcpySizeY = tileSizeYExternal - top - bottom;

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_floor(right, 16);

		const uchar* s = src.ptr<uchar>(firstYInternalGrid - stop, firstXInternalGrid);
		uchar* d = dest.ptr<uchar>(top);

		const int STORE_OFFSET = tileSizeXInternal + sright;

		__m128i a;
		for (int j = 0; j < memcpySizeY - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_set1_epi8(s[-firstXInternalGrid]);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (memcpySizeX));
			for (int i = 0; i < right; i += 16)
			{
				a = _mm_set1_epi8(s[-firstXInternalGrid + src.cols - 1]);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_set1_epi8(s[-firstXInternalGrid]);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (memcpySizeX));
			for (int i = 0; i < RIGHT; i += 16)
			{
				a = _mm_set1_epi8(s[-firstXInternalGrid + src.cols - 1]);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[tileSizeXInternal - 1];
			}

			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			uchar* s = dest.ptr<uchar>(top);
			uchar* d = dest.ptr<uchar>(j);
			memcpy(d, s, sizeof(uchar) * (tileSizeXExternal));
		}

		const int sidx = tileSizeYInternal * (div_size.height - idx.y) + topb - 1;
		const int didx = tileSizeYInternal * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar) * tileSizeXExternal);
		}
	}

	static void cropTile32F_Replicate(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;
		const int dest_tilex = tilex + leftb + rightb;
		const int dest_tiley = tiley + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_32F);

		const int top_tilex = tilex * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int top_tiley = tiley * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley - (src.rows - top_tiley + topb));

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const float* s = src.ptr<float>(tiley * idx.y - stop, top_tilex);
		float* d = dest.ptr<float>(top);

		const int STORE_OFFSET = tilex + sright;

		__m256 a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_set1_ps(s[-top_tilex]);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < right; i += 8)
			{
				a = _mm256_set1_ps(s[-top_tilex + src.cols - 1]);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_set1_ps(s[0]);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < RIGHT; i += 8)
			{
				a = _mm256_set1_ps(s[tilex - 1]);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[tilex - 1];
			}

			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			float* s = dest.ptr<float>(top);
			float* d = dest.ptr<float>(j);
			memcpy(d, s, sizeof(float) * (dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx));
			float* d = dest.ptr<float>(didx + j);
			memcpy(d, s, sizeof(float) * dest_tilex);
		}
	}

	static void cropTile64F_Replicate(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;
		const int dest_tilex = tilex + leftb + rightb;
		const int dest_tiley = tiley + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_64F);

		const int top_tilex = tilex * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int top_tiley = tiley * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley - (src.rows - top_tiley + topb));

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const double* s = src.ptr<double>(tiley * idx.y - stop, top_tilex);
		double* d = dest.ptr<double>(top);

		const int STORE_OFFSET = tilex + sright;

		__m256d a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_set1_pd(s[-top_tilex]);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < right; i += 4)
			{
				a = _mm256_set1_pd(s[-top_tilex + src.cols - 1]);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_set1_pd(s[0]);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < RIGHT; i += 4)
			{
				a = _mm256_set1_pd(s[tilex - 1]);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[tilex - 1];
			}

			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			double* s = dest.ptr<double>(top);
			double* d = dest.ptr<double>(j);
			memcpy(d, s, sizeof(double) * (dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx));
			double* d = dest.ptr<double>(didx + j);
			memcpy(d, s, sizeof(double) * dest_tilex);
		}
	}


	static void cropTile8U_Reflect101(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tileSizeXInternal = src.cols / div_size.width;//src.cols / div_size.width
		const int tileSizeYInternal = src.rows / div_size.height;//src.rows / div_size.height
		const int tileSizeXExternal = tileSizeXInternal + leftb + rightb;
		const int tileSizeYExternal = tileSizeYInternal + topb + bottomb;

		if (dest.size() != Size(tileSizeXExternal, tileSizeYExternal)) dest.create(Size(tileSizeXExternal, tileSizeYExternal), CV_8U);

		const int firstXInternalGrid = tileSizeXInternal * idx.x;
		const int left = max(0, leftb - firstXInternalGrid);
		const int sleft = leftb - left;
		const int right = max(0, tileSizeXExternal - (src.cols - firstXInternalGrid + leftb));
		const int sright = leftb + rightb - right;
		const int copysizex = tileSizeXExternal - left - right;

		const int firstYInternalGrid = tileSizeYInternal * idx.y;
		const int top = max(0, topb - firstYInternalGrid);
		const int stop = topb - top;
		const int bottom = max(0, tileSizeYExternal - (src.rows - firstYInternalGrid + topb));

		const int copysizey = tileSizeYExternal - top - bottom;

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_floor(right, 16);

		const uchar* s = src.ptr<uchar>(firstYInternalGrid - stop, firstXInternalGrid);
		uchar* d = dest.ptr<uchar>(top);

		const int LOAD_OFFSET1 = left - 16 - firstXInternalGrid;
		const int LOAD_OFFSET2 = src.cols - 16 - firstXInternalGrid;
		const int LOAD_OFFSET3 = src.cols - 1 - firstXInternalGrid;
		const int STORE_OFFSET = tileSizeXInternal + sright;

		const __m128i vm = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		__m128i a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i + 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < right; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i - 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i + 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < RIGHT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i - 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i - 1];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			uchar* s = dest.ptr<uchar>(2 * top - j);
			uchar* d = dest.ptr<uchar>(j);
			memcpy(d, s, sizeof(uchar) * (tileSizeXExternal));
		}

		const int sidx = tileSizeYInternal * (div_size.height - idx.y) + topb - 2;
		const int didx = tileSizeYInternal * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx - j));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar) * tileSizeXExternal);
		}
	}

	static void cropTile32F_Reflect101(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;
		const int dest_tilex = tilex + leftb + rightb;
		const int dest_tiley = tiley + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_32F);

		const int top_tilex = tilex * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int top_tiley = tiley * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley - (src.rows - top_tiley + topb));

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const float* s = src.ptr<float>(tiley * idx.y - stop) + top_tilex;
		float* d = dest.ptr<float>(top);

		const int LOAD_OFFSET1 = left - 8 - top_tilex;
		const int LOAD_OFFSET2 = src.cols - 8 - top_tilex;
		const int LOAD_OFFSET3 = src.cols - 1 - top_tilex;
		const int STORE_OFFSET = tilex + sright;

		__m256 a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_loadu_reverse_ps(s + LOAD_OFFSET1 - i + 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < right; i += 8)
			{
				a = _mm256_loadu_ps(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_loadu_ps(s + LOAD_OFFSET1 - i + 1);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < RIGHT; i += 8)
			{
				a = _mm256_loadu_ps(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i - 1];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			float* s = dest.ptr<float>(2 * top - j);
			float* d = dest.ptr<float>(j);
			memcpy(d, s, sizeof(float) * (dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 2;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx - j));
			float* d = dest.ptr<float>(didx + j);
			memcpy(d, s, sizeof(float) * dest_tilex);
		}
	}

	static void cropTile64F_Reflect101(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;
		const int dest_tilex = tilex + leftb + rightb;
		const int dest_tiley = tiley + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_64F);

		const int top_tilex = tilex * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int top_tiley = tiley * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley - (src.rows - top_tiley + topb));

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const double* s = src.ptr<double>(tiley * idx.y - stop) + top_tilex;
		double* d = dest.ptr<double>(top);

		const int LOAD_OFFSET1 = left - 4 - top_tilex;
		const int LOAD_OFFSET2 = src.cols - 4 - top_tilex;
		const int LOAD_OFFSET3 = src.cols - 1 - top_tilex;
		const int STORE_OFFSET = tilex + sright;

		__m256d a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET1 - i + 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < right; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET1 - i + 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < RIGHT; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i - 1];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			double* s = dest.ptr<double>(2 * top - j);
			double* d = dest.ptr<double>(j);

			memcpy(d, s, sizeof(double) * (dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 2;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx - j));
			double* d = dest.ptr<double>(didx + j);

			memcpy(d, s, sizeof(double) * dest_tilex);
		}
	}


	static void cropTile8U_Reflect(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tileSizeXInternal = src.cols / div_size.width;
		const int tileSizeYInternal = src.rows / div_size.height;
		const int tileSizeXExternal = tileSizeXInternal + leftb + rightb;
		const int tileSizeYExternal = tileSizeYInternal + topb + bottomb;

		if (dest.size() != Size(tileSizeXExternal, tileSizeYExternal)) dest.create(Size(tileSizeXExternal, tileSizeYExternal), CV_8U);

		const int firstXInternalGrid = tileSizeXInternal * idx.x;
		const int left = max(0, leftb - firstXInternalGrid);
		const int sleft = leftb - left;
		const int right = max(0, tileSizeXExternal - (src.cols - firstXInternalGrid + leftb));
		const int sright = leftb + rightb - right;
		const int copysizex = tileSizeXExternal - left - right;

		const int firstYInternalGrid = tileSizeYInternal * idx.y;
		const int top = max(0, topb - firstYInternalGrid);
		const int stop = topb - top;
		const int bottom = max(0, tileSizeYExternal - (src.rows - firstYInternalGrid + topb));

		const int copysizey = tileSizeYExternal - top - bottom;

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_floor(right, 16);

		const uchar* s = src.ptr<uchar>(firstYInternalGrid - stop, firstXInternalGrid);
		uchar* d = dest.ptr<uchar>(top);

		const int LOAD_OFFSET1 = left - 16 - firstXInternalGrid;
		const int LOAD_OFFSET2 = src.cols - 16 - firstXInternalGrid;
		const int LOAD_OFFSET3 = src.cols - 1 - firstXInternalGrid;
		const int STORE_OFFSET = tileSizeXInternal + sright;

		const __m128i vm = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		__m128i a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < right; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < RIGHT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, 2 * top - j - 1));
			uchar* d = dest.ptr<uchar>(j);
			memcpy(d, s, sizeof(uchar) * (tileSizeXExternal));
		}

		const int sidx = tileSizeYInternal * (div_size.height - idx.y) + topb - 1;
		const int didx = tileSizeYInternal * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx - j));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar) * tileSizeXExternal);
		}
	}

	static void cropTile32F_Reflect(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;
		const int dest_tilex = tilex + leftb + rightb;
		const int dest_tiley = tiley + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_32F);

		const int top_tilex = tilex * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int top_tiley = tiley * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley + top_tiley - src.rows - topb);

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const float* s = src.ptr<float>(tiley * idx.y - stop, top_tilex);
		float* d = dest.ptr<float>(top);

		const int LOAD_OFFSET1 = left - 8 - top_tilex;
		const int LOAD_OFFSET2 = src.cols - 8 - top_tilex;
		const int LOAD_OFFSET3 = src.cols - 1 - top_tilex;
		const int STORE_OFFSET = tilex + sright;

		__m256 a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < right; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < RIGHT; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			float* s = dest.ptr<float>(max(0, 2 * top - j - 1));
			float* d = dest.ptr<float>(j);
			memcpy(d, s, sizeof(float) * (dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx - j));
			float* d = dest.ptr<float>(didx + j);

			memcpy(d, s, sizeof(float) * dest_tilex);
		}
	}

	static void cropTile64F_Reflect(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;
		const int dest_tilex = tilex + leftb + rightb;
		const int dest_tiley = tiley + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_64F);

		const int top_tilex = tilex * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int top_tiley = tiley * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley + top_tiley - src.rows - topb);

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const double* s = src.ptr<double>(tiley * idx.y - stop) + top_tilex;
		double* d = dest.ptr<double>(top);

		const int LOAD_OFFSET1 = left - 4 - top_tilex;
		const int LOAD_OFFSET2 = src.cols - 4 - top_tilex;
		const int LOAD_OFFSET3 = src.cols - 1 - top_tilex;
		const int STORE_OFFSET = tilex + sright;

		__m256d a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < right; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < RIGHT; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			double* s = dest.ptr<double>(max(0, 2 * top - j - 1));
			double* d = dest.ptr<double>(j);

			memcpy(d, s, sizeof(double) * (dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx - j));
			double* d = dest.ptr<double>(didx + j);

			memcpy(d, s, sizeof(double) * dest_tilex);
		}
	}

	void cropTile(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType)
	{
		CV_Assert(borderType == BORDER_REFLECT101 || borderType == BORDER_REFLECT || borderType == BORDER_REPLICATE);

		CV_Assert(src.channels() == 1);
		
		if (!dest.empty() && dest.depth() != src.depth())dest.release();

		if (src.depth() == CV_8U)
		{
			if (borderType == BORDER_REFLECT101) cropTile8U_Reflect101(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) cropTile8U_Reflect(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) cropTile8U_Replicate(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
		}
		else if (src.depth() == CV_32F)
		{
			if (borderType == BORDER_REFLECT101) cropTile32F_Reflect101(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) cropTile32F_Reflect(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) cropTile32F_Replicate(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
		}
		else if (src.depth() == CV_64F)
		{
			if (borderType == BORDER_REFLECT101) cropTile64F_Reflect101(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) cropTile64F_Reflect(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) cropTile64F_Replicate(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
		}
		else
		{
			cout << "cropTile does not support this depth type." << endl;
		}
	}

	void cropTile(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int r, const int borderType)
	{
		cropTile(src, dest, div_size, idx, r, r, r, r, borderType);
	}

	void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;

		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);

		const int align_width = get_simd_ceil(tilex + L + r, align_x);
		const int padx = align_width - (tilex + L + r);
		const int align_height = get_simd_ceil(tiley + T + r, align_y);
		const int pady = align_height - (tiley + T + r);
		const int R = r + padx;
		const int B = r + pady;

		//printf("xpad%d,ypad%d\n",padx,pady);
		cropTile(src, dest, div_size, idx, T, B, L, R, borderType);
		//cout << format("%d %d %d %d\n", L, R, T, B);
	}
#pragma endregion

#pragma region cropTile roi
	static void cropTile8U_Replicate(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int dest_tile_w = roi_w + leftb + rightb;
		const int dest_tile_h = roi_h + topb + bottomb;
		if (dest.size() != Size(dest_tile_w, dest_tile_h)) dest.create(Size(dest_tile_w, dest_tile_h), CV_8U);

		const int roi_x = roi.x;
		const int left = max(0, leftb - roi_x);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_w + roi_x - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int memcpySizeX = dest_tile_w - left - right;

		const int roi_y = roi.y;
		const int top = max(0, topb - roi_y);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_h + roi_y - src.rows - topb);
		const int memcpySizeY = dest_tile_h - top - bottom;

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_floor(right, 16);

		const uchar* s = src.ptr<uchar>(roi_y - stop, roi_x);
		uchar* d = dest.ptr<uchar>(top);

		const int STORE_OFFSET = roi_w + sright;

		__m128i a;
		for (int j = 0; j < memcpySizeY - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_set1_epi8(s[-roi_x]);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (memcpySizeX));
			for (int i = 0; i < right; i += 16)
			{
				a = _mm_set1_epi8(s[-roi_x + src.cols - 1]);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_set1_epi8(s[-roi_x]);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (memcpySizeX));
			for (int i = 0; i < RIGHT; i += 16)
			{
				a = _mm_set1_epi8(s[-roi_x + src.cols - 1]);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[roi_w - 1];
			}

			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			uchar* s = dest.ptr<uchar>(top);
			uchar* d = dest.ptr<uchar>(j);
			memcpy(d, s, sizeof(uchar) * (dest_tile_w));
		}

		const int sidx = roi_h + topb - 1;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar) * dest_tile_w);
		}
	}

	static void cropTile32F_Replicate(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int dest_tile_w = roi_w + leftb + rightb;
		const int dest_tile_h = roi_h + topb + bottomb;
		if (dest.size() != Size(dest_tile_w, dest_tile_h)) dest.create(Size(dest_tile_w, dest_tile_h), CV_32F);

		const int roi_x = roi.x;
		const int left = max(0, leftb - roi_x);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_w + roi_x - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int memcpySizeX = dest_tile_w - left - right;

		const int roi_y = roi.y;
		const int top = max(0, topb - roi_y);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_h - (src.rows - roi_y + topb));
		const int memcpySizeY = dest_tile_h - top - bottom;

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const float* s = src.ptr<float>(roi_y - stop, roi_x);
		float* d = dest.ptr<float>(top);

		const int STORE_OFFSET = roi_w + sright;

		__m256 a;
		for (int j = 0; j < memcpySizeY - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_set1_ps(s[-roi_x]);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (memcpySizeX));
			for (int i = 0; i < right; i += 8)
			{
				a = _mm256_set1_ps(s[-roi_x + src.cols - 1]);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_set1_ps(s[0]);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (memcpySizeX));
			for (int i = 0; i < RIGHT; i += 8)
			{
				a = _mm256_set1_ps(s[roi_w - 1]);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[roi_w - 1];
			}

			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			float* s = dest.ptr<float>(top);
			float* d = dest.ptr<float>(j);
			memcpy(d, s, sizeof(float) * (dest_tile_w));
		}

		const int sidx = roi_h + topb - 1;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx));
			float* d = dest.ptr<float>(didx + j);
			memcpy(d, s, sizeof(float) * dest_tile_w);
		}
	}

	static void cropTile64F_Replicate(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int dest_tilex = roi_w + leftb + rightb;
		const int dest_tiley = roi_h + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_64F);

		const int roi_x = roi.x;
		const int left = max(0, leftb - roi_x);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + roi_x - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int memcpySizeX = dest_tilex - left - right;

		const int roi_y = roi.y;
		const int top = max(0, topb - roi_y);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley - (src.rows - roi_y + topb));
		const int memcpySizeY = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const double* s = src.ptr<double>(roi_y - stop, roi_x);
		double* d = dest.ptr<double>(top);

		const int STORE_OFFSET = roi_w + sright;

		__m256d a;
		for (int j = 0; j < memcpySizeY - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_set1_pd(s[-roi_x]);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (memcpySizeX));
			for (int i = 0; i < right; i += 4)
			{
				a = _mm256_set1_pd(s[-roi_x + src.cols - 1]);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_set1_pd(s[0]);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (memcpySizeX));
			for (int i = 0; i < RIGHT; i += 4)
			{
				a = _mm256_set1_pd(s[roi_w - 1]);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[roi_w - 1];
			}

			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			double* s = dest.ptr<double>(top);
			double* d = dest.ptr<double>(j);
			memcpy(d, s, sizeof(double) * (dest_tilex));
		}

		const int sidx = roi_h + topb - 1;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx));
			double* d = dest.ptr<double>(didx + j);
			memcpy(d, s, sizeof(double) * dest_tilex);
		}
	}


	static void cropTile8U_Reflect101(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int tileSizeXExternal = roi_w + leftb + rightb;
		const int tileSizeYExternal = roi_h + topb + bottomb;

		if (dest.size() != Size(tileSizeXExternal, tileSizeYExternal)) dest.create(Size(tileSizeXExternal, tileSizeYExternal), CV_8U);

		const int firstXInternalGrid = roi.x;
		const int left = max(0, leftb - firstXInternalGrid);
		const int sleft = leftb - left;
		const int right = max(0, tileSizeXExternal - (src.cols - firstXInternalGrid + leftb));
		const int sright = leftb + rightb - right;
		const int copysizex = tileSizeXExternal - left - right;

		const int firstYInternalGrid = roi.y;
		const int top = max(0, topb - firstYInternalGrid);
		const int stop = topb - top;
		const int bottom = max(0, tileSizeYExternal - (src.rows - firstYInternalGrid + topb));

		const int copysizey = tileSizeYExternal - top - bottom;

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_floor(right, 16);

		const uchar* s = src.ptr<uchar>(firstYInternalGrid - stop, firstXInternalGrid);
		uchar* d = dest.ptr<uchar>(top);

		const int LOAD_OFFSET1 = left - 16 - firstXInternalGrid;
		const int LOAD_OFFSET2 = src.cols - 16 - firstXInternalGrid;
		const int LOAD_OFFSET3 = src.cols - 1 - firstXInternalGrid;
		const int STORE_OFFSET = roi_w + sright;

		const __m128i vm = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		__m128i a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i + 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < right; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i - 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i + 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < RIGHT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i - 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i - 1];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			uchar* s = dest.ptr<uchar>(2 * top - j);
			uchar* d = dest.ptr<uchar>(j);
			memcpy(d, s, sizeof(uchar) * (tileSizeXExternal));
		}

		const int sidx = roi_h + topb - 2;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx - j));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar) * tileSizeXExternal);
		}
	}

	static void cropTile32F_Reflect101(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int dest_tilex = roi_w + leftb + rightb;
		const int dest_tiley = roi_h + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_32F);

		const int roi_x = roi.x;
		const int left = max(0, leftb - roi_x);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + roi_x - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int roi_y = roi.y;
		const int top = max(0, topb - roi_y);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley - (src.rows - roi_y + topb));

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const float* s = src.ptr<float>(roi_y - stop) + roi_x;
		float* d = dest.ptr<float>(top);

		const int LOAD_OFFSET1 = left - 8 - roi_x;
		const int LOAD_OFFSET2 = src.cols - 8 - roi_x;
		const int LOAD_OFFSET3 = src.cols - 1 - roi_x;
		const int STORE_OFFSET = roi_w + sright;

		__m256 a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_loadu_reverse_ps(s + LOAD_OFFSET1 - i + 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < right; i += 8)
			{
				a = _mm256_loadu_ps(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_loadu_ps(s + LOAD_OFFSET1 - i + 1);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < RIGHT; i += 8)
			{
				a = _mm256_loadu_ps(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i - 1];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			float* s = dest.ptr<float>(2 * top - j);
			float* d = dest.ptr<float>(j);
			memcpy(d, s, sizeof(float) * (dest_tilex));
		}

		const int sidx = roi_h + topb - 2;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx - j));
			float* d = dest.ptr<float>(didx + j);
			memcpy(d, s, sizeof(float) * dest_tilex);
		}
	}

	static void cropTile64F_Reflect101(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int dest_tilex = roi_w + leftb + rightb;
		const int dest_tiley = roi_h + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_64F);

		const int roi_x = roi.x;
		const int left = max(0, leftb - roi_x);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + roi_x - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int roi_y = roi.y;
		const int top = max(0, topb - roi_y);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley - (src.rows - roi_y + topb));

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const double* s = src.ptr<double>(roi_y - stop) + roi_x;
		double* d = dest.ptr<double>(top);

		const int LOAD_OFFSET1 = left - 4 - roi_x;
		const int LOAD_OFFSET2 = src.cols - 4 - roi_x;
		const int LOAD_OFFSET3 = src.cols - 1 - roi_x;
		const int STORE_OFFSET = roi_w + sright;

		__m256d a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET1 - i + 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < right; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET1 - i + 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_store_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < RIGHT; i += 4)
			{
				a = _mm256_loadu_pd(s + LOAD_OFFSET2 - i - 1);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i - 1];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			double* s = dest.ptr<double>(2 * top - j);
			double* d = dest.ptr<double>(j);

			memcpy(d, s, sizeof(double) * (dest_tilex));
		}

		const int sidx = roi_h + topb - 2;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx - j));
			double* d = dest.ptr<double>(didx + j);

			memcpy(d, s, sizeof(double) * dest_tilex);
		}
	}


	static void cropTile8U_Reflect(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int tileSizeXExternal = roi_w + leftb + rightb;
		const int tileSizeYExternal = roi_h + topb + bottomb;

		if (dest.size() != Size(tileSizeXExternal, tileSizeYExternal)) dest.create(Size(tileSizeXExternal, tileSizeYExternal), CV_8U);

		const int firstXInternalGrid = roi.x;
		const int left = max(0, leftb - firstXInternalGrid);
		const int sleft = leftb - left;
		const int right = max(0, tileSizeXExternal - (src.cols - firstXInternalGrid + leftb));
		const int sright = leftb + rightb - right;
		const int copysizex = tileSizeXExternal - left - right;

		const int firstYInternalGrid = roi.y;
		const int top = max(0, topb - firstYInternalGrid);
		const int stop = topb - top;
		const int bottom = max(0, tileSizeYExternal - (src.rows - firstYInternalGrid + topb));

		const int copysizey = tileSizeYExternal - top - bottom;

		const int LEFT = get_simd_ceil(left, 16);
		const int RIGHT = get_simd_floor(right, 16);

		const uchar* s = src.ptr<uchar>(firstYInternalGrid - stop, firstXInternalGrid);
		uchar* d = dest.ptr<uchar>(top);

		const int LOAD_OFFSET1 = left - 16 - firstXInternalGrid;
		const int LOAD_OFFSET2 = src.cols - 16 - firstXInternalGrid;
		const int LOAD_OFFSET3 = src.cols - 1 - firstXInternalGrid;
		const int STORE_OFFSET = roi_w + sright;

		const __m128i vm = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		__m128i a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < right; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar) * (copysizex));
			for (int i = 0; i < RIGHT; i += 16)
			{
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET2 - i));
				a = _mm_shuffle_epi8(a, vm);
				_mm_storeu_si128((__m128i*)(d + STORE_OFFSET + i), a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, 2 * top - j - 1));
			uchar* d = dest.ptr<uchar>(j);
			memcpy(d, s, sizeof(uchar) * (tileSizeXExternal));
		}

		const int sidx = roi_h + topb - 1;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx - j));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar) * tileSizeXExternal);
		}
	}

	static void cropTile32F_Reflect(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int dest_tilex = roi_w + leftb + rightb;
		const int dest_tiley = roi_h + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_32F);

		const int roi_x = roi.x;
		const int left = max(0, leftb - roi_x);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + roi_x - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int roi_y = roi.y;
		const int top = max(0, topb - roi_y);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley + roi_y - src.rows - topb);

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const float* s = src.ptr<float>(roi_y - stop, roi_x);
		float* d = dest.ptr<float>(top);

		const int LOAD_OFFSET1 = left - 8 - roi_x;
		const int LOAD_OFFSET2 = src.cols - 8 - roi_x;
		const int LOAD_OFFSET3 = src.cols - 1 - roi_x;
		const int STORE_OFFSET = roi_w + sright;

		__m256 a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < right; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float) * (copysizex));
			for (int i = 0; i < RIGHT; i += 8)
			{
				a = _mm256_load_ps(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_storeu_ps(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			float* s = dest.ptr<float>(max(0, 2 * top - j - 1));
			float* d = dest.ptr<float>(j);
			memcpy(d, s, sizeof(float) * (dest_tilex));
		}

		const int sidx = roi_h + topb - 1;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx - j));
			float* d = dest.ptr<float>(didx + j);

			memcpy(d, s, sizeof(float) * dest_tilex);
		}
	}

	static void cropTile64F_Reflect(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int roi_w = roi.width;
		const int roi_h = roi.height;
		const int dest_tilex = roi_w + leftb + rightb;
		const int dest_tiley = roi_h + topb + bottomb;
		if (dest.size() != Size(dest_tilex, dest_tiley)) dest.create(Size(dest_tilex, dest_tiley), CV_64F);

		const int roi_x = roi.x;
		const int left = max(0, leftb - roi_x);
		const int sleft = leftb - left;
		const int right = max(0, dest_tilex + roi_x - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tilex - left - right;

		const int roi_y = roi.y;
		const int top = max(0, topb - roi_y);
		const int stop = topb - top;
		const int bottom = max(0, dest_tiley + roi_y - src.rows - topb);

		const int copysizey = dest_tiley - top - bottom;

		const int LEFT = get_simd_ceil(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const double* s = src.ptr<double>(roi_y - stop) + roi_x;
		double* d = dest.ptr<double>(top);

		const int LOAD_OFFSET1 = left - 4 - roi_x;
		const int LOAD_OFFSET2 = src.cols - 4 - roi_x;
		const int LOAD_OFFSET3 = src.cols - 1 - roi_x;
		const int STORE_OFFSET = roi_w + sright;

		__m256d a;
		for (int j = 0; j < copysizey - 1; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < right; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			s += src.cols;
			d += dest.cols;
		}
		//overshoot handling for the last loop
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET1 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(double) * (copysizex));
			for (int i = 0; i < RIGHT; i += 4)
			{
				a = _mm256_load_pd(s + LOAD_OFFSET2 - i);
				a = _mm256_shuffle_pd(a, a, 0b0101);
				a = _mm256_permute2f128_pd(a, a, 1);
				_mm256_storeu_pd(d + STORE_OFFSET + i, a);
			}
			for (int i = RIGHT; i < right; i++)
			{
				d[STORE_OFFSET + i] = s[LOAD_OFFSET3 - i];
			}
			s += src.cols;
			d += dest.cols;
		}

		for (int j = 0; j < top; j++)
		{
			double* s = dest.ptr<double>(max(0, 2 * top - j - 1));
			double* d = dest.ptr<double>(j);

			memcpy(d, s, sizeof(double) * (dest_tilex));
		}

		const int sidx = roi_h + topb - 1;
		const int didx = roi_h + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx - j));
			double* d = dest.ptr<double>(didx + j);

			memcpy(d, s, sizeof(double) * dest_tilex);
		}
	}


	void cropTile(const Mat& src, Mat& dest, const Rect roi, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType)
	{
		CV_Assert(src.channels() == 1);
		CV_Assert(borderType == BORDER_REFLECT101 || borderType == BORDER_REFLECT || borderType == BORDER_REPLICATE);
		if (!dest.empty() && dest.depth() != src.depth())dest.release();

		if (src.depth() == CV_8U)
		{
			if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT) cropTile8U_Reflect101(src, dest, roi, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) cropTile8U_Reflect(src, dest, roi, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) cropTile8U_Replicate(src, dest, roi, topb, bottomb, leftb, rightb);
		}
		else if (src.depth() == CV_32F)
		{
			if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT) cropTile32F_Reflect101(src, dest, roi, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) cropTile32F_Reflect(src, dest, roi, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) cropTile32F_Replicate(src, dest, roi, topb, bottomb, leftb, rightb);
		}
		else if (src.depth() == CV_64F)
		{
			/*if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT) cropTile64F_Reflect101(src, dest, roi, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) cropTile64F_Reflect(src, dest, roi, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) cropTile64F_Replicate(src, dest, roi, topb, bottomb, leftb, rightb);*/
		}
		else
		{
			cout << "cropTile does not support this depth type." << endl;
		}
	}

	void cropTile(const Mat& src, Mat& dest, const Rect roi, const int r, const int borderType)
	{
		cropTile(src, dest, roi, r, r, r, r, borderType);
	}

	void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const Rect roi, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
	{
		const int tilex = roi.width;
		const int tiley = roi.height;

		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);

		const int align_width = get_simd_ceil(tilex + L + r, align_x);
		const int padx = align_width - (tilex + L + r);
		const int align_height = get_simd_ceil(tiley + T + r, align_y);
		const int pady = align_height - (tiley + T + r);
		const int R = r + padx;
		const int B = r + pady;
		//printf("xpad%d,ypad%d\n",padx,pady);
		cropTile(src, dest, roi, T, B, L, R, borderType);
		//cout << format("%d %d %d %d\n", L, R, T, B);
	}
#pragma endregion

	Size getTileSize(const Size src, const Size div_size, const int r)
	{
		const int tilex = src.width / div_size.width;
		const int tiley = src.height / div_size.height;
		return Size(tilex + 2 * r, tiley + 2 * r);
	}

	Size getTileAlignSize(const Size src, const Size div_size, const int r, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
	{
		const int tilex = src.width / div_size.width;
		const int tiley = src.height / div_size.height;

		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);

		const int align_width = get_simd_ceil(tilex + L + r, align_x);
		const int padx = align_width - (tilex + L + r);
		const int align_height = get_simd_ceil(tiley + T + r, align_y);
		const int pady = align_height - (tiley + T + r);
		const int R = r + padx;
		const int B = r + pady;

		return Size(tilex + L + R, tiley + T + B);
	}
#pragma endregion

#pragma region cropSplitTile

	static void cropSplitTile8UC3_Replicate(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_8U);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 8);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const int STORE_OFFSET = tile_size_x + sright;
		const uchar* s = src.ptr<uchar>(tile_size_y * idx.y - stop, top_tilex);
		uchar* db = dest[0].ptr<uchar>(top);
		uchar* dg = dest[1].ptr<uchar>(top);
		uchar* dr = dest[2].ptr<uchar>(top);

		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				_mm_storel_epi64((__m128i*)(db + i), _mm_set1_epi8(s[-top_tilex * 3 + 0]));
				_mm_storel_epi64((__m128i*)(dg + i), _mm_set1_epi8(s[-top_tilex * 3 + 1]));
				_mm_storel_epi64((__m128i*)(dr + i), _mm_set1_epi8(s[-top_tilex * 3 + 2]));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[-top_tilex * 3 + 0];
				dg[i] = s[-top_tilex * 3 + 1];
				dr[i] = s[-top_tilex * 3 + 2];
			}

			__m128i mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 8)
			{
				_mm_load_cvtepu8bgr2planar_epi64(s + 3 * (i - sleft), mb, mg, mr);
				_mm_storel_epi64((__m128i*)(db + i + left), mb);
				_mm_storel_epi64((__m128i*)(dg + i + left), mg);
				_mm_storel_epi64((__m128i*)(dr + i + left), mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 8)
			{
				_mm_storel_epi64((__m128i*)(db + STORE_OFFSET + i), _mm_set1_epi8(s[3 * (-top_tilex + src.cols - 1) + 0]));
				_mm_storel_epi64((__m128i*)(dg + STORE_OFFSET + i), _mm_set1_epi8(s[3 * (-top_tilex + src.cols - 1) + 1]));
				_mm_storel_epi64((__m128i*)(dr + STORE_OFFSET + i), _mm_set1_epi8(s[3 * (-top_tilex + src.cols - 1) + 2]));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 0];
				dg[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 1];
				dr[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 2];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				uchar* s = dest[c].ptr<uchar>(top);
				uchar* d = dest[c].ptr<uchar>(j);
				memcpy(d, s, sizeof(uchar) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 1;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				uchar* s = dest[c].ptr<uchar>(max(0, sidx));
				uchar* d = dest[c].ptr<uchar>(didx + j);
				memcpy(d, s, sizeof(uchar) * dest_tile_size_x);
			}
		}
	}

	static void cropSplitTile32FC3_Replicate(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_32F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 8);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const int STORE_OFFSET = tile_size_x + sright;
		const float* s = src.ptr<float>(tile_size_y * idx.y - stop, top_tilex);
		float* db = dest[0].ptr<float>(top);
		float* dg = dest[1].ptr<float>(top);
		float* dr = dest[2].ptr<float>(top);

		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				_mm256_storeu_ps(db + i, _mm256_set1_ps(s[-top_tilex * 3 + 0]));
				_mm256_storeu_ps(dg + i, _mm256_set1_ps(s[-top_tilex * 3 + 1]));
				_mm256_storeu_ps(dr + i, _mm256_set1_ps(s[-top_tilex * 3 + 2]));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[-top_tilex * 3 + 0];
				dg[i] = s[-top_tilex * 3 + 1];
				dr[i] = s[-top_tilex * 3 + 2];
			}

			__m256 mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 8)
			{
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * (i - sleft), mb, mg, mr);
				_mm256_storeu_ps(db + i + left, mb);
				_mm256_storeu_ps(dg + i + left, mg);
				_mm256_storeu_ps(dr + i + left, mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 8)
			{
				_mm256_storeu_ps(db + STORE_OFFSET + i, _mm256_set1_ps(s[3 * (-top_tilex + src.cols - 1) + 0]));
				_mm256_storeu_ps(dg + STORE_OFFSET + i, _mm256_set1_ps(s[3 * (-top_tilex + src.cols - 1) + 1]));
				_mm256_storeu_ps(dr + STORE_OFFSET + i, _mm256_set1_ps(s[3 * (-top_tilex + src.cols - 1) + 2]));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 0];
				dg[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 1];
				dr[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 2];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				float* s = dest[c].ptr<float>(top);
				float* d = dest[c].ptr<float>(j);
				memcpy(d, s, sizeof(float) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 1;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				float* s = dest[c].ptr<float>(max(0, sidx));
				float* d = dest[c].ptr<float>(didx + j);
				memcpy(d, s, sizeof(float) * dest_tile_size_x);
			}
		}
	}

	static void cropSplitTile64FC3_Replicate(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_64F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 4);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const int STORE_OFFSET = tile_size_x + sright;
		const double* s = src.ptr<double>(tile_size_y * idx.y - stop, top_tilex);
		double* db = dest[0].ptr<double>(top);
		double* dg = dest[1].ptr<double>(top);
		double* dr = dest[2].ptr<double>(top);

		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				_mm256_storeu_pd(db + i, _mm256_set1_pd(s[-top_tilex * 3 + 0]));
				_mm256_storeu_pd(dg + i, _mm256_set1_pd(s[-top_tilex * 3 + 1]));
				_mm256_storeu_pd(dr + i, _mm256_set1_pd(s[-top_tilex * 3 + 2]));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[-top_tilex * 3 + 0];
				dg[i] = s[-top_tilex * 3 + 1];
				dr[i] = s[-top_tilex * 3 + 2];
			}

			__m256d mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 4)
			{
				_mm256_load_cvtpd_bgr2planar_pd(s + 3 * (i - sleft), mb, mg, mr);
				_mm256_storeu_pd(db + i + left, mb);
				_mm256_storeu_pd(dg + i + left, mg);
				_mm256_storeu_pd(dr + i + left, mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 4)
			{
				_mm256_storeu_pd(db + STORE_OFFSET + i, _mm256_set1_pd(s[3 * (-top_tilex + src.cols - 1) + 0]));
				_mm256_storeu_pd(dg + STORE_OFFSET + i, _mm256_set1_pd(s[3 * (-top_tilex + src.cols - 1) + 1]));
				_mm256_storeu_pd(dr + STORE_OFFSET + i, _mm256_set1_pd(s[3 * (-top_tilex + src.cols - 1) + 2]));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 0];
				dg[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 1];
				dr[i + STORE_OFFSET] = s[3 * (-top_tilex + src.cols - 1) + 2];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				double* s = dest[c].ptr<double>(top);
				double* d = dest[c].ptr<double>(j);
				memcpy(d, s, sizeof(double) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 1;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				double* s = dest[c].ptr<double>(max(0, sidx));
				double* d = dest[c].ptr<double>(didx + j);
				memcpy(d, s, sizeof(double) * dest_tile_size_x);
			}
		}
	}


	static void cropSplitTile8UC3_Reflect101(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_32F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 8);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const int STORE_OFFSET = tile_size_x + sright;
		const uchar* s = src.ptr<uchar>(tile_size_y * idx.y - stop, top_tilex);
		uchar* db = dest[0].ptr <uchar>(top);
		uchar* dg = dest[1].ptr <uchar>(top);
		uchar* dr = dest[2].ptr <uchar>(top);

		/*print_debug4(tile_size_x, leftb, rightb, dest_tile_size_x);
		print_debug(top_tilex);
		print_debug2(left, LEFT);
		print_debug2(right, RIGHT);
		print_debug(sleft);
		print_debug2(copysizex, copysizey);*/

		const int LOAD_OFFSET1 = 3 * (left - 8 - top_tilex);
		const int LOAD_OFFSET2 = 3 * (src.cols - 1 - 8 - top_tilex);
		__m256i midx = _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0);
		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				_mm_storel_epi64((__m128i*)(db + i), _mm256_i32gather_epu8(s + LOAD_OFFSET1 - 3 * i + 3, midx));
				_mm_storel_epi64((__m128i*)(dg + i), _mm256_i32gather_epu8(s + LOAD_OFFSET1 - 3 * i + 4, midx));
				_mm_storel_epi64((__m128i*)(dr + i), _mm256_i32gather_epu8(s + LOAD_OFFSET1 - 3 * i + 5, midx));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[(-top_tilex + left - i) * 3 + 0];
				dg[i] = s[(-top_tilex + left - i) * 3 + 1];
				dr[i] = s[(-top_tilex + left - i) * 3 + 2];
			}

			__m128i mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 8)
			{
				_mm_load_cvtepu8bgr2planar_epi64(s + 3 * (i - sleft), mb, mg, mr);
				_mm_storel_epi64((__m128i*)(db + i + left), mb);
				_mm_storel_epi64((__m128i*)(dg + i + left), mg);
				_mm_storel_epi64((__m128i*)(dr + i + left), mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 8)
			{
				_mm_storel_epi64((__m128i*)(db + STORE_OFFSET + i), _mm256_i32gather_epu8(s + LOAD_OFFSET2 - 3 * i + 0, midx));
				_mm_storel_epi64((__m128i*)(dg + STORE_OFFSET + i), _mm256_i32gather_epu8(s + LOAD_OFFSET2 - 3 * i + 1, midx));
				_mm_storel_epi64((__m128i*)(dr + STORE_OFFSET + i), _mm256_i32gather_epu8(s + LOAD_OFFSET2 - 3 * i + 2, midx));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 0];
				dg[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 1];
				dr[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 2];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				uchar* s = dest[c].ptr<uchar>(2 * top - j);
				uchar* d = dest[c].ptr<uchar>(j);
				memcpy(d, s, sizeof(uchar) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 2;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				uchar* s = dest[c].ptr<uchar>(max(0, sidx - j));
				uchar* d = dest[c].ptr<uchar>(didx + j);
				memcpy(d, s, sizeof(uchar) * dest_tile_size_x);
			}
		}
	}

	static void cropSplitTile32FC3_Reflect101(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_32F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 8);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const int STORE_OFFSET = tile_size_x + sright;
		const float* s = src.ptr<float>(tile_size_y * idx.y - stop, top_tilex);
		float* db = dest[0].ptr<float>(top);
		float* dg = dest[1].ptr<float>(top);
		float* dr = dest[2].ptr<float>(top);

		/*print_debug4(tile_size_x, leftb, rightb, dest_tile_size_x);
		print_debug(top_tilex);
		print_debug2(left, LEFT);
		print_debug2(right, RIGHT);
		print_debug(sleft);
		print_debug2(copysizex, copysizey);*/

		const int LOAD_OFFSET1 = 3 * (left - 8 - top_tilex);
		const int LOAD_OFFSET2 = 3 * (src.cols - 1 - 8 - top_tilex);
		__m256i midx = _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0);
		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				_mm256_storeu_ps(db + i, _mm256_i32gather_ps(s + LOAD_OFFSET1 - 3 * i + 3, midx, sizeof(float)));
				_mm256_storeu_ps(dg + i, _mm256_i32gather_ps(s + LOAD_OFFSET1 - 3 * i + 4, midx, sizeof(float)));
				_mm256_storeu_ps(dr + i, _mm256_i32gather_ps(s + LOAD_OFFSET1 - 3 * i + 5, midx, sizeof(float)));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[(-top_tilex + left - i) * 3 + 0];
				dg[i] = s[(-top_tilex + left - i) * 3 + 1];
				dr[i] = s[(-top_tilex + left - i) * 3 + 2];
			}

			__m256 mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 8)
			{
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * (i - sleft), mb, mg, mr);
				_mm256_storeu_ps(db + i + left, mb);
				_mm256_storeu_ps(dg + i + left, mg);
				_mm256_storeu_ps(dr + i + left, mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 8)
			{
				_mm256_storeu_ps(db + STORE_OFFSET + i, _mm256_i32gather_ps(s + LOAD_OFFSET2 - 3 * i + 0, midx, sizeof(float)));
				_mm256_storeu_ps(dg + STORE_OFFSET + i, _mm256_i32gather_ps(s + LOAD_OFFSET2 - 3 * i + 1, midx, sizeof(float)));
				_mm256_storeu_ps(dr + STORE_OFFSET + i, _mm256_i32gather_ps(s + LOAD_OFFSET2 - 3 * i + 2, midx, sizeof(float)));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 0];
				dg[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 1];
				dr[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 2];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				float* s = dest[c].ptr<float>(2 * top - j);
				float* d = dest[c].ptr<float>(j);
				memcpy(d, s, sizeof(float) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 2;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				float* s = dest[c].ptr<float>(max(0, sidx - j));
				float* d = dest[c].ptr<float>(didx + j);
				memcpy(d, s, sizeof(float) * dest_tile_size_x);
			}
		}
	}

	static void cropSplitTile64FC3_Reflect101(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_64F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 4);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const int STORE_OFFSET = tile_size_x + sright;
		const double* s = src.ptr<double>(tile_size_y * idx.y - stop, top_tilex);
		double* db = dest[0].ptr<double>(top);
		double* dg = dest[1].ptr<double>(top);
		double* dr = dest[2].ptr<double>(top);

		/*print_debug4(tile_size_x, leftb, rightb, dest_tile_size_x);
		print_debug(top_tilex);
		print_debug2(left, LEFT);
		print_debug2(right, RIGHT);
		print_debug(sleft);
		print_debug2(copysizex, copysizey);*/

		const int LOAD_OFFSET1 = 3 * (left - 4 - top_tilex);
		const int LOAD_OFFSET2 = 3 * (src.cols - 1 - 4 - top_tilex);
		__m128i midx = _mm_setr_epi32(9, 6, 3, 0);
		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				_mm256_storeu_pd(db + i, _mm256_i32gather_pd(s + LOAD_OFFSET1 - 3 * i + 3, midx, sizeof(double)));
				_mm256_storeu_pd(dg + i, _mm256_i32gather_pd(s + LOAD_OFFSET1 - 3 * i + 4, midx, sizeof(double)));
				_mm256_storeu_pd(dr + i, _mm256_i32gather_pd(s + LOAD_OFFSET1 - 3 * i + 5, midx, sizeof(double)));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[(-top_tilex + left - i) * 3 + 0];
				dg[i] = s[(-top_tilex + left - i) * 3 + 1];
				dr[i] = s[(-top_tilex + left - i) * 3 + 2];
			}

			__m256d mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 4)
			{
				_mm256_load_cvtpd_bgr2planar_pd(s + 3 * (i - sleft), mb, mg, mr);
				_mm256_storeu_pd(db + i + left, mb);
				_mm256_storeu_pd(dg + i + left, mg);
				_mm256_storeu_pd(dr + i + left, mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 4)
			{
				_mm256_storeu_pd(db + STORE_OFFSET + i, _mm256_i32gather_pd(s + LOAD_OFFSET2 - 3 * i + 0, midx, sizeof(double)));
				_mm256_storeu_pd(dg + STORE_OFFSET + i, _mm256_i32gather_pd(s + LOAD_OFFSET2 - 3 * i + 1, midx, sizeof(double)));
				_mm256_storeu_pd(dr + STORE_OFFSET + i, _mm256_i32gather_pd(s + LOAD_OFFSET2 - 3 * i + 2, midx, sizeof(double)));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 3) + 0];
				dg[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 3) + 1];
				dr[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 3) + 2];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				double* s = dest[c].ptr<double>(2 * top - j);
				double* d = dest[c].ptr<double>(j);
				memcpy(d, s, sizeof(double) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 2;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				double* s = dest[c].ptr<double>(max(0, sidx - j));
				double* d = dest[c].ptr<double>(didx + j);
				memcpy(d, s, sizeof(double) * dest_tile_size_x);
			}
		}
	}


	static void cropSplitTile8UC3_Reflect(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_32F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 8);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const int STORE_OFFSET = tile_size_x + sright;
		const uchar* s = src.ptr<uchar>(tile_size_y * idx.y - stop, top_tilex);
		uchar* db = dest[0].ptr <uchar>(top);
		uchar* dg = dest[1].ptr <uchar>(top);
		uchar* dr = dest[2].ptr <uchar>(top);

		/*print_debug4(tile_size_x, leftb, rightb, dest_tile_size_x);
		print_debug(top_tilex);
		print_debug2(left, LEFT);
		print_debug2(right, RIGHT);
		print_debug(sleft);
		print_debug2(copysizex, copysizey);*/

		const int LOAD_OFFSET1 = 3 * (left - 8 - top_tilex);
		const int LOAD_OFFSET2 = 3 * (src.cols - 1 - 8 - top_tilex);
		__m256i midx = _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0);
		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				_mm_storel_epi64((__m128i*)(db + i), _mm256_i32gather_epu8(s + LOAD_OFFSET1 - 3 * i + 0, midx));
				_mm_storel_epi64((__m128i*)(dg + i), _mm256_i32gather_epu8(s + LOAD_OFFSET1 - 3 * i + 1, midx));
				_mm_storel_epi64((__m128i*)(dr + i), _mm256_i32gather_epu8(s + LOAD_OFFSET1 - 3 * i + 2, midx));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[(-top_tilex + left - i - 1) * 3 + 0];
				dg[i] = s[(-top_tilex + left - i - 1) * 3 + 1];
				dr[i] = s[(-top_tilex + left - i - 1) * 3 + 2];
			}

			__m128i mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 8)
			{
				_mm_load_cvtepu8bgr2planar_epi64(s + 3 * (i - sleft), mb, mg, mr);
				_mm_storel_epi64((__m128i*)(db + i + left), mb);
				_mm_storel_epi64((__m128i*)(dg + i + left), mg);
				_mm_storel_epi64((__m128i*)(dr + i + left), mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 8)
			{
				_mm_storel_epi64((__m128i*)(db + STORE_OFFSET + i), _mm256_i32gather_epu8(s + LOAD_OFFSET2 - 3 * i + 3, midx));
				_mm_storel_epi64((__m128i*)(dg + STORE_OFFSET + i), _mm256_i32gather_epu8(s + LOAD_OFFSET2 - 3 * i + 4, midx));
				_mm_storel_epi64((__m128i*)(dr + STORE_OFFSET + i), _mm256_i32gather_epu8(s + LOAD_OFFSET2 - 3 * i + 5, midx));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 3];
				dg[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 4];
				dr[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 5];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				uchar* s = dest[c].ptr<uchar>(2 * top - j - 1);
				uchar* d = dest[c].ptr<uchar>(j);
				memcpy(d, s, sizeof(uchar) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 1;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				uchar* s = dest[c].ptr<uchar>(max(0, sidx - j));
				uchar* d = dest[c].ptr<uchar>(didx + j);
				memcpy(d, s, sizeof(uchar) * dest_tile_size_x);
			}
		}
	}

	static void cropSplitTile32FC3_Reflect(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_32F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 8);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 8);
		const int RIGHT = get_simd_floor(right, 8);

		const int STORE_OFFSET = tile_size_x + sright;
		const float* s = src.ptr<float>(tile_size_y * idx.y - stop, top_tilex);
		float* db = dest[0].ptr<float>(top);
		float* dg = dest[1].ptr<float>(top);
		float* dr = dest[2].ptr<float>(top);

		/*print_debug4(tile_size_x, leftb, rightb, dest_tile_size_x);
		print_debug(top_tilex);
		print_debug2(left, LEFT);
		print_debug2(right, RIGHT);
		print_debug(sleft);
		print_debug2(copysizex, copysizey);*/

		const int LOAD_OFFSET1 = 3 * (left - 8 - top_tilex);
		const int LOAD_OFFSET2 = 3 * (src.cols - 1 - 8 - top_tilex);
		__m256i midx = _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0);
		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 8)
			{
				_mm256_storeu_ps(db + i, _mm256_i32gather_ps(s + LOAD_OFFSET1 - 3 * i + 0, midx, sizeof(float)));
				_mm256_storeu_ps(dg + i, _mm256_i32gather_ps(s + LOAD_OFFSET1 - 3 * i + 1, midx, sizeof(float)));
				_mm256_storeu_ps(dr + i, _mm256_i32gather_ps(s + LOAD_OFFSET1 - 3 * i + 2, midx, sizeof(float)));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[(-top_tilex + left - i - 1) * 3 + 0];
				dg[i] = s[(-top_tilex + left - i - 1) * 3 + 1];
				dr[i] = s[(-top_tilex + left - i - 1) * 3 + 2];
			}

			__m256 mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 8)
			{
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * (i - sleft), mb, mg, mr);
				_mm256_storeu_ps(db + i + left, mb);
				_mm256_storeu_ps(dg + i + left, mg);
				_mm256_storeu_ps(dr + i + left, mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 8)
			{
				_mm256_storeu_ps(db + STORE_OFFSET + i, _mm256_i32gather_ps(s + LOAD_OFFSET2 - 3 * i + 3, midx, sizeof(float)));
				_mm256_storeu_ps(dg + STORE_OFFSET + i, _mm256_i32gather_ps(s + LOAD_OFFSET2 - 3 * i + 4, midx, sizeof(float)));
				_mm256_storeu_ps(dr + STORE_OFFSET + i, _mm256_i32gather_ps(s + LOAD_OFFSET2 - 3 * i + 5, midx, sizeof(float)));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 3];
				dg[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 4];
				dr[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 7) + 5];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				float* s = dest[c].ptr<float>(2 * top - j - 1);
				float* d = dest[c].ptr<float>(j);
				memcpy(d, s, sizeof(float) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 1;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				float* s = dest[c].ptr<float>(max(0, sidx - j));
				float* d = dest[c].ptr<float>(didx + j);
				memcpy(d, s, sizeof(float) * dest_tile_size_x);
			}
		}
	}

	static void cropSplitTile64FC3_Reflect(const Mat& src, vector<Mat>& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
	{
		const int tile_size_x = src.cols / div_size.width;
		const int tile_size_y = src.rows / div_size.height;
		const int dest_tile_size_x = tile_size_x + leftb + rightb;
		const int dest_tile_size_y = tile_size_y + topb + bottomb;
		if (dest.size() != 3)dest.resize(3);
		for (int c = 0; c < 3; c++)
		{
			if (dest[c].size() != Size(dest_tile_size_x, dest_tile_size_y)) dest[c].create(Size(dest_tile_size_x, dest_tile_size_y), CV_64F);
		}

		const int top_tilex = tile_size_x * idx.x;
		const int left = max(0, leftb - top_tilex);
		const int sleft = leftb - left;
		const int right = max(0, dest_tile_size_x + top_tilex - src.cols - leftb);
		const int sright = leftb + rightb - right;
		const int copysizex = dest_tile_size_x - left - right;
		const int COPYSIZEX = get_simd_floor(copysizex, 4);

		const int top_tiley = tile_size_y * idx.y;
		const int top = max(0, topb - top_tiley);
		const int stop = topb - top;
		const int bottom = max(0, dest_tile_size_y - (src.rows - top_tiley + topb));

		const int copysizey = dest_tile_size_y - top - bottom;

		const int LEFT = get_simd_floor(left, 4);
		const int RIGHT = get_simd_floor(right, 4);

		const int STORE_OFFSET = tile_size_x + sright;
		const double* s = src.ptr<double>(tile_size_y * idx.y - stop, top_tilex);
		double* db = dest[0].ptr<double>(top);
		double* dg = dest[1].ptr<double>(top);
		double* dr = dest[2].ptr<double>(top);

		/*print_debug4(tile_size_x, leftb, rightb, dest_tile_size_x);
		print_debug(top_tilex);
		print_debug2(left, LEFT);
		print_debug2(right, RIGHT);
		print_debug(sleft);
		print_debug2(copysizex, copysizey);*/

		const int LOAD_OFFSET1 = 3 * (left - 4 - top_tilex);
		const int LOAD_OFFSET2 = 3 * (src.cols - 1 - 4 - top_tilex);
		__m128i midx = _mm_setr_epi32(9, 6, 3, 0);
		for (int j = 0; j < copysizey; j++)
		{
			for (int i = 0; i < LEFT; i += 4)
			{
				_mm256_storeu_pd(db + i, _mm256_i32gather_pd(s + LOAD_OFFSET1 - 3 * i + 0, midx, sizeof(double)));
				_mm256_storeu_pd(dg + i, _mm256_i32gather_pd(s + LOAD_OFFSET1 - 3 * i + 1, midx, sizeof(double)));
				_mm256_storeu_pd(dr + i, _mm256_i32gather_pd(s + LOAD_OFFSET1 - 3 * i + 2, midx, sizeof(double)));
			}
			for (int i = LEFT; i < left; i++)
			{
				db[i] = s[(-top_tilex + left - i - 1) * 3 + 0];
				dg[i] = s[(-top_tilex + left - i - 1) * 3 + 1];
				dr[i] = s[(-top_tilex + left - i - 1) * 3 + 2];
			}

			__m256d mb, mg, mr;
			for (int i = 0; i < COPYSIZEX; i += 4)
			{
				_mm256_load_cvtpd_bgr2planar_pd(s + 3 * (i - sleft), mb, mg, mr);
				_mm256_storeu_pd(db + i + left, mb);
				_mm256_storeu_pd(dg + i + left, mg);
				_mm256_storeu_pd(dr + i + left, mr);
			}
			for (int i = COPYSIZEX; i < copysizex; i++)
			{
				db[i + left] = s[3 * (i - sleft) + 0];
				dg[i + left] = s[3 * (i - sleft) + 1];
				dr[i + left] = s[3 * (i - sleft) + 2];
			}

			for (int i = 0; i < RIGHT; i += 4)
			{
				_mm256_storeu_pd(db + STORE_OFFSET + i, _mm256_i32gather_pd(s + LOAD_OFFSET2 - 3 * i + 3, midx, sizeof(double)));
				_mm256_storeu_pd(dg + STORE_OFFSET + i, _mm256_i32gather_pd(s + LOAD_OFFSET2 - 3 * i + 4, midx, sizeof(double)));
				_mm256_storeu_pd(dr + STORE_OFFSET + i, _mm256_i32gather_pd(s + LOAD_OFFSET2 - 3 * i + 5, midx, sizeof(double)));
			}
			for (int i = RIGHT; i < right; i++)
			{
				db[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 3) + 3];
				dg[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 3) + 4];
				dr[STORE_OFFSET + i] = s[LOAD_OFFSET2 - 3 * (i - 3) + 5];
			}

			s += 3 * src.cols;
			db += dest[0].cols;
			dg += dest[0].cols;
			dr += dest[0].cols;
		}

		for (int c = 0; c < 3; c++)
		{
			for (int j = 0; j < top; j++)
			{
				double* s = dest[c].ptr<double>(2 * top - j - 1);
				double* d = dest[c].ptr<double>(j);
				memcpy(d, s, sizeof(double) * (dest_tile_size_x));
			}

			const int sidx = tile_size_y * (div_size.height - idx.y) + topb - 1;
			const int didx = tile_size_y * (div_size.height - idx.y) + topb;
			for (int j = 0; j < bottom; j++)
			{
				double* s = dest[c].ptr<double>(max(0, sidx - j));
				double* d = dest[c].ptr<double>(didx + j);
				memcpy(d, s, sizeof(double) * dest_tile_size_x);
			}
		}
	}


	void cropSplitTile(const Mat& src, vector<Mat>& vdst, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType)
	{
		CV_Assert(borderType == BORDER_REFLECT101 || borderType == BORDER_REFLECT || borderType == BORDER_REPLICATE);

		if (src.channels() == 3)
		{
			if (src.depth() == CV_8U)
			{
				if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT)cropSplitTile8UC3_Reflect101(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
				else if (borderType == BORDER_REFLECT) cropSplitTile8UC3_Reflect(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
				else if (borderType == BORDER_REPLICATE) cropSplitTile8UC3_Replicate(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
			}
			else if (src.depth() == CV_32F)
			{
				if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT) cropSplitTile32FC3_Reflect101(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
				else if (borderType == BORDER_REFLECT) cropSplitTile32FC3_Reflect(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
				else if (borderType == BORDER_REPLICATE) cropSplitTile32FC3_Replicate(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
			}
			else if (src.depth() == CV_64F)
			{
				if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT)cropSplitTile64FC3_Reflect101(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
				else if (borderType == BORDER_REFLECT) cropSplitTile64FC3_Reflect(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
				else if (borderType == BORDER_REPLICATE) cropSplitTile64FC3_Replicate(src, vdst, div_size, idx, topb, bottomb, leftb, rightb);
			}
			else
			{
				cout << "cropSplitTile does not support this depth type." << endl;
			}
		}
		else
		{
			vector<Mat> vsrc;
			split(src, vsrc);
			vdst.resize(vsrc.size());
			for (int i = 0; i < vsrc.size(); i++)
			{
				cropTile(vsrc[i], vdst[i], div_size, idx, topb, bottomb, leftb, rightb, borderType);
			}
		}
	}

	void cropSplitTileAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
	{
		const int tilex = src.cols / div_size.width;
		const int tiley = src.rows / div_size.height;

		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);

		const int align_width = get_simd_ceil(tilex + L + r, align_x);
		const int padx = align_width - (tilex + L + r);
		const int align_height = get_simd_ceil(tiley + T + r, align_y);
		const int pady = align_height - (tiley + T + r);
		const int R = r + padx;
		const int B = r + pady;

		cropSplitTile(src, dest, div_size, idx, T, B, L, R, borderType);
	}

#pragma endregion

#pragma region pasteTile
	template<typename srcType>
	void pasteTile_internal(const Mat& src, Mat& dest, Rect roi, const int top, const int left)
	{
		const int tilex = roi.width;
		const int tiley = roi.height;

		int align = 0;
		if (typeid(srcType) == typeid(uchar))align = 8;
		if (typeid(srcType) == typeid(float))align = 8;
		if (typeid(srcType) == typeid(double))align = 4;
		const int simd_tile_width = get_simd_floor(tilex, align) * src.channels();
		const int rem = tilex * src.channels() - simd_tile_width;

		if (src.channels() == 1)
		{
			for (int j = 0; j < tiley; j++)
			{
				srcType* d = dest.ptr<srcType>(roi.y + j, roi.x);
				const srcType* s = src.ptr<srcType>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_auto(d + i, _mm256_loadu_auto(s + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[simd_tile_width + i] = s[simd_tile_width + i];
				}
			}
		}
		else if (src.channels() == 3)
		{
			for (int j = 0; j < tiley; j++)
			{
				srcType* d = dest.ptr<srcType>(roi.y + j, roi.x);
				const srcType* s = src.ptr<srcType>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_auto(d + i, _mm256_loadu_auto(s + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[simd_tile_width + i] = s[simd_tile_width + i];
				}
			}
		}
	}

	void pasteTile(const Mat& src, Mat& dest, const Rect roi, const int top, const int left)
	{
		CV_Assert(!dest.empty());
		CV_Assert(src.depth() == dest.depth());
		CV_Assert(src.channels() == dest.channels());
		//CV_Assert(dest.cols % div_size.width == 0 && dest.rows % div_size.height == 0);

		if (src.depth() == CV_8U)
		{
			pasteTile_internal<uchar>(src, dest, roi, top, left);
		}
		else if (src.depth() == CV_32F)
		{
			pasteTile_internal<float>(src, dest, roi, top, left);
		}
		else if (src.depth() == CV_64F)
		{
			pasteTile_internal<double>(src, dest, roi, top, left);
		}
	}

	void pasteTile(const Mat& src, Mat& dest, const Rect roi, const int r)
	{
		pasteTile(src, dest, roi, r, r);
	}

	void pasteTileAlign(const Mat& src, Mat& dest, Rect roi, const int r, const int left_multiple, const int top_multiple)
	{
		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);
		pasteTile(src, dest, roi, L, T);
	}

	template<typename srcType>
	void pasteTile_internal(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		const int tilex = dest.cols / div_size.width;
		const int tiley = dest.rows / div_size.height;

		int align = 0;
		if (typeid(srcType) == typeid(uchar))align = 8;
		if (typeid(srcType) == typeid(float))align = 8;
		if (typeid(srcType) == typeid(double))align = 4;
		const int simd_tile_width = get_simd_floor(tilex, align) * src.channels();
		const int rem = tilex * src.channels() - simd_tile_width;
		const int channels = src.channels();

		if (channels == 1)
		{
			for (int j = 0; j < tiley; j++)
			{
				srcType* d = dest.ptr<srcType>(tiley * idx.y + j, tilex * idx.x);
				const srcType* s = src.ptr<srcType>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_auto(d + i, _mm256_loadu_auto(s + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[simd_tile_width + i] = s[simd_tile_width + i];
				}
			}
		}
		else // n-channel case if (channels == 3)
		{
			for (int j = 0; j < tiley; j++)
			{
				srcType* d = dest.ptr<srcType>(tiley * idx.y + j, tilex * idx.x);
				const srcType* s = src.ptr<srcType>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_auto(d + i, _mm256_loadu_auto(s + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[simd_tile_width + i] = s[simd_tile_width + i];
				}
			}
		}
	}

	void pasteTile(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		CV_Assert(!dest.empty());
		CV_Assert(src.depth() == dest.depth());
		CV_Assert(src.channels() == dest.channels());
		//CV_Assert(dest.cols % div_size.width == 0 && dest.rows % div_size.height == 0);

		if (src.depth() == CV_8U)
		{
			pasteTile_internal<uchar>(src, dest, div_size, idx, top, left);
		}
		else if (src.depth() == CV_32F)
		{
			pasteTile_internal<float>(src, dest, div_size, idx, top, left);
		}
		else if (src.depth() == CV_64F)
		{
			pasteTile_internal<double>(src, dest, div_size, idx, top, left);
		}
	}

	void pasteTile(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int r)
	{
		pasteTile(src, dest, div_size, idx, r, r);
	}

	void pasteTileAlign(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int r, const int left_multiple, const int top_multiple)
	{
		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);
		pasteTile(src, dest, div_size, idx, L, T);
	}
#pragma endregion

#pragma region pastMergeTile
	template<typename srcType>
	void pasteMergeTile_internal(const vector<Mat>& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		const int channels = (int)src.size();
		const int tilex = dest.cols / div_size.width;
		const int tiley = dest.rows / div_size.height;

		int align = 0;
		if (typeid(srcType) == typeid(float))align = 8;
		if (typeid(srcType) == typeid(double))align = 4;
		const int simd_tile_width = get_simd_floor(tilex, align) * channels;
		const int rem = tilex * channels - simd_tile_width;

		if (channels == 1)
		{
			for (int j = 0; j < tiley; j++)
			{
				srcType* d = dest.ptr<srcType>(tiley * idx.y + j, tilex * idx.x);
				const srcType* s = src[0].ptr<srcType>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_auto(d + i, _mm256_loadu_auto(s + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[simd_tile_width + i] = s[simd_tile_width + i];
				}
			}
		}
		else if (channels == 3)
		{
			for (int j = 0; j < tiley; j++)
			{
				srcType* d = dest.ptr<srcType>(tiley * idx.y + j, tilex * idx.x);
				const srcType* s0 = src[0].ptr<srcType>(top + j, left);
				const srcType* s1 = src[1].ptr<srcType>(top + j, left);
				const srcType* s2 = src[2].ptr<srcType>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_store_auto_color(d + 3 * i, _mm256_loadu_auto(s0 + i), _mm256_loadu_auto(s1 + i), _mm256_loadu_auto(s2 + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[3 * (simd_tile_width + i) + 0] = s0[simd_tile_width + i];
					d[3 * (simd_tile_width + i) + 1] = s0[simd_tile_width + i];
					d[3 * (simd_tile_width + i) + 2] = s0[simd_tile_width + i];
				}
			}
		}
	}

	template<>
	void pasteMergeTile_internal<double>(const vector<Mat>& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		const int channels = (int)src.size();
		const int tilex = dest.cols / div_size.width;
		const int tiley = dest.rows / div_size.height;

		int align = 4;
		const int simd_tile_width = get_simd_floor(tilex, align);
		const int rem = tilex - simd_tile_width;

		if (channels == 1)
		{
			for (int j = 0; j < tiley; j++)
			{
				double* d = dest.ptr<double>(tiley * idx.y + j, tilex * idx.x);
				const double* s = src[0].ptr<double>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_pd(d + i, _mm256_loadu_pd(s + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[simd_tile_width + i] = s[simd_tile_width + i];
				}
			}
		}
		else if (channels == 3)
		{
			for (int j = 0; j < tiley; j++)
			{
				double* d = dest.ptr<double>(tiley * idx.y + j, tilex * idx.x);
				const double* s0 = src[0].ptr<double>(top + j, left);
				const double* s1 = src[1].ptr<double>(top + j, left);
				const double* s2 = src[2].ptr<double>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_store_interleave_pd(d + 3 * i, _mm256_loadu_pd(s0 + i), _mm256_loadu_pd(s1 + i), _mm256_loadu_pd(s2 + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[3 * (simd_tile_width + i) + 0] = s0[simd_tile_width + i];
					d[3 * (simd_tile_width + i) + 1] = s0[simd_tile_width + i];
					d[3 * (simd_tile_width + i) + 2] = s0[simd_tile_width + i];
				}
			}
		}
	}

	template<>
	void pasteMergeTile_internal<float>(const vector<Mat>& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		const int channels = (int)src.size();
		const int tilex = dest.cols / div_size.width;
		const int tiley = dest.rows / div_size.height;

		int align = 8;
		const int simd_tile_width = get_simd_floor(tilex, align);
		const int rem = tilex - simd_tile_width;

		if (channels == 1)
		{
			for (int j = 0; j < tiley; j++)
			{
				float* d = dest.ptr<float>(tiley * idx.y + j, tilex * idx.x);
				const float* s = src[0].ptr<float>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_ps(d + i, _mm256_loadu_ps(s + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[simd_tile_width + i] = s[simd_tile_width + i];
				}
			}
		}
		else if (channels == 3)
		{
			for (int j = 0; j < tiley; j++)
			{
				float* d = dest.ptr<float>(tiley * idx.y + j, tilex * idx.x);
				const float* s0 = src[0].ptr<float>(top + j, left);
				const float* s1 = src[1].ptr<float>(top + j, left);
				const float* s2 = src[2].ptr<float>(top + j, left);
				for (int i = 0; i < simd_tile_width; i += align)
				{
					_mm256_storeu_interleave_ps(d + 3 * i, _mm256_loadu_ps(s0 + i), _mm256_loadu_ps(s1 + i), _mm256_loadu_ps(s2 + i));
				}
				for (int i = 0; i < rem; i++)
				{
					d[3 * (simd_tile_width + i) + 0] = s0[simd_tile_width + i];
					d[3 * (simd_tile_width + i) + 1] = s0[simd_tile_width + i];
					d[3 * (simd_tile_width + i) + 2] = s0[simd_tile_width + i];
				}
			}
		}
	}

	template<>
	void pasteMergeTile_internal<uchar>(const vector<Mat>& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		const int channels = (int)src.size();
		const int tilex = dest.cols / div_size.width;
		const int tiley = dest.rows / div_size.height;

		const int align0 = 32;
		const int align1 = 16;
		const int align2 = 8;
		const int simd_tile_width0 = get_simd_floor(tilex, align0);
		const int rem0 = tilex - simd_tile_width0;
		const int simd_tile_width1 = get_simd_floor(rem0, align1);
		const int rem1 = align1 - simd_tile_width1;
		const int simd_tile_width2 = get_simd_floor(rem1, align2);
		const int rem2 = align2 - simd_tile_width2;

		if (channels == 1)
		{
			for (int j = 0; j < tiley; j++)
			{
				uchar* d = dest.ptr<uchar>(tiley * idx.y + j, tilex * idx.x);
				const uchar* s = src[0].ptr<uchar>(top + j, left);
				for (int i = 0; i < simd_tile_width0; i += align0)
				{
					_mm256_storeu_si256((__m256i*)(d + i), _mm256_loadu_si256((__m256i*)(s + i)));
				}
				for (int i = simd_tile_width0; i < simd_tile_width1; i += align1)
				{
					_mm_storeu_si128((__m128i*)(d + i), _mm_loadu_si128((__m128i*)(s + i)));
				}
				for (int i = simd_tile_width1; i < simd_tile_width2; i += align2)
				{
					_mm_storel_epi64((__m128i*)(d + i), _mm_loadl_epi64((__m128i*)(s + i)));
				}
				for (int i = simd_tile_width2; i < tilex; i++)
				{
					d[i] = s[i];
				}
			}
		}
		else if (channels == 3)
		{
			for (int j = 0; j < tiley; j++)
			{
				uchar* d = dest.ptr<uchar>(tiley * idx.y + j, tilex * idx.x);
				const uchar* s0 = src[0].ptr<uchar>(top + j, left);
				const uchar* s1 = src[1].ptr<uchar>(top + j, left);
				const uchar* s2 = src[2].ptr<uchar>(top + j, left);
				for (int i = 0; i < simd_tile_width0; i += align0)
				{
					_mm256_storeu_interleave_epi8_si256((d + 3 * i), _mm256_loadu_si256((__m256i*)(s0 + i)), _mm256_loadu_si256((__m256i*)(s1 + i)), _mm256_loadu_si256((__m256i*)(s2 + i)));
				}
				for (int i = simd_tile_width0; i < simd_tile_width1; i += align1)
				{
					_mm_storeu_interleave_epi8_si128((d + 3 * i), _mm_loadu_si128((__m128i*)(s0 + i)), _mm_loadu_si128((__m128i*)(s1 + i)), _mm_loadu_si128((__m128i*)(s2 + i)));
				}
				for (int i = simd_tile_width1; i < simd_tile_width2; i += align2)
				{
					_mm_store_interleave_epi8_epi64((d + 3 * i), _mm_loadl_epi64((__m128i*)(s0 + i)), _mm_loadl_epi64((__m128i*)(s1 + i)), _mm_loadl_epi64((__m128i*)(s2 + i)));
				}
				for (int i = simd_tile_width2; i < tilex; i++)
				{
					d[3 * i + 0] = s0[i];
					d[3 * i + 1] = s1[i];
					d[3 * i + 2] = s2[i];
				}
			}
		}
	}

	void pasteMergeTile(const vector<Mat>& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		CV_Assert(!dest.empty());
		CV_Assert(src[0].depth() == dest.depth());
		CV_Assert(src.size() == dest.channels());
		//CV_Assert(dest.cols % div_size.width == 0 && dest.rows % div_size.height == 0);

		if (src[0].depth() == CV_8U)
		{
			pasteMergeTile_internal<uchar>(src, dest, div_size, idx, top, left);
		}
		else if (src[0].depth() == CV_32F)
		{
			pasteMergeTile_internal<float>(src, dest, div_size, idx, top, left);
		}
		else if (src[0].depth() == CV_64F)
		{
			pasteMergeTile_internal<double>(src, dest, div_size, idx, top, left);
		}
	}

	void pasteMergeTile(const vector<Mat>& src, Mat& dest, const Size div_size, const Point idx, const int r)
	{
		pasteMergeTile(src, dest, div_size, idx, r, r);
	}

	void pasteMergeTileAlign(const vector<Mat>& src, Mat& dest, const Size div_size, const Point idx, const int r, const int left_multiple, const int top_multiple)
	{
		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);
		pasteMergeTile(src, dest, div_size, idx, L, T);
	}
#pragma endregion

#pragma region divide and conquer tiles
	void divideTiles(const Mat& src, vector<Mat>& dest, const Size div_size, const int r, const int borderType)
	{
		CV_Assert(src.channels() == 1);
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);
		if (dest.size() == 0)dest.resize(div_size.area());

		int sub_index = 0;
		for (int j = 0; j < div_size.height; j++)
		{
			for (int i = 0; i < div_size.width; i++)
			{
				Point idx = Point(i, j);
				cropTile(src, dest[sub_index], div_size, idx, r, borderType);
				sub_index++;
			}
		}
	}

	void divideTilesAlign(const Mat& src, vector<Mat>& dest, const Size div_size, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
	{
		CV_Assert(src.channels() == 1);
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);
		if (dest.size() == 0)dest.resize(div_size.area());

		int sub_index = 0;
		for (int j = 0; j < div_size.height; j++)
		{
			for (int i = 0; i < div_size.width; i++)
			{
				Point idx = Point(i, j);
				cropTileAlign(src, dest[sub_index], div_size, idx, r, borderType, align_x, align_y);
				sub_index++;
			}
		}
	}

	void conquerTiles(const vector<Mat>& src, Mat& dest, const Size div_size, const int r)
	{
		CV_Assert(src[0].channels() == 1);
		CV_Assert(src[0].depth() == CV_32F || src[0].depth() == CV_64F);

		int numBlockImages = div_size.area();
		//#pragma omp parallel for
		for (int n = 0; n < numBlockImages; n++)
		{
			int j = n / div_size.width;
			int i = n % div_size.width;
			Point idx = Point(i, j);
			pasteTile(src[n], dest, div_size, idx, r);
		}
	}

	void conquerTilesAlign(const vector<Mat>& src, Mat& dest, const Size div_size, const int r, const int left_multiple, const int top_multiple)
	{
		CV_Assert(src[0].channels() == 1);
		CV_Assert(src[0].depth() == CV_32F || src[0].depth() == CV_64F);

		for (int j = 0; j < div_size.height; j++)
		{
			for (int i = 0; i < div_size.width; i++)
			{
				int sub_index = div_size.width * j + i;
				Point idx = Point(i, j);
				pasteTile(src[sub_index], dest, div_size, idx, r);
			}
		}
	}

	void TileDivision::update_pt()
	{
		for (int i = 0; i < div.width; i++) pt[i].y = 0;
		pt[0].x = 0;
		for (int i = 1; i < div.width; i++)
			pt[i].x = pt[i - 1].x + tileSize[i - 1].width;

		for (int j = 1; j < div.height; j++)
		{
			pt[div.width * j].x = 0;
			pt[div.width * j].y = pt[div.width * (j - 1)].y + tileSize[div.width * (j - 1)].height;
			for (int i = 1; i < div.width; i++)
			{
				const int ptidx = div.width * (j - 1) + i;
				const int tidx = div.width * j + i;
				pt[tidx].x = pt[tidx - 1].x + tileSize[tidx - 1].width;
				pt[tidx].y = pt[ptidx].y + tileSize[ptidx].height;
			}
		}
	}

	//div.width * y + x;
	cv::Rect TileDivision::getROI(const int x, int y)
	{
		const int tindex = div.width * y + x;
		return getROI(tindex);
	}

	cv::Rect TileDivision::getROI(const int index)
	{
		return cv::Rect(pt[index].x, pt[index].y, tileSize[index].width, tileSize[index].height);
	}

	TileDivision::TileDivision()
	{
		;
	}

	void TileDivision::init(cv::Size imgSize, cv::Size div)
	{
		if (this->imgSize != imgSize)isRecompute = true;
		if (this->div != div)isRecompute = true;

		if (isRecompute)
		{
			this->imgSize = imgSize;
			this->div.width = std::max(div.width, 1);
			this->div.height = std::max(div.height, 1);

			tileSize.resize(this->div.area());
			pt.resize(this->div.area());
		}
	}

	TileDivision::TileDivision(cv::Size imgSize, cv::Size div)
	{
		init(imgSize, div);
	}

	bool TileDivision::compute(const int width_step_, const int height_step_)
	{
		if (width_step != width_step_ || height_step != height_step_)
			isRecompute = true;

		if (!isRecompute) return preReturnFlag;

		isRecompute = false;
		width_step = width_step_;
		height_step = height_step_;

		const int xmulti = width_step * div.width;
		const int ymulti = width_step * div.height;
		const int x_base_width = get_simd_floor(imgSize.width, xmulti);
		const int x_base_height = get_simd_floor(imgSize.height, ymulti);
		const int x_rem = imgSize.width - x_base_width;
		const int y_rem = imgSize.height - x_base_height;

		const int x_base_tilewidth = x_base_width / div.width;
		const int y_base_tileheight = x_base_height / div.height;

		const int num_base_padtile_x = x_rem / width_step;
		const int num_base_padtile_y = y_rem / height_step;

		const int num_last_padtile_x = x_rem - num_base_padtile_x * width_step;
		const int num_last_padtile_y = y_rem - num_base_padtile_y * height_step;

		for (int i = 0; i < div.area(); i++)
		{
			tileSize[i].width = x_base_tilewidth;
		}
		if (num_last_padtile_x == 0)
		{
			for (int j = 0; j < div.height; j++)
			{
				for (int i = 0; i < num_base_padtile_x; i++)
				{
					const int tidx = div.width * j + (div.width - 1 - i);
					tileSize[tidx].width = x_base_tilewidth + width_step;
				}
			}
		}
		else
		{
			for (int j = 0; j < div.height; j++)
			{
				for (int i = 0; i < num_base_padtile_x; i++)
				{
					const int tidx = div.width * j + (div.width - 2 - i);
					tileSize[tidx].width = x_base_tilewidth + width_step;
				}
				tileSize[div.width * j + (div.width - 1)].width = x_base_tilewidth + num_last_padtile_x;
			}
		}

		for (int i = 0; i < div.area(); i++)
		{
			tileSize[i].height = y_base_tileheight;
		}
		if (num_last_padtile_y == 0)
		{
			for (int i = 0; i < div.width; i++)
			{
				for (int j = 0; j < num_base_padtile_y; j++)
				{
					const int tidx = div.width * (div.height - 1 - j) + i;
					tileSize[tidx].height = y_base_tileheight + height_step;
				}
			}
		}
		else
		{
			for (int i = 0; i < div.width; i++)
			{
				for (int j = 0; j < num_base_padtile_y; j++)
				{
					const int tidx = div.width * (div.height - 2 - j) + i;
					tileSize[tidx].height = y_base_tileheight + height_step;
				}
				tileSize[div.width * (div.height - 1) + i].height = y_base_tileheight + num_last_padtile_y;
			}
		}

		update_pt();
		if (threadnum.size() != div.area())threadnum.resize(div.area());
#pragma omp parallel for schedule (static)
		for (int i = 0; i < div.area(); i++)
		{
			threadnum[i] = omp_get_thread_num();
		}

		preReturnFlag = (num_last_padtile_x == 0 && num_last_padtile_y == 0);
		return preReturnFlag;
	}

	void TileDivision::draw(cv::Mat& src, cv::Mat& dst)
	{
		if (src.data == dst.data)
		{
			dst = src;
		}
		else
		{
			dst.create(src.size(), CV_8UC3);
			switch (src.type())
			{
			case CV_8UC3: dst = src.clone(); break;
			case CV_8UC1: cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR); break;

			default:
			{
				if (src.channels() == 3)
				{
					dst = cp::convert(src, CV_8U); break;
				}
				else
				{
					cv::Mat temp = cp::convert(src, CV_8U);
					cv::cvtColor(temp, dst, cv::COLOR_GRAY2BGR); break;
				}
				break;
			}
			}
		}

		const int threadMax = omp_get_max_threads();

		for (int i = 0; i < div.area(); i++)
		{
			const int threadNumber = threadnum[i];
			//const int threadNumber = 0;
			cv::Rect roi = getROI(i);
			if (tileSize[i].width % width_step == 0 && tileSize[i].height % height_step == 0)
			{
				rectangle(dst, roi, COLOR_GREEN);
			}
			else
			{
				rectangle(dst, roi, COLOR_YELLOW);
			}
			if (pt[i].x + tileSize[i].width > imgSize.width || pt[i].y + tileSize[i].height > imgSize.height)
			{
				rectangle(dst, roi, COLOR_RED);
			}
			const Scalar txtcolor = COLOR_WHITE;
			//const Scalar txtcolor = COLOR_GREEN;
			addText(dst, std::to_string(tileSize[i].width) + "x" + std::to_string(tileSize[i].height), pt[i] + cv::Point(10, 20), "Consolas", 12, txtcolor);
			addText(dst, cv::format("#T %d/%d", threadNumber, threadMax - 1), pt[i] + cv::Point(10, 40), "Consolas", 12, txtcolor);

			//cout << roi << endl;
		}
	}

	void TileDivision::draw(cv::Mat& src, cv::Mat& dst, std::vector<std::string>& info)
	{
		if (src.data == dst.data)
		{
			dst = src;
		}
		else
		{
			dst.create(src.size(), CV_8UC3);
			switch (src.type())
			{
			case CV_8UC3: dst = src.clone(); break;
			case CV_8UC1: cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR); break;

			default:
			{
				if (src.channels() == 3)
				{
					dst = cp::convert(src, CV_8U); break;
				}
				else
				{
					cv::Mat temp = cp::convert(src, CV_8U);
					cv::cvtColor(temp, dst, cv::COLOR_GRAY2BGR); break;
				}
				break;
			}
			}
		}

		const int threadMax = omp_get_max_threads();

		for (int i = 0; i < div.area(); i++)
		{
			const int threadNumber = threadnum[i];
			//const int threadNumber = 0;
			cv::Rect roi = getROI(i);
			if (tileSize[i].width % width_step == 0 && tileSize[i].height % height_step == 0)
			{
				rectangle(dst, roi, COLOR_GREEN);
			}
			else
			{
				rectangle(dst, roi, COLOR_YELLOW);
			}
			if (pt[i].x + tileSize[i].width > imgSize.width || pt[i].y + tileSize[i].height > imgSize.height)
			{
				rectangle(dst, roi, COLOR_RED);
			}
			const Scalar txtcolor = COLOR_WHITE;
			//const Scalar txtcolor = COLOR_GREEN;
			addText(dst, std::to_string(tileSize[i].width) + "x" + std::to_string(tileSize[i].height), pt[i] + cv::Point(10, 20), "Consolas", 12, txtcolor);
			addText(dst, cv::format("#T %d/%d", threadNumber, threadMax - 1), pt[i] + cv::Point(10, 40), "Consolas", 12, txtcolor);
			addText(dst, cv::format("%s", info[i]), pt[i] + cv::Point(10, 60), "Consolas", 12, txtcolor);

			//cout << roi << endl;
		}
	}

	void TileDivision::draw(cv::Mat& src, cv::Mat& dst, std::vector<std::string>& info, std::vector<std::string>& info2)
	{
		if (src.data == dst.data)
		{
			dst = src;
		}
		else
		{
			dst.create(src.size(), CV_8UC3);
			switch (src.type())
			{
			case CV_8UC3: dst = src.clone(); break;
			case CV_8UC1: cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR); break;

			default:
			{
				if (src.channels() == 3)
				{
					dst = cp::convert(src, CV_8U); break;
				}
				else
				{
					cv::Mat temp = cp::convert(src, CV_8U);
					cv::cvtColor(temp, dst, cv::COLOR_GRAY2BGR); break;
				}
				break;
			}
			}
		}

		const int threadMax = omp_get_max_threads();

		for (int i = 0; i < div.area(); i++)
		{
			const int threadNumber = threadnum[i];
			//const int threadNumber = 0;
			cv::Rect roi = getROI(i);
			if (tileSize[i].width % width_step == 0 && tileSize[i].height % height_step == 0)
			{
				rectangle(dst, roi, COLOR_GREEN);
			}
			else
			{
				rectangle(dst, roi, COLOR_YELLOW);
			}
			if (pt[i].x + tileSize[i].width > imgSize.width || pt[i].y + tileSize[i].height > imgSize.height)
			{
				rectangle(dst, roi, COLOR_RED);
			}
			const Scalar txtcolor = COLOR_WHITE;
			//const Scalar txtcolor = COLOR_GREEN;
			addText(dst, std::to_string(tileSize[i].width) + "x" + std::to_string(tileSize[i].height), pt[i] + cv::Point(10, 20), "Consolas", 12, txtcolor);
			addText(dst, cv::format("#T %d/%d", threadNumber, threadMax - 1), pt[i] + cv::Point(10, 40), "Consolas", 12, txtcolor);
			addText(dst, cv::format("%s", info[i]), pt[i] + cv::Point(10, 60), "Consolas", 12, txtcolor);
			addText(dst, cv::format("%s", info2[i]), pt[i] + cv::Point(10, 80), "Consolas", 12, txtcolor);

			//cout << roi << endl;
		}
	}

	void TileDivision::show(std::string wname)
	{
		cv::Mat show = cv::Mat::zeros(imgSize, CV_8UC3);
		draw(show, show);
		imshow(wname, show);
	}

#pragma region TileParallelBody
	void TileParallelBody::init(const Size div)
	{
		this->div = div;

		if (srcTile.size() != div.area())srcTile.resize(div.area());
		if (dstTile.size() != div.area())dstTile.resize(div.area());
	}

	void TileParallelBody::initGuide(const Size div, std::vector<Mat>& guide)
	{
		isUseGuide = true;
		this->guideMaps = guide;
		const int numGuides = (int)guide.size();
		if (guideTile.size() != numGuides)
		{
			guideTile.resize(numGuides);
			for (int i = 0; i < numGuides; i++)
			{
				guideTile[i].resize(div.area());
			}
		}
	}

	void TileParallelBody::unsetUseGuide()
	{
		isUseGuide = false;
	}

	void TileParallelBody::drawMinMax(string wname, cv::Mat& src)
	{
		vector<string>info(div.area());
		//vector<string>info2(div.area());

		for (int i = 0; i < div.area(); i++)
		{
			double minv, maxv;
			minMaxLoc(srcTile[i], &minv, &maxv);
			info[i] = format("%4.1f %4.1f", minv, maxv);
			//minMaxLoc(dstTile[i], &minv, &maxv);
			//info2[i] = format("%5.2f %5.2f", minv, maxv);
		}
		Mat dst;
		tdiv.draw(src, dst, info);
		imshow(wname, dst);
	}

	void TileParallelBody::invoker(const cv::Size div, const cv::Mat& src, cv::Mat& dst, int tileBoundary, const int borderType, const int depth)
	{
		init(div);

		dst.create(src.size(), (depth < 0) ? src.depth() : depth);
		int vecsize = (dst.depth() == CV_64F) ? 4 : 8;

		tdiv.init(src.size(), div);
		tdiv.compute(vecsize, vecsize);

		if (0 < tileBoundary && tileBoundary < vecsize)
		{
			tileBoundary = vecsize;
		}

#pragma omp parallel for schedule (static)
		for (int n = 0; n < div.area(); n++)
		{
			const int threadNumber = omp_get_thread_num();
			Rect roi = tdiv.getROI(n);
			cp::cropTileAlign(src, srcTile[n], roi, tileBoundary, borderType, vecsize, vecsize, 1);
			if (isUseGuide)
			{
				for (int i = 0; i < guideTile.size(); i++)
				{
					cp::cropTileAlign(guideMaps[i], guideTile[i][n], roi, tileBoundary, borderType, vecsize, vecsize, 1);
				}
			}
			if (src.data == dst.data)
			{
				process(srcTile[n], srcTile[n], threadNumber, n);
				if (srcTile[n].depth() != dst.depth())
				{
					print_matinfo(src);
					print_matinfo(srcTile[n]);
					print_matinfo(dst);
				}
				cp::pasteTile(srcTile[n], dst, roi, tileBoundary);
			}
			else
			{
				process(srcTile[n], dstTile[n], threadNumber, n);
				if (dstTile[n].depth() != dst.depth())
				{
					print_matinfo(dstTile[n]);
					print_matinfo(dst);
				}
				cp::pasteTile(dstTile[n], dst, roi, tileBoundary);
			}
		}
		tileSize = srcTile[0].size();

		//Mat show;
		//tdiv.draw(dst, show);
		//imshow("tile", show);
	}

	cv::Size TileParallelBody::getTileSize()
	{
		return tileSize;
	}
#pragma endregion

#pragma endregion
}
