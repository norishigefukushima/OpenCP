#include "tiling.hpp"
#include "../include/inlineSIMDFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	void createSubImage8U_Reflect(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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
			memcpy(d + left, s - sleft, sizeof(uchar)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(uchar)*(copysizex));
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
			memcpy(d, s, sizeof(uchar)*(tileSizeXExternal));
		}

		const int sidx = tileSizeYInternal * (div_size.height - idx.y) + topb - 1;
		const int didx = tileSizeYInternal * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx - j));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar)*tileSizeXExternal);
		}
	}

	void createSubImage8U_Reflect101(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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
				a = _mm_load_si128((__m128i*)(s + LOAD_OFFSET1 - i + 1));
				a = _mm_shuffle_epi8(a, vm);
				_mm_store_si128((__m128i*)(d + i), a);
			}
			memcpy(d + left, s - sleft, sizeof(uchar)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(uchar)*(copysizex));
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
			memcpy(d, s, sizeof(uchar)*(tileSizeXExternal));
		}

		const int sidx = tileSizeYInternal * (div_size.height - idx.y) + topb - 2;
		const int didx = tileSizeYInternal * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx - j));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar)*tileSizeXExternal);
		}
	}

	void createSubImage8U_Replicate(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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
			memcpy(d + left, s - sleft, sizeof(uchar)*(memcpySizeX));
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
			memcpy(d + left, s - sleft, sizeof(uchar)*(memcpySizeX));
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
			memcpy(d, s, sizeof(uchar)*(tileSizeXExternal));
		}

		const int sidx = tileSizeYInternal * (div_size.height - idx.y) + topb - 1;
		const int didx = tileSizeYInternal * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			uchar* s = dest.ptr<uchar>(max(0, sidx));
			uchar* d = dest.ptr<uchar>(didx + j);
			memcpy(d, s, sizeof(uchar)*tileSizeXExternal);
		}
	}


	void createSubImage32F_Reflect(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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

		const float* s = src.ptr<float>(tiley*idx.y - stop, top_tilex);
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
			memcpy(d + left, s - sleft, sizeof(float)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(float)*(copysizex));
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
			memcpy(d, s, sizeof(float)*(dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx - j));
			float* d = dest.ptr<float>(didx + j);

			memcpy(d, s, sizeof(float)*dest_tilex);
		}
	}

	void createSubImage32F_Reflect101(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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

		const float* s = src.ptr<float>(tiley*idx.y - stop) + top_tilex;
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
				a = _mm256_loadu_ps(s + LOAD_OFFSET1 - i + 1);
				a = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
				a = _mm256_permute2f128_ps(a, a, 1);
				_mm256_store_ps(d + i, a);
			}
			memcpy(d + left, s - sleft, sizeof(float)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(float)*(copysizex));
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
			memcpy(d, s, sizeof(float)*(dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 2;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx - j));
			float* d = dest.ptr<float>(didx + j);
			memcpy(d, s, sizeof(float)*dest_tilex);
		}
	}

	void createSubImage32F_Replicate(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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

		const float* s = src.ptr<float>(tiley*idx.y - stop, top_tilex);
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
			memcpy(d + left, s - sleft, sizeof(float)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(float)*(copysizex));
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
			memcpy(d, s, sizeof(float)*(dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			float* s = dest.ptr<float>(max(0, sidx));
			float* d = dest.ptr<float>(didx + j);
			memcpy(d, s, sizeof(float)*dest_tilex);
		}
	}


	void createSubImage64F_Reflect(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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

		const double* s = src.ptr<double>(tiley*idx.y - stop) + top_tilex;
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
			memcpy(d + left, s - sleft, sizeof(double)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(double)*(copysizex));
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

			memcpy(d, s, sizeof(double)*(dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx - j));
			double* d = dest.ptr<double>(didx + j);

			memcpy(d, s, sizeof(double)*dest_tilex);
		}
	}

	void createSubImage64F_Reflect101(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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

		const double* s = src.ptr<double>(tiley*idx.y - stop) + top_tilex;
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
			memcpy(d + left, s - sleft, sizeof(double)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(double)*(copysizex));
			for (int i = 0; i < RIGHT; i += 8)
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

			memcpy(d, s, sizeof(double)*(dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 2;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx - j));
			double* d = dest.ptr<double>(didx + j);

			memcpy(d, s, sizeof(double)*dest_tilex);
		}
	}

	void createSubImage64F_Replicate(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb)
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

		const double* s = src.ptr<double>(tiley*idx.y - stop, top_tilex);
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
			memcpy(d + left, s - sleft, sizeof(double)*(copysizex));
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
			memcpy(d + left, s - sleft, sizeof(double)*(copysizex));
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
			memcpy(d, s, sizeof(double)*(dest_tilex));
		}

		const int sidx = tiley * (div_size.height - idx.y) + topb - 1;
		const int didx = tiley * (div_size.height - idx.y) + topb;
		for (int j = 0; j < bottom; j++)
		{
			double* s = dest.ptr<double>(max(0, sidx));
			double* d = dest.ptr<double>(didx + j);
			memcpy(d, s, sizeof(double)*dest_tilex);
		}
	}


	void createSubImage(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType)
	{
		CV_Assert(borderType == BORDER_REFLECT101 || borderType == BORDER_REFLECT || borderType == BORDER_REPLICATE);

		if (src.depth() == CV_8U)
		{
			if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT) createSubImage8U_Reflect101(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) createSubImage8U_Reflect(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) createSubImage8U_Replicate(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
		}
		else if (src.depth() == CV_32F)
		{
			if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT) createSubImage32F_Reflect101(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) createSubImage32F_Reflect(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) createSubImage32F_Replicate(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
		}
		else if (src.depth() == CV_64F)
		{
			if (borderType == BORDER_REFLECT101 || borderType == BORDER_DEFAULT) createSubImage64F_Reflect101(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REFLECT) createSubImage64F_Reflect(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
			else if (borderType == BORDER_REPLICATE) createSubImage64F_Replicate(src, dest, div_size, idx, topb, bottomb, leftb, rightb);
		}
	}

	void createSubImage(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int r, const int borderType)
	{
		createSubImage(src, dest, div_size, idx, r, r, r, r, borderType);
	}

	Size getSubImageSize(const Size src, const Size div_size, const int r)
	{
		const int tilex = src.width / div_size.width;
		const int tiley = src.height / div_size.height;
		return Size(tilex + 2 * r, tiley + 2 * r);
	}

	Size getSubImageAlignSize(const Size src, const Size div_size, const int r, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
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

	void createSubImageAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
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
		createSubImage(src, dest, div_size, idx, T, B, L, R, borderType);
		//cout << format("%d %d %d %d\n", L, R, T, B);
	}

	void setSubImage32F_(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		CV_Assert(!dest.empty());
		CV_Assert(dest.cols % div_size.width == 0 && dest.rows % div_size.height == 0);

		const int tilex = dest.cols / div_size.width;
		const int tiley = dest.rows / div_size.height;

		const int simd_tile_width = get_simd_floor(tilex, 8);
		const int rem = tilex - simd_tile_width;

		for (int j = 0; j < tiley; j++)
		{
			float* d = dest.ptr<float>(tiley*idx.y + j, tilex*idx.x);
			const float * s = src.ptr<float>(top + j, left);
			for (int i = 0; i < simd_tile_width; i += 8)
			{
				_mm256_storeu_ps(d + i, _mm256_loadu_ps(s + i));
			}
			for (int i = 0; i < rem; i++)
			{
				d[simd_tile_width + i] = s[simd_tile_width + i];
			}
		}
	}

	void setSubImage64F_(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		CV_Assert(!dest.empty());
		CV_Assert(dest.cols % div_size.width == 0 && dest.rows % div_size.height == 0);

		const int tilex = dest.cols / div_size.width;
		const int tiley = dest.rows / div_size.height;

		const int simd_tile_width = get_simd_floor(tilex, 4);
		const int rem = tilex - simd_tile_width;

		for (int j = 0; j < tiley; j++)
		{
			double* d = dest.ptr<double>(tiley*idx.y + j, tilex*idx.x);
			const double * s = src.ptr<double>(top + j, left);
			for (int i = 0; i < simd_tile_width; i += 4)
			{
				_mm256_storeu_pd(d + i, _mm256_loadu_pd(s + i));
			}
			for (int i = 0; i < rem; i++)
			{
				d[simd_tile_width + i] = s[simd_tile_width + i];
			}
		}
	}

	void setSubImage(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int top, const int left)
	{
		if (src.depth() == CV_32F)
			setSubImage32F_(src, dest, div_size, idx, top, left);
		else
			setSubImage64F_(src, dest, div_size, idx, top, left);
	}

	void setSubImage(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int r)
	{
		setSubImage(src, dest, div_size, idx, r, r);
	}

	void setSubImageAlign(const Mat& src, Mat& dest, const Size div_size, const Point idx, const int r, const int left_multiple, const int top_multiple)
	{
		const int L = get_simd_ceil(r, left_multiple);
		const int T = get_simd_ceil(r, top_multiple);
		setSubImage(src, dest, div_size, idx, L, T);
	}


	void splitSubImage(const Mat& src, vector<Mat>& dest, const Size div_size, const int r, const int borderType)
	{
		CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);
		if (dest.size() == 0)dest.resize(div_size.area());

		int sub_index = 0;
		for (int j = 0; j < div_size.height; j++)
		{
			for (int i = 0; i < div_size.width; i++)
			{
				Point idx = Point(i, j);
				createSubImage(src, dest[sub_index], div_size, idx, r, borderType);
				sub_index++;
			}
		}
	}

	void splitSubImageAlign(const Mat& src, vector<Mat>& dest, const Size div_size, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple, const int top_multiple)
	{
		if (dest.size() == 0)dest.resize(div_size.area());

		int sub_index = 0;
		for (int j = 0; j < div_size.height; j++)
		{
			for (int i = 0; i < div_size.width; i++)
			{
				Point idx = Point(i, j);
				createSubImageAlign(src, dest[sub_index], div_size, idx, r, borderType, align_x, align_y);
				sub_index++;
			}
		}
	}

	void mergeSubImage(const vector<Mat>& src, Mat& dest, const Size div_size, const int r)
	{
		int numBlockImages = div_size.area();
		//#pragma omp parallel for
		for (int n = 0; n < numBlockImages; n++)
		{
			int j = n / div_size.width;
			int i = n % div_size.width;
			Point idx = Point(i, j);
			setSubImage(src[n], dest, div_size, idx, r);
		}
	}

	void mergeSubImageAlign(const vector<Mat>& src, Mat& dest, const Size div_size, const int r, const int left_multiple, const int top_multiple)
	{
		for (int j = 0; j < div_size.height; j++)
		{
			for (int i = 0; i < div_size.width; i++)
			{
				int sub_index = div_size.width*j + i;
				Point idx = Point(i, j);
				setSubImage(src[sub_index], dest, div_size, idx, r);
			}
		}
	}
}
