#include "color.hpp"
#include "arithmetic.hpp"
#include "statistic.hpp"
#include "inlineSIMDfunctions.hpp"
#include "metrics.hpp"
#include "concat.hpp"

#include "debugcp.hpp"
using namespace std;
using namespace cv;

namespace cp
{
#pragma region merge
	template<typename srcType, typename dstType>
	static void mergeConvertBase_(const vector<Mat>& src, Mat& dest, const double scale, const double offset)
	{
		if (dest.empty())dest.create(src[0].size(), src[0].depth());
		const int size = src[0].size().area();
		dstType* d = dest.ptr<dstType>();
		const srcType* s0 = src[0].ptr<srcType>();
		const srcType* s1 = src[1].ptr<srcType>();
		const srcType* s2 = src[2].ptr<srcType>();
		for (int i = 0; i < size; i++)
		{
			d[0] = saturate_cast<dstType>(scale * s0[0] + offset);
			d[1] = saturate_cast<dstType>(scale * s1[0] + offset);
			d[2] = saturate_cast<dstType>(scale * s2[0] + offset);
			d += 3;
			s0++;
			s1++;
			s2++;
		}
	}

	template<class srcType>
	static void mergeBase_(vector<Mat>& src, Mat& dest)
	{
		if (dest.empty())dest.create(src[0].size(), src[0].depth());
		const int size = src[0].size().area();
		srcType* d = dest.ptr<srcType>(0);
		srcType* s0 = src[0].ptr<srcType>(0);
		srcType* s1 = src[1].ptr<srcType>(0);
		srcType* s2 = src[2].ptr<srcType>(0);
		for (int i = 0; i < size; i++)
		{
			d[0] = s0[0];
			d[1] = s1[0];
			d[2] = s2[0];
			d += 3;
			s0++;
			s1++;
			s2++;
		}
	}

	static void mergeStore_8U(vector<Mat>& src, Mat& dest)
	{
		if (dest.empty())dest.create(src[0].size(), src[0].depth());

		const uchar* bptr = src[0].ptr<uchar>(0);
		const uchar* gptr = src[1].ptr<uchar>(0);
		const uchar* rptr = src[2].ptr<uchar>(0);
		uchar* dptr = dest.ptr<uchar>(0);

		const __m256i mask1 = _mm256_set_epi8(5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0);
		const __m256i mask2 = _mm256_set_epi8(10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5);
		const __m256i mask3 = _mm256_set_epi8(15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10);

		const __m256i bmask1 = _mm256_set_epi8
		(255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
		const __m256i bmask2 = _mm256_set_epi8
		(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255);

		const int size = src[0].size().area();
		const int sizeavx = src[0].size().area() / 32;
		const int rem = sizeavx * 32;

		for (int i = 0; i < sizeavx; i++)
		{
			__m256i a = _mm256_load_si256((__m256i*)(bptr));
			__m256i b = _mm256_load_si256((__m256i*)(gptr));
			__m256i c = _mm256_load_si256((__m256i*)(rptr));

			a = _mm256_shuffle_epi8(a, mask1);
			b = _mm256_shuffle_epi8(b, mask2);
			c = _mm256_shuffle_epi8(c, mask3);
			__m256i aa = _mm256_permute2x128_si256(a, a, 0x00);
			__m256i bb = _mm256_permute2x128_si256(b, b, 0x00);
			__m256i cc = _mm256_permute2x128_si256(c, c, 0x00);

			_mm256_store_si256((__m256i*)(dptr), _mm256_blendv_epi8(cc, _mm256_blendv_epi8(aa, bb, bmask1), bmask2));
			_mm256_store_si256((__m256i*)(dptr + 32), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask2), bmask1));
			aa = _mm256_permute2x128_si256(a, a, 0x11);
			bb = _mm256_permute2x128_si256(b, b, 0x11);
			cc = _mm256_permute2x128_si256(c, c, 0x11);
			_mm256_store_si256((__m256i*)(dptr + 64), _mm256_blendv_epi8(aa, _mm256_blendv_epi8(bb, cc, bmask1), bmask2));

			bptr += 32;
			gptr += 32;
			rptr += 32;
			dptr += 96;
		}

		const int f = size - rem;
		for (int i = 0; i < f; i++)
		{
			dptr[i] = bptr[3 * i];
			dptr[i] = gptr[3 * i + 1];
			dptr[i] = rptr[3 * i + 2];
		}
	}

	static void mergeStream_8U(vector<Mat>& src, Mat& dest)
	{
		if (dest.empty())dest.create(src[0].size(), src[0].depth());

		const uchar* bptr = src[0].ptr<uchar>(0);
		const uchar* gptr = src[1].ptr<uchar>(0);
		const uchar* rptr = src[2].ptr<uchar>(0);
		uchar* dptr = dest.ptr<uchar>(0);

		const __m256i mask1 = _mm256_set_epi8(5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0);
		const __m256i mask2 = _mm256_set_epi8(10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5);
		const __m256i mask3 = _mm256_set_epi8(15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10);

		const __m256i bmask1 = _mm256_set_epi8
		(255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
		const __m256i bmask2 = _mm256_set_epi8
		(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255);

		const int size = src[0].size().area();
		const int sizeavx = src[0].size().area() / 32;
		const int rem = sizeavx * 32;

		for (int i = 0; i < sizeavx; i++)
		{
			__m256i a = _mm256_load_si256((__m256i*)(bptr));
			__m256i b = _mm256_load_si256((__m256i*)(gptr));
			__m256i c = _mm256_load_si256((__m256i*)(rptr));

			a = _mm256_shuffle_epi8(a, mask1);
			b = _mm256_shuffle_epi8(b, mask2);
			c = _mm256_shuffle_epi8(c, mask3);
			__m256i aa = _mm256_permute2x128_si256(a, a, 0x00);
			__m256i bb = _mm256_permute2x128_si256(b, b, 0x00);
			__m256i cc = _mm256_permute2x128_si256(c, c, 0x00);

			_mm256_stream_si256((__m256i*)(dptr), _mm256_blendv_epi8(cc, _mm256_blendv_epi8(aa, bb, bmask1), bmask2));
			_mm256_stream_si256((__m256i*)(dptr + 32), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask2), bmask1));
			aa = _mm256_permute2x128_si256(a, a, 0x11);
			bb = _mm256_permute2x128_si256(b, b, 0x11);
			cc = _mm256_permute2x128_si256(c, c, 0x11);
			_mm256_stream_si256((__m256i*)(dptr + 64), _mm256_blendv_epi8(aa, _mm256_blendv_epi8(bb, cc, bmask1), bmask2));

			bptr += 32;
			gptr += 32;
			rptr += 32;
			dptr += 96;
		}

		const int f = size - rem;
		for (int i = 0; i < f; i++)
		{
			dptr[i] = bptr[3 * i];
			dptr[i] = gptr[3 * i + 1];
			dptr[i] = rptr[3 * i + 2];
		}
	}

	void mergeConvert(cv::InputArrayOfArrays src, cv::OutputArray dest, const int depth, const double scale, const double offset, const bool isCache)
	{
		vector<Mat> s;
		src.getMatVector(s);
		dest.create(s[0].size(), CV_MAKETYPE(depth, (int)s.size()));
		Mat dst = dest.getMat();

		if (depth == s[0].depth() && scale == 1.0 && offset == 1.0)
		{
			switch (s[0].depth())
			{
			case CV_8U:
				if (isCache) mergeStore_8U(s, dst);
				else mergeStream_8U(s, dst);
				break;
			case CV_32F:
				mergeBase_<float>(s, dst);
				break;
			default:
				cout << "not support type" << endl;
				break;
			}
		}
		else
		{
			if (s[0].depth() == CV_8U)
			{
				if (depth == CV_8U)  mergeConvertBase_<uchar, uchar>(s, dst, scale, offset);
				if (depth == CV_32F) mergeConvertBase_<uchar, float>(s, dst, scale, offset);
			}
			if (s[0].depth() == CV_32F)
			{
				if (depth == CV_8U) mergeConvertBase_<float, uchar>(s, dst, scale, offset);
				if (depth == CV_32F) mergeConvertBase_<float, float>(s, dst, scale, offset);
			}
		}
	}
#pragma endregion

#pragma region split
	template<class srcType>
	void splitBase_(Mat& src, vector<Mat>& dst)
	{
		const int size = src.size().area();
		srcType* s = src.ptr<srcType>(0);
		srcType* d0 = dst[0].ptr<srcType>(0);
		srcType* d1 = dst[1].ptr<srcType>(0);
		srcType* d2 = dst[2].ptr<srcType>(0);
		for (int i = 0; i < size; i++)
		{
			d0[0] = s[0];
			d1[0] = s[1];
			d2[0] = s[2];
			s += 3;
			d0++;
			d1++;
			d2++;
		}
	}

	void splitStore_8U(Mat& src, vector<Mat>& dst)
	{
		uchar* s = src.ptr<uchar>(0);
		uchar* B = dst[0].ptr<uchar>(0);
		uchar* G = dst[1].ptr<uchar>(0);
		uchar* R = dst[2].ptr<uchar>(0);

		//BGR BGR BGR BGR BGR B x2
		//GR BGR BGR BGR BGR BG x2
		//R BGR BGR BGR BGR BGR x2
		//BBBBBBGGGGGRRRRR shuffle
		const __m256i mask1 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14,
			0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
		//GGGGGBBBBBBRRRRR shuffle
		const __m256i smask1 = _mm256_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15,
			6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
		const __m256i ssmask1 = _mm256_setr_epi8(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
			11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

		//GGGGGGBBBBBRRRRR shuffle
		const __m256i mask2 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
			0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
		const __m256i ssmask2 = _mm256_setr_epi8(0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10,
			0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10);

		const __m256i bmask1 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		const __m256i bmask2 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0);

		const __m256i bmask3 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		const __m256i bmask4 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0);

		const int size = src.size().area();
		const int sizeavx = size / 32;
		const int rem = sizeavx * 32;
		for (int i = 0; i < sizeavx; i++)
		{
			const __m256i _a = _mm256_load_si256((__m256i*)(s + 0));
			const __m256i _b = _mm256_load_si256((__m256i*)(s + 32));
			const __m256i _c = _mm256_load_si256((__m256i*)(s + 64));
			__m256i a = _mm256_permute2x128_si256(_a, _b, 0x30);
			__m256i b = _mm256_permute2x128_si256(_a, _c, 0x21);
			__m256i c = _mm256_permute2x128_si256(_b, _c, 0x30);
			a = _mm256_shuffle_epi8(a, mask1);
			b = _mm256_shuffle_epi8(b, mask2);
			c = _mm256_shuffle_epi8(c, mask2);
			_mm256_store_si256((__m256i*)(B), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask1), bmask2));

			a = _mm256_shuffle_epi8(a, smask1);
			b = _mm256_shuffle_epi8(b, smask1);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_store_si256((__m256i*)(G), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask2));

			a = _mm256_shuffle_epi8(a, ssmask1);
			b = _mm256_shuffle_epi8(b, ssmask2);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_store_si256((__m256i*)(R), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask4));

			B += 32;
			G += 32;
			R += 32;
			s += 96;
		}

		const int f = size - rem;
		for (int i = 0; i < f; i++)
		{
			B[i] = s[3 * i];
			G[i] = s[3 * i + 1];
			R[i] = s[3 * i + 2];
		}
	}

	void splitStream_8U(Mat& src, vector<Mat>& dst)
	{
		uchar* s = src.ptr<uchar>(0);
		uchar* B = dst[0].ptr<uchar>(0);
		uchar* G = dst[1].ptr<uchar>(0);
		uchar* R = dst[2].ptr<uchar>(0);

		//BGR BGR BGR BGR BGR B x2
		//GR BGR BGR BGR BGR BG x2
		//R BGR BGR BGR BGR BGR x2
		//BBBBBBGGGGGRRRRR shuffle
		const __m256i mask1 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14,
			0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
		//GGGGGBBBBBBRRRRR shuffle
		const __m256i smask1 = _mm256_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15,
			6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
		const __m256i ssmask1 = _mm256_setr_epi8(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
			11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

		//GGGGGGBBBBBRRRRR shuffle
		const __m256i mask2 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
			0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
		const __m256i ssmask2 = _mm256_setr_epi8(0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10,
			0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10);

		const __m256i bmask1 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		const __m256i bmask2 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0);

		const __m256i bmask3 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		const __m256i bmask4 = _mm256_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0);

		const int size = src.size().area();
		const int sizeavx = size / 32;
		const int rem = sizeavx * 32;
		for (int i = 0; i < sizeavx; i++)
		{
			const __m256i _a = _mm256_load_si256((__m256i*)(s + 0));
			const __m256i _b = _mm256_load_si256((__m256i*)(s + 32));
			const __m256i _c = _mm256_load_si256((__m256i*)(s + 64));
			__m256i a = _mm256_permute2x128_si256(_a, _b, 0x30);
			__m256i b = _mm256_permute2x128_si256(_a, _c, 0x21);
			__m256i c = _mm256_permute2x128_si256(_b, _c, 0x30);
			a = _mm256_shuffle_epi8(a, mask1);
			b = _mm256_shuffle_epi8(b, mask2);
			c = _mm256_shuffle_epi8(c, mask2);
			_mm256_stream_si256((__m256i*)(B), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask1), bmask2));

			a = _mm256_shuffle_epi8(a, smask1);
			b = _mm256_shuffle_epi8(b, smask1);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_stream_si256((__m256i*)(G), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask2));

			a = _mm256_shuffle_epi8(a, ssmask1);
			b = _mm256_shuffle_epi8(b, ssmask2);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_stream_si256((__m256i*)(R), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask4));

			B += 32;
			G += 32;
			R += 32;
			s += 96;
		}

		const int f = size - rem;
		for (int i = 0; i < f; i++)
		{
			B[i] = s[3 * i];
			G[i] = s[3 * i + 1];
			R[i] = s[3 * i + 2];
		}
	}

	enum STORE_METHOD
	{
		STOREU,
		STORE,
		STREAM,
	};

	template<int store_method>
	void v_store32f(float* dst, __m256 src)
	{
		_mm256_storeu_ps(dst, src);
	}

	template<>
	void v_store32f<STOREU>(float* dst, __m256 src)
	{
		_mm256_storeu_ps(dst, src);
	}

	template<>
	void v_store32f<STORE>(float* dst, __m256 src)
	{
		_mm256_store_ps(dst, src);
	}

	template<>
	void v_store32f<STREAM>(float* dst, __m256 src)
	{
		_mm256_stream_ps(dst, src);
	}

	template<int store_method, typename srcType, typename dstType>
	void splitConvert_(Mat& src, vector<Mat>& dst, const int unroll = 4)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(1 <= unroll && unroll <= 4);

		srcType* s = src.ptr<srcType>();
		dstType* b = dst[0].ptr<dstType>();
		dstType* g = dst[1].ptr<dstType>();
		dstType* r = dst[2].ptr<dstType>();

		for (int i = 0; i < src.size().area(); i++)
		{
			b[i] = saturate_cast<dstType>(s[3 * i + 0]);
			g[i] = saturate_cast<dstType>(s[3 * i + 1]);
			r[i] = saturate_cast<dstType>(s[3 * i + 2]);
		}
	}

	template<int store_method>
	void splitConvert32F32F_(Mat& src, vector<Mat>& dst, const int unroll = 4)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(1 <= unroll && unroll <= 4);

		float* s = src.ptr   <float>();
		float* b = dst[0].ptr<float>();
		float* g = dst[1].ptr<float>();
		float* r = dst[2].ptr<float>();
		const int step = 8 * unroll;
		const int simdsize = get_simd_floor(src.size().area(), step);

		if (unroll == 1)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);
			}
		}
		else if (unroll == 2)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store32f<store_method>(b + i + 8, mb);
				v_store32f<store_method>(g + i + 8, mg);
				v_store32f<store_method>(r + i + 8, mr);
			}
		}
		else if (unroll == 3)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store32f<store_method>(b + i + 8, mb);
				v_store32f<store_method>(g + i + 8, mg);
				v_store32f<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store32f<store_method>(b + i + 16, mb);
				v_store32f<store_method>(g + i + 16, mg);
				v_store32f<store_method>(r + i + 16, mr);
			}
		}
		else if (unroll == 4)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store32f<store_method>(b + i + 8, mb);
				v_store32f<store_method>(g + i + 8, mg);
				v_store32f<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store32f<store_method>(b + i + 16, mb);
				v_store32f<store_method>(g + i + 16, mg);
				v_store32f<store_method>(r + i + 16, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 72, mb, mg, mr);
				v_store32f<store_method>(b + i + 24, mb);
				v_store32f<store_method>(g + i + 24, mg);
				v_store32f<store_method>(r + i + 24, mr);
			}
		}

		for (int i = simdsize; i < src.size().area(); i++)
		{
			b[i] = s[3 * i + 0];
			g[i] = s[3 * i + 1];
			r[i] = s[3 * i + 2];
		}
	}

	template<int store_method>
	void splitConvert8U32F_(Mat& src, vector<Mat>& dst, const int unroll = 4)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(1 <= unroll && unroll <= 4);

		uchar* s = src.ptr<uchar>();
		float* b = dst[0].ptr<float>();
		float* g = dst[1].ptr<float>();
		float* r = dst[2].ptr<float>();
		const int step = 8 * unroll;
		const int simdsize = get_simd_floor(src.size().area(), step);

		if (unroll == 1)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtepu8bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);
			}
		}
		else if (unroll == 2)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb0, mg0, mr0, mb1, mg1, mr1;
				_mm256_load_cvtepu8bgr2planar_psx2(s + 3 * i, mb0, mb1, mg0, mg1, mr0, mr1);
				v_store32f<store_method>(b + i, mb0);
				v_store32f<store_method>(b + i + 8, mb1);
				v_store32f<store_method>(g + i, mg0);
				v_store32f<store_method>(g + i + 8, mg1);
				v_store32f<store_method>(r + i, mr0);
				v_store32f<store_method>(r + i + 8, mr1);
			}
		}
		/*
		else if (unroll == 3)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store<store_method>(b + i, mb);
				v_store<store_method>(g + i, mg);
				v_store<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store<store_method>(b + i + 8, mb);
				v_store<store_method>(g + i + 8, mg);
				v_store<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store<store_method>(b + i + 16, mb);
				v_store<store_method>(g + i + 16, mg);
				v_store<store_method>(r + i + 16, mr);
			}
		}
		else if (unroll == 4)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store<store_method>(b + i, mb);
				v_store<store_method>(g + i, mg);
				v_store<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store<store_method>(b + i + 8, mb);
				v_store<store_method>(g + i + 8, mg);
				v_store<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store<store_method>(b + i + 16, mb);
				v_store<store_method>(g + i + 16, mg);
				v_store<store_method>(r + i + 16, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 72, mb, mg, mr);
				v_store<store_method>(b + i + 24, mb);
				v_store<store_method>(g + i + 24, mg);
				v_store<store_method>(r + i + 24, mr);
			}
		}

		for (int i = simdsize; i < src.size().area(); i++)
		{
			b[i] = float(s[3 * i + 0]);
			g[i] = float(s[3 * i + 1]);
			r[i] = float(s[3 * i + 2]);
		}
		*/
	}

	void splitConvert(cv::InputArray src, cv::OutputArrayOfArrays dest, const int depth, const double scale, const double offset, const bool isCache)
	{
		Mat s = src.getMat();

		vector<Mat> dst;
		if (dest.empty())
		{
			dest.create(3, 1, src.type());

			dest.getMatVector(dst);

			dst[0].create(src.size(), depth);
			dst[1].create(src.size(), depth);
			dst[2].create(src.size(), depth);

			dest.getMatRef(0) = dst[0];
			dest.getMatRef(1) = dst[1];
			dest.getMatRef(2) = dst[2];
		}
		else
		{
			dest.getMatVector(dst);

			for (int i = 0; i < dst.size(); i++)
			{
				if (dst[i].empty() ||
					dst[i].depth() != src.depth())
				{
					dst[i].create(src.size(), depth);
					dest.getMatRef(i) = dst[i];
				}
			}
		}


		if (src.depth() == CV_32F && depth == CV_32F)
		{
			const int unroll = 1;
			if (isCache && dst[0].cols % 8 == 0)splitConvert32F32F_<STORE>(s, dst, unroll);
			else if (isCache && dst[0].cols % 8 != 0)splitConvert32F32F_<STOREU>(s, dst, unroll);
			else splitConvert32F32F_<STREAM>(s, dst, unroll);
		}
		else if (src.depth() == CV_8U && depth == CV_32F)
		{
			const int unroll = 2;
			if (isCache && dst[0].cols % 8 == 0)splitConvert8U32F_<STORE>(s, dst, unroll);
			else if (isCache && dst[0].cols % 8 != 0)splitConvert8U32F_<STOREU>(s, dst, unroll);
			else splitConvert8U32F_<STREAM>(s, dst, unroll);
		}
		/*
		switch (s.depth())
		{
		case CV_8U:
			if (isCache)splitStore_8U(s, dst);
			else splitStream_8U(s, dst);
			break;
		case CV_32F:

			break;
		default:
			//splitBase_<float>(s, dst);
			cout << "not support type" << endl;
			break;
		}*/
	}


	template<int store_method, typename srcType, typename dstType>
	void splitConvertYCrCb_(Mat& src, vector<Mat>& dst, const int unroll = 4)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(1 <= unroll && unroll <= 4);

		srcType* s = src.ptr<srcType>();
		dstType* y_ = dst[0].ptr<dstType>();
		dstType* cr = dst[1].ptr<dstType>();
		dstType* cb = dst[2].ptr<dstType>();

		for (int i = 0; i < src.size().area(); i++)
		{
			y_[i] = saturate_cast<dstType>(0.114 * s[3 * i + 0] + 0.587 * s[3 * i + 1] + 0.299 * s[3 * i + 2]);
			cr[i] = saturate_cast<dstType>(0.5 * s[3 * i + 0] - 0.331264 * s[3 * i + 1] - 0.168736 * s[3 * i + 2]);
			cb[i] = saturate_cast<dstType>(-0.081312 * s[3 * i + 0] - 0.418688 * s[3 * i + 1] + 0.5 * s[3 * i + 2]);
		}
	}

	template<int store_method>
	void splitConvertYCrCb32F32F_(Mat& src, vector<Mat>& dst, const int unroll = 4)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(1 <= unroll && unroll <= 4);

		float* __restrict s = src.ptr   <float>();
		float* __restrict b = dst[0].ptr<float>();
		float* __restrict g = dst[1].ptr<float>();
		float* __restrict r = dst[2].ptr<float>();
		const int step = 8 * unroll;
		const int simdsize = get_simd_floor(src.size().area(), step);
		const __m256 cy0 = _mm256_set1_ps(0.114f);
		const __m256 cy1 = _mm256_set1_ps(0.587f);
		const __m256 cy2 = _mm256_set1_ps(0.299f);

		const __m256 ccr = _mm256_set1_ps(0.713f);
		const __m256 ccb = _mm256_set1_ps(0.564f);
		const __m256 m128 = _mm256_set1_ps(128.f);
		if (unroll == 1)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb0, mg0, mr0;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb0, mg0, mr0);
				const __m256 my0 = _mm256_fmadd_ps(cy0, mb0, _mm256_fmadd_ps(cy1, mg0, _mm256_mul_ps(cy2, mr0)));
				v_store32f<store_method>(b + i + 0, my0);
				v_store32f<store_method>(g + i + 0, _mm256_fmadd_ps(ccr, _mm256_sub_ps(mr0, my0), m128));
				v_store32f<store_method>(r + i + 0, _mm256_fmadd_ps(ccb, _mm256_sub_ps(mb0, my0), m128));
			}
		}
		else if (unroll == 2)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store32f<store_method>(b + i + 8, mb);
				v_store32f<store_method>(g + i + 8, mg);
				v_store32f<store_method>(r + i + 8, mr);
			}
		}
		else if (unroll == 3)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store32f<store_method>(b + i + 8, mb);
				v_store32f<store_method>(g + i + 8, mg);
				v_store32f<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store32f<store_method>(b + i + 16, mb);
				v_store32f<store_method>(g + i + 16, mg);
				v_store32f<store_method>(r + i + 16, mr);
			}
		}
		else if (unroll == 4)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store32f<store_method>(b + i, mb);
				v_store32f<store_method>(g + i, mg);
				v_store32f<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store32f<store_method>(b + i + 8, mb);
				v_store32f<store_method>(g + i + 8, mg);
				v_store32f<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store32f<store_method>(b + i + 16, mb);
				v_store32f<store_method>(g + i + 16, mg);
				v_store32f<store_method>(r + i + 16, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 72, mb, mg, mr);
				v_store32f<store_method>(b + i + 24, mb);
				v_store32f<store_method>(g + i + 24, mg);
				v_store32f<store_method>(r + i + 24, mr);
			}
		}

		for (int i = simdsize; i < src.size().area(); i++)
		{
			b[i] = saturate_cast<float>(0.114f * s[3 * i + 0] + 0.587f * s[3 * i + 1] + 0.299f * s[3 * i + 2]);
			g[i] = saturate_cast<float>(0.5f * s[3 * i + 0] - 0.331264f * s[3 * i + 1] - 0.168736f * s[3 * i + 2] + 128.f);
			r[i] = saturate_cast<float>(-0.081312f * s[3 * i + 0] - 0.418688f * s[3 * i + 1] + 0.5f * s[3 * i + 2] + 128.f);
		}
	}

	template<int store_method>
	void splitConvertYCrCb8U32F_(Mat& src, vector<Mat>& dst, const int unroll = 4)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(1 <= unroll && unroll <= 4);

		const uchar* __restrict s = src.ptr<uchar>();
		float* __restrict b = dst[0].ptr<float>();
		float* __restrict g = dst[1].ptr<float>();
		float* __restrict r = dst[2].ptr<float>();
		const int step = 8 * unroll;
		const int simdsize = get_simd_floor(src.size().area(), step);

		//y_[i] = saturate_cast<dstType>(0.114 * s[3 * i + 0] + 0.587 * s[3 * i + 1] + 0.299 * s[3 * i + 2]);
		//cr[i] = saturate_cast<dstType>(0.5 * s[3 * i + 0] - 0.331264 * s[3 * i + 1] - 0.168736 * s[3 * i + 2]);
		//cb[i] = saturate_cast<dstType>(-0.081312 * s[3 * i + 0] - 0.418688 * s[3 * i + 1] + 0.5 * s[3 * i + 2]);
		const __m256 cy0 = _mm256_set1_ps(0.114f);
		const __m256 cy1 = _mm256_set1_ps(0.587f);
		const __m256 cy2 = _mm256_set1_ps(0.299f);

		const __m256 ccr = _mm256_set1_ps(0.713f);
		const __m256 ccb = _mm256_set1_ps(0.564f);
		const __m256 m128 = _mm256_set1_ps(128.f);
		if (unroll == 1)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb0, mg0, mr0;
				_mm256_load_cvtepu8bgr2planar_ps(s + 3 * i, mb0, mr0, mg0);
				const __m256 my0 = _mm256_fmadd_ps(cy0, mb0, _mm256_fmadd_ps(cy1, mg0, _mm256_mul_ps(cy2, mr0)));
				v_store32f<store_method>(b + i + 0, my0);
				v_store32f<store_method>(g + i + 0, _mm256_fmadd_ps(ccr, _mm256_sub_ps(mr0, my0), m128));
				v_store32f<store_method>(r + i + 0, _mm256_fmadd_ps(ccb, _mm256_sub_ps(mb0, my0), m128));
			}
		}
		else if (unroll == 2)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb0, mg0, mr0, mb1, mg1, mr1;
				_mm256_load_cvtepu8bgr2planar_psx2(s + 3 * i, mb0, mb1, mg0, mg1, mr0, mr1);
				const __m256 my0 = _mm256_fmadd_ps(cy0, mb0, _mm256_fmadd_ps(cy1, mg0, _mm256_mul_ps(cy2, mr0)));
				const __m256 my1 = _mm256_fmadd_ps(cy0, mb1, _mm256_fmadd_ps(cy1, mg1, _mm256_mul_ps(cy2, mr1)));
				v_store32f<store_method>(b + i + 0, my0);
				v_store32f<store_method>(g + i + 0, _mm256_fmadd_ps(ccr, _mm256_sub_ps(mr0, my0), m128));
				v_store32f<store_method>(r + i + 0, _mm256_fmadd_ps(ccb, _mm256_sub_ps(mb0, my0), m128));
				v_store32f<store_method>(b + i + 8, my1);
				v_store32f<store_method>(g + i + 8, _mm256_fmadd_ps(ccr, _mm256_sub_ps(mr1, my1), m128));
				v_store32f<store_method>(r + i + 8, _mm256_fmadd_ps(ccb, _mm256_sub_ps(mb1, my1), m128));
			}
		}
		for (int i = simdsize; i < src.size().area(); i++)
		{
			const float y = saturate_cast<float>(0.114f * s[3 * i + 0] + 0.587f * s[3 * i + 1] + 0.299f * s[3 * i + 2]);
			b[i] = y;
			g[i] = saturate_cast<float>((s[3 * i + 2] - y) * 0.713f + 128.f);
			r[i] = saturate_cast<float>((s[3 * i + 0] - y) * 0.564f + 128.f);
		}
		/*
		else if (unroll == 3)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store<store_method>(b + i, mb);
				v_store<store_method>(g + i, mg);
				v_store<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store<store_method>(b + i + 8, mb);
				v_store<store_method>(g + i + 8, mg);
				v_store<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store<store_method>(b + i + 16, mb);
				v_store<store_method>(g + i + 16, mg);
				v_store<store_method>(r + i + 16, mr);
			}
		}
		else if (unroll == 4)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
				v_store<store_method>(b + i, mb);
				v_store<store_method>(g + i, mg);
				v_store<store_method>(r + i, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 24, mb, mg, mr);
				v_store<store_method>(b + i + 8, mb);
				v_store<store_method>(g + i + 8, mg);
				v_store<store_method>(r + i + 8, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 48, mb, mg, mr);
				v_store<store_method>(b + i + 16, mb);
				v_store<store_method>(g + i + 16, mg);
				v_store<store_method>(r + i + 16, mr);

				_mm256_load_cvtps_bgr2planar_ps(s + 3 * i + 72, mb, mg, mr);
				v_store<store_method>(b + i + 24, mb);
				v_store<store_method>(g + i + 24, mg);
				v_store<store_method>(r + i + 24, mr);
			}
		}

		for (int i = simdsize; i < src.size().area(); i++)
		{
			b[i] = float(s[3 * i + 0]);
			g[i] = float(s[3 * i + 1]);
			r[i] = float(s[3 * i + 2]);
		}
		*/
	}

	template<int store_method>
	void splitConvertYCrCb8U8U_(Mat& src, vector<Mat>& dst, const int unroll = 4)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(1 <= unroll && unroll <= 4);

		const uchar* __restrict s = src.ptr<uchar>();
		uchar* __restrict b = dst[0].ptr<uchar>();
		uchar* __restrict g = dst[1].ptr<uchar>();
		uchar* __restrict r = dst[2].ptr<uchar>();
		const int step = 8 * unroll;
		const int simdsize = get_simd_floor(src.size().area(), step);

		//y_[i] = saturate_cast<dstType>(0.114 * s[3 * i + 0] + 0.587 * s[3 * i + 1] + 0.299 * s[3 * i + 2]);
		//cr[i] = saturate_cast<dstType>(0.5 * s[3 * i + 0] - 0.331264 * s[3 * i + 1] - 0.168736 * s[3 * i + 2]);
		//cb[i] = saturate_cast<dstType>(-0.081312 * s[3 * i + 0] - 0.418688 * s[3 * i + 1] + 0.5 * s[3 * i + 2]);
		const __m256 cy0 = _mm256_set1_ps(0.114f);
		const __m256 cy1 = _mm256_set1_ps(0.587f);
		const __m256 cy2 = _mm256_set1_ps(0.299f);

		const __m256 ccr = _mm256_set1_ps(0.713f);
		const __m256 ccb = _mm256_set1_ps(0.564f);
		const __m256 m128 = _mm256_set1_ps(128.f);
		if (unroll == 1)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb0, mg0, mr0;
				_mm256_load_cvtepu8bgr2planar_ps(s + 3 * i, mb0, mr0, mg0);
				const __m256 my0 = _mm256_fmadd_ps(cy0, mb0, _mm256_fmadd_ps(cy1, mg0, _mm256_mul_ps(cy2, mr0)));
				_mm_storel_epi64((__m128i*)(b + i + 0), _mm256_cvtps_epu8(my0));
				_mm_storel_epi64((__m128i*)(g + i + 0), _mm256_cvtps_epu8(_mm256_fmadd_ps(ccr, _mm256_sub_ps(mr0, my0), m128)));
				_mm_storel_epi64((__m128i*)(r + i + 0), _mm256_cvtps_epu8(_mm256_fmadd_ps(ccb, _mm256_sub_ps(mb0, my0), m128)));
			}
		}
		else if (unroll == 2)
		{
			for (int i = 0; i < simdsize; i += step)
			{
				__m256 mb0, mg0, mr0, mb1, mg1, mr1;
				_mm256_load_cvtepu8bgr2planar_psx2(s + 3 * i, mb0, mb1, mg0, mg1, mr0, mr1);
				const __m256 my0 = _mm256_fmadd_ps(cy0, mb0, _mm256_fmadd_ps(cy1, mg0, _mm256_mul_ps(cy2, mr0)));
				const __m256 my1 = _mm256_fmadd_ps(cy0, mb1, _mm256_fmadd_ps(cy1, mg1, _mm256_mul_ps(cy2, mr1)));
				_mm_storel_epi64((__m128i*)(b + i + 0), _mm256_cvtps_epu8(my0));
				_mm_storel_epi64((__m128i*)(g + i + 0), _mm256_cvtps_epu8(_mm256_fmadd_ps(ccr, _mm256_sub_ps(mr0, my0), m128)));
				_mm_storel_epi64((__m128i*)(r + i + 0), _mm256_cvtps_epu8(_mm256_fmadd_ps(ccb, _mm256_sub_ps(mb0, my0), m128)));
				_mm_storel_epi64((__m128i*)(b + i + 8), _mm256_cvtps_epu8(my1));
				_mm_storel_epi64((__m128i*)(g + i + 8), _mm256_cvtps_epu8(_mm256_fmadd_ps(ccr, _mm256_sub_ps(mr1, my1), m128)));
				_mm_storel_epi64((__m128i*)(r + i + 8), _mm256_cvtps_epu8(_mm256_fmadd_ps(ccb, _mm256_sub_ps(mb1, my1), m128)));
			}
		}
		for (int i = simdsize; i < src.size().area(); i++)
		{
			const float y = saturate_cast<float>(0.114f * s[3 * i + 0] + 0.587f * s[3 * i + 1] + 0.299f * s[3 * i + 2]);
			b[i] = saturate_cast<uchar>(y);
			g[i] = saturate_cast<uchar>((s[3 * i + 2] - y) * 0.713f + 128.f);
			r[i] = saturate_cast<uchar>((s[3 * i + 0] - y) * 0.564f + 128.f);
		}
	}

	void splitConvertYCrCb(cv::InputArray src, cv::OutputArrayOfArrays dest, const int depth, const double scale, const double offset, const bool isCache)
	{
		Mat s = src.getMat();

		vector<Mat> dst;
		if (dest.empty() || src.size() != dest.size())
		{
			dest.create(3, 1, src.type());

			dest.getMatVector(dst);

			dst[0].create(src.size(), depth);
			dst[1].create(src.size(), depth);
			dst[2].create(src.size(), depth);

			dest.getMatRef(0) = dst[0];
			dest.getMatRef(1) = dst[1];
			dest.getMatRef(2) = dst[2];
		}
		else
		{
			dest.getMatVector(dst);

			for (int i = 0; i < dst.size(); i++)
			{
				if (dst[i].empty() ||
					dst[i].depth() != src.depth())
				{
					dst[i].create(src.size(), depth);
					dest.getMatRef(i) = dst[i];
				}
			}
		}

		if (src.depth() == CV_32F && depth == CV_32F)
		{
			const int unroll = 1;
			if (isCache && dst[0].cols % 8 == 0)splitConvertYCrCb32F32F_<STORE>(s, dst, unroll);
			else if (isCache && dst[0].cols % 8 != 0)splitConvertYCrCb32F32F_<STOREU>(s, dst, unroll);
			else splitConvertYCrCb32F32F_<STREAM>(s, dst, unroll);
		}
		else if (src.depth() == CV_8U && depth == CV_32F)
		{
			const int unroll = 2;
			if (isCache && dst[0].cols % 8 == 0)splitConvertYCrCb8U32F_<STORE>(s, dst, unroll);
			else if (isCache && dst[0].cols % 8 != 0)splitConvertYCrCb8U32F_<STOREU>(s, dst, unroll);
			else splitConvertYCrCb8U32F_<STREAM>(s, dst, unroll);
		}
		else if (src.depth() == CV_8U && depth == CV_8U)
		{
			const int unroll = 2;
			if (isCache && dst[0].cols % 8 == 0)splitConvertYCrCb8U8U_<STORE>(s, dst, unroll);
			else if (isCache && dst[0].cols % 8 != 0)splitConvertYCrCb8U8U_<STOREU>(s, dst, unroll);
			else splitConvertYCrCb8U8U_<STREAM>(s, dst, unroll);
		}
		/*
		switch (s.depth())
		{
		case CV_8U:
			if (isCache)splitStore_8U(s, dst);
			else splitStream_8U(s, dst);
			break;
		case CV_32F:

			break;
		default:
			//splitBase_<float>(s, dst);
			cout << "not support type" << endl;
			break;
		}*/
	}

#pragma endregion


#pragma region convert
	void cvtBGR2RawVector(cv::InputArray src, vector<float>& dest)
	{
		if (dest.size() < src.size().area() * src.channels())dest.resize(src.size().area() * src.channels());
		vector<Mat> v(3);
		split(src, v);

		int sz = src.size().area();
		float* d = &dest[0];
		memcpy(d, v[2].data, sizeof(float) * sz);
		memcpy(d + sz, v[1].data, sizeof(float) * sz);
		memcpy(d + 2 * sz, v[0].data, sizeof(float) * sz);
	}

	void cvtRAWVector2BGR(vector<float>& src, OutputArray dest, Size size)
	{
		vector<Mat> v(3);

		Mat a(size, CV_32F);
		Mat b(size, CV_32F);
		Mat c(size, CV_32F);


		int sz = size.area();
		float* s = &src[0];
		memcpy(c.data, s, sizeof(float) * sz);
		memcpy(b.data, s + size.area(), sizeof(float) * sz);
		memcpy(a.data, s + 2 * size.area(), sizeof(float) * sz);

		v[0] = a;
		v[1] = b;
		v[2] = c;

		merge(v, dest);
	}

#pragma region splitLineInterleave
	//8u
	void splitBGRLineInterleave_8u(const Mat& src, Mat& dest)
	{
		const int size = src.size().area();
		const int dstep = src.cols * 3;
		const int sstep = src.cols * 3;

		const uchar* s = src.ptr<uchar>(0);
		uchar* B = dest.ptr<uchar>(0);//line by line interleave
		uchar* G = dest.ptr<uchar>(1);
		uchar* R = dest.ptr<uchar>(2);

		//BGR BGR BGR BGR BGR B
		//GR BGR BGR BGR BGR BG
		//R BGR BGR BGR BGR BGR
		//BBBBBBGGGGGRRRRR shuffle
		const __m128i mask1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
		//GGGGGBBBBBBRRRRR shuffle
		const __m128i smask1 = _mm_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
		const __m128i ssmask1 = _mm_setr_epi8(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

		//GGGGGGBBBBBRRRRR shuffle
		const __m128i mask2 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
		//const __m128i smask2 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
		const __m128i ssmask2 = _mm_setr_epi8(0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10);

		//RRRRRRGGGGGBBBBB shuffle -> same mask2
		//__m128i mask3 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);

		//const __m128i smask3 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,6,7,8,9,10);
		//const __m128i ssmask3 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

		const __m128i bmask1 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		const __m128i bmask2 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0);

		const __m128i bmask3 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		const __m128i bmask4 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0);

		__m128i a, b, c;

		const int simd_end = get_simd_floor(src.cols, 16);
		for (int j = 0; j < src.rows; j++)
		{
			int i = 0;
			for (; i < simd_end; i += 16)
			{
				a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i)), mask1);
				b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i + 16)), mask2);
				c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i + 32)), mask2);
				_mm_storeu_si128((__m128i*)(B + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask1), bmask2));

				a = _mm_shuffle_epi8(a, smask1);
				b = _mm_shuffle_epi8(b, smask1);
				c = _mm_shuffle_epi8(c, ssmask1);
				_mm_storeu_si128((__m128i*)(G + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask2));

				a = _mm_shuffle_epi8(a, ssmask1);
				c = _mm_shuffle_epi8(c, ssmask1);
				b = _mm_shuffle_epi8(b, ssmask2);

				_mm_storeu_si128((__m128i*)(R + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask4));
			}
			for (; i < src.cols; i++)
			{
				B[i] = s[3 * i + 0];
				G[i] = s[3 * i + 1];
				R[i] = s[3 * i + 2];
			}
			R += dstep;
			G += dstep;
			B += dstep;
			s += sstep;
		}
	}

	void splitBGRLineInterleave_32f(const Mat& src, Mat& dest)
	{
		const int size = src.size().area();
		const int dstep = src.cols * 3;
		const int sstep = src.cols * 3;

		const float* s = src.ptr<float>(0);
		float* B = dest.ptr<float>(0);//line by line interleave
		float* G = dest.ptr<float>(1);
		float* R = dest.ptr<float>(2);

		for (int j = 0; j < src.rows; j++)
		{
			int i = 0;
			for (; i < src.cols; i += 4)
			{
				__m128 a = _mm_load_ps((s + 3 * i));
				__m128 b = _mm_load_ps((s + 3 * i + 4));
				__m128 c = _mm_load_ps((s + 3 * i + 8));

				__m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
				aa = _mm_blend_ps(aa, b, 4);
				__m128 cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
				aa = _mm_blend_ps(aa, cc, 8);
				_mm_stream_ps((B + i), aa);

				aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
				__m128 bb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
				bb = _mm_blend_ps(bb, aa, 1);
				cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
				bb = _mm_blend_ps(bb, cc, 8);
				_mm_stream_ps((G + i), bb);

				aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
				bb = _mm_blend_ps(aa, b, 2);
				cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
				cc = _mm_blend_ps(bb, cc, 12);
				_mm_stream_ps((R + i), cc);

			}
			R += dstep;
			G += dstep;
			B += dstep;
			s += sstep;
		}
	}

	void splitBGRLineInterleave_32fcast(const Mat& src, Mat& dest)
	{
		Mat a, b;
		src.convertTo(a, CV_32F);
		splitBGRLineInterleave_32f(a, b);
		b.convertTo(dest, src.type());
	}

	void splitBGRLineInterleaveAVX_64f(const Mat& src, Mat& dest)
	{
		const int size = src.size().area();
		const int dstep = src.cols * 3;
		const int sstep = src.cols * 3;

		const double* s = src.ptr<double>(0);
		double* B = dest.ptr<double>(0);//line by line interleave
		double* G = dest.ptr<double>(1);
		double* R = dest.ptr<double>(2);

		for (int j = 0; j < src.rows; j++)
		{
			int i = 0;
			for (; i < src.cols; i += 4)
			{
				const __m256d aa = _mm256_load_pd((s + 3 * i));
				const __m256d bb = _mm256_load_pd((s + 3 * i + 4));
				const __m256d cc = _mm256_load_pd((s + 3 * i + 8));

#if CP_AVX2
				__m256d a = _mm256_blend_pd(aa, bb, 0b0110);
				__m256d b = _mm256_blend_pd(a, cc, 0b0010);
				__m256d c = _mm256_permute4x64_pd(b, 0b01101100);
				_mm256_stream_pd((B + i), c);

				a = _mm256_blend_pd(aa, bb, 0b1001);
				b = _mm256_blend_pd(a, cc, 0b0100);
				c = _mm256_permute4x64_pd(b, 0b10110001);
				_mm256_stream_pd((G + i), c);

				a = _mm256_blend_pd(aa, bb, 0b1011);
				b = _mm256_blend_pd(a, cc, 0b1001);
				c = _mm256_permute4x64_pd(b, 0b11000110);
				_mm256_stream_pd((R + i), c);
#else
				__m256d a = _mm256_blend_pd(_mm256_permute2f128_pd(aa, aa, 0b00000001), aa, 0b0001);
				__m256d b = _mm256_blend_pd(_mm256_permute2f128_pd(cc, cc, 0b00000000), bb, 0b0100);
				__m256d c = _mm256_blend_pd(a, b, 0b1100);
				_mm256_stream_pd((B + i), c);

				a = _mm256_blend_pd(aa, bb, 0b1001);
				b = _mm256_blend_pd(a, cc, 0b0100);
				c = _mm256_permute_pd(b, 0b0101);
				_mm256_stream_pd((G + i), c);

				a = _mm256_blend_pd(_mm256_permute2f128_pd(aa, aa, 0b0001), bb, 0b0010);
				b = _mm256_blend_pd(_mm256_permute2f128_pd(cc, cc, 0b0001), cc, 0b1000);
				c = _mm256_blend_pd(a, b, 0b1100);
				_mm256_stream_pd((R + i), c);
#endif
			}
			R += dstep;
			G += dstep;
			B += dstep;
			s += sstep;
		}
	}

	void splitBGRLineInterleave(cv::InputArray src_, cv::OutputArray dest_)
	{
		dest_.create(Size(src_.size().width, src_.size().height * 3), src_.depth());
		Mat src = src_.getMat();
		Mat dest = dest_.getMat();
		if (src.type() == CV_8UC3)
		{
			CV_Assert(src.cols % 32 == 0);
			splitBGRLineInterleave_8u(src, dest);
		}
		else if (src.type() == CV_32FC3)
		{
			CV_Assert(src.cols % 8 == 0);
			splitBGRLineInterleave_32f(src, dest);
		}
		else if (src.type() == CV_64FC3)
		{
			CV_Assert(src.cols % 4 == 0);
			splitBGRLineInterleaveAVX_64f(src, dest);
		}
		else
		{
			CV_Assert(src.cols % 8 == 0);
			splitBGRLineInterleave_32fcast(src, dest);
		}
	}
#pragma endregion


	void cvtColorBGR2PLANE_8u(const Mat& src, Mat& dest)
	{
		dest.create(Size(src.cols, src.rows * 3), CV_8U);

		const int size = src.size().area();
		const int ssesize = 3 * size - ((48 - (3 * size) % 48) % 48);
		const int ssecount = ssesize / 48;
		const uchar* s = src.ptr<uchar>(0);
		uchar* B = dest.ptr<uchar>(0);//line by line interleave
		uchar* G = dest.ptr<uchar>(src.rows);
		uchar* R = dest.ptr<uchar>(2 * src.rows);

		//BGR BGR BGR BGR BGR B	-> GGGGG RRRRR BBBBBB
		//GR BGR BGR BGR BGR BG -> GGGGGG RRRRR BBBBB
		//R BGR BGR BGR BGR BGR -> BBBBB GGGGG RRRRR

		const __m128i mask0 = _mm_setr_epi8(1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 0, 3, 6, 9, 12, 15);
		const __m128i mask1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);

		__m128i a, b, c, d, e;

		for (int i = 0; i < ssecount; i++)
		{
			a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 0)), mask0);
			b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 16)), mask1);
			c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 32)), mask0);

			d = _mm_alignr_epi8(c, b, 11);
			e = _mm_alignr_epi8(d, a, 10);
			_mm_storeu_si128((__m128i*)(B), e);

			d = _mm_alignr_epi8(_mm_srli_si128(c, 5), _mm_slli_si128(b, 10), 10);
			e = _mm_alignr_epi8(d, _mm_slli_si128(a, 11), 11);
			_mm_storeu_si128((__m128i*)(G), e);

			d = _mm_alignr_epi8(_mm_srli_si128(c, 10), _mm_slli_si128(b, 5), 11);
			e = _mm_alignr_epi8(d, _mm_slli_si128(a, 6), 11);
			_mm_storeu_si128((__m128i*)(R), e);

			s += 48;
			R += 16;
			G += 16;
			B += 16;
		}
		for (int i = ssesize; i < 3 * size; i += 3)
		{
			B[0] = s[0];
			G[0] = s[1];
			R[0] = s[2];
			s += 3, R++, G++, B++;
		}
	}

	void cvtColorBGR2PLANE_32f(const Mat& src, Mat& dest)
	{
		const int size = src.size().area();
		const int ssesize = 3 * size - ((12 - (3 * size) % 12) % 12);
		const int ssecount = ssesize / 12;
		const float* s = src.ptr<float>(0);
		float* B = dest.ptr<float>(0);//line by line interleave
		float* G = dest.ptr<float>(src.rows);
		float* R = dest.ptr<float>(2 * src.rows);

		for (int i = 0; i < ssecount; i++)
		{
			__m128 a = _mm_load_ps(s);
			__m128 b = _mm_load_ps(s + 4);
			__m128 c = _mm_load_ps(s + 8);

			__m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
			aa = _mm_blend_ps(aa, b, 4);
			__m128 cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
			aa = _mm_blend_ps(aa, cc, 8);
			_mm_storeu_ps((B), aa);

			aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
			__m128 bb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
			bb = _mm_blend_ps(bb, aa, 1);
			cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
			bb = _mm_blend_ps(bb, cc, 8);
			_mm_storeu_ps((G), bb);

			aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
			bb = _mm_blend_ps(aa, b, 2);
			cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
			cc = _mm_blend_ps(bb, cc, 12);
			_mm_storeu_ps((R), cc);

			s += 12;
			R += 4;
			G += 4;
			B += 4;
		}
		for (int i = ssesize; i < 3 * size; i += 3)
		{
			B[0] = s[0];
			G[0] = s[1];
			R[0] = s[2];
			s += 3, R++, G++, B++;
		}
	}

	template <class srcType>
	void cvtColorBGR2PLANE_(const Mat& src, Mat& dest, int depth)
	{
		vector<Mat> v(3);
		split(src, v);
		dest.create(Size(src.cols, src.rows * 3), depth);

		memcpy(dest.data, v[0].data, src.size().area() * sizeof(srcType));
		memcpy(dest.data + src.size().area() * sizeof(srcType), v[1].data, src.size().area() * sizeof(srcType));
		memcpy(dest.data + 2 * src.size().area() * sizeof(srcType), v[2].data, src.size().area() * sizeof(srcType));
	}

	void cvtColorBGR2PLANE(cv::InputArray src_, cv::OutputArray dest_)
	{
		CV_Assert(src_.channels() == 3);

		Mat src = src_.getMat();
		dest_.create(Size(src.cols, src.rows * 3), src.depth());
		Mat dest = dest_.getMat();

		if (src.depth() == CV_8U)
		{
			//cvtColorBGR2PLANE_<uchar>(src, dest, CV_8U);
			cvtColorBGR2PLANE_8u(src, dest);
		}
		else if (src.depth() == CV_16U)
		{
			cvtColorBGR2PLANE_<ushort>(src, dest, CV_16U);
		}
		if (src.depth() == CV_16S)
		{
			cvtColorBGR2PLANE_<short>(src, dest, CV_16S);
		}
		if (src.depth() == CV_32S)
		{
			cvtColorBGR2PLANE_<int>(src, dest, CV_32S);
		}
		if (src.depth() == CV_32F)
		{
			//cvtColorBGR2PLANE_<float>(src, dest, CV_32F);
			cvtColorBGR2PLANE_32f(src, dest);
		}
		if (src.depth() == CV_64F)
		{
			cvtColorBGR2PLANE_<double>(src, dest, CV_64F);
		}
	}

	template <class srcType>
	void cvtColorPLANE2BGR_(const Mat& src, Mat& dest, int depth)
	{
		int width = src.cols;
		int height = src.rows / 3;
		srcType* b = (srcType*)src.ptr<srcType>(0);
		srcType* g = (srcType*)src.ptr<srcType>(height);
		srcType* r = (srcType*)src.ptr<srcType>(2 * height);

		Mat B(height, width, src.type(), b);
		Mat G(height, width, src.type(), g);
		Mat R(height, width, src.type(), r);
		vector<Mat> v(3);
		v[0] = B;
		v[1] = G;
		v[2] = R;
		merge(v, dest);
	}

	void cvtColorPLANE2BGR_8u_align(const Mat& src, Mat& dest)
	{
		int width = src.cols;
		int height = src.rows / 3;

		if (dest.empty()) dest.create(Size(width, height), CV_8UC3);
		else if (width != dest.cols || height != dest.rows) dest.create(Size(width, height), CV_8UC3);
		else if (dest.type() != CV_8UC3) dest.create(Size(width, height), CV_8UC3);

		uchar* B = (uchar*)src.ptr<uchar>(0);
		uchar* G = (uchar*)src.ptr<uchar>(height);
		uchar* R = (uchar*)src.ptr<uchar>(2 * height);

		uchar* D = (uchar*)dest.ptr<uchar>(0);

		int ssecount = width * height * 3 / 48;

		const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
		const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
		const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
		const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
		const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

		for (int i = ssecount; i--;)
		{
			__m128i a = _mm_load_si128((const __m128i*)B);
			__m128i b = _mm_load_si128((const __m128i*)G);
			__m128i c = _mm_load_si128((const __m128i*)R);

			a = _mm_shuffle_epi8(a, mask1);
			b = _mm_shuffle_epi8(b, mask2);
			c = _mm_shuffle_epi8(c, mask3);
			_mm_stream_si128((__m128i*)(D), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
			_mm_stream_si128((__m128i*)(D + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
			_mm_stream_si128((__m128i*)(D + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));

			D += 48;
			B += 16;
			G += 16;
			R += 16;
		}
	}

	void cvtColorPLANE2BGR_8u(const Mat& src, Mat& dest)
	{
		int width = src.cols;
		int height = src.rows / 3;

		if (dest.empty()) dest.create(Size(width, height), CV_8UC3);
		else if (width != dest.cols || height != dest.rows) dest.create(Size(width, height), CV_8UC3);
		else if (dest.type() != CV_8UC3) dest.create(Size(width, height), CV_8UC3);

		uchar* B = (uchar*)src.ptr<uchar>(0);
		uchar* G = (uchar*)src.ptr<uchar>(height);
		uchar* R = (uchar*)src.ptr<uchar>(2 * height);

		uchar* D = (uchar*)dest.ptr<uchar>(0);

		int ssecount = width * height * 3 / 48;
		int rem = width * height * 3 - ssecount * 48;

		const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
		const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
		const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
		const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
		const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

		for (int i = ssecount; i--;)
		{
			__m128i a = _mm_loadu_si128((const __m128i*)B);
			__m128i b = _mm_loadu_si128((const __m128i*)G);
			__m128i c = _mm_loadu_si128((const __m128i*)R);

			a = _mm_shuffle_epi8(a, mask1);
			b = _mm_shuffle_epi8(b, mask2);
			c = _mm_shuffle_epi8(c, mask3);

			_mm_storeu_si128((__m128i*)(D), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
			_mm_storeu_si128((__m128i*)(D + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
			_mm_storeu_si128((__m128i*)(D + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));

			D += 48;
			B += 16;
			G += 16;
			R += 16;
		}
		for (int i = rem; i--;)
		{
			D[0] = *B;
			D[1] = *G;
			D[2] = *R;
			D += 3;
			B++, G++, R++;
		}
	}

	void cvtColorPLANE2BGR(cv::InputArray src_, cv::OutputArray dest_)
	{
		CV_Assert(src_.channels() == 1);
		Mat src = src_.getMat();
		if (dest_.empty())dest_.create(Size(src.cols, src.rows), CV_MAKETYPE(src.depth(), 3));
		Mat dest = dest_.getMat();

		if (src.depth() == CV_8U)
		{
			//cvtColorPLANE2BGR_<uchar>(src, dest, CV_8U);	
			if (src.cols % 16 == 0)
				cvtColorPLANE2BGR_8u_align(src, dest);
			else
				cvtColorPLANE2BGR_8u(src, dest);
		}
		else if (src.depth() == CV_16U)
		{
			cvtColorPLANE2BGR_<ushort>(src, dest, CV_16U);
		}
		if (src.depth() == CV_16S)
		{
			cvtColorPLANE2BGR_<short>(src, dest, CV_16S);
		}
		if (src.depth() == CV_32S)
		{
			cvtColorPLANE2BGR_<int>(src, dest, CV_32S);
		}
		if (src.depth() == CV_32F)
		{
			cvtColorPLANE2BGR_<float>(src, dest, CV_32F);
		}
		if (src.depth() == CV_64F)
		{
			cvtColorPLANE2BGR_<double>(src, dest, CV_64F);
		}
	}

	void cvtColorBGR8u2BGRA32f(const Mat& src, Mat& dest, const float alpha)
	{
		if (dest.empty()) dest.create(src.size(), CV_32FC4);

		int size = src.size().area();
		uchar* s = (uchar*)src.ptr<uchar>(0);
		float* d = dest.ptr<float>(0);

		for (int i = 0; i < size; i++)
		{
			*d++ = *s++;
			*d++ = *s++;
			*d++ = *s++;
			*d++ = alpha;
		}
	}

	template<typename T>
	void cvtColorBGR2BGRA_(const Mat& src, Mat& dest, const T a)
	{
		const int size = src.size().area();
		const T* s = src.ptr<T>();
		T* d = dest.ptr<T>();

		for (int i = 0; i < size; i++)
		{
			*d++ = *s++;
			*d++ = *s++;
			*d++ = *s++;
			*d++ = a;
		}
	}

	void cvtColorBGR2BGRA(InputArray src_, OutputArray dest_, const double alpha)
	{
		Mat src = src_.getMat();
		dest_.create(src.size(), CV_MAKETYPE(src.depth(), 4));
		Mat dest = dest_.getMat();

		if (src.depth() == CV_8U)  cvtColorBGR2BGRA_<uchar>(src, dest, (uchar)alpha);
		if (src.depth() == CV_8S)  cvtColorBGR2BGRA_<char>(src, dest, (char)alpha);
		if (src.depth() == CV_16U) cvtColorBGR2BGRA_<ushort>(src, dest, (ushort)alpha);
		if (src.depth() == CV_16S) cvtColorBGR2BGRA_<short>(src, dest, (short)alpha);
		if (src.depth() == CV_32S) cvtColorBGR2BGRA_<int>(src, dest, (int)alpha);
		if (src.depth() == CV_32F) cvtColorBGR2BGRA_<float>(src, dest, (float)alpha);
		if (src.depth() == CV_64F) cvtColorBGR2BGRA_<double>(src, dest, alpha);
	}

	void cvtColorBGRA32f2BGR8u(const Mat& src, Mat& dest)
	{
		CV_Assert(src.type() == CV_32FC4);
		if (dest.empty())dest.create(src.size(), CV_8UC3);

		int size = src.size().area();
		float* s = (float*)src.ptr<float>(0);
		uchar* d = dest.ptr<uchar>(0);

		for (int i = 0; i < size; i++)
		{
			*d++ = saturate_cast<uchar>(*s++ + 0.5f);
			*d++ = saturate_cast<uchar>(*s++ + 0.5f);
			*d++ = saturate_cast<uchar>(*s++ + 0.5f);
			*s++;
		}
	}

	template<typename T>
	void cvtColorBGRA2BGR_(const Mat& src, Mat& dest)
	{
		const int size = src.size().area();
		const T* s = src.ptr<T>(0);
		T* d = dest.ptr<T>();

		for (int i = 0; i < size; i++)
		{
			*d++ = *s++;
			*d++ = *s++;
			*d++ = *s++;
			*s++;
		}
	}

	void cvtColorBGRA2BGR(InputArray src_, OutputArray dest_)
	{
		Mat src = src_.getMat();
		dest_.create(src.size(), CV_MAKETYPE(src.depth(), 3));
		Mat dest = dest_.getMat();

		if (src.depth() == CV_8U) cvtColorBGRA2BGR_<uchar>(src, dest);
		if (src.depth() == CV_8S) cvtColorBGRA2BGR_<char>(src, dest);
		if (src.depth() == CV_16U) cvtColorBGRA2BGR_<ushort>(src, dest);
		if (src.depth() == CV_16S) cvtColorBGRA2BGR_<short>(src, dest);
		if (src.depth() == CV_32S) cvtColorBGRA2BGR_<int>(src, dest);
		if (src.depth() == CV_32F) cvtColorBGRA2BGR_<float>(src, dest);
		if (src.depth() == CV_64F) cvtColorBGRA2BGR_<double>(src, dest);
	}

	void makemultichannel(Mat& gray, Mat& color)
	{
		Mat src;
		int channel = 9;
		color.create(gray.size(), CV_8UC(channel));
		copyMakeBorder(gray, src, 1, 1, 1, 1, BORDER_REPLICATE);
		//copyMakeBorder(gray,src,0,0,0,0,BORDER_REPLICATE);

		for (int j = 0; j < gray.rows; j++)
		{
			uchar* s = src.ptr(j + 1); s++;
			uchar* d = color.ptr(j);

			for (int i = 0; i < gray.cols; i++)
			{
				//d[channel*i+0]=s[i];
				//d[channel*i+1]=s[i-1];
				//d[channel*i+2]=s[i+1];

				//d[channel*i+3]=s[i-src.cols];
				//d[channel*i+4]=s[i-src.cols-1];
				//d[channel*i+5]=s[i-src.cols+1];

				//d[channel*i+6]=s[i+src.cols];
				//d[channel*i+7]=s[i+src.cols-1];
				//d[channel*i+8]=s[i+src.cols+1];

				d[channel * i + 0] = s[i];
				d[channel * i + 1] = abs(s[i] - s[i - 1]);
				d[channel * i + 2] = abs(s[i] - s[i + 1]);

				d[channel * i + 3] = abs(s[i] - s[i - src.cols]);
				d[channel * i + 4] = abs(s[i] - s[i - src.cols - 1]);
				d[channel * i + 5] = abs(s[i] - s[i - src.cols + 1]);

				d[channel * i + 6] = abs(s[i] - s[i + src.cols]);
				d[channel * i + 7] = abs(s[i] - s[i + src.cols - 1]);
				d[channel * i + 8] = abs(s[i] - s[i + src.cols + 1]);
			}
		}
	}


	void DecorrelateColorInvert(float* src, float* dest, int width, int height)
	{
		const float c00 = 0.57735025882720947265625f;
		const float c01 = 0.70710676908493041992187f;
		const float c02 = 0.40824830532073974609375f;
		const float c12 = -0.8164966106414794921875f;

		const int size1 = width * height;
		const int size2 = 2 * size1;

		//#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			float* s0 = src + width * j;
			float* s1 = s0 + size1;
			float* s2 = s0 + size2;
			float* d0 = dest + width * j;
			float* d1 = d0 + size1;
			float* d2 = d0 + size2;

			const __m128 mc00 = _mm_set1_ps(c00);
			const __m128 mc01 = _mm_set1_ps(c01);
			const __m128 mc02 = _mm_set1_ps(c02);
			const __m128 mc12 = _mm_set1_ps(c12);
			int i = 0;
			//#ifdef _SSE
			for (i = 0; i < width - 4; i += 4)
			{
				__m128 ms0 = _mm_load_ps(s0);
				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(s2);

				__m128 cs000 = _mm_mul_ps(mc00, ms0);
				__m128 cs002 = _mm_add_ps(cs000, _mm_mul_ps(mc02, ms2));
				_mm_store_ps(d0, _mm_add_ps(cs002, _mm_mul_ps(mc01, ms1)));
				_mm_store_ps(d1, _mm_add_ps(cs000, _mm_mul_ps(mc12, ms2)));
				_mm_store_ps(d2, _mm_sub_ps(cs002, _mm_mul_ps(mc01, ms1)));

				d0 += 4, d1 += 4, d2 += 4, s0 += 4, s1 += 4, s2 += 4;
			}
			//#endif
			for (; i < width; i++)
			{
				float v0 = c00 * *s0 + c01 * *s1 + c02 * *s2;
				float v1 = c00 * *s0 + c12 * *s2;
				float v2 = c00 * *s0 - c01 * *s1 + c02 * *s2;

				*d0++ = v0;
				*d1++ = v1;
				*d2++ = v2;
				s0++, s1++, s2++;
			}
		}
	}

	void DecorrelateColorForward(float* src, float* dest, int width,
		int height)
	{
		const float c0 = 0.57735025882720947265625f;
		const float c1 = 0.70710676908493041992187f;
		const float c20 = 0.40824830532073974609375f;
		const float c21 = -0.8164966106414794921875f;

		const int size1 = width * height;
		const int size2 = 2 * size1;


		//#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			float* s0 = src + width * j;
			float* s1 = s0 + size1;
			float* s2 = s0 + size2;
			float* d0 = dest + width * j;
			float* d1 = d0 + size1;
			float* d2 = d0 + size2;
			const __m128 mc0 = _mm_set1_ps(c0);
			const __m128 mc1 = _mm_set1_ps(c1);
			const __m128 mc20 = _mm_set1_ps(c20);
			const __m128 mc21 = _mm_set1_ps(c21);
			int i = 0;
			//#ifdef _SSE
			for (i = 0; i < width - 4; i += 4)
			{
				__m128 ms0 = _mm_load_ps(s0);
				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(s2);

				__m128 ms02a = _mm_add_ps(ms0, ms2);

				_mm_store_ps(d0, _mm_mul_ps(mc0, _mm_add_ps(ms1, ms02a)));
				_mm_store_ps(d1, _mm_mul_ps(mc1, _mm_sub_ps(ms0, ms2)));
				_mm_store_ps(d2, _mm_add_ps(_mm_mul_ps(mc20, ms02a), _mm_mul_ps(mc21, ms1)));

				d0 += 4, d1 += 4, d2 += 4, s0 += 4, s1 += 4, s2 += 4;
			}
			//#endif
			for (; i < width; i++)
			{
				float v0 = c0 * (*s0 + *s1 + *s2);
				float v1 = c1 * (*s0 - *s2);
				float v2 = (*s0 + *s2) * c20 + *s1 * c21;

				*d0++ = v0;
				*d1++ = v1;
				*d2++ = v2;
				s0++, s1++, s2++;
			}
		}
	}

	void cvtColorBGR2OPP(InputArray src, OutputArray dest)
	{
		vector<Mat> v(3);
		Mat dst(Size(src.size().width, src.size().height * 3), CV_32F);
		Mat a(src.size(), CV_32F, dst.ptr<float>(0));
		Mat b(src.size(), CV_32F, dst.ptr<float>(src.size().height));
		Mat c(src.size(), CV_32F, dst.ptr<float>(2 * src.size().height));
		v[0] = a;
		v[1] = b;
		v[2] = c;
		split(src, v);

		DecorrelateColorForward((float*)dst.data, (float*)dst.data, src.size().width, src.size().height);

		Size size_ = src.size();
		Mat aa(size_, CV_32F, (float*)dst.ptr<float>(0));
		Mat bb(size_, CV_32F, (float*)dst.ptr<float>(size_.height));
		Mat cc(size_, CV_32F, (float*)dst.ptr<float>(2 * size_.height));

		v[0] = aa;
		v[1] = bb;
		v[2] = cc;

		merge(v, dest);
	}

	void cvtColorOPP2BGR(InputArray src, OutputArray dest)
	{
		CV_Assert(src.depth() == CV_32F);

		vector<Mat> v(3);
		Mat dst(Size(src.size().width, src.size().height * 3), CV_32F);
		Mat a(src.size(), CV_32F, dst.ptr<float>(0));
		Mat b(src.size(), CV_32F, dst.ptr<float>(src.size().height));
		Mat c(src.size(), CV_32F, dst.ptr<float>(2 * src.size().height));
		v[0] = a;
		v[1] = b;
		v[2] = c;
		split(src, v);


		DecorrelateColorInvert((float*)dst.data, (float*)dst.data, src.size().width, src.size().height);

		Size size_ = src.size();
		Mat aa(size_, CV_32F, (float*)dst.ptr<float>(0));
		Mat bb(size_, CV_32F, (float*)dst.ptr<float>(size_.height));
		Mat cc(size_, CV_32F, (float*)dst.ptr<float>(2 * size_.height));

		v[0] = aa;
		v[1] = bb;
		v[2] = cc;

		merge(v, dest);
	}

	/*!\fn cvtColorMatrix(cv::Mat& src, cv::Mat& C, cv::Mat& dest,int omp_core)
	* \brief
	* Converting image color to multiply input image by 3x3 or 3x4 color matrix
	*
	* \param src
	* input image
	*
	* \param C
	* 3x3 or 3x4 color matrix
	*
	* \param dest
	* dest image
	*
	* \param omp_core
	* using number of core for openmp
	*
	* \throws <exception class>
	* Description of criteria for throwing this exception.
	*
	* Write detailed description for cvtColorMatrix here.
	*
	* src homogeneous image vector = (r1, g1 b1, 1)^t, dest homogeneousimage vector = (r2, g2, b2, 1)^t
	* C is 3x3 or 3x4 color matrix @n
	* |r2|  |c0  c1 c2  c3 ||r1|@n
	* |g2| =|c4  c5 c6  c7 ||g1|@n
	* |b2|  |c8  c9 c10 c11||b1|@n
	* | 1|                  | 1|
	*
	*/
	void cvtColorMatrix(InputArray src_, OutputArray dest, InputArray C_)
	{
		if (dest.empty())dest.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat C = C_.getMat();

		Mat cmat;
		if (C.cols < C.rows)cmat = C.t();
		else cmat = C;

		vector<Mat> s;
		split(src, s);

		vector<Mat> d(3);
		for (int i = 0; i < src_.channels(); i++)
			d[i].create(src.size(), cmat.depth());
		if (C.cols == 4)
		{
			if (cmat.depth() == CV_32F)
			{
				d[0] = cmat.at<float>(0, 0) * s[0] + cmat.at<float>(0, 1) * s[1] + cmat.at<float>(0, 2) * s[2] + cmat.at<float>(0, 3);
				d[1] = cmat.at<float>(1, 0) * s[0] + cmat.at<float>(1, 1) * s[1] + cmat.at<float>(1, 2) * s[2] + cmat.at<float>(1, 3);
				d[2] = cmat.at<float>(2, 0) * s[0] + cmat.at<float>(2, 1) * s[1] + cmat.at<float>(2, 2) * s[2] + cmat.at<float>(2, 3);
			}
			else if (cmat.depth() == CV_64F)
			{
				d[0] = cmat.at<double>(0, 0) * s[0] + cmat.at<double>(0, 1) * s[1] + cmat.at<double>(0, 2) * s[2] + cmat.at<double>(0, 3);
				d[1] = cmat.at<double>(1, 0) * s[0] + cmat.at<double>(1, 1) * s[1] + cmat.at<double>(1, 2) * s[2] + cmat.at<double>(1, 3);
				d[2] = cmat.at<double>(2, 0) * s[0] + cmat.at<double>(2, 1) * s[1] + cmat.at<double>(2, 2) * s[2] + cmat.at<double>(2, 3);
			}
		}
		else if (C.cols == 3)
		{
			if (cmat.depth() == CV_32F)
			{
				d[0] = cmat.at<float>(0, 0) * s[0] + cmat.at<float>(0, 1) * s[1] + cmat.at<float>(0, 2) * s[2];
				d[1] = cmat.at<float>(1, 0) * s[0] + cmat.at<float>(1, 1) * s[1] + cmat.at<float>(1, 2) * s[2];
				d[2] = cmat.at<float>(2, 0) * s[0] + cmat.at<float>(2, 1) * s[1] + cmat.at<float>(2, 2) * s[2];
			}
			else if (cmat.depth() == CV_64F)
			{
				d[0] = cmat.at<double>(0, 0) * s[0] + cmat.at<double>(0, 1) * s[1] + cmat.at<double>(0, 2) * s[2];
				d[1] = cmat.at<double>(1, 0) * s[0] + cmat.at<double>(1, 1) * s[1] + cmat.at<double>(1, 2) * s[2];
				d[2] = cmat.at<double>(2, 0) * s[0] + cmat.at<double>(2, 1) * s[1] + cmat.at<double>(2, 2) * s[2];
			}
		}

		Mat temp;
		merge(d, temp);
		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			temp.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			temp.copyTo(dest);
	}


	/*!
	* \brief
	* find color correct matrix from images. pt1 = C pt2
	*
	* \param src_point_crowd1
	* input point crowd1. this point crowds and point crowd2 must be corespondence points
	*
	* \param src_point_crowd2
	* input point crowd2
	*
	* \param C
	* color correction matrix
	*
	* findColorMatrixAvgSdv using image pair is more robust but this function is accurate.
	*
	*/
	void findColorMatrixAvgStdDev(InputArray ref_image, InputArray target_image, OutputArray colorMatrix, const double validMin, const double validMax)
	{
		if (colorMatrix.empty())colorMatrix.create(3, 4, CV_64F);
		Mat cmat = colorMatrix.getMat();
		Scalar mean1, mean2;
		Scalar std1, std2;

		Mat mL, mR;
		compareRange(ref_image, mL, validMin, validMax);
		compareRange(target_image, mR, validMin, validMax);
		meanStdDev(ref_image, mean1, std1, mL);
		meanStdDev(target_image, mean2, std2, mR);

		cmat.at<double>(0, 0) = std1.val[0] / std2.val[0];
		cmat.at<double>(1, 1) = std1.val[1] / std2.val[1];
		cmat.at<double>(2, 2) = std1.val[2] / std2.val[2];

		cmat.at<double>(0, 3) = -std1.val[0] / std2.val[0] * mean2.val[0] + mean1.val[0];
		cmat.at<double>(1, 3) = -std1.val[1] / std2.val[1] * mean2.val[1] + mean1.val[1];
		cmat.at<double>(2, 3) = -std1.val[2] / std2.val[2] * mean2.val[2] + mean1.val[2];
	}

	void correctColor(InputArray src, InputArray targetcolor, OutputArray dest)
	{
		Mat va = targetcolor.getMat().reshape(1, targetcolor.size().area());
		Mat vb = src.getMat().reshape(1, src.size().area());

		va.convertTo(va, CV_64F);
		vb.convertTo(vb, CV_64F);
		Mat c = va.t() * vb * (vb.t() * vb).inv();

		transform(src, dest, c);
	}

#pragma endregion

	static void cvtColorGray8U32F(cv::InputArray in, cv::OutputArray out)
	{
		CV_Assert(in.depth() == CV_8U);
		CV_Assert(in.channels() == 3);

		if (out.empty() || in.size() != out.size()) out.create(in.size(), CV_32F);
		const Mat src = in.getMat();
		Mat dst = out.getMat();

		const __m256 cb = _mm256_set1_ps(0.114f);
		const __m256 cg = _mm256_set1_ps(0.587f);
		const __m256 cr = _mm256_set1_ps(0.299f);
		const int size = src.size().area();
		const int simdsize = get_simd_floor(size, 16);

		const uchar* __restrict s = src.ptr<uchar>();
		float* __restrict d = dst.ptr<float>();
		for (int i = 0; i < simdsize; i += 16)
		{
			__m256 b0, b1, g0, g1, r0, r1;
			_mm256_load_cvtepu8bgr2planar_psx2(s + 3 * i, b0, b1, g0, g1, r0, r1);
			_mm256_storeu_ps(d + i + 0, _mm256_fmadd_ps(cb, b0, _mm256_fmadd_ps(cg, g0, _mm256_mul_ps(cr, r0))));
			_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(cb, b1, _mm256_fmadd_ps(cg, g1, _mm256_mul_ps(cr, r1))));
		}
		for (int i = simdsize; i < size; i++)
		{
			d[i] = float(0.114f * s[3 * i + 0] + 0.587f * s[3 * i + 1] + 0.299f * s[3 * i + 2]);
		}
	}

	static void cvtColorGray32F32F(cv::InputArray in, cv::OutputArray out)
	{
		CV_Assert(in.depth() == CV_32F);
		CV_Assert(in.channels() == 3);

		if (out.empty() || in.size() != out.size()) out.create(in.size(), CV_32F);
		const Mat src = in.getMat();
		Mat dst = out.getMat();

		const __m256 cb = _mm256_set1_ps(0.114f);
		const __m256 cg = _mm256_set1_ps(0.587f);
		const __m256 cr = _mm256_set1_ps(0.299f);
		const int size = src.size().area();
		const int simdsize = get_simd_floor(size, 8);

		const float* __restrict s = src.ptr<float>();
		float* __restrict d = dst.ptr<float>();
		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 b0, g0, r0;
			_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, b0, g0, r0);
			_mm256_storeu_ps(d + i + 0, _mm256_fmadd_ps(cb, b0, _mm256_fmadd_ps(cg, g0, _mm256_mul_ps(cr, r0))));
		}
		for (int i = simdsize; i < size; i++)
		{
			d[i] = float(0.114f * s[3 * i + 0] + 0.587f * s[3 * i + 1] + 0.299f * s[3 * i + 2]);
		}
	}

	void cvtColorGray(cv::InputArray in, cv::OutputArray out, int depth)
	{
		if (in.depth() == CV_8U && depth == CV_32F) cvtColorGray8U32F(in, out);
		else if (in.depth() == CV_32F && depth == CV_32F) cvtColorGray32F32F(in, out);
		else
		{
			cout << "do not support this type cvtColorGray: in(depth)" << in.depth() << ", dest depth " << depth << endl;
		}
	}

#pragma region cvtColorAverageGray
	static void cvtColorAverageGray_32F(vector<Mat>& src, Mat& dest, const float normalize)
	{
		const int channels = (int)src.size();
		CV_Assert(src[0].depth() == CV_32F);

		if (channels == 3)
		{
			__m256* b = (__m256*)src[0].ptr<float>();
			__m256* g = (__m256*)src[1].ptr<float>();
			__m256* r = (__m256*)src[2].ptr<float>();
			float* dptr = dest.ptr<float>();
			const int size = src[0].size().area() / 8;
			const __m256 mnorm = _mm256_set1_ps(normalize);
			for (int i = 0; i < size; i++)
			{
				_mm256_store_ps(dptr, _mm256_mul_ps(mnorm, _mm256_add_ps(*r, _mm256_add_ps(*b, *g))));
				b++;
				g++;
				r++;
				dptr += 8;
			}
		}
		else
		{
			AutoBuffer<__m256*> ptr(channels);
			for (int c = 0; c < channels; c++)
			{
				ptr[c] = (__m256*)src[c].ptr<float>();
			}
			float* dptr = dest.ptr<float>();
			const int size = src[0].size().area() / 8;
			const __m256 mnorm = _mm256_set1_ps(normalize);
			for (int i = 0; i < size; i++)
			{
				__m256 v = *ptr[0]++;
				for (int c = 1; c < channels; c++)
				{
					v = _mm256_add_ps(v, *ptr[c]++);
				}
				_mm256_store_ps(dptr, _mm256_mul_ps(mnorm, v));
				dptr += 8;
			}
		}
	}

	static void cvtColorAverageGray_8U(Mat& src, Mat& dest, const float normalize)
	{
		CV_Assert(src.type() == CV_8UC3);

		uchar* sptr = src.ptr<uchar>();
		uchar* dptr = dest.ptr<uchar>();
		const int size = src.size().area() / 8;
		const __m256 mnorm = _mm256_set1_ps(normalize);
		__m256 b, g, r;
		for (int i = 0; i < size; i++)
		{
			_mm256_load_cvtepu8bgr2planar_ps(sptr, b, g, r);
			_mm_storel_epi64((__m128i*)dptr, _mm256_cvtps_epu8(_mm256_mul_ps(mnorm, _mm256_add_ps(r, _mm256_add_ps(b, g)))));
			sptr += 24;
			dptr += 8;
		}
	}

	static void cvtColorAverageGray_32F(Mat& src, Mat& dest, const float normalize)
	{
		CV_Assert(src.type() == CV_32FC3);

		float* sptr = src.ptr<float>();
		float* dptr = dest.ptr<float>();
		const int size = src.size().area() / 8;
		const __m256 mnorm = _mm256_set1_ps(normalize);
		__m256 b, g, r;
		for (int i = 0; i < size; i++)
		{
			_mm256_load_cvtps_bgr2planar_ps(sptr, b, g, r);
			_mm256_store_ps(dptr, _mm256_mul_ps(mnorm, _mm256_add_ps(r, _mm256_add_ps(b, g))));
			sptr += 24;
			dptr += 8;
		}
	}

	void cvtColorAverageGray(InputArray src_, OutputArray dest, const bool isKeepDistance)
	{
		if (src_.isMatVector())
		{
			vector<Mat> src;
			src_.getMatVector(src);
			if (src.size() == 1)
			{
				src[0].copyTo(dest);
				return;
			}

			dest.create(src[0].size(), src[0].depth());
			const float normalize = (isKeepDistance) ? 1.f / sqrt((float)src.size()) : 1.f / (float)src.size();
			Mat dst = dest.getMat();
			if (src[0].depth() == CV_32F) cvtColorAverageGray_32F(src, dst, normalize);
			else cout << "do not support this depth (cvtColorAverageGray)" << endl;
		}
		else
		{
			if (src_.channels() == 1)
			{
				src_.copyTo(dest);
				return;
			}

			Mat src = src_.getMat();
			dest.create(src_.size(), src.depth());

			const float normalize = (isKeepDistance) ? 1.f / sqrt(3.f) : 1.f / 3.f;
			Mat dst = dest.getMat();
			if (src.depth() == CV_32F)cvtColorAverageGray_32F(src, dst, normalize);
			else if (src.depth() == CV_8U)cvtColorAverageGray_8U(src, dst, normalize);
			else cout << "do not support this depth (cvtColorAverageGray)" << endl;
		}
	}
#pragma endregion

#pragma region cvtColorIntegerY
	static void cvtColorIntegerY_32F(Mat& src, Mat& dest)
	{
		CV_Assert(src.type() == CV_32FC3);

		float* sptr = src.ptr<float>();
		float* dptr = dest.ptr<float>();
		const int size = src.size().area() / 8;
		const __m256 rc = _mm256_set1_ps(66.f / 256.f);
		const __m256 gc = _mm256_set1_ps(129.f / 256.f);
		const __m256 bc = _mm256_set1_ps(25.f / 256.f);
		const __m256 base = _mm256_set1_ps(16.f);
		__m256 b, g, r;
		for (int i = 0; i < size; i++)
		{
			_mm256_load_cvtps_bgr2planar_ps(sptr, b, g, r);
			_mm256_store_ps(dptr, _mm256_fmadd_ps(bc, b, _mm256_fmadd_ps(gc, g, _mm256_fmadd_ps(rc, r, base))));
			sptr += 24;
			dptr += 8;
		}
		sptr = src.ptr<float>();
		dptr = dest.ptr<float>();
		for (int i = size * 8; i < src.size().area(); i++)
		{
			dptr[i] = 66.f / 256.f * sptr[3 * i + 2] + 129.f / 256.f * sptr[3 * i + 1] + 25.f / 256.f * sptr[3 * i + 0];
		}
	}

	void cvtColorIntegerY(InputArray src_, OutputArray dest)
	{
		if (src_.channels() == 1)
		{
			src_.copyTo(dest);
			return;
		}

		if (src_.depth() == CV_32F)
		{
			Mat a = src_.getMat();
			dest.create(src_.size(), CV_32F);
			Mat b = dest.getMat();
			cvtColorIntegerY_32F(a, b);
		}
		else
		{
			Mat a;
			src_.getMat().convertTo(a, CV_32F);
			Mat b(src_.size(), CV_32F);
			cvtColorIntegerY_32F(a, b);
			b.convertTo(dest, src_.depth());
		}
	}
#pragma endregion

#pragma region PCA
	static void cvtColorPCAOpenCVPCA(Mat& src, Mat& dest, const int dest_channels)
	{
		Mat x = src.reshape(1, src.size().area());
		PCA pca(x, cv::Mat(), cv::PCA::DATA_AS_ROW, dest_channels);
		dest = pca.project(x).reshape(dest_channels, src.rows);
	}

	static void cvtColorPCAOpenCVCovMat(Mat& src, Mat& dest, const int dest_channels, Mat& evec, Mat& eval)
	{
		Mat x = src.reshape(1, src.size().area());
		Mat cov, mean;
		cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
		eigen(cov, eval, evec);
		Mat transmat;
		evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);
		cv::transform(src, dest, transmat);
	}

#pragma region calcCovarMatrixN_32F
	static void calcCovarMatrix2_32F_(const Mat& src, Mat& dest, Mat& average_value)
	{
		dest.create(src.channels(), src.channels(), CV_64F);
		CV_Assert(src.channels() <= 4);

		Scalar ave = mean(src);
		average_value.create(1, 4, CV_64F);
		average_value.at<double>(0) = ave.val[0];
		average_value.at<double>(1) = ave.val[1];
		average_value.at<double>(2) = ave.val[2];
		average_value.at<double>(3) = ave.val[3];

		const int simdsize = get_simd_ceil(src.size().area(), 8);
		const float* s = src.ptr<float>();

		__m256 mbb = _mm256_setzero_ps();
		__m256 mgg = _mm256_setzero_ps();
		__m256 mbg = _mm256_setzero_ps();
		const __m256 mba = _mm256_set1_ps(float(ave.val[0]));
		const __m256 mga = _mm256_set1_ps(float(ave.val[1]));
		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 mb, mg;
			_mm256_load_deinterleave_ps(s + 2 * i, mb, mg);
			mb = _mm256_sub_ps(mb, mba);
			mg = _mm256_sub_ps(mg, mga);
			mbb = _mm256_fmadd_ps(mb, mb, mbb);
			mgg = _mm256_fmadd_ps(mg, mg, mgg);
			mbg = _mm256_fmadd_ps(mb, mg, mbg);
		}
		double bb = _mm256_reduceadd_ps(mbb);
		double gg = _mm256_reduceadd_ps(mgg);
		double bg = _mm256_reduceadd_ps(mbg);
		dest.at<double>(0, 0) = bb / simdsize;
		dest.at<double>(0, 1) = bg / simdsize;
		dest.at<double>(1, 0) = bg / simdsize;
		dest.at<double>(1, 1) = gg / simdsize;
	}

	static void calcCovarMatrix3_32F_(const Mat& src, Mat& dest, Mat& average_value)
	{
		dest.create(src.channels(), src.channels(), CV_64F);
		CV_Assert(src.channels() <= 4);

		Scalar ave = mean(src);
		average_value.create(1, 4, CV_64F);
		average_value.at<double>(0) = ave.val[0];
		average_value.at<double>(1) = ave.val[1];
		average_value.at<double>(2) = ave.val[2];
		average_value.at<double>(3) = ave.val[3];

		const int simdsize = get_simd_ceil(src.size().area(), 8);
		const float* s = src.ptr<float>();

		__m256 mbb = _mm256_setzero_ps();
		__m256 mgg = _mm256_setzero_ps();
		__m256 mrr = _mm256_setzero_ps();
		__m256 mbg = _mm256_setzero_ps();
		__m256 mbr = _mm256_setzero_ps();
		__m256 mgr = _mm256_setzero_ps();
		const __m256 mba = _mm256_set1_ps(float(ave.val[0]));
		const __m256 mga = _mm256_set1_ps(float(ave.val[1]));
		const __m256 mra = _mm256_set1_ps(float(ave.val[2]));
		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 mb, mg, mr;
			_mm256_load_cvtps_bgr2planar_ps(s + 3 * i, mb, mg, mr);
			mb = _mm256_sub_ps(mb, mba);
			mg = _mm256_sub_ps(mg, mga);
			mr = _mm256_sub_ps(mr, mra);
			mbb = _mm256_fmadd_ps(mb, mb, mbb);
			mgg = _mm256_fmadd_ps(mg, mg, mgg);
			mrr = _mm256_fmadd_ps(mr, mr, mrr);
			mbg = _mm256_fmadd_ps(mb, mg, mbg);
			mbr = _mm256_fmadd_ps(mb, mr, mbr);
			mgr = _mm256_fmadd_ps(mg, mr, mgr);
		}
		double bb = _mm256_reduceadd_ps(mbb);
		double gg = _mm256_reduceadd_ps(mgg);
		double rr = _mm256_reduceadd_ps(mrr);
		double bg = _mm256_reduceadd_ps(mbg);
		double br = _mm256_reduceadd_ps(mbr);
		double gr = _mm256_reduceadd_ps(mgr);
		dest.at<double>(0, 0) = bb / simdsize;
		dest.at<double>(0, 1) = bg / simdsize;
		dest.at<double>(0, 2) = br / simdsize;
		dest.at<double>(1, 0) = bg / simdsize;
		dest.at<double>(1, 1) = gg / simdsize;
		dest.at<double>(1, 2) = gr / simdsize;
		dest.at<double>(2, 0) = br / simdsize;
		dest.at<double>(2, 1) = gr / simdsize;
		dest.at<double>(2, 2) = rr / simdsize;
	}


	static void calcCovarMatrix2_32F_(const vector<Mat>& src, Mat& dest)
	{
		dest.create((int)src.size(), (int)src.size(), CV_64F);
		CV_Assert(src.size() <= 4);
		const float m0 = (float)cp::average(src[0]);
		const float m1 = (float)cp::average(src[1]);

		const int simdsize = get_simd_ceil(src[0].size().area(), 8);
		const float* s0 = src[0].ptr<float>();
		const float* s1 = src[1].ptr<float>();

		__m256 mbb = _mm256_setzero_ps();
		__m256 mgg = _mm256_setzero_ps();
		__m256 mbg = _mm256_setzero_ps();

		const __m256 mba = _mm256_set1_ps(m0);
		const __m256 mga = _mm256_set1_ps(m1);
		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 mb = _mm256_load_ps(s0 + i);
			__m256 mg = _mm256_load_ps(s1 + i);

			mb = _mm256_sub_ps(mb, mba);
			mg = _mm256_sub_ps(mg, mga);
			mbb = _mm256_fmadd_ps(mb, mb, mbb);
			mgg = _mm256_fmadd_ps(mg, mg, mgg);
			mbg = _mm256_fmadd_ps(mb, mg, mbg);
		}
		const double bb = _mm256_reduceadd_ps(mbb);
		const double gg = _mm256_reduceadd_ps(mgg);
		const double bg = _mm256_reduceadd_ps(mbg);
		dest.at<double>(0, 0) = bb / simdsize;
		dest.at<double>(0, 1) = bg / simdsize;
		dest.at<double>(1, 0) = bg / simdsize;
		dest.at<double>(1, 1) = gg / simdsize;
	}

	static void calcCovarMatrix3_32F_(const vector<Mat>& src, Mat& dest)
	{
		dest.create((int)src.size(), (int)src.size(), CV_64F);
		CV_Assert(src.size() <= 4);
		const float m0 = (float)cp::average(src[0]);
		const float m1 = (float)cp::average(src[1]);
		const float m2 = (float)cp::average(src[2]);

		const int simdsize = get_simd_ceil(src[0].size().area(), 8);
		const float* s0 = src[0].ptr<float>();
		const float* s1 = src[1].ptr<float>();
		const float* s2 = src[2].ptr<float>();

		__m256 mbb = _mm256_setzero_ps();
		__m256 mgg = _mm256_setzero_ps();
		__m256 mrr = _mm256_setzero_ps();
		__m256 mbg = _mm256_setzero_ps();
		__m256 mbr = _mm256_setzero_ps();
		__m256 mgr = _mm256_setzero_ps();
		const __m256 mba = _mm256_set1_ps(m0);
		const __m256 mga = _mm256_set1_ps(m1);
		const __m256 mra = _mm256_set1_ps(m2);
		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 mb = _mm256_load_ps(s0 + i);
			__m256 mg = _mm256_load_ps(s1 + i);
			__m256 mr = _mm256_load_ps(s2 + i);

			mb = _mm256_sub_ps(mb, mba);
			mg = _mm256_sub_ps(mg, mga);
			mr = _mm256_sub_ps(mr, mra);
			mbb = _mm256_fmadd_ps(mb, mb, mbb);
			mbg = _mm256_fmadd_ps(mb, mg, mbg);
			mbr = _mm256_fmadd_ps(mb, mr, mbr);
			mgg = _mm256_fmadd_ps(mg, mg, mgg);
			mgr = _mm256_fmadd_ps(mg, mr, mgr);
			mrr = _mm256_fmadd_ps(mr, mr, mrr);
		}
		const double bb = _mm256_reduceadd_ps(mbb);
		const double gg = _mm256_reduceadd_ps(mgg);
		const double rr = _mm256_reduceadd_ps(mrr);
		const double bg = _mm256_reduceadd_ps(mbg);
		const double br = _mm256_reduceadd_ps(mbr);
		const double gr = _mm256_reduceadd_ps(mgr);
		dest.at<double>(0, 0) = bb / simdsize;
		dest.at<double>(0, 1) = bg / simdsize;
		dest.at<double>(0, 2) = br / simdsize;
		dest.at<double>(1, 0) = bg / simdsize;
		dest.at<double>(1, 1) = gg / simdsize;
		dest.at<double>(1, 2) = gr / simdsize;
		dest.at<double>(2, 0) = br / simdsize;
		dest.at<double>(2, 1) = gr / simdsize;
		dest.at<double>(2, 2) = rr / simdsize;
	}

	static void calcCovarMatrix4_32F_(const vector<Mat>& src, Mat& dest)
	{
		dest.create((int)src.size(), (int)src.size(), CV_64F);

		const float m0 = (float)cp::average(src[0]);
		const float m1 = (float)cp::average(src[1]);
		const float m2 = (float)cp::average(src[2]);
		const float m3 = (float)cp::average(src[3]);

		const int simdsize = get_simd_ceil(src[0].size().area(), 8);
		const float* s0 = src[0].ptr<float>();
		const float* s1 = src[1].ptr<float>();
		const float* s2 = src[2].ptr<float>();
		const float* s3 = src[3].ptr<float>();

		__m256 m00 = _mm256_setzero_ps();
		__m256 m01 = _mm256_setzero_ps();
		__m256 m02 = _mm256_setzero_ps();
		__m256 m03 = _mm256_setzero_ps();
		__m256 m11 = _mm256_setzero_ps();
		__m256 m12 = _mm256_setzero_ps();
		__m256 m13 = _mm256_setzero_ps();
		__m256 m22 = _mm256_setzero_ps();
		__m256 m23 = _mm256_setzero_ps();
		__m256 m33 = _mm256_setzero_ps();

		const __m256 ma0 = _mm256_set1_ps(m0);
		const __m256 ma1 = _mm256_set1_ps(m1);
		const __m256 ma2 = _mm256_set1_ps(m2);
		const __m256 ma3 = _mm256_set1_ps(m3);

		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 m0 = _mm256_load_ps(s0 + i);
			__m256 m1 = _mm256_load_ps(s1 + i);
			__m256 m2 = _mm256_load_ps(s2 + i);
			__m256 m3 = _mm256_load_ps(s3 + i);

			m0 = _mm256_sub_ps(m0, ma0);
			m1 = _mm256_sub_ps(m1, ma1);
			m2 = _mm256_sub_ps(m2, ma2);
			m3 = _mm256_sub_ps(m3, ma3);

			m00 = _mm256_fmadd_ps(m0, m0, m00);
			m01 = _mm256_fmadd_ps(m0, m1, m01);
			m02 = _mm256_fmadd_ps(m0, m2, m02);
			m03 = _mm256_fmadd_ps(m0, m3, m03);
			m11 = _mm256_fmadd_ps(m1, m1, m11);
			m12 = _mm256_fmadd_ps(m1, m2, m12);
			m13 = _mm256_fmadd_ps(m1, m3, m13);
			m22 = _mm256_fmadd_ps(m2, m2, m22);
			m23 = _mm256_fmadd_ps(m2, m3, m23);
			m33 = _mm256_fmadd_ps(m3, m3, m33);
		}
		const double v00 = _mm256_reduceadd_ps(m00);
		const double v01 = _mm256_reduceadd_ps(m01);
		const double v02 = _mm256_reduceadd_ps(m02);
		const double v03 = _mm256_reduceadd_ps(m03);
		const double v11 = _mm256_reduceadd_ps(m11);
		const double v12 = _mm256_reduceadd_ps(m12);
		const double v13 = _mm256_reduceadd_ps(m13);
		const double v22 = _mm256_reduceadd_ps(m22);
		const double v23 = _mm256_reduceadd_ps(m23);
		const double v33 = _mm256_reduceadd_ps(m33);

		dest.at<double>(0, 0) = v00 / simdsize;
		dest.at<double>(0, 1) = v01 / simdsize;
		dest.at<double>(0, 2) = v02 / simdsize;
		dest.at<double>(0, 3) = v03 / simdsize;

		dest.at<double>(1, 0) = v01 / simdsize;
		dest.at<double>(1, 1) = v11 / simdsize;
		dest.at<double>(1, 2) = v12 / simdsize;
		dest.at<double>(1, 3) = v13 / simdsize;

		dest.at<double>(2, 0) = v02 / simdsize;
		dest.at<double>(2, 1) = v12 / simdsize;
		dest.at<double>(2, 2) = v22 / simdsize;
		dest.at<double>(2, 3) = v23 / simdsize;

		dest.at<double>(3, 0) = v03 / simdsize;
		dest.at<double>(3, 1) = v13 / simdsize;
		dest.at<double>(3, 2) = v23 / simdsize;
		dest.at<double>(3, 3) = v33 / simdsize;
	}

	static void calcCovarMatrix5_32F_(const vector<Mat>& src, Mat& dest)
	{
		dest.create((int)src.size(), (int)src.size(), CV_64F);

		const float m0 = (float)cp::average(src[0]);
		const float m1 = (float)cp::average(src[1]);
		const float m2 = (float)cp::average(src[2]);
		const float m3 = (float)cp::average(src[3]);
		const float m4 = (float)cp::average(src[4]);

		const int simdsize = get_simd_ceil(src[0].size().area(), 8);
		const float* s0 = src[0].ptr<float>();
		const float* s1 = src[1].ptr<float>();
		const float* s2 = src[2].ptr<float>();
		const float* s3 = src[3].ptr<float>();
		const float* s4 = src[4].ptr<float>();

		__m256 m00 = _mm256_setzero_ps();
		__m256 m01 = _mm256_setzero_ps();
		__m256 m02 = _mm256_setzero_ps();
		__m256 m03 = _mm256_setzero_ps();
		__m256 m04 = _mm256_setzero_ps();
		__m256 m11 = _mm256_setzero_ps();
		__m256 m12 = _mm256_setzero_ps();
		__m256 m13 = _mm256_setzero_ps();
		__m256 m14 = _mm256_setzero_ps();
		__m256 m22 = _mm256_setzero_ps();
		__m256 m23 = _mm256_setzero_ps();
		__m256 m24 = _mm256_setzero_ps();
		__m256 m33 = _mm256_setzero_ps();
		__m256 m34 = _mm256_setzero_ps();
		__m256 m44 = _mm256_setzero_ps();

		const __m256 ma0 = _mm256_set1_ps(m0);
		const __m256 ma1 = _mm256_set1_ps(m1);
		const __m256 ma2 = _mm256_set1_ps(m2);
		const __m256 ma3 = _mm256_set1_ps(m3);
		const __m256 ma4 = _mm256_set1_ps(m4);

		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 m0 = _mm256_load_ps(s0 + i);
			__m256 m1 = _mm256_load_ps(s1 + i);
			__m256 m2 = _mm256_load_ps(s2 + i);
			__m256 m3 = _mm256_load_ps(s3 + i);
			__m256 m4 = _mm256_load_ps(s4 + i);

			m0 = _mm256_sub_ps(m0, ma0);
			m1 = _mm256_sub_ps(m1, ma1);
			m2 = _mm256_sub_ps(m2, ma2);
			m3 = _mm256_sub_ps(m3, ma3);
			m4 = _mm256_sub_ps(m4, ma4);

			m00 = _mm256_fmadd_ps(m0, m0, m00);
			m01 = _mm256_fmadd_ps(m0, m1, m01);
			m02 = _mm256_fmadd_ps(m0, m2, m02);
			m03 = _mm256_fmadd_ps(m0, m3, m03);
			m04 = _mm256_fmadd_ps(m0, m4, m04);
			m11 = _mm256_fmadd_ps(m1, m1, m11);
			m12 = _mm256_fmadd_ps(m1, m2, m12);
			m13 = _mm256_fmadd_ps(m1, m3, m13);
			m14 = _mm256_fmadd_ps(m1, m4, m14);
			m22 = _mm256_fmadd_ps(m2, m2, m22);
			m23 = _mm256_fmadd_ps(m2, m3, m23);
			m24 = _mm256_fmadd_ps(m2, m4, m24);
			m33 = _mm256_fmadd_ps(m3, m3, m33);
			m34 = _mm256_fmadd_ps(m3, m4, m34);
			m44 = _mm256_fmadd_ps(m4, m4, m44);
		}
		const double v00 = _mm256_reduceadd_ps(m00);
		const double v01 = _mm256_reduceadd_ps(m01);
		const double v02 = _mm256_reduceadd_ps(m02);
		const double v03 = _mm256_reduceadd_ps(m03);
		const double v04 = _mm256_reduceadd_ps(m04);
		const double v11 = _mm256_reduceadd_ps(m11);
		const double v12 = _mm256_reduceadd_ps(m12);
		const double v13 = _mm256_reduceadd_ps(m13);
		const double v14 = _mm256_reduceadd_ps(m14);
		const double v22 = _mm256_reduceadd_ps(m22);
		const double v23 = _mm256_reduceadd_ps(m23);
		const double v24 = _mm256_reduceadd_ps(m24);
		const double v33 = _mm256_reduceadd_ps(m33);
		const double v34 = _mm256_reduceadd_ps(m34);
		const double v44 = _mm256_reduceadd_ps(m44);

		dest.at<double>(0, 0) = v00 / simdsize;
		dest.at<double>(0, 1) = v01 / simdsize;
		dest.at<double>(0, 2) = v02 / simdsize;
		dest.at<double>(0, 3) = v03 / simdsize;
		dest.at<double>(0, 4) = v04 / simdsize;

		dest.at<double>(1, 0) = v01 / simdsize;
		dest.at<double>(1, 1) = v11 / simdsize;
		dest.at<double>(1, 2) = v12 / simdsize;
		dest.at<double>(1, 3) = v13 / simdsize;
		dest.at<double>(1, 4) = v14 / simdsize;

		dest.at<double>(2, 0) = v02 / simdsize;
		dest.at<double>(2, 1) = v12 / simdsize;
		dest.at<double>(2, 2) = v22 / simdsize;
		dest.at<double>(2, 3) = v23 / simdsize;
		dest.at<double>(2, 4) = v24 / simdsize;

		dest.at<double>(3, 0) = v03 / simdsize;
		dest.at<double>(3, 1) = v13 / simdsize;
		dest.at<double>(3, 2) = v23 / simdsize;
		dest.at<double>(3, 3) = v33 / simdsize;
		dest.at<double>(3, 4) = v34 / simdsize;

		dest.at<double>(4, 0) = v04 / simdsize;
		dest.at<double>(4, 1) = v14 / simdsize;
		dest.at<double>(4, 2) = v24 / simdsize;
		dest.at<double>(4, 3) = v34 / simdsize;
		dest.at<double>(4, 4) = v44 / simdsize;
	}

	static void calcCovarMatrix6_32F_(const vector<Mat>& src, Mat& dest)
	{
		dest.create((int)src.size(), (int)src.size(), CV_64F);

		const float m0 = (float)cp::average(src[0]);
		const float m1 = (float)cp::average(src[1]);
		const float m2 = (float)cp::average(src[2]);
		const float m3 = (float)cp::average(src[3]);
		const float m4 = (float)cp::average(src[4]);
		const float m5 = (float)cp::average(src[5]);

		const int simdsize = get_simd_ceil(src[0].size().area(), 8);
		const float* s0 = src[0].ptr<float>();
		const float* s1 = src[1].ptr<float>();
		const float* s2 = src[2].ptr<float>();
		const float* s3 = src[3].ptr<float>();
		const float* s4 = src[4].ptr<float>();
		const float* s5 = src[5].ptr<float>();

		__m256 m00 = _mm256_setzero_ps();
		__m256 m01 = _mm256_setzero_ps();
		__m256 m02 = _mm256_setzero_ps();
		__m256 m03 = _mm256_setzero_ps();
		__m256 m04 = _mm256_setzero_ps();
		__m256 m05 = _mm256_setzero_ps();
		__m256 m11 = _mm256_setzero_ps();
		__m256 m12 = _mm256_setzero_ps();
		__m256 m13 = _mm256_setzero_ps();
		__m256 m14 = _mm256_setzero_ps();
		__m256 m15 = _mm256_setzero_ps();
		__m256 m22 = _mm256_setzero_ps();
		__m256 m23 = _mm256_setzero_ps();
		__m256 m24 = _mm256_setzero_ps();
		__m256 m25 = _mm256_setzero_ps();
		__m256 m33 = _mm256_setzero_ps();
		__m256 m34 = _mm256_setzero_ps();
		__m256 m35 = _mm256_setzero_ps();
		__m256 m44 = _mm256_setzero_ps();
		__m256 m45 = _mm256_setzero_ps();
		__m256 m55 = _mm256_setzero_ps();

		const __m256 ma0 = _mm256_set1_ps(m0);
		const __m256 ma1 = _mm256_set1_ps(m1);
		const __m256 ma2 = _mm256_set1_ps(m2);
		const __m256 ma3 = _mm256_set1_ps(m3);
		const __m256 ma4 = _mm256_set1_ps(m4);
		const __m256 ma5 = _mm256_set1_ps(m5);

		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 m0 = _mm256_load_ps(s0 + i);
			__m256 m1 = _mm256_load_ps(s1 + i);
			__m256 m2 = _mm256_load_ps(s2 + i);
			__m256 m3 = _mm256_load_ps(s3 + i);
			__m256 m4 = _mm256_load_ps(s4 + i);
			__m256 m5 = _mm256_load_ps(s5 + i);

			m0 = _mm256_sub_ps(m0, ma0);
			m1 = _mm256_sub_ps(m1, ma1);
			m2 = _mm256_sub_ps(m2, ma2);
			m3 = _mm256_sub_ps(m3, ma3);
			m4 = _mm256_sub_ps(m4, ma4);
			m5 = _mm256_sub_ps(m5, ma5);

			m00 = _mm256_fmadd_ps(m0, m0, m00);
			m01 = _mm256_fmadd_ps(m0, m1, m01);
			m02 = _mm256_fmadd_ps(m0, m2, m02);
			m03 = _mm256_fmadd_ps(m0, m3, m03);
			m04 = _mm256_fmadd_ps(m0, m4, m04);
			m05 = _mm256_fmadd_ps(m0, m5, m05);
			m11 = _mm256_fmadd_ps(m1, m1, m11);
			m12 = _mm256_fmadd_ps(m1, m2, m12);
			m13 = _mm256_fmadd_ps(m1, m3, m13);
			m14 = _mm256_fmadd_ps(m1, m4, m14);
			m15 = _mm256_fmadd_ps(m1, m5, m15);
			m22 = _mm256_fmadd_ps(m2, m2, m22);
			m23 = _mm256_fmadd_ps(m2, m3, m23);
			m24 = _mm256_fmadd_ps(m2, m4, m24);
			m25 = _mm256_fmadd_ps(m2, m5, m25);
			m33 = _mm256_fmadd_ps(m3, m3, m33);
			m34 = _mm256_fmadd_ps(m3, m4, m34);
			m35 = _mm256_fmadd_ps(m3, m5, m35);
			m44 = _mm256_fmadd_ps(m4, m4, m44);
			m45 = _mm256_fmadd_ps(m4, m5, m45);
			m55 = _mm256_fmadd_ps(m5, m5, m55);
		}
		const double v00 = _mm256_reduceadd_ps(m00);
		const double v01 = _mm256_reduceadd_ps(m01);
		const double v02 = _mm256_reduceadd_ps(m02);
		const double v03 = _mm256_reduceadd_ps(m03);
		const double v04 = _mm256_reduceadd_ps(m04);
		const double v05 = _mm256_reduceadd_ps(m05);
		const double v11 = _mm256_reduceadd_ps(m11);
		const double v12 = _mm256_reduceadd_ps(m12);
		const double v13 = _mm256_reduceadd_ps(m13);
		const double v14 = _mm256_reduceadd_ps(m14);
		const double v15 = _mm256_reduceadd_ps(m15);
		const double v22 = _mm256_reduceadd_ps(m22);
		const double v23 = _mm256_reduceadd_ps(m23);
		const double v24 = _mm256_reduceadd_ps(m24);
		const double v25 = _mm256_reduceadd_ps(m25);
		const double v33 = _mm256_reduceadd_ps(m33);
		const double v34 = _mm256_reduceadd_ps(m34);
		const double v35 = _mm256_reduceadd_ps(m35);
		const double v44 = _mm256_reduceadd_ps(m44);
		const double v45 = _mm256_reduceadd_ps(m45);
		const double v55 = _mm256_reduceadd_ps(m55);

		dest.at<double>(0, 0) = v00 / simdsize;
		dest.at<double>(0, 1) = v01 / simdsize;
		dest.at<double>(0, 2) = v02 / simdsize;
		dest.at<double>(0, 3) = v03 / simdsize;
		dest.at<double>(0, 4) = v04 / simdsize;
		dest.at<double>(0, 5) = v05 / simdsize;

		dest.at<double>(1, 0) = v01 / simdsize;
		dest.at<double>(1, 1) = v11 / simdsize;
		dest.at<double>(1, 2) = v12 / simdsize;
		dest.at<double>(1, 3) = v13 / simdsize;
		dest.at<double>(1, 4) = v14 / simdsize;
		dest.at<double>(1, 5) = v15 / simdsize;

		dest.at<double>(2, 0) = v02 / simdsize;
		dest.at<double>(2, 1) = v12 / simdsize;
		dest.at<double>(2, 2) = v22 / simdsize;
		dest.at<double>(2, 3) = v23 / simdsize;
		dest.at<double>(2, 4) = v24 / simdsize;
		dest.at<double>(2, 5) = v25 / simdsize;

		dest.at<double>(3, 0) = v03 / simdsize;
		dest.at<double>(3, 1) = v13 / simdsize;
		dest.at<double>(3, 2) = v23 / simdsize;
		dest.at<double>(3, 3) = v33 / simdsize;
		dest.at<double>(3, 4) = v34 / simdsize;
		dest.at<double>(3, 5) = v35 / simdsize;

		dest.at<double>(4, 0) = v04 / simdsize;
		dest.at<double>(4, 1) = v14 / simdsize;
		dest.at<double>(4, 2) = v24 / simdsize;
		dest.at<double>(4, 3) = v34 / simdsize;
		dest.at<double>(4, 4) = v44 / simdsize;
		dest.at<double>(4, 5) = v45 / simdsize;

		dest.at<double>(5, 0) = v05 / simdsize;
		dest.at<double>(5, 1) = v15 / simdsize;
		dest.at<double>(5, 2) = v25 / simdsize;
		dest.at<double>(5, 3) = v35 / simdsize;
		dest.at<double>(5, 4) = v45 / simdsize;
		dest.at<double>(5, 5) = v55 / simdsize;
	}

	template<int N>
	static void calcCovarMatrix__32F_(const vector<Mat>& src, Mat& dest)
	{
		const int simdsize = get_simd_ceil(src[0].size().area(), 8);

		dest.create((int)src.size(), (int)src.size(), CV_64F);

		AutoBuffer<float> ave(N);
		for (int c = 0; c < N; c++) ave[c] = (float)cp::average(src[c]);
		AutoBuffer<const float*> s(N);
		for (int c = 0; c < N; c++) s[c] = src[c].ptr<float>();

		AutoBuffer<__m256> mc(N * N);
		for (int c = 0; c < N * N; c++) mc[c] = _mm256_setzero_ps();
		AutoBuffer<__m256> ma(N);
		for (int c = 0; c < N; c++) ma[c] = _mm256_set1_ps(ave[c]);
		AutoBuffer<__m256> m(N);

		for (int i = 0; i < simdsize; i += 8)
		{
			for (int c = 0; c < N; c++) m[c] = _mm256_sub_ps(_mm256_load_ps(s[c] + i), ma[c]);

			int idx = 0;
			for (int c = 0; c < N; c++)
			{
				for (int d = c; d < N; d++)
				{
					mc[idx] = _mm256_fmadd_ps(m[c], m[d], mc[idx]);
					idx++;
				}
			}
		}
		int idx = 0;
		for (int c = 0; c < N; c++)
		{
			for (int d = c; d < N; d++)
			{
				dest.at<double>(c, d) = _mm256_reduceadd_ps(mc[idx]) / simdsize;
				idx++;
			}
		}
		for (int c = 0; c < N; c++)
		{
			for (int d = 0; d < c; d++)
			{
				dest.at<double>(c, d) = dest.at<double>(d, c);
			}
		}
	}

	static void calcCovarMatrixN_32F_(const vector<Mat>& src, Mat& dest)
	{
		const int N = (int)src.size();
		const int simdsize = get_simd_ceil(src[0].size().area(), 8);

		dest.create((int)src.size(), (int)src.size(), CV_64F);

		AutoBuffer<float> ave(N);
		for (int c = 0; c < N; c++) ave[c] = (float)cp::average(src[c]);
		AutoBuffer<const float*> s(N);
		for (int c = 0; c < N; c++) s[c] = src[c].ptr<float>();

		AutoBuffer<__m256> mc(N * N);
		for (int c = 0; c < N * N; c++) mc[c] = _mm256_setzero_ps();
		AutoBuffer<__m256> ma(N);
		for (int c = 0; c < N; c++) ma[c] = _mm256_set1_ps(ave[c]);
		AutoBuffer<__m256> m(N);

		for (int i = 0; i < simdsize; i += 8)
		{
			for (int c = 0; c < N; c++) m[c] = _mm256_sub_ps(_mm256_load_ps(s[c] + i), ma[c]);

			int idx = 0;
			for (int c = 0; c < N; c++)
			{
				for (int d = c; d < N; d++)
				{
					mc[idx] = _mm256_fmadd_ps(m[c], m[d], mc[idx]);
					idx++;
				}
			}
		}
		int idx = 0;
		for (int c = 0; c < N; c++)
		{
			for (int d = c; d < N; d++)
			{
				dest.at<double>(c, d) = _mm256_reduceadd_ps(mc[idx]) / simdsize;
				idx++;
			}
		}
		for (int c = 0; c < N; c++)
		{
			for (int d = 0; d < c; d++)
			{
				dest.at<double>(c, d) = dest.at<double>(d, c);
			}
		}
	}
#pragma endregion
#pragma region calcCovarMatrixN_64F
	static void calcCovarMatrixN_64F_(const vector<Mat>& src, Mat& dest)
	{
		const int N = (int)src.size();
		const int simdsize = get_simd_ceil(src[0].size().area(), 4);

		dest.create((int)src.size(), (int)src.size(), CV_64F);

		AutoBuffer<double> ave(N);
		//for (int c = 0; c < N; c++) ave[c] = cp::average(src[c]);
		for (int c = 0; c < N; c++) ave[c] = cv::mean(src[c])[0];
		AutoBuffer<const double*> s(N);
		for (int c = 0; c < N; c++) s[c] = src[c].ptr<double>();

		AutoBuffer<__m256d> mc(N * N);
		for (int c = 0; c < N * N; c++) mc[c] = _mm256_setzero_pd();
		AutoBuffer<__m256d> ma(N);
		for (int c = 0; c < N; c++) ma[c] = _mm256_set1_pd(ave[c]);
		AutoBuffer<__m256d> m(N);

		for (int i = 0; i < simdsize; i += 4)
		{
			for (int c = 0; c < N; c++) m[c] = _mm256_sub_pd(_mm256_load_pd(s[c] + i), ma[c]);

			int idx = 0;
			for (int c = 0; c < N; c++)
			{
				for (int d = c; d < N; d++)
				{
					mc[idx] = _mm256_fmadd_pd(m[c], m[d], mc[idx]);
					idx++;
				}
			}
		}
		int idx = 0;
		for (int c = 0; c < N; c++)
		{
			for (int d = c; d < N; d++)
			{
				dest.at<double>(c, d) = _mm256_reduceadd_pd(mc[idx]) / simdsize;
				idx++;
			}
		}
		for (int c = 0; c < N; c++)
		{
			for (int d = 0; d < c; d++)
			{
				dest.at<double>(c, d) = dest.at<double>(d, c);
			}
		}
	}
#pragma endregion
#pragma region projectPCA_MxN_32F
	//src 3 x dest n(1-3)
	static void projectPCA_2xn_32F(const Mat& src, Mat& dest, Mat& evec)
	{
		CV_Assert(src.channels() == 2);
		const int sizesimd = src.size().area() / 8;
		const float* sptr = src.ptr<float>();
		float* dptr = dest.ptr<float>();
		AutoBuffer<__m256> mv(2 * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				mv[2 * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		if (evec.rows == 1)
		{
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb, mg;
				_mm256_load_deinterleave_ps(sptr, mb, mg);

				__m256 d0 = _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0]));
				_mm256_store_ps(dptr, d0);
				sptr += 16;
				dptr += 8;
			}
		}
		else if (evec.rows == 2)
		{
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb, mg;
				_mm256_load_deinterleave_ps(sptr, mb, mg);

				__m256 d0 = _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0]));
				__m256 d1 = _mm256_fmadd_ps(mg, mv[3], _mm256_mul_ps(mb, mv[2]));

				_mm256_store_interleave_ps(dptr, d0, d1);
				sptr += 16;
				dptr += 16;
			}
		}
	}

	static void projectPCA_3xN_32F(const Mat& src, Mat& dest, Mat& evec)
	{
		CV_Assert(src.channels() == 3);
		const int sizesimd = src.size().area() / 8;
		const float* sptr = src.ptr<float>();
		float* dptr = dest.ptr<float>();
		AutoBuffer<__m256> mv(3 * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				mv[3 * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		if (evec.rows == 1)
		{
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(sptr, mb, mg, mr);

				__m256 d0 = _mm256_fmadd_ps(mr, mv[2], _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0])));
				_mm256_store_ps(dptr, d0);
				sptr += 24;
				dptr += 8;
			}
		}
		else if (evec.rows == 2)
		{
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(sptr, mb, mg, mr);

				__m256 d0 = _mm256_fmadd_ps(mr, mv[2], _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0])));
				__m256 d1 = _mm256_fmadd_ps(mr, mv[5], _mm256_fmadd_ps(mg, mv[4], _mm256_mul_ps(mb, mv[3])));

				_mm256_store_interleave_ps(dptr, d0, d1);
				sptr += 24;
				dptr += 16;
			}
		}
		else if (evec.rows == 3)
		{
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb, mg, mr;
				_mm256_load_cvtps_bgr2planar_ps(sptr, mb, mg, mr);

				__m256 d0 = _mm256_fmadd_ps(mr, mv[2], _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0])));
				__m256 d1 = _mm256_fmadd_ps(mr, mv[5], _mm256_fmadd_ps(mg, mv[4], _mm256_mul_ps(mb, mv[3])));
				__m256 d2 = _mm256_fmadd_ps(mr, mv[8], _mm256_fmadd_ps(mg, mv[7], _mm256_mul_ps(mb, mv[6])));

				_mm256_store_ps_color(dptr, d0, d1, d2);
				sptr += 24;
				dptr += 24;
			}
		}
	}

	static void projectPCA_MxN_32F(const Mat& src, Mat& dest, Mat& evec)
	{
		cv::transform(src, dest, evec);
	}

	//src 3 x dest n(1-3)
	static void projectPCA_2xN_32F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		CV_Assert(src.size() == 2);
		const int sizesimd = src[0].size().area() / 8;
		const float* sptr0 = src[0].ptr<float>();
		const float* sptr1 = src[1].ptr<float>();

		AutoBuffer<__m256> mv(2 * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				mv[2 * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		if (evec.rows == 1)
		{
			float* dptr0 = dest[0].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb = _mm256_load_ps(sptr0);
				__m256 mg = _mm256_load_ps(sptr1);

				__m256 d0 = _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0]));

				_mm256_store_ps(dptr0, d0);
				sptr0 += 8;
				sptr1 += 8;
				dptr0 += 8;
			}
		}
		else if (evec.rows == 2)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb = _mm256_load_ps(sptr0);
				__m256 mg = _mm256_load_ps(sptr1);

				__m256 d0 = _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0]));
				__m256 d1 = _mm256_fmadd_ps(mg, mv[3], _mm256_mul_ps(mb, mv[2]));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				sptr0 += 8;
				sptr1 += 8;
				dptr0 += 8;
				dptr1 += 8;
			}
		}
	}

	static void projectPCA_3xN_32F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		CV_Assert(src.size() == 3);
		const int sizesimd = src[0].size().area() / 8;
		const float* sptr0 = src[0].ptr<float>();
		const float* sptr1 = src[1].ptr<float>();
		const float* sptr2 = src[2].ptr<float>();

		AutoBuffer<__m256> mv(3 * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				mv[3 * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		if (evec.rows == 1)
		{
			float* dptr0 = dest[0].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb = _mm256_load_ps(sptr0);
				__m256 mg = _mm256_load_ps(sptr1);
				__m256 mr = _mm256_load_ps(sptr2);

				__m256 d0 = _mm256_fmadd_ps(mr, mv[2], _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0])));

				_mm256_store_ps(dptr0, d0);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				dptr0 += 8;
			}
		}
		else if (evec.rows == 2)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb = _mm256_load_ps(sptr0);
				__m256 mg = _mm256_load_ps(sptr1);
				__m256 mr = _mm256_load_ps(sptr2);

				__m256 d0 = _mm256_fmadd_ps(mr, mv[2], _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0])));
				__m256 d1 = _mm256_fmadd_ps(mr, mv[5], _mm256_fmadd_ps(mg, mv[4], _mm256_mul_ps(mb, mv[3])));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				dptr0 += 8;
				dptr1 += 8;
			}
		}
		else if (evec.rows == 3)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 mb = _mm256_load_ps(sptr0);
				__m256 mg = _mm256_load_ps(sptr1);
				__m256 mr = _mm256_load_ps(sptr2);

				__m256 d0 = _mm256_fmadd_ps(mr, mv[2], _mm256_fmadd_ps(mg, mv[1], _mm256_mul_ps(mb, mv[0])));
				__m256 d1 = _mm256_fmadd_ps(mr, mv[5], _mm256_fmadd_ps(mg, mv[4], _mm256_mul_ps(mb, mv[3])));
				__m256 d2 = _mm256_fmadd_ps(mr, mv[8], _mm256_fmadd_ps(mg, mv[7], _mm256_mul_ps(mb, mv[6])));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
	}

	static void projectPCA_4xN_32F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		CV_Assert(src.size() == 4);
		const int sizesimd = src[0].size().area() / 8;
		const float* sptr0 = src[0].ptr<float>();
		const float* sptr1 = src[1].ptr<float>();
		const float* sptr2 = src[2].ptr<float>();
		const float* sptr3 = src[3].ptr<float>();

		AutoBuffer<__m256> mv(4 * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				mv[4 * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		if (evec.rows == 1)
		{
			float* dptr0 = dest[0].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);

				__m256 d0 = _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))));

				_mm256_store_ps(dptr0, d0);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				dptr0 += 8;
			}
		}
		else if (evec.rows == 2)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);

				__m256 d0 = _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))));
				__m256 d1 = _mm256_fmadd_ps(m3, mv[7], _mm256_fmadd_ps(m2, mv[6], _mm256_fmadd_ps(m1, mv[5], _mm256_mul_ps(m0, mv[4]))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				dptr0 += 8;
				dptr1 += 8;
			}
		}
		else if (evec.rows == 3)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);

				__m256 d0 = _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))));
				__m256 d1 = _mm256_fmadd_ps(m3, mv[7], _mm256_fmadd_ps(m2, mv[6], _mm256_fmadd_ps(m1, mv[5], _mm256_mul_ps(m0, mv[4]))));
				__m256 d2 = _mm256_fmadd_ps(m3, mv[11], _mm256_fmadd_ps(m2, mv[10], _mm256_fmadd_ps(m1, mv[9], _mm256_mul_ps(m0, mv[8]))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
		else if (evec.rows == 4)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			float* dptr3 = dest[3].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);

				__m256 d0 = _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))));
				__m256 d1 = _mm256_fmadd_ps(m3, mv[7], _mm256_fmadd_ps(m2, mv[6], _mm256_fmadd_ps(m1, mv[5], _mm256_mul_ps(m0, mv[4]))));
				__m256 d2 = _mm256_fmadd_ps(m3, mv[11], _mm256_fmadd_ps(m2, mv[10], _mm256_fmadd_ps(m1, mv[9], _mm256_mul_ps(m0, mv[8]))));
				__m256 d3 = _mm256_fmadd_ps(m3, mv[15], _mm256_fmadd_ps(m2, mv[14], _mm256_fmadd_ps(m1, mv[13], _mm256_mul_ps(m0, mv[12]))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				_mm256_store_ps(dptr3, d3);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
				dptr3 += 8;
			}
		}
	}

	static void projectPCA_5xN_32F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		CV_Assert(src.size() == 5);
		const int sizesimd = src[0].size().area() / 8;
		const float* sptr0 = src[0].ptr<float>();
		const float* sptr1 = src[1].ptr<float>();
		const float* sptr2 = src[2].ptr<float>();
		const float* sptr3 = src[3].ptr<float>();
		const float* sptr4 = src[4].ptr<float>();

		AutoBuffer<__m256> mv(5 * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				mv[5 * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		if (evec.rows == 1)
		{
			float* dptr0 = dest[0].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);

				__m256 d0 = _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0])))));

				_mm256_store_ps(dptr0, d0);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				dptr0 += 8;
			}
		}
		else if (evec.rows == 2)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);

				__m256 d0 = _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0])))));
				__m256 d1 = _mm256_fmadd_ps(m4, mv[9], _mm256_fmadd_ps(m3, mv[8], _mm256_fmadd_ps(m2, mv[7], _mm256_fmadd_ps(m1, mv[6], _mm256_mul_ps(m0, mv[5])))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				dptr0 += 8;
				dptr1 += 8;
			}
		}
		else if (evec.rows == 3)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);

				__m256 d0 = _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0])))));
				__m256 d1 = _mm256_fmadd_ps(m4, mv[9], _mm256_fmadd_ps(m3, mv[8], _mm256_fmadd_ps(m2, mv[7], _mm256_fmadd_ps(m1, mv[6], _mm256_mul_ps(m0, mv[5])))));
				__m256 d2 = _mm256_fmadd_ps(m4, mv[14], _mm256_fmadd_ps(m3, mv[13], _mm256_fmadd_ps(m2, mv[12], _mm256_fmadd_ps(m1, mv[11], _mm256_mul_ps(m0, mv[10])))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
		else if (evec.rows == 4)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			float* dptr3 = dest[3].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);

				__m256 d0 = _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0])))));
				__m256 d1 = _mm256_fmadd_ps(m4, mv[9], _mm256_fmadd_ps(m3, mv[8], _mm256_fmadd_ps(m2, mv[7], _mm256_fmadd_ps(m1, mv[6], _mm256_mul_ps(m0, mv[5])))));
				__m256 d2 = _mm256_fmadd_ps(m4, mv[14], _mm256_fmadd_ps(m3, mv[13], _mm256_fmadd_ps(m2, mv[12], _mm256_fmadd_ps(m1, mv[11], _mm256_mul_ps(m0, mv[10])))));
				__m256 d3 = _mm256_fmadd_ps(m4, mv[19], _mm256_fmadd_ps(m3, mv[18], _mm256_fmadd_ps(m2, mv[17], _mm256_fmadd_ps(m1, mv[16], _mm256_mul_ps(m0, mv[15])))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				_mm256_store_ps(dptr3, d3);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
				dptr3 += 8;
			}
		}
		else if (evec.rows == 5)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			float* dptr3 = dest[3].ptr<float>();
			float* dptr4 = dest[4].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);

				__m256 d0 = _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0])))));
				__m256 d1 = _mm256_fmadd_ps(m4, mv[9], _mm256_fmadd_ps(m3, mv[8], _mm256_fmadd_ps(m2, mv[7], _mm256_fmadd_ps(m1, mv[6], _mm256_mul_ps(m0, mv[5])))));
				__m256 d2 = _mm256_fmadd_ps(m4, mv[14], _mm256_fmadd_ps(m3, mv[13], _mm256_fmadd_ps(m2, mv[12], _mm256_fmadd_ps(m1, mv[11], _mm256_mul_ps(m0, mv[10])))));
				__m256 d3 = _mm256_fmadd_ps(m4, mv[19], _mm256_fmadd_ps(m3, mv[18], _mm256_fmadd_ps(m2, mv[17], _mm256_fmadd_ps(m1, mv[16], _mm256_mul_ps(m0, mv[15])))));
				__m256 d4 = _mm256_fmadd_ps(m4, mv[24], _mm256_fmadd_ps(m3, mv[23], _mm256_fmadd_ps(m2, mv[22], _mm256_fmadd_ps(m1, mv[21], _mm256_mul_ps(m0, mv[20])))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				_mm256_store_ps(dptr3, d3);
				_mm256_store_ps(dptr4, d4);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
				dptr3 += 8;
				dptr4 += 8;
			}
		}
	}

	static void projectPCA_6xN_32F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		CV_Assert(src.size() == 6);
		const int sizesimd = src[0].size().area() / 8;
		const float* sptr0 = src[0].ptr<float>();
		const float* sptr1 = src[1].ptr<float>();
		const float* sptr2 = src[2].ptr<float>();
		const float* sptr3 = src[3].ptr<float>();
		const float* sptr4 = src[4].ptr<float>();
		const float* sptr5 = src[5].ptr<float>();

		AutoBuffer<__m256> mv(6 * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				mv[6 * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		if (evec.rows == 1)
		{
			float* dptr0 = dest[0].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);
				__m256 m5 = _mm256_load_ps(sptr5);

				__m256 d0 = _mm256_fmadd_ps(m5, mv[5], _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))))));

				_mm256_store_ps(dptr0, d0);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				sptr5 += 8;
				dptr0 += 8;
			}
		}
		else if (evec.rows == 2)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);
				__m256 m5 = _mm256_load_ps(sptr5);

				__m256 d0 = _mm256_fmadd_ps(m5, mv[5], _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))))));
				__m256 d1 = _mm256_fmadd_ps(m5, mv[11], _mm256_fmadd_ps(m4, mv[10], _mm256_fmadd_ps(m3, mv[9], _mm256_fmadd_ps(m2, mv[8], _mm256_fmadd_ps(m1, mv[7], _mm256_mul_ps(m0, mv[6]))))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				sptr5 += 8;
				dptr0 += 8;
				dptr1 += 8;
			}
		}
		else if (evec.rows == 3)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);
				__m256 m5 = _mm256_load_ps(sptr5);

				__m256 d0 = _mm256_fmadd_ps(m5, mv[5], _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))))));
				__m256 d1 = _mm256_fmadd_ps(m5, mv[11], _mm256_fmadd_ps(m4, mv[10], _mm256_fmadd_ps(m3, mv[9], _mm256_fmadd_ps(m2, mv[8], _mm256_fmadd_ps(m1, mv[7], _mm256_mul_ps(m0, mv[6]))))));
				__m256 d2 = _mm256_fmadd_ps(m5, mv[17], _mm256_fmadd_ps(m4, mv[16], _mm256_fmadd_ps(m3, mv[15], _mm256_fmadd_ps(m2, mv[14], _mm256_fmadd_ps(m1, mv[13], _mm256_mul_ps(m0, mv[12]))))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				sptr5 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
			}
		}
		else if (evec.rows == 4)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			float* dptr3 = dest[3].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);
				__m256 m5 = _mm256_load_ps(sptr5);

				__m256 d0 = _mm256_fmadd_ps(m5, mv[5], _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))))));
				__m256 d1 = _mm256_fmadd_ps(m5, mv[11], _mm256_fmadd_ps(m4, mv[10], _mm256_fmadd_ps(m3, mv[9], _mm256_fmadd_ps(m2, mv[8], _mm256_fmadd_ps(m1, mv[7], _mm256_mul_ps(m0, mv[6]))))));
				__m256 d2 = _mm256_fmadd_ps(m5, mv[17], _mm256_fmadd_ps(m4, mv[16], _mm256_fmadd_ps(m3, mv[15], _mm256_fmadd_ps(m2, mv[14], _mm256_fmadd_ps(m1, mv[13], _mm256_mul_ps(m0, mv[12]))))));
				__m256 d3 = _mm256_fmadd_ps(m5, mv[23], _mm256_fmadd_ps(m4, mv[22], _mm256_fmadd_ps(m3, mv[21], _mm256_fmadd_ps(m2, mv[20], _mm256_fmadd_ps(m1, mv[19], _mm256_mul_ps(m0, mv[18]))))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				_mm256_store_ps(dptr3, d3);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				sptr5 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
				dptr3 += 8;
			}
		}
		else if (evec.rows == 5)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			float* dptr3 = dest[3].ptr<float>();
			float* dptr4 = dest[4].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);
				__m256 m5 = _mm256_load_ps(sptr5);

				__m256 d0 = _mm256_fmadd_ps(m5, mv[5], _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))))));
				__m256 d1 = _mm256_fmadd_ps(m5, mv[11], _mm256_fmadd_ps(m4, mv[10], _mm256_fmadd_ps(m3, mv[9], _mm256_fmadd_ps(m2, mv[8], _mm256_fmadd_ps(m1, mv[7], _mm256_mul_ps(m0, mv[6]))))));
				__m256 d2 = _mm256_fmadd_ps(m5, mv[17], _mm256_fmadd_ps(m4, mv[16], _mm256_fmadd_ps(m3, mv[15], _mm256_fmadd_ps(m2, mv[14], _mm256_fmadd_ps(m1, mv[13], _mm256_mul_ps(m0, mv[12]))))));
				__m256 d3 = _mm256_fmadd_ps(m5, mv[23], _mm256_fmadd_ps(m4, mv[22], _mm256_fmadd_ps(m3, mv[21], _mm256_fmadd_ps(m2, mv[20], _mm256_fmadd_ps(m1, mv[19], _mm256_mul_ps(m0, mv[18]))))));
				__m256 d4 = _mm256_fmadd_ps(m5, mv[29], _mm256_fmadd_ps(m4, mv[28], _mm256_fmadd_ps(m3, mv[27], _mm256_fmadd_ps(m2, mv[26], _mm256_fmadd_ps(m1, mv[25], _mm256_mul_ps(m0, mv[24]))))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				_mm256_store_ps(dptr3, d3);
				_mm256_store_ps(dptr4, d4);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				sptr5 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
				dptr3 += 8;
				dptr4 += 8;
			}
		}
		else if (evec.rows == 6)
		{
			float* dptr0 = dest[0].ptr<float>();
			float* dptr1 = dest[1].ptr<float>();
			float* dptr2 = dest[2].ptr<float>();
			float* dptr3 = dest[3].ptr<float>();
			float* dptr4 = dest[4].ptr<float>();
			float* dptr5 = dest[5].ptr<float>();
			for (int i = 0; i < sizesimd; i++)
			{
				__m256 m0 = _mm256_load_ps(sptr0);
				__m256 m1 = _mm256_load_ps(sptr1);
				__m256 m2 = _mm256_load_ps(sptr2);
				__m256 m3 = _mm256_load_ps(sptr3);
				__m256 m4 = _mm256_load_ps(sptr4);
				__m256 m5 = _mm256_load_ps(sptr5);

				__m256 d0 = _mm256_fmadd_ps(m5, mv[5], _mm256_fmadd_ps(m4, mv[4], _mm256_fmadd_ps(m3, mv[3], _mm256_fmadd_ps(m2, mv[2], _mm256_fmadd_ps(m1, mv[1], _mm256_mul_ps(m0, mv[0]))))));
				__m256 d1 = _mm256_fmadd_ps(m5, mv[11], _mm256_fmadd_ps(m4, mv[10], _mm256_fmadd_ps(m3, mv[9], _mm256_fmadd_ps(m2, mv[8], _mm256_fmadd_ps(m1, mv[7], _mm256_mul_ps(m0, mv[6]))))));
				__m256 d2 = _mm256_fmadd_ps(m5, mv[17], _mm256_fmadd_ps(m4, mv[16], _mm256_fmadd_ps(m3, mv[15], _mm256_fmadd_ps(m2, mv[14], _mm256_fmadd_ps(m1, mv[13], _mm256_mul_ps(m0, mv[12]))))));
				__m256 d3 = _mm256_fmadd_ps(m5, mv[23], _mm256_fmadd_ps(m4, mv[22], _mm256_fmadd_ps(m3, mv[21], _mm256_fmadd_ps(m2, mv[20], _mm256_fmadd_ps(m1, mv[19], _mm256_mul_ps(m0, mv[18]))))));
				__m256 d4 = _mm256_fmadd_ps(m5, mv[29], _mm256_fmadd_ps(m4, mv[28], _mm256_fmadd_ps(m3, mv[27], _mm256_fmadd_ps(m2, mv[26], _mm256_fmadd_ps(m1, mv[25], _mm256_mul_ps(m0, mv[24]))))));
				__m256 d5 = _mm256_fmadd_ps(m5, mv[35], _mm256_fmadd_ps(m4, mv[34], _mm256_fmadd_ps(m3, mv[33], _mm256_fmadd_ps(m2, mv[32], _mm256_fmadd_ps(m1, mv[31], _mm256_mul_ps(m0, mv[30]))))));

				_mm256_store_ps(dptr0, d0);
				_mm256_store_ps(dptr1, d1);
				_mm256_store_ps(dptr2, d2);
				_mm256_store_ps(dptr3, d3);
				_mm256_store_ps(dptr4, d4);
				_mm256_store_ps(dptr5, d5);
				sptr0 += 8;
				sptr1 += 8;
				sptr2 += 8;
				sptr3 += 8;
				sptr4 += 8;
				sptr5 += 8;
				dptr0 += 8;
				dptr1 += 8;
				dptr2 += 8;
				dptr3 += 8;
				dptr4 += 8;
				dptr5 += 8;
			}
		}
	}

	//src M x dest N
	static void projectPCA_MxN_32F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		const int src_channels = (int)src.size();
		const int dest_channels = (int)dest.size();

		const int sizesimd = src[0].size().area() / 8;

		AutoBuffer < const float*> sptr(src_channels);
		for (int c = 0; c < src_channels; c++)sptr[c] = src[c].ptr<float>();
		AutoBuffer <float*> dptr(dest_channels);
		for (int c = 0; c < dest_channels; c++)dptr[c] = (float*)dest[c].ptr<float>();

		AutoBuffer<__m256> mv(evec.cols * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < evec.cols; j++)
			{
				mv[evec.cols * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		AutoBuffer<__m256> ms(src_channels);
		for (int i = 0; i < sizesimd; i++)
		{
			for (int c = 0; c < src_channels; c++)
			{
				ms[c] = _mm256_load_ps(sptr[c]);
				sptr[c] += 8;
			}
			for (int d = 0, idx = 0; d < dest_channels; d++)
			{
				__m256 md = _mm256_mul_ps(ms[0], mv[idx++]);
				for (int c = 1; c < src_channels; c++)
				{
					md = _mm256_fmadd_ps(ms[c], mv[idx++], md);
				}
				_mm256_store_ps(dptr[d], md);
				dptr[d] += 8;
			}
		}
	}

	//src M x dest N
	template<int src_channels>
	static void projectPCA_MxN_32F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		const int dest_channels = (int)dest.size();

		const int sizesimd = src[0].size().area() / 8;

		AutoBuffer < const float*> sptr(src_channels);
		for (int c = 0; c < src_channels; c++)sptr[c] = src[c].ptr<float>();
		AutoBuffer <float*> dptr(dest_channels);
		for (int c = 0; c < dest_channels; c++)dptr[c] = (float*)dest[c].ptr<float>();

		AutoBuffer<__m256> mv(evec.cols * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < evec.cols; j++)
			{
				mv[evec.cols * i + j] = _mm256_set1_ps(evec.at<float>(i, j));
			}
		}

		AutoBuffer<__m256> ms(src_channels);
		for (int i = 0; i < sizesimd; i++)
		{
			for (int c = 0; c < src_channels; c++)
			{
				ms[c] = _mm256_load_ps(sptr[c]);
				sptr[c] += 8;
			}
			for (int d = 0, idx = 0; d < dest_channels; d++)
			{
				__m256 md = _mm256_mul_ps(ms[0], mv[idx++]);
				for (int c = 1; c < src_channels; c++)
				{
					md = _mm256_fmadd_ps(ms[c], mv[idx++], md);
				}
				_mm256_store_ps(dptr[d], md);
				dptr[d] += 8;
			}
		}
	}
#pragma endregion
#pragma region projectPCA_MxN_64F
	//src M x dest N
	static void projectPCA_MxN_64F(const vector<Mat>& src, vector<Mat>& dest, const Mat& evec)
	{
		const int src_channels = (int)src.size();
		const int dest_channels = (int)dest.size();

		const int sizesimd = src[0].size().area() / 4;

		AutoBuffer<const double*> sptr(src_channels);
		for (int c = 0; c < src_channels; c++)sptr[c] = src[c].ptr<double>();
		AutoBuffer<double*> dptr(dest_channels);
		for (int c = 0; c < dest_channels; c++)dptr[c] = (double*)dest[c].ptr<double>();

		AutoBuffer<__m256d> mv(evec.cols * evec.rows);
		for (int i = 0; i < evec.rows; i++)
		{
			for (int j = 0; j < evec.cols; j++)
			{
				mv[evec.cols * i + j] = _mm256_set1_pd(evec.at<double>(i, j));
			}
		}

		AutoBuffer<__m256d> ms(src_channels);
		for (int i = 0; i < sizesimd; i++)
		{
			for (int c = 0; c < src_channels; c++)
			{
				ms[c] = _mm256_load_pd(sptr[c]);
				sptr[c] += 4;
			}
			for (int d = 0, idx = 0; d < dest_channels; d++)
			{
				__m256d md = _mm256_mul_pd(ms[0], mv[idx++]);
				for (int c = 1; c < src_channels; c++)
				{
					md = _mm256_fmadd_pd(ms[c], mv[idx++], md);
				}
				_mm256_store_pd(dptr[d], md);
				dptr[d] += 4;
			}
		}
	}
#pragma endregion
	void projectPCA(const vector<Mat>& src, vector<Mat>& dest, const Mat& projectionMatrix)
	{
		const int channels = projectionMatrix.rows;
		dest.resize(channels);

		if (src[0].depth() == CV_32F)
		{
			for (int c = 0; c < channels; c++)
			{
				dest[c].create(src[0].size(), CV_32F);
			}

			if (src.size() == 2) projectPCA_2xN_32F(src, dest, projectionMatrix);
			else if (src.size() == 3) projectPCA_3xN_32F(src, dest, projectionMatrix);
			else if (src.size() == 4) projectPCA_4xN_32F(src, dest, projectionMatrix);
			else if (src.size() == 5) projectPCA_5xN_32F(src, dest, projectionMatrix);
			else if (src.size() == 6) projectPCA_6xN_32F(src, dest, projectionMatrix);
			else if (src.size() == 33) projectPCA_MxN_32F<33>(src, dest, projectionMatrix);
			else projectPCA_MxN_32F(src, dest, projectionMatrix);
		}
		else
		{
			for (int c = 0; c < channels; c++)
			{
				dest[c].create(src[0].size(), CV_64F);
			}
			projectPCA_MxN_64F(src, dest, projectionMatrix);
		}
	}

	static void eigenVecConvert(Mat& evec)
	{
		CV_Assert(evec.depth() == CV_64F);
		for (int j = 0; j < evec.rows; j++)
		{
			double sum = 0.0;
			for (int i = 0; i < evec.cols; i++)
			{
				sum += evec.at<double>(j, i);
			}
			if (sum < 0)
			{
				for (int i = 0; i < evec.cols; i++)
				{
					evec.at<double>(j, i) *= -1.0;
				}
			}
		}
	}

	void cvtColorPCA(InputArray src_, OutputArray dest_, const int dest_channels, Mat& evec, Mat& eval, Mat& mean)
	{
		CV_Assert(src_.depth() == CV_32F || src_.depth() == CV_64F);
		const int depth = src_.depth();
		if (src_.channels() == 1)
		{
			src_.copyTo(dest_);
			return;
		}

		const int channels = min(dest_channels, src_.channels());
		dest_.create(src_.size(), CV_MAKE_TYPE(depth, channels));

		Mat src = src_.getMat();
		Mat dest = dest_.getMat();
		CV_Assert(src.data != dest.data);
		Mat cov;
		if (depth == CV_32F)
		{
			if (channels <= 3 && src_.channels() <= 3)
			{
				{
					//cp::Timer t("cov");
					if (src.channels() == 2) calcCovarMatrix2_32F_(src, cov, mean);
					if (src.channels() == 3) calcCovarMatrix3_32F_(src, cov, mean);
				}
				eigen(cov, eval, evec);
				eigenVecConvert(evec);
				Mat transmat;
				evec(Rect(0, 0, evec.cols, channels)).convertTo(transmat, CV_32F);
				//print_matinfo(transmat);
				{
					if (src.channels() == 2) projectPCA_2xn_32F(src, dest, transmat);
					else if (src.channels() == 3) projectPCA_3xN_32F(src, dest, transmat);
					else projectPCA_MxN_32F(src, dest, transmat);
				}
			}
			else
			{
				//if (src.channels() <= 6)
				{
					vector<Mat> vsrc; split(src, vsrc);
					vector<Mat> vdst;
					cvtColorPCA(vsrc, vdst, dest_channels);
					merge(vdst, dest);
				}
				//else
				{
					//cvtColorPCAOpenCVCovMat(src, dest, channels, eval, evec);
				}
			}
		}
		else if (depth == CV_64F)
		{
			vector<Mat> vsrc; split(src, vsrc);
			vector<Mat> vdst;
			cvtColorPCA(vsrc, vdst, dest_channels);
			merge(vdst, dest);
			{
				//if (src.channels() <= 6)
				{

				}
				//else
				{
					//cvtColorPCAOpenCVCovMat(src, dest, channels, eval, evec);
				}
			}
		}
	}

	template<typename T>
	void eigenvectorWeightedBlend(Mat& src_, Mat& dest, Mat& eval, const int dest_channels)
	{
		Mat src = src_.clone();

		vector <double> w(src.rows);

		/*for (int i = dest_channels - 1; i < src.rows; i++)
		{
			w[i] = pow(eval.at<T>(i),4);
		}*/

		double etotal = 1.0 + pow(eval.at<T>(dest_channels) / eval.at<T>(dest_channels - 1), 1.8);

		if (dest_channels == src.rows)etotal = 1.0;
		/*for (int i = dest_channels - 1; i < src.rows; i++)
		{
			etotal += w[i];
		}*/
		//etotal *= 0.95;

		for (int j = 0; j < src.rows; j++)
		{
			if (j < dest_channels - 1)
			{
				for (int i = 0; i < src.cols; i++)
				{
					dest.at<T>(j, i) = src.at<T>(j, i);
				}
			}
			else //last channel
			{
				for (int i = 0; i < src.cols; i++)
				{
					double v = 0.0;
					/*for (int k = dest_channels - 1; k < src.rows; k++)
					{
						v += src.at<T>(k, i) * w[k];
					}*/
					dest.at<T>(j, i) = src.at<T>(j, i) * etotal;
				}
			}
		}
	}

	void cvtColorPCA(InputArray src, OutputArray dest, const int dest_channels)
	{
		Mat evec;
		Mat eval;
		Mat mean;
		cvtColorPCA(src, dest, dest_channels, evec, eval, mean);
	}

	void cvtColorPCA2(InputArray src_, OutputArray dest_, const int dest_channels, Mat& evec, Mat& eval, Mat& mean)
	{
		CV_Assert(src_.depth() == CV_32F);

		if (src_.channels() == 1)
		{
			src_.copyTo(dest_);
			return;
		}

		const int channels = min(dest_channels, src_.channels());
		dest_.create(src_.size(), CV_MAKE_TYPE(CV_32F, channels));

		Mat src = src_.getMat();
		Mat dest = dest_.getMat();
		Mat cov;
		if (channels <= 3 && src_.channels() <= 3)
		{
			{
				//cp::Timer t("cov");
				if (src.channels() == 2) calcCovarMatrix2_32F_(src, cov, mean);
				if (src.channels() == 3) calcCovarMatrix3_32F_(src, cov, mean);
			}

			eigen(cov, eval, evec);
			eigenVecConvert(evec);

			eigenvectorWeightedBlend<double>(evec, evec, eval, channels);
			Mat transmat;
			evec(Rect(0, 0, evec.cols, channels)).convertTo(transmat, CV_32F);


			{
				if (src.channels() == 2) projectPCA_2xn_32F(src, dest, transmat);
				else if (src.channels() == 3) projectPCA_3xN_32F(src, dest, transmat);
				else projectPCA_MxN_32F(src, dest, transmat);
			}
		}
		else
		{
			//if (src.channels() <= 6)
			{
				vector<Mat> vsrc; split(src, vsrc);
				vector<Mat> vdst;
				cvtColorPCA(vsrc, vdst, dest_channels);
				merge(vdst, dest);
			}
			//else
			{
				//cvtColorPCAOpenCVCovMat(src, dest, channels, eval, evec);
			}
		}
	}

	void cvtColorPCA2(InputArray src, OutputArray dest, const int dest_channels)
	{
		Mat evec;
		Mat eval;
		Mat mean;
		cvtColorPCA2(src, dest, dest_channels, evec, eval, mean);
	}

	double cvtColorPCAErrorPSNR(const vector<Mat>& src, const int dest_channels)
	{
		CV_Assert(src[0].depth() == CV_32F);

		vector<Mat> dest;
		if (src.size() == 1)
		{
			dest.resize(1);
			src[0].copyTo(dest[0]);
			return 0.0;
		}

		const int channels = min(dest_channels, (int)src.size());
		dest.resize(src.size());
		for (int c = 0; c < (int)src.size(); c++)
		{
			dest[c].create(src[0].size(), CV_32F);
		}

		Mat cov;
		{
			//cp::Timer t("cov");
			if (src.size() == 2) calcCovarMatrix2_32F_(src, cov);
			else if (src.size() == 3) calcCovarMatrix3_32F_(src, cov);
			else if (src.size() == 4) calcCovarMatrix4_32F_(src, cov);
			else if (src.size() == 5) calcCovarMatrix5_32F_(src, cov);
			else if (src.size() == 6) calcCovarMatrix6_32F_(src, cov);
			else if (src.size() == 33) calcCovarMatrix__32F_<33>(src, cov);
			else calcCovarMatrixN_32F_(src, cov);
		}

		Mat eval, evec;
		eigen(cov, eval, evec);

		Mat transmat;
		evec.convertTo(transmat, CV_32F);
		if (src.size() == 2) projectPCA_2xN_32F(src, dest, transmat);
		else if (src.size() == 3) projectPCA_3xN_32F(src, dest, transmat);
		else if (src.size() == 4) projectPCA_4xN_32F(src, dest, transmat);
		else if (src.size() == 5) projectPCA_5xN_32F(src, dest, transmat);
		else if (src.size() == 6) projectPCA_6xN_32F(src, dest, transmat);
		else if (src.size() == 33) projectPCA_MxN_32F<33>(src, dest, transmat);
		else  projectPCA_MxN_32F(src, dest, transmat);

		const int s = (int)src.size() - channels;
		vector<Mat> a(s);
		vector<Mat> b(s);
		for (int i = 0; i < s; i++)
		{
			a[i] = dest[i + channels];
			b[i] = Mat::zeros(src[0].size(), CV_32F);
		}

		if (s == 0) return 0.0;
		Mat v0, v1;
		cp::concat(a, v0, s);
		cp::concat(b, v1, s);
		return cp::getPSNR(v0, v1);
	}

	double cvtColorPCAErrorPSNR(const Mat& src, const int dest_channels)
	{
		vector<Mat> vsrc; split(src, vsrc);
		return cvtColorPCAErrorPSNR(vsrc, dest_channels);
	}


	static void sortEigenVecVal(Mat& vec, Mat& val)
	{
		CV_Assert(vec.depth() == CV_64F);
		CV_Assert(val.depth() == CV_64F);
		//ordering check
		bool ret = true;
		for (int i = 0; i < val.size().area() - 1; i++)
		{
			if (abs(val.at<double>(i)) < abs(val.at<double>(i + 1)))
			{
				ret = false;
			}
		}
		if (ret)return;

		Mat a = vec.clone();
		Mat b = val.clone();
		for (int j = 0; j < b.size().area(); j++)
		{
			double maxval = 0.0;
			int argmax = 0;
			for (int i = 0; i < b.size().area(); i++)
			{
				if (maxval < abs(b.at<double>(i)))
				{
					maxval = abs(b.at<double>(i));
					argmax = i;
				}
			}
			val.at <double>(j) = abs(b.at<double>(argmax));
			double* src = a.ptr<double>(argmax);
			double* dst = vec.ptr<double>(j);
			if (b.at<double>(argmax) >= 0)
			{
				for (int i = 0; i < val.cols; i++)
				{
					dst[i] = src[i];
				}
			}
			else
			{
				for (int i = 0; i < val.cols; i++)
				{
					dst[i] = -src[i];
				}
			}
			b.at<double>(argmax) = 0.0;
		}
	}

	void computePCA(const vector<Mat>& src, Mat& evec, Mat& eval)
	{
		Mat cov;
		if (src[0].depth() == CV_32F)
		{
			//cp::Timer t("cov");
			if (src.size() == 2) calcCovarMatrix2_32F_(src, cov);
			else if (src.size() == 3) calcCovarMatrix3_32F_(src, cov);
			else if (src.size() == 4) calcCovarMatrix4_32F_(src, cov);
			else if (src.size() == 5) calcCovarMatrix5_32F_(src, cov);
			else if (src.size() == 6) calcCovarMatrix6_32F_(src, cov);
			else if (src.size() == 33) calcCovarMatrix__32F_<33>(src, cov);
			else calcCovarMatrixN_32F_(src, cov);
		}
		else if (src[0].depth() == CV_64F)
		{
			calcCovarMatrixN_64F_(src, cov);
		}
		eigen(cov, eval, evec);
		sortEigenVecVal(evec, eval);
	}

	void cvtColorPCA(const vector<Mat>& src, vector<Mat>& dest, const int dest_channels, Mat& projectionMatrix, Mat& eval, Mat& mean)
	{
		CV_Assert(src[0].depth() == CV_32F || src[0].depth() == CV_64F);

		if (src.size() == 1)
		{
			dest.resize(1);
			src[0].copyTo(dest[0]);
			return;
		}

		const int channels = min(dest_channels, (int)src.size());

		Mat evec;
		computePCA(src, evec, eval);
		evec(Rect(0, 0, evec.cols, channels)).convertTo(projectionMatrix, src[0].depth());
		projectPCA(src, dest, projectionMatrix);
	}

	void cvtColorPCA(const vector<Mat>& src, vector<Mat>& dest, const int dest_channels, Mat& projectionMatrix, Mat& eval)
	{
		Mat mean;
		cvtColorPCA(src, dest, dest_channels, projectionMatrix, eval, mean);
	}
	void cvtColorPCA(const vector<Mat>& src, vector<Mat>& dest, const int dest_channels, Mat& projectionMatrix)
	{
		Mat mean;
		Mat eval;
		cvtColorPCA(src, dest, dest_channels, projectionMatrix, eval, mean);
	}
	void cvtColorPCA(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, const int dest_channels)
	{
		Mat projectionMatrix;
		Mat eval;
		Mat mean;
		cvtColorPCA(src, dest, dest_channels, projectionMatrix, eval, mean);
	}

	void guiSplit(InputArray src, string wname)
	{
		namedWindow(wname);
		static int ch_guiSplit = 0; createTrackbar("ch", wname, &ch_guiSplit, src.channels() - 1);
		vector<Mat> sp;
		split(src, sp);
		Mat show;
		int key = 0;
		while (key != 'q')
		{
			sp[ch_guiSplit].convertTo(show, CV_8U);
			imshow(wname, show);
			key = waitKey(1);
		}
		destroyWindow(wname);
	}


	template<typename S, typename D>
	void cvtColorHSI2BGR_round(Mat& src, Mat& dest)
	{
		const int channel = src.channels();

		const int bc = channel / 3;
		const int gc = 2 * bc;
		const int size = src.size().area();

		S* s = src.ptr <S>();
		D* d = dest.ptr<D>();
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			double b = 0.0;
			double g = 0.0;
			double r = 0.0;
			for (int c = 0; c < bc; c++)
			{
				b += (double)s[channel * i + c + 0];
				g += (double)s[channel * i + c + bc];
				r += (double)s[channel * i + c + gc];
			}
			d[3 * i + 0] = (D)cvRound(b / bc);
			d[3 * i + 1] = (D)cvRound(g / bc);
			d[3 * i + 2] = (D)cvRound(r / bc);
		}
	}

	template<typename S, typename D>
	void cvtColorHSI2BGR_(Mat& src, Mat& dest)
	{
		const int channel = src.channels();

		const int bc = channel / 3;
		const int gc = 2 * bc;
		const int size = src.size().area();

		S* s = src.ptr <S>();
		D* d = dest.ptr<D>();
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			double b = 0.0;
			double g = 0.0;
			double r = 0.0;
			for (int c = 0; c < bc; c++)
			{
				b += (double)s[channel * i + c + 0];
				g += (double)s[channel * i + c + bc];
				r += (double)s[channel * i + c + gc];
			}
			d[3 * i + 0] = (D)(b / bc);
			d[3 * i + 1] = (D)(g / bc);
			d[3 * i + 2] = (D)(r / bc);
		}
	}

	void cvtColorHSI2BGR(Mat& src, Mat& dest, const int depth)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);
		CV_Assert(depth == CV_8U || depth == CV_32F || depth == CV_64F);
		dest.create(src.size(), CV_MAKETYPE(depth, 3));

		if (depth == CV_8U)
		{
			if (src.depth() == CV_8U)cvtColorHSI2BGR_round<uchar, uchar>(src, dest);
			else if (src.depth() == CV_32F)cvtColorHSI2BGR_round<float, uchar>(src, dest);
			else if (src.depth() == CV_64F)cvtColorHSI2BGR_round<double, uchar>(src, dest);
		}
		else if (depth == CV_32F)
		{
			if (src.depth() == CV_8U)cvtColorHSI2BGR_<uchar, float>(src, dest);
			else if (src.depth() == CV_32F)cvtColorHSI2BGR_<float, float>(src, dest);
			else if (src.depth() == CV_64F)cvtColorHSI2BGR_<double, float>(src, dest);
		}
		else if (depth == CV_64F)
		{
			if (src.depth() == CV_8U)cvtColorHSI2BGR_<uchar, double>(src, dest);
			else if (src.depth() == CV_32F)cvtColorHSI2BGR_<float, double>(src, dest);
			else if (src.depth() == CV_64F)cvtColorHSI2BGR_<double, double>(src, dest);
		}
	}

#pragma endregion

	//TODO: support OpenCV 4
#if CV_MAJOR_VERSION == 3
	void xcvFindWhiteBlanceMatrix(IplImage* src, CvMat* C, IplImage* mask)
	{
		uchar* data = (uchar*)src->imageData;
		int size = src->width * src->height;
		unsigned char* mdata = new unsigned char[size];
		memset(mdata, 1, size);

		uchar* m = mdata;
		if (mask != NULL)
		{
			m = (uchar*)mask->imageData;
		}

		int count = 0;
		for (int j = 0; j < size; j++)
		{
			if (m[j] != 0)
			{
				count++;
			}
		}

		CvMat* abr = cvCreateMat(2, 1, CV_64F);
		CvMat* abb = cvCreateMat(2, 1, CV_64F);
		double rxr, bxb, r, b, rsd, bsd, dtr, dtb;
		rxr = bxb = r = b = rsd = bsd = dtr = dtb = 0.0;
		for (int j = 0; j < size; j++)
		{
			if (m[j] != 0)
			{
				double rs = data[3 * j + 2];
				double gs = data[3 * j + 1];
				double bs = data[3 * j + 0];

				rxr += rs * rs;
				bxb += bs * bs;

				r += rs;
				b += bs;

				dtr += gs;
				dtb += gs;

				rsd += rs * gs;
				bsd += bs * gs;
			}
		}

		CvMat* Am = cvCreateMat(2, 2, CV_64F);
		CvMat* Bm = cvCreateMat(2, 1, CV_64F);

		cvmSet(Am, 0, 0, rxr);
		cvmSet(Am, 0, 1, r);
		cvmSet(Am, 1, 0, r);
		cvmSet(Am, 1, 1, count);

		cvmSet(Bm, 0, 0, rsd);
		cvmSet(Bm, 0, 1, dtr);
		cvSolve(Am, Bm, abr, CV_LU);

		cvmSet(Am, 0, 0, bxb);
		cvmSet(Am, 0, 1, b);
		cvmSet(Am, 1, 0, b);
		cvmSet(Am, 1, 1, count);

		cvmSet(Bm, 0, 0, bsd);
		cvmSet(Bm, 0, 1, dtb);
		cvSolve(Am, Bm, abb, CV_LU);

		cvReleaseMat(&Am);
		cvReleaseMat(&Bm);

		//printf("R: a =  %f b = %f\n",cvmGet(abr,0,0),cvmGet(abr,1,0));
		//printf("B: a =  %f b = %f\n",cvmGet(abb,0,0),cvmGet(abb,1,0));

		cvSetIdentity(C);
		cvmSet(C, 0, 0, cvmGet(abb, 0, 0));
		cvmSet(C, 0, 3, cvmGet(abb, 1, 0));
		cvmSet(C, 2, 2, cvmGet(abr, 0, 0));
		cvmSet(C, 2, 3, cvmGet(abr, 1, 0));

		cvReleaseMat(&abr);
		cvReleaseMat(&abb);
		delete[] mdata;
	}

	/*!
	* \brief
	* find white balance color matrix
	*
	* \param src
	* input image
	*
	* \param C
	* output 3x4 color matrix, if C is not allocated, matrix will be allocated automatically
	*
	* \param mask
	* masking data for WB
	*
	* \remarks
	* Write remarks for findWhiteBlanceMatrix here.
	*
	* \see
	* Separate items with the '|' character.
	*/
	void findWhiteBlanceMatrix(Mat& src, Mat& C, Mat& mask)
	{
		if (C.empty())C.create(3, 4, CV_64F);
		if (mask.empty())
			xcvFindWhiteBlanceMatrix(&IplImage(src), &CvMat(C), NULL);
		else
			xcvFindWhiteBlanceMatrix(&IplImage(src), &CvMat(C), &IplImage(mask));
	}

	/*!
	* \brief
	* find color correct matrix from images. im1 = C im2
	*
	* \param srcim1
	* Description of parameter srcim1.
	*
	* \param srcim2
	* Description of parameter srcim2.
	*
	* \param C
	* Description of parameter C.
	*
	* \throws <exception class>
	* Description of criteria for throwing this exception.
	*
	* Write detailed description for findColorMatrixAvgSdv here.
	*
	* \remarks
	* findColorMatrix using corresponding point crowds is more accurate but this function is robust.
	*
	* \see
	* Separate items with the '|' character.
	*/

	void xcvFindColorMatrix(CvMat* _src, CvMat* _dest, CvMat* C)
	{
		CvMat* ret = C;
		CvMat* cmat = cvCreateMat(12, 1, CV_64F);

		CvMat* src = _src;
		CvMat* dest = _dest;

		int num = _src->rows;

		CvMat* y = cvCreateMat(3 * num, 1, CV_64F);
		CvMat* A = cvCreateMat(3 * num, 12, CV_64F);
		//cvZero(A);
		for (int i = 0; i < num; i++)
		{
			cvmSet(A, 3 * i + 0, 0, cvmGet(src, i, 0));
			cvmSet(A, 3 * i + 0, 1, cvmGet(src, i, 1));
			cvmSet(A, 3 * i + 0, 2, cvmGet(src, i, 2));
			cvmSet(A, 3 * i + 0, 3, 1.0);

			cvmSet(A, 3 * i + 1, 4, cvmGet(src, i, 0));
			cvmSet(A, 3 * i + 1, 5, cvmGet(src, i, 1));
			cvmSet(A, 3 * i + 1, 6, cvmGet(src, i, 2));
			cvmSet(A, 3 * i + 1, 7, 1.0);

			cvmSet(A, 3 * i + 2, 8, cvmGet(src, i, 0));
			cvmSet(A, 3 * i + 2, 9, cvmGet(src, i, 1));
			cvmSet(A, 3 * i + 2, 10, cvmGet(src, i, 2));
			cvmSet(A, 3 * i + 2, 11, 1.0);

			cvmSet(y, 3 * i + 0, 0, cvmGet(dest, i, 0));
			cvmSet(y, 3 * i + 1, 0, cvmGet(dest, i, 1));
			cvmSet(y, 3 * i + 2, 0, cvmGet(dest, i, 2));
		}
		cvSolve(A, y, cmat, CV_SVD);

		cvmSet(C, 0, 0, cvmGet(cmat, 0, 0));
		cvmSet(C, 0, 1, cvmGet(cmat, 1, 0));
		cvmSet(C, 0, 2, cvmGet(cmat, 2, 0));
		cvmSet(C, 0, 3, cvmGet(cmat, 3, 0));

		cvmSet(C, 1, 0, cvmGet(cmat, 4, 0));
		cvmSet(C, 1, 1, cvmGet(cmat, 5, 0));
		cvmSet(C, 1, 2, cvmGet(cmat, 6, 0));
		cvmSet(C, 1, 3, cvmGet(cmat, 7, 0));

		cvmSet(C, 2, 0, cvmGet(cmat, 8, 0));
		cvmSet(C, 2, 1, cvmGet(cmat, 9, 0));
		cvmSet(C, 2, 2, cvmGet(cmat, 10, 0));
		cvmSet(C, 2, 3, cvmGet(cmat, 11, 0));


		cvReleaseMat(&y);
		cvReleaseMat(&A);
		cvReleaseMat(&cmat);
	}

	void findColorMatrix(Mat& src_point_crowd1, Mat& src_point_crowd2, Mat& C)
	{
		if (C.empty())C.create(3, 4, CV_64F);

		xcvFindColorMatrix(&CvMat(src_point_crowd1), &CvMat(src_point_crowd2), &CvMat(C));
	}
#endif
}