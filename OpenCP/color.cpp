#include "color.hpp"
#include "arithmetic.hpp"
#include "inlineSIMDfunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
#pragma region merge
	template<class T>
	void mergeBase_(vector<Mat>& src, Mat& dest)
	{
		if (dest.empty())dest.create(src[0].size(), src[0].depth());
		const int size = src[0].size().area();
		T* d = dest.ptr<T>(0);
		T* s0 = src[0].ptr<T>(0);
		T* s1 = src[1].ptr<T>(0);
		T* s2 = src[2].ptr<T>(0);
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

	void mergeStore_8U(vector<Mat>& src, Mat& dest)
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

	void mergeStream_8U(vector<Mat>& src, Mat& dest)
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
		if (dest.empty())
		{
			dest.create(s[0].size(), s[0].type());
		}
		Mat dst = dest.getMat();

		switch (s[0].depth())
		{
		case CV_8U:
			if (isCache)mergeStore_8U(s, dst);
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
#pragma endregion

#pragma region split
	template<class T>
	void splitBase_(Mat& src, vector<Mat>& dst)
	{
		const int size = src.size().area();
		T* s = src.ptr<T>(0);
		T* d0 = dst[0].ptr<T>(0);
		T* d1 = dst[1].ptr<T>(0);
		T* d2 = dst[2].ptr<T>(0);
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
		if constexpr (store_method == STOREU) _mm256_storeu_ps(dst, src);
		else if constexpr (store_method == STORE) _mm256_store_ps(dst, src);
		else if constexpr (store_method == STREAM) _mm256_stream_ps(dst, src);
	}

	template<int store_method>
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

#pragma endregion

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

		for (int j = 0; j < src.rows; j++)
		{
			int i = 0;
			for (; i < src.cols; i += 16)
			{
				a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i)), mask1);
				b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i + 16)), mask2);
				c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i + 32)), mask2);
				_mm_stream_si128((__m128i*)(B + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask1), bmask2));

				a = _mm_shuffle_epi8(a, smask1);
				b = _mm_shuffle_epi8(b, smask1);
				c = _mm_shuffle_epi8(c, ssmask1);
				_mm_stream_si128((__m128i*)(G + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask2));

				a = _mm_shuffle_epi8(a, ssmask1);
				c = _mm_shuffle_epi8(c, ssmask1);
				b = _mm_shuffle_epi8(b, ssmask2);

				_mm_stream_si128((__m128i*)(R + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask4));
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

	void splitBGRLineInterleave(cv::InputArray src_, cv::OutputArray dest_)
	{
		dest_.create(Size(src_.size().width, src_.size().height * 3), src_.depth());
		Mat src = src_.getMat();
		Mat dest = dest_.getMat();
		if (src.type() == CV_MAKE_TYPE(CV_8U, 3))
		{
			CV_Assert(src.cols % 16 == 0);
			splitBGRLineInterleave_8u(src, dest);
		}
		else if (src.type() == CV_MAKE_TYPE(CV_32F, 3))
		{
			CV_Assert(src.cols % 4 == 0);
			splitBGRLineInterleave_32f(src, dest);
		}
		else
		{
			CV_Assert(src.cols % 4 == 0);
			splitBGRLineInterleave_32fcast(src, dest);
		}
	}

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

	template <class T>
	void cvtColorBGR2PLANE_(const Mat& src, Mat& dest, int depth)
	{
		vector<Mat> v(3);
		split(src, v);
		dest.create(Size(src.cols, src.rows * 3), depth);

		memcpy(dest.data, v[0].data, src.size().area() * sizeof(T));
		memcpy(dest.data + src.size().area() * sizeof(T), v[1].data, src.size().area() * sizeof(T));
		memcpy(dest.data + 2 * src.size().area() * sizeof(T), v[2].data, src.size().area() * sizeof(T));
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

	template <class T>
	void cvtColorPLANE2BGR_(const Mat& src, Mat& dest, int depth)
	{
		int width = src.cols;
		int height = src.rows / 3;
		T* b = (T*)src.ptr<T>(0);
		T* g = (T*)src.ptr<T>(height);
		T* r = (T*)src.ptr<T>(2 * height);

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
		if (dest.empty())dest.create(src.size(), CV_32FC4);

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

	void cvtColorBGR2BGRA(const Mat& src, Mat& dest, const uchar alpha)
	{
		if (dest.empty())dest.create(src.size(), CV_8UC4);

		int size = src.size().area();
		uchar* s = (uchar*)src.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);

		for (int i = 0; i < size; i++)
		{
			*d++ = *s++;
			*d++ = *s++;
			*d++ = *s++;
			*d++ = alpha;
		}
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

	void cvtColorBGRA2BGR(const Mat& src, Mat& dest)
	{
		CV_Assert(src.type() == CV_8UC4);
		if (dest.empty())dest.create(src.size(), CV_8UC3);

		int size = src.size().area();
		uchar* s = (uchar*)src.ptr<uchar>(0);
		uchar* d = dest.ptr<uchar>(0);

		for (int i = 0; i < size; i++)
		{
			*d++ = *s++;
			*d++ = *s++;
			*d++ = *s++;
			*s++;
		}
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