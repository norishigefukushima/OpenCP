#include "bitconvert.hpp"

using namespace cv;

namespace cp
{

	void cvt32f8u(const Mat& src, Mat& dest)
	{
		if (dest.empty()) dest.create(src.size(), CV_8U);

		const int imsize = src.size().area() / 16;
		const int nn = src.size().area() - imsize * 16;
		float* s = (float*)src.ptr<float>(0);
		uchar* d = dest.ptr<uchar>(0);

		for (int i = imsize; i--;)
		{
			__m128 src128 = _mm_load_ps(s);
			__m128i src_int128 = _mm_cvtps_epi32(src128);

			src128 = _mm_load_ps(s + 4);
			__m128i src1_int128 = _mm_cvtps_epi32(src128);

			__m128i src2_int128 = _mm_packs_epi32(src_int128, src1_int128);

			src128 = _mm_load_ps(s + 8);
			src_int128 = _mm_cvtps_epi32(src128);

			src128 = _mm_load_ps(s + 12);
			src1_int128 = _mm_cvtps_epi32(src128);

			src1_int128 = _mm_packs_epi32(src_int128, src1_int128);

			src1_int128 = _mm_packus_epi16(src2_int128, src1_int128);

			_mm_store_si128((__m128i*)(d), src1_int128);

			s += 16;
			d += 16;
		}
		for (int i = 0; i < nn; i++)
		{
			*d = (uchar)(*s + 0.5f);
			s++, d++;
		}
	}

	void cvt8u32f(const Mat& src, Mat& dest, const float amp)
	{
		const int imsize = src.size().area() / 8;
		const int nn = src.size().area() - imsize * 8;
		uchar* s = (uchar*)src.ptr(0);
		float* d = dest.ptr<float>(0);
		const __m128 mamp = _mm_set_ps1(amp);
		const __m128i zero = _mm_setzero_si128();
		for (int i = imsize; i--;)
		{
			__m128i s1 = _mm_loadl_epi64((__m128i*)s);

			_mm_store_ps(d, _mm_mul_ps(mamp, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(s1))));
			_mm_store_ps(d + 4, _mm_mul_ps(mamp, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(s1, 4)))));
			s += 8;
			d += 8;
		}
		for (int i = 0; i < nn; i++)
		{
			*d = (float)*s * amp;
			s++, d++;
		}
	}

	void cvt8u32f(const Mat& src, Mat& dest)
	{
		if (dest.empty()) dest.create(src.size(), CV_32F);
		const int imsize = src.size().area() / 8;
		const int nn = src.size().area() - imsize * 8;
		uchar* s = (uchar*)src.ptr(0);
		float* d = dest.ptr<float>(0);
		const __m128i zero = _mm_setzero_si128();
		for (int i = imsize; i--;)
		{
			__m128i s1 = _mm_loadl_epi64((__m128i*)s);

			_mm_store_ps(d, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(s1)));
			_mm_store_ps(d + 4, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(s1, 4))));
			s += 8;
			d += 8;
		}
		for (int i = 0; i < nn; i++)
		{
			*d = (float)*s;
			s++, d++;
		}
	}

	void cvt32F16F(cv::Mat& srcdst)
	{
		float* s = srcdst.ptr<float>(0);
		for (int i = 0; i < srcdst.size().area(); i += 8)
		{
			__m256 mv = _mm256_load_ps(s + i);
			__m128i ms = _mm256_cvtps_ph(mv, 0);
			mv = _mm256_cvtph_ps(ms);
			_mm256_store_ps(s + i, mv);
		}
	}
}