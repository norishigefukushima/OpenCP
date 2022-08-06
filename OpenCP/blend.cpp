#include "imshowExtension.hpp"
#include "blend.hpp"
#include "imagediff.hpp"
#include "matinfo.hpp"
#include "metrics.hpp"
#include "inlineSIMDFunctions.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	static void alphaBlend_AVX_8U(Mat& src1, Mat& src2, const float alpha, Mat& dst)
	{
		int size = src1.size().area() * src1.channels();
		int simdsize = get_simd_floor(size, 16);
		uchar* s1 = src1.data;
		uchar* s2 = src2.data;
		uchar* d = dst.data;

		const __m256 ma = _mm256_set1_ps(alpha);
		for (int i = 0; i < simdsize; i += 16)
		{
			__m256 ms1 = _mm256_load_epu8cvtps((__m128i*)s1);
			__m256 ms2 = _mm256_load_epu8cvtps((__m128i*)s2);
			__m256 mt1 = _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2);

			ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 8));
			ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 8));
			__m256 mt2 = _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2);

			_mm_store_si128((__m128i*)d, _mm256_cvtpsx2_epu8(mt1, mt2));

			s1 += 16;
			s2 += 16;
			d += 16;
		}

		s1 = src1.data;
		s2 = src2.data;
		d = dst.data;
		for (int i = simdsize; i < size; i++)
		{
			d[i] = saturate_cast<uchar>(alpha * (s1[i] - s2[i]) + s2[i]);
		}
	}

	static void alphaBlend_AVX_32F(Mat& src1, Mat& src2, const float alpha, Mat& dst)
	{
		int size = src1.size().area() * src1.channels();
		int simdsize = get_simd_floor(size, 8);
		float* s1 = src1.ptr<float>();
		float* s2 = src2.ptr<float>();
		float* d = dst.ptr<float>();

		const __m256 ma = _mm256_set1_ps(alpha);
		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 ms1 = _mm256_load_ps(s1);
			__m256 ms2 = _mm256_load_ps(s2);
			_mm256_store_ps(d, _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2));

			s1 += 8;
			s2 += 8;
			d += 8;
		}

		s1 = src1.ptr<float>();
		s2 = src2.ptr<float>();
		d = dst.ptr<float>();
		for (int i = simdsize; i < size; i++)
		{
			d[i] = alpha * (s1[i] - s2[i]) + s2[i];
		}
	}

	static void alphaBlend_AVX_64F(Mat& src1, Mat& src2, const double alpha, Mat& dst)
	{
		int size = src1.size().area() * src1.channels();
		int simdsize = get_simd_floor(size, 8);
		double* s1 = src1.ptr<double>();
		double* s2 = src2.ptr<double>();
		double* d = dst.ptr<double>();

		const __m256d ma = _mm256_set1_pd(alpha);
		const double ia = 1.0 - alpha;
		const __m256d mia = _mm256_set1_pd(ia);
		for (int i = 0; i < simdsize; i += 8)
		{
			__m256d ms1 = _mm256_load_pd(s1);
			__m256d ms2 = _mm256_load_pd(s2);
			_mm256_store_pd(d, _mm256_fmadd_pd(ma, ms1, _mm256_mul_pd(mia, ms2)));

			ms1 = _mm256_load_pd(s1 + 4);
			ms2 = _mm256_load_pd(s2 + 4);
			_mm256_store_pd(d + 4, _mm256_fmadd_pd(ma, ms1, _mm256_mul_pd(mia, ms2)));

			s1 += 8;
			s2 += 8;
			d += 8;
		}

		s1 = src1.ptr<double>();
		s2 = src2.ptr<double>();
		d = dst.ptr<double>();
		for (int i = simdsize; i < size; i++)
		{
			d[i] = alpha * s1[i] + ia * s2[i];
		}
	}

	void alphaBlend(InputArray src1, InputArray src2, const double alpha, OutputArray dest)
	{
		CV_Assert(src1.size() == src2.size());
		Mat s1, s2;
		const int depth = max(src1.depth(), src2.depth());
		const int channel = max(src1.channels(), src2.channels());
		if (src1.depth() == src2.depth())
		{
			if (src1.channels() == src2.channels())
			{
				s1 = src1.getMat();
				s2 = src2.getMat();
			}
			else if (src2.channels() == 3)
			{
				cvtColor(src1, s1, COLOR_GRAY2BGR);
				s2 = src2.getMat();
			}
			else
			{
				cvtColor(src2, s2, COLOR_GRAY2BGR);
				s1 = src1.getMat();
			}
		}
		else if (src1.depth() != src2.depth())
		{
			if (src1.channels() == src2.channels())
			{
				s1 = src1.getMat();
				s2 = src2.getMat();
			}
			else if (src2.channels() == 3)
			{
				cvtColor(src1, s1, COLOR_GRAY2BGR);
				s2 = src2.getMat();
			}
			else
			{
				cvtColor(src2, s2, COLOR_GRAY2BGR);
				s1 = src1.getMat();
			}
			s1.convertTo(s1, depth);
			s2.convertTo(s2, depth);
		}

		dest.create(src1.size(), CV_MAKETYPE(depth, channel));
		Mat dst = dest.getMat();

		if (depth == CV_8U)
		{
			alphaBlend_AVX_8U(s1, s2, (float)alpha, dst);
		}
		else if (depth == CV_32F)
		{
			alphaBlend_AVX_32F(s1, s2, (float)alpha, dst);
		}
		else if (depth == CV_64F)
		{
			alphaBlend_AVX_64F(s1, s2, alpha, dst);
		}
		else
		{
			cv::addWeighted(s1, alpha, s2, 1.0 - alpha, 0.0, dest);
		}
	}


	static void alphaBlendFixedPoint_AVX(Mat& src1, Mat& src2, int alpha, Mat& dst)
	{
		int size = src1.size().area() * src1.channels();
		int simdsize = get_simd_floor(size, 32);
		uchar* s1 = src1.data;
		uchar* s2 = src2.data;
		uchar* d = dst.data;

		const __m256i ma = _mm256_set1_epi16(alpha << 7);
		//a*(s1-s2)+s2
		for (int i = 0; i < simdsize; i += 32)
		{
			__m256i ms1 = _mm256_load_epu8cvtepi16((__m128i*)s1);
			__m256i ms2 = _mm256_load_epu8cvtepi16((__m128i*)s2);
			_mm_store_si128((__m128i*)d, _mm256_cvtepi16_epu8(_mm256_add_epi16(ms2, _mm256_mulhrs_epi16(ma, _mm256_sub_epi16(ms1, ms2)))));

			ms1 = _mm256_load_epu8cvtepi16((__m128i*)(s1 + 16));
			ms2 = _mm256_load_epu8cvtepi16((__m128i*)(s2 + 16));
			_mm_store_si128((__m128i*)(d + 16), _mm256_cvtepi16_epu8(_mm256_add_epi16(ms2, _mm256_mulhrs_epi16(ma, _mm256_sub_epi16(ms1, ms2)))));

			s1 += 32;
			s2 += 32;
			d += 32;
		}

		s1 = src1.data;
		s2 = src2.data;
		d = dst.data;
		double a = alpha / 255.0;
		for (int i = simdsize; i < size; i++)
		{
			d[i] = saturate_cast<uchar>(a * (s1[i] - s2[i]) + s2[i]);
		}
	}

	void alphaBlendFixedPoint(InputArray src1, InputArray src2, const int alpha, OutputArray dest)
	{
		CV_Assert(src1.size() == src2.size());
		CV_Assert(src1.depth() == CV_8U);
		CV_Assert(src2.depth() == CV_8U);
		Mat s1, s2;

		const int channel = max(src1.channels(), src2.channels());
		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
		}
		else if (src2.channels() == 3)
		{
			cvtColor(src1, s1, COLOR_GRAY2BGR);
			s2 = src2.getMat();
		}
		else
		{
			cvtColor(src2, s2, COLOR_GRAY2BGR);
			s1 = src1.getMat();
		}

		dest.create(src1.size(), CV_MAKETYPE(CV_8U, channel));
		Mat dst = dest.getMat();
		alphaBlendFixedPoint_AVX(s1, s2, alpha, dst);
	}


	static void alphaBlendMask8U_AVX_8U(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
	{
		uchar* s1 = src1.data;
		uchar* s2 = src2.data;
		uchar* a = alpha.data;
		uchar* d = dst.data;
		const float inv = 1.f / 255.f;
		const __m256 minv = _mm256_set1_ps(inv);

		if (src1.channels() == 1)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 16);

			for (int i = 0; i < simdsize; i += 16)
			{
				__m256 ms1 = _mm256_load_epu8cvtps((__m128i*)s1);
				__m256 ms2 = _mm256_load_epu8cvtps((__m128i*)s2);
				__m256 ma = _mm256_load_epu8cvtps((__m128i*)a);
				__m256 mt1 = _mm256_fmadd_ps(_mm256_mul_ps(minv, ma), _mm256_sub_ps(ms1, ms2), ms2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 8));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 8));
				ma = _mm256_load_epu8cvtps((__m128i*)(a + 8));
				__m256 mt2 = _mm256_fmadd_ps(_mm256_mul_ps(minv, ma), _mm256_sub_ps(ms1, ms2), ms2);

				_mm_store_si128((__m128i*)d, _mm256_cvtpsx2_epu8(mt1, mt2));

				s1 += 16;
				s2 += 16;
				a += 16;
				d += 16;
			}

			s1 = src1.data;
			s2 = src2.data;
			d = dst.data;
			a = alpha.data;

			for (int i = simdsize; i < size; i++)
			{
				const float aa = inv * a[i];
				d[i] = saturate_cast<uchar>(aa * (s1[i] - s2[i]) + s2[i]);
			}
		}
		else if (src1.channels() == 3)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 48);

			__m256 ma0, ma1, ma2;
			for (int i = 0; i < simdsize; i += 48)
			{
				__m256 ma = _mm256_mul_ps(minv, _mm256_load_epu8cvtps((__m128i*)a));
				_mm256_cvtps_gray2bgr(ma, ma0, ma1, ma2);

				__m256 ms1 = _mm256_load_epu8cvtps((__m128i*)s1);
				__m256 ms2 = _mm256_load_epu8cvtps((__m128i*)s2);
				__m256 mt1 = _mm256_fmadd_ps(ma0, _mm256_sub_ps(ms1, ms2), ms2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 8));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 8));
				__m256 mt2 = _mm256_fmadd_ps(ma1, _mm256_sub_ps(ms1, ms2), ms2);
				_mm_store_si128((__m128i*)d, _mm256_cvtpsx2_epu8(mt1, mt2));

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 16));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 16));
				mt1 = _mm256_fmadd_ps(ma2, _mm256_sub_ps(ms1, ms2), ms2);

				ma = _mm256_mul_ps(minv, _mm256_load_epu8cvtps((__m128i*)(a + 8)));
				_mm256_cvtps_gray2bgr(ma, ma0, ma1, ma2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 24));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 24));
				mt2 = _mm256_fmadd_ps(ma0, _mm256_sub_ps(ms1, ms2), ms2);
				_mm_store_si128((__m128i*)(d + 16), _mm256_cvtpsx2_epu8(mt1, mt2));

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 32));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 32));
				mt1 = _mm256_fmadd_ps(ma1, _mm256_sub_ps(ms1, ms2), ms2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 40));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 40));
				mt2 = _mm256_fmadd_ps(ma2, _mm256_sub_ps(ms1, ms2), ms2);
				_mm_store_si128((__m128i*)(d + 32), _mm256_cvtpsx2_epu8(mt1, mt2));

				s1 += 48;
				s2 += 48;
				a += 16;
				d += 48;
			}

			s1 = src1.data;
			s2 = src2.data;
			d = dst.data;
			a = alpha.data;

			for (int i = simdsize; i < size; i++)
			{
				const float aa = inv * a[i / 3];
				d[i] = saturate_cast<uchar>(aa * s1[i] + (1.f - aa) * s2[i]);
			}
		}
	}

	static void alphaBlendMask32F_AVX_8U(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
	{
		uchar* s1 = src1.data;
		uchar* s2 = src2.data;
		float* a = alpha.ptr<float>();
		uchar* d = dst.data;

		if (src1.channels() == 1)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 16);

			for (int i = 0; i < simdsize; i += 16)
			{
				__m256 ms1 = _mm256_load_epu8cvtps((__m128i*)s1);
				__m256 ms2 = _mm256_load_epu8cvtps((__m128i*)s2);
				__m256 ma = _mm256_load_ps(a);
				__m256 mt1 = _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 8));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 8));
				ma = _mm256_load_ps(a + 8);
				__m256 mt2 = _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2);

				_mm_store_si128((__m128i*)d, _mm256_cvtpsx2_epu8(mt1, mt2));

				s1 += 16;
				s2 += 16;
				a += 16;
				d += 16;
			}

			s1 = src1.data;
			s2 = src2.data;
			d = dst.data;
			a = alpha.ptr<float>();

			for (int i = simdsize; i < size; i++)
			{
				d[i] = saturate_cast<uchar>(a[i] * (s1[i] - s2[i]) + s2[i]);
			}
		}
		else if (src1.channels() == 3)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 48);

			__m256 ma0, ma1, ma2;
			for (int i = 0; i < simdsize; i += 48)
			{
				__m256 ma = _mm256_load_ps(a);
				_mm256_cvtps_gray2bgr(ma, ma0, ma1, ma2);

				__m256 ms1 = _mm256_load_epu8cvtps((__m128i*)s1);
				__m256 ms2 = _mm256_load_epu8cvtps((__m128i*)s2);
				__m256 mt1 = _mm256_fmadd_ps(ma0, _mm256_sub_ps(ms1, ms2), ms2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 8));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 8));
				__m256 mt2 = _mm256_fmadd_ps(ma1, _mm256_sub_ps(ms1, ms2), ms2);
				_mm_store_si128((__m128i*)d, _mm256_cvtpsx2_epu8(mt1, mt2));

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 16));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 16));
				mt1 = _mm256_fmadd_ps(ma2, _mm256_sub_ps(ms1, ms2), ms2);

				ma = _mm256_load_ps(a + 8);
				_mm256_cvtps_gray2bgr(ma, ma0, ma1, ma2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 24));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 24));
				mt2 = _mm256_fmadd_ps(ma0, _mm256_sub_ps(ms1, ms2), ms2);
				_mm_store_si128((__m128i*)(d + 16), _mm256_cvtpsx2_epu8(mt1, mt2));

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 32));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 32));
				mt1 = _mm256_fmadd_ps(ma1, _mm256_sub_ps(ms1, ms2), ms2);

				ms1 = _mm256_load_epu8cvtps((__m128i*)(s1 + 40));
				ms2 = _mm256_load_epu8cvtps((__m128i*)(s2 + 40));
				mt2 = _mm256_fmadd_ps(ma2, _mm256_sub_ps(ms1, ms2), ms2);
				_mm_store_si128((__m128i*)(d + 32), _mm256_cvtpsx2_epu8(mt1, mt2));

				s1 += 48;
				s2 += 48;
				a += 16;
				d += 48;
			}

			s1 = src1.data;
			s2 = src2.data;
			d = dst.data;
			a = alpha.ptr<float>();

			for (int i = simdsize; i < size; i++)
			{
				d[i] = saturate_cast<uchar>(a[i / 3] * s1[i] + (1.f - a[i / 3]) * s2[i]);
			}
		}
	}

	static void alphaBlendMask8U_AVX_32F(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
	{
		float* s1 = src1.ptr<float>();
		float* s2 = src2.ptr<float>();
		uchar* a = alpha.ptr<uchar>();
		float* d = dst.ptr<float>();
		float inv = 1.f / 255.f;
		const __m256 minv = _mm256_set1_ps(inv);

		if (src1.channels() == 1)
		{
			const int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 8);

			for (int i = 0; i < simdsize; i += 8)
			{
				__m256 ms1 = _mm256_load_ps(s1);
				__m256 ms2 = _mm256_load_ps(s2);
				__m256 ma = _mm256_mul_ps(minv, _mm256_load_epu8cvtps((__m128i*)a));
				_mm256_store_ps(d, _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2));

				s1 += 8;
				s2 += 8;
				a += 8;
				d += 8;
			}

			s1 = src1.ptr<float>();
			s2 = src2.ptr<float>();
			d = dst.ptr<float>();
			a = alpha.ptr<uchar>();

			for (int i = simdsize; i < size; i++)
			{
				d[i] = saturate_cast<float>(a[i] * (s1[i] - s2[i]) + s2[i]);
			}
		}
		else if (src1.channels() == 3)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 24);

			__m256 ma0, ma1, ma2;
			for (int i = 0; i < simdsize; i += 24)
			{
				__m256 ma = _mm256_mul_ps(minv, _mm256_load_epu8cvtps((__m128i*)a));
				_mm256_cvtps_gray2bgr(ma, ma0, ma1, ma2);

				__m256 ms1 = _mm256_load_ps(s1);
				__m256 ms2 = _mm256_load_ps(s2);
				_mm256_store_ps(d, _mm256_fmadd_ps(ma0, _mm256_sub_ps(ms1, ms2), ms2));

				ms1 = _mm256_load_ps(s1 + 8);
				ms2 = _mm256_load_ps(s2 + 8);
				_mm256_store_ps(d + 8, _mm256_fmadd_ps(ma1, _mm256_sub_ps(ms1, ms2), ms2));

				ms1 = _mm256_load_ps(s1 + 16);
				ms2 = _mm256_load_ps(s2 + 16);
				_mm256_store_ps(d + 16, _mm256_fmadd_ps(ma2, _mm256_sub_ps(ms1, ms2), ms2));

				s1 += 24;
				s2 += 24;
				a += 8;
				d += 24;
			}

			s1 = src1.ptr<float>();
			s2 = src2.ptr<float>();
			d = dst.ptr<float>();
			uchar* a = alpha.ptr<uchar>();
			for (int i = simdsize; i < size; i++)
			{
				d[i] = saturate_cast<float>(inv * a[i / 3] * s1[i] + (1.f - inv * a[i / 3]) * s2[i]);
			}
		}
	}

	static void alphaBlendMask32F_AVX_32F(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
	{
		float* s1 = src1.ptr<float>();
		float* s2 = src2.ptr<float>();
		float* a = alpha.ptr<float>();
		float* d = dst.ptr<float>();

		if (src1.channels() == 1)
		{
			const int size = src1.size().area() * src1.channels();

#if 1
			int simdsize = get_simd_floor(size, 8);
			for (int i = 0; i < simdsize; i += 8)
			{
				__m256 ms1 = _mm256_load_ps(s1);
				__m256 ms2 = _mm256_load_ps(s2);
				__m256 ma = _mm256_load_ps(a);
				_mm256_store_ps(d, _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2));

				s1 += 8;
				s2 += 8;
				a += 8;
				d += 8;
			}
#else
			int simdsize = get_simd_floor(size, 16);

			for (int i = 0; i < simdsize; i += 16)
			{
				__m256 ms1 = _mm256_load_ps(s1);
				__m256 ms2 = _mm256_load_ps(s2);
				__m256 ma = _mm256_load_ps(a);
				_mm256_store_ps(d, _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2));

				ms1 = _mm256_load_ps(s1 + 8);
				ms2 = _mm256_load_ps(s2 + 8);
				ma = _mm256_load_ps(a + 8);
				_mm256_store_ps(d, _mm256_fmadd_ps(ma, _mm256_sub_ps(ms1, ms2), ms2));

				s1 += 16;
				s2 += 16;
				a += 16;
				d += 16;
			}
#endif

			s1 = src1.ptr<float>();
			s2 = src2.ptr<float>();
			d = dst.ptr<float>();
			a = alpha.ptr<float>();

			for (int i = simdsize; i < size; i++)
			{
				d[i] = saturate_cast<float>(a[i] * (s1[i] - s2[i]) + s2[i]);
			}
		}
		else if (src1.channels() == 3)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 24);

			__m256 ma0, ma1, ma2;
			for (int i = 0; i < simdsize; i += 24)
			{
				__m256 ma = _mm256_load_ps(a);
				_mm256_cvtps_gray2bgr(ma, ma0, ma1, ma2);

				__m256 ms1 = _mm256_load_ps(s1);
				__m256 ms2 = _mm256_load_ps(s2);
				_mm256_store_ps(d, _mm256_fmadd_ps(ma0, _mm256_sub_ps(ms1, ms2), ms2));

				ms1 = _mm256_load_ps(s1 + 8);
				ms2 = _mm256_load_ps(s2 + 8);
				_mm256_store_ps(d + 8, _mm256_fmadd_ps(ma1, _mm256_sub_ps(ms1, ms2), ms2));

				ms1 = _mm256_load_ps(s1 + 16);
				ms2 = _mm256_load_ps(s2 + 16);
				_mm256_store_ps(d + 16, _mm256_fmadd_ps(ma2, _mm256_sub_ps(ms1, ms2), ms2));

				s1 += 24;
				s2 += 24;
				a += 8;
				d += 24;
			}

			s1 = src1.ptr<float>();
			s2 = src2.ptr<float>();
			d = dst.ptr<float>();
			a = alpha.ptr<float>();

			for (int i = simdsize; i < size; i++)
			{
				d[i] = saturate_cast<float>(a[i / 3] * s1[i] + (1.f - a[i / 3]) * s2[i]);
			}
		}
	}

	void alphaBlend(cv::InputArray src1, cv::InputArray src2, cv::InputArray alpha, cv::OutputArray dest)
	{
		int T;
		Mat s1, s2;
		if (src1.channels() <= src2.channels())T = src2.type();
		else T = src1.type();

		dest.create(src1.size(), T);

		if (dest.type() != T)dest.create(src1.size(), T);

		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
		}
		else if (src2.channels() == 3)
		{
			cvtColor(src1, s1, COLOR_GRAY2BGR);
			s2 = src2.getMat();
		}
		else
		{
			cvtColor(src2, s2, COLOR_GRAY2BGR);
			s1 = src1.getMat();
		}

		Mat a = alpha.getMat();
		Mat dst = dest.getMat();
		if (src1.depth() == CV_8U && alpha.depth() == CV_8U)
		{
			alphaBlendMask8U_AVX_8U(s1, s2, a, dst);
		}
		else if (src1.depth() == CV_8U && alpha.depth() == CV_32F)
		{
			alphaBlendMask32F_AVX_8U(s1, s2, a, dst);
		}
		else if (src1.depth() == CV_32F && alpha.depth() == CV_8U)
		{
			alphaBlendMask8U_AVX_32F(s1, s2, a, dst);
		}
		else if (src1.depth() == CV_32F && alpha.depth() == CV_32F)
		{
			alphaBlendMask32F_AVX_32F(s1, s2, a, dst);
		}
		else
		{
			cout << "src image (8U/32F), alpha mask(8U/32F) supported." << endl;
		}
	}


	static void alphaBlendFixedPointMask8U_AVX(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
	{
		uchar* s1 = src1.data;
		uchar* s2 = src2.data;
		uchar* a = alpha.data;
		uchar* d = dst.data;

		const float inv = 1.f / 255.f;

		if (src1.channels() == 1)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 16);

			for (int i = 0; i < simdsize; i += 16)
			{
				__m256i ms1 = _mm256_load_epu8cvtepi16((__m128i*)s1);
				__m256i ms2 = _mm256_load_epu8cvtepi16((__m128i*)s2);
				__m256i ma = _mm256_slli_epi16(_mm256_load_epu8cvtepi16((__m128i*)a), 7);
				_mm_store_si128((__m128i*)d, _mm256_cvtepi16_epu8(_mm256_add_epi16(ms2, _mm256_mulhrs_epi16(ma, _mm256_sub_epi16(ms1, ms2)))));

				s1 += 16;
				s2 += 16;
				a += 16;
				d += 16;
			}

			s1 = src1.data;
			s2 = src2.data;
			d = dst.data;
			a = alpha.data;

			for (int i = simdsize; i < size; i++)
			{
				const float aa = inv * a[i];
				d[i] = saturate_cast<uchar>(aa * (s1[i] - s2[i]) + s2[i]);
			}
		}
		else if (src1.channels() == 3)
		{
			int size = src1.size().area() * src1.channels();
			int simdsize = get_simd_floor(size, 48);

			__m256i ma0, ma1, ma2;
			for (int i = 0; i < simdsize; i += 48)
			{
				__m256i ma = _mm256_slli_epi16(_mm256_load_epu8cvtepi16((__m128i*)a), 7);
				_mm256_cvtepi16_gray2bgr(ma, ma0, ma1, ma2);

				__m256i ms1 = _mm256_load_epu8cvtepi16((__m128i*)(s1 + 0));
				__m256i ms2 = _mm256_load_epu8cvtepi16((__m128i*)(s2 + 0));
				_mm_store_si128((__m128i*)(d + 0), _mm256_cvtepi16_epu8(_mm256_add_epi16(ms2, _mm256_mulhrs_epi16(ma0, _mm256_sub_epi16(ms1, ms2)))));
				ms1 = _mm256_load_epu8cvtepi16((__m128i*)(s1 + 16));
				ms2 = _mm256_load_epu8cvtepi16((__m128i*)(s2 + 16));
				_mm_store_si128((__m128i*)(d + 16), _mm256_cvtepi16_epu8(_mm256_add_epi16(ms2, _mm256_mulhrs_epi16(ma1, _mm256_sub_epi16(ms1, ms2)))));

				ms1 = _mm256_load_epu8cvtepi16((__m128i*)(s1 + 32));
				ms2 = _mm256_load_epu8cvtepi16((__m128i*)(s2 + 32));
				_mm_store_si128((__m128i*)(d + 32), _mm256_cvtepi16_epu8(_mm256_add_epi16(ms2, _mm256_mulhrs_epi16(ma2, _mm256_sub_epi16(ms1, ms2)))));

				s1 += 48;
				s2 += 48;
				a += 16;
				d += 48;
			}

			s1 = src1.data;
			s2 = src2.data;
			d = dst.data;
			a = alpha.data;

			for (int i = simdsize; i < size; i++)
			{
				const float aa = inv * a[i / 3];
				d[i] = saturate_cast<uchar>(aa * s1[i] + (1.f - aa) * s2[i]);
			}
		}
	}

	void alphaBlendFixedPoint(InputArray src1, InputArray src2, InputArray alpha, OutputArray dest)
	{
		CV_Assert(src1.depth() == CV_8U);
		CV_Assert(src2.depth() == CV_8U);
		CV_Assert(alpha.depth() == CV_8U);
		int T;
		Mat s1, s2;
		if (src1.channels() <= src2.channels())T = src2.type();
		else T = src1.type();

		dest.create(src1.size(), T);

		if (dest.type() != T)dest.create(src1.size(), T);

		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
		}
		else if (src2.channels() == 3)
		{
			cvtColor(src1, s1, COLOR_GRAY2BGR);
			s2 = src2.getMat();
		}
		else
		{
			cvtColor(src2, s2, COLOR_GRAY2BGR);
			s1 = src1.getMat();
		}

		Mat a = alpha.getMat();
		Mat dst = dest.getMat();
		alphaBlendFixedPointMask8U_AVX(s1, s2, a, dst);
	}

	static void imageCast(InputArray src, Mat& dest, string mes = "")
	{
		if (src.depth() == CV_8U)
		{
			if (src.channels() == 1)cvtColor(src, dest, COLOR_GRAY2BGR);
			else dest = src.getMat();
		}
		else
		{
			Mat s = src.getMat();
			s.convertTo(dest, CV_32F);
			if (src.channels() == 1)cvtColor(dest, dest, COLOR_GRAY2BGR);

			double minv, maxv;
			minMaxLoc(src, &minv, &maxv);
			bool isMulFF = (maxv <= 1.0 && minv >= 0.0) ? true : false;
			bool isNormalized = (maxv > 255 || minv < 0.0) ? true : false;
			if (isMulFF)
			{
				cout << mes + ": scale 0.0-1.0 -> 0-255" << endl;
				dest.convertTo(dest, CV_8U, 255);
			}
			else if (isNormalized)
			{
				cout << mes + ": normalize min-max -> 0-255" << endl;

				normalize(src, dest, 255, 0, NORM_MINMAX, CV_8U);
			}
			else
			{
				dest.convertTo(dest, CV_8U);
			}
		}
	}

	cv::Mat guiAlphaBlend(InputArray src1, InputArray src2, bool isShowImageStats, std::string wname)
	{
		if (isShowImageStats)
		{
			showMatInfo(src1, "src1");
			cout << endl;
			showMatInfo(src2, "src2");
		}

		Mat s1, s2;
		imageCast(src1, s1, "src1");
		imageCast(src2, s2, "src2");

		namedWindow(wname);
		int a = 0;
		cv::createTrackbar("a", wname, &a, 100);
		int key = 0;
		Mat show;

		cv::VideoWriter video;
		bool isVideo = false;
		bool printAlpha = false;

		while (key != 'q')
		{
			alphaBlend(s1, s2, 1.0 - a / 100.0, show);

			if (printAlpha) addText(show, cv::format("a = %3d", a), Point(20, 30), "Consolas", 16);
			cv::imshow("alphaBlend", show);

			if (isVideo)video << show;

			key = waitKey(1);
			if (key == 'd')
			{
				guiDiff(src1, src2);
			}
			if (key == 'f')
			{
				a = (a > 0) ? 0 : 100;
				setTrackbarPos("a", "alphaBlend", a);
			}
			if (key == 'p')
			{
				cout << "PSNR: " << getPSNR(src1, src2) << "dB" << endl;
				cout << "MSE: " << getMSE(src1, src2) << endl;
			}
			if (key == 'i')
			{
				showMatInfo(src1, "src1");
				cout << endl;
				showMatInfo(src2, "src2");
			}
			if (key == 'v')
			{
				if (isVideo == false)
				{
					cout << "start capturing" << endl;
					imshowCountDown(wname, show);

					//int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
					int codec = VideoWriter::fourcc('X', '2', '6', '4');
					if (!video.isOpened()) video.open(wname + ".mp4", codec, 29.97, show.size(), show.channels() == 3);
					isVideo = true;
					printAlpha = true;
				}
				else
				{
					cout << "stop capturing." << endl;
					isVideo = false;
					printAlpha = false;
					video.release();
				}
			}
			if (key == '?')
			{
				cout << "d: call guiDiff." << endl;
				cout << "f: flip blend parameter alpha." << endl;
				cout << "i: call showMatInfo." << endl;
				cout << "p: compute PSNR and MSE." << endl;
				cout << "v: start/stop video capture (toggle)." << endl;
			}
		}
		destroyWindow(wname);
		return show;
	}

	template <class srcType>
	static void setTrapezoidMaskH(Mat& src, double ratio, double slant_ratio, Point& start_pt, Point& end_pt)
	{
		const int offset = (int)((0.5 - ratio) * src.cols);
		const int boundary = int(src.cols * slant_ratio);
		float aspect = (float)(src.cols - 2 * boundary) / src.rows;
		src.setTo(0);
		for (int j = 0; j < src.rows; j++)
		{
			srcType* s = src.ptr<srcType>(j);
			int v = (int)(j * aspect) + boundary + offset;
			memset(s, 1, (sizeof(srcType) * min(src.cols, max(src.cols - v, 0))));
		}

		start_pt.x = src.cols - 1 - offset - boundary;
		start_pt.y = 0;
		end_pt.x = boundary - offset;
		end_pt.y = src.rows - 1;
	}

	template <class srcType>
	static void setTrapezoidMaskV(Mat& src, double ratio, double slant_ratio, Point& start_pt, Point& end_pt)
	{
		const int offset = int((0.5 - ratio) * src.rows);

		src.setTo(1);

		if (slant_ratio < 0.5)
		{
			const int boundary = int(src.rows * slant_ratio);
			float aspect = (float)(src.cols) / (src.rows - 2 * boundary);
			int sty = offset + boundary;
			int edy = src.rows + offset - boundary;

			for (int j = sty; j < edy; j++)
			{
				if (j >= 0 && j < src.rows)
				{
					srcType* s = src.ptr<srcType>(j);
					int v = min((int)((j - sty) * aspect), src.cols);
					memset(s, 0, sizeof(srcType) * v);
				}
			}
			for (int j = edy; j < src.rows; j++)
			{
				srcType* s = src.ptr<srcType>(j);
				memset(s, 0, sizeof(srcType) * src.cols);
			}
			start_pt.x = 0;
			start_pt.y = sty;
			end_pt.x = src.cols - 1;
			end_pt.y = edy;
		}
		else
		{
			const int boundary = int(src.rows * (1.0 - slant_ratio));
			float aspect = (float)(src.cols) / (src.rows - 2 * boundary);
			int sty = offset + boundary;
			int edy = src.rows + offset - boundary;

			for (int j = sty; j <= edy; j++)
			{
				if (j >= 1 && j < src.rows)
				{
					int v = max(0, min(src.cols, (int)((j - sty) * aspect)));
					srcType* s = src.ptr<srcType>(j) - v;
					memset(s, 0, sizeof(srcType) * v);
				}
			}
			for (int j = edy; j < src.rows; j++)
			{
				srcType* s = src.ptr<srcType>(j);
				memset(s, 0, sizeof(srcType) * src.cols);
			}
			start_pt.x = 0;
			start_pt.y = edy;
			end_pt.x = src.cols - 1;
			end_pt.y = sty;
		}
	}

	void dissolveSlideBlend(InputArray src1, InputArray src2, OutputArray dest, const double ratio, const double slant_ratio, const int direction, cv::Scalar line_color, const int line_thickness)
	{
		if (direction == 0)
		{
			if (ratio == 0.0)
			{
				src2.copyTo(dest);
				return;
			}
			else if (ratio == 1.0)
			{
				src1.copyTo(dest);
				return;
			}
		}
		else if (direction == 1)
		{
			if (ratio == 0.0)
			{
				src1.copyTo(dest);
				return;
			}
			else if (ratio == 1.0)
			{
				src2.copyTo(dest);
				return;
			}
		}

		CV_Assert(src1.size() == src2.size());
		Mat s1;
		Mat s2;
		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
			dest.create(src1.size(), src1.type());
		}
		else
		{
			if (src1.channels() == 1)cvtColor(src1, s1, COLOR_GRAY2BGR);
			else s1 = src1.getMat();
			if (src2.channels() == 1)cvtColor(src2, s2, COLOR_GRAY2BGR);
			else s2 = src2.getMat();
			dest.create(src1.size(), CV_MAKETYPE(src1.depth(), 3));
		}

		Mat dst;
		if (direction == 0)//vertical split
		{
			Mat mask = Mat::zeros(s1.size(), CV_8U);
			Point st;
			Point ed;
			setTrapezoidMaskH<uchar>(mask, ratio, slant_ratio, st, ed);
			s2.copyTo(dst);
			s1.copyTo(dst, mask);
			if (line_thickness != 0) line(dst, st, ed, line_color, line_thickness);
		}
		else if (direction == 1)//horizontal split
		{
			Mat mask = Mat::zeros(s1.size(), CV_8U);
			Point st;
			Point ed;
			setTrapezoidMaskV<uchar>(mask, ratio, slant_ratio, st, ed);
			s2.copyTo(dst);
			s1.copyTo(dst, mask);
			//cvtColor(mask, dst, COLOR_GRAY2BGR);
			if (line_thickness != 0) line(dst, st, ed, line_color, line_thickness);
		}
		dst.copyTo(dest);
	}

	cv::Mat guiDissolveSlideBlend(InputArray src1, InputArray src2, string wname)
	{
		namedWindow(wname);
		static bool isBorderLine = true;
		static int a = 50; createTrackbar("ratio", wname, &a, 100);
		static int sa = 30; createTrackbar("slant_ratio", wname, &sa, 100);
		static int direction = 1; createTrackbar("direction", wname, &direction, 1);
		static int line_width = 2; createTrackbar("line_width", wname, &line_width, 20);

		int key = 0;
		Mat show;
		while (key != 'q')
		{
			dissolveSlideBlend(src1, src2, show, a / 100.0, sa / 100.0, direction, Scalar::all(255), line_width);
			imshow(wname, show);
			key = waitKey(1);

			if (key == 'k' || key == 'l')
			{
				a++;
				a = min(100, a);
				setTrackbarPos("ratio", wname, a);
			}
			if (key == 'i' || key == 'j')
			{
				a--;
				a = max(0, a);
				setTrackbarPos("ratio", wname, a);
			}
			if (key == 'f')
			{
				a = (a == 0) ? 100 : 0;
				setTrackbarPos("ratio", wname, a);
			}
			if (key == 'd')
			{
				direction = (direction == 0) ? 1 : 0;
				setTrackbarPos("direction", wname, direction);
			}
			if (key == 'b')isBorderLine = (isBorderLine) ? false : true;
			if (key == '?')
			{
				cout << "Help: " << endl;
				cout << "i,j,k,l: move ratio slider" << endl;
				cout << "f: flip ratio" << endl;
				cout << "d: flip direction" << endl;
				cout << "b: flip flag of isBorderline" << endl;
				cout << "q: quit" << endl;
			}
		}

		destroyWindow(wname);
		return show;
	}
}
