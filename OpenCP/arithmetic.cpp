#include "arithmetic.hpp"
#include "fmath/fmath.hpp"
#include "inlineSIMDFunctions.hpp"
#include "checkSameImage.hpp"

using namespace std;
using namespace cv;
using namespace fmath;

namespace cp
{
	// if you do not use fmath.hpp
	/*
	// Fast SSE pow for range [0, 1]
	// Adapted from C. Schlick with one more iteration each for exp(x) and ln(x)
	// 8 muls, 5 adds, 1 rcp
	inline __m128 _mm_pow01_ps(__m128 x, __m128 y)
	{
	static const __m128 fourOne = _mm_set1_ps(1.0f);
	static const __m128 fourHalf = _mm_set1_ps(0.5f);

	__m128 a = _mm_sub_ps(fourOne, y);
	__m128 b = _mm_sub_ps(x, fourOne);
	__m128 aSq = _mm_mul_ps(a, a);
	__m128 bSq = _mm_mul_ps(b, b);
	__m128 c = _mm_mul_ps(fourHalf, bSq);
	__m128 d = _mm_sub_ps(b, c);
	__m128 dSq = _mm_mul_ps(d, d);
	__m128 e = _mm_mul_ps(aSq, dSq);
	__m128 f = _mm_mul_ps(a, d);
	__m128 g = _mm_mul_ps(fourHalf, e);
	__m128 h = _mm_add_ps(fourOne, f);
	__m128 i = _mm_add_ps(h, g);
	__m128 iRcp = _mm_rcp_ps(i);
	//__m128 iRcp = _mm_rcp_22bit_ps(i);
	__m128 result = _mm_mul_ps(x, iRcp);

	return result;
	}
	#define _mm_pow_ps _mm_pow01_ps
	*/

	inline __m128 _mm_pow_ps(__m128 a, __m128 b)
	{
		return exp_ps(_mm_mul_ps(b, log_ps(a)));
	}


	//sign(src)*pow(abs(src),v)
	void powsign(cv::InputArray src, const float v, cv::OutputArray dest)
	{
		dest.create(src.size(), src.type());
		const float* s = src.getMat().ptr<float>();
		float* d = dest.getMat().ptr<float>();
		__m256 mv = _mm256_set1_ps(v);
		const int simd_width = get_simd_floor((int)src.total(), 8);
		for (int i = 0; i < simd_width; i += 8)
		{
			__m256 ms = _mm256_load_ps(s + i);
			_mm256_store_ps(d + i, _mm256_mul_ps(_mm256_sign_ps(ms), _mm256_pow_ps(_mm256_abs_ps(ms), mv)));
		}

		for (int i = simd_width; i < src.total(); i++)
		{
			if (s[i] >= 0)d[i] = pow(s[i], v);
			else d[i] = -pow(-s[i], v);
		}
	}

	//sign(src)*pow(abs(src),v)
	void powZeroClip(cv::InputArray src, const float v, cv::OutputArray dest)
	{
		dest.create(src.size(), src.type());
		const float* s = src.getMat().ptr<float>();
		float* d = dest.getMat().ptr<float>();
		__m256 mv = _mm256_set1_ps(v);
		const int simd_width = get_simd_floor((int)src.total(), 8);
		for (int i = 0; i < simd_width; i += 8)
		{
			__m256 ms = _mm256_max_ps(_mm256_load_ps(s + i), _mm256_setzero_ps());
			_mm256_store_ps(d + i, _mm256_pow_ps(ms, mv));
		}

		for (int i = simd_width; i < src.total(); i++)
		{
			d[i] = -pow(max(s[i], 0.f), v);
		}
	}

	void pow_fmath(const float a, const Mat& src, Mat& dest)
	{
		if (dest.empty())dest.create(src.size(), CV_32F);

		int size = src.size().area();
		int i = 0;

		const float* s = src.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		const __m128 ma = _mm_set1_ps(a);
		for (i = 0; i <= size - 4; i += 4)
		{
			_mm_store_ps(d + i, cp::_mm_pow_ps(ma, _mm_load_ps(s + i)));
		}
		for (; i < size; i++)
		{
			d[i] = cv::pow(a, s[i]);
		}
	}

	void pow_fmath(const Mat& src, const float a, Mat& dest)
	{
		if (dest.empty())dest.create(src.size(), CV_32F);

		int size = src.size().area();
		int i = 0;

		const float* s = src.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		const __m128 ma = _mm_set1_ps(a);
		for (i = 0; i <= size - 4; i += 4)
		{
			_mm_store_ps(d + i, cp::_mm_pow_ps(_mm_load_ps(s + i), ma));
		}
		for (; i < size; i++)
		{
			d[i] = cv::pow(s[i], a);
		}
	}

	void pow_fmath(const Mat& src1, const Mat& src2, Mat& dest)
	{
		if (dest.empty())dest.create(src1.size(), CV_32F);


		int size = src1.size().area();
		int i = 0;

		const float* s1 = src1.ptr<float>(0);
		const float* s2 = src2.ptr<float>(0);
		float* d = dest.ptr<float>(0);

		for (i = 0; i <= size - 4; i += 4)
		{
			_mm_store_ps(d + i, cp::_mm_pow_ps(_mm_load_ps(s1 + i), _mm_load_ps(s2 + i)));
		}
		for (; i < size; i++)
		{
			d[i] = cv::pow(s1[i], s2[i]);
		}
	}

	void compareRange(InputArray src, OutputArray destMask, const double validMin, const double validMax)
	{
		Mat gray;
		if (src.channels() == 1) gray = src.getMat();
		else cvtColor(src, gray, COLOR_BGR2GRAY);

		Mat mask1;
		Mat mask2;
		compare(gray, validMin, mask1, cv::CMP_GE);
		compare(gray, validMax, mask2, cv::CMP_LE);
		bitwise_and(mask1, mask2, destMask);
	}

	void setTypeMaxValue(InputOutputArray src)
	{
		Mat s = src.getMat();
		if (s.depth() == CV_8U)s.setTo(UCHAR_MAX);
		else if (s.depth() == CV_16U)s.setTo(USHRT_MAX);
		else if (s.depth() == CV_16S)s.setTo(SHRT_MAX);
		else if (s.depth() == CV_32S)s.setTo(INT_MAX);
		else if (s.depth() == CV_32F)s.setTo(FLT_MAX);
		else if (s.depth() == CV_64F)s.setTo(DBL_MAX);
	}

	void setTypeMinValue(InputOutputArray src)
	{
		Mat s = src.getMat();
		if (s.depth() == CV_8U)s.setTo(0);
		else if (s.depth() == CV_16U)s.setTo(0);
		else if (s.depth() == CV_16S)s.setTo(SHRT_MIN);
		else if (s.depth() == CV_32S)s.setTo(INT_MIN);
		else if (s.depth() == CV_32F)s.setTo(FLT_MIN);
		else if (s.depth() == CV_64F)s.setTo(DBL_MIN);
	}


	void fnmsub(const Mat& a, const Mat& x, const Mat& b, Mat& dest)
	{
		CV_Assert(!a.empty());
		CV_Assert(!x.empty());
		CV_Assert(!b.empty());
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area() * a.channels();
		if (a.depth() == CV_32F)
		{
			const float* aptr = a.ptr<float>();
			const float* xptr = x.ptr<float>();
			const float* bptr = b.ptr<float>();
			float* dptr = dest.ptr<float>();
			const int simdsize = size / 8;
			const int rem = simdsize * 8;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_ps(dptr, _mm256_fnmsub_ps(_mm256_load_ps(aptr), _mm256_load_ps(xptr), _mm256_load_ps(bptr)));
				aptr += 8;
				xptr += 8;
				bptr += 8;
				dptr += 8;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<float>(i) = -a.at<float>(i) * x.at<float>(i) - b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			const double* aptr = a.ptr<double>();
			const double* xptr = x.ptr<double>();
			const double* bptr = b.ptr<double>();
			double* dptr = dest.ptr<double>();
			const int simdsize = size / 4;
			const int rem = simdsize * 4;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_pd(dptr, _mm256_fnmsub_pd(_mm256_load_pd(aptr), _mm256_load_pd(xptr), _mm256_load_pd(bptr)));
				aptr += 4;
				xptr += 4;
				bptr += 4;
				dptr += 4;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<double>(i) = -a.at<double>(i) * x.at<double>(i) - b.at<double>(i);
			}
		}
	}

	void fnmadd(const Mat& a, const Mat& x, const Mat& b, Mat& dest)
	{
		CV_Assert(!a.empty());
		CV_Assert(!x.empty());
		CV_Assert(!b.empty());
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area() * a.channels();
		if (a.depth() == CV_32F)
		{
			const float* aptr = a.ptr<float>();
			const float* xptr = x.ptr<float>();
			const float* bptr = b.ptr<float>();
			float* dptr = dest.ptr<float>();
			const int simdsize = size / 8;
			const int rem = simdsize * 8;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_ps(dptr, _mm256_fnmadd_ps(_mm256_load_ps(aptr), _mm256_load_ps(xptr), _mm256_load_ps(bptr)));
				aptr += 8;
				xptr += 8;
				bptr += 8;
				dptr += 8;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<float>(i) = -a.at<float>(i) * x.at<float>(i) + b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			const double* aptr = a.ptr<double>();
			const double* xptr = x.ptr<double>();
			const double* bptr = b.ptr<double>();
			double* dptr = dest.ptr<double>();
			const int simdsize = size / 4;
			const int rem = simdsize * 4;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_pd(dptr, _mm256_fnmadd_pd(_mm256_load_pd(aptr), _mm256_load_pd(xptr), _mm256_load_pd(bptr)));
				aptr += 4;
				xptr += 4;
				bptr += 4;
				dptr += 4;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<double>(i) = -a.at<double>(i) * x.at<double>(i) + b.at<double>(i);
			}
		}
	}

	void fmsub(const Mat& a, const Mat& x, const Mat& b, Mat& dest)
	{
		CV_Assert(!a.empty());
		CV_Assert(!x.empty());
		CV_Assert(!b.empty());
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area() * a.channels();
		if (a.depth() == CV_32F)
		{
			const float* aptr = a.ptr<float>();
			const float* xptr = x.ptr<float>();
			const float* bptr = b.ptr<float>();
			float* dptr = dest.ptr<float>();
			const int simdsize = size / 8;
			const int rem = simdsize * 8;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_ps(dptr, _mm256_fmsub_ps(_mm256_load_ps(aptr), _mm256_load_ps(xptr), _mm256_load_ps(bptr)));
				aptr += 8;
				xptr += 8;
				bptr += 8;
				dptr += 8;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<float>(i) = a.at<float>(i) * x.at<float>(i) - b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			const double* aptr = a.ptr<double>();
			const double* xptr = x.ptr<double>();
			const double* bptr = b.ptr<double>();
			double* dptr = dest.ptr<double>();
			const int simdsize = size / 4;
			const int rem = simdsize * 4;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_pd(dptr, _mm256_fmsub_pd(_mm256_load_pd(aptr), _mm256_load_pd(xptr), _mm256_load_pd(bptr)));
				aptr += 4;
				xptr += 4;
				bptr += 4;
				dptr += 4;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<double>(i) = a.at<double>(i) * x.at<double>(i) - b.at<double>(i);
			}
		}
	}

	void fmadd(const Mat& a, const Mat& x, const Mat& b, Mat& dest)
	{
		CV_Assert(!a.empty());
		CV_Assert(!x.empty());
		CV_Assert(!b.empty());
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area() * a.channels();
		if (a.depth() == CV_32F)
		{
			const float* aptr = a.ptr<float>();
			const float* xptr = x.ptr<float>();
			const float* bptr = b.ptr<float>();
			float* dptr = dest.ptr<float>();
			const int simdsize = size / 8;
			const int rem = simdsize * 8;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_ps(dptr, _mm256_fmadd_ps(_mm256_load_ps(aptr), _mm256_load_ps(xptr), _mm256_load_ps(bptr)));
				aptr += 8;
				xptr += 8;
				bptr += 8;
				dptr += 8;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<float>(i) = a.at<float>(i) * x.at<float>(i) + b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			const double* aptr = a.ptr<double>();
			const double* xptr = x.ptr<double>();
			const double* bptr = b.ptr<double>();
			double* dptr = dest.ptr<double>();
			const int simdsize = size / 4;
			const int rem = simdsize * 4;
			for (int i = 0; i < simdsize; i++)
			{
				_mm256_store_pd(dptr, _mm256_fmadd_pd(_mm256_load_pd(aptr), _mm256_load_pd(xptr), _mm256_load_pd(bptr)));
				aptr += 4;
				xptr += 4;
				bptr += 4;
				dptr += 4;
			}
			for (int i = rem; i < size; i++)
			{
				dest.at<double>(i) = a.at<double>(i) * x.at<double>(i) + b.at<double>(i);
			}
		}
	}


	void bitshiftRight_nonsimd(Mat& src, Mat& dest, const int shift)
	{
		CV_Assert(src.depth() == CV_8U);
		if (src.data != dest.data) dest.create(src.size(), src.type());

		const int size = src.size().area() * src.channels();
		uchar* s = src.ptr<uchar>();
		uchar* d = dest.ptr<uchar>();

		for (int i = 0; i < size; i++)
		{
			d[i] = s[i] >> shift;
		}
	}

	void bitshiftRight(cv::InputArray src_, cv::OutputArray dest_, const int shift)
	{
		CV_Assert(src_.depth() == CV_8U);
		if (shift == 0)
		{
			if (&src_ == &dest_)return;
			else
			{
				src_.copyTo(dest_);
				return;
			}
		}

		Mat src = src_.getMat();
		if (dest_.empty()) dest_.create(src.size(), src.type());
		Mat dest = dest_.getMat();
		//if (src.data != dest.data) dest_.create(src.size(), src.type());

		const int size = src.size().area() * src.channels();
		uchar* s = src.ptr<uchar>();
		uchar* d = dest.ptr<uchar>();
		const int simdsize = size / 16;
		const int remstart = simdsize * 16;
		for (int i = 0; i < simdsize; i++)
		{
			__m128i v0 = _mm_srli_epi16(_mm_cvtepu8_epi16(_mm_load_si128((__m128i*)s)), shift);
			__m128i v1 = _mm_srli_epi16(_mm_cvtepu8_epi16(_mm_load_si128((__m128i*)(s + 8))), shift);
			_mm_store_si128((__m128i*)d, _mm_packus_epi16(v0, v1));
			s += 16;
			d += 16;
		}

		s = src.ptr<uchar>();
		d = dest.ptr<uchar>();
		for (int i = remstart; i < size; i++)
		{
			d[i] = s[i] >> shift;
		}
	}

	void bitshiftRight_nonsimd(Mat& src, Mat& dest, Mat& lostbit, const int shift)
	{
		CV_Assert(src.depth() == CV_8U);

		if (src.data != dest.data) dest.create(src.size(), src.type());
		lostbit.create(src.size(), src.type());

		const int size = src.size().area() * src.channels();
		uchar* s = src.ptr<uchar>();
		uchar* d = dest.ptr<uchar>();
		uchar* o = lostbit.ptr<uchar>();

		for (int i = 0; i < size; i++)
		{
			d[i] = s[i] >> shift;
			o[i] = s[i] - (d[i] << shift);
		}
	}

	void bitshiftRight(cv::InputArray src_, cv::OutputArray dest_, cv::OutputArray lostbit_, const int shift)
	{
		CV_Assert(src_.depth() == CV_8U);

		Mat src = src_.getMat();
		if (dest_.empty()) dest_.create(src.size(), src.type());
		lostbit_.create(src.size(), src.type());

		Mat dest = dest_.getMat();
		Mat lostbit = lostbit_.getMat();

		const int size = src.size().area() * src.channels();
		uchar* s = src.ptr<uchar>();
		uchar* d = dest.ptr<uchar>();
		uchar* o = lostbit.ptr<uchar>();
		const int simdsize = size / 16;
		const int remstart = simdsize * 16;

		uchar bitmask = 255;

		bitmask = ~(bitmask << shift);
		//	printf("%08d\n", BCD(bitmask));
		//	printf("%08d\n", BCD(bitmask));

		__m128i mmask = _mm_set1_epi8(bitmask);
		for (int i = 0; i < simdsize; i++)
		{
			__m128i ms = _mm_load_si128((__m128i*)s);
			__m128i v0 = _mm_srli_epi16(_mm_cvtepu8_epi16(ms), shift);
			__m128i v1 = _mm_srli_epi16(_mm_cvtepu8_epi16(_mm_load_si128((__m128i*)(s + 8))), shift);
			v0 = _mm_packus_epi16(v0, v1);
			_mm_store_si128((__m128i*)d, v0);
			_mm_store_si128((__m128i*)o, _mm_and_si128(ms, mmask));

			s += 16;
			o += 16;
			d += 16;
		}

		s = src.ptr<uchar>();
		d = dest.ptr<uchar>();
		o = lostbit.ptr<uchar>();

		for (int i = remstart; i < size; i++)
		{
			d[i] = s[i] >> shift;
			o[i] = s[i] - (d[i] << shift);
		}
	}

	void sqrtZeroClip(InputArray src_, OutputArray dst_)
	{
		CV_Assert(src_.depth() == CV_32F || src_.depth() == CV_64F);

		Mat src = src_.getMat();
		dst_.create(src.size(), src.type());
		Mat dst = dst_.getMat();

		if (src.depth() == CV_32F)
		{
			float* s = src.ptr <float>();
			float* d = dst.ptr <float>();
			const int simd_end = get_simd_floor((int)src.total(), 8);
			for (int i = 0; i < simd_end; i += 8)
			{
				_mm256_store_ps(d + i, _mm256_sqrt_ps(_mm256_max_ps(_mm256_setzero_ps(), _mm256_load_ps(s + i))));
			}
			for (int i = simd_end; i < src.total(); i++)
			{
				d[i] = sqrt(max(s[i], 0.f));
			}
		}
		else if (src.depth() == CV_64F)
		{
			double* s = src.ptr <double>();
			double* d = dst.ptr <double>();
			const int simd_end = get_simd_floor((int)src.total(), 4);
			for (int i = 0; i < simd_end; i += 4)
			{
				_mm256_store_pd(d + i, _mm256_sqrt_pd(_mm256_max_pd(_mm256_setzero_pd(), _mm256_load_pd(s + i))));
			}
			for (int i = simd_end; i < src.total(); i++)
			{
				d[i] = sqrt(max(s[i], 0.0));
			}
		}
	}

	template<class T>
	void clip(Mat& src, Mat& dst, const T minval, const T maxval)
	{
		const int size = (int)src.total();
		T* sptr = src.ptr<T>();
		T* dptr = dst.ptr<T>();
		for (int i = 0; i < size; i++)
		{
			dptr[i] = std::min(std::max(sptr[i], minval), maxval);
		}
	}

	template<>
	void clip<uchar>(Mat& src, Mat& dst, const uchar minval, const uchar maxval)
	{
		const __m256i mmin = _mm256_set1_epi8(minval);
		const __m256i mmax = _mm256_set1_epi8(maxval);

		const int size = (int)src.total();
		const int simdsize = get_simd_floor(size, 128);
		const int loopsize = size / 128;

		uchar* sptr = src.ptr<uchar>();
		uchar* dptr = dst.ptr<uchar>();
		__m256i* ms = (__m256i*)sptr;
		__m256i* md = (__m256i*)dptr;
		for (int i = 0; i < loopsize; i++)
		{
			*md++ = _mm256_min_epu8(_mm256_max_epu8(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_epu8(_mm256_max_epu8(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_epu8(_mm256_max_epu8(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_epu8(_mm256_max_epu8(*ms, mmin), mmax); ms++;
		}
		for (int i = simdsize; i < size; i++)
		{
			dptr[i] = std::min(std::max(sptr[i], minval), maxval);
		}
	}

	template<>
	void clip<char>(Mat& src, Mat& dst, const char minval, const char maxval)
	{
		const __m256i mmin = _mm256_set1_epi8(minval);
		const __m256i mmax = _mm256_set1_epi8(maxval);

		const int size = (int)src.total();
		const int simdsize = get_simd_floor(size, 128);
		const int loopsize = size / 128;

		char* sptr = src.ptr<char>();
		char* dptr = dst.ptr<char>();
		__m256i* ms = (__m256i*)sptr;
		__m256i* md = (__m256i*)dptr;
		for (int i = 0; i < loopsize; i++)
		{
			*md++ = _mm256_min_epi8(_mm256_max_epi8(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_epi8(_mm256_max_epi8(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_epi8(_mm256_max_epi8(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_epi8(_mm256_max_epi8(*ms, mmin), mmax); ms++;
		}
		for (int i = simdsize; i < size; i++)
		{
			dptr[i] = std::min(std::max(sptr[i], minval), maxval);
		}
	}


	template<>
	void clip<float>(Mat& src, Mat& dst, const float minval, const float maxval)
	{
		const __m256 mmin = _mm256_set1_ps(minval);
		const __m256 mmax = _mm256_set1_ps(maxval);

		const int size = (int)src.total();
		const int simdsize = get_simd_floor(size, 32);
		const int loopsize = size / 32;

		float* sptr = src.ptr<float>();
		float* dptr = dst.ptr<float>();
		__m256* ms = (__m256*)sptr;
		__m256* md = (__m256*)dptr;
		for (int i = 0; i < loopsize; i++)
		{
			*md++ = _mm256_min_ps(_mm256_max_ps(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_ps(_mm256_max_ps(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_ps(_mm256_max_ps(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_ps(_mm256_max_ps(*ms, mmin), mmax); ms++;
		}
		for (int i = simdsize; i < size; i++)
		{
			dptr[i] = std::min(std::max(sptr[i], minval), maxval);
		}
	}

	template<>
	void clip<double>(Mat& src, Mat& dst, const double minval, const double maxval)
	{
		const __m256d mmin = _mm256_set1_pd(minval);
		const __m256d mmax = _mm256_set1_pd(maxval);

		const int size = (int)src.total();
		const int simdsize = get_simd_floor(size, 16);
		const int loopsize = size / 16;

		double* sptr = src.ptr<double>();
		double* dptr = dst.ptr<double>();
		__m256d* ms = (__m256d*)sptr;
		__m256d* md = (__m256d*)dptr;
		for (int i = 0; i < loopsize; i++)
		{
			*md++ = _mm256_min_pd(_mm256_max_pd(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_pd(_mm256_max_pd(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_pd(_mm256_max_pd(*ms, mmin), mmax); ms++;
			*md++ = _mm256_min_pd(_mm256_max_pd(*ms, mmin), mmax); ms++;
		}
		for (int i = simdsize; i < size; i++)
		{
			dptr[i] = std::min(std::max(sptr[i], minval), maxval);
		}
	}

	void clip(InputArray src, OutputArray dst, const double minval, const double maxval)
	{
		Mat s = src.getMat();
		dst.create(src.size(), src.type());
		Mat d = dst.getMat();
		if (src.depth() == CV_8U)clip<uchar>(s, d, uchar(minval), uchar(maxval));
		if (src.depth() == CV_8S)clip<char>(s, d, char(minval), char(maxval));
		if (src.depth() == CV_16U)clip<ushort>(s, d, ushort(minval), ushort(maxval));
		if (src.depth() == CV_16S)clip<short>(s, d, short(minval), short(maxval));
		if (src.depth() == CV_32S)clip<int>(s, d, int(minval), int(maxval));
		if (src.depth() == CV_32F)clip<float>(s, d, float(minval), float(maxval));
		if (src.depth() == CV_64F)clip <double>(s, d, double(minval), double(maxval));
	}

	static void squareDiff_32F(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dest)
	{
		const int size = src1.size().area() * src1.channels();
		const int size32 = get_simd_floor(size, 32);
		const int size16 = get_simd_floor(size, 16);
		const int size8 = get_simd_floor(size, 8);
		const float* s1 = src1.ptr<float>();
		const float* s2 = src2.ptr<float>();
		float* d = dest.ptr<float>();
		const __m256i mask = get_simd_residualmask_epi32(size);
		for (int i = 0; i < size32; i += 32)
		{
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			__m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 8), _mm256_loadu_ps(s2 + i + 8));
			__m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 16), _mm256_loadu_ps(s2 + i + 16));
			__m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 24), _mm256_loadu_ps(s2 + i + 24));
			_mm256_storeu_ps(d + i + 0, _mm256_mul_ps(d0, d0));
			_mm256_storeu_ps(d + i + 8, _mm256_mul_ps(d1, d1));
			_mm256_storeu_ps(d + i + 16, _mm256_mul_ps(d2, d2));
			_mm256_storeu_ps(d + i + 24, _mm256_mul_ps(d3, d3));
		}
		for (int i = size32; i < size16; i += 16)
		{
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			__m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 8), _mm256_loadu_ps(s2 + i + 8));
			_mm256_storeu_ps(d + i + 0, _mm256_mul_ps(d0, d0));
			_mm256_storeu_ps(d + i + 8, _mm256_mul_ps(d1, d1));
		}
		for (int i = size16; i < size8; i += 8)
		{
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			_mm256_storeu_ps(d + i + 0, _mm256_mul_ps(d0, d0));
		}
		if (size8 < size)
		{
			int i = size8;
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			_mm256_maskstore_ps(d + i + 0, mask, _mm256_mul_ps(d0, d0));
		}
	}

	static void squareDiff_64F(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dest)
	{
		const int size = src1.size().area() * src1.channels();
		const int size16 = get_simd_floor(size, 16);
		const int size8 = get_simd_floor(size, 8);
		const int size4 = get_simd_floor(size, 4);
		const float* s1 = src1.ptr<float>();
		const float* s2 = src2.ptr<float>();
		float* d = dest.ptr<float>();
		const __m256i mask = get_simd_residualmask_epi32(size);
		for (int i = 0; i < size16; i += 16)
		{
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			__m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 4), _mm256_loadu_ps(s2 + i + 4));
			__m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 8), _mm256_loadu_ps(s2 + i + 8));
			__m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 12), _mm256_loadu_ps(s2 + i + 12));
			_mm256_storeu_ps(d + i + 0, _mm256_mul_ps(d0, d0));
			_mm256_storeu_ps(d + i + 4, _mm256_mul_ps(d1, d1));
			_mm256_storeu_ps(d + i + 8, _mm256_mul_ps(d2, d2));
			_mm256_storeu_ps(d + i + 12, _mm256_mul_ps(d3, d3));
		}
		for (int i = size16; i < size8; i += 8)
		{
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			__m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 4), _mm256_loadu_ps(s2 + i + 4));
			_mm256_storeu_ps(d + i + 0, _mm256_mul_ps(d0, d0));
			_mm256_storeu_ps(d + i + 4, _mm256_mul_ps(d1, d1));
		}
		for (int i = size8; i < size4; i += 4)
		{
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			_mm256_storeu_ps(d + i + 0, _mm256_mul_ps(d0, d0));
		}
		if (size4 < size)
		{
			int i = size4;
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(s1 + i + 0), _mm256_loadu_ps(s2 + i + 0));
			_mm256_maskstore_ps(d + i + 0, mask, _mm256_mul_ps(d0, d0));
		}
	}

	void squareDiff(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dest)
	{
		CV_Assert(src1.depth() == CV_32F || src1.depth() == CV_64F);
		CV_Assert(src1.isContinuous());
		if (dest.size() != src1.size() || src1.type() != dest.type()) dest.create(src1.size(), src1.type());
		Mat s1 = src1.getMat();
		Mat s2 = src2.getMat();
		Mat d = dest.getMat();
		if (s1.depth() == CV_32F) squareDiff_32F(s1, s2, d);
		else if (s1.depth() == CV_64F) squareDiff_64F(s1, s2, d);
	}
}