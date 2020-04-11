#include "arithmetic.hpp"
#include "fmath/fmath.hpp"
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


	void pow_fmath(const float a, const Mat& src, Mat & dest)
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

	void pow_fmath(const Mat& src, const float a, Mat & dest)
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

	void pow_fmath(const Mat& src1, const Mat& src2, Mat & dest)
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


	void fnmsub(Mat& a, Mat& x, Mat& b, Mat& dest)
	{
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area()*a.channels();
		if (a.depth() == CV_32F)
		{
			float* aptr = a.ptr<float>();
			float* xptr = x.ptr<float>();
			float* bptr = b.ptr<float>();
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
				dest.at<float>(i) = -a.at<float>(i)*x.at<float>(i) - b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			double* aptr = a.ptr<double>();
			double* xptr = x.ptr<double>();
			double* bptr = b.ptr<double>();
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
				dest.at<double>(i) = -a.at<double>(i)*x.at<double>(i) - b.at<double>(i);
			}
		}
	}

	void fnmadd(Mat& a, Mat& x, Mat& b, Mat& dest)
	{
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area()*a.channels();
		if (a.depth() == CV_32F)
		{
			float* aptr = a.ptr<float>();
			float* xptr = x.ptr<float>();
			float* bptr = b.ptr<float>();
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
				dest.at<float>(i) = -a.at<float>(i)*x.at<float>(i) + b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			double* aptr = a.ptr<double>();
			double* xptr = x.ptr<double>();
			double* bptr = b.ptr<double>();
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
				dest.at<double>(i) = -a.at<double>(i)*x.at<double>(i) + b.at<double>(i);
			}
		}
	}

	void fmsub(Mat& a, Mat& x, Mat& b, Mat& dest)
	{
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area()*a.channels();
		if (a.depth() == CV_32F)
		{
			float* aptr = a.ptr<float>();
			float* xptr = x.ptr<float>();
			float* bptr = b.ptr<float>();
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
				dest.at<float>(i) = a.at<float>(i)*x.at<float>(i) - b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			double* aptr = a.ptr<double>();
			double* xptr = x.ptr<double>();
			double* bptr = b.ptr<double>();
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
				dest.at<double>(i) = a.at<double>(i)*x.at<double>(i) - b.at<double>(i);
			}
		}
	}

	void fmadd(Mat& a, Mat& x, Mat& b, Mat& dest)
	{
		CV_Assert(a.depth() == CV_32F || a.depth() == CV_64F);
		dest.create(a.size(), a.type());

		const int size = a.size().area()*a.channels();
		if (a.depth() == CV_32F)
		{
			float* aptr = a.ptr<float>();
			float* xptr = x.ptr<float>();
			float* bptr = b.ptr<float>();
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
				dest.at<float>(i) = a.at<float>(i)*x.at<float>(i) + b.at<float>(i);
			}
		}
		else if (a.depth() == CV_64F)
		{
			double* aptr = a.ptr<double>();
			double* xptr = x.ptr<double>();
			double* bptr = b.ptr<double>();
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
				dest.at<double>(i) = a.at<double>(i)*x.at<double>(i) + b.at<double>(i);
			}
		}
	}


	void bitshiftRight_nonsimd(Mat& src, Mat& dest, const int shift)
	{
		CV_Assert(src.depth() == CV_8U);
		if (src.data != dest.data) dest.create(src.size(), src.type());

		const int size = src.size().area()*src.channels();
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

		Mat src = src_.getMat();
		if (dest_.empty()) dest_.create(src.size(), src.type());
		Mat dest = dest_.getMat();
		//if (src.data != dest.data) dest_.create(src.size(), src.type());

		const int size = src.size().area()*src.channels();
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

		const int size = src.size().area()*src.channels();
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
	
		const int size = src.size().area()*src.channels();
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
}