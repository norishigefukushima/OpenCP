#include "metricsDD.hpp"
#include "inlineSIMDFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	double getPSNR_DD(doubledouble* src1, doubledouble* src2, const int size)
	{
		doubledouble ret = doubledouble{ 0.0,0.0 };
		doubledouble m = doubledouble{ -1.0,0.0 };
		
		for (int i = 0; i < size; i++)
		{
			doubledouble v = ddsub(src2[i], src1[i]);
			v = ddmul(v, v);
			ret = ddadd(ret, v);
		}

		double mse = ret.hi / (double)size;

		if (mse == 0.0)
		{
			return 0;
		}
		else if (cvIsNaN(mse) || cvIsInf(mse))
		{
			std::cout << "mse = NaN" << std::endl;
			return 0;
		}
		else if (cvIsInf(mse))
		{
			std::cout << "mse = Inf" << std::endl;
			return 0;
		}

		return 10.0 * log10(255.0 * 255.0 / mse);
	}

	double getPSNR_DD(doubledouble* src1, cv::Mat& src2)
	{
		if (src2.depth() == CV_64F && src2.channels() == 2)
		{
			doubledouble* a = (doubledouble*)src2.ptr<double>();
			return getPSNR_DD(src1, a, src2.size().area());
		}
		const int size = src2.size().area();
		doubledouble ret = doubledouble{ 0.0, 0.0 };

		if (src2.depth() == CV_64F)
		{
			const int SIZE = get_simd_floor(size, 4);
			double* s2 = src2.ptr<double>();
			__m256dd r = _mm256_setzero_pdd();
			for (int i = 0; i < SIZE; i += 4)
			{
				__m256dd s = _mm256_loaddeinterleave_pdd(src1 + i);
				s = _mm256_subw_pdd(s, _mm256_loadu_pd(s2 + i));
				s = _mm256_mul_pdd(s, s);
				r = _mm256_add_pdd(r, s);
			}
			ret = _mm256_reduce_add_pdd(r);

			for (int i = SIZE; i < size; i++)
			{
				doubledouble v = ddsubw(src1[i], src2.at<double>(i));
				v = ddmul(v, v);
				ret = ddadd(ret, v);
			}
		}
		else if (src2.depth() == CV_32F)
		{
#if 1
			const int SIZE = get_simd_floor(size, 4);
			float* s2 = src2.ptr<float>();
			__m256dd r = _mm256_setzero_pdd();
			for (int i = 0; i < SIZE; i += 4)
			{
				__m256dd s = _mm256_loaddeinterleave_pdd(src1 + i);
				s = _mm256_subw_pdd(s, _mm256_cvtps_pd(_mm_loadu_ps(s2 + i)));
				s = _mm256_mul_pdd(s, s);
				r = _mm256_add_pdd(r, s);
			}
			ret = _mm256_reduce_add_pdd(r);

			for (int i = SIZE; i < size; i++)
			{
				doubledouble v = ddsubw(src1[i], src2.at<double>(i));
				v = ddmul(v, v);
				ret = ddadd(ret, v);
			}
#else
			for (int i = 0; i < size; i++)
			{
				doubledouble v = ddaddw(src1[i], -src2.at<float>(i));
				v = ddmul(v, v);
				ret = ddadd(ret, v);
			}
#endif
		}
		else if (src2.depth() == CV_8U)
		{
#if 1
			const int SIZE = get_simd_floor(size, 4);
			uchar* s2 = src2.ptr<uchar>();
			__m256dd r = _mm256_setzero_pdd();
			for (int i = 0; i < SIZE; i += 4)
			{
				__m256dd s = _mm256_loaddeinterleave_pdd(src1 + i);

				s = _mm256_subw_pdd(s, _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(s2 + i)))));
				s = _mm256_mul_pdd(s, s);
				r = _mm256_add_pdd(r, s);
			}
			ret = _mm256_reduce_add_pdd(r);

			for (int i = SIZE; i < size; i++)
			{
				doubledouble v = ddsubw(src1[i], src2.at<double>(i));
				v = ddmul(v, v);
				ret = ddadd(ret, v);
			}
#else
			for (int i = 0; i < size; i++)
			{
				doubledouble v = ddaddw(src1[i], -double(src2.at<uchar>(i)));
				v = ddmul(v, v);
				ret = ddadd(ret, v);
			}
#endif
		}
		else
		{
			cout << "do not support this type getPSNR_DD" << endl;
		}

		double mse = ret.hi / (double)size;

		if (mse == 0.0)
		{
			return 0.0;
		}
		else if (cvIsNaN(mse) || cvIsInf(mse))
		{
			std::cout << "mse = NaN" << std::endl;
			return 0.0;
		}
		else if (cvIsInf(mse))
		{
			std::cout << "mse = Inf" << std::endl;
			return 0.0;
		}

		return 10.0 * log10(255.0 * 255.0 / mse);
	}

	double getPSNR_DD(cv::Mat& src1, doubledouble* src2)
	{
		return getPSNR_DD(src2, src1);
	}

	double getPSNR_DD(cv::Mat& src1, cv::Mat& src2)
	{
		double ret = 0.0;
		if (src1.depth() == CV_64F && src1.channels() == 2 && src2.depth() == CV_64F && src2.channels() == 2)
		{
			doubledouble* a = (doubledouble*)src1.ptr<double>();
			doubledouble* b = (doubledouble*)src2.ptr<double>();
			ret = getPSNR_DD(a, b, src1.size().area());
		}
		else if (src1.depth() == CV_64F && src1.channels() == 2)
		{
			doubledouble* a = (doubledouble*)src1.ptr<double>();
			ret = getPSNR_DD(a, src2);
		}
		else if (src2.depth() == CV_64F && src2.channels() == 2)
		{
			doubledouble* b = (doubledouble*)src2.ptr<double>();
			ret = getPSNR_DD(src1, b);
		}
		else
		{
			std::cout << "input is not DD" << std::endl;
		}
		return ret;
	}
}