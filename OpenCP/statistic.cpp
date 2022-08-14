#include "statistic.hpp"
#include "inlineSIMDFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	void drawMinMaxPoints(InputArray src_, OutputArray dest, const uchar minv, const uchar maxv, Scalar minColor, Scalar maxColor, const int circle_r)
	{
		Mat src = src_.getMat();

		CV_Assert(src.channels() == 1);

		Mat s8u;
		src.convertTo(s8u, CV_8U);
		cvtColor(s8u, dest, COLOR_GRAY2BGR);
		Mat dst = dest.getMat();
		for (int j = 0; j < src.rows; j++)
		{
			const float* s = src.ptr<const float>(j);
			for (int i = 0; i < src.cols; i++)
			{
				if (s[i] == minv)
					circle(dst, Point(i, j), 3, minColor);
				if (s[i] == maxv)
					circle(dst, Point(i, j), 3, maxColor);
			}
		}
	}

	void calcMinMax(InputArray src_, uchar& minv, uchar& maxv)
	{
		Mat src = src_.getMat();
		const uchar* s = src.ptr<uchar>(0);
		const int size = src.size().area();
		const int simdsize = get_simd_floor(size, 128);

		__m256i minvec = _mm256_set1_epi8(0xFF);
		__m256i maxvec = _mm256_setzero_si256();
		for (int i = 0; i < simdsize; i += 128)
		{
			__m256i v1 = _mm256_load_si256((__m256i*)(s + i + 0));
			__m256i v2 = _mm256_load_si256((__m256i*)(s + i + 32));
			__m256i v3 = _mm256_load_si256((__m256i*)(s + i + 64));
			__m256i v4 = _mm256_load_si256((__m256i*)(s + i + 96));
			maxvec = _mm256_max_epu8(maxvec, v1);
			minvec = _mm256_min_epu8(minvec, v1);
			maxvec = _mm256_max_epu8(maxvec, v2);
			minvec = _mm256_min_epu8(minvec, v2);
			maxvec = _mm256_max_epu8(maxvec, v3);
			minvec = _mm256_min_epu8(minvec, v3);
			maxvec = _mm256_max_epu8(maxvec, v4);
			minvec = _mm256_min_epu8(minvec, v4);
		}

		uchar lminv = 255;
		uchar lmaxv = 0;
		for (int i = 0; i < 32; i++)
		{
			lmaxv = max(lmaxv, maxvec.m256i_u8[i]);
			lminv = min(lminv, minvec.m256i_u8[i]);
		}
		for (int i = simdsize; i < size; i++)
		{
			lmaxv = max(lmaxv, s[i]);
			lminv = min(lminv, s[i]);
		}

		minv = lminv;
		maxv = lmaxv;
	}

	double average(InputArray src_, const int left, const int right, const int top, const int bottom, const bool isNormalize)
	{
		Mat src = src_.getMat();
		CV_Assert(src.type() == CV_32FC1);
		__m256 msum1 = _mm256_setzero_ps();
		__m256 msum2 = _mm256_setzero_ps();
		__m256 msum3 = _mm256_setzero_ps();
		__m256 msum4 = _mm256_setzero_ps();

		const bool isFull = (left == 0 && right == 0 && top == 0 && bottom == 0);
		const int size = (isFull) ? src.size().area() : (src.cols - (left + right)) * (src.rows - (top + bottom));

		double sum = 0.0;
		if (isFull)
		{
			const float* sptr = src.ptr<float>();

			const int simdX = 256;
			const int simdY = size / simdX;
			const int remSt = simdY * simdX;
			for (int j = 0; j < simdY; j++)
			{
				msum1 = _mm256_setzero_ps();
				msum2 = _mm256_setzero_ps();
				msum3 = _mm256_setzero_ps();
				msum4 = _mm256_setzero_ps();
				const float* sptr2 = sptr + simdX * j;
				for (int i = 0; i < simdX; i += 32)
				{
					msum1 = _mm256_add_ps(_mm256_load_ps(sptr2 + i + 0), msum1);
					msum2 = _mm256_add_ps(_mm256_load_ps(sptr2 + i + 8), msum2);
					msum3 = _mm256_add_ps(_mm256_load_ps(sptr2 + i + 16), msum3);
					msum4 = _mm256_add_ps(_mm256_load_ps(sptr2 + i + 24), msum4);
				}
				sum += _mm256_reduceadd_pspd(msum1) + _mm256_reduceadd_pspd(msum2) + _mm256_reduceadd_pspd(msum3) + _mm256_reduceadd_pspd(msum4);
			}
			double rem = 0.0;
			for (int i = remSt; i < size; i++)
			{
				rem += sptr[i];
			}
			sum += rem;
		}
		else
		{
			const int simdend = get_simd_floor(src.cols - (left + right), 32) + left;
			for (int j = top; j < src.rows - bottom; j++)
			{
				const float* sptr = src.ptr<float>(j);
				msum1 = _mm256_setzero_ps();
				msum2 = _mm256_setzero_ps();
				msum3 = _mm256_setzero_ps();
				msum4 = _mm256_setzero_ps();
				for (int i = left; i < simdend; i += 32)
				{
					msum1 = _mm256_add_ps(_mm256_loadu_ps(sptr + i + 0), msum1);
					msum2 = _mm256_add_ps(_mm256_loadu_ps(sptr + i + 8), msum2);
					msum3 = _mm256_add_ps(_mm256_loadu_ps(sptr + i + 16), msum3);
					msum4 = _mm256_add_ps(_mm256_loadu_ps(sptr + i + 24), msum4);
				}
				sum += _mm256_reduceadd_pspd(msum1) + _mm256_reduceadd_pspd(msum2) + _mm256_reduceadd_pspd(msum3) + _mm256_reduceadd_pspd(msum4);
				double rem = 0.0;
				for (int i = simdend; i < src.cols - right; i++)
				{
					rem += sptr[i];
				}
				sum += rem;
			}
		}

		if (isNormalize)return sum / size;
		else return sum;
	}

	void weightedAverageVariance(InputArray src_, InputArray weight_, double& ave, double& var, const int left, const int right, const int top, const int bottom)
	{
		Mat src = src_.getMat();
		Mat weight = weight_.getMat();
		double wsum = 0.0;
		double sum = 0.0;
		double mulsum = 0.0;
		for (int j = top; j < src.rows - bottom; j++)
		{
			const float* sptr = src.ptr<float>(j);
			const float* wptr = weight.ptr<float>(j);
			for (int i = left; i < src.cols - right; i++)
			{
				wsum += wptr[i];
				sum += sptr[i] * wptr[i];
				mulsum += (sptr[i] * sptr[i] * wptr[i]);
			}
		}
		if (wsum == 0.0)
		{
			ave = 1.0;
			var = 0.0;
		}
		else
		{
			ave = sum / wsum;
			var = mulsum / wsum - ave * ave;
		}
	}

	void average_variance(InputArray src_, double& ave, double& var, const int left, const int right, const int top, const int bottom, const bool isNormalize)
	{
		Mat src = src_.getMat();
		CV_Assert(src.type() == CV_32FC1);
		__m256 msum1 = _mm256_setzero_ps();
		__m256 msum2 = _mm256_setzero_ps();
		__m256 msum3 = _mm256_setzero_ps();
		__m256 msum4 = _mm256_setzero_ps();
		__m256 mvar1 = _mm256_setzero_ps();
		__m256 mvar2 = _mm256_setzero_ps();
		__m256 mvar3 = _mm256_setzero_ps();
		__m256 mvar4 = _mm256_setzero_ps();

		const bool isFull = (left == 0 && right == 0 && top == 0 && bottom == 0);
		const int size = (isFull) ? src.size().area() : (src.cols - (left + right)) * (src.rows - (top + bottom));

		double sum = 0.0;
		double mulsum = 0.0;
		/*if (isFull)
		{
			const float* sptr = src.ptr<float>();
			const int simdSize = get_simd_floor(size, 32);
			for (int i = 0; i < simdSize; i += 32)
			{
				const __m256 ms1 = _mm256_load_ps(sptr + i + 0);
				const __m256 ms2 = _mm256_load_ps(sptr + i + 8);
				const __m256 ms3 = _mm256_load_ps(sptr + i + 16);
				const __m256 ms4 = _mm256_load_ps(sptr + i + 24);
				msum1 = _mm256_add_ps(ms1, msum1);
				msum2 = _mm256_add_ps(ms2, msum2);
				msum3 = _mm256_add_ps(ms3, msum3);
				msum4 = _mm256_add_ps(ms4, msum4);
				mvar1 = _mm256_fmadd_ps(ms1, ms1, mvar1);
				mvar2 = _mm256_fmadd_ps(ms2, ms2, mvar2);
				mvar3 = _mm256_fmadd_ps(ms3, ms3, mvar3);
				mvar4 = _mm256_fmadd_ps(ms4, ms4, mvar4);
			}
			sum += _mm256_reduceadd_pspd(msum1) + _mm256_reduceadd_pspd(msum2) + _mm256_reduceadd_pspd(msum3) + _mm256_reduceadd_pspd(msum4);
			mulsum += _mm256_reduceadd_pspd(mvar1) + _mm256_reduceadd_pspd(mvar2) + _mm256_reduceadd_pspd(mvar3) + _mm256_reduceadd_pspd(mvar4);
			float remsum = 0.f;
			float remmulsum = 0.f;
			for (int i = simdSize; i < size; i++)
			{
				remsum += sptr[i];
				remmulsum += sptr[i] * sptr[i];
			}
			sum += remsum;
			mulsum += remmulsum;
		}
		else*/
		{
			for (int j = top; j < src.rows - bottom; j++)
			{
				const float* sptr = src.ptr<float>(j);
				for (int i = left; i < src.cols - right; i++)
				{
					sum += sptr[i];
					mulsum += (sptr[i] * sptr[i]);
				}
			}
			/*
			const int simdend = get_simd_floor(src.cols - (left + right), 32) + left;
			for (int j = top; j < src.rows - bottom; j++)
			{
				const float* sptr = src.ptr<float>(j);
				msum1 = _mm256_setzero_ps();
				msum2 = _mm256_setzero_ps();
				msum3 = _mm256_setzero_ps();
				msum4 = _mm256_setzero_ps();
				mvar1 = _mm256_setzero_ps();
				mvar2 = _mm256_setzero_ps();
				mvar3 = _mm256_setzero_ps();
				mvar4 = _mm256_setzero_ps();
				for (int i = left; i < simdend; i += 32)
				{
					const __m256 ms1 = _mm256_loadu_ps(sptr + i + 0);
					const __m256 ms2 = _mm256_loadu_ps(sptr + i + 8);
					const __m256 ms3 = _mm256_loadu_ps(sptr + i + 16);
					const __m256 ms4 = _mm256_loadu_ps(sptr + i + 24);
					msum1 = _mm256_add_ps(ms1, msum1);
					msum2 = _mm256_add_ps(ms2, msum2);
					msum3 = _mm256_add_ps(ms3, msum3);
					msum4 = _mm256_add_ps(ms4, msum4);
					mvar1 = _mm256_fmadd_ps(ms1, ms1, mvar1);
					mvar2 = _mm256_fmadd_ps(ms2, ms2, mvar2);
					mvar3 = _mm256_fmadd_ps(ms3, ms3, mvar3);
					mvar4 = _mm256_fmadd_ps(ms4, ms4, mvar4);
				}
				sum += _mm256_reduceadd_pspd(msum1) + _mm256_reduceadd_pspd(msum2) + _mm256_reduceadd_pspd(msum3) + _mm256_reduceadd_pspd(msum4);
				mulsum += _mm256_reduceadd_pspd(mvar1) + _mm256_reduceadd_pspd(mvar2) + _mm256_reduceadd_pspd(mvar3) + _mm256_reduceadd_pspd(mvar4);
				float remsum = 0.f;
				float remmulsum = 0.f;
				for (int i = simdend; i < src.cols - right; i++)
				{
					remsum += sptr[i];
					remmulsum += sptr[i] * sptr[i];
				}
				sum += remsum;
				mulsum += remmulsum;
			}*/
		}
		if (isNormalize)
		{
			ave = sum / size;
			var = mulsum / size - ave * ave;
		}
		else
		{
			ave = sum;
			var = mulsum;
		}
	}

	//do not have interface in opencp.hpp
	void average_var_accurate(const Mat& src, double& ave, double& var, const bool isNormalize = true)
	{
		const float* sptr = src.ptr<float>();
		const int size = src.size().area();
		const int simdSize = get_simd_floor(size, 32);

		__m256 msum1 = _mm256_setzero_ps();
		__m256 msum2 = _mm256_setzero_ps();
		__m256 msum3 = _mm256_setzero_ps();
		__m256 msum4 = _mm256_setzero_ps();
		__m256 mvar1 = _mm256_setzero_ps();
		__m256 mvar2 = _mm256_setzero_ps();
		__m256 mvar3 = _mm256_setzero_ps();
		__m256 mvar4 = _mm256_setzero_ps();
		for (int i = 0; i < simdSize; i += 32)
		{
			const __m256 ms1 = _mm256_load_ps(sptr + i + 0);
			const __m256 ms2 = _mm256_load_ps(sptr + i + 8);
			const __m256 ms3 = _mm256_load_ps(sptr + i + 16);
			const __m256 ms4 = _mm256_load_ps(sptr + i + 24);
			msum1 = _mm256_add_ps(ms1, msum1);
			msum2 = _mm256_add_ps(ms2, msum2);
			msum3 = _mm256_add_ps(ms3, msum3);
			msum4 = _mm256_add_ps(ms4, msum4);
		}
		double sum = _mm256_reduceadd_pspd(msum1) + _mm256_reduceadd_pspd(msum2) + _mm256_reduceadd_pspd(msum3) + _mm256_reduceadd_pspd(msum4);
		float remsum = 0.f;
		for (int i = simdSize; i < size; i++)
		{
			remsum += sptr[i];
		}
		sum += remsum;
		ave = sum / size;
		const __m256 mave = _mm256_set1_ps((float)ave);
		for (int i = 0; i < simdSize; i += 32)
		{
			const __m256 ms1 = _mm256_sub_ps(_mm256_load_ps(sptr + i + 0), mave);
			const __m256 ms2 = _mm256_sub_ps(_mm256_load_ps(sptr + i + 8), mave);
			const __m256 ms3 = _mm256_sub_ps(_mm256_load_ps(sptr + i + 16), mave);
			const __m256 ms4 = _mm256_sub_ps(_mm256_load_ps(sptr + i + 24), mave);

			mvar1 = _mm256_fmadd_ps(ms1, ms1, mvar1);
			mvar2 = _mm256_fmadd_ps(ms2, ms2, mvar2);
			mvar3 = _mm256_fmadd_ps(ms3, ms3, mvar3);
			mvar4 = _mm256_fmadd_ps(ms4, ms4, mvar4);
		}
		float remmulsum = 0.f;
		for (int i = simdSize; i < size; i++)
		{
			remmulsum += (sptr[i] - (float)ave) * (sptr[i] - (float)ave);
		}
		double mulsum = _mm256_reduceadd_pspd(mvar1) + _mm256_reduceadd_pspd(mvar2) + _mm256_reduceadd_pspd(mvar3) + _mm256_reduceadd_pspd(mvar4);
		mulsum += remmulsum;

		if (isNormalize)
		{
			var = mulsum / size;
		}
		else
		{
			ave = sum;
			var = mulsum;
		}
	}
}