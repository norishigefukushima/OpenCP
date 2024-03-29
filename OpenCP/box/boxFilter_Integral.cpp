#include "boxFilter_Integral.h"

using namespace cv;
using namespace std;

/* --- boxFilter Integral --- */
boxFilterIntegralScalar::boxFilterIntegralScalar(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: BoxFilterBase(_src, _dest, _r, _parallelType)
{
	ksize = 2 * r + 1;
	cn = src.channels();
	copyMakeBorder(src, copy, r, r, r, r, BOX_FILTER_BORDER_TYPE);
	cv::integral(copy, sum, CV_32F);
}

/*
 * nonVec
 */
void boxFilterIntegralScalar::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		float* s1 = sum.ptr<float>(y);
		float* s2 = sum.ptr<float>(y) + cn * ksize;
		float* s3 = sum.ptr<float>(y + ksize);
		float* s4 = sum.ptr<float>(y + ksize) + cn * ksize;
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x++)
		{
			*dp = (*s4 - *s3 - *s2 + *s1)*div;
			s1++, s2++, s3++, s4++, dp++;
		}
	}
}

void boxFilterIntegralScalar::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		float* s1 = sum.ptr<float>(y);
		float* s2 = sum.ptr<float>(y) + cn * ksize;
		float* s3 = sum.ptr<float>(y + ksize);
		float* s4 = sum.ptr<float>(y + ksize) + cn * ksize;
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x++)
		{
			*dp = (*s4 - *s3 - *s2 + *s1)*div;
			s1++, s2++, s3++, s4++, dp++;
		}
	}
}

void boxFilterIntegralScalar::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x++)
		{
			const float* s1 = sum.ptr<float>(y) + x;
			const float* s2 = sum.ptr<float>(y) + cn * ksize + x;
			const float* s3 = sum.ptr<float>(y + ksize) + x;
			const float* s4 = sum.ptr<float>(y + ksize) + cn * ksize + x;

			*dp = (*s4 - *s3 - *s2 + *s1)*div;
			dp++;
		}
	}
}



/*
 * SSE
 */
void boxFilterIntegralSSE::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x += 4)
		{
			//*dp = (*s4 - *s3 - *s2 + *s1)*div;
			//s1++, s2++, s3++, s4++, dp++;
			const float* s1 = sum.ptr<float>(y) + x;
			const float* s2 = sum.ptr<float>(y) + cn * ksize + x;
			const float* s3 = sum.ptr<float>(y + ksize) + x;
			const float* s4 = sum.ptr<float>(y + ksize) + cn * ksize + x;

			__m128 mTmp = _mm_loadu_ps(s4);
			mTmp = _mm_sub_ps(mTmp, _mm_loadu_ps(s3));
			mTmp = _mm_sub_ps(mTmp, _mm_loadu_ps(s2));
			mTmp = _mm_add_ps(mTmp, _mm_loadu_ps(s1));
			mTmp = _mm_mul_ps(mTmp, mDiv);
			_mm_storeu_ps(dp, mTmp);
			dp += 4;
		}
	}
}

void boxFilterIntegralSSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x += 4)
		{
			//*dp = (*s4 - *s3 - *s2 + *s1)*div;
			//s1++, s2++, s3++, s4++, dp++;
			const float* s1 = sum.ptr<float>(y) + x;
			const float* s2 = sum.ptr<float>(y) + cn * ksize + x;
			const float* s3 = sum.ptr<float>(y + ksize) + x;
			const float* s4 = sum.ptr<float>(y + ksize) + cn * ksize + x;

			__m128 mTmp = _mm_loadu_ps(s4);
			mTmp = _mm_sub_ps(mTmp, _mm_loadu_ps(s3));
			mTmp = _mm_sub_ps(mTmp, _mm_loadu_ps(s2));
			mTmp = _mm_add_ps(mTmp, _mm_loadu_ps(s1));
			mTmp = _mm_mul_ps(mTmp, mDiv);
			_mm_storeu_ps(dp, mTmp);
			dp += 4;
		}
	}
}

void boxFilterIntegralSSE::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x += 4)
		{
			//*dp = (*s4 - *s3 - *s2 + *s1)*div;
			//s1++, s2++, s3++, s4++, dp++;
			const float* s1 = sum.ptr<float>(y) + x;
			const float* s2 = sum.ptr<float>(y) + cn * ksize + x;
			const float* s3 = sum.ptr<float>(y + ksize) + x;
			const float* s4 = sum.ptr<float>(y + ksize) + cn * ksize + x;

			__m128 mTmp = _mm_loadu_ps(s4);
			mTmp = _mm_sub_ps(mTmp, _mm_loadu_ps(s3));
			mTmp = _mm_sub_ps(mTmp, _mm_loadu_ps(s2));
			mTmp = _mm_add_ps(mTmp, _mm_loadu_ps(s1));
			mTmp = _mm_mul_ps(mTmp, mDiv);
			_mm_storeu_ps(dp, mTmp);
			dp += 4;
		}
	}
}



/*
 * AVX
 */
void boxFilterIntegralAVX::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x += 8)
		{
			//*dp = (*s4 - *s3 - *s2 + *s1)*div;
			//s1++, s2++, s3++, s4++, dp++;
			const float* s1 = sum.ptr<float>(y) + x;
			const float* s2 = sum.ptr<float>(y) + cn * ksize + x;
			const float* s3 = sum.ptr<float>(y + ksize) + x;
			const float* s4 = sum.ptr<float>(y + ksize) + cn * ksize + x;

			__m256 mTmp = _mm256_loadu_ps(s4);
			mTmp = _mm256_sub_ps(mTmp, _mm256_loadu_ps(s3));
			mTmp = _mm256_sub_ps(mTmp, _mm256_loadu_ps(s2));
			mTmp = _mm256_add_ps(mTmp, _mm256_loadu_ps(s1));
			mTmp = _mm256_mul_ps(mTmp, mDiv);
			_mm256_storeu_ps(dp, mTmp);
			dp += 8;
		}
	}
}

void boxFilterIntegralAVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x += 8)
		{
			//*dp = (*s4 - *s3 - *s2 + *s1)*div;
			//s1++, s2++, s3++, s4++, dp++;
			const float* s1 = sum.ptr<float>(y) + x;
			const float* s2 = sum.ptr<float>(y) + cn * ksize + x;
			const float* s3 = sum.ptr<float>(y + ksize) + x;
			const float* s4 = sum.ptr<float>(y + ksize) + cn * ksize + x;

			__m256 mTmp = _mm256_loadu_ps(s4);
			mTmp = _mm256_sub_ps(mTmp, _mm256_loadu_ps(s3));
			mTmp = _mm256_sub_ps(mTmp, _mm256_loadu_ps(s2));
			mTmp = _mm256_add_ps(mTmp, _mm256_loadu_ps(s1));
			mTmp = _mm256_mul_ps(mTmp, mDiv);
			_mm256_storeu_ps(dp, mTmp);
			dp += 8;
		}
	}
}

void boxFilterIntegralAVX::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col * cn; x += 8)
		{
			//*dp = (*s4 - *s3 - *s2 + *s1)*div;
			//s1++, s2++, s3++, s4++, dp++;
			const float* s1 = sum.ptr<float>(y) + x;
			const float* s2 = sum.ptr<float>(y) + cn * ksize + x;
			const float* s3 = sum.ptr<float>(y + ksize) + x;
			const float* s4 = sum.ptr<float>(y + ksize) + cn * ksize + x;

			__m256 mTmp = _mm256_loadu_ps(s4);
			mTmp = _mm256_sub_ps(mTmp, _mm256_loadu_ps(s3));
			mTmp = _mm256_sub_ps(mTmp, _mm256_loadu_ps(s2));
			mTmp = _mm256_add_ps(mTmp, _mm256_loadu_ps(s1));
			mTmp = _mm256_mul_ps(mTmp, mDiv);
			_mm256_storeu_ps(dp, mTmp);
			dp += 8;
		}
	}
}
