#include "boxFilter_SSAT_HV.h"

using namespace cv;
using namespace std;

void boxFilter_SSAT_HV_nonVec::filter()
{
	RowSumFilter(src, temp, r, parallelType).filter();
	ColumnSumFilter_nonVec(temp, dest, r, parallelType).filter();
}

void boxFilter_SSAT_HV_SSE::filter()
{
	if (src.channels() % 4 == 0)
		RowSumFilter_SSE(src, temp, r, parallelType).filter();
	else
		RowSumFilter(src, temp, r, parallelType).filter();
	ColumnSumFilter_SSE(temp, dest, r, parallelType).filter();
}

void boxFilter_SSAT_HV_AVX::filter()
{
	if (src.channels() % 8 == 0)
		RowSumFilter_AVX(src, temp, r, parallelType).filter();
	else
		RowSumFilter(src, temp, r, parallelType).filter();
	ColumnSumFilter_AVX(temp, dest, r, parallelType).filter();
}




void RowSumFilter::filter_naive_impl()
{
	for (int j = 0; j < row; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;

		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(j) + k;

			float sum = 0.f;

			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}
}

void RowSumFilter::filter_omp_impl()
{
#pragma omp parallel for
	//#pragma omp parallel for num_threads(numOfThreads)
	for (int j = 0; j < row; j++)
	{
		for (int k = 0; k < cn; k++)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			float sum = 0.f;

			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}
}

void RowSumFilter::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;

		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(j) + k;

			float sum = 0.f;

			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}
}



void RowSumFilter_SSE::filter_naive_impl()
{
	for (int j = 0; j < row; j++)
	{
		for (int k = 0; k < cn; k += 4)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m128 mSum = _mm_setzero_ps();

			__m128 mPrev = _mm_load_ps(sp1);
			__m128 mNext;
			mSum = _mm_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < row; j++)
	{
		for (int k = 0; k < cn; k += 4)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m128 mSum = _mm_setzero_ps();

			__m128 mPrev = _mm_load_ps(sp1);
			__m128 mNext;
			mSum = _mm_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_SSE::operator()(const cv::Range& range) const
{

}



void RowSumFilter_AVX::filter_naive_impl()
{
	for (int j = 0; j < row; j++)
	{
		for (int k = 0; k < cn; k += 8)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m256 mSum = _mm256_setzero_ps();

			__m256 mPrev = _mm256_load_ps(sp1);
			__m256 mNext;
			mSum = _mm256_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm256_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm256_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < row; j++)
	{
		for (int k = 0; k < cn; k += 8)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m256 mSum = _mm256_setzero_ps();

			__m256 mPrev = _mm256_load_ps(sp1);
			__m256 mNext;
			mSum = _mm256_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm256_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm256_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_AVX::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonVec::filter_naive_impl()
{
	for (int i = 0; i < step; i++)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		float sum = 0.f;
		sum += *sp1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = sum*div;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = sum*div;
			dp += step;
		}
	}
}

void ColumnSumFilter_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < step; i++)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		float sum = 0.f;
		sum += *sp1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = sum*div;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = sum*div;
			dp += step;
		}
	}

	/*for (int i = 0; i < col; i++)
	{
#pragma omp parallel for
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(0) + i * cn + k;
			float* sp2 = src.ptr<float>(1) + i * cn + k;
			float* dp = dest.ptr<float>(0) + i * cn + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int j = 1; j <= r; j++)
			{
				sum += *sp2;
				sp2 += step;
			}
			*dp = sum*div;
			dp += step;

			for (int j = 1; j <= r; j++)
			{
				sum += *sp2 - *sp1;
				sp2 += step;
				*dp = sum*div;
				dp += step;
			}
			for (int j = r + 1; j < row - r - 1; j++)
			{
				sum += *sp2 - *sp1;
				sp1 += step;
				sp2 += step;
				*dp = sum*div;
				dp += step;
			}
			for (int j = row - r - 1; j < row; j++)
			{
				sum += *sp2 - *sp1;
				sp1 += step;
				*dp = sum*div;
				dp += step;
			}
		}
	}*/
}

void ColumnSumFilter_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		float sum = 0.f;
		sum += *sp1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = sum*div;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = sum*div;
			dp += step;
		}
	}
}

void ColumnSumFilter_nonVec::filter()
{
	if (parallelType == ParallelTypes::NAIVE)
	{
		filter_naive_impl();
	}
	else if (parallelType == ParallelTypes::OMP)
	{
		filter_omp_impl();
	}
	else if (parallelType == ParallelTypes::PARALLEL_FOR_)
	{
		cv::parallel_for_(cv::Range(0, step), *this, cv::getNumThreads() - 1);
	}
	else
	{

	}
}



void ColumnSumFilter_SSE::filter_naive_impl()
{
	for (int i = 0; i < step; i += 4)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		__m128 mTmp = _mm_setzero_ps();
		__m128 mSum = _mm_setzero_ps();
		mSum = _mm_mul_ps(_mm_set1_ps((float)r + 1), _mm_load_ps(sp1));
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
		}
		_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, mTmp);
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm_add_ps(mSum, mTmp);
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
	}

	//if (step != nn)
	//{
	//	for (int i = nn; i < step; i++)
	//	{
	//		float* sp1 = src.ptr<float>(0) + i;
	//		float* sp2 = src.ptr<float>(1) + i;
	//		float* dp = dest.ptr<float>(0) + i;

	//		float sum = 0.f;
	//		sum += *sp1 * (r + 1);
	//		for (int j = 1; j <= r; j++)
	//		{
	//			sum += *sp2;
	//			sp2 += step;
	//		}
	//		*dp = sum*div;
	//		dp += step;

	//		for (int j = 1; j <= r; j++)
	//		{
	//			sum += *sp2 - *sp1;
	//			sp2 += step;
	//			*dp = sum*div;
	//			dp += step;
	//		}
	//		for (int j = r + 1; j < row - r - 1; j++)
	//		{
	//			sum += *sp2 - *sp1;
	//			sp1 += step;
	//			sp2 += step;
	//			*dp = sum*div;
	//			dp += step;
	//		}
	//		for (int j = row - r - 1; j < row; j++)
	//		{
	//			sum += *sp2 - *sp1;
	//			sp1 += step;
	//			*dp = sum*div;
	//			dp += step;
	//		}
	//	}
	//}
}

void ColumnSumFilter_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < step; i += 4)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		__m128 mTmp = _mm_setzero_ps();
		__m128 mSum = _mm_setzero_ps();
		mSum = _mm_mul_ps(_mm_set1_ps((float)r + 1), _mm_load_ps(sp1));
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
		}
		_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, mTmp);
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm_add_ps(mSum, mTmp);
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
	}

	/*for (int i = 0; i < col; i++)
	{
#pragma omp parallel for
		for (int k = 0; k < cn; k += 4)
		{
			float* sp1 = src.ptr<float>(0) + i * cn + k;
			float* sp2 = src.ptr<float>(1) + i * cn + k;
			float* dp = dest.ptr<float>(0) + i * cn + k;

			__m128 mTmp = _mm_setzero_ps();
			__m128 mSum = _mm_setzero_ps();
			mSum = _mm_mul_ps(_mm_set1_ps((float)r + 1), _mm_load_ps(sp1));
			for (int j = 1; j <= r; j++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
				sp2 += step;
			}
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;

			mTmp = _mm_load_ps(sp1);
			for (int j = 1; j <= r; j++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
				sp2 += step;
				mSum = _mm_sub_ps(mSum, mTmp);
				_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			for (int j = r + 1; j < row - r - 1; j++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
				sp2 += step;
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += step;
				_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			mTmp = _mm_load_ps(sp2);
			for (int j = row - r - 1; j < row; j++)
			{
				mSum = _mm_add_ps(mSum, mTmp);
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += step;
				_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
		}
	}*/
}

void ColumnSumFilter_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i += 4)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		__m128 mTmp = _mm_setzero_ps();
		__m128 mSum = _mm_setzero_ps();
		mSum = _mm_mul_ps(_mm_set1_ps((float)r + 1), _mm_load_ps(sp1));
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
		}
		_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, mTmp);
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm_add_ps(mSum, mTmp);
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}



void ColumnSumFilter_AVX::filter_naive_impl()
{
	for (int i = 0; i < step; i += 8)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		__m256 mTmp = _mm256_setzero_ps();
		__m256 mSum = _mm256_setzero_ps();
		mSum = _mm256_mul_ps(_mm256_set1_ps((float)r + 1), _mm256_load_ps(sp1));
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
		}
		_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm256_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, mTmp);
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
			sp2 += step;
			//_mm_prefetch((char *)sp2, _MM_HINT_NTA);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm256_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm256_add_ps(mSum, mTmp);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}

void ColumnSumFilter_AVX::filter_omp_impl()
{
#pragma omp parallel for
	//#pragma omp parallel for num_threads(numOfThreads)
	for (int i = 0; i < step; i += 8)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		__m256 mTmp = _mm256_setzero_ps();
		__m256 mSum = _mm256_setzero_ps();
		mSum = _mm256_mul_ps(_mm256_set1_ps((float)r + 1), _mm256_load_ps(sp1));
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
		}
		_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm256_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, mTmp);
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
			sp2 += step;
			//_mm_prefetch((char *)sp2, _MM_HINT_NTA);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm256_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm256_add_ps(mSum, mTmp);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
	}

	//	for (int i = 0; i < col; i++)
	//	{
	//#pragma omp parallel for
	//		for (int k = 0; k < cn; k += 8)
	//		{
	//			float* sp1 = src.ptr<float>(0) + i * cn + k;
	//			float* sp2 = src.ptr<float>(1) + i * cn + k;
	//			float* dp = dest.ptr<float>(0) + i * cn + k;
	//			
	//			__m256 mTmp = _mm256_setzero_ps();
	//			__m256 mSum = _mm256_setzero_ps();
	//			mSum = _mm256_mul_ps(_mm256_set1_ps((float)r + 1), _mm256_load_ps(sp1));
	//			for (int j = 1; j <= r; j++)
	//			{
	//				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
	//				sp2 += step;
	//			}
	//			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
	//			dp += step;
	//
	//			mTmp = _mm256_load_ps(sp1);
	//			for (int j = 1; j <= r; j++)
	//			{
	//				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
	//				sp2 += step;
	//				mSum = _mm256_sub_ps(mSum, mTmp);
	//				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
	//				dp += step;
	//			}
	//			for (int j = r + 1; j < row - r - 1; j++)
	//			{
	//				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
	//				sp2 += step;
	//				//_mm_prefetch((char *)sp2, _MM_HINT_NTA);
	//				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
	//				sp1 += step;
	//				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
	//				dp += step;
	//			}
	//			mTmp = _mm256_load_ps(sp2);
	//			for (int j = row - r - 1; j < row; j++)
	//			{
	//				mSum = _mm256_add_ps(mSum, mTmp);
	//				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
	//				sp1 += step;
	//				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
	//				dp += step;
	//			}
	//		}
	//	}
}

void ColumnSumFilter_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i += 8)
	{
		float* sp1 = src.ptr<float>(0) + i;
		float* sp2 = src.ptr<float>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		__m256 mTmp = _mm256_setzero_ps();
		__m256 mSum = _mm256_setzero_ps();
		mSum = _mm256_mul_ps(_mm256_set1_ps((float)r + 1), _mm256_load_ps(sp1));
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
		}
		_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm256_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, mTmp);
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
			sp2 += step;
			//_mm_prefetch((char *)sp2, _MM_HINT_NTA);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm256_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm256_add_ps(mSum, mTmp);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}





void boxFilter_SSAT_Channel_nonVec::filter()
{
	RowSumFilter_CN(src, temp, r, parallelType).filter();
	ColumnSumFilter_nonVec(temp, dest, r, parallelType).filter();
}

void boxFilter_SSAT_Channel_SSE::filter()
{
	RowSumFilter_SSE_CN(src, temp, r, parallelType).filter();
	ColumnSumFilter_SSE(temp, dest, r, parallelType).filter();
}

void boxFilter_SSAT_Channel_AVX::filter()
{
	RowSumFilter_AVX_CN(src, temp, r, parallelType).filter();
	ColumnSumFilter_AVX(temp, dest, r, parallelType).filter();
}



void RowSumFilter_CN::filter_naive_impl()
{
	for (int k = 0; k < cn; k++)
	{
		for (int j = 0; j < row; j++)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			float sum = 0.f;

			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}
}

void RowSumFilter_CN::filter_omp_impl()
{
#pragma omp parallel for
	//#pragma omp parallel for num_threads(numOfThreads)
	for (int k = 0; k < cn; k++)
	{
		for (int j = 0; j < row; j++)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			float sum = 0.f;

			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}
}

void RowSumFilter_CN::operator()(const cv::Range& range) const
{

}



void RowSumFilter_SSE_CN::filter_naive_impl()
{
	for (int k = 0; k < cn; k += 4)
	{
		for (int j = 0; j < row; j++)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m128 mSum = _mm_setzero_ps();

			__m128 mPrev = _mm_load_ps(sp1);
			__m128 mNext;
			mSum = _mm_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_SSE_CN::filter_omp_impl()
{
#pragma omp parallel for
	for (int k = 0; k < cn; k += 4)
	{
		for (int j = 0; j < row; j++)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m128 mSum = _mm_setzero_ps();

			__m128 mPrev = _mm_load_ps(sp1);
			__m128 mNext;
			mSum = _mm_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mNext = _mm_load_ps(sp2);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm_load_ps(sp1);
				mSum = _mm_sub_ps(mSum, mPrev);
				mSum = _mm_add_ps(mSum, mNext);
				sp1 += cn;

				_mm_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_SSE_CN::operator()(const cv::Range& range) const
{

}



void RowSumFilter_AVX_CN::filter_naive_impl()
{
	for (int k = 0; k < cn; k += 8)
	{
		for (int j = 0; j < row; j++)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m256 mSum = _mm256_setzero_ps();

			__m256 mPrev = _mm256_load_ps(sp1);
			__m256 mNext;
			mSum = _mm256_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm256_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm256_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_AVX_CN::filter_omp_impl()
{
#pragma omp parallel for
	for (int k = 0; k < cn; k += 8)
	{
		for (int j = 0; j < row; j++)
		{
			float *sp1 = src.ptr<float>(j) + k;
			float *sp2 = src.ptr<float>(j) + k + cn;
			float *dp = dest.ptr<float>(j) + k;

			__m256 mSum = _mm256_setzero_ps();

			__m256 mPrev = _mm256_load_ps(sp1);
			__m256 mNext;
			mSum = _mm256_mul_ps(mPrev, mBorder);
			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;
			}
			_mm256_stream_ps(dp, mSum);
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mNext = _mm256_load_ps(sp2);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;
				sp2 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
			mNext = _mm256_load_ps(sp2);
			for (int i = col - r - 1; i < col; i++)
			{
				mPrev = _mm256_load_ps(sp1);
				mSum = _mm256_sub_ps(mSum, mPrev);
				mSum = _mm256_add_ps(mSum, mNext);
				sp1 += cn;

				_mm256_stream_ps(dp, mSum);
				dp += cn;
			}
		}
	}
}

void RowSumFilter_AVX_CN::operator()(const cv::Range& range) const
{

}





/*
 * uchar implement
 */
void boxFilter_SSAT_8u_nonVec::filter()
{
	RowSumFilter_8u(src, temp, r, parallelType).filter();
	ColumnSumFilter_8u_nonVec(temp, dest, r, parallelType).filter();
}

void RowSumFilter_8u::filter_naive_impl()
{
	for (int j = 0; j < row; j++)
	{
		uint8_t* sp1;
		uint8_t* sp2;
		uint32_t* dp;

		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<uint8_t>(j) + k;
			sp2 = src.ptr<uint8_t>(j) + k + cn;
			dp = dest.ptr<uint32_t>(j) + k;

			uint32_t sum = 0;

			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}
}

void RowSumFilter_8u::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < row; j++)
	{
		uint8_t* sp1;
		uint8_t* sp2;
		uint32_t* dp;

		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<uint8_t>(j) + k;
			sp2 = src.ptr<uint8_t>(j) + k + cn;
			dp = dest.ptr<uint32_t>(j) + k;

			uint32_t sum = 0;

			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}
}

void RowSumFilter_8u::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_8u_nonVec::filter_naive_impl()
{
	for (int i = 0; i < step; i++)
	{
		uint32_t* sp1 = src.ptr<uint32_t>(0) + i;
		uint32_t* sp2 = src.ptr<uint32_t>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		uint32_t sum = 0;
		sum += *sp1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = (float)sum*div;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = (float)sum*div;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = (float)sum*div;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = (float)sum*div;
			dp += step;
		}
	}
}

void ColumnSumFilter_8u_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < step; i++)
	{
		uint32_t* sp1 = src.ptr<uint32_t>(0) + i;
		uint32_t* sp2 = src.ptr<uint32_t>(1) + i;
		float* dp = dest.ptr<float>(0) + i;

		uint32_t sum = 0;
		sum += *sp1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = (float)sum*div;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = (float)sum*div;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = (float)sum*div;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = (float)sum*div;
			dp += step;
		}
	}
}

void ColumnSumFilter_8u_nonVec::operator()(const cv::Range& range) const
{

}
