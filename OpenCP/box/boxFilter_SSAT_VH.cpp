#include "boxFilter_SSAT_VH.h"

using namespace cv;
using namespace std;

boxFilter_SSAT_VH_nonVec::boxFilter_SSAT_VH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(src.size(), src.type());
}

void boxFilter_SSAT_VH_nonVec::filter()
{
	ColumnSumFilter_VH_nonVec(src, temp, r, parallelType).filter();
	RowSumFilter_VH(temp, dest, r, parallelType).filter();
}

void boxFilter_SSAT_VH_SSE::filter()
{
	ColumnSumFilter_VH_SSE(src, temp, r, parallelType).filter();
	RowSumFilter_VH(temp, dest, r, parallelType).filter();
}

void boxFilter_SSAT_VH_AVX::filter()
{
	ColumnSumFilter_VH_AVX(src, temp, r, parallelType).filter();
	RowSumFilter_VH(temp, dest, r, parallelType).filter();
}



void RowSumFilter_VH::filter_naive_impl()
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
			*dp = sum * div;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum * div;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum * div;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum * div;
				dp += cn;
			}
		}
	}
}

void RowSumFilter_VH::filter_omp_impl()
{
#pragma omp parallel for
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
			*dp = sum * div;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum * div;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum * div;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum * div;
				dp += cn;
			}
		}
	}
}

void RowSumFilter_VH::operator()(const cv::Range& range) const
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
			*dp = sum * div;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum * div;
				dp += cn;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum * div;
				dp += cn;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum * div;
				dp += cn;
			}
		}
	}
}



void ColumnSumFilter_VH_nonVec::filter_naive_impl()
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
		*dp = sum;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = sum;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = sum;
			dp += step;
		}
	}
}

void ColumnSumFilter_VH_nonVec::filter_omp_impl()
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
		*dp = sum;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = sum;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = sum;
			dp += step;
		}
	}
}

void ColumnSumFilter_VH_nonVec::operator()(const cv::Range& range) const
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
		*dp = sum;
		dp += step;

		for (int j = 1; j <= r; j++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = sum;
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum;
			dp += step;
		}
		for (int j = row - r - 1; j < row; j++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = sum;
			dp += step;
		}
	}
}

void ColumnSumFilter_VH_nonVec::filter()
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



void ColumnSumFilter_VH_SSE::filter_naive_impl()
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
		_mm_store_ps(dp, mSum);
		dp += step;

		mTmp = _mm_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, mTmp);
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
		mTmp = _mm_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm_add_ps(mSum, mTmp);
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, mSum);
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

void ColumnSumFilter_VH_SSE::filter_omp_impl()
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
		_mm_store_ps(dp, mSum);
		dp += step;

		mTmp = _mm_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, mTmp);
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
		mTmp = _mm_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm_add_ps(mSum, mTmp);
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
	}
}

void ColumnSumFilter_VH_SSE::operator()(const cv::Range& range) const
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
		_mm_storeu_ps(dp, mSum);
		dp += step;

		mTmp = _mm_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, mTmp);
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
		mTmp = _mm_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm_add_ps(mSum, mTmp);
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_storeu_ps(dp, mSum);
			dp += step;
		}
	}
}



void ColumnSumFilter_VH_AVX::filter_naive_impl()
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
		_mm256_store_ps(dp, mSum);
		dp += step;

		mTmp = _mm256_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, mTmp);
			_mm256_storeu_ps(dp, mSum);
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
			sp2 += step;
			//_mm_prefetch((char *)sp2, _MM_HINT_NTA);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, mSum);
			dp += step;
		}
		mTmp = _mm256_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm256_add_ps(mSum, mTmp);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, mSum);
			dp += step;
		}
	}
}

void ColumnSumFilter_VH_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < step; i += 8)
	//for (int I = 0; I < 2; I++)
	//	for (int i = I * step; i < (I + 1) * step; i += 8)
	{
		//int i = I * 8;

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
		_mm256_store_ps(dp, mSum);
		dp += step;

		mTmp = _mm256_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, mTmp);
			_mm256_store_ps(dp, mSum);
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			//_mm_prefetch((char *)sp2, _MM_HINT_NTA);
			mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
			sp1 += step;
			_mm256_store_ps(dp, mSum);
			dp += step;
		}
		mTmp = _mm256_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm256_add_ps(mSum, mTmp);
			mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
			sp1 += step;
			_mm256_store_ps(dp, mSum);
			dp += step;
		}
	}
}

void ColumnSumFilter_VH_AVX::operator()(const cv::Range& range) const
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
		_mm256_storeu_ps(dp, mSum);
		dp += step;

		mTmp = _mm256_load_ps(sp1);
		for (int j = 1; j <= r; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, mTmp);
			_mm256_storeu_ps(dp, mSum);
			dp += step;
		}
		for (int j = r + 1; j < row - r - 1; j++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
			sp2 += step;
			//_mm_prefetch((char *)sp2, _MM_HINT_NTA);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, mSum);
			dp += step;
		}
		mTmp = _mm256_load_ps(sp2);
		for (int j = row - r - 1; j < row; j++)
		{
			mSum = _mm256_add_ps(mSum, mTmp);
			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
			sp1 += step;
			_mm256_storeu_ps(dp, mSum);
			dp += step;
		}
	}
}
