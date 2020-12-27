#include "boxFilter_SSAT_RowSumGather.h"
#include "boxFilter_SSAT_HV.h"
#include "boxFilter_SSAT_VH.h"
#include <inlineSIMDFunctions.hpp>

using namespace std;
using namespace cv;

boxFilter_SSAT_HV_RowSumGather_AVX::boxFilter_SSAT_HV_RowSumGather_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(src.size(), src.type());
}

void boxFilter_SSAT_HV_RowSumGather_AVX::filter()
{
	RowSumFilter_HV_AVXgather(src, temp, r, parallelType).filter();
	ColumnSumFilter_AVX(temp, dest, r, parallelType).filter();
}



void RowSumFilter_HV_AVXgather::filter_naive_impl()
{
	const int step = cn * col;
	const __m256i vindex = _mm256_set_epi32(step * 7, step * 6, step * 5, step * 4, step * 3, step * 2, step, 0);

	for (int i = 0; i < row; i += 8)
	{
		__m256 mTmp;

		float* sp_prev = src.ptr<float>(i);
		float* sp_next = src.ptr<float>(i) + cn;
		float* dp = dest.ptr<float>(i);

		__m256 mRef_prev, mRef_next;
		__m256 mSum = _mm256_setzero_ps();

#ifndef _USE_GATHER_
		mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
			*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
		mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
		mSum = _mm256_mul_ps(mRef_prev, mBorder);
		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			mSum = _mm256_add_ps(mSum, mRef_next);
			sp_next += cn;
		}

		_mm256_i32scaterscalar_ps(dp, vindex, mSum);
		dp++;

		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			_mm256_i32scaterscalar_ps(dp, vindex, mSum);
			dp++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_prev += cn;
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			_mm256_i32scaterscalar_ps(dp, vindex, mSum);
			dp++;
		}
		for (int j = col - r; j < col; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
			sp_prev += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			_mm256_i32scaterscalar_ps(dp, vindex, mSum);
			dp++;
		}
	}
}

void RowSumFilter_HV_AVXgather::filter_omp_impl()
{
	const int step = cn * col;
	__m256i vindex = _mm256_set_epi32(step * 7, step * 6, step * 5, step * 4, step * 3, step * 2, step, 0);

#pragma omp parallel for
	for (int i = 0; i < row; i += 8)
	{
		__m256 mTmp;

		float* sp_prev = src.ptr<float>(i);
		float* sp_next = src.ptr<float>(i) + cn;
		float* dp = dest.ptr<float>(i);

		__m256 mRef_prev, mRef_next;
		__m256 mSum = _mm256_setzero_ps();

#ifndef _USE_GATHER_
		mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
			*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
		mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
		mSum = _mm256_mul_ps(mRef_prev, mBorder);
		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			mSum = _mm256_add_ps(mSum, mRef_next);
			sp_next += cn;
		}

		_mm256_i32scaterscalar_ps(dp, vindex, mSum);
		dp++;

		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			_mm256_i32scaterscalar_ps(dp, vindex, mSum);
			dp++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_prev += cn;
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			_mm256_i32scaterscalar_ps(dp, vindex, mSum);
			dp++;
		}
		for (int j = col - r; j < col; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
			sp_prev += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			_mm256_i32scaterscalar_ps(dp, vindex, mSum);
			dp++;
		}
	}
}

void RowSumFilter_HV_AVXgather::operator()(const cv::Range& range) const
{

}




boxFilter_SSAT_VH_RowSumGather_SSE::boxFilter_SSAT_VH_RowSumGather_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(src.size(), src.type());
}

void boxFilter_SSAT_VH_RowSumGather_SSE::filter()
{
	ColumnSumFilter_VH_SSE(src, temp, r, parallelType).filter();
	RowSumFilter_VH_SSEgather(temp, dest, r, parallelType).filter();
}



void RowSumFilter_VH_SSEgather::filter_naive_impl()
{
	const int step = cn * col;
	__m128i vindex = _mm_set_epi32(step * 3, step * 2, step, 0);

	for (int i = 0; i < row; i += 4)
	{
		__m128 mTmp;

		float* sp_prev = src.ptr<float>(i);
		float* sp_next = src.ptr<float>(i) + cn;
		float* dp = dest.ptr<float>(i);

		__m128 mRef_prev, mRef_next;
		__m128 mSum = _mm_setzero_ps();

#ifndef _USE_GATHER_
		mRef_prev = _mm_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3));
#else
		mRef_prev = _mm_i32gather_ps(sp_prev, vindex, 4);
#endif
		mSum = _mm_mul_ps(mRef_prev, mBorder);
		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3));
#else
			mRef_next = _mm_i32gather_ps(sp_next, vindex, 4);
#endif
			mSum = _mm_add_ps(mSum, mRef_next);
			sp_next += cn;
		}
		mTmp = _mm_mul_ps(mSum, mDiv);
		_mm_i32scaterscalar_ps(dp++, vindex, mTmp);

		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3));
#else
			mRef_next = _mm_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_next += cn;

			mSum = _mm_add_ps(mSum, mRef_next);
			mSum = _mm_sub_ps(mSum, mRef_prev);

			mTmp = _mm_mul_ps(mSum, mDiv);
			_mm_i32scaterscalar_ps(dp++, vindex, mTmp);

		}
		for (int j = r + 1; j < col - r; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3));
			mRef_next = _mm_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3));
#else
			mRef_prev = _mm_i32gather_ps(sp_prev, vindex, 4);
			mRef_next = _mm_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_prev += cn;
			sp_next += cn;

			mSum = _mm_add_ps(mSum, mRef_next);
			mSum = _mm_sub_ps(mSum, mRef_prev);

			mTmp = _mm_mul_ps(mSum, mDiv);
			_mm_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
		for (int j = col - r; j < col; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3));
#else
			mRef_prev = _mm_i32gather_ps(sp_prev, vindex, 4);
#endif
			sp_prev += cn;

			mSum = _mm_add_ps(mSum, mRef_next);
			mSum = _mm_sub_ps(mSum, mRef_prev);

			mTmp = _mm_mul_ps(mSum, mDiv);
			_mm_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
	}
}

void RowSumFilter_VH_SSEgather::filter_omp_impl()
{
	const int step = cn * col;
	__m128i vindex = _mm_set_epi32(step * 3, step * 2, step, 0);

#pragma omp parallel for
	for (int i = 0; i < row; i += 4)
	{
		__m128 mTmp;

		float* sp_prev = src.ptr<float>(i);
		float* sp_next = src.ptr<float>(i) + cn;
		float* dp = dest.ptr<float>(i);

		__m128 mRef_prev, mRef_next;
		__m128 mSum = _mm_setzero_ps();

#ifndef _USE_GATHER_
		mRef_prev = _mm_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3));
#else
		mRef_prev = _mm_i32gather_ps(sp_prev, vindex, 4);
#endif
		mSum = _mm_mul_ps(mRef_prev, mBorder);
		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3));
#else
			mRef_next = _mm_i32gather_ps(sp_next, vindex, 4);
#endif
			mSum = _mm_add_ps(mSum, mRef_next);
			sp_next += cn;
		}
		mTmp = _mm_mul_ps(mSum, mDiv);
		_mm_i32scaterscalar_ps(dp++, vindex, mTmp);

		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3));
#else
			mRef_next = _mm_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_next += cn;

			mSum = _mm_add_ps(mSum, mRef_next);
			mSum = _mm_sub_ps(mSum, mRef_prev);

			mTmp = _mm_mul_ps(mSum, mDiv);
			_mm_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
		for (int j = r + 1; j < col - r; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3));
			mRef_next = _mm_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3));
#else
			mRef_prev = _mm_i32gather_ps(sp_prev, vindex, 4);
			mRef_next = _mm_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_prev += cn;
			sp_next += cn;

			mSum = _mm_add_ps(mSum, mRef_next);
			mSum = _mm_sub_ps(mSum, mRef_prev);

			mTmp = _mm_mul_ps(mSum, mDiv);
			_mm_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
		for (int j = col - r; j < col; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3));
#else
			mRef_prev = _mm_i32gather_ps(sp_prev, vindex, 4);
#endif
			sp_prev += cn;

			mSum = _mm_add_ps(mSum, mRef_next);
			mSum = _mm_sub_ps(mSum, mRef_prev);

			mTmp = _mm_mul_ps(mSum, mDiv);
			_mm_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
	}
}

void RowSumFilter_VH_SSEgather::operator()(const cv::Range& range) const
{

}



boxFilter_SSAT_VH_RowSumGather_AVX::boxFilter_SSAT_VH_RowSumGather_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(src.size(), src.type());
}

void boxFilter_SSAT_VH_RowSumGather_AVX::filter()
{
	ColumnSumFilter_VH_AVX(src, temp, r, parallelType).filter();
	RowSumFilter_VH_AVXgather(temp, dest, r, parallelType).filter();
}



void RowSumFilter_VH_AVXgather::filter_naive_impl()
{
	const int step = cn * col;
	__m256i vindex = _mm256_set_epi32(step * 7, step * 6, step * 5, step * 4, step * 3, step * 2, step, 0);

	for (int i = 0; i < row; i += 8)
	{
		__m256 mTmp;

		float* sp_prev = src.ptr<float>(i);
		float* sp_next = src.ptr<float>(i) + cn;
		float* dp = dest.ptr<float>(i);

		__m256 mRef_prev, mRef_next;
		__m256 mSum = _mm256_setzero_ps();

#ifndef _USE_GATHER_
		mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
			*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
		mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
		mSum = _mm256_mul_ps(mRef_prev, mBorder);
		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			mSum = _mm256_add_ps(mSum, mRef_next);
			sp_next += cn;
		}
		mTmp = _mm256_mul_ps(mSum, mDiv);
		_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);

		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			mTmp = _mm256_mul_ps(mSum, mDiv);
			_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
		for (int j = r + 1; j < col - r; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_prev += cn;
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			mTmp = _mm256_mul_ps(mSum, mDiv);
			_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
		for (int j = col - r; j < col; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
			sp_prev += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			mTmp = _mm256_mul_ps(mSum, mDiv);
			_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
	}
}

void RowSumFilter_VH_AVXgather::filter_omp_impl()
{
	const int step = cn * col;
	static const __m256i vindex = _mm256_set_epi32(step * 7, step * 6, step * 5, step * 4, step * 3, step * 2, step, 0);

#pragma omp parallel for
	for (int i = 0; i < row; i += 8)
		//for (int I = 0; I < row / 8; I++)
	{
		//int i = 8 * I;
		__m256 mTmp;

		float* sp_prev = src.ptr<float>(i);
		float* sp_next = src.ptr<float>(i) + cn;
		float* dp = dest.ptr<float>(i);

		__m256 mRef_prev, mRef_next;
		__m256 mSum = _mm256_setzero_ps();

#ifndef _USE_GATHER_
		mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
			*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
		mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
		mSum = _mm256_mul_ps(mRef_prev, mBorder);
		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			mSum = _mm256_add_ps(mSum, mRef_next);
			sp_next += cn;
		}
		mTmp = _mm256_mul_ps(mSum, mDiv);
		_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);

		for (int j = 1; j <= r; j++)
		{
#ifndef _USE_GATHER_
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			mTmp = _mm256_mul_ps(mSum, mDiv);
			_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
		for (int j = r + 1; j < col - r; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
			mRef_next = _mm256_setr_ps(*sp_next, *(sp_next + step), *(sp_next + step * 2), *(sp_next + step * 3),
				*(sp_next + step * 4), *(sp_next + step * 5), *(sp_next + step * 6), *(sp_next + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
			mRef_next = _mm256_i32gather_ps(sp_next, vindex, 4);
#endif
			sp_prev += cn;
			sp_next += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			mTmp = _mm256_mul_ps(mSum, mDiv);
			_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
		for (int j = col - r; j < col; j++)
		{
#ifndef _USE_GATHER_
			mRef_prev = _mm256_setr_ps(*sp_prev, *(sp_prev + step), *(sp_prev + step * 2), *(sp_prev + step * 3),
				*(sp_prev + step * 4), *(sp_prev + step * 5), *(sp_prev + step * 6), *(sp_prev + step * 7));
#else
			mRef_prev = _mm256_i32gather_ps(sp_prev, vindex, 4);
#endif
			sp_prev += cn;

			mSum = _mm256_add_ps(mSum, mRef_next);
			mSum = _mm256_sub_ps(mSum, mRef_prev);

			mTmp = _mm256_mul_ps(mSum, mDiv);
			_mm256_i32scaterscalar_ps(dp++, vindex, mTmp);
		}
	}
}

void RowSumFilter_VH_AVXgather::operator()(const cv::Range& range) const
{

}
