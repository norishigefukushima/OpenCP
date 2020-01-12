#include "boxFilter_SSAT_HV_CacheBlocking.h"

using namespace std;
using namespace cv;

boxFilter_SSAT_HV_CacheBlock_nonVec::boxFilter_SSAT_HV_CacheBlock_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	row = src.rows;
	col = src.cols;
	cn = src.channels();
	div = 1.f / ((2 * r + 1)*(2 * r + 1));
	step = col * cn;

	divRow = row / OMP_THREADS_MAX;
}

void boxFilter_SSAT_HV_CacheBlock_nonVec::upper_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);
	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x++)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		float sum = 0.f;
		sum += *sp2 * (r + 1);
		sp2 += step;
		for (int y = 1; y <= r; y++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = sum*div;
		dp += step;

		for (int y = 1; y <= r; y++)
		{
			sum += *sp2 - *sp1;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int y = r + 1; y < divRow; y++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
	}
}

void boxFilter_SSAT_HV_CacheBlock_nonVec::middle_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);
	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x++)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		float sum = 0.f;
		for (int y = -r; y <= r; y++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = sum*div;
		dp += step;

		for (int y = y_dest + 1; y < y_dest + divRow; y++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
	}
}

void boxFilter_SSAT_HV_CacheBlock_nonVec::lower_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);
	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x++)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		float sum = 0.f;
		for (int y = -r; y <= r; y++)
		{
			sum += *sp2;
			sp2 += step;
		}
		*dp = sum*div;
		dp += step;

		for (int y = y_dest + 1; y < row - r - 1; y++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			sp2 += step;
			*dp = sum*div;
			dp += step;
		}
		for (int y = row - r - 1; y < row; y++)
		{
			sum += *sp2 - *sp1;
			sp1 += step;
			*dp = sum*div;
			dp += step;
		}
	}
}

void boxFilter_SSAT_HV_CacheBlock_nonVec::filter()
{
	if (parallelType == NAIVE)
	{
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int idxDiv = 0; idxDiv < OMP_THREADS_MAX; idxDiv++)
		{
			if (idxDiv == 0)
				upper_impl(idxDiv);
			else if (idxDiv == OMP_THREADS_MAX - 1)
				lower_impl(idxDiv);
			else
				middle_impl(idxDiv);
		}
	}
}



boxFilter_SSAT_HV_CacheBlock_SSE::boxFilter_SSAT_HV_CacheBlock_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_SSAT_HV_CacheBlock_nonVec(_src, _dest, _r, _parallelType)
{
	mBorder = _mm_set1_ps(static_cast<float>(r + 1));
	mDiv = _mm_set1_ps(div);
}

void boxFilter_SSAT_HV_CacheBlock_SSE::upper_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);

	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x += 4)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		__m128 mTmp = _mm_setzero_ps();
		__m128 mSum = _mm_setzero_ps();
		mSum = _mm_mul_ps(_mm_set1_ps((float)r + 1), _mm_load_ps(sp2));
		sp2 += step;
		for (int y = 1; y <= r; y++)
		{
			mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			sp2 += step;
		}
		_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm_load_ps(sp1);
		for (int y = 1; y <= r; y++)
		{
			mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, mTmp);
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int y = r + 1; y < divRow; y++)
		{
			mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}

void boxFilter_SSAT_HV_CacheBlock_SSE::middle_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);

	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x += 4)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		__m128 mTmp = _mm_setzero_ps();
		__m128 mSum = _mm_setzero_ps();
		for (int y = -r; y <= r; y++)
		{
			mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			sp2 += step;
		}
		_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
		dp += step;

		for (int y = y_dest + 1; y < y_dest + divRow; y++)
		{
			mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}

void boxFilter_SSAT_HV_CacheBlock_SSE::lower_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);

	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x += 4)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		__m128 mTmp = _mm_setzero_ps();
		__m128 mSum = _mm_setzero_ps();
		for (int y = -r; y <= r; y++)
		{
			mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			sp2 += step;
		}
		_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
		dp += step;

		for (int y = y_dest + 1; y < row - r - 1; y++)
		{
			mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			sp2 += step;
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm_load_ps(sp2);
		for (int y = row - r - 1; y < row; y++)
		{
			mSum = _mm_add_ps(mSum, mTmp);
			mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
			sp1 += step;
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}



boxFilter_SSAT_HV_CacheBlock_AVX::boxFilter_SSAT_HV_CacheBlock_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_SSAT_HV_CacheBlock_nonVec(_src, _dest, _r, _parallelType)
{
	mBorder = _mm256_set1_ps(static_cast<float>(r + 1));
	mDiv = _mm256_set1_ps(div);
}

void boxFilter_SSAT_HV_CacheBlock_AVX::upper_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);

	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x += 8)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		__m256 mTmp = _mm256_setzero_ps();
		__m256 mSum = _mm256_setzero_ps();
		mSum = _mm256_mul_ps(_mm256_set1_ps((float)r + 1), _mm256_load_ps(sp2));
		sp2 += step;
		for (int y = 1; y <= r; y++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
		}
		_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
		dp += step;

		mTmp = _mm256_load_ps(sp1);
		for (int y = 1; y <= r; y++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, mTmp);
			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		for (int y = r + 1; y < divRow; y++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
			sp1 += step;
			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}

void boxFilter_SSAT_HV_CacheBlock_AVX::middle_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);
	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x += 8)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		__m256 mTmp = _mm256_setzero_ps();
		__m256 mSum = _mm256_setzero_ps();
		for (int y = -r; y <= r; y++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
		}
		_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
		dp += step;

		for (int y = y_dest + 1; y < y_dest + divRow; y++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
			sp1 += step;
			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}

void boxFilter_SSAT_HV_CacheBlock_AVX::lower_impl(const int idxDiv)
{
	Mat temp(Size(col, divRow + r), src.type());

	const int start = max(0, divRow * idxDiv - r);
	const int end = min(row, divRow * (idxDiv + 1) + r);
	for (int y = start; y < end; y++)
	{
		for (int k = 0; k < cn; k++)
		{
			float* sp1 = src.ptr<float>(y) + k;
			float* sp2 = src.ptr<float>(y) + k + cn;
			float* dp = temp.ptr<float>(y - start) + k;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int x = 1; x <= r; x++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = r + 1; x < col - r - 1; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += cn;
			}
			for (int x = col - r - 1; x < col; x++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += cn;
			}
		}
	}

	const int y_dest = divRow * idxDiv;
	for (int x = 0; x < step; x += 8)
	{
		float* sp1 = temp.ptr<float>(0) + x;
		float* sp2 = sp1;
		float* dp = dest.ptr<float>(y_dest) + x;

		__m256 mTmp = _mm256_setzero_ps();
		__m256 mSum = _mm256_setzero_ps();
		for (int y = -r; y <= r; y++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
		}
		_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
		dp += step;

		for (int y = y_dest + 1; y < row - r - 1; y++)
		{
			mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			sp2 += step;
			mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
			sp1 += step;
			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
		mTmp = _mm256_load_ps(sp2);
		for (int y = row - r - 1; y < row; y++)
		{
			mSum = _mm256_add_ps(mSum, mTmp);
			mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
			sp1 += step;
			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;
		}
	}
}


//#include "boxFilter_Summed_cachebrocking.h"
//#include <opencp.hpp>
//#include <omp.h>
//
//using namespace cv;
//using namespace std;
//using namespace cp;
//
//void RowSumFilter_cache(Mat& src, Mat& dest, int r)
//{
//	const int col = src.cols;
//	const int row = src.rows;
//	const int cn = src.channels();
//
//#pragma omp parallel for
//	for (int j = 0; j < row; j++)
//	{
//		float* sp1;
//		float* sp2;
//		float* dp;
//		for (int k = 0; k < cn; k++)
//		{
//			sp1 = src.ptr<float>(j)+k;
//			sp2 = src.ptr<float>(j)+k + cn;
//			dp = dest.ptr<float>(j+r)+k;
//
//			float sum = 0.f;
//
//			sum += *sp1 * (r + 1);
//			for (int i = 1; i <= r; i++)
//			{
//				sum += *sp2;
//				sp2 += cn;
//			}
//			*dp = sum;
//			dp += cn;
//
//			for (int i = 1; i <= r; i++)
//			{
//				sum += *sp2 - *sp1;
//				sp2 += cn;
//
//				*dp = sum;
//				dp += cn;
//			}
//			for (int i = r + 1; i < col - r - 1; i++)
//			{
//				sum += *sp2 - *sp1;
//				sp1 += cn;
//				sp2 += cn;
//
//				*dp = sum;
//				dp += cn;
//			}
//			for (int i = col - r - 1; i < col; i++)
//			{
//				sum += *sp2 - *sp1;
//				sp1 += cn;
//
//				*dp = sum;
//				dp += cn;
//			}
//		}
//	}
//#pragma omp parallel for
//	for (int j = 0; j < r; j++)
//	{
//		float* sp = dest.ptr<float>(r);
//		float* dp = dest.ptr<float>(j);
//		for (int i = 0; i < col*cn; i++)
//		{
//			*dp = *sp;
//			sp++;
//			dp++;
//		}
//	}
//#pragma omp parallel for
//	for (int j = 0; j < r; j++)
//	{
//		float* sp = dest.ptr<float>(row + r - 1);
//		float* dp = dest.ptr<float>(row + r + j);
//		for (int i = 0; i < col*cn; i++)
//		{
//			*dp = *sp;
//			sp++;
//			dp++;
//		}
//	}
//}
//
//void ColumnSumFilter_cache(Mat& src, Mat& dest, int r)
//{
//
//}
//
//#define COL_SUM_CACHE_AVX(k)									\
//	for(int i = 0; i < SIMDsize; i++)							\
//	{															\
//		float* sp1 = src.ptr<float>(index[k]) + i * 8;			\
//		float* sp2 = src.ptr<float>(index[k]) + i * 8;			\
//		float* dp = dest.ptr<float>(index[k]) + i * 8;			\
//																\
//		__m256 mSum = _mm256_setzero_ps();						\
//																\
//		for (int j = -r; j <= r; j++)							\
//		{														\
//			mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));	\
//			sp2 += step;										\
//		}														\
//		_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));		\
//		dp += step;												\
//																\
//		for (int j = index[k] + 1; j < index[k + 1]; j++)		\
//		{														\
//			mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));	\
//			mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));	\
//			sp1 += step;										\
//			sp2 += step;										\
//			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));	\
//			dp += step;											\
//		}														\
//	}
//
//
//void ColumnSumFilter_cache_AVX(Mat& src, Mat& dest, int r, int division)
//{
//	const int ksize = 2 * r + 1;
//	const int col = src.cols;
//	const int row = src.rows;
//	const int cn = src.channels();
//
//	const int step = col*cn;
//
//	//const int division = 8; //“ü—Í‰æ‘œ‚Ìc‚Ì‰æ‘f”‚Ì–ñ”‚È‚çOK
//
//	const int SIMDsize = col*cn / 8;
//	const int nn = SIMDsize * 8;
//	const int rowsize = (row - r - r) / division;
//	const int colsize = SIMDsize / division;
//
//	const float div = 1.f / (ksize*ksize);
//	const __m256 mDiv = _mm256_set1_ps(div);
//
//	// divide horizon
//	//int* index = new int[division];
//	//for (int i = 0; i < division; i++)
//	//{
//	//	index[i] = i*rowsize;
//	//}
//	
//	// divide var
//	//int* index = new int[division];
//	//for (int i = 0; i < division; i++)
//	//{
//	//	index[i] = i*colsize;
//	//}
//
//	// divide horizon
//	int* index = new int[division + 1];
//	for (int i = 0; i < division; i++)
//	{
//		index[i] = i*rowsize;
//	}
//	index[division] = row - r - r;
//
//	// divide var
//	//int* index = new int[division + 1];
//	//for (int i = 0; i < division; i++)
//	//{
//	//	index[i] = i*colsize;
//	//}
//	//index[division] = SIMDsize;
//
//	//‰¡•ªŠ„”CˆÓ
//#pragma omp parallel for
//	for (int k = 0; k < division; k++)
//	{
//		for (int i = 0; i < SIMDsize; i++)
//		{
//			float* sp1 = src.ptr<float>(index[k]) + i * 8;
//			float* sp2 = src.ptr<float>(index[k]) + i * 8;
//			float* dp = dest.ptr<float>(index[k]) + i * 8;
//
//			__m256 mSum = _mm256_setzero_ps();
//
//			for (int j = -r; j <= r; j++)
//			{
//				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
//				sp2 += step;
//			}
//			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
//			dp += step;
//
//			for (int j = index[k] + 1; j < index[k + 1]; j++)
//			{
//				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
//				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
//				sp1 += step;
//				sp2 += step;
//				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
//				dp += step;
//			}
//		}
//		//Mat show; dest.convertTo(show, CV_8U); resize(show, show, Size(), 0.5, 0.5); imshow("test", show); waitKey(0);
//	}
//
//	//c•ªŠ„”CˆÓ
////#pragma omp parallel for
////	for (int k = 0; k < division; k++)
////	{
////		for (int i = index[k]; i < index[k + 1]; i++)
////		{
////			float* sp1 = src.ptr<float>(0) + i * 8;
////			float* sp2 = src.ptr<float>(0) + i * 8;
////			float* dp = dest.ptr<float>(0) + i * 8;
////
////			__m256 mSum = _mm256_setzero_ps();
////
////			for (int j = -r; j <= r; j++)
////			{
////				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
////				sp2 += step;
////			}
////			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
////			dp += step;
////
////			for (int j = 1; j < dest.rows; j++)
////			{
////				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
////				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
////				sp1 += step;
////				sp2 += step;
////				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
////				dp += step;
////			}
////		}
////		//Mat show; dest.convertTo(show, CV_8U); resize(show, show, Size(), 0.5, 0.5); imshow("test", show); waitKey(0);
////	}
//
////‰¡•ªŠ„–ñ”
////#pragma omp parallel for
////	for (int k = 0; k < division; k++)
////	{
////		for (int i = 0; i < SIMDsize; i++)
////		{
////			float* sp1 = src.ptr<float>(index[k]) + i * 8;
////			float* sp2 = src.ptr<float>(index[k]) + i * 8;
////			float* dp = dest.ptr<float>(index[k]) + i * 8;
////
////			__m256 mSum = _mm256_setzero_ps();
////
////			for (int j = -r; j <= r; j++)
////			{
////				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
////				sp2 += step;
////			}
////			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
////			dp += step;
////
////			for (int j = 1; j < rowsize; j++)
////			{
////				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
////				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
////				sp1 += step;
////				sp2 += step;
////				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
////				dp += step;
////			}
////		}
////		//Mat show; dest.convertTo(show, CV_8U); resize(show, show, Size(), 0.5, 0.5); imshow("test", show); waitKey(0);
////	}
//
////c•ªŠ„–ñ”
////#pragma omp parallel for
////	for (int k = 0; k < division; k++)
////	{
////		for (int i = 0; i < colsize; i++)
////		{
////			float* sp1 = src.ptr<float>(0) + (index[k]+i) * 8;
////			float* sp2 = src.ptr<float>(0) + (index[k]+i) * 8;
////			float* dp = dest.ptr<float>(0) + (index[k]+i) * 8;
////
////			__m256 mSum = _mm256_setzero_ps();
////
////			for (int j = -r; j <= r; j++)
////			{
////				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
////				sp2 += step;
////			}
////			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
////			dp += step;
////
////			for (int j = 1; j < dest.rows; j++)
////			{
////				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp2));
////				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp1));
////				sp1 += step;
////				sp2 += step;
////				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
////				dp += step;
////			}
////		}
////		//Mat show; dest.convertTo(show, CV_8U); resize(show, show, Size(), 0.5, 0.5); imshow("test", show); waitKey(0);
////	}
//
//	delete[] index;
//}
//
//void ColumnSumFilter_cache_AVX_th1(Mat& src, Mat& dest, int r, int division)
//{
//	const int ksize = 2 * r + 1;
//	const int col = src.cols;
//	const int row = src.rows;
//	const int cn = src.channels();
//
//	const int step = col*cn;
//
//	const int SIMDsize = col*cn / 8;
//	const int rowsize = (row - r - r) / division;
//
//	const float div = 1.f / (ksize*ksize);
//	const __m256 mDiv = _mm256_set1_ps(div);
//
//	// divide horizon
//	int* index = new int[division + 1];
//	for (int i = 0; i < division; i++)
//	{
//		index[i] = i*rowsize;
//	}
//	index[division] = row - r - r;
//
//	//‰¡•ªŠ„”CˆÓ
////#pragma omp sections
//	{
////#pragma omp section
//		{	COL_SUM_CACHE_AVX(0); }
//	}
//
//	delete[] index;
//}
//
//void ColumnSumFilter_cache_AVX_th2(Mat& src, Mat& dest, int r, int division)
//{
//	const int ksize = 2 * r + 1;
//	const int col = src.cols;
//	const int row = src.rows;
//	const int cn = src.channels();
//
//	const int step = col*cn;
//
//	const int SIMDsize = col*cn / 8;
//	const int rowsize = (row - r - r) / division;
//
//	const float div = 1.f / (ksize*ksize);
//	const __m256 mDiv = _mm256_set1_ps(div);
//
//	// divide horizon
//	int* index = new int[division + 1];
//	for (int i = 0; i < division; i++)
//	{
//		index[i] = i*rowsize;
//	}
//	index[division] = row - r - r;
//
//	//‰¡•ªŠ„”CˆÓ
//#pragma omp sections
//	{
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(0); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(1); }
//	}
//
//	delete[] index;
//}
//
//void ColumnSumFilter_cache_AVX_th8(Mat& src, Mat& dest, int r, int division)
//{
//	const int ksize = 2 * r + 1;
//	const int col = src.cols;
//	const int row = src.rows;
//	const int cn = src.channels();
//
//	const int step = col*cn;
//
//	const int SIMDsize = col*cn / 8;
//	const int rowsize = (row - r - r) / division;
//
//	const float div = 1.f / (ksize*ksize);
//	const __m256 mDiv = _mm256_set1_ps(div);
//
//	// divide horizon
//	int* index = new int[division + 1];
//	for (int i = 0; i < division; i++)
//	{
//		index[i] = i*rowsize;
//	}
//	index[division] = row - r - r;
//
//	//‰¡•ªŠ„”CˆÓ
//#pragma omp sections
//	{
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(0); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(1); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(2); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(3); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(4); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(5); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(6); }
//#pragma omp section
//		{	COL_SUM_CACHE_AVX(7); }
//	}
//
//	delete[] index;
//}
//
//void boxFilter_Summed_cache::filter(Mat& src, Mat& dest, int r)
//{
//
//}
//
//void boxFilter_Summed_cache::filter_SSE(Mat& src, Mat& dest, int r)
//{
//
//}
//
//void boxFilter_Summed_cache::filter_AVX(Mat& src, Mat& dest, int r, int division)
//{
//	{
//		//Timer t("row_for");
//		RowSumFilter_cache(src, temp, r);
//	}
//	{
//		//Timer t("col_for");
//		ColumnSumFilter_cache_AVX(temp, dest, r, division);
//		//if (division == 1)ColumnSumFilter_cache_AVX_th1(temp, dest, r, division);
//		//else if (division == 2)ColumnSumFilter_cache_AVX_th2(temp, dest, r, division);
//		//else if (division == 8)ColumnSumFilter_cache_AVX_th8(temp, dest, r, division);
//	}
//}