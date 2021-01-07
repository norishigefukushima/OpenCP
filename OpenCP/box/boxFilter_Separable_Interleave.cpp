#include "boxFilter_Separable_Interleave.h"

using namespace std;
using namespace cv;

boxFilter_Separable_VHI_nonVec::boxFilter_Separable_VHI_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: BoxFilterBase(_src, _dest, _r, _parallelType)
{
	ksize = 2 * r + 1;
	init();
}

void boxFilter_Separable_VHI_nonVec::init()
{
}

void boxFilter_Separable_VHI_nonVec::filter_naive_impl()
{
	for (int i = 0; i < row; i++)
	{
		Mat temp = Mat::zeros(Size(col, 1), CV_32FC1);

		for (int j = 0; j < col; j++)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(0) + j;

			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col : src.ptr<float>(row - 1) + j;
				sum += *sp1;
				sum += *sp2;
			}
			*tp = sum;
		}
		for (int j = 0; j < col; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = j - k >= 0 ? tp - k : temp.ptr<float>(0);
				const float* tp2 = j + k < col ? tp + k : temp.ptr<float>(0) + (col - 1);
				sum += *tp1;
				sum += *tp2;
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Separable_VHI_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		Mat temp = Mat::zeros(Size(col, 1), CV_32FC1);

		for (int j = 0; j < col; j++)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(0) + j;

			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col : src.ptr<float>(row - 1) + j;
				sum += *sp1;
				sum += *sp2;
			}
			*tp = sum;
		}
		for (int j = 0; j < col; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = j - k >= 0 ? tp - k : temp.ptr<float>(0);
				const float* tp2 = j + k < col ? tp + k : temp.ptr<float>(0) + (col - 1);
				sum += *tp1;
				sum += *tp2;
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Separable_VHI_nonVec::operator()(const cv::Range& range) const
{
	
}



/*
 * SSE
 */
boxFilter_Separable_VHI_SSE::boxFilter_Separable_VHI_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_Separable_VHI_nonVec(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_Separable_VHI_SSE::init()
{
	mDiv = _mm_set1_ps(div);
}

void boxFilter_Separable_VHI_SSE::filter_naive_impl()
{
	for (int i = 0; i < row; i++)
	{
		Mat temp = Mat::zeros(Size(col, 1), CV_32FC1);

		for (int j = 0; j < col; j += 4)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(0) + j;

			__m128 mSum = _mm_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col : src.ptr<float>(row - 1) + j;
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp1));
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			}
			_mm_store_ps(tp, mSum);
		}

		for (int j = 0; j < r; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = j - k >= 0 ? tp - k : temp.ptr<float>(0);
				sum += *tp1;
				sum += *(tp + k);
			}
			*dp = sum * div;
		}
		for (int j = r; j < col - r; j += 4)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m128 mSum = _mm_loadu_ps(tp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(tp - k));
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(tp + k));
			}
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
		}
		for (int j = col - r; j < col; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp2 = j + k < col ? tp + k : temp.ptr<float>(0) + (col - 1);
				sum += *(tp - k);
				sum += *tp2;
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Separable_VHI_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		Mat temp = Mat::zeros(Size(col, 1), CV_32FC1);

		for (int j = 0; j < col; j += 4)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(0) + j;

			__m128 mSum = _mm_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col : src.ptr<float>(row - 1) + j;
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp1));
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
			}
			_mm_store_ps(tp, mSum);
		}

		for (int j = 0; j < r; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = j - k >= 0 ? tp - k : temp.ptr<float>(0);
				sum += *tp1;
				sum += *(tp + k);
			}
			*dp = sum * div;
		}
		for (int j = r; j < col - r; j += 4)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m128 mSum = _mm_loadu_ps(tp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(tp - k));
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(tp + k));
			}
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
		}
		for (int j = col - r; j < col; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp2 = j + k < col ? tp + k : temp.ptr<float>(0) + (col - 1);
				sum += *(tp - k);
				sum += *tp2;
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Separable_VHI_SSE::operator()(const cv::Range& range) const
{
	
}



/*
 * AVX
 */
boxFilter_Separable_VHI_AVX::boxFilter_Separable_VHI_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_Separable_VHI_nonVec(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_Separable_VHI_AVX::init()
{
	mDiv = _mm256_set1_ps(div);
}

void boxFilter_Separable_VHI_AVX::filter_naive_impl()
{
	for (int i = 0; i < row; i++)
	{
		Mat temp = Mat::zeros(Size(col, 1), CV_32FC1);

		for (int j = 0; j < col; j += 8)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(0) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col : src.ptr<float>(row - 1) + j;
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp1));
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			}
			_mm256_store_ps(tp, mSum);
		}

		for (int j = 0; j < r; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = j - k > 0 ? tp - k : temp.ptr<float>(0);
				sum += *tp1;
				sum += *(tp + k);
			}
			*dp = sum * div;
		}
		for (int j = r; j < col - r; j += 8)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m256 mSum = _mm256_loadu_ps(tp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(tp - k));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(tp + k));
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
		for (int j = col - r; j < col; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp2 = j + k < col ? tp + k : temp.ptr<float>(0) + (col - 1);
				sum += *(tp - k);
				sum += *tp2;
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Separable_VHI_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		Mat temp = Mat::zeros(Size(col, 1), CV_32FC1);

		for (int j = 0; j < col; j += 8)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(0) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col : src.ptr<float>(row - 1) + j;
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp1));
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			}
			_mm256_store_ps(tp, mSum);
		}

		for (int j = 0; j < r; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = j - k > 0 ? tp - k : temp.ptr<float>(0);
				sum += *tp1;
				sum += *(tp + k);
			}
			*dp = sum * div;
		}
		for (int j = r; j < col - r; j += 8)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m256 mSum = _mm256_loadu_ps(tp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(tp - k));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(tp + k));
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
		for (int j = col - r; j < col; j++)
		{
			float* tp = temp.ptr<float>(0) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *tp;
			for (int k = 1; k <= r; k++)
			{
				const float* tp2 = j + k < col ? tp + k : temp.ptr<float>(0) + (col - 1);
				sum += *(tp - k);
				sum += *tp2;
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Separable_VHI_AVX::operator()(const cv::Range& range) const
{
	
}