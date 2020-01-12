#include "boxFilter_Separable.h"

using namespace std;
using namespace cv;

#define _WITHOUT_COPYMAKE_ 1

boxFilter_Separable_HV_nonVec::boxFilter_Separable_HV_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	ksize = 2 * r + 1;

#if _WITHOUT_COPYMAKE_
	temp = Mat::zeros(Size(col, row), src.type());
#else
	copyMakeBorder(src, copy, r, r, r, r, BORDER_TYPE);
	temp = Mat::zeros(Size(col, row + r + r), src.type());
#endif
}

void boxFilter_Separable_HV_nonVec::filter_naive_impl()
{
#if _WITHOUT_COPYMAKE_
	//TODO: Without CopyMake
#else
	for (int i = 0; i < row + r + r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			float* sp = copy.ptr<float>(i) + j + r;
			float* dp = temp.ptr<float>(i) + j;

			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				sum += *(sp + k);
				sum += *(sp - k);
			}
			*dp = sum;
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			float* sp = temp.ptr<float>(i + r) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				sum += *(sp + k*col);
				sum += *(sp - k*col);
			}
			*dp = sum * div;
		}
	}
#endif
}

void boxFilter_Separable_HV_nonVec::filter_omp_impl()
{
#if _WITHOUT_COPYMAKE_
	//TODO: Without CopyMake
#else
#pragma omp parallel for
	for (int i = 0; i < row + r + r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			float* sp = copy.ptr<float>(i) + j + r;
			float* dp = temp.ptr<float>(i) + j;

			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				sum += *(sp + k);
				sum += *(sp - k);
			}
			*dp = sum;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			float* sp = temp.ptr<float>(i + r) + j;
			float* dp = dest.ptr<float>(i) + j;

			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				sum += *(sp + k*col);
				sum += *(sp - k*col);
			}
			*dp = sum * div;
		}
	}
#endif
}

void boxFilter_Separable_HV_nonVec::operator()(const cv::Range& range) const
{

}



boxFilter_Separable_HV_SSE::boxFilter_Separable_HV_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	:boxFilter_Separable_HV_nonVec(_src, _dest, _r, _parallelType)
{
	mDiv = _mm_set1_ps(div);
}

void boxFilter_Separable_HV_SSE::filter_naive_impl()
{
#if _WITHOUT_COPYMAKE_
	//TODO: Without CopyMake
#else
	for (int i = 0; i < row + r + r; i++)
	{
		for (int j = 0; j < col; j += 4)
		{
			float* sp = copy.ptr<float>(i) + j + r;
			float* dp = temp.ptr<float>(i) + j;

			__m128 mSum = _mm_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp + k));
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp - k));
			}
			_mm_store_ps(dp, mSum);
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j += 4)
		{
			float* sp = temp.ptr<float>(i + r) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m128 mSum = _mm_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp + k*col));
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp - k*col));
			}
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
		}
	}
#endif
}

void boxFilter_Separable_HV_SSE::filter_omp_impl()
{
#if _WITHOUT_COPYMAKE_
	//TODO: Without CopyMake
#else
#pragma omp parallel for
	for (int i = 0; i < row + r + r; i++)
	{
		for (int j = 0; j < col; j += 4)
		{
			float* sp = copy.ptr<float>(i) + j + r;
			float* dp = temp.ptr<float>(i) + j;

			__m128 mSum = _mm_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp + k));
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp - k));
			}
			_mm_store_ps(dp, mSum);
		}
	}
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j += 4)
		{
			float* sp = temp.ptr<float>(i + r) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m128 mSum = _mm_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp + k*col));
				mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp - k*col));
			}
			_mm_store_ps(dp, _mm_mul_ps(mSum, mDiv));
		}
	}
#endif
}

void boxFilter_Separable_HV_SSE::operator()(const cv::Range& range) const
{
	
}



boxFilter_Separable_HV_AVX::boxFilter_Separable_HV_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	:boxFilter_Separable_HV_nonVec(_src, _dest, _r, _parallelType)
{
	cn = src.channels();
	mDiv = _mm256_set1_ps(div);
}

void boxFilter_Separable_HV_AVX::filter_naive_impl()
{
	// without
#if _WITHOUT_COPYMAKE_
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < r; j++)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = j - k > 0 ? sp - k : src.ptr<float>(i);
				sum += *sp1;
				sum += *(sp + k);
			}
			*tp = sum;
		}
		for (int j = r; j < col - r; j += 8)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(i) + j;
			__m256 mSum = _mm256_loadu_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k));
			}
			_mm256_storeu_ps(tp, mSum);
		}
		for (int j = col - r; j < col; j++)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp2 = j + k < col ? sp + k : src.ptr<float>(i) + (col - 1);
				sum += *(sp - k);
				sum += *sp2;
			}
			*tp = sum;
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j += 8)
		{
			float* tp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			__m256 mSum = _mm256_load_ps(tp);
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = i - k >= 0 ? tp - k*col : temp.ptr<float>(0) + j;
				const float* tp2 = i + k < row ? tp + k*col : temp.ptr<float>(row - 1) + j;
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(tp1));
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(tp2));
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
#else
	for (int i = 0; i < row + r + r; i++)
	{
		for (int j = 0; j < col * cn; j += 8)
		{
			float* sp = copy.ptr<float>(i) + j + r * cn;
			float* dp = temp.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*cn));
			}
			_mm256_store_ps(dp, mSum);
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col * cn; j += 8)
		{
			float* sp = temp.ptr<float>(i + r) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*col*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*col*cn));
			}
			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
#endif
}

void boxFilter_Separable_HV_AVX::filter_omp_impl()
{
#if _WITHOUT_COPYMAKE_
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < r; j++)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = j - k > 0 ? sp - k : src.ptr<float>(i);
				sum += *sp1;
				sum += *(sp + k);
			}
			*tp = sum;
		}
		for (int j = r; j < col - r; j += 8)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(i) + j;
			__m256 mSum = _mm256_loadu_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k));
			}
			_mm256_storeu_ps(tp, mSum);
		}
		for (int j = col - r; j < col; j++)
		{
			float* sp = src.ptr<float>(i) + j;
			float* tp = temp.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp2 = j + k < col ? sp + k : src.ptr<float>(i) + (col - 1);
				sum += *(sp - k);
				sum += *sp2;
			}
			*tp = sum;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j += 8)
		{
			float* tp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			__m256 mSum = _mm256_load_ps(tp);
			for (int k = 1; k <= r; k++)
			{
				const float* tp1 = i - k >= 0 ? tp - k*col : temp.ptr<float>(0) + j;
				const float* tp2 = i + k < row ? tp + k*col : temp.ptr<float>(row - 1) + j;
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(tp1));
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(tp2));
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
#else
#pragma omp parallel for
	for (int i = 0; i < row + r + r; i++)
	{
		for (int j = 0; j < col * cn; j += 8)
		{
			float* sp = copy.ptr<float>(i) + j + r * cn;
			float* dp = temp.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*cn));
			}
			_mm256_store_ps(dp, mSum);
		}
	}
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col * cn; j += 8)
		{
			float* sp = temp.ptr<float>(i + r) + j;
			float* dp = dest.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*col*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*col*cn));
			}
			_mm256_store_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
#endif
}

void boxFilter_Separable_HV_AVX::operator()(const cv::Range& range) const
{
	
}



boxFilter_Separable_VH_AVX::boxFilter_Separable_VH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	padded = 8 % (8 - (col + r + r) % 8);
	ksize = 2 * r + 1;

#if _WITHOUT_COPYMAKE_
	temp = Mat::zeros(Size(col, row), src.type());
#else
	copyMakeBorder(src, copy, r, r, r, r + padded, BORDER_TYPE);
	temp = Mat::zeros(Size(col + r + r, row), src.type());
#endif

	cn = src.channels();
	mDiv = _mm256_set1_ps(div);
}

void boxFilter_Separable_VH_AVX::filter_naive_impl()
{
#if _WITHOUT_COPYMAKE_
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col*cn; j += 8)
		{
			float* sp = src.ptr<float>(i) + j;
			float* dp = temp.ptr<float>(i) + j;
			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col*cn : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col*cn : src.ptr<float>(row - 1) + j;
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp1));
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			}
			_mm256_store_ps(dp, mSum);
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < r; j++)
		{
			float* sp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = j - k >= 0 ? sp - k*cn : temp.ptr<float>(i);
				sum += *sp1;
				sum += *(sp + k);
			}
			*dp = sum * div;
		}
		for (int j = r; j < col - r; j += 8)
		{
			float* sp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			__m256 mSum = _mm256_loadu_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k));
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
		for (int j = col - r; j < col; j++)
		{
			float* sp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp2 = j + k < col ? sp + k*cn : temp.ptr<float>(i) + (col - 1);
				sum += *(sp - k);
				sum += *sp2;
			}
			*dp = sum * div;
		}
	}
#else
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < temp.cols * cn; j += 8)
		{
			float* sp = copy.ptr<float>(i + r) + j;
			float* dp = temp.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*copy.cols*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*copy.cols*cn));
			}
			_mm256_store_ps(dp, mSum);
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col * cn; j += 8)
		{
			float* sp = temp.ptr<float>(i) + j + r * cn;
			float* dp = dest.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*cn));
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
#endif
}

void boxFilter_Separable_VH_AVX::filter_omp_impl()
{
#if _WITHOUT_COPYMAKE_
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col*cn; j += 8)
		{
			float* sp = src.ptr<float>(i) + j;
			float* dp = temp.ptr<float>(i) + j;
			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = i - k >= 0 ? sp - k*col*cn : src.ptr<float>(0) + j;
				const float* sp2 = i + k < row ? sp + k*col*cn : src.ptr<float>(row - 1) + j;
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp1));
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
			}
			_mm256_store_ps(dp, mSum);
		}
	}
#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < r; j++)
		{
			float* sp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp1 = j - k >= 0 ? sp - k*cn : temp.ptr<float>(i);
				sum += *sp1;
				sum += *(sp + k);
			}
			*dp = sum * div;
		}
		for (int j = r; j < col - r; j += 8)
		{
			float* sp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			__m256 mSum = _mm256_loadu_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k));
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
		for (int j = col - r; j < col; j++)
		{
			float* sp = temp.ptr<float>(i) + j;
			float* dp = dest.ptr<float>(i) + j;
			float sum = *sp;
			for (int k = 1; k <= r; k++)
			{
				const float* sp2 = j + k < col ? sp + k*cn : temp.ptr<float>(i) + (col - 1);
				sum += *(sp - k);
				sum += *sp2;
			}
			*dp = sum * div;
		}
	}
#else
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < temp.cols * cn; j += 8)
		{
			float* sp = copy.ptr<float>(i + r) + j;
			float* dp = temp.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*copy.cols*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*copy.cols*cn));
			}
			_mm256_store_ps(dp, mSum);
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col * cn; j += 8)
		{
			float* sp = temp.ptr<float>(i) + j + r * cn;
			float* dp = dest.ptr<float>(i) + j;

			__m256 mSum = _mm256_load_ps(sp);
			for (int k = 1; k <= r; k++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + k*cn));
				mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - k*cn));
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
#endif
}

void boxFilter_Separable_VH_AVX::operator()(const cv::Range& range) const
{
	
}
