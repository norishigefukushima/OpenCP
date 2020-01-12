#include "boxFilter_Naive.h"

using namespace cv;
using namespace std;


/* --- Naive Gray --- */
boxFilter_Naive_nonVec_Gray::boxFilter_Naive_nonVec_Gray(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	ksize = 2 * r + 1;
	copyMakeBorder(src, copy, r, r, r, r, BOX_FILTER_BORDER_TYPE);
}

/*
 * nonVec
 */
void boxFilter_Naive_nonVec_Gray::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			float* dp = dest.ptr<float>(y) + x;
			float sum = 0.f;
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					sum += *cp;
				}
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Naive_nonVec_Gray::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			float* dp = dest.ptr<float>(y) + x;
			float sum = 0.f;
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					sum += *cp;
				}
			}
			*dp = sum * div;
		}
	}
}

void boxFilter_Naive_nonVec_Gray::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		for (int x = 0; x < col; x ++)
		{
			float* dp = dest.ptr<float>(y) + x;
			float sum = 0.f;
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					sum += *cp;
				}
			}
			*dp = sum * div;
		}
	}
}



/*
 * SSE
 */
void boxFilter_Naive_SSE_Gray::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x += 4)
		{
			float* dp = dest.ptr<float>(y) + x;
			__m128 mSum = _mm_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					mSum = _mm_add_ps(mSum, _mm_loadu_ps(cp));
				}
			}
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
		}
	}
}

void boxFilter_Naive_SSE_Gray::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x += 4)
		{
			float* dp = dest.ptr<float>(y) + x;
			__m128 mSum = _mm_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					mSum = _mm_add_ps(mSum, _mm_loadu_ps(cp));
				}
			}
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
		}
	}
}

void boxFilter_Naive_SSE_Gray::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		for (int x = 0; x < col; x += 4)
		{
			float* dp = dest.ptr<float>(y) + x;
			__m128 mSum = _mm_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					mSum = _mm_add_ps(mSum, _mm_loadu_ps(cp));
				}
			}
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
		}
	}
}



/*
 * AVX
 */
void boxFilter_Naive_AVX_Gray::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x += 8)
		{
			float* dp = dest.ptr<float>(y) + x;
			__m256 mSum = _mm256_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(cp));
				}
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
}

void boxFilter_Naive_AVX_Gray::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x += 8)
		{
			float* dp = dest.ptr<float>(y) + x;
			__m256 mSum = _mm256_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(cp));
				}
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
}

void boxFilter_Naive_AVX_Gray::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		for (int x = 0; x < col; x += 8)
		{
			float* dp = dest.ptr<float>(y) + x;
			__m256 mSum = _mm256_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp = copy.ptr<float>(y + j) + x + i;
					mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(cp));
				}
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
		}
	}
}



/* --- Naive Color --- */
boxFilter_Naive_nonVec_Color::boxFilter_Naive_nonVec_Color(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	ksize = 2 * r + 1;
	copyMakeBorder(src, copy, r, r, r, r, BOX_FILTER_BORDER_TYPE);
	split(copy, vCopy);
}

/*
 * nonVec
 */
void boxFilter_Naive_nonVec_Color::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x++)
		{
			float sumB, sumG, sumR;
			sumB = 0.f, sumG = 0.f, sumR = 0.f;
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					sumB += *cp_B;
					sumG += *cp_G;
					sumR += *cp_R;
				}
			}
			sumB *= div;
			sumG *= div;
			sumR *= div;

			float* dpx = dp + 3 * x;

			*dpx = sumB; dpx++;
			*dpx = sumG; dpx++;
			*dpx = sumR;
		}
	}
}

void boxFilter_Naive_nonVec_Color::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x++)
		{
			float sumB, sumG, sumR;
			sumB = 0.f, sumG = 0.f, sumR = 0.f;
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					sumB += *cp_B;
					sumG += *cp_G;
					sumR += *cp_R;
				}
			}
			sumB *= div;
			sumG *= div;
			sumR *= div;

			float* dpx = dp + 3 * x;

			*dpx = sumB; dpx++;
			*dpx = sumG; dpx++;
			*dpx = sumR;
		}
	}
}

void boxFilter_Naive_nonVec_Color::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x++)
		{
			float sumB, sumG, sumR;
			sumB = 0.f, sumG = 0.f, sumR = 0.f;
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					sumB += *cp_B;
					sumG += *cp_G;
					sumR += *cp_R;
				}
			}
			sumB *= div;
			sumG *= div;
			sumR *= div;

			float* dpx = dp + 3 * x;

			*dpx = sumB; dpx++;
			*dpx = sumG; dpx++;
			*dpx = sumR;
		}
	}
}



/*
 * SSE
 */
void boxFilter_Naive_SSE_Color::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x += 4)
		{
			__m128 mSumB = _mm_setzero_ps();
			__m128 mSumG = _mm_setzero_ps();
			__m128 mSumR = _mm_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					mSumB = _mm_add_ps(mSumB, _mm_loadu_ps(cp_B));
					mSumG = _mm_add_ps(mSumG, _mm_loadu_ps(cp_G));
					mSumR = _mm_add_ps(mSumR, _mm_loadu_ps(cp_R));
				}
			}
			mSumB = _mm_mul_ps(mSumB, mDiv);
			mSumG = _mm_mul_ps(mSumG, mDiv);
			mSumR = _mm_mul_ps(mSumR, mDiv);

			float* dpx = dp + 3 * x;

			const __m128 mb = _mm_permute_ps(mSumB, 0b01101100);
			const __m128 mg = _mm_permute_ps(mSumG, 0b10110001);
			const __m128 mr = _mm_permute_ps(mSumR, 0b11000110);

			const __m128 v0 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b0110), mr, 0b0100);
			const __m128 v1 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b1011), mr, 0b0010);
			const __m128 v2 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b1101), mr, 0b1001);

			_mm_store_ps((dpx + 0), v0);
			_mm_store_ps((dpx + 4), v1);
			_mm_store_ps((dpx + 8), v2);
		}
	}
}

void boxFilter_Naive_SSE_Color::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x += 4)
		{
			__m128 mSumB = _mm_setzero_ps();
			__m128 mSumG = _mm_setzero_ps();
			__m128 mSumR = _mm_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					mSumB = _mm_add_ps(mSumB, _mm_loadu_ps(cp_B));
					mSumG = _mm_add_ps(mSumG, _mm_loadu_ps(cp_G));
					mSumR = _mm_add_ps(mSumR, _mm_loadu_ps(cp_R));
				}
			}
			mSumB = _mm_mul_ps(mSumB, mDiv);
			mSumG = _mm_mul_ps(mSumG, mDiv);
			mSumR = _mm_mul_ps(mSumR, mDiv);

			float* dpx = dp + 3 * x;

			const __m128 mb = _mm_permute_ps(mSumB, 0b01101100);
			const __m128 mg = _mm_permute_ps(mSumG, 0b10110001);
			const __m128 mr = _mm_permute_ps(mSumR, 0b11000110);

			const __m128 v0 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b0110), mr, 0b0100);
			const __m128 v1 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b1011), mr, 0b0010);
			const __m128 v2 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b1101), mr, 0b1001);

			_mm_store_ps((dpx + 0), v0);
			_mm_store_ps((dpx + 4), v1);
			_mm_store_ps((dpx + 8), v2);
		}
	}
}

void boxFilter_Naive_SSE_Color::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x += 4)
		{
			__m128 mSumB = _mm_setzero_ps();
			__m128 mSumG = _mm_setzero_ps();
			__m128 mSumR = _mm_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					mSumB = _mm_add_ps(mSumB, _mm_loadu_ps(cp_B));
					mSumG = _mm_add_ps(mSumG, _mm_loadu_ps(cp_G));
					mSumR = _mm_add_ps(mSumR, _mm_loadu_ps(cp_R));
				}
			}
			mSumB = _mm_mul_ps(mSumB, mDiv);
			mSumG = _mm_mul_ps(mSumG, mDiv);
			mSumR = _mm_mul_ps(mSumR, mDiv);

			float* dpx = dp + 3 * x;

			const __m128 mb = _mm_permute_ps(mSumB, 0b01101100);
			const __m128 mg = _mm_permute_ps(mSumG, 0b10110001);
			const __m128 mr = _mm_permute_ps(mSumR, 0b11000110);

			const __m128 v0 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b0110), mr, 0b0100);
			const __m128 v1 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b1011), mr, 0b0010);
			const __m128 v2 = _mm_blend_ps(_mm_blend_ps(mb, mg, 0b1101), mr, 0b1001);

			_mm_store_ps((dpx + 0), v0);
			_mm_store_ps((dpx + 4), v1);
			_mm_store_ps((dpx + 8), v2);
		}
	}
}



/*
 * AVX
 */
void boxFilter_Naive_AVX_Color::filter_naive_impl()
{
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x += 8)
		{
			__m256 mSumB = _mm256_setzero_ps();
			__m256 mSumG = _mm256_setzero_ps();
			__m256 mSumR = _mm256_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					mSumB = _mm256_add_ps(mSumB, _mm256_loadu_ps(cp_B));
					mSumG = _mm256_add_ps(mSumG, _mm256_loadu_ps(cp_G));
					mSumR = _mm256_add_ps(mSumR, _mm256_loadu_ps(cp_R));
				}
			}
			mSumB = _mm256_mul_ps(mSumB, mDiv);
			mSumG = _mm256_mul_ps(mSumG, mDiv);
			mSumR = _mm256_mul_ps(mSumR, mDiv);

			float* dpx = dp + 3 * x;

			const __m256 ma = _mm256_permutevar8x32_ps(mSumB, mask_b);
			const __m256 mb = _mm256_permutevar8x32_ps(mSumG, mask_g);
			const __m256 mc = _mm256_permutevar8x32_ps(mSumR, mask_r);

			const __m256 v0 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 146), mc, 36);
			const __m256 v1 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 36), mc, 73);
			const __m256 v2 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 73), mc, 146);

			_mm256_storeu_ps((dpx + 0), v0);
			_mm256_storeu_ps((dpx + 8), v1);
			_mm256_storeu_ps((dpx + 16), v2);
		}
	}
}

void boxFilter_Naive_AVX_Color::filter_omp_impl()
{
#pragma omp parallel for
	for (int y = 0; y < row; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x += 8)
		{
			__m256 mSumB = _mm256_setzero_ps();
			__m256 mSumG = _mm256_setzero_ps();
			__m256 mSumR = _mm256_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;

					mSumB = _mm256_add_ps(mSumB, _mm256_loadu_ps(cp_B));
					mSumG = _mm256_add_ps(mSumG, _mm256_loadu_ps(cp_G));
					mSumR = _mm256_add_ps(mSumR, _mm256_loadu_ps(cp_R));
				}
			}
			
			mSumB = _mm256_mul_ps(mSumB, mDiv);
			mSumG = _mm256_mul_ps(mSumG, mDiv);
			mSumR = _mm256_mul_ps(mSumR, mDiv);

			float* dpx = dp + 3 * x;

			const __m256 ma = _mm256_permutevar8x32_ps(mSumB, mask_b);
			const __m256 mb = _mm256_permutevar8x32_ps(mSumG, mask_g);
			const __m256 mc = _mm256_permutevar8x32_ps(mSumR, mask_r);

			const __m256 v0 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 146), mc, 36);
			const __m256 v1 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 36), mc, 73);
			const __m256 v2 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 73), mc, 146);

			_mm256_storeu_ps((dpx + 0), v0);
			_mm256_storeu_ps((dpx + 8), v1);
			_mm256_storeu_ps((dpx + 16), v2);
		}
	}
}

void boxFilter_Naive_AVX_Color::operator()(const cv::Range& range) const
{
	for (int y = range.start; y < range.end; y++)
	{
		float* dp = dest.ptr<float>(y);
		for (int x = 0; x < col; x += 8)
		{
			__m256 mSumB = _mm256_setzero_ps();
			__m256 mSumG = _mm256_setzero_ps();
			__m256 mSumR = _mm256_setzero_ps();
			for (int j = 0; j < ksize; j++)
			{
				for (int i = 0; i < ksize; i++)
				{
					const float* cp_B = vCopy[0].ptr<float>(y + j) + x + i;
					const float* cp_G = vCopy[1].ptr<float>(y + j) + x + i;
					const float* cp_R = vCopy[2].ptr<float>(y + j) + x + i;
					
					mSumB = _mm256_add_ps(mSumB, _mm256_loadu_ps(cp_B));
					mSumG = _mm256_add_ps(mSumG, _mm256_loadu_ps(cp_G));
					mSumR = _mm256_add_ps(mSumR, _mm256_loadu_ps(cp_R));
				}
			}

			mSumB = _mm256_mul_ps(mSumB, mDiv);
			mSumG = _mm256_mul_ps(mSumG, mDiv);
			mSumR = _mm256_mul_ps(mSumR, mDiv);

			float* dpx = dp + 3 * x;

			const __m256 ma = _mm256_permutevar8x32_ps(mSumB, mask_b);
			const __m256 mb = _mm256_permutevar8x32_ps(mSumG, mask_g);
			const __m256 mc = _mm256_permutevar8x32_ps(mSumR, mask_r);

			const __m256 v0 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 146), mc, 36);
			const __m256 v1 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 36), mc, 73);
			const __m256 v2 = _mm256_blend_ps(_mm256_blend_ps(ma, mb, 73), mc, 146);

			_mm256_storeu_ps((dpx + 0), v0);
			_mm256_storeu_ps((dpx + 8), v1);
			_mm256_storeu_ps((dpx + 16), v2);
		}
	}
}
