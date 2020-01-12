#include "boxFilter_SSAT_HV_8x8.h"

using namespace std;
using namespace cv;

boxFilter_SSAT_HV_8x8::boxFilter_SSAT_HV_8x8(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_base(_src, _dest, _r, _parallelType)
{
	padding = (r % 4 == 0) ? 0 : (4 - (r % 4)) * 2;
	copyMakeBorder(src, copy, r, r, r, r + padding, BORDER_REPLICATE);
	mDiv = _mm256_set1_ps(div);
}

void boxFilter_SSAT_HV_8x8::filter_naive_impl()
{
	/*   8x8 row -> transpose -> row   */
	{
		Mat temp(Size(8 + r + r + padding, 8), CV_32F);
		Mat temp_t(Size(8, 8 + r + r + padding), CV_32F);
		Mat sum(Size(8, 8), CV_32F);

		for (int i = 0; i < row; i += 8)
		{
			for (int j = 0; j < col; j += 8)
			{
				for (int x = 0; x < temp.cols; x += 8)
				{
					__m256 mSum = _mm256_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(copy.ptr<float>(y + i) + j + x));
					}
					_mm256_store_ps(temp.ptr<float>(0) + x, mSum);

					// y = 1 ~ 7
					for (int y = 1; y < 8; y++)
					{
						mSum = _mm256_sub_ps(mSum, _mm256_load_ps(copy.ptr<float>(y + i - 1) + j + x));
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(copy.ptr<float>(y + i + r + r) + j + x));
						_mm256_store_ps(temp.ptr<float>(y) + x, mSum);
					}
				}
				temp_t = temp.t();
				{
					__m256 mSum = _mm256_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(temp_t.ptr<float>(y)));
					}
					_mm256_store_ps(sum.ptr<float>(0), mSum);

					// y = 1 ~ 7
					for (int y = 1; y < 8; y++)
					{
						mSum = _mm256_sub_ps(mSum, _mm256_load_ps(temp_t.ptr<float>(y - 1)));
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(temp_t.ptr<float>(y + r + r)));
						_mm256_store_ps(sum.ptr<float>(y), mSum);
					}
				}
				sum = sum.t();
				for (int y = 0; y < 8; y++)
				{
					_mm256_store_ps(dest.ptr<float>(i + y) + j, _mm256_mul_ps(_mm256_load_ps(sum.ptr<float>(y)), mDiv));
				}
			}
		}
	}

	/*   8x8 row -> col   */
	/*{
		Mat temp(Size(8, 8 + r + r), CV_32F);

		for (int i = 0; i < row; i += 8)
		{
			for (int j = 0; j < col; j += 8)
			{
				for (int y = 0; y < 8 + r + r; y++)
				{
					float* sp = copy.ptr<float>(i + y) + j + r;
					float* dp = temp.ptr<float>(y);

					__m256 mSum = _mm256_load_ps(sp);
					for (int x = 1; x <= r; x++)
					{
						mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp + x));
						mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp - x));
					}
					_mm256_storeu_ps(dp, mSum);
				}
				{
					float* sp_prev = temp.ptr<float>(0);
					float* sp_next = temp.ptr<float>(0);
					float* dp = dest.ptr<float>(i) + j;
					__m256 mSum = _mm256_setzero_ps();

					for (int y = 0; y < 2 * r + 1; y++)
					{
						mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp_next));
						sp_next += 8;
					}
					_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
					dp += col;
					for (int y = 2 * r + 1; y < 8 + r + r; y++)
					{
						mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(sp_prev));
						sp_prev += 8;
						mSum = _mm256_add_ps(mSum, _mm256_loadu_ps(sp_next));
						sp_next += 8;
						_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
						dp += col;
					}
				}
			}
		}
	}*/

	/*   8x8 row -> col (colSum)   */
	/*{
		Mat temp(Size(col, row + r + r), CV_32F);
		Mat sum(Size(col, 1), CV_32F);
		__m256 mSum = _mm256_setzero_ps();
		for (int j = 0; j < col; j += 8)
		{
			mSum = _mm256_setzero_ps();
			float* dp = dest.ptr<float>(0) + j;
			for (int y = 0; y < r + r + 1; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp = temp.ptr<float>(y) + j;

				__m256 mRowSum = _mm256_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp + x));
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp - x));
				}
				_mm256_storeu_ps(tp, mRowSum);
				mSum = _mm256_add_ps(mSum, mRowSum);
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += col;
			for (int y = r + r + 1; y < r + r + 8; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp_prev = temp.ptr<float>(0) + j;
				float* tp_next = temp.ptr<float>(y) + j;

				__m256 mRowSum = _mm256_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp + x));
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp - x));
				}
				_mm256_storeu_ps(tp_next, mRowSum);

				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(tp_prev));
				mSum = _mm256_add_ps(mSum, mRowSum);
				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += col;
			}
			_mm256_storeu_ps(sum.ptr<float>(0) + j, mSum);
		}

		for (int i = 8; i < row; i += 8)
		{
			for (int j = 0; j < col; j += 8)
			{
				float* dp = dest.ptr<float>(i) + j;
				mSum = _mm256_loadu_ps(sum.ptr<float>(0) + j);
				for (int y = 0; y < 8; y++)
				{
					float* sp = copy.ptr<float>(y + i + r + r) + j + r;
					float* tp_prev = temp.ptr<float>(y + i - 1) + j;
					float* tp_next = temp.ptr<float>(y + i + r + r) + j;

					__m256 mRowSum = _mm256_load_ps(sp);
					for (int x = 1; x <= r; x++)
					{
						mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp + x));
						mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp - x));
					}
					_mm256_storeu_ps(tp_next, mRowSum);

					mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(tp_prev));
					mSum = _mm256_add_ps(mSum, mRowSum);
					_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
					dp += col;
				}
				_mm256_storeu_ps(sum.ptr<float>(0) + j, mSum);
			}
		}
	}*/
}

void boxFilter_SSAT_HV_8x8::filter_omp_impl()
{
	/*   8x8 row -> transpose -> row   */
	{
#pragma omp parallel for
		for (int i = 0; i < row; i += 8)
		{
			for (int j = 0; j < col; j += 8)
			{
				Mat temp(Size(8 + r + r + padding, 8), CV_32F);
				Mat temp_t(Size(8, 8 + r + r + padding), CV_32F);
				Mat sum(Size(8, 8), CV_32F);
				for (int x = 0; x < temp.cols; x += 8)
				{
					__m256 mSum = _mm256_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(copy.ptr<float>(y + i) + j + x));
					}
					_mm256_store_ps(temp.ptr<float>(0) + x, mSum);

					// y = 1 ~ 7
					for (int y = 1; y < 8; y++)
					{
						mSum = _mm256_sub_ps(mSum, _mm256_load_ps(copy.ptr<float>(y + i - 1) + j + x));
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(copy.ptr<float>(y + i + r + r) + j + x));
						_mm256_store_ps(temp.ptr<float>(y) + x, mSum);
					}
				}
				temp_t = temp.t();
				{
					__m256 mSum = _mm256_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(temp_t.ptr<float>(y)));
					}
					_mm256_store_ps(sum.ptr<float>(0), mSum);

					// y = 1 ~ 7
					for (int y = 1; y < 8; y++)
					{
						mSum = _mm256_sub_ps(mSum, _mm256_load_ps(temp_t.ptr<float>(y - 1)));
						mSum = _mm256_add_ps(mSum, _mm256_load_ps(temp_t.ptr<float>(y + r + r)));
						_mm256_store_ps(sum.ptr<float>(y), mSum);
					}
				}
				sum = sum.t();
				for (int y = 0; y < 8; y++)
				{
					_mm256_store_ps(dest.ptr<float>(i + y) + j, _mm256_mul_ps(_mm256_load_ps(sum.ptr<float>(y)), mDiv));
				}
			}
		}
	}

	/*   8x8 row -> col (colSum)   */
	/*{
		Mat temp(Size(col, row + r + r), CV_32F);
		Mat sum(Size(col, 1), CV_32F);
#pragma omp parallel for
		for (int j = 0; j < col; j += 8)
		{
			__m256 mSum = _mm256_setzero_ps();
			float* dp = dest.ptr<float>(0) + j;
			for (int y = 0; y < r + r + 1; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp = temp.ptr<float>(y) + j;

				__m256 mRowSum = _mm256_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp + x));
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp - x));
				}
				_mm256_storeu_ps(tp, mRowSum);
				mSum = _mm256_add_ps(mSum, mRowSum);
			}
			_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += col;
			for (int y = r + r + 1; y < r + r + 8; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp_prev = temp.ptr<float>(0) + j;
				float* tp_next = temp.ptr<float>(y) + j;

				__m256 mRowSum = _mm256_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp + x));
					mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp - x));
				}
				_mm256_storeu_ps(tp_next, mRowSum);

				mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(tp_prev));
				mSum = _mm256_add_ps(mSum, mRowSum);
				_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += col;
			}
			_mm256_storeu_ps(sum.ptr<float>(0) + j, mSum);
		}

		for (int i = 8; i < row; i += 8)
		{
#pragma omp parallel for
			for (int j = 0; j < col; j += 8)
			{
				float* dp = dest.ptr<float>(i) + j;
				__m256 mSum = _mm256_loadu_ps(sum.ptr<float>(0) + j);
				for (int y = 0; y < 8; y++)
				{
					float* sp = copy.ptr<float>(y + i + r + r) + j + r;
					float* tp_prev = temp.ptr<float>(y + i - 1) + j;
					float* tp_next = temp.ptr<float>(y + i + r + r) + j;

					__m256 mRowSum = _mm256_load_ps(sp);
					for (int x = 1; x <= r; x++)
					{
						mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp + x));
						mRowSum = _mm256_add_ps(mRowSum, _mm256_loadu_ps(sp - x));
					}
					_mm256_storeu_ps(tp_next, mRowSum);

					mSum = _mm256_sub_ps(mSum, _mm256_loadu_ps(tp_prev));
					mSum = _mm256_add_ps(mSum, mRowSum);
					_mm256_storeu_ps(dp, _mm256_mul_ps(mSum, mDiv));
					dp += col;
				}
				_mm256_storeu_ps(sum.ptr<float>(0) + j, mSum);
			}
		}
	}*/
}

void boxFilter_SSAT_HV_8x8::operator()(const cv::Range& range) const
{

}
