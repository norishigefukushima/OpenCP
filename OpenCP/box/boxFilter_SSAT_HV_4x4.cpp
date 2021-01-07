#include "boxFilter_SSAT_HV_4x4.h"

using namespace std;
using namespace cv;

boxFilter_SSAT_HV_4x4::boxFilter_SSAT_HV_4x4(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: BoxFilterBase(_src, _dest, _r, _parallelType)
{
	padding = (r % 2) * 2;
	copyMakeBorder(src, copy, r, r, r, r + padding, BORDER_REPLICATE);
	mDiv = _mm_set1_ps(div);
}

void boxFilter_SSAT_HV_4x4::filter_naive_impl()
{
	/*   4x4 row -> transpose -> row   */
	{
		Mat temp(Size(4 + r + r + padding, 4), CV_32F);
		Mat temp_t(Size(4, 4 + r + r + padding), CV_32F);
		Mat sum(Size(4, 4), CV_32F);

		for (int i = 0; i < row; i += 4)
		{
			for (int j = 0; j < col; j += 4)
			{
				for (int x = 0; x < temp.cols; x += 4)
				{
					__m128 mSum = _mm_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm_add_ps(mSum, _mm_load_ps(copy.ptr<float>(y + i) + j + x));
					}
					_mm_store_ps(temp.ptr<float>(0) + x, mSum);

					// y = 1 ~ 3
					for (int y = 1; y < 4; y++)
					{
						mSum = _mm_sub_ps(mSum, _mm_load_ps(copy.ptr<float>(y + i - 1) + j + x));
						mSum = _mm_add_ps(mSum, _mm_load_ps(copy.ptr<float>(y + i + r + r) + j + x));
						_mm_store_ps(temp.ptr<float>(y) + x, mSum);
					}
				}
				temp_t = temp.t();
				{
					__m128 mSum = _mm_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm_add_ps(mSum, _mm_load_ps(temp_t.ptr<float>(y)));
					}
					_mm_store_ps(sum.ptr<float>(0), mSum);

					// y = 1 ~ 3
					for (int y = 1; y < 4; y++)
					{
						mSum = _mm_sub_ps(mSum, _mm_load_ps(temp_t.ptr<float>(y - 1)));
						mSum = _mm_add_ps(mSum, _mm_load_ps(temp_t.ptr<float>(y + r + r)));
						_mm_store_ps(sum.ptr<float>(y), mSum);
					}
				}
				sum = sum.t();
				for (int y = 0; y < 4; y++)
				{
					_mm_store_ps(dest.ptr<float>(i + y) + j, _mm_mul_ps(_mm_load_ps(sum.ptr<float>(y)), mDiv));
				}
			}
		}
	}

	/*   4x4 row -> col   */
	/*{
		Mat temp(Size(4, 4 + r + r), CV_32F);

		for (int i = 0; i < row; i += 4)
		{
			for (int j = 0; j < col; j += 4)
			{
				for (int y = 0; y < 4 + r + r; y++)
				{
					float* sp = copy.ptr<float>(i + y) + j + r;
					float* dp = temp.ptr<float>(y);

					__m128 mSum = _mm_load_ps(sp);
					for (int x = 1; x <= r; x++)
					{
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp + x));
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp - x));
					}
					_mm_storeu_ps(dp, mSum);
				}
				{
					float* sp_prev = temp.ptr<float>(0);
					float* sp_next = temp.ptr<float>(0);
					float* dp = dest.ptr<float>(i) + j;
					__m128 mSum = _mm_setzero_ps();

					for (int y = 0; y < 2 * r + 1; y++)
					{
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp_next));
						sp_next += 4;
					}
					_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
					dp += col;
					for (int y = 2 * r + 1; y < 4 + r + r; y++)
					{
						mSum = _mm_sub_ps(mSum, _mm_loadu_ps(sp_prev));
						sp_prev += 4;
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp_next));
						sp_next += 4;
						_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
						dp += col;
					}
				}
			}
		}
	}*/

	/*   4x4 row -> col (colSum)   */
	/*{
		Mat temp(Size(col, row + r + r), CV_32F);
		Mat sum(Size(col, 1), CV_32F);
		__m128 mSum = _mm_setzero_ps();
		for (int j = 0; j < col; j += 4)
		{
			mSum = _mm_setzero_ps();
			float* dp = dest.ptr<float>(0) + j;
			for (int y = 0; y < r + r + 1; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp = temp.ptr<float>(y) + j;

				__m128 mRowSum = _mm_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp + x));
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp - x));
				}
				_mm_storeu_ps(tp, mRowSum);
				mSum = _mm_add_ps(mSum, mRowSum);
			}
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += col;
			for (int y = r + r + 1; y < r + r + 4; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp_prev = temp.ptr<float>(0) + j;
				float* tp_next = temp.ptr<float>(y) + j;

				__m128 mRowSum = _mm_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp + x));
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp - x));
				}
				_mm_storeu_ps(tp_next, mRowSum);

				mSum = _mm_sub_ps(mSum, _mm_loadu_ps(tp_prev));
				mSum = _mm_add_ps(mSum, mRowSum);
				_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += col;
			}
			_mm_storeu_ps(sum.ptr<float>(0) + j, mSum);
		}

		for (int i = 4; i < row; i += 4)
		{
			for (int j = 0; j < col; j += 4)
			{
				float* dp = dest.ptr<float>(i) + j;
				mSum = _mm_loadu_ps(sum.ptr<float>(0) + j);
				for (int y = 0; y < 4; y++)
				{
					float* sp = copy.ptr<float>(y + i + r + r) + j + r;
					float* tp_prev = temp.ptr<float>(y + i - 1) + j;
					float* tp_next = temp.ptr<float>(y + i + r + r) + j;

					__m128 mRowSum = _mm_load_ps(sp);
					for (int x = 1; x <= r; x++)
					{
						mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp + x));
						mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp - x));
					}
					_mm_storeu_ps(tp_next, mRowSum);

					mSum = _mm_sub_ps(mSum, _mm_loadu_ps(tp_prev));
					mSum = _mm_add_ps(mSum, mRowSum);
					_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
					dp += col;
				}
				_mm_storeu_ps(sum.ptr<float>(0) + j, mSum);
			}
		}
	}*/
}

void boxFilter_SSAT_HV_4x4::filter_omp_impl()
{
	/*   4x4 row -> transpose -> row   */
	{
#pragma omp parallel for
		for (int i = 0; i < row; i += 4)
		{
			for (int j = 0; j < col; j += 4)
			{
				Mat temp(Size(4 + r + r + padding, 4), CV_32F);
				Mat temp_t(Size(4, 4 + r + r + padding), CV_32F);
				Mat sum(Size(4, 4), CV_32F);

				for (int x = 0; x < temp.cols; x += 4)
				{
					__m128 mSum = _mm_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm_add_ps(mSum, _mm_load_ps(copy.ptr<float>(y + i) + j + x));
					}
					_mm_store_ps(temp.ptr<float>(0) + x, mSum);

					// y = 1 ~ 3
					for (int y = 1; y < 4; y++)
					{
						mSum = _mm_sub_ps(mSum, _mm_load_ps(copy.ptr<float>(y + i - 1) + j + x));
						mSum = _mm_add_ps(mSum, _mm_load_ps(copy.ptr<float>(y + i + r + r) + j + x));
						_mm_store_ps(temp.ptr<float>(y) + x, mSum);
					}
				}
				temp_t = temp.t();
				{
					__m128 mSum = _mm_setzero_ps();

					// y = 0
					for (int y = 0; y < r + r + 1; y++)
					{
						mSum = _mm_add_ps(mSum, _mm_load_ps(temp_t.ptr<float>(y)));
					}
					_mm_store_ps(sum.ptr<float>(0), mSum);

					// y = 1 ~ 3
					for (int y = 1; y < 4; y++)
					{
						mSum = _mm_sub_ps(mSum, _mm_load_ps(temp_t.ptr<float>(y - 1)));
						mSum = _mm_add_ps(mSum, _mm_load_ps(temp_t.ptr<float>(y + r + r)));
						_mm_store_ps(sum.ptr<float>(y), mSum);
					}
				}
				sum = sum.t();
				for (int y = 0; y < 4; y++)
				{
					_mm_store_ps(dest.ptr<float>(i + y) + j, _mm_mul_ps(_mm_load_ps(sum.ptr<float>(y)), mDiv));
				}
			}
		}
	}

	/*   4x4 row -> col   */
	/*{
#pragma omp parallel for
		for (int i = 0; i < row; i += 4)
		{
			Mat temp(Size(4, 4 + r + r), CV_32F);
			for (int j = 0; j < col; j += 4)
			{
				for (int y = 0; y < 4 + r + r; y++)
				{
					float* sp = copy.ptr<float>(i + y) + j + r;
					float* dp = temp.ptr<float>(y);

					__m128 mSum = _mm_load_ps(sp);
					for (int x = 1; x <= r; x++)
					{
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp + x));
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp - x));
					}
					_mm_storeu_ps(dp, mSum);
				}
				{
					float* sp_prev = temp.ptr<float>(0);
					float* sp_next = temp.ptr<float>(0);
					float* dp = dest.ptr<float>(i) + j;
					__m128 mSum = _mm_setzero_ps();

					for (int y = 0; y < 2 * r + 1; y++)
					{
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp_next));
						sp_next += 4;
					}
					_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
					dp += col;
					for (int y = 2 * r + 1; y < 4 + r + r; y++)
					{
						mSum = _mm_sub_ps(mSum, _mm_loadu_ps(sp_prev));
						sp_prev += 4;
						mSum = _mm_add_ps(mSum, _mm_loadu_ps(sp_next));
						sp_next += 4;
						_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
						dp += col;
					}
				}
			}
		}
	}*/

	/*   4x4 row -> col (colSum)   */
	/*{
		Mat temp(Size(col, row + r + r), CV_32F);
		Mat sum(Size(col, 1), CV_32F);

#pragma omp parallel for
		for (int j = 0; j < col; j += 4)
		{
			__m128 mSum = _mm_setzero_ps();
			float* dp = dest.ptr<float>(0) + j;
			for (int y = 0; y < r + r + 1; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp = temp.ptr<float>(y) + j;

				__m128 mRowSum = _mm_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp + x));
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp - x));
				}
				_mm_storeu_ps(tp, mRowSum);
				mSum = _mm_add_ps(mSum, mRowSum);
			}
			_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += col;
			for (int y = r + r + 1; y < r + r + 4; y++)
			{
				float* sp = copy.ptr<float>(y) + j + r;
				float* tp_prev = temp.ptr<float>(0) + j;
				float* tp_next = temp.ptr<float>(y) + j;

				__m128 mRowSum = _mm_load_ps(sp);
				for (int x = 1; x <= r; x++)
				{
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp + x));
					mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp - x));
				}
				_mm_storeu_ps(tp_next, mRowSum);

				mSum = _mm_sub_ps(mSum, _mm_loadu_ps(tp_prev));
				mSum = _mm_add_ps(mSum, mRowSum);
				_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += col;
			}
			_mm_storeu_ps(sum.ptr<float>(0) + j, mSum);
		}

		for (int i = 4; i < row; i += 4)
		{
#pragma omp parallel for
			for (int j = 0; j < col; j += 4)
			{
				float* dp = dest.ptr<float>(i) + j;
				__m128 mSum = _mm_loadu_ps(sum.ptr<float>(0) + j);
				for (int y = 0; y < 4; y++)
				{
					float* sp = copy.ptr<float>(y + i + r + r) + j + r;
					float* tp_prev = temp.ptr<float>(y + i - 1) + j;
					float* tp_next = temp.ptr<float>(y + i + r + r) + j;

					__m128 mRowSum = _mm_load_ps(sp);
					for (int x = 1; x <= r; x++)
					{
						mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp + x));
						mRowSum = _mm_add_ps(mRowSum, _mm_loadu_ps(sp - x));
					}
					_mm_storeu_ps(tp_next, mRowSum);

					mSum = _mm_sub_ps(mSum, _mm_loadu_ps(tp_prev));
					mSum = _mm_add_ps(mSum, mRowSum);
					_mm_storeu_ps(dp, _mm_mul_ps(mSum, mDiv));
					dp += col;
				}
				_mm_storeu_ps(sum.ptr<float>(0) + j, mSum);
			}
		}
	}*/
}

void boxFilter_SSAT_HV_4x4::operator()(const cv::Range& range) const
{

}
