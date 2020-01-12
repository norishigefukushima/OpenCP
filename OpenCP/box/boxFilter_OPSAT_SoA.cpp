#include "boxFilter_OPSAT_SoA.h"

using namespace std;
using namespace cv;

void boxFilter_OPSAT_SoA::AoS2SoA()
{
	const int loop = src.channels() >> 2;

	for (int k = 0; k < loop; k++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			float* sp = src.ptr<float>(i) + 4 * k;
			float* dp0 = vSrc[4 * k].ptr<float>(i);
			float* dp1 = vSrc[4 * k + 1].ptr<float>(i);
			float* dp2 = vSrc[4 * k + 2].ptr<float>(i);
			float* dp3 = vSrc[4 * k + 3].ptr<float>(i);

			for (int j = 0; j < src.cols; j += 4)
			{
				const __m128 mRef0 = _mm_load_ps(sp);
				const __m128 mRef1 = _mm_load_ps(sp + 4 * loop);
				const __m128 mRef2 = _mm_load_ps(sp + 8 * loop);
				const __m128 mRef3 = _mm_load_ps(sp + 12 * loop);
				sp += 16 * loop;

				const __m128 mTmp0 = _mm_unpacklo_ps(mRef0, mRef2);
				const __m128 mTmp1 = _mm_unpacklo_ps(mRef1, mRef3);
				const __m128 mTmp2 = _mm_unpackhi_ps(mRef0, mRef2);
				const __m128 mTmp3 = _mm_unpackhi_ps(mRef1, mRef3);

				const __m128 mDst0 = _mm_unpacklo_ps(mTmp0, mTmp1);
				const __m128 mDst1 = _mm_unpackhi_ps(mTmp0, mTmp1);
				const __m128 mDst2 = _mm_unpacklo_ps(mTmp2, mTmp3);
				const __m128 mDst3 = _mm_unpackhi_ps(mTmp2, mTmp3);

				_mm_stream_ps(dp0, mDst0);
				_mm_stream_ps(dp1, mDst1);
				_mm_stream_ps(dp2, mDst2);
				_mm_stream_ps(dp3, mDst3);

				dp0 += 4;
				dp1 += 4;
				dp2 += 4;
				dp3 += 4;
			}
		}
	}
}

void boxFilter_OPSAT_SoA::SoA2AoS()
{
	const int loop = static_cast<int>(vDest.size()) >> 2;

	for (int k = 0; k < loop; k++)
	{
		for (int i = 0; i < vDest[0].rows; i++)
		{
			float* sp0 = vDest[4 * k].ptr<float>(i);
			float* sp1 = vDest[4 * k + 1].ptr<float>(i);
			float* sp2 = vDest[4 * k + 2].ptr<float>(i);
			float* sp3 = vDest[4 * k + 3].ptr<float>(i);
			float* dp = dest.ptr<float>(i) + 4 * k;

			for (int j = 0; j < vDest[0].cols; j += 4)
			{
				const __m128 mRef0 = _mm_load_ps(sp0);
				const __m128 mRef1 = _mm_load_ps(sp1);
				const __m128 mRef2 = _mm_load_ps(sp2);
				const __m128 mRef3 = _mm_load_ps(sp3);
				sp0 += 4;
				sp1 += 4;
				sp2 += 4;
				sp3 += 4;

				const __m128 mTmp0 = _mm_unpacklo_ps(mRef0, mRef2);
				const __m128 mTmp1 = _mm_unpacklo_ps(mRef1, mRef3);
				const __m128 mTmp2 = _mm_unpackhi_ps(mRef0, mRef2);
				const __m128 mTmp3 = _mm_unpackhi_ps(mRef1, mRef3);

				const __m128 mDst0 = _mm_unpacklo_ps(mTmp0, mTmp1);
				const __m128 mDst1 = _mm_unpackhi_ps(mTmp0, mTmp1);
				const __m128 mDst2 = _mm_unpacklo_ps(mTmp2, mTmp3);
				const __m128 mDst3 = _mm_unpackhi_ps(mTmp2, mTmp3);

				_mm_store_ps(dp, mDst0);
				_mm_store_ps(dp + 4 * loop, mDst1);
				_mm_store_ps(dp + 8 * loop, mDst2);
				_mm_store_ps(dp + 12 * loop, mDst3);
				dp += 16 * loop;
			}
		}
	}
}

void boxFilter_OPSAT_SoA_SSE::AoS2SoA()
{
	const int loop = src.channels() >> 2;
#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j * 4;

				_mm_stream_ps(dp, _mm_load_ps(sp));
				sp += 4;
			}
		}
	}
}

void boxFilter_OPSAT_SoA_SSE::SoA2AoS()
{
	const int loop = static_cast<int>(vDest.size());
#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j * 4;

				_mm_stream_ps(dp, _mm_load_ps(sp));
				dp += 4;
			}
		}
	}
}

void boxFilter_OPSAT_SoA_AVX::AoS2SoA()
{
	const int loop = src.channels() >> 3;
#pragma omp parallel for
	for (int i = 0; i < src.rows; i++)
	{
		float* sp = src.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* dp = vSrc[k].ptr<float>(i) + j * 8;

				_mm256_stream_ps(dp, _mm256_load_ps(sp));
				sp += 8;
			}
		}
	}
}

void boxFilter_OPSAT_SoA_AVX::SoA2AoS()
{
	const int loop = static_cast<int>(vDest.size());
#pragma omp parallel for
	for (int i = 0; i < vDest[0].rows; i++)
	{
		float* dp = dest.ptr<float>(i);

		for (int j = 0; j < vDest[0].cols; j++)
		{
			for (int k = 0; k < loop; k++)
			{
				float* sp = vDest[k].ptr<float>(i) + j * 8;

				_mm256_stream_ps(dp, _mm256_load_ps(sp));
				dp += 8;
			}
		}
	}
}



void boxFilter_OPSAT_SoA::filter_impl(cv::Mat& input, cv::Mat& output)
{
	float sum = 0.f;
	Mat columnSum = Mat::zeros(1, col, CV_32FC1);

	float* dp = output.ptr<float>(0);

	float* sp_next = input.ptr<float>(0);
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = columnSum.ptr<float>(0);
	for (int j = 0; j < col; j++)
	{
		*cp_next = *sp_next * (r + 1);
		sp_next++;
		cp_next++;
	}
	cp_next -= col;
	for (int i = 1; i < r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			*cp_next += *sp_next;
			sp_next++;
			cp_next++;
		}
		cp_next -= col;
	}

	*cp_next += *sp_next;
	sum += *cp_next * (r + 1);
	sp_next++;
	cp_next++;
	for (int j = 1; j <= r; j++)
	{
		*cp_next += *sp_next;
		sum += *cp_next;
		sp_next++;
		cp_next++;
	}
	*dp = sum * div;
	dp++;
	for (int j = 1; j <= r; j++)
	{
		*cp_next += *sp_next;
		sum = sum - *cp_prev + *cp_next;
		sp_next++;
		cp_next++;
		*dp = sum * div;
		dp++;
	}
	for (int j = r + 1; j < col - r; j++)
	{
		*cp_next += *sp_next;
		sum = sum - *cp_prev + *cp_next;
		sp_next++;
		cp_prev++;
		cp_next++;
		*dp = sum * div;
		dp++;
	}
	cp_next--;
	for (int j = col - r; j < col; j++)
	{
		sum = sum - *cp_prev + *cp_next;
		cp_prev++;
		*dp = sum * div;
		dp++;
	}
	cp_prev -= (col - r - 1);
	cp_next = cp_prev;


	float* sp_prev = input.ptr<float>(0);
	/*   0 < i <= r   */
	for (int i = 1; i <= r; i++)
	{
		sum = 0.f;

		*cp_next = *cp_next - *sp_prev + *sp_next;
		sum += *cp_next * (r + 1);
		sp_prev++;
		sp_next++;
		cp_next++;
		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum += *cp_next;
			sp_prev++;
			sp_next++;
			cp_next++;
		}
		*dp = sum * div;
		dp++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev++;
			sp_next++;
			cp_next++;
			*dp = sum * div;
			dp++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev++;
			sp_next++;
			cp_prev++;
			cp_next++;
			*dp = sum * div;
			dp++;
		}
		cp_next--;
		for (int j = col - r; j < col; j++)
		{
			sum = sum - *cp_prev + *cp_next;
			cp_prev++;
			*dp = sum * div;
			dp++;
		}
		sp_prev -= col;
		cp_prev -= (col - r - 1);
		cp_next = cp_prev;
	}

	/*   r < i < row - r - 1   */
	for (int i = r + 1; i < row - r - 1; i++)
	{
		sum = 0.f;

		*cp_next = *cp_next - *sp_prev + *sp_next;
		sum += *cp_next * (r + 1);
		sp_prev++;
		sp_next++;
		cp_next++;
		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum += *cp_next;
			sp_prev++;
			sp_next++;
			cp_next++;
		}
		*dp = sum * div;
		dp++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev++;
			sp_next++;
			cp_next++;
			*dp = sum * div;
			dp++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev++;
			sp_next++;
			cp_prev++;
			cp_next++;
			*dp = sum * div;
			dp++;
		}
		cp_next--;
		for (int j = col - r; j < col; j++)
		{
			sum = sum - *cp_prev + *cp_next;
			cp_prev++;
			*dp = sum * div;
			dp++;
		}
		cp_prev -= (col - r - 1);
		cp_next = cp_prev;
	}

	/*   row - r - 1 <= i < row   */
	for (int i = row - r - 1; i < row; i++)
	{
		sum = 0.f;

		*cp_next = *cp_next - *sp_prev + *sp_next;
		sum += *cp_next * (r + 1);
		sp_prev++;
		sp_next++;
		cp_next++;
		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum += *cp_next;
			sp_prev++;
			sp_next++;
			cp_next++;
		}
		*dp = sum * div;
		dp++;

		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev++;
			sp_next++;
			cp_next++;
			*dp = sum * div;
			dp++;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev++;
			sp_next++;
			cp_prev++;
			cp_next++;
			*dp = sum * div;
			dp++;
		}
		cp_next--;
		for (int j = col - r; j < col; j++)
		{
			sum = sum - *cp_prev + *cp_next;
			cp_prev++;
			*dp = sum * div;
			dp++;
		}
		sp_next -= col;
		cp_prev -= (col - r - 1);
		cp_next = cp_prev;
	}
}



void boxFilter_OPSAT_SoA_SSE::filter_impl(cv::Mat& input, cv::Mat& output)
{
	__m128 mSum = _mm_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC4);

	float* sp_next = input.ptr<float>(0);
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m128 mCol_prev = _mm_setzero_ps();
	__m128 mCol_next = _mm_setzero_ps();
	__m128 mRef_next = _mm_setzero_ps();

	for (int j = 0; j < col; j++)
	{
		mRef_next = _mm_loadu_ps(sp_next);

		mCol_next = _mm_mul_ps(mRef_next, mBorder);
		_mm_store_ps(cp_next, mCol_next);

		sp_next += 4;
		cp_next += 4;
	}
	cp_next = cp_prev;
	for (int i = 1; i <= r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_next = _mm_load_ps(sp_next);
			mCol_next = _mm_load_ps(cp_next);

			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_store_ps(cp_next, mCol_next);

			sp_next += 4;
			cp_next += 4;
		}
		cp_next = cp_prev;
	}

	mCol_next = _mm_load_ps(cp_next);
	mSum = _mm_mul_ps(mCol_next, mBorder);
	cp_next += 4;

	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm_load_ps(cp_next);
		mSum = _mm_add_ps(mSum, mCol_next);
		cp_next += 4;
	}

	float* dp = output.ptr<float>(0);
	__m128 mDest = _mm_setzero_ps();

	mDest = _mm_mul_ps(mSum, mDiv);
	_mm_store_ps(dp, mDest);
	dp += 4;

	mCol_prev = _mm_load_ps(cp_prev);
	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm_load_ps(cp_next);

		mSum = _mm_sub_ps(mSum, mCol_prev);
		mSum = _mm_add_ps(mSum, mCol_next);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_store_ps(dp, mDest);

		cp_next += 4;
		dp += 4;
	}
	for (int j = r + 1; j < col - r; j++)
	{
		mCol_prev = _mm_load_ps(cp_prev);
		mCol_next = _mm_load_ps(cp_next);

		mSum = _mm_sub_ps(mSum, mCol_prev);
		mSum = _mm_add_ps(mSum, mCol_next);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_store_ps(dp, mDest);

		cp_prev += 4;
		cp_next += 4;
		dp += 4;
	}
	cp_next -= 4;
	mCol_next = _mm_load_ps(cp_next);
	for (int j = col - r; j < col; j++)
	{
		mCol_prev = _mm_load_ps(cp_prev);

		mSum = _mm_sub_ps(mSum, mCol_prev);
		mSum = _mm_add_ps(mSum, mCol_next);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_store_ps(dp, mDest);

		cp_prev += 4;
		dp += 4;
	}
	cp_prev = cp_next = columnSum.ptr<float>(0);


	float* sp_prev = input.ptr<float>(0);
	__m128 mRef_prev = _mm_setzero_ps();
	/*   0 < i <= r   */
	for (int i = 1; i <= r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_prev = _mm_load_ps(sp_prev);
			mRef_next = _mm_load_ps(sp_next);
			mCol_next = _mm_load_ps(cp_next);

			mCol_next = _mm_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_store_ps(cp_next, mCol_next);

			sp_prev += 4;
			sp_next += 4;
			cp_next += 4;
		}
		sp_prev = input.ptr<float>(0);
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm_setzero_ps();

		mCol_next = _mm_load_ps(cp_next);
		mSum = _mm_mul_ps(mCol_next, mBorder);
		cp_next += 4;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);
			mSum = _mm_add_ps(mSum, mCol_next);
			cp_next += 4;
		}
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_store_ps(dp, mDest);
		dp += 4;

		mCol_prev = _mm_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_next += 4;
			dp += 4;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm_load_ps(cp_prev);
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_prev += 4;
			cp_next += 4;
			dp += 4;
		}
		cp_next -= 4;
		mCol_next = _mm_load_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm_load_ps(cp_prev);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_prev += 4;
			dp += 4;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
	/*   r < i < row - r - 1   */
	for (int i = r + 1; i < row - r - 1; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_prev = _mm_load_ps(sp_prev);
			mRef_next = _mm_load_ps(sp_next);
			mCol_next = _mm_load_ps(cp_next);

			mCol_next = _mm_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_store_ps(cp_next, mCol_next);

			sp_prev += 4;
			sp_next += 4;
			cp_next += 4;
		}
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm_setzero_ps();

		mCol_next = _mm_load_ps(cp_next);
		mSum = _mm_mul_ps(mCol_next, mBorder);
		cp_next += 4;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);
			mSum = _mm_add_ps(mSum, mCol_next);
			cp_next += 4;
		}
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_store_ps(dp, mDest);
		dp += 4;

		mCol_prev = _mm_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_next += 4;
			dp += 4;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm_load_ps(cp_prev);
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_prev += 4;
			cp_next += 4;
			dp += 4;
		}
		cp_next -= 4;
		mCol_next = _mm_load_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm_load_ps(cp_prev);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_prev += 4;
			dp += 4;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}

	/*   row - r - 1 <= i < row   */
	for (int i = row - r - 1; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_prev = _mm_load_ps(sp_prev);
			mRef_next = _mm_load_ps(sp_next);
			mCol_next = _mm_load_ps(cp_next);

			mCol_next = _mm_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_store_ps(cp_next, mCol_next);

			sp_prev += 4;
			sp_next += 4;
			cp_next += 4;
		}
		sp_next = input.ptr<float>(row - 1);
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm_setzero_ps();

		mCol_next = _mm_load_ps(cp_next);
		mSum = _mm_mul_ps(mCol_next, mBorder);
		cp_next += 4;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);
			mSum = _mm_add_ps(mSum, mCol_next);
			cp_next += 4;
		}
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_store_ps(dp, mDest);
		dp += 4;

		mCol_prev = _mm_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_next += 4;
			dp += 4;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm_load_ps(cp_prev);
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_prev += 4;
			cp_next += 4;
			dp += 4;
		}
		cp_next -= 4;
		mCol_next = _mm_load_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm_load_ps(cp_prev);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_prev += 4;
			dp += 4;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
}



void boxFilter_OPSAT_SoA_AVX::filter_impl(cv::Mat& input, cv::Mat& output)
{
	__m256 mSum = _mm256_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(8));

	float* sp_next = input.ptr<float>(0);
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m256 mCol_prev = _mm256_setzero_ps();
	__m256 mCol_next = _mm256_setzero_ps();
	__m256 mRef_next = _mm256_setzero_ps();

	for (int j = 0; j < col; j++)
	{
		mRef_next = _mm256_load_ps(sp_next);

		mCol_next = _mm256_mul_ps(mRef_next, mBorder);
		_mm256_store_ps(cp_next, mCol_next);

		sp_next += 8;
		cp_next += 8;
	}
	cp_next = cp_prev;
	for (int i = 1; i <= r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_next = _mm256_load_ps(sp_next);
			mCol_next = _mm256_load_ps(cp_next);

			mCol_next = _mm256_add_ps(mCol_next, mRef_next);
			_mm256_store_ps(cp_next, mCol_next);

			sp_next += 8;
			cp_next += 8;
		}
		cp_next = cp_prev;
	}

	mCol_next = _mm256_load_ps(cp_next);
	mSum = _mm256_mul_ps(mCol_next, mBorder);
	cp_next += 8;

	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm256_load_ps(cp_next);
		mSum = _mm256_add_ps(mSum, mCol_next);
		cp_next += 8;
	}

	float* dp = output.ptr<float>(0);
	__m256 mDest = _mm256_setzero_ps();

	mDest = _mm256_mul_ps(mSum, mDiv);
	_mm256_store_ps(dp, mDest);
	dp += 8;

	mCol_prev = _mm256_load_ps(cp_prev);
	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm256_load_ps(cp_next);

		mSum = _mm256_sub_ps(mSum, mCol_prev);
		mSum = _mm256_add_ps(mSum, mCol_next);
		mDest = _mm256_mul_ps(mSum, mDiv);
		_mm256_store_ps(dp, mDest);

		cp_next += 8;
		dp += 8;
	}
	for (int j = r + 1; j < col - r; j++)
	{
		mCol_prev = _mm256_load_ps(cp_prev);
		mCol_next = _mm256_load_ps(cp_next);

		mSum = _mm256_sub_ps(mSum, mCol_prev);
		mSum = _mm256_add_ps(mSum, mCol_next);
		mDest = _mm256_mul_ps(mSum, mDiv);
		_mm256_store_ps(dp, mDest);

		cp_prev += 8;
		cp_next += 8;
		dp += 8;
	}
	cp_next -= 8;
	mCol_next = _mm256_load_ps(cp_next);
	for (int j = col - r; j < col; j++)
	{
		mCol_prev = _mm256_load_ps(cp_prev);

		mSum = _mm256_sub_ps(mSum, mCol_prev);
		mSum = _mm256_add_ps(mSum, mCol_next);
		mDest = _mm256_mul_ps(mSum, mDiv);
		_mm256_store_ps(dp, mDest);

		cp_prev += 8;
		dp += 8;
	}
	cp_prev = cp_next = columnSum.ptr<float>(0);


	float* sp_prev = input.ptr<float>(0);
	__m256 mRef_prev = _mm256_setzero_ps();
	/*   0 < i <= r   */
	for (int i = 1; i <= r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_prev = _mm256_load_ps(sp_prev);
			mRef_next = _mm256_load_ps(sp_next);
			mCol_next = _mm256_load_ps(cp_next);

			mCol_next = _mm256_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm256_add_ps(mCol_next, mRef_next);
			_mm256_store_ps(cp_next, mCol_next);

			sp_prev += 8;
			sp_next += 8;
			cp_next += 8;
		}
		sp_prev = input.ptr<float>(0);
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm256_setzero_ps();

		mCol_next = _mm256_load_ps(cp_next);
		mSum = _mm256_mul_ps(mCol_next, mBorder);
		cp_next += 8;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);
			mSum = _mm256_add_ps(mSum, mCol_next);
			cp_next += 8;
		}
		mDest = _mm256_mul_ps(mSum, mDiv);
		_mm256_store_ps(dp, mDest);
		dp += 8;

		mCol_prev = _mm256_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_next += 8;
			dp += 8;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm256_load_ps(cp_prev);
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_prev += 8;
			cp_next += 8;
			dp += 8;
		}
		cp_next -= 8;
		mCol_next = _mm256_load_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm256_load_ps(cp_prev);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_prev += 8;
			dp += 8;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
	/*   r < i < row - r - 1   */
	for (int i = r + 1; i < row - r - 1; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_prev = _mm256_load_ps(sp_prev);
			mRef_next = _mm256_load_ps(sp_next);
			mCol_next = _mm256_load_ps(cp_next);

			mCol_next = _mm256_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm256_add_ps(mCol_next, mRef_next);
			_mm256_store_ps(cp_next, mCol_next);

			sp_prev += 8;
			sp_next += 8;
			cp_next += 8;
		}
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm256_setzero_ps();

		mCol_next = _mm256_load_ps(cp_next);
		mSum = _mm256_mul_ps(mCol_next, mBorder);
		cp_next += 8;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);
			mSum = _mm256_add_ps(mSum, mCol_next);
			cp_next += 8;
		}
		mDest = _mm256_mul_ps(mSum, mDiv);
		_mm256_store_ps(dp, mDest);
		dp += 8;

		mCol_prev = _mm256_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_next += 8;
			dp += 8;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm256_load_ps(cp_prev);
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_prev += 8;
			cp_next += 8;
			dp += 8;
		}
		cp_next -= 8;
		mCol_next = _mm256_load_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm256_load_ps(cp_prev);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_prev += 8;
			dp += 8;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}

	/*   row - r - 1 <= i < row   */
	for (int i = row - r - 1; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			mRef_prev = _mm256_load_ps(sp_prev);
			mRef_next = _mm256_load_ps(sp_next);
			mCol_next = _mm256_load_ps(cp_next);

			mCol_next = _mm256_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm256_add_ps(mCol_next, mRef_next);
			_mm256_store_ps(cp_next, mCol_next);

			sp_prev += 8;
			sp_next += 8;
			cp_next += 8;
		}
		sp_next = input.ptr<float>(row - 1);
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm256_setzero_ps();

		mCol_next = _mm256_load_ps(cp_next);
		mSum = _mm256_mul_ps(mCol_next, mBorder);
		cp_next += 8;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);
			mSum = _mm256_add_ps(mSum, mCol_next);
			cp_next += 8;
		}
		mDest = _mm256_mul_ps(mSum, mDiv);
		_mm256_store_ps(dp, mDest);
		dp += 8;

		mCol_prev = _mm256_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_next += 8;
			dp += 8;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm256_load_ps(cp_prev);
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_prev += 8;
			cp_next += 8;
			dp += 8;
		}
		cp_next -= 8;
		mCol_next = _mm256_load_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm256_load_ps(cp_prev);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_prev += 8;
			dp += 8;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
}
