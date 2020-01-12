#include "boxFilter_OPSAT_AoS.h"

using namespace std;
using namespace cv;

void boxFilter_OPSAT_AoS::filter_impl(int cnNum)
{
	float sum = 0.f;
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC1);

	float* dp = dest.ptr<float>(0) + cnNum;

	float* sp_next = src.ptr<float>(0) + cnNum;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	for (int j = 0; j < col; j++)
	{
		*cp_next = *sp_next * (r + 1);
		sp_next += cn;
		cp_next++;
	}
	cp_next -= col;
	for (int i = 1; i < r; i++)
	{
		for (int j = 0; j < col; j++)
		{
			*cp_next += *sp_next;
			sp_next += cn;
			cp_next++;
		}
		cp_next -= col;
	}

	*cp_next += *sp_next;
	sum += *cp_next * (r + 1);
	sp_next += cn;
	cp_next++;
	for (int j = 1; j <= r; j++)
	{
		*cp_next += *sp_next;
		sum += *cp_next;
		sp_next += cn;
		cp_next++;
	}
	*dp = sum * div;
	dp += cn;
	for (int j = 1; j <= r; j++)
	{
		*cp_next += *sp_next;
		sum = sum - *cp_prev + *cp_next;
		sp_next += cn;
		cp_next++;
		*dp = sum * div;
		dp += cn;
	}
	for (int j = r + 1; j < col - r; j++)
	{
		*cp_next += *sp_next;
		sum = sum - *cp_prev + *cp_next;
		sp_next += cn;
		cp_prev++;
		cp_next++;
		*dp = sum * div;
		dp += cn;
	}
	cp_next--;
	for (int j = col - r; j < col; j++)
	{
		sum = sum - *cp_prev + *cp_next;
		cp_prev++;
		*dp = sum * div;
		dp += cn;
	}
	cp_prev -= (col - r - 1);
	cp_next = cp_prev;


	float* sp_prev = src.ptr<float>(0) + cnNum;
	/*   0 < i <= r   */
	for (int i = 1; i <= r; i++)
	{
		sum = 0.f;

		*cp_next = *cp_next - *sp_prev + *sp_next;
		sum += *cp_next * (r + 1);
		sp_prev += cn;
		sp_next += cn;
		cp_next++;
		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum += *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_next++;
		}
		*dp = sum * div;
		dp += cn;

		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_next++;
			*dp = sum * div;
			dp += cn;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_prev++;
			cp_next++;
			*dp = sum * div;
			dp += cn;
		}
		cp_next--;
		for (int j = col - r; j < col; j++)
		{
			sum = sum - *cp_prev + *cp_next;
			cp_prev++;
			*dp = sum * div;
			dp += cn;
		}
		sp_prev -= col*cn;
		cp_prev -= (col - r - 1);
		cp_next = cp_prev;
	}

	/*   r < i < row - r - 1   */
	for (int i = r + 1; i < row - r - 1; i++)
	{
		sum = 0.f;

		*cp_next = *cp_next - *sp_prev + *sp_next;
		sum += *cp_next * (r + 1);
		sp_prev += cn;
		sp_next += cn;
		cp_next++;
		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum += *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_next++;
		}
		*dp = sum * div;
		dp += cn;

		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_next++;
			*dp = sum * div;
			dp += cn;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_prev++;
			cp_next++;
			*dp = sum * div;
			dp += cn;
		}
		cp_next--;
		for (int j = col - r; j < col; j++)
		{
			sum = sum - *cp_prev + *cp_next;
			cp_prev++;
			*dp = sum * div;
			dp += cn;
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
		sp_prev += cn;
		sp_next += cn;
		cp_next++;
		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum += *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_next++;
		}
		*dp = sum * div;
		dp += cn;

		for (int j = 1; j <= r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_next++;
			*dp = sum * div;
			dp += cn;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			*cp_next = *cp_next - *sp_prev + *sp_next;
			sum = sum - *cp_prev + *cp_next;
			sp_prev += cn;
			sp_next += cn;
			cp_prev++;
			cp_next++;
			*dp = sum * div;
			dp += cn;
		}
		cp_next--;
		for (int j = col - r; j < col; j++)
		{
			sum = sum - *cp_prev + *cp_next;
			cp_prev++;
			*dp = sum * div;
			dp += cn;
		}
		sp_next -= col*cn;
		cp_prev -= (col - r - 1);
		cp_next = cp_prev;
	}
}

void boxFilter_OPSAT_AoS_SSE::filter_impl(int cnNum)
{
	__m128 mSum = _mm_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC4);

	float* sp_next = src.ptr<float>(0) + cnNum * 4;
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

		sp_next += cn;
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

			sp_next += cn;
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

	float* dp = dest.ptr<float>(0) + cnNum * 4;
	__m128 mDest = _mm_setzero_ps();

	mDest = _mm_mul_ps(mSum, mDiv);
	_mm_store_ps(dp, mDest);
	dp += cn;

	mCol_prev = _mm_load_ps(cp_prev);
	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm_load_ps(cp_next);

		mSum = _mm_sub_ps(mSum, mCol_prev);
		mSum = _mm_add_ps(mSum, mCol_next);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_store_ps(dp, mDest);

		cp_next += 4;
		dp += cn;
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
		dp += cn;
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
		dp += cn;
	}
	cp_prev = cp_next = columnSum.ptr<float>(0);


	float* sp_prev = src.ptr<float>(0) + cnNum * 4;
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

			sp_prev += cn;
			sp_next += cn;
			cp_next += 4;
		}
		sp_prev = src.ptr<float>(0) + cnNum * 4;
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
		dp += cn;

		mCol_prev = _mm_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_next += 4;
			dp += cn;
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
			dp += cn;
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
			dp += cn;
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

			sp_prev += cn;
			sp_next += cn;
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
		dp += cn;

		mCol_prev = _mm_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_next += 4;
			dp += cn;
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
			dp += cn;
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
			dp += cn;
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

			sp_prev += cn;
			sp_next += cn;
			cp_next += 4;
		}
		sp_next = src.ptr<float>(row - 1) + cnNum * 4;
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
		dp += cn;

		mCol_prev = _mm_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_load_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_store_ps(dp, mDest);

			cp_next += 4;
			dp += cn;
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
			dp += cn;
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
			dp += cn;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
}

void boxFilter_OPSAT_AoS_AVX::filter_impl(int cnNum)
{
	__m256 mSum = _mm256_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(8));

	float* sp_next = src.ptr<float>(0) + cnNum * 8;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m256 mCol_prev = _mm256_setzero_ps();
	__m256 mCol_next = _mm256_setzero_ps();
	__m256 mRef_next = _mm256_setzero_ps();

	for (int j = 0; j < col; j++)
	{
		mRef_next = _mm256_loadu_ps(sp_next);

		mCol_next = _mm256_mul_ps(mRef_next, mBorder);
		_mm256_store_ps(cp_next, mCol_next);

		sp_next += cn;
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

			sp_next += cn;
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

	float* dp = dest.ptr<float>(0) + cnNum * 8;
	__m256 mDest = _mm256_setzero_ps();

	mDest = _mm256_mul_ps(mSum, mDiv);
	_mm256_store_ps(dp, mDest);
	dp += cn;

	mCol_prev = _mm256_load_ps(cp_prev);
	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm256_load_ps(cp_next);

		mSum = _mm256_sub_ps(mSum, mCol_prev);
		mSum = _mm256_add_ps(mSum, mCol_next);
		mDest = _mm256_mul_ps(mSum, mDiv);
		_mm256_store_ps(dp, mDest);

		cp_next += 8;
		dp += cn;
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
		dp += cn;
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
		dp += cn;
	}
	cp_prev = cp_next = columnSum.ptr<float>(0);


	float* sp_prev = src.ptr<float>(0) + cnNum * 8;
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

			sp_prev += cn;
			sp_next += cn;
			cp_next += 8;
		}
		sp_prev = src.ptr<float>(0) + cnNum * 8;
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
		dp += cn;

		mCol_prev = _mm256_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_next += 8;
			dp += cn;
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
			dp += cn;
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
			dp += cn;
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

			sp_prev += cn;
			sp_next += cn;
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
		dp += cn;

		mCol_prev = _mm256_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_next += 8;
			dp += cn;
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
			dp += cn;
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
			dp += cn;
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

			sp_prev += cn;
			sp_next += cn;
			cp_next += 8;
		}
		sp_next = src.ptr<float>(row - 1) + cnNum * 8;
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
		dp += cn;

		mCol_prev = _mm256_load_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm256_load_ps(cp_next);

			mSum = _mm256_sub_ps(mSum, mCol_prev);
			mSum = _mm256_add_ps(mSum, mCol_next);
			mDest = _mm256_mul_ps(mSum, mDiv);
			_mm256_store_ps(dp, mDest);

			cp_next += 8;
			dp += cn;
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
			dp += cn;
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
			dp += cn;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
}




void boxFilter_OPSAT_BGR::filter_impl()
{
	if (cn == 1)
	{
		src.copyTo(dest);
		return;
	}

	__m128 mSum = _mm_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col + 1, 1), CV_32FC3);

	float* sp_next = src.ptr<float>(0);
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m128 mCol_prev = _mm_setzero_ps();
	__m128 mCol_next = _mm_setzero_ps();
	__m128 mRef_next = _mm_setzero_ps();

	for (int j = 0; j < col * cn; j += 4)
	{
		mRef_next = _mm_loadu_ps(sp_next);

		mCol_next = _mm_mul_ps(mRef_next, mBorder);
		_mm_storeu_ps(cp_next, mCol_next);

		sp_next += 4;
		cp_next += 4;
	}
	cp_next = cp_prev;
	for (int i = 1; i <= r; i++)
	{
		sp_next = src.ptr<float>(i);
		for (int j = 0; j < col * cn; j += 4)
		{
			mRef_next = _mm_loadu_ps(sp_next);
			mCol_next = _mm_loadu_ps(cp_next);

			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_storeu_ps(cp_next, mCol_next);

			sp_next += 4;
			cp_next += 4;
		}
		cp_next = cp_prev;
	}

	mCol_next = _mm_loadu_ps(cp_next);
	mSum = _mm_mul_ps(mCol_next, mBorder);
	cp_next += cn;

	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm_loadu_ps(cp_next);
		mSum = _mm_add_ps(mSum, mCol_next);
		cp_next += cn;
	}

	float* dp = temp.ptr<float>(0);
	__m128 mDest = _mm_setzero_ps();

	mDest = _mm_mul_ps(mSum, mDiv);
	_mm_storeu_ps(dp, mDest);
	dp += cn;

	mCol_prev = _mm_loadu_ps(cp_prev);
	for (int j = 1; j <= r; j++)
	{
		mCol_next = _mm_loadu_ps(cp_next);

		mSum = _mm_sub_ps(mSum, mCol_prev);
		mSum = _mm_add_ps(mSum, mCol_next);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_storeu_ps(dp, mDest);

		cp_next += cn;
		dp += cn;
	}
	for (int j = r + 1; j < col - r; j++)
	{
		mCol_prev = _mm_loadu_ps(cp_prev);
		mCol_next = _mm_loadu_ps(cp_next);

		mSum = _mm_sub_ps(mSum, mCol_prev);
		mSum = _mm_add_ps(mSum, mCol_next);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_storeu_ps(dp, mDest);

		cp_prev += cn;
		cp_next += cn;
		dp += cn;
	}
	cp_next -= cn;
	mCol_next = _mm_loadu_ps(cp_next);
	for (int j = col - r; j < col; j++)
	{
		mCol_prev = _mm_loadu_ps(cp_prev);

		mSum = _mm_sub_ps(mSum, mCol_prev);
		mSum = _mm_add_ps(mSum, mCol_next);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_storeu_ps(dp, mDest);

		cp_prev += cn;
		dp += cn;
	}
	cp_prev = cp_next = columnSum.ptr<float>(0);


	float* sp_prev = src.ptr<float>(0);
	__m128 mRef_prev = _mm_setzero_ps();
	/*   0 < i <= r   */
	for (int i = 1; i <= r; i++)
	{
		sp_next = src.ptr<float>(i + r);
		for (int j = 0; j < col * cn; j += 4)
		{
			mRef_prev = _mm_loadu_ps(sp_prev);
			mRef_next = _mm_loadu_ps(sp_next);
			mCol_next = _mm_loadu_ps(cp_next);

			mCol_next = _mm_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_store_ps(cp_next, mCol_next);

			sp_prev += 4;
			sp_next += 4;
			cp_next += 4;
		}
		sp_prev = src.ptr<float>(0);
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm_setzero_ps();

		mCol_next = _mm_loadu_ps(cp_next);
		mSum = _mm_mul_ps(mCol_next, mBorder);
		cp_next += cn;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);
			mSum = _mm_add_ps(mSum, mCol_next);
			cp_next += cn;
		}
		dp = temp.ptr<float>(i);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_storeu_ps(dp, mDest);
		dp += cn;

		mCol_prev = _mm_loadu_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);
			
			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_next += cn;
			dp += cn;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm_loadu_ps(cp_prev);
			mCol_next = _mm_loadu_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_prev += cn;
			cp_next += cn;
			dp += cn;
		}
		cp_next -= cn;
		mCol_next = _mm_loadu_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm_loadu_ps(cp_prev);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_prev += cn;
			dp += cn;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
	/*   r < i < row - r - 1   */
	for (int i = r + 1; i < row - r - 1; i++)
	{
		sp_prev = src.ptr<float>(i - r - 1);
		sp_next = src.ptr<float>(i + r);
		for (int j = 0; j < col * cn; j += 4)
		{
			mRef_prev = _mm_loadu_ps(sp_prev);
			mRef_next = _mm_loadu_ps(sp_next);
			mCol_next = _mm_loadu_ps(cp_next);

			mCol_next = _mm_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_store_ps(cp_next, mCol_next);

			sp_prev += 4;
			sp_next += 4;
			cp_next += 4;
		}
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm_setzero_ps();

		mCol_next = _mm_loadu_ps(cp_next);
		mSum = _mm_mul_ps(mCol_next, mBorder);
		cp_next += cn;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);
			mSum = _mm_add_ps(mSum, mCol_next);
			cp_next += cn;
		}
		dp = temp.ptr<float>(i);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_storeu_ps(dp, mDest);
		dp += cn;

		mCol_prev = _mm_loadu_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_next += cn;
			dp += cn;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm_loadu_ps(cp_prev);
			mCol_next = _mm_loadu_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_prev += cn;
			cp_next += cn;
			dp += cn;
		}
		cp_next -= cn;
		mCol_next = _mm_loadu_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm_loadu_ps(cp_prev);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_prev += cn;
			dp += cn;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}

	/*   row - r - 1 <= i < row   */
	for (int i = row - r - 1; i < row; i++)
	{
		sp_prev = src.ptr<float>(i - r - 1);
		sp_next = src.ptr<float>(row - 1);
		for (int j = 0; j < col * cn; j += 4)
		{
			mRef_prev = _mm_loadu_ps(sp_prev);
			mRef_next = _mm_loadu_ps(sp_next);
			mCol_next = _mm_loadu_ps(cp_next);

			mCol_next = _mm_sub_ps(mCol_next, mRef_prev);
			mCol_next = _mm_add_ps(mCol_next, mRef_next);
			_mm_store_ps(cp_next, mCol_next);

			sp_prev += 4;
			sp_next += 4;
			cp_next += 4;
		}
		cp_next = columnSum.ptr<float>(0);

		mSum = _mm_setzero_ps();

		mCol_next = _mm_loadu_ps(cp_next);
		mSum = _mm_mul_ps(mCol_next, mBorder);
		cp_next += cn;

		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);
			mSum = _mm_add_ps(mSum, mCol_next);
			cp_next += cn;
		}
		dp = temp.ptr<float>(i);
		mDest = _mm_mul_ps(mSum, mDiv);
		_mm_storeu_ps(dp, mDest);
		dp += cn;

		mCol_prev = _mm_loadu_ps(cp_prev);
		for (int j = 1; j <= r; j++)
		{
			mCol_next = _mm_loadu_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_next += cn;
			dp += cn;
		}
		for (int j = r + 1; j < col - r; j++)
		{
			mCol_prev = _mm_loadu_ps(cp_prev);
			mCol_next = _mm_loadu_ps(cp_next);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_prev += cn;
			cp_next += cn;
			dp += cn;
		}
		cp_next -= cn;
		mCol_next = _mm_loadu_ps(cp_next);
		for (int j = col - r; j < col; j++)
		{
			mCol_prev = _mm_loadu_ps(cp_prev);

			mSum = _mm_sub_ps(mSum, mCol_prev);
			mSum = _mm_add_ps(mSum, mCol_next);
			mDest = _mm_mul_ps(mSum, mDiv);
			_mm_storeu_ps(dp, mDest);

			cp_prev += cn;
			dp += cn;
		}
		cp_prev = cp_next = columnSum.ptr<float>(0);
	}
	Mat temp_roi = temp(Rect(0, 0, col, row));
	temp_roi.copyTo(dest);
}


void boxFilter_OPSAT_BGRA::filter_impl()
{
	if (cn == 1)
	{
		src.copyTo(dest);
		return;
	}

	cvtColor(src, srcBGRA, COLOR_BGR2BGRA);

	__m128 mSum = _mm_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC4);

	float* sp_next = srcBGRA.ptr<float>(0);
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

	float* dp = destBGRA.ptr<float>(0);
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


	float* sp_prev = srcBGRA.ptr<float>(0);
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
		sp_prev = srcBGRA.ptr<float>(0);
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
		sp_next = srcBGRA.ptr<float>(row - 1);
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

	cvtColor(destBGRA, dest, COLOR_BGRA2BGR);
}
