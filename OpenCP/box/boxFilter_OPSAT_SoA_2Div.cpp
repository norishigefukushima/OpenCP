#include "boxFilter_OPSAT_SoA_2Div.h"

using namespace std;
using namespace cv;

boxFilter_OPSAT_SoA_2Div::boxFilter_OPSAT_SoA_2Div(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	div = 1.f / ((2 * r + 1)*(2 * r + 1));
	row = src.rows;
	col = src.cols;
	cn = src.channels();

	init();
}

void boxFilter_OPSAT_SoA_2Div::init()
{
	loop = cn;
	divNum = 2;
	divRow = row / divNum;

	vSrc.resize(loop);
	vDest.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC1);
		vDest[i].create(src.size(), CV_32FC1);
	}
}

void boxFilter_OPSAT_SoA_2Div::AoS2SoA()
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

void boxFilter_OPSAT_SoA_2Div::SoA2AoS()
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

void boxFilter_OPSAT_SoA_2Div::filter_upper_impl(cv::Mat& input, cv::Mat& output)
{
	float sum = 0.f;
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC1);

	float* dp = output.ptr<float>(0);

	float* sp_next = input.ptr<float>(0);
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
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

	/*   r < i < row / 2   */
	for (int i = r + 1; i < divRow; i++)
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
}

void boxFilter_OPSAT_SoA_2Div::filter_lower_impl(cv::Mat& input, cv::Mat& output)
{
	int rowOffset = (divNum - 1) * divRow;

	float sum = 0.f;
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC1);

	float* dp = output.ptr<float>(rowOffset);

	float* sp_next = input.ptr<float>(rowOffset - r);
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	for (int i = -r; i < r; i++)
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


	float* sp_prev = input.ptr<float>(rowOffset - r);
	/*   row / 2 < i < row - r - 1   */
	for (int i = rowOffset + 1; i < row - r - 1; i++)
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

void boxFilter_OPSAT_SoA_2Div::filter_impl(int idx)
{
	if (idx < loop)
		filter_upper_impl(vSrc[idx], vDest[idx]);
	else
		filter_lower_impl(vSrc[idx - loop], vDest[idx - loop]);
}

void boxFilter_OPSAT_SoA_2Div::filterOnly()
{
	if (parallelType == NAIVE)
	{
		for (int idx = 0; idx < divNum * loop; idx++)
		{
			filter_impl(idx);
		}
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int idx = 0; idx < divNum * loop; idx++)
		{
			filter_impl(idx);
		}
	}
}

void boxFilter_OPSAT_SoA_2Div::filter()
{
	AoS2SoA();
	filterOnly();
	SoA2AoS();
}



boxFilter_OPSAT_SoA_nDiv::boxFilter_OPSAT_SoA_nDiv(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num)
	: boxFilter_OPSAT_SoA_2Div(_src, _dest, _r, _parallelType), num(_num)
{
	init();
}

void boxFilter_OPSAT_SoA_nDiv::init()
{
	loop = cn;
	divNum = num;
	divRow = row / divNum;

	vSrc.resize(loop);
	vDest.resize(loop);
	for (int i = 0; i < loop; i++)
	{
		vSrc[i].create(src.size(), CV_32FC1);
		vDest[i].create(src.size(), CV_32FC1);
	}
}

void boxFilter_OPSAT_SoA_nDiv::filter_middle_impl(cv::Mat& input, cv::Mat& output, int idx)
{
	int rowOffset = idx * divRow;

	float sum = 0.f;
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC1);

	float* dp = output.ptr<float>(rowOffset);

	float* sp_next = input.ptr<float>(rowOffset - r);
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	for (int i = -r; i < r; i++)
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


	float* sp_prev = input.ptr<float>(rowOffset - r);
	/*   row / 2 < i < row - r - 1   */
	for (int i = rowOffset + 1; i < rowOffset + divRow; i++)
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
}

void boxFilter_OPSAT_SoA_nDiv::filter_impl(int idx)
{
	if (idx < cn)
		filter_upper_impl(vSrc[idx], vDest[idx]);
	else if (idx + cn >= divNum * cn)
		filter_lower_impl(vSrc[idx % cn], vDest[idx % cn]);
	else
		filter_middle_impl(vSrc[idx % cn], vDest[idx % cn], idx / cn);
}
