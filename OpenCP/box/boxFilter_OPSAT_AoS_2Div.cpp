#include "boxFilter_OPSAT_AoS_2Div.h"

using namespace std;
using namespace cv;

//2Div nonVec

boxFilter_OPSAT_AoS_2Div::boxFilter_OPSAT_AoS_2Div(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	div = 1.f / ((2 * r + 1)*(2 * r + 1));
	row = src.rows;
	col = src.cols;
	cn = src.channels();

	init();
}

void boxFilter_OPSAT_AoS_2Div::init()
{
	divNum = 2;
	divRow = row / divNum;
	cnNum = cn;
}

void boxFilter_OPSAT_AoS_2Div::filter_upper_impl(int cnNum)
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
		sp_prev -= col * cn;
		cp_prev -= (col - r - 1);
		cp_next = cp_prev;
	}

	/*   r < i < row / 2   */
	for (int i = r + 1; i < divRow; i++)
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
}

void boxFilter_OPSAT_AoS_2Div::filter_lower_impl(int cnNum)
{
	int rowOffset = (divNum - 1) * divRow;

	float sum = 0.f;
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC1);

	float* dp = dest.ptr<float>(rowOffset) + cnNum;

	float* sp_next = src.ptr<float>(rowOffset - r) + cnNum;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	for (int i = -r; i < r; i++)
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


	float* sp_prev = src.ptr<float>(rowOffset - r) + cnNum;
	/*   row / 2 < i < row - r - 1   */
	for (int i = rowOffset + 1; i < row - r - 1; i++)
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
		sp_next -= col * cn;
		cp_prev -= (col - r - 1);
		cp_next = cp_prev;
	}
}

void boxFilter_OPSAT_AoS_2Div::filter()
{
	//	if (parallelType == NAIVE)
	//	{
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//		}
	//	}
	//	else if (parallelType == OMP)
	//	{
	//#pragma omp parallel for
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//		}
	//	}

	const int loop = cnNum * divNum;
	if (parallelType == NAIVE)
	{
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else
				filter_lower_impl(idx - cnNum);
		}
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else
				filter_lower_impl(idx - cnNum);
		}
	}
}



/*
 *  nDiv nonVec
 */
boxFilter_OPSAT_AoS_nDiv::boxFilter_OPSAT_AoS_nDiv(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num)
	: boxFilter_OPSAT_AoS_2Div(_src, _dest, _r, _parallelType), num(_num)
{
	init();
}

void boxFilter_OPSAT_AoS_nDiv::init()
{
	divNum = num;
	divRow = row / divNum;
	cnNum = cn;
}

void boxFilter_OPSAT_AoS_nDiv::filter_middle_impl(int cnNum, int idx)
{
	int rowOffset = idx * divRow;

	float sum = 0.f;
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC1);

	float* dp = dest.ptr<float>(rowOffset) + cnNum;

	float* sp_next = src.ptr<float>(rowOffset - r) + cnNum;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	for (int i = -r; i < r; i++)
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


	float* sp_prev = src.ptr<float>(rowOffset - r) + cnNum;
	/*   row / 2 < i < row - r - 1   */
	for (int i = rowOffset + 1; i < rowOffset + divRow; i++)
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
}

void boxFilter_OPSAT_AoS_nDiv::filter()
{
	//	if (parallelType == NAIVE)
	//	{
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else if (idxDiv == divNum - 1)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_middle_impl(idxCn, idxDiv);
	//		}
	//	}
	//	else if (parallelType == OMP)
	//	{
	//#pragma omp parallel for
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else if (idxDiv == divNum - 1)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_middle_impl(idxCn, idxDiv);
	//		}
	//	}
	//	else if (parallelType == PARALLEL_FOR_)
	//	{
	//		if (num == 2)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 4; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4; idxCn < cnNum / 2; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum / 4 * 3; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4 * 3; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 4; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4; idxCn < cnNum / 2; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum / 4 * 3; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4 * 3; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//			}
	//		}
	//		else if (num == 4)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//			}
	//		}
	//		else if (num == 8)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 3);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 4);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 5);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 6);
	//				}
	//			}
	//		}
	//		else
	//		{
	//		}
	//	}

	const int loop = divNum * cnNum;
	if (parallelType == NAIVE)
	{
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else if (idx + cnNum >= loop)
				filter_lower_impl(idx % cnNum);
			else
				filter_middle_impl(idx % cnNum, idx / cnNum);
		}
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else if (idx + cnNum >= loop)
				filter_lower_impl(idx % cnNum);
			else
				filter_middle_impl(idx % cnNum, idx / cnNum);
		}
	}
}



/*
 * 2Div SSE
 */
boxFilter_OPSAT_AoS_2Div_SSE::boxFilter_OPSAT_AoS_2Div_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_OPSAT_AoS_2Div(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_OPSAT_AoS_2Div_SSE::init()
{
	mDiv = _mm_set1_ps(div);
	mBorder = _mm_set1_ps(static_cast<float>(r + 1));

	divNum = 2;
	divRow = row / divNum;
	cnNum = cn / 4;
}

void boxFilter_OPSAT_AoS_2Div_SSE::filter_upper_impl(int cnNum)
{
	__m128 mSum = _mm_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(4));

	float* sp_next = src.ptr<float>(0) + cnNum * 4;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m128 mCol_prev = _mm_setzero_ps();
	__m128 mCol_next = _mm_setzero_ps();
	__m128 mRef_next = _mm_setzero_ps();

	for (int j = 0; j < col; j++)
	{
		mRef_next = _mm_load_ps(sp_next);

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
	for (int i = r + 1; i < divRow; i++)
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
}

void boxFilter_OPSAT_AoS_2Div_SSE::filter_lower_impl(int cnNum)
{
	int rowOffset = (divNum - 1) * divRow;

	__m128 mSum = _mm_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(4));

	float* sp_next = src.ptr<float>(rowOffset - r) + cnNum * 4;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m128 mCol_prev = _mm_setzero_ps();
	__m128 mCol_next = _mm_setzero_ps();
	__m128 mRef_next = _mm_setzero_ps();

	for (int i = -r; i <= r; i++)
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

	float* dp = dest.ptr<float>(rowOffset) + cnNum * 4;
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


	float* sp_prev = src.ptr<float>(rowOffset - r) + cnNum * 4;
	__m128 mRef_prev = _mm_setzero_ps();

	/*   r < i < row - r - 1   */
	for (int i = rowOffset + 1; i < row - r - 1; i++)
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



/*
 * nDiv SSE
 */
boxFilter_OPSAT_AoS_nDiv_SSE::boxFilter_OPSAT_AoS_nDiv_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num)
	: boxFilter_OPSAT_AoS_2Div_SSE(_src, _dest, _r, _parallelType), num(_num)
{
	init();
}

void boxFilter_OPSAT_AoS_nDiv_SSE::init()
{
	mDiv = _mm_set1_ps(div);
	mBorder = _mm_set1_ps(static_cast<float>(r + 1));

	divNum = num;
	divRow = row / divNum;
	cnNum = cn / 4;
}

void boxFilter_OPSAT_AoS_nDiv_SSE::filter_middle_impl(int cnNum, int idx)
{
	int rowOffset = idx * divRow;

	__m128 mSum = _mm_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(4));

	float* sp_next = src.ptr<float>(rowOffset - r) + cnNum * 4;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m128 mCol_prev = _mm_setzero_ps();
	__m128 mCol_next = _mm_setzero_ps();
	__m128 mRef_next = _mm_setzero_ps();

	for (int i = -r; i <= r; i++)
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

	float* dp = dest.ptr<float>(rowOffset) + cnNum * 4;
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


	float* sp_prev = src.ptr<float>(rowOffset - r) + cnNum * 4;
	__m128 mRef_prev = _mm_setzero_ps();

	/*   r < i < row - r - 1   */
	for (int i = rowOffset + 1; i < rowOffset + divRow; i++)
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
}

void boxFilter_OPSAT_AoS_nDiv_SSE::filter()
{
	//	if (parallelType == NAIVE)
	//	{
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else if (idxDiv == divNum - 1)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_middle_impl(idxCn, idxDiv);
	//		}
	//	}
	//	else if (parallelType == OMP)
	//	{
	//#pragma omp parallel for
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else if (idxDiv == divNum - 1)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_middle_impl(idxCn, idxDiv);
	//		}
	//	}
	//	else if (parallelType == PARALLEL_FOR_)
	//	{
	//		if (num == 2)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 4; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4; idxCn < cnNum / 2; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum / 4 * 3; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4 * 3; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 4; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4; idxCn < cnNum / 2; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum / 4 * 3; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4 * 3; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//			}
	//		}
	//		else if (num == 4)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//			}
	//		}
	//		else if (num == 8)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 3);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 4);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 5);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 6);
	//				}
	//			}
	//		}
	//		else
	//		{
	//		}
	//	}

	const int loop = divNum * cnNum;
	if (parallelType == NAIVE)
	{
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else if (idx + cnNum >= loop)
				filter_lower_impl(idx % cnNum);
			else
				filter_middle_impl(idx % cnNum, idx / cnNum);
		}
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else if (idx + cnNum >= loop)
				filter_lower_impl(idx % cnNum);
			else
				filter_middle_impl(idx % cnNum, idx / cnNum);
		}
	}
}



/*
 * 2Div AVX
 */
boxFilter_OPSAT_AoS_2Div_AVX::boxFilter_OPSAT_AoS_2Div_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: boxFilter_OPSAT_AoS_2Div(_src, _dest, _r, _parallelType)
{
	init();
}

void boxFilter_OPSAT_AoS_2Div_AVX::init()
{
	mDiv = _mm256_set1_ps(div);
	mBorder = _mm256_set1_ps(static_cast<float>(r + 1));

	divNum = 2;
	divRow = row / divNum;
	cnNum = cn / 8;
}

void boxFilter_OPSAT_AoS_2Div_AVX::filter_upper_impl(int cnNum)
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
		mRef_next = _mm256_load_ps(sp_next);

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
	for (int i = r + 1; i < divRow; i++)
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
}

void boxFilter_OPSAT_AoS_2Div_AVX::filter_lower_impl(int cnNum)
{
	int rowOffset = (divNum - 1) * divRow;

	__m256 mSum = _mm256_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(8));

	float* sp_next = src.ptr<float>(rowOffset - r) + cnNum * 8;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m256 mCol_prev = _mm256_setzero_ps();
	__m256 mCol_next = _mm256_setzero_ps();
	__m256 mRef_next = _mm256_setzero_ps();

	for (int i = -r; i <= r; i++)
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

	float* dp = dest.ptr<float>(rowOffset) + cnNum * 8;
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


	float* sp_prev = src.ptr<float>(rowOffset - r) + cnNum * 8;
	__m256 mRef_prev = _mm256_setzero_ps();

	/*   r < i < row - r - 1   */
	for (int i = rowOffset + 1; i < row - r - 1; i++)
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



/*
 * nDiv AVX
 */
boxFilter_OPSAT_AoS_nDiv_AVX::boxFilter_OPSAT_AoS_nDiv_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num)
	: boxFilter_OPSAT_AoS_2Div_AVX(_src, _dest, _r, _parallelType), num(_num)
{
	init();
}

void boxFilter_OPSAT_AoS_nDiv_AVX::init()
{
	mDiv = _mm256_set1_ps(div);
	mBorder = _mm256_set1_ps(static_cast<float>(r + 1));

	divNum = num;
	divRow = row / divNum;
	cnNum = cn / 8;
}

void boxFilter_OPSAT_AoS_nDiv_AVX::filter_middle_impl(int cnNum, int idx)
{
	int rowOffset = idx * divRow;

	__m256 mSum = _mm256_setzero_ps();
	Mat columnSum = Mat::zeros(Size(col, 1), CV_32FC(8));

	float* sp_next = src.ptr<float>(rowOffset - r) + cnNum * 8;
	float* cp_prev = columnSum.ptr<float>(0);
	float* cp_next = cp_prev;
	__m256 mCol_prev = _mm256_setzero_ps();
	__m256 mCol_next = _mm256_setzero_ps();
	__m256 mRef_next = _mm256_setzero_ps();

	for (int i = -r; i <= r; i++)
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

	float* dp = dest.ptr<float>(rowOffset) + cnNum * 8;
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


	float* sp_prev = src.ptr<float>(rowOffset - r) + cnNum * 8;
	__m256 mRef_prev = _mm256_setzero_ps();

	/*   r < i < row - r - 1   */
	for (int i = rowOffset + 1; i < rowOffset + divRow; i++)
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
}

void boxFilter_OPSAT_AoS_nDiv_AVX::filter()
{
	//	if (parallelType == NAIVE)
	//	{
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else if (idxDiv == divNum - 1)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_middle_impl(idxCn, idxDiv);
	//		}
	//	}
	//	else if (parallelType == OMP)
	//	{
	//#pragma omp parallel for
	//		for (int idxDiv = 0; idxDiv < divNum; idxDiv++)
	//		{
	//			if (idxDiv == 0)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_upper_impl(idxCn);
	//			else if (idxDiv == divNum - 1)
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_lower_impl(idxCn);
	//			else
	//				for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//					filter_middle_impl(idxCn, idxDiv);
	//		}
	//	}
	//	else if (parallelType == PARALLEL_FOR_)
	//	{
	//		if (num == 2)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 4; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4; idxCn < cnNum / 2; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum / 4 * 3; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4 * 3; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 4; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4; idxCn < cnNum / 2; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum / 4 * 3; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 4 * 3; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//			}
	//		}
	//		else if (num == 4)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum / 2; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = cnNum / 2; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//			}
	//		}
	//		else if (num == 8)
	//		{
	//#pragma omp parallel sections
	//			{
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_upper_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_lower_impl(idxCn);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 1);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 2);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 3);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 4);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 5);
	//				}
	//#pragma omp section
	//				{
	//					for (int idxCn = 0; idxCn < cnNum; idxCn++)
	//						filter_middle_impl(idxCn, 6);
	//				}
	//			}
	//		}
	//		else
	//		{
	//		}
	//	}

	const int loop = divNum * cnNum;
	if (parallelType == NAIVE)
	{
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else if (idx + cnNum >= loop)
				filter_lower_impl(idx % cnNum);
			else
				filter_middle_impl(idx % cnNum, idx / cnNum);
		}
	}
	else if (parallelType == OMP)
	{
#pragma omp parallel for
		for (int idx = 0; idx < loop; idx++)
		{
			if (idx < cnNum)
				filter_upper_impl(idx);
			else if (idx + cnNum >= loop)
				filter_lower_impl(idx % cnNum);
			else
				filter_middle_impl(idx % cnNum, idx / cnNum);
		}
	}
}
