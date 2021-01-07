#include "boxFilter_SSAT_HtH.h"

using namespace cv;
using namespace std;

boxFilter_SSAT_HtH_nonVec::boxFilter_SSAT_HtH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(Size(src.rows, src.cols), src.type());
}

void boxFilter_SSAT_HtH_nonVec::filter()
{
	RowSumFilter_HtH_nonVec(src, temp, r, parallelType).filter();
	ColumnSumFilter_HtH_nonVec(temp, dest, r, parallelType).filter();
}



boxFilter_SSAT_HtH_SSE::boxFilter_SSAT_HtH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(Size(src.rows * 4, src.cols / 4), src.type());
}

void boxFilter_SSAT_HtH_SSE::filter()
{
	RowSumFilter_HtH_SSE(src, temp, r, parallelType).filter();
	ColumnSumFilter_HtH_SSE(temp, dest, r, parallelType).filter();
}



boxFilter_SSAT_HtH_AVX::boxFilter_SSAT_HtH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
{
	temp.create(Size(src.rows * 8, src.cols / 8), src.type());
}

void boxFilter_SSAT_HtH_AVX::filter()
{
	RowSumFilter_HtH_AVX(src, temp, r, parallelType).filter();
	ColumnSumFilter_HtH_AVX(temp, dest, r, parallelType).filter();
}



RowSumFilter_HtH_nonVec::RowSumFilter_HtH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: BoxFilterBase(_src, _dest, _r, _parallelType)
{
	
}

void RowSumFilter_HtH_nonVec::filter_naive_impl()
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;

	for (int j = 0; j < row; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + cn * j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += step;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += step;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += step;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += step;
			}
		}
	}
}

void RowSumFilter_HtH_nonVec::filter_omp_impl()
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;

#pragma omp parallel for
	for (int j = 0; j < row; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + cn * j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += step;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += step;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += step;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += step;
			}
		}
	}
}

void RowSumFilter_HtH_nonVec::operator()(const cv::Range& range) const
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;

	for (int j = range.start; j < range.end; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + cn * j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += step;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				dp += step;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				dp += step;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				dp += step;
			}
		}
	}
}

void RowSumFilter_HtH_nonVec::filter()
{
	if (parallelType == ParallelTypes::NAIVE)
	{
		filter_naive_impl();
	}
	else if (parallelType == ParallelTypes::OMP)
	{
		filter_omp_impl();
	}
	else if (parallelType == ParallelTypes::PARALLEL_FOR_)
	{
		cv::parallel_for_(cv::Range(0, src.rows), *this, cv::getNumThreads() - 1);
	}
	else
	{

	}
}



RowSumFilter_HtH_SSE::RowSumFilter_HtH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: RowSumFilter_HtH_nonVec(_src, _dest, _r, _parallelType)
{

}

void RowSumFilter_HtH_SSE::filter_naive_impl()
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const int back = 3 * cn;

	for (int j = 0; j < row; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + 4 * cn*j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
		}
	}
}

void RowSumFilter_HtH_SSE::filter_omp_impl()
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const int back = 3 * cn;

#pragma omp parallel for
	for (int j = 0; j < row; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + 4 * cn*j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
		}
	}
}

void RowSumFilter_HtH_SSE::operator()(const cv::Range& range) const
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const int back = 3 * cn;

	for (int j = range.start; j < range.end; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + 4 * cn*j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				if ((i & 3) == 3)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
		}
	}
}



RowSumFilter_HtH_AVX::RowSumFilter_HtH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: RowSumFilter_HtH_nonVec(_src, _dest, _r, _parallelType)
{

}

void RowSumFilter_HtH_AVX::filter_naive_impl()
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const int back = 7 * cn;

	for (int j = 0; j < row; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + 8 * cn*j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
		}
	}
}

void RowSumFilter_HtH_AVX::filter_omp_impl()
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const int back = 7 * cn;

#pragma omp parallel for
	for (int j = 0; j < row; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + 8 * cn*j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
		}
	}
}

void RowSumFilter_HtH_AVX::operator()(const cv::Range& range) const
{
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const int back = 7 * cn;

	for (int j = range.start; j < range.end; j++)
	{
		float* sp1;
		float* sp2;
		float* dp;
		for (int k = 0; k < cn; k++)
		{
			sp1 = src.ptr<float>(j) + k;
			sp2 = src.ptr<float>(j) + k + cn;
			dp = dest.ptr<float>(0) + k + 8 * cn*j;

			float sum = 0.f;
			sum += *sp1 * (r + 1);
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2 += cn;
			}
			*dp = sum;
			dp += cn;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;
				sp2 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1 += cn;

				*dp = sum;
				if ((i & 7) == 7)
				{
					dp += step - back;
				}
				else
				{
					dp += cn;
				}
			}
		}
	}
}



ColumnSumFilter_HtH_nonVec::ColumnSumFilter_HtH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: RowSumFilter_HtH_nonVec(_src, _dest, _r, _parallelType)
{
	
}

void ColumnSumFilter_HtH_nonVec::filter_naive_impl()
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
		for (int j = 0; j < row; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 1;

			float* dp = dest.ptr<float>(0) + j * 1;

			float sum = 0.f;

			sum = (r + 1) * *sp1;
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2++;
			}
			*dp = sum * div;
			dp += step;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2++;
				*dp = sum * div;
				dp += step;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1++;
				sp2++;
				*dp = sum * div;
				dp += step;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1++;
				*dp = sum * div;
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
		for (int j = 0; j < row; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 1;
			float* sp1_R2 = src.ptr<float>(j) + 2;
			float* sp2_R0 = src.ptr<float>(j) + 3;
			float* sp2_R1 = src.ptr<float>(j) + 4;
			float* sp2_R2 = src.ptr<float>(j) + 5;

			float* dp = dest.ptr<float>(0) + j * 3;

			float sum0 = 0.f;
			float sum1 = 0.f;
			float sum2 = 0.f;

			sum0 = (r + 1) * *sp1_R0;
			sum1 = (r + 1) * *sp1_R1;
			sum2 = (r + 1) * *sp1_R2;
			for (int i = 1; i <= r; i++)
			{
				sum0 += *sp2_R0;
				sp2_R0 += 3;
				sum1 += *sp2_R1;
				sp2_R1 += 3;
				sum2 += *sp2_R2;
				sp2_R2 += 3;
			}
			*dp = sum0 * div;
			dp++;
			*dp = sum1 * div;
			dp++;
			*dp = sum2 * div;
			dp += step - 2;

			for (int i = 1; i <= r; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp2_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp2_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp2_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp1_R0 += 3;
				*sp2_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp1_R1 += 3;
				*sp2_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp1_R2 += 3;
				*sp2_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp1_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp1_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp1_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
		}
	}
}

void ColumnSumFilter_HtH_nonVec::filter_omp_impl()
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
#pragma omp parallel for
		for (int j = 0; j < row; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 1;

			float* dp = dest.ptr<float>(0) + j * 1;

			float sum = 0.f;

			sum = (r + 1) * *sp1;
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2++;
			}
			*dp = sum * div;
			dp += step;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2++;
				*dp = sum * div;
				dp += step;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1++;
				sp2++;
				*dp = sum * div;
				dp += step;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1++;
				*dp = sum * div;
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
#pragma omp parallel for
		for (int j = 0; j < row; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 1;
			float* sp1_R2 = src.ptr<float>(j) + 2;
			float* sp2_R0 = src.ptr<float>(j) + 3;
			float* sp2_R1 = src.ptr<float>(j) + 4;
			float* sp2_R2 = src.ptr<float>(j) + 5;

			float* dp = dest.ptr<float>(0) + j * 3;

			float sum0 = 0.f;
			float sum1 = 0.f;
			float sum2 = 0.f;

			sum0 = (r + 1) * *sp1_R0;
			sum1 = (r + 1) * *sp1_R1;
			sum2 = (r + 1) * *sp1_R2;
			for (int i = 1; i <= r; i++)
			{
				sum0 += *sp2_R0;
				sp2_R0 += 3;
				sum1 += *sp2_R1;
				sp2_R1 += 3;
				sum2 += *sp2_R2;
				sp2_R2 += 3;
			}
			*dp = sum0 * div;
			dp++;
			*dp = sum1 * div;
			dp++;
			*dp = sum2 * div;
			dp += step - 2;

			for (int i = 1; i <= r; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp2_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp2_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp2_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp1_R0 += 3;
				*sp2_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp1_R1 += 3;
				*sp2_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp1_R2 += 3;
				*sp2_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp1_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp1_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp1_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
		}
	}
}

void ColumnSumFilter_HtH_nonVec::operator()(const cv::Range& range) const
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
		for (int j = range.start; j < range.end; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 1;

			float* dp = dest.ptr<float>(0) + j * 1;

			float sum = 0.f;

			sum = (r + 1) * *sp1;
			for (int i = 1; i <= r; i++)
			{
				sum += *sp2;
				sp2++;
			}
			*dp = sum * div;
			dp += step;

			for (int i = 1; i <= r; i++)
			{
				sum += *sp2 - *sp1;
				sp2++;
				*dp = sum * div;
				dp += step;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum += *sp2 - *sp1;
				sp1++;
				sp2++;
				*dp = sum * div;
				dp += step;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum += *sp2 - *sp1;
				sp1++;
				*dp = sum * div;
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
		for (int j = range.start; j < range.end; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 1;
			float* sp1_R2 = src.ptr<float>(j) + 2;
			float* sp2_R0 = src.ptr<float>(j) + 3;
			float* sp2_R1 = src.ptr<float>(j) + 4;
			float* sp2_R2 = src.ptr<float>(j) + 5;

			float* dp = dest.ptr<float>(0) + j * 3;

			float sum0 = 0.f;
			float sum1 = 0.f;
			float sum2 = 0.f;

			sum0 = (r + 1) * *sp1_R0;
			sum1 = (r + 1) * *sp1_R1;
			sum2 = (r + 1) * *sp1_R2;
			for (int i = 1; i <= r; i++)
			{
				sum0 += *sp2_R0;
				sp2_R0 += 3;
				sum1 += *sp2_R1;
				sp2_R1 += 3;
				sum2 += *sp2_R2;
				sp2_R2 += 3;
			}
			*dp = sum0 * div;
			dp++;
			*dp = sum1 * div;
			dp++;
			*dp = sum2 * div;
			dp += step - 2;

			for (int i = 1; i <= r; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp2_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp2_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp2_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
			for (int i = r + 1; i < col - r - 1; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp1_R0 += 3;
				*sp2_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp1_R1 += 3;
				*sp2_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp1_R2 += 3;
				*sp2_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
			for (int i = col - r - 1; i < col; i++)
			{
				sum0 += *sp2_R0 - *sp1_R0;
				*sp1_R0 += 3;
				sum1 += *sp2_R1 - *sp1_R1;
				*sp1_R1 += 3;
				sum2 += *sp2_R2 - *sp1_R2;
				*sp1_R2 += 3;

				*dp = sum0 * div;
				dp++;
				*dp = sum1 * div;
				dp++;
				*dp = sum2 * div;
				dp += step - 2;
			}
		}
	}
}



ColumnSumFilter_HtH_SSE::ColumnSumFilter_HtH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: RowSumFilter_HtH_nonVec(_src, _dest, _r, _parallelType)
{
	
}

void ColumnSumFilter_HtH_SSE::filter_naive_impl()
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	const __m128 mDiv = _mm_set1_ps(div);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
		for (int j = 0; j < row; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 4;

			float* dp = dest.ptr<float>(0) + j * 4;

			__m128 mSum = _mm_setzero_ps();
			__m128 mTmp;

			mSum = _mm_mul_ps(_mm_set1_ps((float)(r + 1)), _mm_load_ps(sp1));
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
			}
			_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;

			mTmp = _mm_load_ps(sp1);
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
				mSum = _mm_sub_ps(mSum, mTmp);
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			for (int i = r + 1; i < col / 4 - r - 1; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			mTmp = _mm_load_ps(sp2);
			for (int i = col / 4 - r - 1; i < col / 4; i++)
			{
				mSum = _mm_add_ps(mSum, mTmp);
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
		for (int j = 0; j < row; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 4;
			float* sp1_R2 = src.ptr<float>(j) + 8;
			float* sp2_R0 = src.ptr<float>(j) + 12;
			float* sp2_R1 = src.ptr<float>(j) + 16;
			float* sp2_R2 = src.ptr<float>(j) + 20;

			float* dp = dest.ptr<float>(0) + j * 12;

			__m128 mSum0 = _mm_setzero_ps();
			__m128 mSum1 = _mm_setzero_ps();
			__m128 mSum2 = _mm_setzero_ps();
			__m128 mTmp0, mTmp1, mTmp2;

			mTmp0 = _mm_set1_ps((float)(r + 1));
			mSum0 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R0));
			mSum1 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R1));
			mSum2 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R2));
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
			}
			_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
			dp += 4;
			_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
			dp += 4;
			_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
			dp += step - 8;

			mTmp0 = _mm_load_ps(sp1_R0);
			mTmp1 = _mm_load_ps(sp1_R1);
			mTmp2 = _mm_load_ps(sp1_R2);
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum0 = _mm_sub_ps(mSum0, mTmp0);
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum1 = _mm_sub_ps(mSum1, mTmp1);
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
				mSum2 = _mm_sub_ps(mSum2, mTmp2);

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
			for (int i = r + 1; i < col / 4 - r - 1; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(sp1_R0));
				sp1_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(sp1_R1));
				sp1_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
				mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(sp1_R2));
				sp1_R2 += 12;

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
			mTmp0 = _mm_load_ps(sp2_R0);
			mTmp1 = _mm_load_ps(sp2_R1);
			mTmp2 = _mm_load_ps(sp2_R2);
			for (int i = col / 4 - r - 1; i < col / 4; i++)
			{
				mSum0 = _mm_add_ps(mSum0, mTmp0);
				mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(sp1_R0));
				sp1_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, mTmp1);
				mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(sp1_R1));
				sp1_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, mTmp2);
				mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(sp1_R2));
				sp1_R2 += 12;

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
		}
	}
}

void ColumnSumFilter_HtH_SSE::filter_omp_impl()
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	const __m128 mDiv = _mm_set1_ps(div);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
#pragma omp parallel for
		for (int j = 0; j < row; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 4;

			float* dp = dest.ptr<float>(0) + j * 4;

			__m128 mSum = _mm_setzero_ps();
			__m128 mTmp;

			mSum = _mm_mul_ps(_mm_set1_ps((float)(r + 1)), _mm_load_ps(sp1));
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
			}
			_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;

			mTmp = _mm_load_ps(sp1);
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
				mSum = _mm_sub_ps(mSum, mTmp);
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			for (int i = r + 1; i < col / 4 - r - 1; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			mTmp = _mm_load_ps(sp2);
			for (int i = col / 4 - r - 1; i < col / 4; i++)
			{
				mSum = _mm_add_ps(mSum, mTmp);
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
#pragma omp parallel for
		for (int j = 0; j < row; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 4;
			float* sp1_R2 = src.ptr<float>(j) + 8;
			float* sp2_R0 = src.ptr<float>(j) + 12;
			float* sp2_R1 = src.ptr<float>(j) + 16;
			float* sp2_R2 = src.ptr<float>(j) + 20;

			float* dp = dest.ptr<float>(0) + j * 12;

			__m128 mSum0 = _mm_setzero_ps();
			__m128 mSum1 = _mm_setzero_ps();
			__m128 mSum2 = _mm_setzero_ps();
			__m128 mTmp0, mTmp1, mTmp2;

			mTmp0 = _mm_set1_ps((float)(r + 1));
			mSum0 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R0));
			mSum1 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R1));
			mSum2 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R2));
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
			}
			_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
			dp += 4;
			_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
			dp += 4;
			_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
			dp += step - 8;

			mTmp0 = _mm_load_ps(sp1_R0);
			mTmp1 = _mm_load_ps(sp1_R1);
			mTmp2 = _mm_load_ps(sp1_R2);
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum0 = _mm_sub_ps(mSum0, mTmp0);
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum1 = _mm_sub_ps(mSum1, mTmp1);
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
				mSum2 = _mm_sub_ps(mSum2, mTmp2);

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
			for (int i = r + 1; i < col / 4 - r - 1; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(sp1_R0));
				sp1_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(sp1_R1));
				sp1_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
				mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(sp1_R2));
				sp1_R2 += 12;

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
			mTmp0 = _mm_load_ps(sp2_R0);
			mTmp1 = _mm_load_ps(sp2_R1);
			mTmp2 = _mm_load_ps(sp2_R2);
			for (int i = col / 4 - r - 1; i < col / 4; i++)
			{
				mSum0 = _mm_add_ps(mSum0, mTmp0);
				mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(sp1_R0));
				sp1_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, mTmp1);
				mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(sp1_R1));
				sp1_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, mTmp2);
				mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(sp1_R2));
				sp1_R2 += 12;

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
		}
	}
}

void ColumnSumFilter_HtH_SSE::operator()(const cv::Range& range) const
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	const __m128 mDiv = _mm_set1_ps(div);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
		for (int j = range.start; j < range.end; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 4;

			float* dp = dest.ptr<float>(0) + j * 4;

			__m128 mSum = _mm_setzero_ps();
			__m128 mTmp;

			mSum = _mm_mul_ps(_mm_set1_ps((float)(r + 1)), _mm_load_ps(sp1));
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
			}
			_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
			dp += step;

			mTmp = _mm_load_ps(sp1);
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
				mSum = _mm_sub_ps(mSum, mTmp);
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			for (int i = r + 1; i < col / 4 - r - 1; i++)
			{
				mSum = _mm_add_ps(mSum, _mm_load_ps(sp2));
				sp2 += 4;
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
			mTmp = _mm_load_ps(sp2);
			for (int i = col / 4 - r - 1; i < col / 4; i++)
			{
				mSum = _mm_add_ps(mSum, mTmp);
				mSum = _mm_sub_ps(mSum, _mm_load_ps(sp1));
				sp1 += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum, mDiv));
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
		for (int j = range.start; j < range.end; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 4;
			float* sp1_R2 = src.ptr<float>(j) + 8;
			float* sp2_R0 = src.ptr<float>(j) + 12;
			float* sp2_R1 = src.ptr<float>(j) + 16;
			float* sp2_R2 = src.ptr<float>(j) + 20;

			float* dp = dest.ptr<float>(0) + j * 12;

			__m128 mSum0 = _mm_setzero_ps();
			__m128 mSum1 = _mm_setzero_ps();
			__m128 mSum2 = _mm_setzero_ps();
			__m128 mTmp0, mTmp1, mTmp2;

			mTmp0 = _mm_set1_ps((float)(r + 1));
			mSum0 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R0));
			mSum1 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R1));
			mSum2 = _mm_mul_ps(mTmp0, _mm_load_ps(sp1_R2));
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
			}
			_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
			dp += 4;
			_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
			dp += 4;
			_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
			dp += step - 8;

			mTmp0 = _mm_load_ps(sp1_R0);
			mTmp1 = _mm_load_ps(sp1_R1);
			mTmp2 = _mm_load_ps(sp1_R2);
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum0 = _mm_sub_ps(mSum0, mTmp0);
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum1 = _mm_sub_ps(mSum1, mTmp1);
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
				mSum2 = _mm_sub_ps(mSum2, mTmp2);

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
			for (int i = r + 1; i < col / 4 - r - 1; i++)
			{
				mSum0 = _mm_add_ps(mSum0, _mm_load_ps(sp2_R0));
				sp2_R0 += 12;
				mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(sp1_R0));
				sp1_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, _mm_load_ps(sp2_R1));
				sp2_R1 += 12;
				mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(sp1_R1));
				sp1_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, _mm_load_ps(sp2_R2));
				sp2_R2 += 12;
				mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(sp1_R2));
				sp1_R2 += 12;

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
			mTmp0 = _mm_load_ps(sp2_R0);
			mTmp1 = _mm_load_ps(sp2_R1);
			mTmp2 = _mm_load_ps(sp2_R2);
			for (int i = col / 4 - r - 1; i < col / 4; i++)
			{
				mSum0 = _mm_add_ps(mSum0, mTmp0);
				mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(sp1_R0));
				sp1_R0 += 12;
				mSum1 = _mm_add_ps(mSum1, mTmp1);
				mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(sp1_R1));
				sp1_R1 += 12;
				mSum2 = _mm_add_ps(mSum2, mTmp2);
				mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(sp1_R2));
				sp1_R2 += 12;

				_mm_stream_ps(dp, _mm_mul_ps(mSum0, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum1, mDiv));
				dp += 4;
				_mm_stream_ps(dp, _mm_mul_ps(mSum2, mDiv));
				dp += step - 8;
			}
		}
	}
}



ColumnSumFilter_HtH_AVX::ColumnSumFilter_HtH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
	: RowSumFilter_HtH_nonVec(_src, _dest, _r, _parallelType)
{
	
}

void ColumnSumFilter_HtH_AVX::filter_naive_impl()
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	const __m256 mDiv = _mm256_set1_ps(div);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
		for (int j = 0; j < row; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 8;

			float* dp = dest.ptr<float>(0) + j * 8;

			__m256 mSum = _mm256_setzero_ps();
			__m256 mTmp;

			mSum = _mm256_mul_ps(_mm256_set1_ps((float)(r + 1)), _mm256_load_ps(sp1));
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;

			mTmp = _mm256_load_ps(sp1);
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
				mSum = _mm256_sub_ps(mSum, mTmp);
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
			for (int i = r + 1; i < col / 8 - r - 1; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
				mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
				sp1 += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
			mTmp = _mm256_load_ps(sp2);
			for (int i = col / 8 - r - 1; i < col / 8; i++)
			{
				mSum = _mm256_add_ps(mSum, mTmp);
				mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
				sp1 += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
		for (int j = 0; j < row; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 8;
			float* sp1_R2 = src.ptr<float>(j) + 16;
			float* sp2_R0 = src.ptr<float>(j) + 24;
			float* sp2_R1 = src.ptr<float>(j) + 32;
			float* sp2_R2 = src.ptr<float>(j) + 40;

			float* dp = dest.ptr<float>(0) + j * 24;

			__m256 mSum0 = _mm256_setzero_ps();
			__m256 mSum1 = _mm256_setzero_ps();
			__m256 mSum2 = _mm256_setzero_ps();
			__m256 mTmp0, mTmp1, mTmp2;

			mTmp0 = _mm256_set1_ps((float)(r + 1));
			mSum0 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R0));
			mSum1 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R1));
			mSum2 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R2));
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
			dp += 8;
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
			dp += 8;
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
			dp += step - 16;

			mTmp0 = _mm256_load_ps(sp1_R0);
			mTmp1 = _mm256_load_ps(sp1_R1);
			mTmp2 = _mm256_load_ps(sp1_R2);
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum0 = _mm256_sub_ps(mSum0, mTmp0);
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum1 = _mm256_sub_ps(mSum1, mTmp1);
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
				mSum2 = _mm256_sub_ps(mSum2, mTmp2);

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
			for (int i = r + 1; i < col / 8 - r - 1; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(sp1_R0));
				sp1_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(sp1_R1));
				sp1_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
				mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(sp1_R2));
				sp1_R2 += 24;

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
			mTmp0 = _mm256_load_ps(sp2_R0);
			mTmp1 = _mm256_load_ps(sp2_R1);
			mTmp2 = _mm256_load_ps(sp2_R2);
			for (int i = col / 8 - r - 1; i < col / 8; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, mTmp0);
				mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(sp1_R0));
				sp1_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, mTmp1);
				mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(sp1_R1));
				sp1_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, mTmp2);
				mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(sp1_R2));
				sp1_R2 += 24;

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
		}
	}
}

void ColumnSumFilter_HtH_AVX::filter_omp_impl()
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	const __m256 mDiv = _mm256_set1_ps(div);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
#pragma omp parallel for
		for (int j = 0; j < row; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 8;

			float* dp = dest.ptr<float>(0) + j * 8;

			__m256 mSum = _mm256_setzero_ps();
			__m256 mTmp;

			mSum = _mm256_mul_ps(_mm256_set1_ps((float)(r + 1)), _mm256_load_ps(sp1));
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;

			mTmp = _mm256_load_ps(sp1);
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
				mSum = _mm256_sub_ps(mSum, mTmp);
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
			for (int i = r + 1; i < col / 8 - r - 1; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
				mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
				sp1 += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
			mTmp = _mm256_load_ps(sp2);
			for (int i = col / 8 - r - 1; i < col / 8; i++)
			{
				mSum = _mm256_add_ps(mSum, mTmp);
				mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
				sp1 += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
#pragma omp parallel for
		for (int j = 0; j < row; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 8;
			float* sp1_R2 = src.ptr<float>(j) + 16;
			float* sp2_R0 = src.ptr<float>(j) + 24;
			float* sp2_R1 = src.ptr<float>(j) + 32;
			float* sp2_R2 = src.ptr<float>(j) + 40;

			float* dp = dest.ptr<float>(0) + j * 24;

			__m256 mSum0 = _mm256_setzero_ps();
			__m256 mSum1 = _mm256_setzero_ps();
			__m256 mSum2 = _mm256_setzero_ps();
			__m256 mTmp0, mTmp1, mTmp2;

			mTmp0 = _mm256_set1_ps((float)(r + 1));
			mSum0 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R0));
			mSum1 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R1));
			mSum2 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R2));
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
			dp += 8;
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
			dp += 8;
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
			dp += step - 16;

			mTmp0 = _mm256_load_ps(sp1_R0);
			mTmp1 = _mm256_load_ps(sp1_R1);
			mTmp2 = _mm256_load_ps(sp1_R2);
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum0 = _mm256_sub_ps(mSum0, mTmp0);
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum1 = _mm256_sub_ps(mSum1, mTmp1);
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
				mSum2 = _mm256_sub_ps(mSum2, mTmp2);

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
			for (int i = r + 1; i < col / 8 - r - 1; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(sp1_R0));
				sp1_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(sp1_R1));
				sp1_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
				mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(sp1_R2));
				sp1_R2 += 24;

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
			mTmp0 = _mm256_load_ps(sp2_R0);
			mTmp1 = _mm256_load_ps(sp2_R1);
			mTmp2 = _mm256_load_ps(sp2_R2);
			for (int i = col / 8 - r - 1; i < col / 8; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, mTmp0);
				mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(sp1_R0));
				sp1_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, mTmp1);
				mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(sp1_R1));
				sp1_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, mTmp2);
				mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(sp1_R2));
				sp1_R2 += 24;

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
		}
	}
}

void ColumnSumFilter_HtH_AVX::operator()(const cv::Range& range) const
{
	const int size = 2 * r + 1;
	const int col = src.cols;
	const int row = src.rows;
	const int cn = src.channels();

	const int step = dest.cols*cn;
	const float div = 1.f / (size*size);

	const __m256 mDiv = _mm256_set1_ps(div);

	if (cn == 1)
	{
		/* ---------- 1 ---------- */
		for (int j = range.start; j < range.end; j++)
		{
			float* sp1 = src.ptr<float>(j);
			float* sp2 = src.ptr<float>(j) + 8;

			float* dp = dest.ptr<float>(0) + j * 8;

			__m256 mSum = _mm256_setzero_ps();
			__m256 mTmp;

			mSum = _mm256_mul_ps(_mm256_set1_ps((float)(r + 1)), _mm256_load_ps(sp1));
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
			dp += step;

			mTmp = _mm256_load_ps(sp1);
			for (int i = 1; i <= r; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
				mSum = _mm256_sub_ps(mSum, mTmp);
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
			for (int i = r + 1; i < col / 8 - r - 1; i++)
			{
				mSum = _mm256_add_ps(mSum, _mm256_load_ps(sp2));
				sp2 += 8;
				mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
				sp1 += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
			mTmp = _mm256_load_ps(sp2);
			for (int i = col / 8 - r - 1; i < col / 8; i++)
			{
				mSum = _mm256_add_ps(mSum, mTmp);
				mSum = _mm256_sub_ps(mSum, _mm256_load_ps(sp1));
				sp1 += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum, mDiv));
				dp += step;
			}
		}
	}
	else if (cn == 3)
	{
		/* ---------- 1 ---------- */
		for (int j = range.start; j < range.end; j++)
		{
			float* sp1_R0 = src.ptr<float>(j);
			float* sp1_R1 = src.ptr<float>(j) + 8;
			float* sp1_R2 = src.ptr<float>(j) + 16;
			float* sp2_R0 = src.ptr<float>(j) + 24;
			float* sp2_R1 = src.ptr<float>(j) + 32;
			float* sp2_R2 = src.ptr<float>(j) + 40;

			float* dp = dest.ptr<float>(0) + j * 24;

			__m256 mSum0 = _mm256_setzero_ps();
			__m256 mSum1 = _mm256_setzero_ps();
			__m256 mSum2 = _mm256_setzero_ps();
			__m256 mTmp0, mTmp1, mTmp2;

			mTmp0 = _mm256_set1_ps((float)(r + 1));
			mSum0 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R0));
			mSum1 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R1));
			mSum2 = _mm256_mul_ps(mTmp0, _mm256_load_ps(sp1_R2));
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
			}
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
			dp += 8;
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
			dp += 8;
			_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
			dp += step - 16;

			mTmp0 = _mm256_load_ps(sp1_R0);
			mTmp1 = _mm256_load_ps(sp1_R1);
			mTmp2 = _mm256_load_ps(sp1_R2);
			for (int i = 1; i <= r; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum0 = _mm256_sub_ps(mSum0, mTmp0);
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum1 = _mm256_sub_ps(mSum1, mTmp1);
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
				mSum2 = _mm256_sub_ps(mSum2, mTmp2);

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
			for (int i = r + 1; i < col / 8 - r - 1; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(sp2_R0));
				sp2_R0 += 24;
				mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(sp1_R0));
				sp1_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(sp2_R1));
				sp2_R1 += 24;
				mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(sp1_R1));
				sp1_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(sp2_R2));
				sp2_R2 += 24;
				mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(sp1_R2));
				sp1_R2 += 24;

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
			mTmp0 = _mm256_load_ps(sp2_R0);
			mTmp1 = _mm256_load_ps(sp2_R1);
			mTmp2 = _mm256_load_ps(sp2_R2);
			for (int i = col / 8 - r - 1; i < col / 8; i++)
			{
				mSum0 = _mm256_add_ps(mSum0, mTmp0);
				mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(sp1_R0));
				sp1_R0 += 24;
				mSum1 = _mm256_add_ps(mSum1, mTmp1);
				mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(sp1_R1));
				sp1_R1 += 24;
				mSum2 = _mm256_add_ps(mSum2, mTmp2);
				mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(sp1_R2));
				sp1_R2 += 24;

				_mm256_stream_ps(dp, _mm256_mul_ps(mSum0, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum1, mDiv));
				dp += 8;
				_mm256_stream_ps(dp, _mm256_mul_ps(mSum2, mDiv));
				dp += step - 16;
			}
		}
	}
}
