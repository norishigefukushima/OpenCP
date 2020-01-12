#include "guidedFilter_Merge_Transpose_Inverse.h"
#include "guidedFilter_Merge_Share_Transpose_Inverse.h"

using namespace std;
using namespace cv;

void guidedFilter_Merge_Share_Transpose_Inverse_nonVec::init()
{
	b.create(t_size, CV_32F);

	if (guide.channels() == 1)
	{
		a.create(t_size, CV_32F);
		var.create(t_size, CV_32F);
		mean_I.create(t_size, CV_32F);

		temp.resize(2);
		temp_ab.resize(2);

		for (int i = 0; i < temp.size(); i++) temp[i].create(t_size, CV_32F);
		for (int i = 0; i < temp_ab.size(); i++) temp_ab[i].create(src.size(), CV_32F);
	}
	else if (guide.channels() == 3)
	{
		split(guide, vI);

		va.resize(3);
		for (int i = 0; i < va.size(); i++) va[i].create(t_size, CV_32F);
		vCov.resize(6);
		for (int i = 0; i < vCov.size(); i++) vCov[i].create(t_size, CV_32F);
		vMean_I.resize(3);
		for (int i = 0; i < vMean_I.size(); i++) vMean_I[i].create(t_size, CV_32F);
		det.create(t_size, CV_32F);

		temp.resize(9);
		temp_ab.resize(4);

		for (int i = 0; i < temp.size(); i++) temp[i].create(t_size, CV_32F);
		for (int i = 0; i < temp_ab.size(); i++) temp_ab[i].create(src.size(), CV_32F);
	}
}

void guidedFilter_Merge_Share_Transpose_Inverse_nonVec::compute_Var()
{
	RowSumFilter_Var_Transpose_Inverse_nonVec rsf_var(guide, temp, r, parallelType); rsf_var.filter();
	ColumnSumFilter_Var_Transpose_nonVec csf_var(temp, var, mean_I, r, eps, parallelType); csf_var.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_nonVec::compute_Cov()
{
	RowSumFilter_Cov_Transpose_Inverse_nonVec rsf_cov(vI, temp, r, parallelType); rsf_cov.filter();
	ColumnSumFilter_Cov_Transpose_nonVec csf_cov(temp, vCov, det, vMean_I, r, eps, parallelType); csf_cov.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_nonVec::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec csf_ip2ab(temp, var, mean_I, a, b, r, parallelType); csf_ip2ab.filter();
	ColumnSumFilter_ab2q_Guide1_Transpose_Inverse_nonVec csf_ab2q(a, b, temp_ab, r, parallelType); csf_ab2q.filter();
	RowSumFilter_ab2q_Guide1_Transpose_Inverse rsf_ab2q(temp_ab, guide, output, r, parallelType); rsf_ab2q.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_nonVec::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType); csf_ip2ab.filter();
	ColumnSumFilter_ab2q_Guide3_Transpose_Inverse_nonVec csf_ab2q(va, b, temp_ab, r, parallelType); csf_ab2q.filter();
	RowSumFilter_ab2q_Guide3_Transpose_Inverse rsf_ab2q(temp_ab, vI, output, r, parallelType); rsf_ab2q.filter();
}



void guidedFilter_Merge_Share_Transpose_Inverse_SSE::compute_Var()
{
	RowSumFilter_Var_Transpose_Inverse_SSE rsf_var(guide, temp, r, parallelType); rsf_var.filter();
	ColumnSumFilter_Var_Transpose_SSE csf_var(temp, var, mean_I, r, eps, parallelType); csf_var.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_SSE::compute_Cov()
{
	RowSumFilter_Cov_Transpose_Inverse_SSE rsf_cov(vI, temp, r, parallelType); rsf_cov.filter();
	ColumnSumFilter_Cov_Transpose_SSE csf_cov(temp, vCov, det, vMean_I, r, eps, parallelType); csf_cov.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_SSE::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share_Transpose_SSE rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_SSE csf_ip2ab(temp, var, mean_I, a, b, r, parallelType); csf_ip2ab.filter();
	ColumnSumFilter_ab2q_Guide1_Transpose_Inverse_SSE csf_ab2q(a, b, temp_ab, r, parallelType); csf_ab2q.filter();
	RowSumFilter_ab2q_Guide1_Transpose_Inverse rsf_ab2q(temp_ab, guide, output, r, parallelType); rsf_ab2q.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_SSE::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share_Transpose_SSE rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_SSE csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType); csf_ip2ab.filter();
	ColumnSumFilter_ab2q_Guide3_Transpose_Inverse_SSE csf_ab2q(va, b, temp_ab, r, parallelType); csf_ab2q.filter();
	RowSumFilter_ab2q_Guide3_Transpose_Inverse rsf_ab2q(temp_ab, vI, output, r, parallelType); rsf_ab2q.filter();
}



void guidedFilter_Merge_Share_Transpose_Inverse_AVX::compute_Var()
{
	RowSumFilter_Var_Transpose_Inverse_AVX rsf_var(guide, temp, r, parallelType); rsf_var.filter();
	ColumnSumFilter_Var_Transpose_AVX csf_var(temp, var, mean_I, r, eps, parallelType); csf_var.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_AVX::compute_Cov()
{
	RowSumFilter_Cov_Transpose_Inverse_AVX rsf_cov(vI, temp, r, parallelType); rsf_cov.filter();
	ColumnSumFilter_Cov_Transpose_AVX csf_cov(temp, vCov, det, vMean_I, r, eps, parallelType); csf_cov.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_AVX::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share_Transpose_AVX rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_AVX csf_ip2ab(temp, var, mean_I, a, b, r, parallelType); csf_ip2ab.filter();
	ColumnSumFilter_ab2q_Guide1_Transpose_Inverse_AVX csf_ab2q(a, b, temp_ab, r, parallelType); csf_ab2q.filter();
	RowSumFilter_ab2q_Guide1_Transpose_Inverse rsf_ab2q(temp_ab, guide, output, r, parallelType); rsf_ab2q.filter();
}

void guidedFilter_Merge_Share_Transpose_Inverse_AVX::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share_Transpose_AVX rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_AVX csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType); csf_ip2ab.filter();
	ColumnSumFilter_ab2q_Guide3_Transpose_Inverse_AVX csf_ab2q(va, b, temp_ab, r, parallelType); csf_ab2q.filter();
	RowSumFilter_ab2q_Guide3_Transpose_Inverse rsf_ab2q(temp_ab, vI, output, r, parallelType); rsf_ab2q.filter();
}



/* --- Compute Var --- */
void RowSumFilter_Var_Transpose_Inverse_nonVec::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p += step;
		*v1_p = sum1;
		v1_p += step;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
	}
}

void RowSumFilter_Var_Transpose_Inverse_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p += step;
		*v1_p = sum1;
		v1_p += step;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
	}
}

void RowSumFilter_Var_Transpose_Inverse_nonVec::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p += step;
		*v1_p = sum1;
		v1_p += step;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			v0_p += step;
			v1_p += step;
		}
	}
}



void RowSumFilter_Var_Transpose_Inverse_SSE::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + 4 * j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + 4 * j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p++;
		*v1_p = sum1;
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
	}
}

void RowSumFilter_Var_Transpose_Inverse_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + 4 * j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + 4 * j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p++;
		*v1_p = sum1;
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
	}
}

void RowSumFilter_Var_Transpose_Inverse_SSE::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + 4 * j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + 4 * j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p++;
		*v1_p = sum1;
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 3) == 3)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
	}
}



void RowSumFilter_Var_Transpose_Inverse_AVX::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + 8 * j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + 8 * j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p++;
		*v1_p = sum1;
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
	}
}

void RowSumFilter_Var_Transpose_Inverse_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + 8 * j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + 8 * j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p++;
		*v1_p = sum1;
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
	}
}

void RowSumFilter_Var_Transpose_Inverse_AVX::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(0) + 8 * j; // mean_I
		float* v1_p = tempVec[1].ptr<float>(0) + 8 * j; // corr_I

		float sum0 = 0.f, sum1 = 0.f;
		sum0 = *I_p1 * (r + 1);
		sum1 = (*I_p1 * *I_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2;
			sum1 += (*I_p2 * *I_p2);
			I_p2++;
		}
		*v0_p = sum0;
		v0_p++;
		*v1_p = sum1;
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			*v1_p = sum1;

			if ((i & 7) == 7)
			{
				v0_p += step;
				v1_p += step;
			}
			else
			{
				v0_p++;
				v1_p++;
			}
		}
	}
}



/* --- Compute Cov --- */
void RowSumFilter_Cov_Transpose_Inverse_nonVec::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p += step;
		*d01_p = sum01;
		d01_p += step;
		*d02_p = sum02;
		d02_p += step;
		*d03_p = sum03;
		d03_p += step;
		*d04_p = sum04;
		d04_p += step;
		*d05_p = sum05;
		d05_p += step;
		*d06_p = sum06;
		d06_p += step;
		*d07_p = sum07;
		d07_p += step;
		*d08_p = sum08;
		d08_p += step;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
	}
}

void RowSumFilter_Cov_Transpose_Inverse_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p += step;
		*d01_p = sum01;
		d01_p += step;
		*d02_p = sum02;
		d02_p += step;
		*d03_p = sum03;
		d03_p += step;
		*d04_p = sum04;
		d04_p += step;
		*d05_p = sum05;
		d05_p += step;
		*d06_p = sum06;
		d06_p += step;
		*d07_p = sum07;
		d07_p += step;
		*d08_p = sum08;
		d08_p += step;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
	}
}

void RowSumFilter_Cov_Transpose_Inverse_nonVec::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p += step;
		*d01_p = sum01;
		d01_p += step;
		*d02_p = sum02;
		d02_p += step;
		*d03_p = sum03;
		d03_p += step;
		*d04_p = sum04;
		d04_p += step;
		*d05_p = sum05;
		d05_p += step;
		*d06_p = sum06;
		d06_p += step;
		*d07_p = sum07;
		d07_p += step;
		*d08_p = sum08;
		d08_p += step;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			d00_p += step;
			d01_p += step;
			d02_p += step;
			d03_p += step;
			d04_p += step;
			d05_p += step;
			d06_p += step;
			d07_p += step;
			d08_p += step;
		}
	}
}



void RowSumFilter_Cov_Transpose_Inverse_SSE::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + 4 * j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + 4 * j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + 4 * j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + 4 * j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + 4 * j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + 4 * j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + 4 * j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + 4 * j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + 4 * j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;
		*d04_p = sum04;
		d04_p++;
		*d05_p = sum05;
		d05_p++;
		*d06_p = sum06;
		d06_p++;
		*d07_p = sum07;
		d07_p++;
		*d08_p = sum08;
		d08_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
	}
}

void RowSumFilter_Cov_Transpose_Inverse_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + 4 * j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + 4 * j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + 4 * j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + 4 * j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + 4 * j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + 4 * j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + 4 * j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + 4 * j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + 4 * j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;
		*d04_p = sum04;
		d04_p++;
		*d05_p = sum05;
		d05_p++;
		*d06_p = sum06;
		d06_p++;
		*d07_p = sum07;
		d07_p++;
		*d08_p = sum08;
		d08_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
	}
}

void RowSumFilter_Cov_Transpose_Inverse_SSE::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + 4 * j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + 4 * j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + 4 * j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + 4 * j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + 4 * j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + 4 * j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + 4 * j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + 4 * j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + 4 * j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;
		*d04_p = sum04;
		d04_p++;
		*d05_p = sum05;
		d05_p++;
		*d06_p = sum06;
		d06_p++;
		*d07_p = sum07;
		d07_p++;
		*d08_p = sum08;
		d08_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			if ((i & 3) == 3)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
	}
}



void RowSumFilter_Cov_Transpose_Inverse_AVX::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + 8 * j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + 8 * j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + 8 * j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + 8 * j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + 8 * j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + 8 * j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + 8 * j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + 8 * j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + 8 * j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;
		*d04_p = sum04;
		d04_p++;
		*d05_p = sum05;
		d05_p++;
		*d06_p = sum06;
		d06_p++;
		*d07_p = sum07;
		d07_p++;
		*d08_p = sum08;
		d08_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
	}
}

void RowSumFilter_Cov_Transpose_Inverse_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + 8 * j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + 8 * j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + 8 * j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + 8 * j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + 8 * j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + 8 * j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + 8 * j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + 8 * j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + 8 * j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;
		*d04_p = sum04;
		d04_p++;
		*d05_p = sum05;
		d05_p++;
		*d06_p = sum06;
		d06_p++;
		*d07_p = sum07;
		d07_p++;
		*d08_p = sum08;
		d08_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
	}
}

void RowSumFilter_Cov_Transpose_Inverse_AVX::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(0) + 8 * j;	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(0) + 8 * j;	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(0) + 8 * j;	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(0) + 8 * j;	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(0) + 8 * j;	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(0) + 8 * j;	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(0) + 8 * j;	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(0) + 8 * j;	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(0) + 8 * j;	// corr_I_rr

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += (*s00_p1 * *s00_p1) * (r + 1);
		sum04 += (*s00_p1 * *s01_p1) * (r + 1);
		sum05 += (*s00_p1 * *s02_p1) * (r + 1);
		sum06 += (*s01_p1 * *s01_p1) * (r + 1);
		sum07 += (*s01_p1 * *s02_p1) * (r + 1);
		sum08 += (*s02_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;

			sum01 += *s01_p2;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;

			sum02 += *s02_p2;
			sum08 += *s02_p2 * *s02_p2;
			s02_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;
		*d04_p = sum04;
		d04_p++;
		*d05_p = sum05;
		d05_p++;
		*d06_p = sum06;
		d06_p++;
		*d07_p = sum07;
		d07_p++;
		*d08_p = sum08;
		d08_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p2++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;
			s00_p2++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;
			s01_p2++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;
			s02_p2++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d03_p = sum03;
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d04_p = sum04;
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d05_p = sum05;
			s00_p1++;

			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d06_p = sum06;
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d07_p = sum07;
			s01_p1++;

			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d08_p = sum08;
			s02_p1++;

			if ((i & 7) == 7)
			{
				d00_p += step;
				d01_p += step;
				d02_p += step;
				d03_p += step;
				d04_p += step;
				d05_p += step;
				d06_p += step;
				d07_p += step;
				d08_p += step;
			}
			else
			{
				d00_p++;
				d01_p++;
				d02_p++;
				d03_p++;
				d04_p++;
				d05_p++;
				d06_p++;
				d07_p++;
				d08_p++;
			}
		}
	}
}



/* --- Guide1 --- */
void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec::filter_naive_impl()
{
	for (int j = 0; j < img_col; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 1;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 1;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		float sum0 = 0.f, sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum1 += *v1_p2;
			v1_p2++;
		}
		*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
		*b_p = (sum0 * div) - *a_p * *meanI_p;

		meanI_p++;
		var_p++;
		a_p++;
		b_p++;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum0 -= *v0_p1;
			sum1 += *v1_p2;
			v1_p2++;
			sum1 -= *v1_p1;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum0 -= *v0_p1;
			v0_p1++;
			sum1 += *v1_p2;
			v1_p2++;
			sum1 -= *v1_p1;
			v1_p1++;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1++;
			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1++;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_col; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 1;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 1;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		float sum0 = 0.f, sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum1 += *v1_p2;
			v1_p2++;
		}
		*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
		*b_p = (sum0 * div) - *a_p * *meanI_p;

		meanI_p++;
		var_p++;
		a_p++;
		b_p++;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum0 -= *v0_p1;
			sum1 += *v1_p2;
			v1_p2++;
			sum1 -= *v1_p1;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum0 -= *v0_p1;
			v0_p1++;
			sum1 += *v1_p2;
			v1_p2++;
			sum1 -= *v1_p1;
			v1_p1++;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1++;
			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1++;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 1;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 1;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		float sum0 = 0.f, sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum1 += *v1_p2;
			v1_p2++;
		}
		*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
		*b_p = (sum0 * div) - *a_p * *meanI_p;

		meanI_p++;
		var_p++;
		a_p++;
		b_p++;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum0 -= *v0_p1;
			sum1 += *v1_p2;
			v1_p2++;
			sum1 -= *v1_p1;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2++;
			sum0 -= *v0_p1;
			v0_p1++;
			sum1 += *v1_p2;
			v1_p2++;
			sum1 -= *v1_p1;
			v1_p1++;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1++;
			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1++;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p++;
			var_p++;
			a_p++;
			b_p++;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_SSE::filter_naive_impl()
{
	for (int j = 0; j < img_col; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 4;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 4;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
		}
		m0 = _mm_load_ps(meanI_p);		// mean_I
		meanI_p += 4;
		m1 = _mm_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm_load_ps(var_p);			// var_I + eps
		var_p += 4;
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm_div_ps(m4, m3);			// a
		m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

		_mm_store_ps(a_p, m5);
		a_p += 4;
		_mm_store_ps(b_p, m6);
		b_p += 4;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}
		for (int i = r + 1; i < img_row / 4 - r - 1; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(v0_p1));
			v0_p1 += 4;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(v1_p1));
			v1_p1 += 4;

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int i = img_row / 4 - r - 1; i < img_row / 4; i++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(v0_p1));
			v0_p1 += 4;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(v1_p1));
			v1_p1 += 4;

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_col; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 4;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 4;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
		}
		m0 = _mm_load_ps(meanI_p);		// mean_I
		meanI_p += 4;
		m1 = _mm_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm_load_ps(var_p);			// var_I + eps
		var_p += 4;
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm_div_ps(m4, m3);			// a
		m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

		_mm_store_ps(a_p, m5);
		a_p += 4;
		_mm_store_ps(b_p, m6);
		b_p += 4;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}
		for (int i = r + 1; i < img_row / 4 - r - 1; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(v0_p1));
			v0_p1 += 4;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(v1_p1));
			v1_p1 += 4;

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int i = img_row / 4 - r - 1; i < img_row / 4; i++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(v0_p1));
			v0_p1 += 4;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(v1_p1));
			v1_p1 += 4;

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_SSE::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 4;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 4;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
		}
		m0 = _mm_load_ps(meanI_p);		// mean_I
		meanI_p += 4;
		m1 = _mm_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm_load_ps(var_p);			// var_I + eps
		var_p += 4;
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm_div_ps(m4, m3);			// a
		m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

		_mm_store_ps(a_p, m5);
		a_p += 4;
		_mm_store_ps(b_p, m6);
		b_p += 4;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}
		for (int i = r + 1; i < img_row / 4 - r - 1; i++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += 4;
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(v0_p1));
			v0_p1 += 4;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += 4;
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(v1_p1));
			v1_p1 += 4;

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int i = img_row / 4 - r - 1; i < img_row / 4; i++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(v0_p1));
			v0_p1 += 4;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(v1_p1));
			v1_p1 += 4;

			m0 = _mm_load_ps(meanI_p);		//mean_I
			meanI_p += 4;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_load_ps(var_p);		//var_I + eps
			var_p += 4;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_store_ps(a_p, m5);
			a_p += 4;
			_mm_store_ps(b_p, m6);
			b_p += 4;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_AVX::filter_naive_impl()
{
	for (int j = 0; j < img_col; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 8;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 8;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
		}
		m0 = _mm256_load_ps(meanI_p);		// mean_I
		meanI_p += 8;
		m1 = _mm256_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm256_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm256_load_ps(var_p);			// var_I + eps
		var_p += 8;
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm256_div_ps(m4, m3);			// a
		m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

		_mm256_store_ps(a_p, m5);
		a_p += 8;
		_mm256_store_ps(b_p, m6);
		b_p += 8;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}
		for (int i = r + 1; i < img_row / 8 - r - 1; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(v0_p1));
			v0_p1 += 8;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(v1_p1));
			v1_p1 += 8;

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int i = img_row / 8 - r - 1; i < img_row / 8; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(v0_p1));
			v0_p1 += 8;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(v1_p1));
			v1_p1 += 8;

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_col; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 8;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 8;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
		}
		m0 = _mm256_load_ps(meanI_p);		// mean_I
		meanI_p += 8;
		m1 = _mm256_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm256_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm256_load_ps(var_p);			// var_I + eps
		var_p += 8;
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm256_div_ps(m4, m3);			// a
		m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

		_mm256_store_ps(a_p, m5);
		a_p += 8;
		_mm256_store_ps(b_p, m6);
		b_p += 8;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}
		for (int i = r + 1; i < img_row / 8 - r - 1; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(v0_p1));
			v0_p1 += 8;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(v1_p1));
			v1_p1 += 8;

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int i = img_row / 8 - r - 1; i < img_row / 8; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(v0_p1));
			v0_p1 += 8;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(v1_p1));
			v1_p1 += 8;

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_AVX::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(j);
		float* v1_p1 = tempVec[1].ptr<float>(j);
		float* v0_p2 = tempVec[0].ptr<float>(j) + 8;
		float* v1_p2 = tempVec[1].ptr<float>(j) + 8;

		float* var_p = var.ptr<float>(j);
		float* meanI_p = mean_I.ptr<float>(j);

		float* a_p = a.ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
		}
		m0 = _mm256_load_ps(meanI_p);		// mean_I
		meanI_p += 8;
		m1 = _mm256_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm256_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm256_load_ps(var_p);			// var_I + eps
		var_p += 8;
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm256_div_ps(m4, m3);			// a
		m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

		_mm256_store_ps(a_p, m5);
		a_p += 8;
		_mm256_store_ps(b_p, m6);
		b_p += 8;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int i = 1; i <= r; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}
		for (int i = r + 1; i < img_row / 8 - r - 1; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += 8;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(v0_p1));
			v0_p1 += 8;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += 8;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(v1_p1));
			v1_p1 += 8;

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int i = img_row / 8 - r - 1; i < img_row / 8; i++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(v0_p1));
			v0_p1 += 8;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(v1_p1));
			v1_p1 += 8;

			m0 = _mm256_load_ps(meanI_p);		//mean_I
			meanI_p += 8;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_load_ps(var_p);		//var_I + eps
			var_p += 8;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_store_ps(a_p, m5);
			a_p += 8;
			_mm256_store_ps(b_p, m6);
			b_p += 8;
		}
	}
}



/* --- Guide3 --- */
void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec::filter_naive_impl()
{
	for (int j = 0; j < img_col; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 1;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 1;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 1;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 1;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		float sum0, sum1, sum2, sum3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		float meanp, corr_Ip_b, corr_Ip_g, corr_Ip_r, cov_Ip_b, cov_Ip_g, cov_Ip_r;

		sum0 = (r + 1) * *s00_p1;
		sum1 = (r + 1) * *s01_p1;
		sum2 = (r + 1) * *s02_p1;
		sum3 = (r + 1) * *s03_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum1 += *s01_p2;
			s01_p2++;
			sum2 += *s02_p2;
			s02_p2++;
			sum3 += *s03_p2;
			s03_p2++;
		}
		meanp = sum0 * div;
		corr_Ip_b = sum1 * div;
		corr_Ip_g = sum2 * div;
		corr_Ip_r = sum3 * div;
		cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
		cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
		cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

		*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
		*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
		*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
		*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

		meanIb_p++;
		meanIg_p++;
		meanIr_p++;

		c0_p++;
		c1_p++;
		c2_p++;
		c4_p++;
		c5_p++;
		c8_p++;
		det_p++;

		a_b_p++;
		a_g_p++;
		a_r_p++;
		b_p++;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2++;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2++;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2++;
			sum3 -= *s03_p1;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum0 -= *s00_p1;
			s00_p1++;
			sum1 += *s01_p2;
			s01_p2++;
			sum1 -= *s01_p1;
			s01_p1++;
			sum2 += *s02_p2;
			s02_p2++;
			sum2 -= *s02_p1;
			s02_p1++;
			sum3 += *s03_p2;
			s03_p2++;
			sum3 -= *s03_p1;
			s03_p1++;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1++;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1++;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1++;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1++;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_col; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 1;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 1;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 1;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 1;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		float sum0, sum1, sum2, sum3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		float meanp, corr_Ip_b, corr_Ip_g, corr_Ip_r, cov_Ip_b, cov_Ip_g, cov_Ip_r;

		sum0 = (r + 1) * *s00_p1;
		sum1 = (r + 1) * *s01_p1;
		sum2 = (r + 1) * *s02_p1;
		sum3 = (r + 1) * *s03_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum1 += *s01_p2;
			s01_p2++;
			sum2 += *s02_p2;
			s02_p2++;
			sum3 += *s03_p2;
			s03_p2++;
		}
		meanp = sum0 * div;
		corr_Ip_b = sum1 * div;
		corr_Ip_g = sum2 * div;
		corr_Ip_r = sum3 * div;
		cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
		cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
		cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

		*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
		*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
		*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
		*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

		meanIb_p++;
		meanIg_p++;
		meanIr_p++;

		c0_p++;
		c1_p++;
		c2_p++;
		c4_p++;
		c5_p++;
		c8_p++;
		det_p++;

		a_b_p++;
		a_g_p++;
		a_r_p++;
		b_p++;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2++;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2++;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2++;
			sum3 -= *s03_p1;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum0 -= *s00_p1;
			s00_p1++;
			sum1 += *s01_p2;
			s01_p2++;
			sum1 -= *s01_p1;
			s01_p1++;
			sum2 += *s02_p2;
			s02_p2++;
			sum2 -= *s02_p1;
			s02_p1++;
			sum3 += *s03_p2;
			s03_p2++;
			sum3 -= *s03_p1;
			s03_p1++;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1++;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1++;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1++;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1++;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 1;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 1;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 1;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 1;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		float sum0, sum1, sum2, sum3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		float meanp, corr_Ip_b, corr_Ip_g, corr_Ip_r, cov_Ip_b, cov_Ip_g, cov_Ip_r;

		sum0 = (r + 1) * *s00_p1;
		sum1 = (r + 1) * *s01_p1;
		sum2 = (r + 1) * *s02_p1;
		sum3 = (r + 1) * *s03_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum1 += *s01_p2;
			s01_p2++;
			sum2 += *s02_p2;
			s02_p2++;
			sum3 += *s03_p2;
			s03_p2++;
		}
		meanp = sum0 * div;
		corr_Ip_b = sum1 * div;
		corr_Ip_g = sum2 * div;
		corr_Ip_r = sum3 * div;
		cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
		cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
		cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

		*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
		*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
		*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
		*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

		meanIb_p++;
		meanIg_p++;
		meanIr_p++;

		c0_p++;
		c1_p++;
		c2_p++;
		c4_p++;
		c5_p++;
		c8_p++;
		det_p++;

		a_b_p++;
		a_g_p++;
		a_r_p++;
		b_p++;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2++;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2++;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2++;
			sum3 -= *s03_p1;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2++;
			sum0 -= *s00_p1;
			s00_p1++;
			sum1 += *s01_p2;
			s01_p2++;
			sum1 -= *s01_p1;
			s01_p1++;
			sum2 += *s02_p2;
			s02_p2++;
			sum2 -= *s02_p1;
			s02_p1++;
			sum3 += *s03_p2;
			s03_p2++;
			sum3 -= *s03_p1;
			s03_p1++;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1++;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1++;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1++;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1++;

			meanp = sum0 * div;
			corr_Ip_b = sum1 * div;
			corr_Ip_g = sum2 * div;
			corr_Ip_r = sum3 * div;
			cov_Ip_b = corr_Ip_b - *meanIb_p * meanp;
			cov_Ip_g = corr_Ip_g - *meanIg_p * meanp;
			cov_Ip_r = corr_Ip_r - *meanIr_p * meanp;

			*a_b_p = *det_p * (cov_Ip_b * *c0_p + cov_Ip_g * *c1_p + cov_Ip_r * *c2_p);
			*a_g_p = *det_p * (cov_Ip_b * *c1_p + cov_Ip_g * *c4_p + cov_Ip_r * *c5_p);
			*a_r_p = *det_p * (cov_Ip_b * *c2_p + cov_Ip_g * *c5_p + cov_Ip_r * *c8_p);
			*b_p = meanp - *a_b_p * *meanIb_p - *a_g_p * *meanIg_p - *a_r_p * *meanIr_p;

			meanIb_p++;
			meanIg_p++;
			meanIr_p++;

			c0_p++;
			c1_p++;
			c2_p++;
			c4_p++;
			c5_p++;
			c8_p++;
			det_p++;

			a_b_p++;
			a_g_p++;
			a_r_p++;
			b_p++;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_SSE::filter_naive_impl()
{
	for (int j = 0; j < img_col; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 4;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 4;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 4;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 4;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m128 mSum00, mSum01, mSum02, mSum03;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m128 mCov00, mCov01, mCov02;
		__m128 mC0, mC1, mC2, mC4, mC5, mC8;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
		}
		mTmp00 = _mm_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += 4;
		mTmp01 = _mm_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += 4;
		mTmp02 = _mm_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += 4;

		mTmp03 = _mm_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm_load_ps(c0_p);
		c0_p += 4;
		mC1 = _mm_load_ps(c1_p);
		c1_p += 4;
		mC2 = _mm_load_ps(c2_p);
		c2_p += 4;
		mC4 = _mm_load_ps(c4_p);
		c4_p += 4;
		mC5 = _mm_load_ps(c5_p);
		c5_p += 4;
		mC8 = _mm_load_ps(c8_p);
		c8_p += 4;

		mDet = _mm_load_ps(det_p);
		det_p += 4;

		/*
		mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm_mul_ps(mTmp04, mDet);
		//_mm_store_ps(a_b_p, mTmp04);
		_mm_stream_ps(a_b_p, mTmp04);
		a_b_p += 4;

		/*
		mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm_mul_ps(mTmp05, mDet);
		//_mm_store_ps(a_g_p, mTmp05);
		_mm_stream_ps(a_g_p, mTmp05);
		a_g_p += 4;

		/*
		mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm_mul_ps(mTmp06, mDet);
		//_mm_store_ps(a_r_p, mTmp06);
		_mm_stream_ps(a_r_p, mTmp06);
		a_r_p += 4;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += 4;

		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}

		for (int i = r + 1; i < img_row / 4 - r - 1; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += 4;

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}

		for (int i = img_row / 4 - r - 1; i < img_row / 4; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += 4;

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_col; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 4;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 4;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 4;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 4;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m128 mSum00, mSum01, mSum02, mSum03;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m128 mCov00, mCov01, mCov02;
		__m128 mC0, mC1, mC2, mC4, mC5, mC8;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
		}
		mTmp00 = _mm_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += 4;
		mTmp01 = _mm_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += 4;
		mTmp02 = _mm_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += 4;

		mTmp03 = _mm_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm_load_ps(c0_p);
		c0_p += 4;
		mC1 = _mm_load_ps(c1_p);
		c1_p += 4;
		mC2 = _mm_load_ps(c2_p);
		c2_p += 4;
		mC4 = _mm_load_ps(c4_p);
		c4_p += 4;
		mC5 = _mm_load_ps(c5_p);
		c5_p += 4;
		mC8 = _mm_load_ps(c8_p);
		c8_p += 4;

		mDet = _mm_load_ps(det_p);
		det_p += 4;

		/*
		mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm_mul_ps(mTmp04, mDet);
		//_mm_store_ps(a_b_p, mTmp04);
		_mm_stream_ps(a_b_p, mTmp04);
		a_b_p += 4;

		/*
		mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm_mul_ps(mTmp05, mDet);
		//_mm_store_ps(a_g_p, mTmp05);
		_mm_stream_ps(a_g_p, mTmp05);
		a_g_p += 4;

		/*
		mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm_mul_ps(mTmp06, mDet);
		//_mm_store_ps(a_r_p, mTmp06);
		_mm_stream_ps(a_r_p, mTmp06);
		a_r_p += 4;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += 4;

		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}

		for (int i = r + 1; i < img_row / 4 - r - 1; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += 4;

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}

		for (int i = img_row / 4 - r - 1; i < img_row / 4; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += 4;

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_SSE::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 4;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 4;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 4;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 4;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m128 mSum00, mSum01, mSum02, mSum03;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m128 mCov00, mCov01, mCov02;
		__m128 mC0, mC1, mC2, mC4, mC5, mC8;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
		}
		mTmp00 = _mm_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += 4;
		mTmp01 = _mm_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += 4;
		mTmp02 = _mm_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += 4;

		mTmp03 = _mm_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm_load_ps(c0_p);
		c0_p += 4;
		mC1 = _mm_load_ps(c1_p);
		c1_p += 4;
		mC2 = _mm_load_ps(c2_p);
		c2_p += 4;
		mC4 = _mm_load_ps(c4_p);
		c4_p += 4;
		mC5 = _mm_load_ps(c5_p);
		c5_p += 4;
		mC8 = _mm_load_ps(c8_p);
		c8_p += 4;

		mDet = _mm_load_ps(det_p);
		det_p += 4;

		/*
		mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm_mul_ps(mTmp04, mDet);
		//_mm_store_ps(a_b_p, mTmp04);
		_mm_stream_ps(a_b_p, mTmp04);
		a_b_p += 4;

		/*
		mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm_mul_ps(mTmp05, mDet);
		//_mm_store_ps(a_g_p, mTmp05);
		_mm_stream_ps(a_g_p, mTmp05);
		a_g_p += 4;

		/*
		mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm_mul_ps(mTmp06, mDet);
		//_mm_store_ps(a_r_p, mTmp06);
		_mm_stream_ps(a_r_p, mTmp06);
		a_r_p += 4;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += 4;

		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}

		for (int i = r + 1; i < img_row / 4 - r - 1; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			s00_p2 += 4;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			s01_p2 += 4;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			s02_p2 += 4;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			s03_p2 += 4;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += 4;

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}

		for (int i = img_row / 4 - r - 1; i < img_row / 4; i++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_load_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += 4;
			mSum01 = _mm_add_ps(mSum01, _mm_load_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += 4;
			mSum02 = _mm_add_ps(mSum02, _mm_load_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += 4;
			mSum03 = _mm_add_ps(mSum03, _mm_load_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += 4;

			mTmp00 = _mm_load_ps(meanIb_p);
			meanIb_p += 4;
			mTmp01 = _mm_load_ps(meanIg_p);
			meanIg_p += 4;
			mTmp02 = _mm_load_ps(meanIr_p);
			meanIr_p += 4;

			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += 4;
			mC1 = _mm_load_ps(c1_p);
			c1_p += 4;
			mC2 = _mm_load_ps(c2_p);
			c2_p += 4;
			mC4 = _mm_load_ps(c4_p);
			c4_p += 4;
			mC5 = _mm_load_ps(c5_p);
			c5_p += 4;
			mC8 = _mm_load_ps(c8_p);
			c8_p += 4;

			mDet = _mm_load_ps(det_p);
			det_p += 4;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += 4;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += 4;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += 4;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += 4;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_AVX::filter_naive_impl()
{
	for (int j = 0; j < img_col; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 8;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 8;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 8;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 8;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m256 mSum00, mSum01, mSum02, mSum03;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m256 mCov00, mCov01, mCov02;
		__m256 mC0, mC1, mC2, mC4, mC5, mC8;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
		}
		mTmp00 = _mm256_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += 8;
		mTmp01 = _mm256_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += 8;
		mTmp02 = _mm256_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += 8;

		mTmp03 = _mm256_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm256_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm256_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm256_load_ps(c0_p);
		c0_p += 8;
		mC1 = _mm256_load_ps(c1_p);
		c1_p += 8;
		mC2 = _mm256_load_ps(c2_p);
		c2_p += 8;
		mC4 = _mm256_load_ps(c4_p);
		c4_p += 8;
		mC5 = _mm256_load_ps(c5_p);
		c5_p += 8;
		mC8 = _mm256_load_ps(c8_p);
		c8_p += 8;

		mDet = _mm256_load_ps(det_p);
		det_p += 8;

		/*
		mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm256_mul_ps(mTmp04, mDet);
		//_mm256_store_ps(a_b_p, mTmp04);
		_mm256_stream_ps(a_b_p, mTmp04);
		a_b_p += 8;

		/*
		mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm256_mul_ps(mTmp05, mDet);
		//_mm256_store_ps(a_g_p, mTmp05);
		_mm256_stream_ps(a_g_p, mTmp05);
		a_g_p += 8;

		/*
		mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm256_mul_ps(mTmp06, mDet);
		//_mm256_store_ps(a_r_p, mTmp06);
		_mm256_stream_ps(a_r_p, mTmp06);
		a_r_p += 8;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += 8;

		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}

		for (int i = r + 1; i < img_row / 8 - r - 1; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += 8;

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}

		for (int i = img_row / 8 - r - 1; i < img_row / 8; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += 8;

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_col; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 8;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 8;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 8;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 8;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m256 mSum00, mSum01, mSum02, mSum03;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m256 mCov00, mCov01, mCov02;
		__m256 mC0, mC1, mC2, mC4, mC5, mC8;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
		}
		mTmp00 = _mm256_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += 8;
		mTmp01 = _mm256_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += 8;
		mTmp02 = _mm256_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += 8;

		mTmp03 = _mm256_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm256_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm256_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm256_load_ps(c0_p);
		c0_p += 8;
		mC1 = _mm256_load_ps(c1_p);
		c1_p += 8;
		mC2 = _mm256_load_ps(c2_p);
		c2_p += 8;
		mC4 = _mm256_load_ps(c4_p);
		c4_p += 8;
		mC5 = _mm256_load_ps(c5_p);
		c5_p += 8;
		mC8 = _mm256_load_ps(c8_p);
		c8_p += 8;

		mDet = _mm256_load_ps(det_p);
		det_p += 8;

		/*
		mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm256_mul_ps(mTmp04, mDet);
		//_mm256_store_ps(a_b_p, mTmp04);
		_mm256_stream_ps(a_b_p, mTmp04);
		a_b_p += 8;

		/*
		mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm256_mul_ps(mTmp05, mDet);
		//_mm256_store_ps(a_g_p, mTmp05);
		_mm256_stream_ps(a_g_p, mTmp05);
		a_g_p += 8;

		/*
		mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm256_mul_ps(mTmp06, mDet);
		//_mm256_store_ps(a_r_p, mTmp06);
		_mm256_stream_ps(a_r_p, mTmp06);
		a_r_p += 8;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += 8;

		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}

		for (int i = r + 1; i < img_row / 8 - r - 1; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += 8;

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}

		for (int i = img_row / 8 - r - 1; i < img_row / 8; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += 8;

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_AVX::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(j);	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(j);	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(j);	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(j);	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(j) + 8;
		float* s01_p2 = tempVec[1].ptr<float>(j) + 8;
		float* s02_p2 = tempVec[2].ptr<float>(j) + 8;
		float* s03_p2 = tempVec[3].ptr<float>(j) + 8;

		float* c0_p = vCov[0].ptr<float>(j);
		float* c1_p = vCov[1].ptr<float>(j);
		float* c2_p = vCov[2].ptr<float>(j);
		float* c4_p = vCov[3].ptr<float>(j);
		float* c5_p = vCov[4].ptr<float>(j);
		float* c8_p = vCov[5].ptr<float>(j);

		float* det_p = det.ptr<float>(j);
		float* meanIb_p = vMean_I[0].ptr<float>(j);
		float* meanIg_p = vMean_I[1].ptr<float>(j);
		float* meanIr_p = vMean_I[2].ptr<float>(j);

		float* a_b_p = va[0].ptr<float>(j);
		float* a_g_p = va[1].ptr<float>(j);
		float* a_r_p = va[2].ptr<float>(j);
		float* b_p = b.ptr<float>(j);

		__m256 mSum00, mSum01, mSum02, mSum03;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m256 mCov00, mCov01, mCov02;
		__m256 mC0, mC1, mC2, mC4, mC5, mC8;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
		}
		mTmp00 = _mm256_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += 8;
		mTmp01 = _mm256_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += 8;
		mTmp02 = _mm256_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += 8;

		mTmp03 = _mm256_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm256_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm256_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm256_load_ps(c0_p);
		c0_p += 8;
		mC1 = _mm256_load_ps(c1_p);
		c1_p += 8;
		mC2 = _mm256_load_ps(c2_p);
		c2_p += 8;
		mC4 = _mm256_load_ps(c4_p);
		c4_p += 8;
		mC5 = _mm256_load_ps(c5_p);
		c5_p += 8;
		mC8 = _mm256_load_ps(c8_p);
		c8_p += 8;

		mDet = _mm256_load_ps(det_p);
		det_p += 8;

		/*
		mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm256_mul_ps(mTmp04, mDet);
		//_mm256_store_ps(a_b_p, mTmp04);
		_mm256_stream_ps(a_b_p, mTmp04);
		a_b_p += 8;

		/*
		mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm256_mul_ps(mTmp05, mDet);
		//_mm256_store_ps(a_g_p, mTmp05);
		_mm256_stream_ps(a_g_p, mTmp05);
		a_g_p += 8;

		/*
		mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm256_mul_ps(mTmp06, mDet);
		//_mm256_store_ps(a_r_p, mTmp06);
		_mm256_stream_ps(a_r_p, mTmp06);
		a_r_p += 8;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += 8;

		for (int i = 1; i <= r; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}

		for (int i = r + 1; i < img_row / 8 - r - 1; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			s00_p2 += 8;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			s01_p2 += 8;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			s02_p2 += 8;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			s03_p2 += 8;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += 8;

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}

		for (int i = img_row / 8 - r - 1; i < img_row / 8; i++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_load_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += 8;
			mSum01 = _mm256_add_ps(mSum01, _mm256_load_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += 8;
			mSum02 = _mm256_add_ps(mSum02, _mm256_load_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += 8;
			mSum03 = _mm256_add_ps(mSum03, _mm256_load_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += 8;

			mTmp00 = _mm256_load_ps(meanIb_p);
			meanIb_p += 8;
			mTmp01 = _mm256_load_ps(meanIg_p);
			meanIg_p += 8;
			mTmp02 = _mm256_load_ps(meanIr_p);
			meanIr_p += 8;

			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += 8;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += 8;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += 8;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += 8;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += 8;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += 8;

			mDet = _mm256_load_ps(det_p);
			det_p += 8;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += 8;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += 8;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += 8;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += 8;
		}
	}
}
