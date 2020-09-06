#include "guidedFilter_Merge_Share.h"

using namespace std;
using namespace cv;

void guidedFilter_Merge_Share_Base::init()
{
	b.create(src.size(), CV_32F);
	if (guide.channels() == 1)
	{
		//buffer size: 6
		//
		temp.resize(2);
		for (int i = 0; i < temp.size(); i++) temp[i].create(src.size(), CV_32F);

		a.create(src.size(), CV_32F);
		var.create(src.size(), CV_32F);
		mean_I.create(src.size(), CV_32F);
	}
	else if (guide.channels() == 3)
	{
		//buffer size: 23

		temp.resize(9);
		for (int i = 0; i < temp.size(); i++) temp[i].create(src.size(), CV_32F);

		va.resize(3);
		for (int i = 0; i < va.size(); i++) va[i].create(src.size(), CV_32F);

		vCov.resize(6);
		for (int i = 0; i < vCov.size(); i++) vCov[i].create(src.size(), CV_32F);

		vMean_I.resize(3);
		for (int i = 0; i < vMean_I.size(); i++) vMean_I[i].create(src.size(), CV_32F);

		det.create(src.size(), CV_32F);
	}
}

void guidedFilter_Merge_Share_Base::compute_Var()
{
	RowSumFilter_Var			rsf_var(guide, temp, r, parallelType);				rsf_var.filter();
	ColumnSumFilter_Var_nonVec	csf_var(temp, var, mean_I, r, eps, parallelType);	csf_var.filter();
}

void guidedFilter_Merge_Share_Base::compute_Cov()
{
	RowSumFilter_Cov			rsf_cov(vI, temp, r, parallelType);							rsf_cov.filter();
	ColumnSumFilter_Cov_nonVec	csf_cov(temp, vCov, det, vMean_I, r, eps, parallelType);	csf_cov.filter();
}

void guidedFilter_Merge_Share_Base::filter()
{
	//cout << "Merge: parallel type " << parallelType << endl;
	if (src.channels() == 1 && guide.channels() == 1)
	{
		compute_Var();

		filter_Guide1(src, dest);
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		split(guide, vI);
		compute_Cov();

		filter_Guide3(src, dest);
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		compute_Var();

		split(src, vsrc);
		split(dest, vdest);
		filter_Guide1(vsrc[0], vdest[0]);
		filter_Guide1(vsrc[1], vdest[1]);
		filter_Guide1(vsrc[2], vdest[2]);

		merge(vdest, dest);
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
		split(guide, vI);
		compute_Cov();

		split(src, vsrc);
		split(dest, vdest);
		filter_Guide3(vsrc[0], vdest[0]);
		filter_Guide3(vsrc[1], vdest[1]);
		filter_Guide3(vsrc[2], vdest[2]);

		merge(vdest, dest);
	}
}

void guidedFilter_Merge_Share_Base::filterVector()
{
	//cout << "Merge: parallel type " << parallelType << endl;
	if (src.channels() == 1 && guide.channels() == 1)
	{
		compute_Var();
		filter_Guide1(vsrc[0], vdest[0]);
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		vI = vguide;
		compute_Cov();

		filter_Guide3(vsrc[0], vdest[0]);
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		compute_Var();

		filter_Guide1(vsrc[0], vdest[0]);
		filter_Guide1(vsrc[1], vdest[1]);
		filter_Guide1(vsrc[2], vdest[2]);
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
		vI = vguide;
		compute_Cov();

		filter_Guide3(vsrc[0], vdest[0]);
		filter_Guide3(vsrc[1], vdest[1]);
		filter_Guide3(vsrc[2], vdest[2]);
	}
}

void guidedFilter_Merge_Share_Base::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share				rsf_ip2ab(input, guide, temp, r, parallelType);				rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_nonVec	csf_ip2ab(temp, var, mean_I, a, b, r, parallelType);	csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1					rsf_ab2q(a, b, temp, r, parallelType);						rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_nonVec			csf_ab2q(temp, guide, output, r, parallelType);				csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Base::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share				rsf_ip2ab(input, vI, temp, r, parallelType);						rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_nonVec	csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType);	csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3					rsf_ab2q(va, b, temp, r, parallelType);								rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_nonVec			csf_ab2q(temp, vI, output, r, parallelType);						csf_ab2q.filter();
}



void guidedFilter_Merge_Share_SSE::compute_Var()
{
	RowSumFilter_Var		rsf_var(guide, temp, r, parallelType);				rsf_var.filter();
	ColumnSumFilter_Var_SSE	csf_var(temp, var, mean_I, r, eps, parallelType);	csf_var.filter();
}

void guidedFilter_Merge_Share_SSE::compute_Cov()
{
	RowSumFilter_Cov		rsf_cov(vI, temp, r, parallelType);							rsf_cov.filter();
	ColumnSumFilter_Cov_SSE	csf_cov(temp, vCov, det, vMean_I, r, eps, parallelType);	csf_cov.filter();
}

void guidedFilter_Merge_Share_SSE::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share			rsf_ip2ab(input, guide, temp, r, parallelType);				rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_SSE	csf_ip2ab(temp, var, mean_I, a, b, r, parallelType);	csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1				rsf_ab2q(a, b, temp, r, parallelType);						rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_SSE			csf_ab2q(temp, guide, output, r, parallelType);				csf_ab2q.filter();
}

void guidedFilter_Merge_Share_SSE::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share			rsf_ip2ab(input, vI, temp, r, parallelType);						rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_SSE	csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType);	csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3				rsf_ab2q(va, b, temp, r, parallelType);								rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_SSE			csf_ab2q(temp, vI, output, r, parallelType);						csf_ab2q.filter();
}


void guidedFilter_Merge_Share_AVX::compute_Var()
{
	RowSumFilter_Var		rsf_var(guide, temp, r, parallelType);				rsf_var.filter();
	ColumnSumFilter_Var_AVX	csf_var(temp, var, mean_I, r, eps, parallelType);	csf_var.filter();
}

void guidedFilter_Merge_Share_AVX::compute_Cov()
{
	RowSumFilter_Cov		rsf_cov(vI, temp, r, parallelType);							rsf_cov.filter();
	ColumnSumFilter_Cov_AVX	csf_cov(temp, vCov, det, vMean_I, r, eps, parallelType);	csf_cov.filter();
}

void guidedFilter_Merge_Share_AVX::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share			rsf_ip2ab(input, guide, temp, r, parallelType);				rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_AVX	csf_ip2ab(temp, var, mean_I, a, b, r, parallelType);	csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1				rsf_ab2q(a, b, temp, r, parallelType);						rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_AVX			csf_ab2q(temp, guide, output, r, parallelType);				csf_ab2q.filter();
}

void guidedFilter_Merge_Share_AVX::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share			rsf_ip2ab(input, vI, temp, r, parallelType);						rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_AVX	csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType);	csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3				rsf_ab2q(va, b, temp, r, parallelType);								rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_AVX			csf_ab2q(temp, vI, output, r, parallelType);						csf_ab2q.filter();
}




/* --- Compute Var --- */
void RowSumFilter_Var::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* v0_p = tempVec[0].ptr<float>(j); // mean_I
		float* v1_p = tempVec[1].ptr<float>(j); // corr_I

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
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
	}
}

void RowSumFilter_Var::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* v0_p = tempVec[0].ptr<float>(j); // mean_I
		float* v1_p = tempVec[1].ptr<float>(j); // corr_I

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
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
	}
}

void RowSumFilter_Var::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* v0_p = tempVec[0].ptr<float>(j); // mean_I
		float* v1_p = tempVec[1].ptr<float>(j); // corr_I

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
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;
			I_p2++;

			*v0_p = sum0;
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *I_p2 - *I_p1;
			sum1 += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			I_p1++;

			*v0_p = sum0;
			v0_p++;
			*v1_p = sum1;
			v1_p++;
		}
	}
}



void ColumnSumFilter_Var_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;

		float* cov_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float sum0, sum1;
		sum0 = sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
		}
		*meanI_p = sum0 * div;
		*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
		meanI_p += step;
		cov_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;

			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			v0_p1 += step;

			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1 += step;

			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
	}
}

void ColumnSumFilter_Var_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;

		float* cov_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float sum0, sum1;
		sum0 = sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
		}
		*meanI_p = sum0 * div;
		*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
		meanI_p += step;
		cov_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;

			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			v0_p1 += step;

			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1 += step;

			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
	}
}

void ColumnSumFilter_Var_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;

		float* cov_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float sum0, sum1;
		sum0 = sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
		}
		*meanI_p = sum0 * div;
		*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
		meanI_p += step;
		cov_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;

			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			v0_p1 += step;

			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1 += step;

			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*meanI_p = sum0 * div;
			*cov_p = ((sum1 * div) - (*meanI_p * *meanI_p)) + eps;
			meanI_p += step;
			cov_p += step;
		}
	}
}



void ColumnSumFilter_Var_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;

		float* cov_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2;
		__m128 mTmp[2];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
		m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
		m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

		_mm_store_ps(meanI_p, m0);
		meanI_p += step;
		_mm_store_ps(cov_p, _mm_add_ps(m2, mEps));
		cov_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}
	}
}

void ColumnSumFilter_Var_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;

		float* cov_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2;
		__m128 mTmp[2];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
		m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
		m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

		_mm_store_ps(meanI_p, m0);
		meanI_p += step;
		_mm_store_ps(cov_p, _mm_add_ps(m2, mEps));
		cov_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}
	}
}

void ColumnSumFilter_Var_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;

		float* cov_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2;
		__m128 mTmp[2];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
		m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
		m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

		_mm_store_ps(meanI_p, m0);
		meanI_p += step;
		_mm_store_ps(cov_p, _mm_add_ps(m2, mEps));
		cov_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm_sub_ps(m1, _mm_mul_ps(m0, m0));	//var_I

			_mm_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm_storeu_ps(cov_p, _mm_add_ps(m2, mEps));
			cov_p += step;
		}
	}
}



void ColumnSumFilter_Var_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;

		float* cov_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2;
		__m256 mTmp[2];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
		m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

		_mm256_store_ps(meanI_p, m0);
		meanI_p += step;
		_mm256_store_ps(cov_p, _mm256_add_ps(m2, mEps));
		cov_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}
	}
}

void ColumnSumFilter_Var_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;

		float* cov_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2;
		__m256 mTmp[2];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
		m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

		_mm256_store_ps(meanI_p, m0);
		meanI_p += step;
		_mm256_store_ps(cov_p, _mm256_add_ps(m2, mEps));
		cov_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}
	}
}

void ColumnSumFilter_Var_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;

		float* cov_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2;
		__m256 mTmp[2];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
		m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

		_mm256_store_ps(meanI_p, m0);
		meanI_p += step;
		_mm256_store_ps(cov_p, _mm256_add_ps(m2, mEps));
		cov_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	// mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	// corr_I
			m2 = _mm256_sub_ps(m1, _mm256_mul_ps(m0, m0));	//var_I

			_mm256_storeu_ps(meanI_p, m0);
			meanI_p += step;
			_mm256_storeu_ps(cov_p, _mm256_add_ps(m2, mEps));
			cov_p += step;
		}
	}
}



/* --- Compute Cov --- */
void RowSumFilter_Cov::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(j);	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(j);	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(j);	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(j);	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(j);	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(j);	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(j);	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(j);	// corr_I_rr

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
			sum01 += *s01_p2;
			sum02 += *s02_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;
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
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p2++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p2++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p2++;

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
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p1++;
			s00_p2++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p1++;
			s01_p2++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;

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
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p1++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p1++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p1++;

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
		}
	}
}

void RowSumFilter_Cov::filter_omp_impl()
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

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(j);	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(j);	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(j);	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(j);	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(j);	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(j);	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(j);	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(j);	// corr_I_rr

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
			sum01 += *s01_p2;
			sum02 += *s02_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;
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
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p2++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p2++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p2++;

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
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p1++;
			s00_p2++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p1++;
			s01_p2++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;

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
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p1++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p1++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p1++;

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
		}
	}
}

void RowSumFilter_Cov::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(j);	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(j);	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(j);	// corr_I_bb
		float* d04_p = tempVec[4].ptr<float>(j);	// corr_I_bg
		float* d05_p = tempVec[5].ptr<float>(j);	// corr_I_br
		float* d06_p = tempVec[6].ptr<float>(j);	// corr_I_gg
		float* d07_p = tempVec[7].ptr<float>(j);	// corr_I_gr
		float* d08_p = tempVec[8].ptr<float>(j);	// corr_I_rr

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
			sum01 += *s01_p2;
			sum02 += *s02_p2;
			sum03 += *s00_p2 * *s00_p2;
			sum04 += *s00_p2 * *s01_p2;
			sum05 += *s00_p2 * *s02_p2;
			s00_p2++;
			sum06 += *s01_p2 * *s01_p2;
			sum07 += *s01_p2 * *s02_p2;
			s01_p2++;
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
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p2++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p2++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p2++;

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
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p1++;
			s00_p2++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p1++;
			s01_p2++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;

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
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			sum01 += *s01_p2 - *s01_p1;
			sum02 += *s02_p2 - *s02_p1;
			sum03 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			sum04 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			sum05 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			s00_p1++;
			sum06 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			sum07 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			s01_p1++;
			sum08 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			s02_p1++;

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
		}
	}
}



void ColumnSumFilter_Cov_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i;	// c8

		float* det_p = det.ptr<float>(0) + i;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i;	// mean_I_r

		float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
		sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = sum7 = sum8 = 0.f;
		float bb, bg, br, gg, gr, rr;

		sum0 = (r + 1) * *s00_p1;
		sum1 = (r + 1) * *s01_p1;
		sum2 = (r + 1) * *s02_p1;
		sum3 = (r + 1) * *s03_p1;
		sum4 = (r + 1) * *s04_p1;
		sum5 = (r + 1) * *s05_p1;
		sum6 = (r + 1) * *s06_p1;
		sum7 = (r + 1) * *s07_p1;
		sum8 = (r + 1) * *s08_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum8 += *s08_p2;
			s08_p2 += step;
		}
		*meanIb_p = sum0 * div;
		*meanIg_p = sum1 * div;
		*meanIr_p = sum2 * div;

		bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
		bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
		br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
		gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
		gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
		rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
		det_p += step;

		*cov0_p = gg * rr - gr * gr;
		cov0_p += step;
		*cov1_p = gr * br - bg * rr;
		cov1_p += step;
		*cov2_p = bg * gr - br * gg;
		cov2_p += step;
		*cov3_p = bb * rr - br * br;
		cov3_p += step;
		*cov4_p = bg * br - bb * gr;
		cov4_p += step;
		*cov5_p = bb * gg - bg * bg;
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum4 -= *s04_p1;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum5 -= *s05_p1;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum6 -= *s06_p1;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum7 -= *s07_p1;
			sum8 += *s08_p2;
			s08_p2 += step;
			sum8 -= *s08_p1;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			s03_p1 += step;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum4 -= *s04_p1;
			s04_p1 += step;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum5 -= *s05_p1;
			s05_p1 += step;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum6 -= *s06_p1;
			s06_p1 += step;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum7 -= *s07_p1;
			s07_p1 += step;
			sum8 += *s08_p2;
			s08_p2 += step;
			sum8 -= *s08_p1;
			s08_p1 += step;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1 += step;
			sum4 += *s04_p2;
			sum4 -= *s04_p1;
			s04_p1 += step;
			sum5 += *s05_p2;
			sum5 -= *s05_p1;
			s05_p1 += step;
			sum6 += *s06_p2;
			sum6 -= *s06_p1;
			s06_p1 += step;
			sum7 += *s07_p2;
			sum7 -= *s07_p1;
			s07_p1 += step;
			sum8 += *s08_p2;
			sum8 -= *s08_p1;
			s08_p1 += step;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}
	}
}

void ColumnSumFilter_Cov_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i;	// c8

		float* det_p = det.ptr<float>(0) + i;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i;	// mean_I_r

		float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
		sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = sum7 = sum8 = 0.f;
		float bb, bg, br, gg, gr, rr;

		sum0 = (r + 1) * *s00_p1;
		sum1 = (r + 1) * *s01_p1;
		sum2 = (r + 1) * *s02_p1;
		sum3 = (r + 1) * *s03_p1;
		sum4 = (r + 1) * *s04_p1;
		sum5 = (r + 1) * *s05_p1;
		sum6 = (r + 1) * *s06_p1;
		sum7 = (r + 1) * *s07_p1;
		sum8 = (r + 1) * *s08_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum8 += *s08_p2;
			s08_p2 += step;
		}
		*meanIb_p = sum0 * div;
		*meanIg_p = sum1 * div;
		*meanIr_p = sum2 * div;

		bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
		bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
		br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
		gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
		gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
		rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
		det_p += step;

		*cov0_p = gg * rr - gr * gr;
		cov0_p += step;
		*cov1_p = gr * br - bg * rr;
		cov1_p += step;
		*cov2_p = bg * gr - br * gg;
		cov2_p += step;
		*cov3_p = bb * rr - br * br;
		cov3_p += step;
		*cov4_p = bg * br - bb * gr;
		cov4_p += step;
		*cov5_p = bb * gg - bg * bg;
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum4 -= *s04_p1;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum5 -= *s05_p1;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum6 -= *s06_p1;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum7 -= *s07_p1;
			sum8 += *s08_p2;
			s08_p2 += step;
			sum8 -= *s08_p1;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			s03_p1 += step;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum4 -= *s04_p1;
			s04_p1 += step;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum5 -= *s05_p1;
			s05_p1 += step;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum6 -= *s06_p1;
			s06_p1 += step;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum7 -= *s07_p1;
			s07_p1 += step;
			sum8 += *s08_p2;
			s08_p2 += step;
			sum8 -= *s08_p1;
			s08_p1 += step;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1 += step;
			sum4 += *s04_p2;
			sum4 -= *s04_p1;
			s04_p1 += step;
			sum5 += *s05_p2;
			sum5 -= *s05_p1;
			s05_p1 += step;
			sum6 += *s06_p2;
			sum6 -= *s06_p1;
			s06_p1 += step;
			sum7 += *s07_p2;
			sum7 -= *s07_p1;
			s07_p1 += step;
			sum8 += *s08_p2;
			sum8 -= *s08_p1;
			s08_p1 += step;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}
	}
}

void ColumnSumFilter_Cov_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i;	// c8

		float* det_p = det.ptr<float>(0) + i;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i;	// mean_I_r

		float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
		sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = sum7 = sum8 = 0.f;
		float bb, bg, br, gg, gr, rr;

		sum0 = (r + 1) * *s00_p1;
		sum1 = (r + 1) * *s01_p1;
		sum2 = (r + 1) * *s02_p1;
		sum3 = (r + 1) * *s03_p1;
		sum4 = (r + 1) * *s04_p1;
		sum5 = (r + 1) * *s05_p1;
		sum6 = (r + 1) * *s06_p1;
		sum7 = (r + 1) * *s07_p1;
		sum8 = (r + 1) * *s08_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum8 += *s08_p2;
			s08_p2 += step;
		}
		*meanIb_p = sum0 * div;
		*meanIg_p = sum1 * div;
		*meanIr_p = sum2 * div;

		bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
		bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
		br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
		gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
		gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
		rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
		det_p += step;

		*cov0_p = gg * rr - gr * gr;
		cov0_p += step;
		*cov1_p = gr * br - bg * rr;
		cov1_p += step;
		*cov2_p = bg * gr - br * gg;
		cov2_p += step;
		*cov3_p = bb * rr - br * br;
		cov3_p += step;
		*cov4_p = bg * br - bb * gr;
		cov4_p += step;
		*cov5_p = bb * gg - bg * bg;
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum4 -= *s04_p1;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum5 -= *s05_p1;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum6 -= *s06_p1;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum7 -= *s07_p1;
			sum8 += *s08_p2;
			s08_p2 += step;
			sum8 -= *s08_p1;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			s03_p1 += step;
			sum4 += *s04_p2;
			s04_p2 += step;
			sum4 -= *s04_p1;
			s04_p1 += step;
			sum5 += *s05_p2;
			s05_p2 += step;
			sum5 -= *s05_p1;
			s05_p1 += step;
			sum6 += *s06_p2;
			s06_p2 += step;
			sum6 -= *s06_p1;
			s06_p1 += step;
			sum7 += *s07_p2;
			s07_p2 += step;
			sum7 -= *s07_p1;
			s07_p1 += step;
			sum8 += *s08_p2;
			s08_p2 += step;
			sum8 -= *s08_p1;
			s08_p1 += step;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1 += step;
			sum4 += *s04_p2;
			sum4 -= *s04_p1;
			s04_p1 += step;
			sum5 += *s05_p2;
			sum5 -= *s05_p1;
			s05_p1 += step;
			sum6 += *s06_p2;
			sum6 -= *s06_p1;
			s06_p1 += step;
			sum7 += *s07_p2;
			sum7 -= *s07_p1;
			s07_p1 += step;
			sum8 += *s08_p2;
			sum8 -= *s08_p1;
			s08_p1 += step;

			*meanIb_p = sum0 * div;
			*meanIg_p = sum1 * div;
			*meanIr_p = sum2 * div;

			bb = ((sum3 * div) - (*meanIb_p * *meanIb_p)) + eps;
			bg = ((sum4 * div) - (*meanIb_p * *meanIg_p));
			br = ((sum5 * div) - (*meanIb_p * *meanIr_p));
			gg = ((sum6 * div) - (*meanIg_p * *meanIg_p)) + eps;
			gr = ((sum7 * div) - (*meanIg_p * *meanIr_p));
			rr = ((sum8 * div) - (*meanIr_p * *meanIr_p)) + eps;

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			*det_p = 1.f / ((bb * gg * rr) + (bg * gr * br) + (bg * gr * br) - (bb * gr * gr) - (bg * bg * rr) - (br * gg * br));
			det_p += step;

			*cov0_p = gg * rr - gr * gr;
			cov0_p += step;
			*cov1_p = gr * br - bg * rr;
			cov1_p += step;
			*cov2_p = bg * gr - br * gg;
			cov2_p += step;
			*cov3_p = bb * rr - br * br;
			cov3_p += step;
			*cov4_p = bg * br - bb * gr;
			cov4_p += step;
			*cov5_p = bb * gg - bg * bg;
			cov5_p += step;
		}
	}
}



void ColumnSumFilter_Cov_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 4;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 4;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 4;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 4;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 4;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 4;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 4;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 4;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 4;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 4;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i * 4;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i * 4;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i * 4;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i * 4;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i * 4;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i * 4;	// c8

		float* det_p = det.ptr<float>(0) + i * 4;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 4;	// mean_I_r

		__m128 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08;
		__m128 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();
		mSum04 = _mm_setzero_ps();
		mSum05 = _mm_setzero_ps();
		mSum06 = _mm_setzero_ps();
		mSum07 = _mm_setzero_ps();
		mSum08 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		mSum04 = _mm_mul_ps(mBorder, _mm_load_ps(s04_p1));
		mSum05 = _mm_mul_ps(mBorder, _mm_load_ps(s05_p1));
		mSum06 = _mm_mul_ps(mBorder, _mm_load_ps(s06_p1));
		mSum07 = _mm_mul_ps(mBorder, _mm_load_ps(s07_p1));
		mSum08 = _mm_mul_ps(mBorder, _mm_load_ps(s08_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
		}
		mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm_mul_ps(mSum03, mDiv);	// corr_I_bb
		mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bg
		mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_br
		mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_gg
		mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gr
		mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_rr
		_mm_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));	// var_I_bb
		mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));	// var_I_bg 
		mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));	// var_I_br
		mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));	// var_I_gg
		mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));	// var_I_gr
		mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));	// var_I_rr

		mVar00 = _mm_add_ps(mVar00, mEps);
		mVar03 = _mm_add_ps(mVar03, mEps);
		mVar05 = _mm_add_ps(mVar05, mEps);

		mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// *bb * *gg * *rr
		mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// *bg * *gr * *br
		mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// *bb * *gr * *gr
		mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// *bg * *bg * *rr
		mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// *br * *gg * *br

		mDet = _mm_add_ps(mTmp04, mTmp05);
		mDet = _mm_add_ps(mDet, mTmp05);
		mDet = _mm_sub_ps(mDet, mTmp06);
		mDet = _mm_sub_ps(mDet, mTmp07);
		mDet = _mm_sub_ps(mDet, mTmp08);	// determinant
		mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);	// 1/det
		_mm_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;

		_mm_store_ps(cov0_p, mTmp03);
		cov0_p += step;
		_mm_store_ps(cov1_p, mTmp04);
		cov1_p += step;
		_mm_store_ps(cov2_p, mTmp05);
		cov2_p += step;
		_mm_store_ps(cov3_p, mTmp06);
		cov3_p += step;
		_mm_store_ps(cov4_p, mTmp07);
		cov4_p += step;
		_mm_store_ps(cov5_p, mTmp08);
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
	}
}

void ColumnSumFilter_Cov_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 4;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 4;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 4;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 4;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 4;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 4;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 4;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 4;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 4;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 4;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i * 4;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i * 4;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i * 4;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i * 4;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i * 4;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i * 4;	// c8

		float* det_p = det.ptr<float>(0) + i * 4;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 4;	// mean_I_r

		__m128 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08;
		__m128 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();
		mSum04 = _mm_setzero_ps();
		mSum05 = _mm_setzero_ps();
		mSum06 = _mm_setzero_ps();
		mSum07 = _mm_setzero_ps();
		mSum08 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		mSum04 = _mm_mul_ps(mBorder, _mm_load_ps(s04_p1));
		mSum05 = _mm_mul_ps(mBorder, _mm_load_ps(s05_p1));
		mSum06 = _mm_mul_ps(mBorder, _mm_load_ps(s06_p1));
		mSum07 = _mm_mul_ps(mBorder, _mm_load_ps(s07_p1));
		mSum08 = _mm_mul_ps(mBorder, _mm_load_ps(s08_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
		}
		mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm_mul_ps(mSum03, mDiv);	// corr_I_bb
		mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bg
		mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_br
		mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_gg
		mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gr
		mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_rr
		_mm_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));	// var_I_bb
		mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));	// var_I_bg 
		mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));	// var_I_br
		mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));	// var_I_gg
		mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));	// var_I_gr
		mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));	// var_I_rr

		mVar00 = _mm_add_ps(mVar00, mEps);
		mVar03 = _mm_add_ps(mVar03, mEps);
		mVar05 = _mm_add_ps(mVar05, mEps);

		mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// *bb * *gg * *rr
		mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// *bg * *gr * *br
		mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// *bb * *gr * *gr
		mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// *bg * *bg * *rr
		mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// *br * *gg * *br

		mDet = _mm_add_ps(mTmp04, mTmp05);
		mDet = _mm_add_ps(mDet, mTmp05);
		mDet = _mm_sub_ps(mDet, mTmp06);
		mDet = _mm_sub_ps(mDet, mTmp07);
		mDet = _mm_sub_ps(mDet, mTmp08);	// determinant
		mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);	// 1/det
		_mm_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;

		_mm_store_ps(cov0_p, mTmp03);
		cov0_p += step;
		_mm_store_ps(cov1_p, mTmp04);
		cov1_p += step;
		_mm_store_ps(cov2_p, mTmp05);
		cov2_p += step;
		_mm_store_ps(cov3_p, mTmp06);
		cov3_p += step;
		_mm_store_ps(cov4_p, mTmp07);
		cov4_p += step;
		_mm_store_ps(cov5_p, mTmp08);
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
	}
}

void ColumnSumFilter_Cov_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 4;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 4;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 4;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 4;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 4;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 4;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 4;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 4;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 4;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 4;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i * 4;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i * 4;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i * 4;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i * 4;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i * 4;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i * 4;	// c8

		float* det_p = det.ptr<float>(0) + i * 4;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 4;	// mean_I_r

		__m128 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08;
		__m128 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();
		mSum04 = _mm_setzero_ps();
		mSum05 = _mm_setzero_ps();
		mSum06 = _mm_setzero_ps();
		mSum07 = _mm_setzero_ps();
		mSum08 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		mSum04 = _mm_mul_ps(mBorder, _mm_load_ps(s04_p1));
		mSum05 = _mm_mul_ps(mBorder, _mm_load_ps(s05_p1));
		mSum06 = _mm_mul_ps(mBorder, _mm_load_ps(s06_p1));
		mSum07 = _mm_mul_ps(mBorder, _mm_load_ps(s07_p1));
		mSum08 = _mm_mul_ps(mBorder, _mm_load_ps(s08_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
		}
		mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm_mul_ps(mSum03, mDiv);	// corr_I_bb
		mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bg
		mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_br
		mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_gg
		mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gr
		mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_rr
		_mm_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));	// var_I_bb
		mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));	// var_I_bg 
		mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));	// var_I_br
		mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));	// var_I_gg
		mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));	// var_I_gr
		mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));	// var_I_rr

		mVar00 = _mm_add_ps(mVar00, mEps);
		mVar03 = _mm_add_ps(mVar03, mEps);
		mVar05 = _mm_add_ps(mVar05, mEps);

		mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// *bb * *gg * *rr
		mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// *bg * *gr * *br
		mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// *bb * *gr * *gr
		mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// *bg * *bg * *rr
		mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// *br * *gg * *br

		mDet = _mm_add_ps(mTmp04, mTmp05);
		mDet = _mm_add_ps(mDet, mTmp05);
		mDet = _mm_sub_ps(mDet, mTmp06);
		mDet = _mm_sub_ps(mDet, mTmp07);
		mDet = _mm_sub_ps(mDet, mTmp08);	// determinant
		mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);	// 1/det
		_mm_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;

		_mm_store_ps(cov0_p, mTmp03);
		cov0_p += step;
		_mm_store_ps(cov1_p, mTmp04);
		cov1_p += step;
		_mm_store_ps(cov2_p, mTmp05);
		cov2_p += step;
		_mm_store_ps(cov3_p, mTmp06);
		cov3_p += step;
		_mm_store_ps(cov4_p, mTmp07);
		cov4_p += step;
		_mm_store_ps(cov5_p, mTmp08);
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			_mm_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
	}
}



void ColumnSumFilter_Cov_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 8;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 8;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 8;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 8;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 8;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i * 8;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i * 8;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i * 8;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i * 8;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i * 8;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i * 8;	// c8

		float* det_p = det.ptr<float>(0) + i * 8;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 8;	// mean_I_r

		__m256 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08;
		__m256 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();
		mSum04 = _mm256_setzero_ps();
		mSum05 = _mm256_setzero_ps();
		mSum06 = _mm256_setzero_ps();
		mSum07 = _mm256_setzero_ps();
		mSum08 = _mm256_setzero_ps();

		mSum00 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s00_p1), _mm256_loadu_ps(s00_p2));
		mSum01 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s01_p1), _mm256_loadu_ps(s01_p2));
		mSum02 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s02_p1), _mm256_loadu_ps(s02_p2));
		mSum03 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s03_p1), _mm256_loadu_ps(s03_p2));
		mSum04 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s04_p1), _mm256_loadu_ps(s04_p2));
		mSum05 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s05_p1), _mm256_loadu_ps(s05_p2));
		mSum06 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s06_p1), _mm256_loadu_ps(s06_p2));
		mSum07 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s07_p1), _mm256_loadu_ps(s07_p2));
		mSum08 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s08_p1), _mm256_loadu_ps(s08_p2));
		s00_p2 += step;
		s01_p2 += step;
		s02_p2 += step;
		s03_p2 += step;
		s04_p2 += step;
		s05_p2 += step;
		s06_p2 += step;
		s07_p2 += step;
		s08_p2 += step;
		for (int j = 2; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// corr_I_bb
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bg
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_br
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_gg
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gr
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_rr
		_mm256_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm256_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm256_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm256_fnmadd_ps(mTmp00, mTmp00, mTmp03);	// var_I_bb
		mVar01 = _mm256_fnmadd_ps(mTmp00, mTmp01, mTmp04);	// var_I_bg 
		mVar02 = _mm256_fnmadd_ps(mTmp00, mTmp02, mTmp05);	// var_I_br
		mVar03 = _mm256_fnmadd_ps(mTmp01, mTmp01, mTmp06);	// var_I_gg
		mVar04 = _mm256_fnmadd_ps(mTmp01, mTmp02, mTmp07);	// var_I_gr
		mVar05 = _mm256_fnmadd_ps(mTmp02, mTmp02, mTmp08);	// var_I_rr

		mVar00 = _mm256_add_ps(mVar00, mEps);
		mVar03 = _mm256_add_ps(mVar03, mEps);
		mVar05 = _mm256_add_ps(mVar05, mEps);

		mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// *bb * *gg * *rr
		mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// *bg * *gr * *br
		mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// *bb * *gr * *gr
		mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// *bg * *bg * *rr
		mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// *br * *gg * *br

		mDet = _mm256_add_ps(mTmp04, mTmp05);
		mDet = _mm256_add_ps(mDet, mTmp05);
		mDet = _mm256_sub_ps(mDet, mTmp06);
		mDet = _mm256_sub_ps(mDet, mTmp07);
		mDet = _mm256_sub_ps(mDet, mTmp08);	// determinant
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);	// 1/det
		_mm256_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;

		_mm256_store_ps(cov0_p, mTmp03);
		cov0_p += step;
		_mm256_store_ps(cov1_p, mTmp04);
		cov1_p += step;
		_mm256_store_ps(cov2_p, mTmp05);
		cov2_p += step;
		_mm256_store_ps(cov3_p, mTmp06);
		cov3_p += step;
		_mm256_store_ps(cov4_p, mTmp07);
		cov4_p += step;
		_mm256_store_ps(cov5_p, mTmp08);
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_fnmadd_ps(mTmp00, mTmp00, mTmp03);
			mVar01 = _mm256_fnmadd_ps(mTmp00, mTmp01, mTmp04);
			mVar02 = _mm256_fnmadd_ps(mTmp00, mTmp02, mTmp05);
			mVar03 = _mm256_fnmadd_ps(mTmp01, mTmp01, mTmp06);
			mVar04 = _mm256_fnmadd_ps(mTmp01, mTmp02, mTmp07);
			mVar05 = _mm256_fnmadd_ps(mTmp02, mTmp02, mTmp08);

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			/*
			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			*/
			mDet = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));// *bg * *gr * *br
			mDet = _mm256_add_ps(mDet, mDet);
			mDet = _mm256_fmadd_ps(mVar00, _mm256_mul_ps(mVar03, mVar05), mDet);// *bb * *gg * *rr
			mDet = _mm256_fnmadd_ps(mVar00, _mm256_mul_ps(mVar04, mVar04), mDet);// *bb * *gr * *gr
			mDet = _mm256_fnmadd_ps(mVar03, _mm256_mul_ps(mVar02, mVar02), mDet);// *bg * *bg * *rr
			mDet = _mm256_fnmadd_ps(mVar05, _mm256_mul_ps(mVar01, mVar01), mDet);// *br * *gg * *br

			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
	}
}

void ColumnSumFilter_Cov_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 8;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 8;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 8;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 8;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 8;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i * 8;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i * 8;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i * 8;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i * 8;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i * 8;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i * 8;	// c8

		float* det_p = det.ptr<float>(0) + i * 8;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 8;	// mean_I_r

		__m256 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08;
		__m256 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();
		mSum04 = _mm256_setzero_ps();
		mSum05 = _mm256_setzero_ps();
		mSum06 = _mm256_setzero_ps();
		mSum07 = _mm256_setzero_ps();
		mSum08 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		mSum04 = _mm256_mul_ps(mBorder, _mm256_load_ps(s04_p1));
		mSum05 = _mm256_mul_ps(mBorder, _mm256_load_ps(s05_p1));
		mSum06 = _mm256_mul_ps(mBorder, _mm256_load_ps(s06_p1));
		mSum07 = _mm256_mul_ps(mBorder, _mm256_load_ps(s07_p1));
		mSum08 = _mm256_mul_ps(mBorder, _mm256_load_ps(s08_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// corr_I_bb
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bg
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_br
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_gg
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gr
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_rr
		_mm256_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm256_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm256_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));	// var_I_bb
		mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));	// var_I_bg 
		mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));	// var_I_br
		mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));	// var_I_gg
		mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));	// var_I_gr
		mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));	// var_I_rr

		mVar00 = _mm256_add_ps(mVar00, mEps);
		mVar03 = _mm256_add_ps(mVar03, mEps);
		mVar05 = _mm256_add_ps(mVar05, mEps);

		mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// *bb * *gg * *rr
		mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// *bg * *gr * *br
		mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// *bb * *gr * *gr
		mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// *bg * *bg * *rr
		mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// *br * *gg * *br

		mDet = _mm256_add_ps(mTmp04, mTmp05);
		mDet = _mm256_add_ps(mDet, mTmp05);
		mDet = _mm256_sub_ps(mDet, mTmp06);
		mDet = _mm256_sub_ps(mDet, mTmp07);
		mDet = _mm256_sub_ps(mDet, mTmp08);	// determinant
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);	// 1/det
		_mm256_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;

		_mm256_store_ps(cov0_p, mTmp03);
		cov0_p += step;
		_mm256_store_ps(cov1_p, mTmp04);
		cov1_p += step;
		_mm256_store_ps(cov2_p, mTmp05);
		cov2_p += step;
		_mm256_store_ps(cov3_p, mTmp06);
		cov3_p += step;
		_mm256_store_ps(cov4_p, mTmp07);
		cov4_p += step;
		_mm256_store_ps(cov5_p, mTmp08);
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
	}
}

void ColumnSumFilter_Cov_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// corr_I_bb
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;	// corr_I_bg
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;	// corr_I_br
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;	// corr_I_gg
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;	// corr_I_gr
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;	// corr_I_rr

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;	// mean_I_b
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;	// mean_I_g
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;	// mean_I_r
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;	// corr_I_bb
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 8;	// corr_I_bg
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 8;	// corr_I_br
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 8;	// corr_I_gg
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 8;	// corr_I_gr
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 8;	// corr_I_rr

		float* cov0_p = vCov[0].ptr<float>(0) + i * 8;	// c0
		float* cov1_p = vCov[1].ptr<float>(0) + i * 8;	// c1
		float* cov2_p = vCov[2].ptr<float>(0) + i * 8;	// c2
		float* cov3_p = vCov[3].ptr<float>(0) + i * 8;	// c4
		float* cov4_p = vCov[4].ptr<float>(0) + i * 8;	// c5
		float* cov5_p = vCov[5].ptr<float>(0) + i * 8;	// c8

		float* det_p = det.ptr<float>(0) + i * 8;	// determinant

		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 8;	// mean_I_r

		__m256 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08;
		__m256 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();
		mSum04 = _mm256_setzero_ps();
		mSum05 = _mm256_setzero_ps();
		mSum06 = _mm256_setzero_ps();
		mSum07 = _mm256_setzero_ps();
		mSum08 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		mSum04 = _mm256_mul_ps(mBorder, _mm256_load_ps(s04_p1));
		mSum05 = _mm256_mul_ps(mBorder, _mm256_load_ps(s05_p1));
		mSum06 = _mm256_mul_ps(mBorder, _mm256_load_ps(s06_p1));
		mSum07 = _mm256_mul_ps(mBorder, _mm256_load_ps(s07_p1));
		mSum08 = _mm256_mul_ps(mBorder, _mm256_load_ps(s08_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// corr_I_bb
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bg
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_br
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_gg
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gr
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_rr
		_mm256_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm256_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm256_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));	// var_I_bb
		mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));	// var_I_bg 
		mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));	// var_I_br
		mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));	// var_I_gg
		mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));	// var_I_gr
		mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));	// var_I_rr

		mVar00 = _mm256_add_ps(mVar00, mEps);
		mVar03 = _mm256_add_ps(mVar03, mEps);
		mVar05 = _mm256_add_ps(mVar05, mEps);

		mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// *bb * *gg * *rr
		mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// *bg * *gr * *br
		mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// *bb * *gr * *gr
		mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// *bg * *bg * *rr
		mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// *br * *gg * *br

		mDet = _mm256_add_ps(mTmp04, mTmp05);
		mDet = _mm256_add_ps(mDet, mTmp05);
		mDet = _mm256_sub_ps(mDet, mTmp06);
		mDet = _mm256_sub_ps(mDet, mTmp07);
		mDet = _mm256_sub_ps(mDet, mTmp08);	// determinant
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);	// 1/det
		_mm256_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;

		_mm256_store_ps(cov0_p, mTmp03);
		cov0_p += step;
		_mm256_store_ps(cov1_p, mTmp04);
		cov1_p += step;
		_mm256_store_ps(cov2_p, mTmp05);
		cov2_p += step;
		_mm256_store_ps(cov3_p, mTmp06);
		cov3_p += step;
		_mm256_store_ps(cov4_p, mTmp07);
		cov4_p += step;
		_mm256_store_ps(cov5_p, mTmp08);
		cov5_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			s03_p1 += step;
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			s04_p1 += step;
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			s05_p1 += step;
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			s06_p1 += step;
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			s07_p1 += step;
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			s08_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			_mm256_storeu_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_storeu_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_storeu_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp02, mTmp02));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_storeu_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp03 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp03 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp04 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp05 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp06 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp07 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp08 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_storeu_ps(cov0_p, mTmp03);
			cov0_p += step;
			_mm256_storeu_ps(cov1_p, mTmp04);
			cov1_p += step;
			_mm256_storeu_ps(cov2_p, mTmp05);
			cov2_p += step;
			_mm256_storeu_ps(cov3_p, mTmp06);
			cov3_p += step;
			_mm256_storeu_ps(cov4_p, mTmp07);
			cov4_p += step;
			_mm256_storeu_ps(cov5_p, mTmp08);
			cov5_p += step;
		}
	}
}



/* --- Guide 1 --- */
void RowSumFilter_Ip2ab_Guide1_Share::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* p_p1 = p.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* p_p2 = p.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(j); // mean_p
		float* v1_p = tempVec[1].ptr<float>(j); // corr_Ip

		float sum[2] = { 0.f };
		sum[0] += *p_p1 * (r + 1);
		sum[1] += (*I_p1 * *p_p1) * (r + 1);

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *p_p2;
			sum[1] += *I_p2 * *p_p2;
			I_p2++;
			p_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide1_Share::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* p_p1 = p.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* p_p2 = p.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(j); // mean_p
		float* v1_p = tempVec[1].ptr<float>(j); // corr_Ip

		float sum[2] = { 0.f };
		sum[0] += *p_p1 * (r + 1);
		sum[1] += (*I_p1 * *p_p1) * (r + 1);

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *p_p2;
			sum[1] += *I_p2 * *p_p2;
			I_p2++;
			p_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide1_Share::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* p_p1 = p.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* p_p2 = p.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(j); // mean_p
		float* v1_p = tempVec[1].ptr<float>(j); // corr_Ip

		float sum[2] = { 0.f };
		sum[0] += *p_p1 * (r + 1);
		sum[1] += (*I_p1 * *p_p1) * (r + 1);

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *p_p2;
			sum[1] += *I_p2 * *p_p2;
			I_p2++;
			p_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *p_p2 - *p_p1;
			sum[1] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_Share_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;

		float* var_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float* a_p = a.ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum0 = 0.f, sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
		}
		*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
		*b_p = (sum0 * div) - *a_p * *meanI_p;

		meanI_p += step;
		var_p += step;
		a_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			v0_p1 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1 += step;
			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;

		float* var_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float* a_p = a.ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum0 = 0.f, sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
		}
		*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
		*b_p = (sum0 * div) - *a_p * *meanI_p;

		meanI_p += step;
		var_p += step;
		a_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			v0_p1 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1 += step;
			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;

		float* var_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float* a_p = a.ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum0 = 0.f, sum1 = 0.f;

		sum0 = (r + 1) * *v0_p1;
		sum1 = (r + 1) * *v1_p1;
		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
		}
		*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
		*b_p = (sum0 * div) - *a_p * *meanI_p;

		meanI_p += step;
		var_p += step;
		a_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *v0_p2;
			v0_p2 += step;
			sum0 -= *v0_p1;
			v0_p1 += step;
			sum1 += *v1_p2;
			v1_p2 += step;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *v0_p2;
			sum0 -= *v0_p1;
			v0_p1 += step;
			sum1 += *v1_p2;
			sum1 -= *v1_p1;
			v1_p1 += step;

			*a_p = ((sum1 * div) - (*meanI_p * (sum0 * div))) / *var_p;
			*b_p = (sum0 * div) - *a_p * *meanI_p;

			meanI_p += step;
			var_p += step;
			a_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_Share_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;

		float* var_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		float* a_p = a.ptr<float>(0) + 4 * i;
		float* b_p = b.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm_load_ps(meanI_p);		// mean_I
		meanI_p += step;
		m1 = _mm_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm_load_ps(var_p);			// var_I + eps
		var_p += step;
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm_div_ps(m4, m3);			// a
		m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

		_mm_store_ps(a_p, m5);
		a_p += step;
		_mm_store_ps(b_p, m6);
		b_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;

		float* var_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		float* a_p = a.ptr<float>(0) + 4 * i;
		float* b_p = b.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm_load_ps(meanI_p);		// mean_I
		meanI_p += step;
		m1 = _mm_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm_load_ps(var_p);			// var_I + eps
		var_p += step;
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm_div_ps(m4, m3);			// a
		m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

		_mm_store_ps(a_p, m5);
		a_p += step;
		_mm_store_ps(b_p, m6);
		b_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;

		float* var_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		float* a_p = a.ptr<float>(0) + 4 * i;
		float* b_p = b.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm_load_ps(meanI_p);		// mean_I
		meanI_p += step;
		m1 = _mm_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm_load_ps(var_p);			// var_I + eps
		var_p += step;
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm_div_ps(m4, m3);			// a
		m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

		_mm_store_ps(a_p, m5);
		a_p += step;
		_mm_store_ps(b_p, m6);
		b_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm_div_ps(m4, m3);		// a
			m6 = _mm_sub_ps(m1, _mm_mul_ps(m5, m0));	// b

			_mm_storeu_ps(a_p, m5);
			a_p += step;
			_mm_storeu_ps(b_p, m6);
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_Share_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;

		float* var_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		float* a_p = a.ptr<float>(0) + 8 * i;
		float* b_p = b.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm256_load_ps(meanI_p);		// mean_I
		meanI_p += step;
		m1 = _mm256_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm256_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm256_load_ps(var_p);			// var_I + eps
		var_p += step;
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm256_div_ps(m4, m3);			// a
		m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

		_mm256_store_ps(a_p, m5);
		a_p += step;
		_mm256_store_ps(b_p, m6);
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;

		float* var_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		float* a_p = a.ptr<float>(0) + 8 * i;
		float* b_p = b.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm256_load_ps(meanI_p);		// mean_I
		meanI_p += step;
		m1 = _mm256_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm256_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm256_load_ps(var_p);			// var_I + eps
		var_p += step;
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm256_div_ps(m4, m3);			// a
		m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

		_mm256_store_ps(a_p, m5);
		a_p += step;
		_mm256_store_ps(b_p, m6);
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_Share_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;

		float* var_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		float* a_p = a.ptr<float>(0) + 8 * i;
		float* b_p = b.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		m0 = _mm256_load_ps(meanI_p);		// mean_I
		meanI_p += step;
		m1 = _mm256_mul_ps(mSum0, mDiv);	// mean_p
		m2 = _mm256_mul_ps(mSum1, mDiv);	// corr_Ip
		m3 = _mm256_load_ps(var_p);			// var_I + eps
		var_p += step;
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

		m5 = _mm256_div_ps(m4, m3);			// a
		m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

		_mm256_store_ps(a_p, m5);
		a_p += step;
		_mm256_store_ps(b_p, m6);
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;

			m0 = _mm256_loadu_ps(meanI_p);		//mean_I
			meanI_p += step;
			m1 = _mm256_mul_ps(mSum0, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum1, mDiv);	//corr_Ip
			m3 = _mm256_loadu_ps(var_p);		//var_I + eps
			var_p += step;
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m1));	//cov_Ip

			m5 = _mm256_div_ps(m4, m3);		// a
			m6 = _mm256_sub_ps(m1, _mm256_mul_ps(m5, m0));	// b

			_mm256_storeu_ps(a_p, m5);
			a_p += step;
			_mm256_storeu_ps(b_p, m6);
			b_p += step;
		}
	}
}



/* --- Guide3 --- */
void RowSumFilter_Ip2ab_Guide3_Share::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s03_p1 = p.ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;
		float* s03_p2 = p.ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_p
		float* d01_p = tempVec[1].ptr<float>(j);	// cov_Ip_b
		float* d02_p = tempVec[2].ptr<float>(j);	// cov_Ip_g
		float* d03_p = tempVec[3].ptr<float>(j);	// cov_Ip_r

		float sum00, sum01, sum02, sum03;
		sum00 = sum01 = sum02 = sum03 = 0.f;

		sum00 += *s03_p1 * (r + 1);
		sum01 += (*s03_p1 * *s00_p1) * (r + 1);
		sum02 += (*s03_p1 * *s01_p1) * (r + 1);
		sum03 += (*s03_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s03_p2;
			sum01 += *s03_p2 * *s00_p2;
			s00_p2++;
			sum02 += *s03_p2 * *s01_p2;
			s01_p2++;
			sum03 += *s03_p2 * *s02_p2;
			s02_p2++;
			s03_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p2++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p2++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p2++;
			s03_p2++;
			*d03_p = sum03;
			d03_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			s00_p2++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			s01_p2++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;
			s03_p1++;
			s03_p2++;
			*d03_p = sum03;
			d03_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s03_p1++;
			*d03_p = sum03;
			d03_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide3_Share::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s03_p1 = p.ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;
		float* s03_p2 = p.ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_p
		float* d01_p = tempVec[1].ptr<float>(j);	// cov_Ip_b
		float* d02_p = tempVec[2].ptr<float>(j);	// cov_Ip_g
		float* d03_p = tempVec[3].ptr<float>(j);	// cov_Ip_r

		float sum00, sum01, sum02, sum03;
		sum00 = sum01 = sum02 = sum03 = 0.f;

		sum00 += *s03_p1 * (r + 1);
		sum01 += (*s03_p1 * *s00_p1) * (r + 1);
		sum02 += (*s03_p1 * *s01_p1) * (r + 1);
		sum03 += (*s03_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s03_p2;
			sum01 += *s03_p2 * *s00_p2;
			s00_p2++;
			sum02 += *s03_p2 * *s01_p2;
			s01_p2++;
			sum03 += *s03_p2 * *s02_p2;
			s02_p2++;
			s03_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p2++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p2++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p2++;
			s03_p2++;
			*d03_p = sum03;
			d03_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			s00_p2++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			s01_p2++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;
			s03_p1++;
			s03_p2++;
			*d03_p = sum03;
			d03_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s03_p1++;
			*d03_p = sum03;
			d03_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide3_Share::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s00_p1 = vI[0].ptr<float>(j);
		float* s01_p1 = vI[1].ptr<float>(j);
		float* s02_p1 = vI[2].ptr<float>(j);
		float* s03_p1 = p.ptr<float>(j);
		float* s00_p2 = vI[0].ptr<float>(j) + 1;
		float* s01_p2 = vI[1].ptr<float>(j) + 1;
		float* s02_p2 = vI[2].ptr<float>(j) + 1;
		float* s03_p2 = p.ptr<float>(j) + 1;

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_p
		float* d01_p = tempVec[1].ptr<float>(j);	// cov_Ip_b
		float* d02_p = tempVec[2].ptr<float>(j);	// cov_Ip_g
		float* d03_p = tempVec[3].ptr<float>(j);	// cov_Ip_r

		float sum00, sum01, sum02, sum03;
		sum00 = sum01 = sum02 = sum03 = 0.f;

		sum00 += *s03_p1 * (r + 1);
		sum01 += (*s03_p1 * *s00_p1) * (r + 1);
		sum02 += (*s03_p1 * *s01_p1) * (r + 1);
		sum03 += (*s03_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s03_p2;
			sum01 += *s03_p2 * *s00_p2;
			s00_p2++;
			sum02 += *s03_p2 * *s01_p2;
			s01_p2++;
			sum03 += *s03_p2 * *s02_p2;
			s02_p2++;
			s03_p2++;
		}
		*d00_p = sum00;
		d00_p++;
		*d01_p = sum01;
		d01_p++;
		*d02_p = sum02;
		d02_p++;
		*d03_p = sum03;
		d03_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p2++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p2++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p2++;
			s03_p2++;
			*d03_p = sum03;
			d03_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			s00_p2++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			s01_p2++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;
			s03_p1++;
			s03_p2++;
			*d03_p = sum03;
			d03_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s03_p2 - *s03_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			*d01_p = sum01;
			d01_p++;
			sum02 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			*d02_p = sum02;
			d02_p++;
			sum03 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s03_p1++;
			*d03_p = sum03;
			d03_p++;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_Share_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;

		float* c0_p = vCov[0].ptr<float>(0) + i;
		float* c1_p = vCov[1].ptr<float>(0) + i;
		float* c2_p = vCov[2].ptr<float>(0) + i;
		float* c4_p = vCov[3].ptr<float>(0) + i;
		float* c5_p = vCov[4].ptr<float>(0) + i;
		float* c8_p = vCov[5].ptr<float>(0) + i;

		float* det_p = det.ptr<float>(0) + i;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i;

		float* a_b_p = va[0].ptr<float>(0) + i;
		float* a_g_p = va[1].ptr<float>(0) + i;
		float* a_r_p = va[2].ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

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
			s00_p2 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
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

		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		c0_p += step;
		c1_p += step;
		c2_p += step;
		c4_p += step;
		c5_p += step;
		c8_p += step;
		det_p += step;

		a_b_p += step;
		a_g_p += step;
		a_r_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2 += step;
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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			s03_p1 += step;

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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1 += step;

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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;

		float* c0_p = vCov[0].ptr<float>(0) + i;
		float* c1_p = vCov[1].ptr<float>(0) + i;
		float* c2_p = vCov[2].ptr<float>(0) + i;
		float* c4_p = vCov[3].ptr<float>(0) + i;
		float* c5_p = vCov[4].ptr<float>(0) + i;
		float* c8_p = vCov[5].ptr<float>(0) + i;

		float* det_p = det.ptr<float>(0) + i;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i;

		float* a_b_p = va[0].ptr<float>(0) + i;
		float* a_g_p = va[1].ptr<float>(0) + i;
		float* a_r_p = va[2].ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

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
			s00_p2 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
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

		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		c0_p += step;
		c1_p += step;
		c2_p += step;
		c4_p += step;
		c5_p += step;
		c8_p += step;
		det_p += step;

		a_b_p += step;
		a_g_p += step;
		a_r_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2 += step;
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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			s03_p1 += step;

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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1 += step;

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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;

		float* c0_p = vCov[0].ptr<float>(0) + i;
		float* c1_p = vCov[1].ptr<float>(0) + i;
		float* c2_p = vCov[2].ptr<float>(0) + i;
		float* c4_p = vCov[3].ptr<float>(0) + i;
		float* c5_p = vCov[4].ptr<float>(0) + i;
		float* c8_p = vCov[5].ptr<float>(0) + i;

		float* det_p = det.ptr<float>(0) + i;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i;

		float* a_b_p = va[0].ptr<float>(0) + i;
		float* a_g_p = va[1].ptr<float>(0) + i;
		float* a_r_p = va[2].ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

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
			s00_p2 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
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

		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		c0_p += step;
		c1_p += step;
		c2_p += step;
		c4_p += step;
		c5_p += step;
		c8_p += step;
		det_p += step;

		a_b_p += step;
		a_g_p += step;
		a_r_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			sum3 += *s03_p2;
			s03_p2 += step;
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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s00_p2;
			s00_p2 += step;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			s01_p2 += step;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			s02_p2 += step;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			s03_p2 += step;
			sum3 -= *s03_p1;
			s03_p1 += step;

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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s00_p2;
			sum0 -= *s00_p1;
			s00_p1 += step;
			sum1 += *s01_p2;
			sum1 -= *s01_p1;
			s01_p1 += step;
			sum2 += *s02_p2;
			sum2 -= *s02_p1;
			s02_p1 += step;
			sum3 += *s03_p2;
			sum3 -= *s03_p1;
			s03_p1 += step;

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

			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			c0_p += step;
			c1_p += step;
			c2_p += step;
			c4_p += step;
			c5_p += step;
			c8_p += step;
			det_p += step;

			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_Share_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;

		float* c0_p = vCov[0].ptr<float>(0) + i * 4;
		float* c1_p = vCov[1].ptr<float>(0) + i * 4;
		float* c2_p = vCov[2].ptr<float>(0) + i * 4;
		float* c4_p = vCov[3].ptr<float>(0) + i * 4;
		float* c5_p = vCov[4].ptr<float>(0) + i * 4;
		float* c8_p = vCov[5].ptr<float>(0) + i * 4;

		float* det_p = det.ptr<float>(0) + i * 4;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 4;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 4;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 4;

		float* a_b_p = va[0].ptr<float>(0) + i * 4;
		float* a_g_p = va[1].ptr<float>(0) + i * 4;
		float* a_r_p = va[2].ptr<float>(0) + i * 4;
		float* b_p = b.ptr<float>(0) + i * 4;

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
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
		}
		mTmp00 = _mm_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += step;
		mTmp01 = _mm_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += step;
		mTmp02 = _mm_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += step;
		mTmp03 = _mm_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm_load_ps(c0_p);
		c0_p += step;
		mC1 = _mm_load_ps(c1_p);
		c1_p += step;
		mC2 = _mm_load_ps(c2_p);
		c2_p += step;
		mC4 = _mm_load_ps(c4_p);
		c4_p += step;
		mC5 = _mm_load_ps(c5_p);
		c5_p += step;
		mC8 = _mm_load_ps(c8_p);
		c8_p += step;

		mDet = _mm_load_ps(det_p);
		det_p += step;

		/*
		mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm_mul_ps(mTmp04, mDet);
		//_mm_store_ps(a_b_p, mTmp04);
		_mm_stream_ps(a_b_p, mTmp04);
		a_b_p += step;

		/*
		mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm_mul_ps(mTmp05, mDet);
		//_mm_store_ps(a_g_p, mTmp05);
		_mm_stream_ps(a_g_p, mTmp05);
		a_g_p += step;

		/*
		mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm_mul_ps(mTmp06, mDet);
		//_mm_store_ps(a_r_p, mTmp06);
		_mm_stream_ps(a_r_p, mTmp06);
		a_r_p += step;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;

		float* c0_p = vCov[0].ptr<float>(0) + i * 4;
		float* c1_p = vCov[1].ptr<float>(0) + i * 4;
		float* c2_p = vCov[2].ptr<float>(0) + i * 4;
		float* c4_p = vCov[3].ptr<float>(0) + i * 4;
		float* c5_p = vCov[4].ptr<float>(0) + i * 4;
		float* c8_p = vCov[5].ptr<float>(0) + i * 4;

		float* det_p = det.ptr<float>(0) + i * 4;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 4;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 4;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 4;

		float* a_b_p = va[0].ptr<float>(0) + i * 4;
		float* a_g_p = va[1].ptr<float>(0) + i * 4;
		float* a_r_p = va[2].ptr<float>(0) + i * 4;
		float* b_p = b.ptr<float>(0) + i * 4;

		__m128 mSum00, mSum01, mSum02, mSum03;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m128 mCov00, mCov01, mCov02;
		__m128 mC0, mC1, mC2, mC4, mC5, mC8;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();

		const __m128 mBorder = _mm_set1_ps((float)(r + 1));
		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
		}
		mTmp00 = _mm_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += step;
		mTmp01 = _mm_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += step;
		mTmp02 = _mm_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += step;
		mTmp03 = _mm_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm_load_ps(c0_p);
		c0_p += step;
		mC1 = _mm_load_ps(c1_p);
		c1_p += step;
		mC2 = _mm_load_ps(c2_p);
		c2_p += step;
		mC4 = _mm_load_ps(c4_p);
		c4_p += step;
		mC5 = _mm_load_ps(c5_p);
		c5_p += step;
		mC8 = _mm_load_ps(c8_p);
		c8_p += step;

		mDet = _mm_load_ps(det_p);
		det_p += step;

		/*
		mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm_mul_ps(mTmp04, mDet);
		//_mm_store_ps(a_b_p, mTmp04);
		_mm_stream_ps(a_b_p, mTmp04);
		a_b_p += step;

		/*
		mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm_mul_ps(mTmp05, mDet);
		//_mm_store_ps(a_g_p, mTmp05);
		_mm_stream_ps(a_g_p, mTmp05);
		a_g_p += step;

		/*
		mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm_mul_ps(mTmp06, mDet);
		//_mm_store_ps(a_r_p, mTmp06);
		_mm_stream_ps(a_r_p, mTmp06);
		a_r_p += step;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;

		float* c0_p = vCov[0].ptr<float>(0) + i * 4;
		float* c1_p = vCov[1].ptr<float>(0) + i * 4;
		float* c2_p = vCov[2].ptr<float>(0) + i * 4;
		float* c4_p = vCov[3].ptr<float>(0) + i * 4;
		float* c5_p = vCov[4].ptr<float>(0) + i * 4;
		float* c8_p = vCov[5].ptr<float>(0) + i * 4;

		float* det_p = det.ptr<float>(0) + i * 4;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 4;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 4;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 4;

		float* a_b_p = va[0].ptr<float>(0) + i * 4;
		float* a_g_p = va[1].ptr<float>(0) + i * 4;
		float* a_r_p = va[2].ptr<float>(0) + i * 4;
		float* b_p = b.ptr<float>(0) + i * 4;

		__m128 mSum00, mSum01, mSum02, mSum03;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m128 mCov00, mCov01, mCov02;
		__m128 mC0, mC1, mC2, mC4, mC5, mC8;
		__m128 mDet;

		mSum00 = _mm_setzero_ps();
		mSum01 = _mm_setzero_ps();
		mSum02 = _mm_setzero_ps();
		mSum03 = _mm_setzero_ps();

		const __m128 mBorder = _mm_set1_ps((float)(r + 1));
		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
		}
		mTmp00 = _mm_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += step;
		mTmp01 = _mm_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += step;
		mTmp02 = _mm_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += step;
		mTmp03 = _mm_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm_load_ps(c0_p);
		c0_p += step;
		mC1 = _mm_load_ps(c1_p);
		c1_p += step;
		mC2 = _mm_load_ps(c2_p);
		c2_p += step;
		mC4 = _mm_load_ps(c4_p);
		c4_p += step;
		mC5 = _mm_load_ps(c5_p);
		c5_p += step;
		mC8 = _mm_load_ps(c8_p);
		c8_p += step;

		mDet = _mm_load_ps(det_p);
		det_p += step;

		/*
		mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
		mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm_mul_ps(mTmp04, mDet);
		//_mm_store_ps(a_b_p, mTmp04);
		_mm_stream_ps(a_b_p, mTmp04);
		a_b_p += step;

		/*
		mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
		mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm_mul_ps(mTmp05, mDet);
		//_mm_store_ps(a_g_p, mTmp05);
		_mm_stream_ps(a_g_p, mTmp05);
		a_g_p += step;

		/*
		mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
		mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm_mul_ps(mTmp06, mDet);
		//_mm_store_ps(a_r_p, mTmp06);
		_mm_stream_ps(a_r_p, mTmp06);
		a_r_p += step;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm_sub_ps(mSum00, _mm_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm_sub_ps(mSum01, _mm_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm_sub_ps(mSum02, _mm_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm_sub_ps(mSum03, _mm_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm_mul_ps(mSum00, mDiv);
			mTmp04 = _mm_mul_ps(mSum01, mDiv);
			mTmp05 = _mm_mul_ps(mSum02, mDiv);
			mTmp06 = _mm_mul_ps(mSum03, mDiv);

			mCov00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp02, mTmp03));

			mC0 = _mm_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm_load_ps(c8_p);
			c8_p += step;

			mDet = _mm_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm_add_ps(_mm_mul_ps(mCov00, mC0), _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_add_ps(mTmp04, _mm_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm_fmadd_ps(mCov00, mC0, _mm_mul_ps(mCov01, mC1));
			mTmp04 = _mm_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm_mul_ps(mTmp04, mDet);
			_mm_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm_add_ps(_mm_mul_ps(mCov00, mC1), _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_add_ps(mTmp05, _mm_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm_fmadd_ps(mCov00, mC1, _mm_mul_ps(mCov01, mC4));
			mTmp05 = _mm_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm_mul_ps(mTmp05, mDet);
			_mm_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm_add_ps(_mm_mul_ps(mCov00, mC2), _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_add_ps(mTmp06, _mm_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm_fmadd_ps(mCov00, mC2, _mm_mul_ps(mCov01, mC5));
			mTmp06 = _mm_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm_mul_ps(mTmp06, mDet);
			_mm_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp06, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_Share_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;

		float* c0_p = vCov[0].ptr<float>(0) + i * 8;
		float* c1_p = vCov[1].ptr<float>(0) + i * 8;
		float* c2_p = vCov[2].ptr<float>(0) + i * 8;
		float* c4_p = vCov[3].ptr<float>(0) + i * 8;
		float* c5_p = vCov[4].ptr<float>(0) + i * 8;
		float* c8_p = vCov[5].ptr<float>(0) + i * 8;

		float* det_p = det.ptr<float>(0) + i * 8;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 8;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 8;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 8;

		float* a_b_p = va[0].ptr<float>(0) + i * 8;
		float* a_g_p = va[1].ptr<float>(0) + i * 8;
		float* a_r_p = va[2].ptr<float>(0) + i * 8;
		float* b_p = b.ptr<float>(0) + i * 8;

		__m256 mSum00, mSum01, mSum02, mSum03;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m256 mCov00, mCov01, mCov02;
		__m256 mC0, mC1, mC2, mC4, mC5, mC8;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();

		const __m256 mBorder = _mm256_set1_ps((float)(r + 1));
		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
		}
		mTmp00 = _mm256_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += step;
		mTmp01 = _mm256_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += step;
		mTmp02 = _mm256_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += step;
		mTmp03 = _mm256_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm256_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm256_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm256_load_ps(c0_p);
		c0_p += step;
		mC1 = _mm256_load_ps(c1_p);
		c1_p += step;
		mC2 = _mm256_load_ps(c2_p);
		c2_p += step;
		mC4 = _mm256_load_ps(c4_p);
		c4_p += step;
		mC5 = _mm256_load_ps(c5_p);
		c5_p += step;
		mC8 = _mm256_load_ps(c8_p);
		c8_p += step;

		mDet = _mm256_load_ps(det_p);
		det_p += step;

		/*
		mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm256_mul_ps(mTmp04, mDet);
		//_mm256_store_ps(a_b_p, mTmp04);
		_mm256_stream_ps(a_b_p, mTmp04);
		a_b_p += step;

		/*
		mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm256_mul_ps(mTmp05, mDet);
		//_mm256_store_ps(a_g_p, mTmp05);
		_mm256_stream_ps(a_g_p, mTmp05);
		a_g_p += step;

		/*
		mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm256_mul_ps(mTmp06, mDet);
		//_mm256_store_ps(a_r_p, mTmp06);
		_mm256_stream_ps(a_r_p, mTmp06);
		a_r_p += step;

		mTmp03 = _mm256_fnmadd_ps(mTmp04, mTmp00, mTmp03);
		mTmp03 = _mm256_fnmadd_ps(mTmp05, mTmp01, mTmp03);
		mTmp03 = _mm256_fnmadd_ps(mTmp06, mTmp02, mTmp03);
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_fnmadd_ps(mTmp00, mTmp03, mTmp04);
			mCov01 = _mm256_fnmadd_ps(mTmp01, mTmp03, mTmp05);
			mCov02 = _mm256_fnmadd_ps(mTmp02, mTmp03, mTmp06);

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_fnmadd_ps(mTmp04, mTmp00, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp05, mTmp01, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp06, mTmp02, mTmp03);
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;

		float* c0_p = vCov[0].ptr<float>(0) + i * 8;
		float* c1_p = vCov[1].ptr<float>(0) + i * 8;
		float* c2_p = vCov[2].ptr<float>(0) + i * 8;
		float* c4_p = vCov[3].ptr<float>(0) + i * 8;
		float* c5_p = vCov[4].ptr<float>(0) + i * 8;
		float* c8_p = vCov[5].ptr<float>(0) + i * 8;

		float* det_p = det.ptr<float>(0) + i * 8;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 8;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 8;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 8;

		float* a_b_p = va[0].ptr<float>(0) + i * 8;
		float* a_g_p = va[1].ptr<float>(0) + i * 8;
		float* a_r_p = va[2].ptr<float>(0) + i * 8;
		float* b_p = b.ptr<float>(0) + i * 8;

		__m256 mSum00, mSum01, mSum02, mSum03;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m256 mCov00, mCov01, mCov02;
		__m256 mC0, mC1, mC2, mC4, mC5, mC8;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();

		const __m256 mBorder = _mm256_set1_ps((float)(r + 1));
		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
		}
		mTmp00 = _mm256_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += step;
		mTmp01 = _mm256_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += step;
		mTmp02 = _mm256_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += step;
		mTmp03 = _mm256_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm256_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm256_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm256_load_ps(c0_p);
		c0_p += step;
		mC1 = _mm256_load_ps(c1_p);
		c1_p += step;
		mC2 = _mm256_load_ps(c2_p);
		c2_p += step;
		mC4 = _mm256_load_ps(c4_p);
		c4_p += step;
		mC5 = _mm256_load_ps(c5_p);
		c5_p += step;
		mC8 = _mm256_load_ps(c8_p);
		c8_p += step;

		mDet = _mm256_load_ps(det_p);
		det_p += step;

		/*
		mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm256_mul_ps(mTmp04, mDet);
		//_mm256_store_ps(a_b_p, mTmp04);
		_mm256_stream_ps(a_b_p, mTmp04);
		a_b_p += step;

		/*
		mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm256_mul_ps(mTmp05, mDet);
		//_mm256_store_ps(a_g_p, mTmp05);
		_mm256_stream_ps(a_g_p, mTmp05);
		a_g_p += step;

		/*
		mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm256_mul_ps(mTmp06, mDet);
		//_mm256_store_ps(a_r_p, mTmp06);
		_mm256_stream_ps(a_r_p, mTmp06);
		a_r_p += step;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_Share_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_p
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// corr_Ip_b
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// corr_Ip_g
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// corr_Ip_r
		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;

		float* c0_p = vCov[0].ptr<float>(0) + i * 8;
		float* c1_p = vCov[1].ptr<float>(0) + i * 8;
		float* c2_p = vCov[2].ptr<float>(0) + i * 8;
		float* c4_p = vCov[3].ptr<float>(0) + i * 8;
		float* c5_p = vCov[4].ptr<float>(0) + i * 8;
		float* c8_p = vCov[5].ptr<float>(0) + i * 8;

		float* det_p = det.ptr<float>(0) + i * 8;
		float* meanIb_p = vMean_I[0].ptr<float>(0) + i * 8;
		float* meanIg_p = vMean_I[1].ptr<float>(0) + i * 8;
		float* meanIr_p = vMean_I[2].ptr<float>(0) + i * 8;

		float* a_b_p = va[0].ptr<float>(0) + i * 8;
		float* a_g_p = va[1].ptr<float>(0) + i * 8;
		float* a_r_p = va[2].ptr<float>(0) + i * 8;
		float* b_p = b.ptr<float>(0) + i * 8;

		__m256 mSum00, mSum01, mSum02, mSum03;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06;
		__m256 mCov00, mCov01, mCov02;
		__m256 mC0, mC1, mC2, mC4, mC5, mC8;
		__m256 mDet;

		mSum00 = _mm256_setzero_ps();
		mSum01 = _mm256_setzero_ps();
		mSum02 = _mm256_setzero_ps();
		mSum03 = _mm256_setzero_ps();

		const __m256 mBorder = _mm256_set1_ps((float)(r + 1));
		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
		}
		mTmp00 = _mm256_load_ps(meanIb_p);	// mean_I_b
		meanIb_p += step;
		mTmp01 = _mm256_load_ps(meanIg_p);	// mean_I_g
		meanIg_p += step;
		mTmp02 = _mm256_load_ps(meanIr_p);	// mean_I_r
		meanIr_p += step;
		mTmp03 = _mm256_mul_ps(mSum00, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum01, mDiv);	// corr_Ip_b
		mTmp05 = _mm256_mul_ps(mSum02, mDiv);	// corr_Ip_g
		mTmp06 = _mm256_mul_ps(mSum03, mDiv);	// corr_Ip_r

		mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));	// cov_Ip_b
		mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));	// cov_Ip_g
		mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));	// cov_Ip_r

		mC0 = _mm256_load_ps(c0_p);
		c0_p += step;
		mC1 = _mm256_load_ps(c1_p);
		c1_p += step;
		mC2 = _mm256_load_ps(c2_p);
		c2_p += step;
		mC4 = _mm256_load_ps(c4_p);
		c4_p += step;
		mC5 = _mm256_load_ps(c5_p);
		c5_p += step;
		mC8 = _mm256_load_ps(c8_p);
		c8_p += step;

		mDet = _mm256_load_ps(det_p);
		det_p += step;

		/*
		mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mC2));
		*/
		mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
		mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
		mTmp04 = _mm256_mul_ps(mTmp04, mDet);
		//_mm256_store_ps(a_b_p, mTmp04);
		_mm256_stream_ps(a_b_p, mTmp04);
		a_b_p += step;

		/*
		mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
		*/
		mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
		mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
		mTmp05 = _mm256_mul_ps(mTmp05, mDet);
		//_mm256_store_ps(a_g_p, mTmp05);
		_mm256_stream_ps(a_g_p, mTmp05);
		a_g_p += step;

		/*
		mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
		*/
		mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
		mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
		mTmp06 = _mm256_mul_ps(mTmp06, mDet);
		//_mm256_store_ps(a_r_p, mTmp06);
		_mm256_stream_ps(a_r_p, mTmp06);
		a_r_p += step;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			mSum00 = _mm256_sub_ps(mSum00, _mm256_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			mSum01 = _mm256_sub_ps(mSum01, _mm256_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			mSum02 = _mm256_sub_ps(mSum02, _mm256_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			mSum03 = _mm256_sub_ps(mSum03, _mm256_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_loadu_ps(s00_p1));
			s00_p1 += step;
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_loadu_ps(s01_p1));
			s01_p1 += step;
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_loadu_ps(s02_p1));
			s02_p1 += step;
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_loadu_ps(s03_p1));
			s03_p1 += step;

			mTmp00 = _mm256_loadu_ps(meanIb_p);
			meanIb_p += step;
			mTmp01 = _mm256_loadu_ps(meanIg_p);
			meanIg_p += step;
			mTmp02 = _mm256_loadu_ps(meanIr_p);
			meanIr_p += step;
			mTmp03 = _mm256_mul_ps(mSum00, mDiv);
			mTmp04 = _mm256_mul_ps(mSum01, mDiv);
			mTmp05 = _mm256_mul_ps(mSum02, mDiv);
			mTmp06 = _mm256_mul_ps(mSum03, mDiv);

			mCov00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp02, mTmp03));

			mC0 = _mm256_load_ps(c0_p);
			c0_p += step;
			mC1 = _mm256_load_ps(c1_p);
			c1_p += step;
			mC2 = _mm256_load_ps(c2_p);
			c2_p += step;
			mC4 = _mm256_load_ps(c4_p);
			c4_p += step;
			mC5 = _mm256_load_ps(c5_p);
			c5_p += step;
			mC8 = _mm256_load_ps(c8_p);
			c8_p += step;

			mDet = _mm256_load_ps(det_p);
			det_p += step;

			/*
			mTmp04 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC0), _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_add_ps(mTmp04, _mm256_mul_ps(mCov02, mC2));
			*/
			mTmp04 = _mm256_fmadd_ps(mCov00, mC0, _mm256_mul_ps(mCov01, mC1));
			mTmp04 = _mm256_fmadd_ps(mCov02, mC2, mTmp04);
			mTmp04 = _mm256_mul_ps(mTmp04, mDet);
			_mm256_storeu_ps(a_b_p, mTmp04);
			a_b_p += step;

			/*
			mTmp05 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC1), _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_add_ps(mTmp05, _mm256_mul_ps(mCov02, mC5));
			*/
			mTmp05 = _mm256_fmadd_ps(mCov00, mC1, _mm256_mul_ps(mCov01, mC4));
			mTmp05 = _mm256_fmadd_ps(mCov02, mC5, mTmp05);
			mTmp05 = _mm256_mul_ps(mTmp05, mDet);
			_mm256_storeu_ps(a_g_p, mTmp05);
			a_g_p += step;

			/*
			mTmp06 = _mm256_add_ps(_mm256_mul_ps(mCov00, mC2), _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_add_ps(mTmp06, _mm256_mul_ps(mCov02, mC8));
			*/
			mTmp06 = _mm256_fmadd_ps(mCov00, mC2, _mm256_mul_ps(mCov01, mC5));
			mTmp06 = _mm256_fmadd_ps(mCov02, mC8, mTmp06);
			mTmp06 = _mm256_mul_ps(mTmp06, mDet);
			_mm256_storeu_ps(a_r_p, mTmp06);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp04, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp05, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp06, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}
