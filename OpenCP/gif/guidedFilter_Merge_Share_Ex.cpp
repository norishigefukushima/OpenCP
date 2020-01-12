#include "guidedFilter_Merge_Share_Ex.h"

using namespace std;
using namespace cv;

void guidedFilter_Merge_Share_Mixed_nonVec::init()
{
	b.create(src.size(), CV_32F);

	if (guide.channels() == 1)
	{
		temp.resize(4);
		for (int i = 0; i < temp.size(); i++) temp[i].create(src.size(), CV_32F);
		a.create(src.size(), CV_32F);
		var.create(src.size(), CV_32F);
		mean_I.create(src.size(), CV_32F);
	}
	else if (guide.channels() == 3)
	{
		temp.resize(13);
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

void guidedFilter_Merge_Share_Mixed_nonVec::filter()
{
	if (src.channels() == 1 && guide.channels() == 1)
	{
		filter_Src1Guide1(src, dest);
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		split(guide, vI);
		filter_Src1Guide3(src, dest);
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		split(src, vSrc);
		split(dest, vDest);
		filter_Src3Guide1_First(vSrc[0], vDest[0]);
		filter_Guide1(vSrc[1], vDest[1]);
		filter_Guide1(vSrc[2], vDest[2]);
		merge(vDest, dest);
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
		split(guide, vI);

		split(src, vSrc);
		split(dest, vDest);
		filter_Src3Guide3_First(vSrc[0], vDest[0]);
		filter_Guide3(vSrc[1], vDest[1]);
		filter_Guide3(vSrc[2], vDest[2]);
		merge(vDest, dest);
	}
}

void guidedFilter_Merge_Share_Mixed_nonVec::filterVector()
{
	if (src.channels() == 1 && guide.channels() == 1)
	{
		filter_Src1Guide1(vsrc[0], vdest[0]);
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		vI = vguide;
		filter_Src1Guide3(vsrc[0], vdest[0]);
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		filter_Src3Guide1_First(vsrc[0], vdest[0]);
		filter_Guide1(vsrc[1], vdest[1]);
		filter_Guide1(vsrc[2], vdest[2]);	
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
		vI = vguide;
		filter_Src3Guide3_First(vsrc[0], vdest[0]);
		filter_Guide3(vsrc[1], vdest[1]);
		filter_Guide3(vsrc[2], vdest[2]);
	}
}

void guidedFilter_Merge_Share_Mixed_nonVec::filter_Src1Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1 rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_nonVec csf_ip2ab(temp, a, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_nonVec csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_nonVec::filter_Src1Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3 rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_nonVec csf_ip2ab(temp, va, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_nonVec csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_nonVec::filter_Src3Guide1_First(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1 rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_First_nonVec csf_ip2ab(temp, var, mean_I, a, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_nonVec csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_nonVec::filter_Src3Guide3_First(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3 rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_First_nonVec csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_nonVec csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_nonVec::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_nonVec csf_ip2ab(temp, var, mean_I, a, b, r, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_nonVec csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_nonVec::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_nonVec csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_nonVec csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}



void guidedFilter_Merge_Share_Mixed_SSE::filter_Src1Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1 rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_SSE csf_ip2ab(temp, a, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_SSE csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_SSE::filter_Src1Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3 rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_SSE csf_ip2ab(temp, va, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_SSE csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_SSE::filter_Src3Guide1_First(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1 rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_First_SSE csf_ip2ab(temp, var, mean_I, a, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_SSE csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_SSE::filter_Src3Guide3_First(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3 rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_First_SSE csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_SSE csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_SSE::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_SSE csf_ip2ab(temp, var, mean_I, a, b, r, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_SSE csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_SSE::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_SSE csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_SSE csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}



void guidedFilter_Merge_Share_Mixed_AVX::filter_Src1Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1 rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_AVX csf_ip2ab(temp, a, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_AVX csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_AVX::filter_Src1Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3 rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_AVX csf_ip2ab(temp, va, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_AVX csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_AVX::filter_Src3Guide1_First(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1 rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_First_AVX csf_ip2ab(temp, var, mean_I, a, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_AVX csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_AVX::filter_Src3Guide3_First(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3 rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_First_AVX csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, eps, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_AVX csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_AVX::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1_Share rsf_ip2ab(input, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide1_Share_AVX csf_ip2ab(temp, var, mean_I, a, b, r, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide1 rsf_ab2q(a, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide1_AVX csf_ab2q(temp, guide, output, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_Merge_Share_Mixed_AVX::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3_Share rsf_ip2ab(input, vI, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_Ip2ab_Guide3_Share_AVX csf_ip2ab(temp, vCov, det, vMean_I, va, b, r, parallelType); csf_ip2ab.filter();
	RowSumFilter_ab2q_Guide3 rsf_ab2q(va, b, temp, r, parallelType); rsf_ab2q.filter();
	ColumnSumFilter_ab2q_Guide3_AVX csf_ab2q(temp, vI, output, r, parallelType); csf_ab2q.filter();
}



/*
 *  Guide1
 */
void ColumnSumFilter_Ip2ab_Guide1_First_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + i;

		float* cov_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float* a_p = a.ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum[4] = { 0.f };
		float tmp[4] = { 0.f };
		float var_I, cov_Ip;

		sum[0] = *v0_p1 * (r + 1);
		sum[1] = *v1_p1 * (r + 1);
		sum[2] = *v2_p1 * (r + 1);
		sum[3] = *v3_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum[0] += *v0_p2;
			v0_p2 += step;
			sum[1] += *v1_p2;
			v1_p2 += step;
			sum[2] += *v2_p2;
			v2_p2 += step;
			sum[3] += *v3_p2;
			v3_p2 += step;
		}
		tmp[0] = sum[0] * div;
		tmp[1] = sum[1] * div;
		tmp[2] = sum[2] * div;
		tmp[3] = sum[3] * div;
		var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
		cov_Ip = tmp[3] - tmp[0] * tmp[1];

		*meanI_p = tmp[0];
		*cov_p = var_I;
		*a_p = cov_Ip / var_I;
		*b_p = tmp[1] - *a_p * tmp[0];

		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p2 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p2 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p2 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p2 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p1 += step;
			v0_p2 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p1 += step;
			v1_p2 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p1 += step;
			v2_p2 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p1 += step;
			v3_p2 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p1 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p1 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p1 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p1 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_First_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + i;

		float* cov_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float* a_p = a.ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum[4] = { 0.f };
		float tmp[4] = { 0.f };
		float var_I, cov_Ip;

		sum[0] = *v0_p1 * (r + 1);
		sum[1] = *v1_p1 * (r + 1);
		sum[2] = *v2_p1 * (r + 1);
		sum[3] = *v3_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum[0] += *v0_p2;
			v0_p2 += step;
			sum[1] += *v1_p2;
			v1_p2 += step;
			sum[2] += *v2_p2;
			v2_p2 += step;
			sum[3] += *v3_p2;
			v3_p2 += step;
		}
		tmp[0] = sum[0] * div;
		tmp[1] = sum[1] * div;
		tmp[2] = sum[2] * div;
		tmp[3] = sum[3] * div;
		var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
		cov_Ip = tmp[3] - tmp[0] * tmp[1];

		*meanI_p = tmp[0];
		*cov_p = var_I;
		*a_p = cov_Ip / var_I;
		*b_p = tmp[1] - *a_p * tmp[0];

		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p2 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p2 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p2 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p2 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p1 += step;
			v0_p2 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p1 += step;
			v1_p2 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p1 += step;
			v2_p2 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p1 += step;
			v3_p2 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p1 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p1 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p1 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p1 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_First_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + i;

		float* cov_p = var.ptr<float>(0) + i;
		float* meanI_p = mean_I.ptr<float>(0) + i;

		float* a_p = a.ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum[4] = { 0.f };
		float tmp[4] = { 0.f };
		float var_I, cov_Ip;

		sum[0] = *v0_p1 * (r + 1);
		sum[1] = *v1_p1 * (r + 1);
		sum[2] = *v2_p1 * (r + 1);
		sum[3] = *v3_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum[0] += *v0_p2;
			v0_p2 += step;
			sum[1] += *v1_p2;
			v1_p2 += step;
			sum[2] += *v2_p2;
			v2_p2 += step;
			sum[3] += *v3_p2;
			v3_p2 += step;
		}
		tmp[0] = sum[0] * div;
		tmp[1] = sum[1] * div;
		tmp[2] = sum[2] * div;
		tmp[3] = sum[3] * div;
		var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
		cov_Ip = tmp[3] - tmp[0] * tmp[1];

		*meanI_p = tmp[0];
		*cov_p = var_I;
		*a_p = cov_Ip / var_I;
		*b_p = tmp[1] - *a_p * tmp[0];

		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p2 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p2 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p2 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p2 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p1 += step;
			v0_p2 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p1 += step;
			v1_p2 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p1 += step;
			v2_p2 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p1 += step;
			v3_p2 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum[0] += *v0_p2 - *v0_p1;
			v0_p1 += step;
			sum[1] += *v1_p2 - *v1_p1;
			v1_p1 += step;
			sum[2] += *v2_p2 - *v2_p1;
			v2_p1 += step;
			sum[3] += *v3_p2 - *v3_p1;
			v3_p1 += step;

			tmp[0] = sum[0] * div;
			tmp[1] = sum[1] * div;
			tmp[2] = sum[2] * div;
			tmp[3] = sum[3] * div;
			var_I = (tmp[2] - tmp[0] * tmp[0]) + eps;
			cov_Ip = tmp[3] - tmp[0] * tmp[1];

			*meanI_p = tmp[0];
			*cov_p = var_I;
			*a_p = cov_Ip / var_I;
			*b_p = tmp[1] - *a_p * tmp[0];

			meanI_p += step;
			cov_p += step;
			a_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_First_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + 4 * i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + 4 * i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + 4 * i;

		float* cov_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		float* a_p = a.ptr<float>(0) + 4 * i;
		float* b_p = b.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 mSum2 = _mm_setzero_ps();
		__m128 mSum3 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6, m7;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		mSum2 = _mm_mul_ps(mBorder, _mm_load_ps(v2_p1));
		mSum3 = _mm_mul_ps(mBorder, _mm_load_ps(v3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			mSum2 = _mm_add_ps(mSum2, _mm_load_ps(v2_p2));
			mSum3 = _mm_add_ps(mSum3, _mm_load_ps(v3_p2));
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
		}
		m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
		m4 = _mm_add_ps(m4, mEps);
		m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
		m6 = _mm_div_ps(m5, m4);
		m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

		_mm_store_ps(meanI_p, m0);
		_mm_store_ps(cov_p, m4);
		_mm_store_ps(a_p, m6);
		_mm_store_ps(b_p, m7);
		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		mTmp[2] = _mm_load_ps(v2_p1);
		mTmp[3] = _mm_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(v2_p2));
			mSum2 = _mm_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(v3_p2));
			mSum3 = _mm_sub_ps(mSum3, mTmp[3]);

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(v2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_loadu_ps(v2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(v3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_loadu_ps(v3_p1));

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		mTmp[2] = _mm_load_ps(v2_p2);
		mTmp[3] = _mm_load_ps(v3_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			mSum2 = _mm_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm_sub_ps(mSum2, _mm_loadu_ps(v2_p1));
			mSum3 = _mm_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm_sub_ps(mSum3, _mm_loadu_ps(v3_p1));

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_First_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + 4 * i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + 4 * i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + 4 * i;

		float* cov_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		float* a_p = a.ptr<float>(0) + 4 * i;
		float* b_p = b.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 mSum2 = _mm_setzero_ps();
		__m128 mSum3 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6, m7;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		mSum2 = _mm_mul_ps(mBorder, _mm_load_ps(v2_p1));
		mSum3 = _mm_mul_ps(mBorder, _mm_load_ps(v3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			mSum2 = _mm_add_ps(mSum2, _mm_load_ps(v2_p2));
			mSum3 = _mm_add_ps(mSum3, _mm_load_ps(v3_p2));
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
		}
		m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
		m4 = _mm_add_ps(m4, mEps);
		m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
		m6 = _mm_div_ps(m5, m4);
		m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

		_mm_store_ps(meanI_p, m0);
		_mm_store_ps(cov_p, m4);
		_mm_store_ps(a_p, m6);
		_mm_store_ps(b_p, m7);
		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		mTmp[2] = _mm_load_ps(v2_p1);
		mTmp[3] = _mm_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(v2_p2));
			mSum2 = _mm_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(v3_p2));
			mSum3 = _mm_sub_ps(mSum3, mTmp[3]);

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(v2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_loadu_ps(v2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(v3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_loadu_ps(v3_p1));

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		mTmp[2] = _mm_load_ps(v2_p2);
		mTmp[3] = _mm_load_ps(v3_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			mSum2 = _mm_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm_sub_ps(mSum2, _mm_loadu_ps(v2_p1));
			mSum3 = _mm_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm_sub_ps(mSum3, _mm_loadu_ps(v3_p1));

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_First_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 4 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 4 * i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + 4 * i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + 4 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 4 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 4 * i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + 4 * i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + 4 * i;

		float* cov_p = var.ptr<float>(0) + 4 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 4 * i;

		float* a_p = a.ptr<float>(0) + 4 * i;
		float* b_p = b.ptr<float>(0) + 4 * i;

		__m128 mSum0 = _mm_setzero_ps();
		__m128 mSum1 = _mm_setzero_ps();
		__m128 mSum2 = _mm_setzero_ps();
		__m128 mSum3 = _mm_setzero_ps();
		__m128 m0, m1, m2, m3, m4, m5, m6, m7;
		__m128 mTmp[4];

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		mSum2 = _mm_mul_ps(mBorder, _mm_load_ps(v2_p1));
		mSum3 = _mm_mul_ps(mBorder, _mm_load_ps(v3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_load_ps(v0_p2));
			mSum1 = _mm_add_ps(mSum1, _mm_load_ps(v1_p2));
			mSum2 = _mm_add_ps(mSum2, _mm_load_ps(v2_p2));
			mSum3 = _mm_add_ps(mSum3, _mm_load_ps(v3_p2));
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
		}
		m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
		m4 = _mm_add_ps(m4, mEps);
		m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
		m6 = _mm_div_ps(m5, m4);
		m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

		_mm_store_ps(meanI_p, m0);
		_mm_store_ps(cov_p, m4);
		_mm_store_ps(a_p, m6);
		_mm_store_ps(b_p, m7);
		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		mTmp[0] = _mm_load_ps(v0_p1);
		mTmp[1] = _mm_load_ps(v1_p1);
		mTmp[2] = _mm_load_ps(v2_p1);
		mTmp[3] = _mm_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			mSum0 = _mm_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			mSum1 = _mm_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(v2_p2));
			mSum2 = _mm_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(v3_p2));
			mSum3 = _mm_sub_ps(mSum3, mTmp[3]);

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(v0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(v1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(v2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_loadu_ps(v2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(v3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_loadu_ps(v3_p1));

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		mTmp[0] = _mm_load_ps(v0_p2);
		mTmp[1] = _mm_load_ps(v1_p2);
		mTmp[2] = _mm_load_ps(v2_p2);
		mTmp[3] = _mm_load_ps(v3_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm_sub_ps(mSum0, _mm_loadu_ps(v0_p1));
			mSum1 = _mm_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm_sub_ps(mSum1, _mm_loadu_ps(v1_p1));
			mSum2 = _mm_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm_sub_ps(mSum2, _mm_loadu_ps(v2_p1));
			mSum3 = _mm_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm_sub_ps(mSum3, _mm_loadu_ps(v3_p1));

			m0 = _mm_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm_sub_ps(m2, _mm_mul_ps(m0, m0));	//var_I
			m4 = _mm_add_ps(m4, mEps);
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm_div_ps(m5, m4);
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(meanI_p, m0);
			_mm_store_ps(cov_p, m4);
			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_First_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + 8 * i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + 8 * i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + 8 * i;

		float* cov_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		float* a_p = a.ptr<float>(0) + 8 * i;
		float* b_p = b.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 mSum2 = _mm256_setzero_ps();
		__m256 mSum3 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6, m7;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		mSum2 = _mm256_mul_ps(mBorder, _mm256_load_ps(v2_p1));
		mSum3 = _mm256_mul_ps(mBorder, _mm256_load_ps(v3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(v2_p2));
			mSum3 = _mm256_add_ps(mSum3, _mm256_load_ps(v3_p2));
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
		m4 = _mm256_add_ps(m4, mEps);
		m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
		m6 = _mm256_div_ps(m5, m4);
		m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

		_mm256_store_ps(meanI_p, m0);
		_mm256_store_ps(cov_p, m4);
		_mm256_store_ps(a_p, m6);
		_mm256_store_ps(b_p, m7);
		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		mTmp[2] = _mm256_load_ps(v2_p1);
		mTmp[3] = _mm256_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			mSum2 = _mm256_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			mSum3 = _mm256_sub_ps(mSum3, mTmp[3]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		mTmp[2] = _mm256_load_ps(v2_p2);
		mTmp[3] = _mm256_load_ps(v3_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			mSum2 = _mm256_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			mSum3 = _mm256_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_First_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + 8 * i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + 8 * i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + 8 * i;

		float* cov_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		float* a_p = a.ptr<float>(0) + 8 * i;
		float* b_p = b.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 mSum2 = _mm256_setzero_ps();
		__m256 mSum3 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6, m7;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		mSum2 = _mm256_mul_ps(mBorder, _mm256_load_ps(v2_p1));
		mSum3 = _mm256_mul_ps(mBorder, _mm256_load_ps(v3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(v2_p2));
			mSum3 = _mm256_add_ps(mSum3, _mm256_load_ps(v3_p2));
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
		m4 = _mm256_add_ps(m4, mEps);
		m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
		m6 = _mm256_div_ps(m5, m4);
		m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

		_mm256_store_ps(meanI_p, m0);
		_mm256_store_ps(cov_p, m4);
		_mm256_store_ps(a_p, m6);
		_mm256_store_ps(b_p, m7);
		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		mTmp[2] = _mm256_load_ps(v2_p1);
		mTmp[3] = _mm256_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			mSum2 = _mm256_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			mSum3 = _mm256_sub_ps(mSum3, mTmp[3]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		mTmp[2] = _mm256_load_ps(v2_p2);
		mTmp[3] = _mm256_load_ps(v3_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			mSum2 = _mm256_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			mSum3 = _mm256_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_First_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + 8 * i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + 8 * i;
		float* v2_p1 = tempVec[2].ptr<float>(0) + 8 * i;
		float* v3_p1 = tempVec[3].ptr<float>(0) + 8 * i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + 8 * i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + 8 * i;
		float* v2_p2 = tempVec[2].ptr<float>(1) + 8 * i;
		float* v3_p2 = tempVec[3].ptr<float>(1) + 8 * i;

		float* cov_p = var.ptr<float>(0) + 8 * i;
		float* meanI_p = mean_I.ptr<float>(0) + 8 * i;

		float* a_p = a.ptr<float>(0) + 8 * i;
		float* b_p = b.ptr<float>(0) + 8 * i;

		__m256 mSum0 = _mm256_setzero_ps();
		__m256 mSum1 = _mm256_setzero_ps();
		__m256 mSum2 = _mm256_setzero_ps();
		__m256 mSum3 = _mm256_setzero_ps();
		__m256 m0, m1, m2, m3, m4, m5, m6, m7;
		__m256 mTmp[4];

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		mSum2 = _mm256_mul_ps(mBorder, _mm256_load_ps(v2_p1));
		mSum3 = _mm256_mul_ps(mBorder, _mm256_load_ps(v3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_load_ps(v0_p2));
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(v2_p2));
			mSum3 = _mm256_add_ps(mSum3, _mm256_load_ps(v3_p2));
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
		m4 = _mm256_add_ps(m4, mEps);
		m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
		m6 = _mm256_div_ps(m5, m4);
		m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

		_mm256_store_ps(meanI_p, m0);
		_mm256_store_ps(cov_p, m4);
		_mm256_store_ps(a_p, m6);
		_mm256_store_ps(b_p, m7);
		meanI_p += step;
		cov_p += step;
		a_p += step;
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		mTmp[2] = _mm256_load_ps(v2_p1);
		mTmp[3] = _mm256_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			mSum2 = _mm256_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			mSum3 = _mm256_sub_ps(mSum3, mTmp[3]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			v2_p2 += step;
			v3_p2 += step;
			a_p += step;
			b_p += step;
		}

		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		mTmp[2] = _mm256_load_ps(v2_p2);
		mTmp[3] = _mm256_load_ps(v3_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, mTmp[0]);
			mSum0 = _mm256_sub_ps(mSum0, _mm256_loadu_ps(v0_p1));
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			mSum2 = _mm256_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			mSum3 = _mm256_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m4 = _mm256_add_ps(m4, mEps);
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip
			m6 = _mm256_div_ps(m5, m4);
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_store_ps(meanI_p, m0);
			_mm256_store_ps(cov_p, m4);
			_mm256_store_ps(a_p, m6);
			_mm256_store_ps(b_p, m7);
			meanI_p += step;
			cov_p += step;

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}



/*
 * Guide3
 */
void ColumnSumFilter_Ip2ab_Guide3_First_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;
		float* s04_p1 = tempVec[4].ptr<float>(0) + i;
		float* s05_p1 = tempVec[5].ptr<float>(0) + i;
		float* s06_p1 = tempVec[6].ptr<float>(0) + i;
		float* s07_p1 = tempVec[7].ptr<float>(0) + i;
		float* s08_p1 = tempVec[8].ptr<float>(0) + i;
		float* s09_p1 = tempVec[9].ptr<float>(0) + i;
		float* s10_p1 = tempVec[10].ptr<float>(0) + i;
		float* s11_p1 = tempVec[11].ptr<float>(0) + i;
		float* s12_p1 = tempVec[12].ptr<float>(0) + i;

		float* s00_p2 = tempVec[0].ptr<float>(1) + i;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i;

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

		float* a_b_p = va[0].ptr<float>(0) + i;
		float* a_g_p = va[1].ptr<float>(0) + i;
		float* a_r_p = va[2].ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08, sum09, sum10, sum11, sum12;
		float tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, tmp10, tmp11, tmp12;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = sum09 = sum10 = sum11 = sum12 = 0.f;

		float bb, bg, br, gg, gr, rr;
		float covb, covg, covr;
		float det, id;


		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += *s03_p1 * (r + 1);
		sum04 += *s04_p1 * (r + 1);
		sum05 += *s05_p1 * (r + 1);
		sum06 += *s06_p1 * (r + 1);
		sum07 += *s07_p1 * (r + 1);
		sum08 += *s08_p1 * (r + 1);
		sum09 += *s09_p1 * (r + 1);
		sum10 += *s10_p1 * (r + 1);
		sum11 += *s11_p1 * (r + 1);
		sum12 += *s12_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum00 += *s00_p2;
			s00_p2 += step;
			sum01 += *s01_p2;
			s01_p2 += step;
			sum02 += *s02_p2;
			s02_p2 += step;
			sum03 += *s03_p2;
			s03_p2 += step;
			sum04 += *s04_p2;
			s04_p2 += step;
			sum05 += *s05_p2;
			s05_p2 += step;
			sum06 += *s06_p2;
			s06_p2 += step;
			sum07 += *s07_p2;
			s07_p2 += step;
			sum08 += *s08_p2;
			s08_p2 += step;
			sum09 += *s09_p2;
			s09_p2 += step;
			sum10 += *s10_p2;
			s10_p2 += step;
			sum11 += *s11_p2;
			s11_p2 += step;
			sum12 += *s12_p2;
			s12_p2 += step;
		}
		tmp00 = sum00 * div;
		tmp01 = sum01 * div;
		tmp02 = sum02 * div;
		tmp03 = sum03 * div;
		tmp04 = sum04 * div;
		tmp05 = sum05 * div;
		tmp06 = sum06 * div;
		tmp07 = sum07 * div;
		tmp08 = sum08 * div;
		tmp09 = sum09 * div;
		tmp10 = sum10 * div;
		tmp11 = sum11 * div;
		tmp12 = sum12 * div;

		*meanIb_p = tmp00;
		*meanIg_p = tmp01;
		*meanIr_p = tmp02;
		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		bb = tmp04 - tmp00 * tmp00;
		bg = tmp05 - tmp00 * tmp01;
		br = tmp06 - tmp00 * tmp02;
		gg = tmp07 - tmp01 * tmp01;
		gr = tmp08 - tmp01 * tmp02;
		rr = tmp09 - tmp02 * tmp02;
		covb = tmp10 - tmp00 * tmp03;
		covg = tmp11 - tmp01 * tmp03;
		covr = tmp12 - tmp02 * tmp03;

		bb += eps;
		gg += eps;
		rr += eps;

		det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
		id = 1.f / det;
		*det_p = id;
		det_p += step;

		float c0 = gg*rr - gr*gr;
		float c1 = br*gr - bg*rr;
		float c2 = bg*gr - br*gg;
		float c4 = bb*rr - br*br;
		float c5 = br*bg - bb*gr;
		float c8 = bb*gg - bg*bg;

		*cov0_p = c0;
		cov0_p += step;
		*cov1_p = c1;
		cov1_p += step;
		*cov2_p = c2;
		cov2_p += step;
		*cov3_p = c4;
		cov3_p += step;
		*cov4_p = c5;
		cov4_p += step;
		*cov5_p = c8;
		cov5_p += step;

		*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
		*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
		*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
		*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
		a_b_p += step;
		a_g_p += step;
		a_r_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p2 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p2 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p2 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p2 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p2 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p2 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p2 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p2 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p2 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p2 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p2 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p2 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p2 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p1 += step;
			s00_p2 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p1 += step;
			s01_p2 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p1 += step;
			s02_p2 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p1 += step;
			s03_p2 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p1 += step;
			s04_p2 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p1 += step;
			s05_p2 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p1 += step;
			s06_p2 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p1 += step;
			s07_p2 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p1 += step;
			s08_p2 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p1 += step;
			s09_p2 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p1 += step;
			s10_p2 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p1 += step;
			s11_p2 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p1 += step;
			s12_p2 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p1 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p1 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p1 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p1 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p1 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p1 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p1 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p1 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p1 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p1 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p1 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p1 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p1 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_First_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;
		float* s04_p1 = tempVec[4].ptr<float>(0) + i;
		float* s05_p1 = tempVec[5].ptr<float>(0) + i;
		float* s06_p1 = tempVec[6].ptr<float>(0) + i;
		float* s07_p1 = tempVec[7].ptr<float>(0) + i;
		float* s08_p1 = tempVec[8].ptr<float>(0) + i;
		float* s09_p1 = tempVec[9].ptr<float>(0) + i;
		float* s10_p1 = tempVec[10].ptr<float>(0) + i;
		float* s11_p1 = tempVec[11].ptr<float>(0) + i;
		float* s12_p1 = tempVec[12].ptr<float>(0) + i;

		float* s00_p2 = tempVec[0].ptr<float>(1) + i;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i;

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

		float* a_b_p = va[0].ptr<float>(0) + i;
		float* a_g_p = va[1].ptr<float>(0) + i;
		float* a_r_p = va[2].ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08, sum09, sum10, sum11, sum12;
		float tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, tmp10, tmp11, tmp12;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = sum09 = sum10 = sum11 = sum12 = 0.f;

		float bb, bg, br, gg, gr, rr;
		float covb, covg, covr;
		float det, id;


		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += *s03_p1 * (r + 1);
		sum04 += *s04_p1 * (r + 1);
		sum05 += *s05_p1 * (r + 1);
		sum06 += *s06_p1 * (r + 1);
		sum07 += *s07_p1 * (r + 1);
		sum08 += *s08_p1 * (r + 1);
		sum09 += *s09_p1 * (r + 1);
		sum10 += *s10_p1 * (r + 1);
		sum11 += *s11_p1 * (r + 1);
		sum12 += *s12_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum00 += *s00_p2;
			s00_p2 += step;
			sum01 += *s01_p2;
			s01_p2 += step;
			sum02 += *s02_p2;
			s02_p2 += step;
			sum03 += *s03_p2;
			s03_p2 += step;
			sum04 += *s04_p2;
			s04_p2 += step;
			sum05 += *s05_p2;
			s05_p2 += step;
			sum06 += *s06_p2;
			s06_p2 += step;
			sum07 += *s07_p2;
			s07_p2 += step;
			sum08 += *s08_p2;
			s08_p2 += step;
			sum09 += *s09_p2;
			s09_p2 += step;
			sum10 += *s10_p2;
			s10_p2 += step;
			sum11 += *s11_p2;
			s11_p2 += step;
			sum12 += *s12_p2;
			s12_p2 += step;
		}
		tmp00 = sum00 * div;
		tmp01 = sum01 * div;
		tmp02 = sum02 * div;
		tmp03 = sum03 * div;
		tmp04 = sum04 * div;
		tmp05 = sum05 * div;
		tmp06 = sum06 * div;
		tmp07 = sum07 * div;
		tmp08 = sum08 * div;
		tmp09 = sum09 * div;
		tmp10 = sum10 * div;
		tmp11 = sum11 * div;
		tmp12 = sum12 * div;

		*meanIb_p = tmp00;
		*meanIg_p = tmp01;
		*meanIr_p = tmp02;
		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		bb = tmp04 - tmp00 * tmp00;
		bg = tmp05 - tmp00 * tmp01;
		br = tmp06 - tmp00 * tmp02;
		gg = tmp07 - tmp01 * tmp01;
		gr = tmp08 - tmp01 * tmp02;
		rr = tmp09 - tmp02 * tmp02;
		covb = tmp10 - tmp00 * tmp03;
		covg = tmp11 - tmp01 * tmp03;
		covr = tmp12 - tmp02 * tmp03;

		bb += eps;
		gg += eps;
		rr += eps;

		det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
		id = 1.f / det;
		*det_p = id;
		det_p += step;

		float c0 = gg*rr - gr*gr;
		float c1 = br*gr - bg*rr;
		float c2 = bg*gr - br*gg;
		float c4 = bb*rr - br*br;
		float c5 = br*bg - bb*gr;
		float c8 = bb*gg - bg*bg;

		*cov0_p = c0;
		cov0_p += step;
		*cov1_p = c1;
		cov1_p += step;
		*cov2_p = c2;
		cov2_p += step;
		*cov3_p = c4;
		cov3_p += step;
		*cov4_p = c5;
		cov4_p += step;
		*cov5_p = c8;
		cov5_p += step;

		*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
		*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
		*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
		*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
		a_b_p += step;
		a_g_p += step;
		a_r_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p2 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p2 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p2 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p2 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p2 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p2 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p2 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p2 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p2 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p2 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p2 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p2 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p2 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p1 += step;
			s00_p2 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p1 += step;
			s01_p2 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p1 += step;
			s02_p2 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p1 += step;
			s03_p2 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p1 += step;
			s04_p2 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p1 += step;
			s05_p2 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p1 += step;
			s06_p2 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p1 += step;
			s07_p2 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p1 += step;
			s08_p2 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p1 += step;
			s09_p2 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p1 += step;
			s10_p2 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p1 += step;
			s11_p2 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p1 += step;
			s12_p2 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p1 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p1 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p1 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p1 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p1 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p1 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p1 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p1 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p1 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p1 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p1 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p1 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p1 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_First_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i;
		float* s01_p1 = tempVec[1].ptr<float>(0) + i;
		float* s02_p1 = tempVec[2].ptr<float>(0) + i;
		float* s03_p1 = tempVec[3].ptr<float>(0) + i;
		float* s04_p1 = tempVec[4].ptr<float>(0) + i;
		float* s05_p1 = tempVec[5].ptr<float>(0) + i;
		float* s06_p1 = tempVec[6].ptr<float>(0) + i;
		float* s07_p1 = tempVec[7].ptr<float>(0) + i;
		float* s08_p1 = tempVec[8].ptr<float>(0) + i;
		float* s09_p1 = tempVec[9].ptr<float>(0) + i;
		float* s10_p1 = tempVec[10].ptr<float>(0) + i;
		float* s11_p1 = tempVec[11].ptr<float>(0) + i;
		float* s12_p1 = tempVec[12].ptr<float>(0) + i;

		float* s00_p2 = tempVec[0].ptr<float>(1) + i;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i;

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

		float* a_b_p = va[0].ptr<float>(0) + i;
		float* a_g_p = va[1].ptr<float>(0) + i;
		float* a_r_p = va[2].ptr<float>(0) + i;
		float* b_p = b.ptr<float>(0) + i;

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08, sum09, sum10, sum11, sum12;
		float tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08, tmp09, tmp10, tmp11, tmp12;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = sum09 = sum10 = sum11 = sum12 = 0.f;

		float bb, bg, br, gg, gr, rr;
		float covb, covg, covr;
		float det, id;


		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += *s03_p1 * (r + 1);
		sum04 += *s04_p1 * (r + 1);
		sum05 += *s05_p1 * (r + 1);
		sum06 += *s06_p1 * (r + 1);
		sum07 += *s07_p1 * (r + 1);
		sum08 += *s08_p1 * (r + 1);
		sum09 += *s09_p1 * (r + 1);
		sum10 += *s10_p1 * (r + 1);
		sum11 += *s11_p1 * (r + 1);
		sum12 += *s12_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum00 += *s00_p2;
			s00_p2 += step;
			sum01 += *s01_p2;
			s01_p2 += step;
			sum02 += *s02_p2;
			s02_p2 += step;
			sum03 += *s03_p2;
			s03_p2 += step;
			sum04 += *s04_p2;
			s04_p2 += step;
			sum05 += *s05_p2;
			s05_p2 += step;
			sum06 += *s06_p2;
			s06_p2 += step;
			sum07 += *s07_p2;
			s07_p2 += step;
			sum08 += *s08_p2;
			s08_p2 += step;
			sum09 += *s09_p2;
			s09_p2 += step;
			sum10 += *s10_p2;
			s10_p2 += step;
			sum11 += *s11_p2;
			s11_p2 += step;
			sum12 += *s12_p2;
			s12_p2 += step;
		}
		tmp00 = sum00 * div;
		tmp01 = sum01 * div;
		tmp02 = sum02 * div;
		tmp03 = sum03 * div;
		tmp04 = sum04 * div;
		tmp05 = sum05 * div;
		tmp06 = sum06 * div;
		tmp07 = sum07 * div;
		tmp08 = sum08 * div;
		tmp09 = sum09 * div;
		tmp10 = sum10 * div;
		tmp11 = sum11 * div;
		tmp12 = sum12 * div;

		*meanIb_p = tmp00;
		*meanIg_p = tmp01;
		*meanIr_p = tmp02;
		meanIb_p += step;
		meanIg_p += step;
		meanIr_p += step;

		bb = tmp04 - tmp00 * tmp00;
		bg = tmp05 - tmp00 * tmp01;
		br = tmp06 - tmp00 * tmp02;
		gg = tmp07 - tmp01 * tmp01;
		gr = tmp08 - tmp01 * tmp02;
		rr = tmp09 - tmp02 * tmp02;
		covb = tmp10 - tmp00 * tmp03;
		covg = tmp11 - tmp01 * tmp03;
		covr = tmp12 - tmp02 * tmp03;

		bb += eps;
		gg += eps;
		rr += eps;

		det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
		id = 1.f / det;
		*det_p = id;
		det_p += step;

		float c0 = gg*rr - gr*gr;
		float c1 = br*gr - bg*rr;
		float c2 = bg*gr - br*gg;
		float c4 = bb*rr - br*br;
		float c5 = br*bg - bb*gr;
		float c8 = bb*gg - bg*bg;

		*cov0_p = c0;
		cov0_p += step;
		*cov1_p = c1;
		cov1_p += step;
		*cov2_p = c2;
		cov2_p += step;
		*cov3_p = c4;
		cov3_p += step;
		*cov4_p = c5;
		cov4_p += step;
		*cov5_p = c8;
		cov5_p += step;

		*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
		*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
		*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
		*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
		a_b_p += step;
		a_g_p += step;
		a_r_p += step;
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p2 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p2 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p2 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p2 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p2 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p2 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p2 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p2 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p2 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p2 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p2 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p2 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p2 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p1 += step;
			s00_p2 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p1 += step;
			s01_p2 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p1 += step;
			s02_p2 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p1 += step;
			s03_p2 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p1 += step;
			s04_p2 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p1 += step;
			s05_p2 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p1 += step;
			s06_p2 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p1 += step;
			s07_p2 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p1 += step;
			s08_p2 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p1 += step;
			s09_p2 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p1 += step;
			s10_p2 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p1 += step;
			s11_p2 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p1 += step;
			s12_p2 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum00 += *s00_p2 - *s00_p1;
			s00_p1 += step;
			sum01 += *s01_p2 - *s01_p1;
			s01_p1 += step;
			sum02 += *s02_p2 - *s02_p1;
			s02_p1 += step;
			sum03 += *s03_p2 - *s03_p1;
			s03_p1 += step;
			sum04 += *s04_p2 - *s04_p1;
			s04_p1 += step;
			sum05 += *s05_p2 - *s05_p1;
			s05_p1 += step;
			sum06 += *s06_p2 - *s06_p1;
			s06_p1 += step;
			sum07 += *s07_p2 - *s07_p1;
			s07_p1 += step;
			sum08 += *s08_p2 - *s08_p1;
			s08_p1 += step;
			sum09 += *s09_p2 - *s09_p1;
			s09_p1 += step;
			sum10 += *s10_p2 - *s10_p1;
			s10_p1 += step;
			sum11 += *s11_p2 - *s11_p1;
			s11_p1 += step;
			sum12 += *s12_p2 - *s12_p1;
			s12_p1 += step;

			tmp00 = sum00 * div;
			tmp01 = sum01 * div;
			tmp02 = sum02 * div;
			tmp03 = sum03 * div;
			tmp04 = sum04 * div;
			tmp05 = sum05 * div;
			tmp06 = sum06 * div;
			tmp07 = sum07 * div;
			tmp08 = sum08 * div;
			tmp09 = sum09 * div;
			tmp10 = sum10 * div;
			tmp11 = sum11 * div;
			tmp12 = sum12 * div;

			*meanIb_p = tmp00;
			*meanIg_p = tmp01;
			*meanIr_p = tmp02;
			meanIb_p += step;
			meanIg_p += step;
			meanIr_p += step;

			bb = tmp04 - tmp00 * tmp00;
			bg = tmp05 - tmp00 * tmp01;
			br = tmp06 - tmp00 * tmp02;
			gg = tmp07 - tmp01 * tmp01;
			gr = tmp08 - tmp01 * tmp02;
			rr = tmp09 - tmp02 * tmp02;
			covb = tmp10 - tmp00 * tmp03;
			covg = tmp11 - tmp01 * tmp03;
			covr = tmp12 - tmp02 * tmp03;

			bb += eps;
			gg += eps;
			rr += eps;

			det = (bb*gg*rr) + (gr*bg*br) + (br*gr*bg) - (bb*gr*gr) - (br*gg*br) - (bg*bg*rr);
			id = 1.f / det;
			*det_p = id;
			det_p += step;

			float c0 = gg*rr - gr*gr;
			float c1 = br*gr - bg*rr;
			float c2 = bg*gr - br*gg;
			float c4 = bb*rr - br*br;
			float c5 = br*bg - bb*gr;
			float c8 = bb*gg - bg*bg;

			*cov0_p = c0;
			cov0_p += step;
			*cov1_p = c1;
			cov1_p += step;
			*cov2_p = c2;
			cov2_p += step;
			*cov3_p = c4;
			cov3_p += step;
			*cov4_p = c5;
			cov4_p += step;
			*cov5_p = c8;
			cov5_p += step;

			*a_b_p = id * (covb*c0 + covg*c1 + covr*c2);
			*a_g_p = id * (covb*c1 + covg*c4 + covr*c5);
			*a_r_p = id * (covb*c2 + covg*c5 + covr*c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_First_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// mean_p
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 4;	// corr_I_bb
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 4;	// corr_I_bg
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 4;	// corr_I_br
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 4;	// corr_I_gg
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 4;	// corr_I_gr
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 4;	// corr_I_rr
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 4;	// cov_Ip_b
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 4;	// cov_Ip_g
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 4;	// cov_Ip_r

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 4;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 4;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 4;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 4;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 4;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i * 4;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i * 4;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i * 4;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i * 4;

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

		float* a_b_p = va[0].ptr<float>(0) + i * 4;
		float* a_g_p = va[1].ptr<float>(0) + i * 4;
		float* a_r_p = va[2].ptr<float>(0) + i * 4;
		float* b_p = b.ptr<float>(0) + i * 4;

		__m128 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08, mSum09, mSum10, mSum11, mSum12;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08, mTmp09, mTmp10, mTmp11, mTmp12;
		__m128 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m128 mCov00, mCov01, mCov02;
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
		mSum09 = _mm_setzero_ps();
		mSum10 = _mm_setzero_ps();
		mSum11 = _mm_setzero_ps();
		mSum12 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		mSum04 = _mm_mul_ps(mBorder, _mm_load_ps(s04_p1));
		mSum05 = _mm_mul_ps(mBorder, _mm_load_ps(s05_p1));
		mSum06 = _mm_mul_ps(mBorder, _mm_load_ps(s06_p1));
		mSum07 = _mm_mul_ps(mBorder, _mm_load_ps(s07_p1));
		mSum08 = _mm_mul_ps(mBorder, _mm_load_ps(s08_p1));
		mSum09 = _mm_mul_ps(mBorder, _mm_load_ps(s09_p1));
		mSum10 = _mm_mul_ps(mBorder, _mm_load_ps(s10_p1));
		mSum11 = _mm_mul_ps(mBorder, _mm_load_ps(s11_p1));
		mSum12 = _mm_mul_ps(mBorder, _mm_load_ps(s12_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;
		}
		mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
		mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
		mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
		mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
		mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
		mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
		mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
		mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
		mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

		_mm_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
		mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
		mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
		mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
		mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
		mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
		mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

		mVar00 = _mm_add_ps(mVar00, mEps);
		mVar03 = _mm_add_ps(mVar03, mEps);
		mVar05 = _mm_add_ps(mVar05, mEps);

		mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
		mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
		mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
		mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
		mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

		mDet = _mm_add_ps(mTmp04, mTmp05);
		mDet = _mm_add_ps(mDet, mTmp05);
		mDet = _mm_sub_ps(mDet, mTmp06);
		mDet = _mm_sub_ps(mDet, mTmp07);
		mDet = _mm_sub_ps(mDet, mTmp08);
		mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
		_mm_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

		_mm_store_ps(cov0_p, mTmp04);
		cov0_p += step;
		_mm_store_ps(cov1_p, mTmp05);
		cov1_p += step;
		_mm_store_ps(cov2_p, mTmp06);
		cov2_p += step;
		_mm_store_ps(cov3_p, mTmp07);
		cov3_p += step;
		_mm_store_ps(cov4_p, mTmp08);
		cov4_p += step;
		_mm_store_ps(cov5_p, mTmp09);
		cov5_p += step;

		//
		/*
		mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
		*/
		mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
		mTmp10 = _mm_mul_ps(mTmp10, mDet);
		//_mm_store_ps(a_b_p, mTmp10);
		_mm_stream_ps(a_b_p, mTmp10);
		a_b_p += step;

		/*
		mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp08));
		mTmp11 = _mm_add_ps(mTmp11, _mm_mul_ps(mCov02, mTmp08));
		*/
		mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
		mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
		mTmp11 = _mm_mul_ps(mTmp11, mDet);
		//_mm_store_ps(a_g_p, mTmp11);
		_mm_stream_ps(a_g_p, mTmp11);
		a_g_p += step;

		/*
		mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp07));
		mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
		*/
		mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
		mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
		mTmp12 = _mm_mul_ps(mTmp12, mDet);
		//_mm_store_ps(a_r_p, mTmp12);
		_mm_stream_ps(a_r_p, mTmp12);
		a_r_p += step;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;


			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_First_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// mean_p
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 4;	// corr_I_bb
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 4;	// corr_I_bg
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 4;	// corr_I_br
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 4;	// corr_I_gg
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 4;	// corr_I_gr
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 4;	// corr_I_rr
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 4;	// cov_Ip_b
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 4;	// cov_Ip_g
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 4;	// cov_Ip_r

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 4;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 4;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 4;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 4;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 4;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i * 4;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i * 4;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i * 4;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i * 4;

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

		float* a_b_p = va[0].ptr<float>(0) + i * 4;
		float* a_g_p = va[1].ptr<float>(0) + i * 4;
		float* a_r_p = va[2].ptr<float>(0) + i * 4;
		float* b_p = b.ptr<float>(0) + i * 4;

		__m128 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08, mSum09, mSum10, mSum11, mSum12;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08, mTmp09, mTmp10, mTmp11, mTmp12;
		__m128 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m128 mCov00, mCov01, mCov02;
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
		mSum09 = _mm_setzero_ps();
		mSum10 = _mm_setzero_ps();
		mSum11 = _mm_setzero_ps();
		mSum12 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		mSum04 = _mm_mul_ps(mBorder, _mm_load_ps(s04_p1));
		mSum05 = _mm_mul_ps(mBorder, _mm_load_ps(s05_p1));
		mSum06 = _mm_mul_ps(mBorder, _mm_load_ps(s06_p1));
		mSum07 = _mm_mul_ps(mBorder, _mm_load_ps(s07_p1));
		mSum08 = _mm_mul_ps(mBorder, _mm_load_ps(s08_p1));
		mSum09 = _mm_mul_ps(mBorder, _mm_load_ps(s09_p1));
		mSum10 = _mm_mul_ps(mBorder, _mm_load_ps(s10_p1));
		mSum11 = _mm_mul_ps(mBorder, _mm_load_ps(s11_p1));
		mSum12 = _mm_mul_ps(mBorder, _mm_load_ps(s12_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;
		}
		mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
		mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
		mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
		mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
		mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
		mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
		mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
		mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
		mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

		_mm_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
		mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
		mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
		mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
		mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
		mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
		mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

		mVar00 = _mm_add_ps(mVar00, mEps);
		mVar03 = _mm_add_ps(mVar03, mEps);
		mVar05 = _mm_add_ps(mVar05, mEps);

		mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
		mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
		mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
		mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
		mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

		mDet = _mm_add_ps(mTmp04, mTmp05);
		mDet = _mm_add_ps(mDet, mTmp05);
		mDet = _mm_sub_ps(mDet, mTmp06);
		mDet = _mm_sub_ps(mDet, mTmp07);
		mDet = _mm_sub_ps(mDet, mTmp08);
		mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
		_mm_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

		_mm_store_ps(cov0_p, mTmp04);
		cov0_p += step;
		_mm_store_ps(cov1_p, mTmp05);
		cov1_p += step;
		_mm_store_ps(cov2_p, mTmp06);
		cov2_p += step;
		_mm_store_ps(cov3_p, mTmp07);
		cov3_p += step;
		_mm_store_ps(cov4_p, mTmp08);
		cov4_p += step;
		_mm_store_ps(cov5_p, mTmp09);
		cov5_p += step;

		//
		/*
		mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
		*/
		mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
		mTmp10 = _mm_mul_ps(mTmp10, mDet);
		//_mm_store_ps(a_b_p, mTmp10);
		_mm_stream_ps(a_b_p, mTmp10);
		a_b_p += step;

		/*
		mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp08));
		mTmp11 = _mm_add_ps(mTmp11, _mm_mul_ps(mCov02, mTmp08));
		*/
		mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
		mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
		mTmp11 = _mm_mul_ps(mTmp11, mDet);
		//_mm_store_ps(a_g_p, mTmp11);
		_mm_stream_ps(a_g_p, mTmp11);
		a_g_p += step;

		/*
		mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp07));
		mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
		*/
		mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
		mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
		mTmp12 = _mm_mul_ps(mTmp12, mDet);
		//_mm_store_ps(a_r_p, mTmp12);
		_mm_stream_ps(a_r_p, mTmp12);
		a_r_p += step;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;


			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_First_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 4;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 4;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 4;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 4;	// mean_p
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 4;	// corr_I_bb
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 4;	// corr_I_bg
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 4;	// corr_I_br
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 4;	// corr_I_gg
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 4;	// corr_I_gr
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 4;	// corr_I_rr
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 4;	// cov_Ip_b
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 4;	// cov_Ip_g
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 4;	// cov_Ip_r

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 4;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 4;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 4;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 4;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 4;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 4;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i * 4;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i * 4;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i * 4;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i * 4;

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

		float* a_b_p = va[0].ptr<float>(0) + i * 4;
		float* a_g_p = va[1].ptr<float>(0) + i * 4;
		float* a_r_p = va[2].ptr<float>(0) + i * 4;
		float* b_p = b.ptr<float>(0) + i * 4;

		__m128 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08, mSum09, mSum10, mSum11, mSum12;
		__m128 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08, mTmp09, mTmp10, mTmp11, mTmp12;
		__m128 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m128 mCov00, mCov01, mCov02;
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
		mSum09 = _mm_setzero_ps();
		mSum10 = _mm_setzero_ps();
		mSum11 = _mm_setzero_ps();
		mSum12 = _mm_setzero_ps();

		mSum00 = _mm_mul_ps(mBorder, _mm_load_ps(s00_p1));
		mSum01 = _mm_mul_ps(mBorder, _mm_load_ps(s01_p1));
		mSum02 = _mm_mul_ps(mBorder, _mm_load_ps(s02_p1));
		mSum03 = _mm_mul_ps(mBorder, _mm_load_ps(s03_p1));
		mSum04 = _mm_mul_ps(mBorder, _mm_load_ps(s04_p1));
		mSum05 = _mm_mul_ps(mBorder, _mm_load_ps(s05_p1));
		mSum06 = _mm_mul_ps(mBorder, _mm_load_ps(s06_p1));
		mSum07 = _mm_mul_ps(mBorder, _mm_load_ps(s07_p1));
		mSum08 = _mm_mul_ps(mBorder, _mm_load_ps(s08_p1));
		mSum09 = _mm_mul_ps(mBorder, _mm_load_ps(s09_p1));
		mSum10 = _mm_mul_ps(mBorder, _mm_load_ps(s10_p1));
		mSum11 = _mm_mul_ps(mBorder, _mm_load_ps(s11_p1));
		mSum12 = _mm_mul_ps(mBorder, _mm_load_ps(s12_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;
		}
		mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
		mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
		mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
		mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
		mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
		mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
		mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
		mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
		mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
		mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

		_mm_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
		mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
		mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
		mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
		mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
		mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
		mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

		mVar00 = _mm_add_ps(mVar00, mEps);
		mVar03 = _mm_add_ps(mVar03, mEps);
		mVar05 = _mm_add_ps(mVar05, mEps);

		mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
		mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
		mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
		mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
		mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

		mDet = _mm_add_ps(mTmp04, mTmp05);
		mDet = _mm_add_ps(mDet, mTmp05);
		mDet = _mm_sub_ps(mDet, mTmp06);
		mDet = _mm_sub_ps(mDet, mTmp07);
		mDet = _mm_sub_ps(mDet, mTmp08);
		mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
		_mm_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

		_mm_store_ps(cov0_p, mTmp04);
		cov0_p += step;
		_mm_store_ps(cov1_p, mTmp05);
		cov1_p += step;
		_mm_store_ps(cov2_p, mTmp06);
		cov2_p += step;
		_mm_store_ps(cov3_p, mTmp07);
		cov3_p += step;
		_mm_store_ps(cov4_p, mTmp08);
		cov4_p += step;
		_mm_store_ps(cov5_p, mTmp09);
		cov5_p += step;

		//
		/*
		mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
		*/
		mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
		mTmp10 = _mm_mul_ps(mTmp10, mDet);
		//_mm_store_ps(a_b_p, mTmp10);
		_mm_stream_ps(a_b_p, mTmp10);
		a_b_p += step;

		/*
		mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp08));
		mTmp11 = _mm_add_ps(mTmp11, _mm_mul_ps(mCov02, mTmp08));
		*/
		mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
		mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
		mTmp11 = _mm_mul_ps(mTmp11, mDet);
		//_mm_store_ps(a_g_p, mTmp11);
		_mm_stream_ps(a_g_p, mTmp11);
		a_g_p += step;

		/*
		mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp07));
		mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
		*/
		mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
		mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
		mTmp12 = _mm_mul_ps(mTmp12, mDet);
		//_mm_store_ps(a_r_p, mTmp12);
		_mm_stream_ps(a_r_p, mTmp12);
		a_r_p += step;

		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
		mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
		//_mm_store_ps(b_p, mTmp03);
		_mm_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;


			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm_add_ps(mSum00, _mm_loadu_ps(s00_p2));
			mSum00 = _mm_sub_ps(mSum00, _mm_load_ps(s00_p1));
			mSum01 = _mm_add_ps(mSum01, _mm_loadu_ps(s01_p2));
			mSum01 = _mm_sub_ps(mSum01, _mm_load_ps(s01_p1));
			mSum02 = _mm_add_ps(mSum02, _mm_loadu_ps(s02_p2));
			mSum02 = _mm_sub_ps(mSum02, _mm_load_ps(s02_p1));
			mSum03 = _mm_add_ps(mSum03, _mm_loadu_ps(s03_p2));
			mSum03 = _mm_sub_ps(mSum03, _mm_load_ps(s03_p1));
			mSum04 = _mm_add_ps(mSum04, _mm_loadu_ps(s04_p2));
			mSum04 = _mm_sub_ps(mSum04, _mm_load_ps(s04_p1));
			mSum05 = _mm_add_ps(mSum05, _mm_loadu_ps(s05_p2));
			mSum05 = _mm_sub_ps(mSum05, _mm_load_ps(s05_p1));
			mSum06 = _mm_add_ps(mSum06, _mm_loadu_ps(s06_p2));
			mSum06 = _mm_sub_ps(mSum06, _mm_load_ps(s06_p1));
			mSum07 = _mm_add_ps(mSum07, _mm_loadu_ps(s07_p2));
			mSum07 = _mm_sub_ps(mSum07, _mm_load_ps(s07_p1));
			mSum08 = _mm_add_ps(mSum08, _mm_loadu_ps(s08_p2));
			mSum08 = _mm_sub_ps(mSum08, _mm_load_ps(s08_p1));
			mSum09 = _mm_add_ps(mSum09, _mm_loadu_ps(s09_p2));
			mSum09 = _mm_sub_ps(mSum09, _mm_load_ps(s09_p1));
			mSum10 = _mm_add_ps(mSum10, _mm_loadu_ps(s10_p2));
			mSum10 = _mm_sub_ps(mSum10, _mm_load_ps(s10_p1));
			mSum11 = _mm_add_ps(mSum11, _mm_loadu_ps(s11_p2));
			mSum11 = _mm_sub_ps(mSum11, _mm_load_ps(s11_p1));
			mSum12 = _mm_add_ps(mSum12, _mm_loadu_ps(s12_p2));
			mSum12 = _mm_sub_ps(mSum12, _mm_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;

			mTmp00 = _mm_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm_sub_ps(mTmp04, _mm_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm_sub_ps(mTmp05, _mm_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm_sub_ps(mTmp06, _mm_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm_sub_ps(mTmp07, _mm_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm_sub_ps(mTmp08, _mm_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm_sub_ps(mTmp09, _mm_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm_sub_ps(mTmp10, _mm_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm_sub_ps(mTmp11, _mm_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm_sub_ps(mTmp12, _mm_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm_add_ps(mVar00, mEps);
			mVar03 = _mm_add_ps(mVar03, mEps);
			mVar05 = _mm_add_ps(mVar05, mEps);

			mTmp04 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm_mul_ps(mVar01, _mm_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm_mul_ps(mVar00, _mm_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm_mul_ps(mVar03, _mm_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm_mul_ps(mVar05, _mm_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm_add_ps(mTmp04, mTmp05);
			mDet = _mm_add_ps(mDet, mTmp05);
			mDet = _mm_sub_ps(mDet, mTmp06);
			mDet = _mm_sub_ps(mDet, mTmp07);
			mDet = _mm_sub_ps(mDet, mTmp08);
			mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);
			_mm_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm_sub_ps(_mm_mul_ps(mVar03, mVar05), _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar04), _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_sub_ps(_mm_mul_ps(mVar01, mVar04), _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar05), _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_sub_ps(_mm_mul_ps(mVar02, mVar01), _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_sub_ps(_mm_mul_ps(mVar00, mVar03), _mm_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm_fmsub_ps(mVar03, mVar05, _mm_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm_fmsub_ps(mVar02, mVar04, _mm_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm_fmsub_ps(mVar01, mVar04, _mm_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm_fmsub_ps(mVar00, mVar05, _mm_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm_fmsub_ps(mVar02, mVar01, _mm_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm_fmsub_ps(mVar00, mVar03, _mm_mul_ps(mVar01, mVar01)); //c8

			_mm_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp04), _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_add_ps(mTmp10, _mm_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm_fmadd_ps(mCov00, mTmp04, _mm_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm_mul_ps(mTmp10, mDet);
			_mm_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp05), _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm_fmadd_ps(mCov00, mTmp05, _mm_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm_mul_ps(mTmp11, mDet);
			_mm_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm_add_ps(_mm_mul_ps(mCov00, mTmp06), _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_add_ps(mTmp12, _mm_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm_fmadd_ps(mCov00, mTmp06, _mm_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm_mul_ps(mTmp12, mDet);
			_mm_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm_sub_ps(mTmp03, _mm_mul_ps(mTmp12, mTmp02));
			_mm_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_First_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// mean_p
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;	// corr_I_bb
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;	// corr_I_bg
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;	// corr_I_br
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;	// corr_I_gg
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;	// corr_I_gr
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 8;	// corr_I_rr
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 8;	// cov_Ip_b
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 8;	// cov_Ip_g
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 8;	// cov_Ip_r

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 8;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 8;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 8;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 8;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 8;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i * 8;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i * 8;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i * 8;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i * 8;

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

		float* a_b_p = va[0].ptr<float>(0) + i * 8;
		float* a_g_p = va[1].ptr<float>(0) + i * 8;
		float* a_r_p = va[2].ptr<float>(0) + i * 8;
		float* b_p = b.ptr<float>(0) + i * 8;

		__m256 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08, mSum09, mSum10, mSum11, mSum12;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08, mTmp09, mTmp10, mTmp11, mTmp12;
		__m256 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m256 mCov00, mCov01, mCov02;
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
		mSum09 = _mm256_setzero_ps();
		mSum10 = _mm256_setzero_ps();
		mSum11 = _mm256_setzero_ps();
		mSum12 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		mSum04 = _mm256_mul_ps(mBorder, _mm256_load_ps(s04_p1));
		mSum05 = _mm256_mul_ps(mBorder, _mm256_load_ps(s05_p1));
		mSum06 = _mm256_mul_ps(mBorder, _mm256_load_ps(s06_p1));
		mSum07 = _mm256_mul_ps(mBorder, _mm256_load_ps(s07_p1));
		mSum08 = _mm256_mul_ps(mBorder, _mm256_load_ps(s08_p1));
		mSum09 = _mm256_mul_ps(mBorder, _mm256_load_ps(s09_p1));
		mSum10 = _mm256_mul_ps(mBorder, _mm256_load_ps(s10_p1));
		mSum11 = _mm256_mul_ps(mBorder, _mm256_load_ps(s11_p1));
		mSum12 = _mm256_mul_ps(mBorder, _mm256_load_ps(s12_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
		mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
		mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
		mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
		mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

		_mm256_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm256_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm256_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
		mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
		mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
		mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
		mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
		mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
		mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

		mVar00 = _mm256_add_ps(mVar00, mEps);
		mVar03 = _mm256_add_ps(mVar03, mEps);
		mVar05 = _mm256_add_ps(mVar05, mEps);

		mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
		mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
		mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
		mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
		mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

		mDet = _mm256_add_ps(mTmp04, mTmp05);
		mDet = _mm256_add_ps(mDet, mTmp05);
		mDet = _mm256_sub_ps(mDet, mTmp06);
		mDet = _mm256_sub_ps(mDet, mTmp07);
		mDet = _mm256_sub_ps(mDet, mTmp08);
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
		_mm256_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

		_mm256_store_ps(cov0_p, mTmp04);
		cov0_p += step;
		_mm256_store_ps(cov1_p, mTmp05);
		cov1_p += step;
		_mm256_store_ps(cov2_p, mTmp06);
		cov2_p += step;
		_mm256_store_ps(cov3_p, mTmp07);
		cov3_p += step;
		_mm256_store_ps(cov4_p, mTmp08);
		cov4_p += step;
		_mm256_store_ps(cov5_p, mTmp09);
		cov5_p += step;

		//
		/*
		mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
		*/
		mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
		mTmp10 = _mm256_mul_ps(mTmp10, mDet);
		//_mm256_store_ps(a_b_p, mTmp10);
		_mm256_stream_ps(a_b_p, mTmp10);
		a_b_p += step;

		/*
		mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp08));
		mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
		*/
		mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
		mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
		mTmp11 = _mm256_mul_ps(mTmp11, mDet);
		//_mm256_store_ps(a_g_p, mTmp11);
		_mm256_stream_ps(a_g_p, mTmp11);
		a_g_p += step;

		/*
		mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp07));
		mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
		*/
		mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
		mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
		mTmp12 = _mm256_mul_ps(mTmp12, mDet);
		//_mm256_store_ps(a_r_p, mTmp12);
		_mm256_stream_ps(a_r_p, mTmp12);
		a_r_p += step;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;


			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_First_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// mean_p
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;	// corr_I_bb
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;	// corr_I_bg
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;	// corr_I_br
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;	// corr_I_gg
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;	// corr_I_gr
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 8;	// corr_I_rr
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 8;	// cov_Ip_b
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 8;	// cov_Ip_g
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 8;	// cov_Ip_r

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 8;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 8;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 8;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 8;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 8;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i * 8;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i * 8;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i * 8;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i * 8;

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

		float* a_b_p = va[0].ptr<float>(0) + i * 8;
		float* a_g_p = va[1].ptr<float>(0) + i * 8;
		float* a_r_p = va[2].ptr<float>(0) + i * 8;
		float* b_p = b.ptr<float>(0) + i * 8;

		__m256 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08, mSum09, mSum10, mSum11, mSum12;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08, mTmp09, mTmp10, mTmp11, mTmp12;
		__m256 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m256 mCov00, mCov01, mCov02;
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
		mSum09 = _mm256_setzero_ps();
		mSum10 = _mm256_setzero_ps();
		mSum11 = _mm256_setzero_ps();
		mSum12 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		mSum04 = _mm256_mul_ps(mBorder, _mm256_load_ps(s04_p1));
		mSum05 = _mm256_mul_ps(mBorder, _mm256_load_ps(s05_p1));
		mSum06 = _mm256_mul_ps(mBorder, _mm256_load_ps(s06_p1));
		mSum07 = _mm256_mul_ps(mBorder, _mm256_load_ps(s07_p1));
		mSum08 = _mm256_mul_ps(mBorder, _mm256_load_ps(s08_p1));
		mSum09 = _mm256_mul_ps(mBorder, _mm256_load_ps(s09_p1));
		mSum10 = _mm256_mul_ps(mBorder, _mm256_load_ps(s10_p1));
		mSum11 = _mm256_mul_ps(mBorder, _mm256_load_ps(s11_p1));
		mSum12 = _mm256_mul_ps(mBorder, _mm256_load_ps(s12_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
		mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
		mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
		mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
		mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

		_mm256_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm256_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm256_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
		mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
		mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
		mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
		mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
		mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
		mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

		mVar00 = _mm256_add_ps(mVar00, mEps);
		mVar03 = _mm256_add_ps(mVar03, mEps);
		mVar05 = _mm256_add_ps(mVar05, mEps);

		mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
		mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
		mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
		mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
		mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

		mDet = _mm256_add_ps(mTmp04, mTmp05);
		mDet = _mm256_add_ps(mDet, mTmp05);
		mDet = _mm256_sub_ps(mDet, mTmp06);
		mDet = _mm256_sub_ps(mDet, mTmp07);
		mDet = _mm256_sub_ps(mDet, mTmp08);
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
		_mm256_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

		_mm256_store_ps(cov0_p, mTmp04);
		cov0_p += step;
		_mm256_store_ps(cov1_p, mTmp05);
		cov1_p += step;
		_mm256_store_ps(cov2_p, mTmp06);
		cov2_p += step;
		_mm256_store_ps(cov3_p, mTmp07);
		cov3_p += step;
		_mm256_store_ps(cov4_p, mTmp08);
		cov4_p += step;
		_mm256_store_ps(cov5_p, mTmp09);
		cov5_p += step;

		//
		/*
		mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
		*/
		mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
		mTmp10 = _mm256_mul_ps(mTmp10, mDet);
		//_mm256_store_ps(a_b_p, mTmp10);
		_mm256_stream_ps(a_b_p, mTmp10);
		a_b_p += step;

		/*
		mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp08));
		mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
		*/
		mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
		mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
		mTmp11 = _mm256_mul_ps(mTmp11, mDet);
		//_mm256_store_ps(a_g_p, mTmp11);
		_mm256_stream_ps(a_g_p, mTmp11);
		a_g_p += step;

		/*
		mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp07));
		mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
		*/
		mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
		mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
		mTmp12 = _mm256_mul_ps(mTmp12, mDet);
		//_mm256_store_ps(a_r_p, mTmp12);
		_mm256_stream_ps(a_r_p, mTmp12);
		a_r_p += step;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;


			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_First_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;	// mean_I_b
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;	// mean_I_g
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;	// mean_I_r
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;	// mean_p
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;	// corr_I_bb
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;	// corr_I_bg
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;	// corr_I_br
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;	// corr_I_gg
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;	// corr_I_gr
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 8;	// corr_I_rr
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 8;	// cov_Ip_b
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 8;	// cov_Ip_g
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 8;	// cov_Ip_r

		float* s00_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s01_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s02_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s03_p2 = tempVec[3].ptr<float>(1) + i * 8;
		float* s04_p2 = tempVec[4].ptr<float>(1) + i * 8;
		float* s05_p2 = tempVec[5].ptr<float>(1) + i * 8;
		float* s06_p2 = tempVec[6].ptr<float>(1) + i * 8;
		float* s07_p2 = tempVec[7].ptr<float>(1) + i * 8;
		float* s08_p2 = tempVec[8].ptr<float>(1) + i * 8;
		float* s09_p2 = tempVec[9].ptr<float>(1) + i * 8;
		float* s10_p2 = tempVec[10].ptr<float>(1) + i * 8;
		float* s11_p2 = tempVec[11].ptr<float>(1) + i * 8;
		float* s12_p2 = tempVec[12].ptr<float>(1) + i * 8;

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

		float* a_b_p = va[0].ptr<float>(0) + i * 8;
		float* a_g_p = va[1].ptr<float>(0) + i * 8;
		float* a_r_p = va[2].ptr<float>(0) + i * 8;
		float* b_p = b.ptr<float>(0) + i * 8;

		__m256 mSum00, mSum01, mSum02, mSum03, mSum04, mSum05, mSum06, mSum07, mSum08, mSum09, mSum10, mSum11, mSum12;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08, mTmp09, mTmp10, mTmp11, mTmp12;
		__m256 mVar00, mVar01, mVar02, mVar03, mVar04, mVar05;
		__m256 mCov00, mCov01, mCov02;
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
		mSum09 = _mm256_setzero_ps();
		mSum10 = _mm256_setzero_ps();
		mSum11 = _mm256_setzero_ps();
		mSum12 = _mm256_setzero_ps();

		mSum00 = _mm256_mul_ps(mBorder, _mm256_load_ps(s00_p1));
		mSum01 = _mm256_mul_ps(mBorder, _mm256_load_ps(s01_p1));
		mSum02 = _mm256_mul_ps(mBorder, _mm256_load_ps(s02_p1));
		mSum03 = _mm256_mul_ps(mBorder, _mm256_load_ps(s03_p1));
		mSum04 = _mm256_mul_ps(mBorder, _mm256_load_ps(s04_p1));
		mSum05 = _mm256_mul_ps(mBorder, _mm256_load_ps(s05_p1));
		mSum06 = _mm256_mul_ps(mBorder, _mm256_load_ps(s06_p1));
		mSum07 = _mm256_mul_ps(mBorder, _mm256_load_ps(s07_p1));
		mSum08 = _mm256_mul_ps(mBorder, _mm256_load_ps(s08_p1));
		mSum09 = _mm256_mul_ps(mBorder, _mm256_load_ps(s09_p1));
		mSum10 = _mm256_mul_ps(mBorder, _mm256_load_ps(s10_p1));
		mSum11 = _mm256_mul_ps(mBorder, _mm256_load_ps(s11_p1));
		mSum12 = _mm256_mul_ps(mBorder, _mm256_load_ps(s12_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
		mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
		mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
		mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
		mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

		_mm256_store_ps(meanIb_p, mTmp00);
		meanIb_p += step;
		_mm256_store_ps(meanIg_p, mTmp01);
		meanIg_p += step;
		_mm256_store_ps(meanIr_p, mTmp02);
		meanIr_p += step;

		mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
		mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
		mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
		mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
		mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
		mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
		mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

		mVar00 = _mm256_add_ps(mVar00, mEps);
		mVar03 = _mm256_add_ps(mVar03, mEps);
		mVar05 = _mm256_add_ps(mVar05, mEps);

		mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
		mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
		mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
		mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
		mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

		mDet = _mm256_add_ps(mTmp04, mTmp05);
		mDet = _mm256_add_ps(mDet, mTmp05);
		mDet = _mm256_sub_ps(mDet, mTmp06);
		mDet = _mm256_sub_ps(mDet, mTmp07);
		mDet = _mm256_sub_ps(mDet, mTmp08);
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
		_mm256_store_ps(det_p, mDet);
		det_p += step;

		/*
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

		_mm256_store_ps(cov0_p, mTmp04);
		cov0_p += step;
		_mm256_store_ps(cov1_p, mTmp05);
		cov1_p += step;
		_mm256_store_ps(cov2_p, mTmp06);
		cov2_p += step;
		_mm256_store_ps(cov3_p, mTmp07);
		cov3_p += step;
		_mm256_store_ps(cov4_p, mTmp08);
		cov4_p += step;
		_mm256_store_ps(cov5_p, mTmp09);
		cov5_p += step;

		//
		/*
		mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
		*/
		mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
		mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
		mTmp10 = _mm256_mul_ps(mTmp10, mDet);
		//_mm256_store_ps(a_b_p, mTmp10);
		_mm256_stream_ps(a_b_p, mTmp10);
		a_b_p += step;

		/*
		mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp08));
		mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
		*/
		mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
		mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
		mTmp11 = _mm256_mul_ps(mTmp11, mDet);
		//_mm256_store_ps(a_g_p, mTmp11);
		_mm256_stream_ps(a_g_p, mTmp11);
		a_g_p += step;

		/*
		mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp07));
		mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
		*/
		mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
		mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
		mTmp12 = _mm256_mul_ps(mTmp12, mDet);
		//_mm256_store_ps(a_r_p, mTmp12);
		_mm256_stream_ps(a_r_p, mTmp12);
		a_r_p += step;

		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
		mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
		//_mm256_store_ps(b_p, mTmp03);
		_mm256_stream_ps(b_p, mTmp03);
		b_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;


			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;
			s00_p2 += step;
			s01_p2 += step;
			s02_p2 += step;
			s03_p2 += step;
			s04_p2 += step;
			s05_p2 += step;
			s06_p2 += step;
			s07_p2 += step;
			s08_p2 += step;
			s09_p2 += step;
			s10_p2 += step;
			s11_p2 += step;
			s12_p2 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}

		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			mSum00 = _mm256_sub_ps(mSum00, _mm256_load_ps(s00_p1));
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			mSum01 = _mm256_sub_ps(mSum01, _mm256_load_ps(s01_p1));
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			mSum02 = _mm256_sub_ps(mSum02, _mm256_load_ps(s02_p1));
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			mSum03 = _mm256_sub_ps(mSum03, _mm256_load_ps(s03_p1));
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			mSum04 = _mm256_sub_ps(mSum04, _mm256_load_ps(s04_p1));
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			mSum05 = _mm256_sub_ps(mSum05, _mm256_load_ps(s05_p1));
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			mSum06 = _mm256_sub_ps(mSum06, _mm256_load_ps(s06_p1));
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			mSum07 = _mm256_sub_ps(mSum07, _mm256_load_ps(s07_p1));
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			mSum08 = _mm256_sub_ps(mSum08, _mm256_load_ps(s08_p1));
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			s00_p1 += step;
			s01_p1 += step;
			s02_p1 += step;
			s03_p1 += step;
			s04_p1 += step;
			s05_p1 += step;
			s06_p1 += step;
			s07_p1 += step;
			s08_p1 += step;
			s09_p1 += step;
			s10_p1 += step;
			s11_p1 += step;
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);	// mean_I_b
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);	// mean_I_g
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);	// mean_I_r
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);	// mean_p
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);	// corr_I_bb
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);	// corr_I_bg
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);	// corr_I_br
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);	// corr_I_gg
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);	// corr_I_gr
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);	// corr_I_rr
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);	// cov_Ip_b
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);	// cov_Ip_g
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);	// cov_Ip_r

			_mm256_store_ps(meanIb_p, mTmp00);
			meanIb_p += step;
			_mm256_store_ps(meanIg_p, mTmp01);
			meanIg_p += step;
			_mm256_store_ps(meanIr_p, mTmp02);
			meanIr_p += step;

			mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));
			mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));
			mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));
			mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));
			mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));
			mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));
			mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
			mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
			mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

			mVar00 = _mm256_add_ps(mVar00, mEps);
			mVar03 = _mm256_add_ps(mVar03, mEps);
			mVar05 = _mm256_add_ps(mVar05, mEps);

			mTmp04 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar03, mVar05));	// bb*gg*rr
			mTmp05 = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));	// bg*br*gr
			mTmp06 = _mm256_mul_ps(mVar00, _mm256_mul_ps(mVar04, mVar04));	// bb*gr*gr
			mTmp07 = _mm256_mul_ps(mVar03, _mm256_mul_ps(mVar02, mVar02));	// gg*br*br
			mTmp08 = _mm256_mul_ps(mVar05, _mm256_mul_ps(mVar01, mVar01));	// rr*bg*bg

			mDet = _mm256_add_ps(mTmp04, mTmp05);
			mDet = _mm256_add_ps(mDet, mTmp05);
			mDet = _mm256_sub_ps(mDet, mTmp06);
			mDet = _mm256_sub_ps(mDet, mTmp07);
			mDet = _mm256_sub_ps(mDet, mTmp08);
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);
			_mm256_store_ps(det_p, mDet);
			det_p += step;

			/*
			mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
			*/
			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

			_mm256_store_ps(cov0_p, mTmp04);
			cov0_p += step;
			_mm256_store_ps(cov1_p, mTmp05);
			cov1_p += step;
			_mm256_store_ps(cov2_p, mTmp06);
			cov2_p += step;
			_mm256_store_ps(cov3_p, mTmp07);
			cov3_p += step;
			_mm256_store_ps(cov4_p, mTmp08);
			cov4_p += step;
			_mm256_store_ps(cov5_p, mTmp09);
			cov5_p += step;

			/*
			mTmp10 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp04), _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_add_ps(mTmp10, _mm256_mul_ps(mCov02, mTmp06));
			*/
			mTmp10 = _mm256_fmadd_ps(mCov00, mTmp04, _mm256_mul_ps(mCov01, mTmp05));
			mTmp10 = _mm256_fmadd_ps(mCov02, mTmp06, mTmp10);
			mTmp10 = _mm256_mul_ps(mTmp10, mDet);
			_mm256_storeu_ps(a_b_p, mTmp10);
			a_b_p += step;

			/*
			mTmp11 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp05), _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_add_ps(mTmp11, _mm256_mul_ps(mCov02, mTmp08));
			*/
			mTmp11 = _mm256_fmadd_ps(mCov00, mTmp05, _mm256_mul_ps(mCov01, mTmp07));
			mTmp11 = _mm256_fmadd_ps(mCov02, mTmp08, mTmp11);
			mTmp11 = _mm256_mul_ps(mTmp11, mDet);
			_mm256_storeu_ps(a_g_p, mTmp11);
			a_g_p += step;

			/*
			mTmp12 = _mm256_add_ps(_mm256_mul_ps(mCov00, mTmp06), _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_add_ps(mTmp12, _mm256_mul_ps(mCov02, mTmp09));
			*/
			mTmp12 = _mm256_fmadd_ps(mCov00, mTmp06, _mm256_mul_ps(mCov01, mTmp08));
			mTmp12 = _mm256_fmadd_ps(mCov02, mTmp09, mTmp12);
			mTmp12 = _mm256_mul_ps(mTmp12, mDet);
			_mm256_storeu_ps(a_r_p, mTmp12);
			a_r_p += step;

			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp10, mTmp00));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp11, mTmp01));
			mTmp03 = _mm256_sub_ps(mTmp03, _mm256_mul_ps(mTmp12, mTmp02));
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}
