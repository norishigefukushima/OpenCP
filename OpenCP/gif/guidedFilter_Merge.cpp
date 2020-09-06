#include "guidedFilter_Merge.h"
#include <iostream>
using namespace cv;
using namespace std;
using namespace cp;
guidedImageFilter_Merge_Base::guidedImageFilter_Merge_Base(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType, const bool isInit)
	: GuidedFilterBase(_src, _guide, _dest, _r, _eps), parallelType(_parallelType)
{
	implementation = GUIDED_MERGE;
	if (isInit)init();
}

void guidedImageFilter_Merge_Base::init()
{
	//cout << "merge_nonvec::init" << endl;
	b.create(src.size(), CV_32F);

	if (guide.channels() == 1)
	{
		temp.resize(4);
		for (int i = 0; i < temp.size(); i++)
		{
			temp[i].create(src.size(), CV_32F);
		}

		a.create(src.size(), CV_32F);
	}
	else if (guide.channels() == 3)
	{
		temp.resize(13);
		for (int i = 0; i < temp.size(); i++)
		{
			temp[i].create(src.size(), CV_32F);
		}

		va.resize(3);
		for (int i = 0; i < va.size(); i++)
		{
			va[i].create(src.size(), CV_32F);
		}
	}
}

void guidedImageFilter_Merge_Base::filter()
{
	//cout << "Merge: parallel type " << parallelType << endl;
	if (src.channels() == 1 && guide.channels() == 1)
	{
		filter_Guide1(src, dest);
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		split(guide, vI);

		filter_Guide3(src, dest);
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
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
		split(src, vsrc);
		split(dest, vdest);

		filter_Guide3(vsrc[0], vdest[0]);
		filter_Guide3(vsrc[1], vdest[1]);
		filter_Guide3(vsrc[2], vdest[2]);

		merge(vdest, dest);
	}
}

void guidedImageFilter_Merge_Base::filterVector()
{
	//cout << "Merge: parallel type " << parallelType << endl;
	if (src.channels() == 1 && guide.channels() == 1)
	{
		filter_Guide1(vsrc[0], vdest[0]);
	}
	else if (src.channels() == 1 && guide.channels() == 3)
	{
		vI = vguide;

		filter_Guide3(vsrc[0], vdest[0]);
	}
	else if (src.channels() == 3 && guide.channels() == 1)
	{
		filter_Guide1(vsrc[0], vdest[0]);
		filter_Guide1(vsrc[1], vdest[1]);
		filter_Guide1(vsrc[2], vdest[2]);
	}
	else if (src.channels() == 3 && guide.channels() == 3)
	{
		vI = vguide;

		filter_Guide3(vsrc[0], vdest[0]);
		filter_Guide3(vsrc[1], vdest[1]);
		filter_Guide3(vsrc[2], vdest[2]);
	}
}

void guidedImageFilter_Merge_Base::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1(input, guide, temp, r, parallelType).filter();
	ColumnSumFilter_Ip2ab_Guide1_nonVec(temp, a, b, r, eps, parallelType).filter();
	RowSumFilter_ab2q_Guide1(a, b, temp, r, parallelType).filter();
	ColumnSumFilter_ab2q_Guide1_nonVec(temp, guide, output, r, parallelType).filter();
}

void guidedImageFilter_Merge_Base::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3(input, vI, temp, r, parallelType).filter();
	ColumnSumFilter_Ip2ab_Guide3_nonVec(temp, va, b, r, eps, parallelType).filter();
	RowSumFilter_ab2q_Guide3(va, b, temp, r, parallelType).filter();
	ColumnSumFilter_ab2q_Guide3_nonVec(temp, vI, output, r, parallelType).filter();
}


void guidedFilter_Merge_SSE::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1(input, guide, temp, r, parallelType).filter();
	ColumnSumFilter_Ip2ab_Guide1_SSE(temp, a, b, r, eps, parallelType).filter();
	RowSumFilter_ab2q_Guide1(a, b, temp, r, parallelType).filter();
	ColumnSumFilter_ab2q_Guide1_SSE(temp, guide, output, r, parallelType).filter();
}

void guidedFilter_Merge_SSE::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3(input, vI, temp, r, parallelType).filter();
	ColumnSumFilter_Ip2ab_Guide3_SSE(temp, va, b, r, eps, parallelType).filter();
	RowSumFilter_ab2q_Guide3(va, b, temp, r, parallelType).filter();
	ColumnSumFilter_ab2q_Guide3_SSE(temp, vI, output, r, parallelType).filter();
}

void guidedFilter_Merge_AVX::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide1(input, guide, temp, r, parallelType).filter();
	ColumnSumFilter_Ip2ab_Guide1_AVX(temp, a, b, r, eps, parallelType).filter();
	RowSumFilter_ab2q_Guide1(a, b, temp, r, parallelType).filter();
	ColumnSumFilter_ab2q_Guide1_AVX(temp, guide, output, r, parallelType).filter();
}


using namespace cv;
void ab2q_guide3_sep_VHI_Unroll2_AVX_omp(Mat& a_b, Mat& a_g, Mat& a_r, Mat& b, Mat& guide_b, Mat& guide_g, Mat& guide_r, const int r, Mat& dest);

void guidedFilter_Merge_AVX::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	RowSumFilter_Ip2ab_Guide3(input, vI, temp, r, parallelType).filter();
	ColumnSumFilter_Ip2ab_Guide3_AVX(temp, va, b, r, eps, parallelType).filter();

	ab2q_guide3_sep_VHI_Unroll2_AVX_omp(va[0], va[1], va[2], b, vI[0], vI[1], vI[2], r, output);
	//RowSumFilter_ab2q_Guide3(va, b, temp, r, parallelType).filter();
	//ColumnSumFilter_ab2q_Guide3_AVX(temp, vI, output, r, parallelType).filter();
}



void RowSumFilter_Ip2ab_Guide1::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* p_p1 = p.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* p_p2 = p.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(j); // mean_I
		float* v1_p = tempVec[1].ptr<float>(j); // mean_p
		float* v2_p = tempVec[2].ptr<float>(j); // corr_I
		float* v3_p = tempVec[3].ptr<float>(j); // corr_Ip

		float sum[4] = { 0.f };
		sum[0] += *I_p1 * (r + 1);
		sum[1] += *p_p1 * (r + 1);
		sum[2] += (*I_p1 * *I_p1) * (r + 1);
		sum[3] += (*I_p1 * *p_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *I_p2;
			sum[1] += *p_p2;
			sum[2] += *I_p2 * *I_p2;
			sum[3] += *I_p2 * *p_p2;
			I_p2++;
			p_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;
		*v2_p = sum[2];
		v2_p++;
		*v3_p = sum[3];
		v3_p++;

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide1::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* p_p1 = p.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* p_p2 = p.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(j); // mean_I
		float* v1_p = tempVec[1].ptr<float>(j); // mean_p
		float* v2_p = tempVec[2].ptr<float>(j); // corr_I
		float* v3_p = tempVec[3].ptr<float>(j); // corr_Ip

		float sum[4] = { 0.f };
		sum[0] += *I_p1 * (r + 1);
		sum[1] += *p_p1 * (r + 1);
		sum[2] += (*I_p1 * *I_p1) * (r + 1);
		sum[3] += (*I_p1 * *p_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *I_p2;
			sum[1] += *p_p2;
			sum[2] += *I_p2 * *I_p2;
			sum[3] += *I_p2 * *p_p2;
			I_p2++;
			p_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;
		*v2_p = sum[2];
		v2_p++;
		*v3_p = sum[3];
		v3_p++;

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide1::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* I_p1 = I.ptr<float>(j);
		float* p_p1 = p.ptr<float>(j);
		float* I_p2 = I.ptr<float>(j) + 1;
		float* p_p2 = p.ptr<float>(j) + 1;

		float* v0_p = tempVec[0].ptr<float>(j); // mean_I
		float* v1_p = tempVec[1].ptr<float>(j); // mean_p
		float* v2_p = tempVec[2].ptr<float>(j); // corr_I
		float* v3_p = tempVec[3].ptr<float>(j); // corr_Ip

		float sum[4] = { 0.f };
		sum[0] += *I_p1 * (r + 1);
		sum[1] += *p_p1 * (r + 1);
		sum[2] += (*I_p1 * *I_p1) * (r + 1);
		sum[3] += (*I_p1 * *p_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *I_p2;
			sum[1] += *p_p2;
			sum[2] += *I_p2 * *I_p2;
			sum[3] += *I_p2 * *p_p2;
			I_p2++;
			p_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;
		*v2_p = sum[2];
		v2_p++;
		*v3_p = sum[3];
		v3_p++;

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;
			I_p2++;
			p_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *I_p2 - *I_p1;
			sum[1] += *p_p2 - *p_p1;
			sum[2] += (*I_p2 * *I_p2) - (*I_p1 * *I_p1);
			sum[3] += (*I_p2 * *p_p2) - (*I_p1 * *p_p1);
			I_p1++;
			p_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
			*v2_p = sum[2];
			v2_p++;
			*v3_p = sum[3];
			v3_p++;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_nonVec::filter_naive_impl()
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
		var_I = tmp[2] - tmp[0] * tmp[0];
		cov_Ip = tmp[3] - tmp[0] * tmp[1];
		*a_p = cov_Ip / (var_I + eps);
		*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_nonVec::filter_omp_impl()
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
		var_I = tmp[2] - tmp[0] * tmp[0];
		cov_Ip = tmp[3] - tmp[0] * tmp[1];
		*a_p = cov_Ip / (var_I + eps);
		*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_nonVec::operator()(const cv::Range& range) const
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
		var_I = tmp[2] - tmp[0] * tmp[0];
		cov_Ip = tmp[3] - tmp[0] * tmp[1];
		*a_p = cov_Ip / (var_I + eps);
		*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
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
			var_I = tmp[2] - tmp[0] * tmp[0];
			cov_Ip = tmp[3] - tmp[0] * tmp[1];
			*a_p = cov_Ip / (var_I + eps);
			*b_p = tmp[1] - *a_p * tmp[0];
			a_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_SSE::filter_naive_impl()
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
		m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

		m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
		m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

		_mm_store_ps(a_p, m6);
		_mm_store_ps(b_p, m7);
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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_SSE::filter_omp_impl()
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
		m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

		m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
		m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

		_mm_store_ps(a_p, m6);
		_mm_store_ps(b_p, m7);
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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_SSE::operator()(const cv::Range& range) const
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
		m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

		m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
		m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

		_mm_store_ps(a_p, m6);
		_mm_store_ps(b_p, m7);
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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

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
			m5 = _mm_sub_ps(m3, _mm_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm_div_ps(m5, _mm_add_ps(m4, mEps));
			m7 = _mm_sub_ps(m1, _mm_mul_ps(m6, m0));

			_mm_store_ps(a_p, m6);
			_mm_store_ps(b_p, m7);

			v0_p1 += step;
			v1_p1 += step;
			v2_p1 += step;
			v3_p1 += step;
			a_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide1_AVX::filter_naive_impl()
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
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(v2_p2));
			v2_p2 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_load_ps(v3_p2));
			v3_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
		m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

		m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
		m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

		_mm256_store_ps(a_p, m6);
		a_p += step;
		_mm256_store_ps(b_p, m7);
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		mTmp[2] = _mm256_load_ps(v2_p1);
		mTmp[3] = _mm256_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			v2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			v3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, mTmp[3]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
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
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			v2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			v2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			v3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));
			v3_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
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
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			v2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));
			v3_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_AVX::filter_omp_impl()
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
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(v2_p2));
			v2_p2 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_load_ps(v3_p2));
			v3_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
		m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

		m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
		m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

		_mm256_store_ps(a_p, m6);
		a_p += step;
		_mm256_store_ps(b_p, m7);
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		mTmp[2] = _mm256_load_ps(v2_p1);
		mTmp[3] = _mm256_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			v2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			v3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, mTmp[3]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
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
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			v2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			v2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			v3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));
			v3_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
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
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			v2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));
			v3_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide1_AVX::operator()(const cv::Range& range) const
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
			v0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_load_ps(v2_p2));
			v2_p2 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_load_ps(v3_p2));
			v3_p2 += step;
		}
		m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
		m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
		m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
		m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
		m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
		m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

		m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
		m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

		_mm256_store_ps(a_p, m6);
		a_p += step;
		_mm256_store_ps(b_p, m7);
		b_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		mTmp[2] = _mm256_load_ps(v2_p1);
		mTmp[3] = _mm256_load_ps(v3_p1);
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(v0_p2));
			v0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, mTmp[0]);
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(v1_p2));
			v1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, mTmp[1]);
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			v2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, mTmp[2]);
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			v3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, mTmp[3]);

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
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
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(v2_p2));
			v2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			v2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(v3_p2));
			v3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));
			v3_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
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
			v0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, mTmp[1]);
			mSum1 = _mm256_sub_ps(mSum1, _mm256_loadu_ps(v1_p1));
			v1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, mTmp[2]);
			mSum2 = _mm256_sub_ps(mSum2, _mm256_loadu_ps(v2_p1));
			v2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, mTmp[3]);
			mSum3 = _mm256_sub_ps(mSum3, _mm256_loadu_ps(v3_p1));
			v3_p1 += step;

			m0 = _mm256_mul_ps(mSum0, mDiv);	//mean_I
			m1 = _mm256_mul_ps(mSum1, mDiv);	//mean_p
			m2 = _mm256_mul_ps(mSum2, mDiv);	//corr_I
			m3 = _mm256_mul_ps(mSum3, mDiv);	//corr_Ip
			m4 = _mm256_sub_ps(m2, _mm256_mul_ps(m0, m0));	//var_I
			m5 = _mm256_sub_ps(m3, _mm256_mul_ps(m0, m1));	//cov_Ip

			m6 = _mm256_div_ps(m5, _mm256_add_ps(m4, mEps));
			m7 = _mm256_sub_ps(m1, _mm256_mul_ps(m6, m0));

			_mm256_storeu_ps(a_p, m6);
			a_p += step;
			_mm256_storeu_ps(b_p, m7);
			b_p += step;
		}
	}
}



void RowSumFilter_ab2q_Guide1::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* a_p1 = a.ptr<float>(j);
		float* b_p1 = b.ptr<float>(j);
		float* a_p2 = a.ptr<float>(j) + 1;
		float* b_p2 = b.ptr<float>(j) + 1;
		float* v0_p = tempVec[0].ptr<float>(j);
		float* v1_p = tempVec[1].ptr<float>(j);

		float sum[2] = { 0.f };
		sum[0] += *a_p1 * (r + 1);
		sum[1] += *b_p1 * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *a_p2;
			a_p2++;
			sum[1] += *b_p2;
			b_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p2++;
			sum[1] += *b_p2 - *b_p1;
			b_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p1++;
			a_p2++;
			sum[1] += *b_p2 - *b_p1;
			b_p1++;
			b_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p1++;
			sum[1] += *b_p2 - *b_p1;
			b_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
	}
}

void RowSumFilter_ab2q_Guide1::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* a_p1 = a.ptr<float>(j);
		float* b_p1 = b.ptr<float>(j);
		float* a_p2 = a.ptr<float>(j) + 1;
		float* b_p2 = b.ptr<float>(j) + 1;
		float* v0_p = tempVec[0].ptr<float>(j);
		float* v1_p = tempVec[1].ptr<float>(j);

		float sum[2] = { 0.f };
		sum[0] += *a_p1 * (r + 1);
		sum[1] += *b_p1 * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *a_p2;
			a_p2++;
			sum[1] += *b_p2;
			b_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p2++;
			sum[1] += *b_p2 - *b_p1;
			b_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p1++;
			a_p2++;
			sum[1] += *b_p2 - *b_p1;
			b_p1++;
			b_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p1++;
			sum[1] += *b_p2 - *b_p1;
			b_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
	}
}

void RowSumFilter_ab2q_Guide1::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* a_p1 = a.ptr<float>(j);
		float* b_p1 = b.ptr<float>(j);
		float* a_p2 = a.ptr<float>(j) + 1;
		float* b_p2 = b.ptr<float>(j) + 1;
		float* v0_p = tempVec[0].ptr<float>(j);
		float* v1_p = tempVec[1].ptr<float>(j);

		float sum[2] = { 0.f };
		sum[0] += *a_p1 * (r + 1);
		sum[1] += *b_p1 * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum[0] += *a_p2;
			a_p2++;
			sum[1] += *b_p2;
			b_p2++;
		}
		*v0_p = sum[0];
		v0_p++;
		*v1_p = sum[1];
		v1_p++;

		for (int i = 1; i <= r; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p2++;
			sum[1] += *b_p2 - *b_p1;
			b_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p1++;
			a_p2++;
			sum[1] += *b_p2 - *b_p1;
			b_p1++;
			b_p2++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum[0] += *a_p2 - *a_p1;
			a_p1++;
			sum[1] += *b_p2 - *b_p1;
			b_p1++;

			*v0_p = sum[0];
			v0_p++;
			*v1_p = sum[1];
			v1_p++;
		}
	}
}



void ColumnSumFilter_ab2q_Guide1_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;
		float* I_p = I.ptr<float>(0) + i;
		float* q_p = q.ptr<float>(0) + i;

		float suma = 0.f, sumb = 0.f;

		suma = *v0_p1 * (r + 1);
		sumb = *v1_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			suma += *v0_p2;
			v0_p2 += step;
			sumb += *v1_p2;
			v1_p2 += step;
		}
		*q_p = (suma * *I_p + sumb) * div;
		I_p += step;
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p2 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p2 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p1 += step;
			v0_p2 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p1 += step;
			v1_p2 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p1 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p1 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide1_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;
		float* I_p = I.ptr<float>(0) + i;
		float* q_p = q.ptr<float>(0) + i;

		float suma = 0.f, sumb = 0.f;

		suma = *v0_p1 * (r + 1);
		sumb = *v1_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			suma += *v0_p2;
			v0_p2 += step;
			sumb += *v1_p2;
			v1_p2 += step;
		}
		*q_p = (suma * *I_p + sumb) * div;
		I_p += step;
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p2 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p2 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p1 += step;
			v0_p2 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p1 += step;
			v1_p2 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p1 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p1 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide1_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i;
		float* I_p = I.ptr<float>(0) + i;
		float* q_p = q.ptr<float>(0) + i;

		float suma = 0.f, sumb = 0.f;

		suma = *v0_p1 * (r + 1);
		sumb = *v1_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			suma += *v0_p2;
			v0_p2 += step;
			sumb += *v1_p2;
			v1_p2 += step;
		}
		*q_p = (suma * *I_p + sumb) * div;
		I_p += step;
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p2 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p2 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p1 += step;
			v0_p2 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p1 += step;
			v1_p2 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			suma += *v0_p2 - *v0_p1;
			v0_p1 += step;
			sumb += *v1_p2 - *v1_p1;
			v1_p1 += step;

			*q_p = (suma * *I_p + sumb) * div;
			I_p += step;
			q_p += step;
		}
	}
}



void ColumnSumFilter_ab2q_Guide1_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i * 4;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i * 4;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* I_p = I.ptr<float>(0) + i * 4;
		float* q_p = q.ptr<float>(0) + i * 4;

		__m128 ma = _mm_setzero_ps();
		__m128 mb = _mm_setzero_ps();
		__m128 mI = _mm_load_ps(I_p);
		__m128 mq = _mm_setzero_ps();
		__m128 mTmp_a, mTmp_b;

		ma = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mb = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			v0_p2 += step;
			v1_p2 += step;
		}
		mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
		_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));
		I_p += step;
		q_p += step;

		mTmp_a = _mm_load_ps(v0_p1);
		mTmp_b = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			ma = _mm_sub_ps(ma, mTmp_a);
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			mb = _mm_sub_ps(mb, mTmp_b);
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p2 += step;
			v1_p2 += step;
			I_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			ma = _mm_sub_ps(ma, _mm_load_ps(v0_p1));
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			mb = _mm_sub_ps(mb, _mm_load_ps(v1_p1));
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p1 += step;
			v1_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			I_p += step;
			q_p += step;
		}
		mTmp_a = _mm_load_ps(v0_p2);
		mTmp_b = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			ma = _mm_add_ps(ma, mTmp_a);
			ma = _mm_sub_ps(ma, _mm_load_ps(v0_p1));
			mb = _mm_add_ps(mb, mTmp_b);
			mb = _mm_sub_ps(mb, _mm_load_ps(v1_p1));
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p1 += step;
			v1_p1 += step;
			I_p += step;
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide1_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i * 4;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i * 4;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* I_p = I.ptr<float>(0) + i * 4;
		float* q_p = q.ptr<float>(0) + i * 4;

		__m128 ma = _mm_setzero_ps();
		__m128 mb = _mm_setzero_ps();
		__m128 mI = _mm_load_ps(I_p);
		__m128 mq = _mm_setzero_ps();
		__m128 mTmp_a, mTmp_b;

		ma = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mb = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			v0_p2 += step;
			v1_p2 += step;
		}
		mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
		_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));
		I_p += step;
		q_p += step;

		mTmp_a = _mm_load_ps(v0_p1);
		mTmp_b = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			ma = _mm_sub_ps(ma, mTmp_a);
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			mb = _mm_sub_ps(mb, mTmp_b);
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p2 += step;
			v1_p2 += step;
			I_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			ma = _mm_sub_ps(ma, _mm_load_ps(v0_p1));
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			mb = _mm_sub_ps(mb, _mm_load_ps(v1_p1));
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p1 += step;
			v1_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			I_p += step;
			q_p += step;
		}
		mTmp_a = _mm_load_ps(v0_p2);
		mTmp_b = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			ma = _mm_add_ps(ma, mTmp_a);
			ma = _mm_sub_ps(ma, _mm_load_ps(v0_p1));
			mb = _mm_add_ps(mb, mTmp_b);
			mb = _mm_sub_ps(mb, _mm_load_ps(v1_p1));
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p1 += step;
			v1_p1 += step;
			I_p += step;
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide1_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i * 4;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i * 4;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* I_p = I.ptr<float>(0) + i * 4;
		float* q_p = q.ptr<float>(0) + i * 4;

		__m128 ma = _mm_setzero_ps();
		__m128 mb = _mm_setzero_ps();
		__m128 mI = _mm_load_ps(I_p);
		__m128 mq = _mm_setzero_ps();
		__m128 mTmp_a, mTmp_b;

		ma = _mm_mul_ps(mBorder, _mm_load_ps(v0_p1));
		mb = _mm_mul_ps(mBorder, _mm_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			v0_p2 += step;
			v1_p2 += step;
		}
		mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
		_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));
		I_p += step;
		q_p += step;

		mTmp_a = _mm_load_ps(v0_p1);
		mTmp_b = _mm_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			ma = _mm_sub_ps(ma, mTmp_a);
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			mb = _mm_sub_ps(mb, mTmp_b);
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p2 += step;
			v1_p2 += step;
			I_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			ma = _mm_add_ps(ma, _mm_load_ps(v0_p2));
			ma = _mm_sub_ps(ma, _mm_load_ps(v0_p1));
			mb = _mm_add_ps(mb, _mm_load_ps(v1_p2));
			mb = _mm_sub_ps(mb, _mm_load_ps(v1_p1));
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p1 += step;
			v1_p1 += step;
			v0_p2 += step;
			v1_p2 += step;
			I_p += step;
			q_p += step;
		}
		mTmp_a = _mm_load_ps(v0_p2);
		mTmp_b = _mm_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			ma = _mm_add_ps(ma, mTmp_a);
			ma = _mm_sub_ps(ma, _mm_load_ps(v0_p1));
			mb = _mm_add_ps(mb, mTmp_b);
			mb = _mm_sub_ps(mb, _mm_load_ps(v1_p1));
			mI = _mm_load_ps(I_p);

			mq = _mm_add_ps(_mm_mul_ps(mI, ma), mb);
			_mm_store_ps(q_p, _mm_mul_ps(mq, mDiv));

			v0_p1 += step;
			v1_p1 += step;
			I_p += step;
			q_p += step;
		}
	}
}



void ColumnSumFilter_ab2q_Guide1_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* I_p = I.ptr<float>(0) + i * 8;
		float* q_p = q.ptr<float>(0) + i * 8;

		__m256 ma = _mm256_setzero_ps();
		__m256 mb = _mm256_setzero_ps();
		__m256 mq = _mm256_setzero_ps();
		__m256 mTmp[2];

		__m256 mI = _mm256_load_ps(I_p);
		I_p += step;

		ma = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mb = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
		_mm256_store_ps(q_p, _mm256_mul_ps(mq, mDiv));
		q_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			ma = _mm256_sub_ps(ma, mTmp[0]);
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mb = _mm256_sub_ps(mb, mTmp[1]);
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			ma = _mm256_sub_ps(ma, _mm256_load_ps(v0_p1));
			v0_p1 += step;
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mb = _mm256_sub_ps(mb, _mm256_load_ps(v1_p1));
			v1_p1 += step;
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			ma = _mm256_add_ps(ma, mTmp[0]);
			ma = _mm256_sub_ps(ma, _mm256_load_ps(v0_p1));
			v0_p1 += step;
			mb = _mm256_add_ps(mb, mTmp[1]);
			mb = _mm256_sub_ps(mb, _mm256_load_ps(v1_p1));
			v1_p1 += step;
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide1_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* I_p = I.ptr<float>(0) + i * 8;
		float* q_p = q.ptr<float>(0) + i * 8;

		__m256 ma = _mm256_setzero_ps();
		__m256 mb = _mm256_setzero_ps();
		__m256 mq = _mm256_setzero_ps();
		__m256 mTmp[2];

		__m256 mI = _mm256_load_ps(I_p);
		I_p += step;

		ma = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mb = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
		_mm256_store_ps(q_p, _mm256_mul_ps(mq, mDiv));
		q_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			ma = _mm256_sub_ps(ma, mTmp[0]);
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mb = _mm256_sub_ps(mb, mTmp[1]);
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			ma = _mm256_sub_ps(ma, _mm256_load_ps(v0_p1));
			v0_p1 += step;
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mb = _mm256_sub_ps(mb, _mm256_load_ps(v1_p1));
			v1_p1 += step;
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			ma = _mm256_add_ps(ma, mTmp[0]);
			ma = _mm256_sub_ps(ma, _mm256_load_ps(v0_p1));
			v0_p1 += step;
			mb = _mm256_add_ps(mb, mTmp[1]);
			mb = _mm256_sub_ps(mb, _mm256_load_ps(v1_p1));
			v1_p1 += step;
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide1_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* v0_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* v1_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* v0_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* v1_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* I_p = I.ptr<float>(0) + i * 8;
		float* q_p = q.ptr<float>(0) + i * 8;

		__m256 ma = _mm256_setzero_ps();
		__m256 mb = _mm256_setzero_ps();
		__m256 mq = _mm256_setzero_ps();
		__m256 mTmp[2];

		__m256 mI = _mm256_load_ps(I_p);
		I_p += step;

		ma = _mm256_mul_ps(mBorder, _mm256_load_ps(v0_p1));
		mb = _mm256_mul_ps(mBorder, _mm256_load_ps(v1_p1));
		for (int j = 1; j <= r; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
		}
		mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
		_mm256_store_ps(q_p, _mm256_mul_ps(mq, mDiv));
		q_p += step;

		mTmp[0] = _mm256_load_ps(v0_p1);
		mTmp[1] = _mm256_load_ps(v1_p1);
		for (int j = 1; j <= r; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			ma = _mm256_sub_ps(ma, mTmp[0]);
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mb = _mm256_sub_ps(mb, mTmp[1]);
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			ma = _mm256_add_ps(ma, _mm256_load_ps(v0_p2));
			v0_p2 += step;
			ma = _mm256_sub_ps(ma, _mm256_load_ps(v0_p1));
			v0_p1 += step;
			mb = _mm256_add_ps(mb, _mm256_load_ps(v1_p2));
			v1_p2 += step;
			mb = _mm256_sub_ps(mb, _mm256_load_ps(v1_p1));
			v1_p1 += step;
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
		mTmp[0] = _mm256_load_ps(v0_p2);
		mTmp[1] = _mm256_load_ps(v1_p2);
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			ma = _mm256_add_ps(ma, mTmp[0]);
			ma = _mm256_sub_ps(ma, _mm256_load_ps(v0_p1));
			v0_p1 += step;
			mb = _mm256_add_ps(mb, mTmp[1]);
			mb = _mm256_sub_ps(mb, _mm256_load_ps(v1_p1));
			v1_p1 += step;
			mI = _mm256_load_ps(I_p);
			I_p += step;

			mq = _mm256_add_ps(_mm256_mul_ps(mI, ma), mb);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mq, mDiv));
			q_p += step;
		}
	}
}



/* --- Guide3 --- */
void RowSumFilter_Ip2ab_Guide3::filter_naive_impl()
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

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(j);	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(j);	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(j);	// mean_p
		float* d04_p = tempVec[4].ptr<float>(j);	// corr_I_bb
		float* d05_p = tempVec[5].ptr<float>(j);	// corr_I_bg
		float* d06_p = tempVec[6].ptr<float>(j);	// corr_I_br
		float* d07_p = tempVec[7].ptr<float>(j);	// corr_I_gg
		float* d08_p = tempVec[8].ptr<float>(j);	// corr_I_gr
		float* d09_p = tempVec[9].ptr<float>(j);	// corr_I_rr
		float* d10_p = tempVec[10].ptr<float>(j);	// cov_Ip_b
		float* d11_p = tempVec[11].ptr<float>(j);	// cov_Ip_g
		float* d12_p = tempVec[12].ptr<float>(j);	// cov_Ip_r

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08, sum09, sum10, sum11, sum12;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = sum09 = sum10 = sum11 = sum12 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += *s03_p1 * (r + 1);
		sum04 += (*s00_p1 * *s00_p1) * (r + 1);
		sum05 += (*s00_p1 * *s01_p1) * (r + 1);
		sum06 += (*s00_p1 * *s02_p1) * (r + 1);
		sum07 += (*s01_p1 * *s01_p1) * (r + 1);
		sum08 += (*s01_p1 * *s02_p1) * (r + 1);
		sum09 += (*s02_p1 * *s02_p1) * (r + 1);
		sum10 += (*s03_p1 * *s00_p1) * (r + 1);
		sum11 += (*s03_p1 * *s01_p1) * (r + 1);
		sum12 += (*s03_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum01 += *s01_p2;
			sum02 += *s02_p2;
			sum03 += *s03_p2;
			sum04 += *s00_p2 * *s00_p2;
			sum05 += *s00_p2 * *s01_p2;
			sum06 += *s00_p2 * *s02_p2;
			sum07 += *s01_p2 * *s01_p2;
			sum08 += *s01_p2 * *s02_p2;
			sum09 += *s02_p2 * *s02_p2;
			sum10 += *s03_p2 * *s00_p2;
			s00_p2++;
			sum11 += *s03_p2 * *s01_p2;
			s01_p2++;
			sum12 += *s03_p2 * *s02_p2;
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
		*d09_p = sum09;
		d09_p++;
		*d10_p = sum10;
		d10_p++;
		*d11_p = sum11;
		d11_p++;
		*d12_p = sum12;
		d12_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p2++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p2++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p2++;
			s03_p2++;
			*d12_p = sum12;
			d12_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			s00_p2++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			s01_p2++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;
			s03_p1++;
			s03_p2++;
			*d12_p = sum12;
			d12_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s03_p1++;
			*d12_p = sum12;
			d12_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide3::filter_omp_impl()
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

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(j);	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(j);	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(j);	// mean_p
		float* d04_p = tempVec[4].ptr<float>(j);	// corr_I_bb
		float* d05_p = tempVec[5].ptr<float>(j);	// corr_I_bg
		float* d06_p = tempVec[6].ptr<float>(j);	// corr_I_br
		float* d07_p = tempVec[7].ptr<float>(j);	// corr_I_gg
		float* d08_p = tempVec[8].ptr<float>(j);	// corr_I_gr
		float* d09_p = tempVec[9].ptr<float>(j);	// corr_I_rr
		float* d10_p = tempVec[10].ptr<float>(j);	// cov_Ip_b
		float* d11_p = tempVec[11].ptr<float>(j);	// cov_Ip_g
		float* d12_p = tempVec[12].ptr<float>(j);	// cov_Ip_r

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08, sum09, sum10, sum11, sum12;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = sum09 = sum10 = sum11 = sum12 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += *s03_p1 * (r + 1);
		sum04 += (*s00_p1 * *s00_p1) * (r + 1);
		sum05 += (*s00_p1 * *s01_p1) * (r + 1);
		sum06 += (*s00_p1 * *s02_p1) * (r + 1);
		sum07 += (*s01_p1 * *s01_p1) * (r + 1);
		sum08 += (*s01_p1 * *s02_p1) * (r + 1);
		sum09 += (*s02_p1 * *s02_p1) * (r + 1);
		sum10 += (*s03_p1 * *s00_p1) * (r + 1);
		sum11 += (*s03_p1 * *s01_p1) * (r + 1);
		sum12 += (*s03_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum01 += *s01_p2;
			sum02 += *s02_p2;
			sum03 += *s03_p2;
			sum04 += *s00_p2 * *s00_p2;
			sum05 += *s00_p2 * *s01_p2;
			sum06 += *s00_p2 * *s02_p2;
			sum07 += *s01_p2 * *s01_p2;
			sum08 += *s01_p2 * *s02_p2;
			sum09 += *s02_p2 * *s02_p2;
			sum10 += *s03_p2 * *s00_p2;
			s00_p2++;
			sum11 += *s03_p2 * *s01_p2;
			s01_p2++;
			sum12 += *s03_p2 * *s02_p2;
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
		*d09_p = sum09;
		d09_p++;
		*d10_p = sum10;
		d10_p++;
		*d11_p = sum11;
		d11_p++;
		*d12_p = sum12;
		d12_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p2++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p2++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p2++;
			s03_p2++;
			*d12_p = sum12;
			d12_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			s00_p2++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			s01_p2++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;
			s03_p1++;
			s03_p2++;
			*d12_p = sum12;
			d12_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s03_p1++;
			*d12_p = sum12;
			d12_p++;
		}
	}
}

void RowSumFilter_Ip2ab_Guide3::operator()(const cv::Range& range) const
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

		float* d00_p = tempVec[0].ptr<float>(j);	// mean_I_b
		float* d01_p = tempVec[1].ptr<float>(j);	// mean_I_g
		float* d02_p = tempVec[2].ptr<float>(j);	// mean_I_r
		float* d03_p = tempVec[3].ptr<float>(j);	// mean_p
		float* d04_p = tempVec[4].ptr<float>(j);	// corr_I_bb
		float* d05_p = tempVec[5].ptr<float>(j);	// corr_I_bg
		float* d06_p = tempVec[6].ptr<float>(j);	// corr_I_br
		float* d07_p = tempVec[7].ptr<float>(j);	// corr_I_gg
		float* d08_p = tempVec[8].ptr<float>(j);	// corr_I_gr
		float* d09_p = tempVec[9].ptr<float>(j);	// corr_I_rr
		float* d10_p = tempVec[10].ptr<float>(j);	// cov_Ip_b
		float* d11_p = tempVec[11].ptr<float>(j);	// cov_Ip_g
		float* d12_p = tempVec[12].ptr<float>(j);	// cov_Ip_r

		float sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07, sum08, sum09, sum10, sum11, sum12;
		sum00 = sum01 = sum02 = sum03 = sum04 = sum05 = sum06 = sum07 = sum08 = sum09 = sum10 = sum11 = sum12 = 0.f;

		sum00 += *s00_p1 * (r + 1);
		sum01 += *s01_p1 * (r + 1);
		sum02 += *s02_p1 * (r + 1);
		sum03 += *s03_p1 * (r + 1);
		sum04 += (*s00_p1 * *s00_p1) * (r + 1);
		sum05 += (*s00_p1 * *s01_p1) * (r + 1);
		sum06 += (*s00_p1 * *s02_p1) * (r + 1);
		sum07 += (*s01_p1 * *s01_p1) * (r + 1);
		sum08 += (*s01_p1 * *s02_p1) * (r + 1);
		sum09 += (*s02_p1 * *s02_p1) * (r + 1);
		sum10 += (*s03_p1 * *s00_p1) * (r + 1);
		sum11 += (*s03_p1 * *s01_p1) * (r + 1);
		sum12 += (*s03_p1 * *s02_p1) * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2;
			sum01 += *s01_p2;
			sum02 += *s02_p2;
			sum03 += *s03_p2;
			sum04 += *s00_p2 * *s00_p2;
			sum05 += *s00_p2 * *s01_p2;
			sum06 += *s00_p2 * *s02_p2;
			sum07 += *s01_p2 * *s01_p2;
			sum08 += *s01_p2 * *s02_p2;
			sum09 += *s02_p2 * *s02_p2;
			sum10 += *s03_p2 * *s00_p2;
			s00_p2++;
			sum11 += *s03_p2 * *s01_p2;
			s01_p2++;
			sum12 += *s03_p2 * *s02_p2;
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
		*d09_p = sum09;
		d09_p++;
		*d10_p = sum10;
		d10_p++;
		*d11_p = sum11;
		d11_p++;
		*d12_p = sum12;
		d12_p++;

		for (int i = 1; i <= r; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p2++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p2++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p2++;
			s03_p2++;
			*d12_p = sum12;
			d12_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			s00_p2++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			s01_p2++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s02_p2++;
			s03_p1++;
			s03_p2++;
			*d12_p = sum12;
			d12_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum00 += *s00_p2 - *s00_p1;
			*d00_p = sum00;
			d00_p++;
			sum01 += *s01_p2 - *s01_p1;
			*d01_p = sum01;
			d01_p++;
			sum02 += *s02_p2 - *s02_p1;
			*d02_p = sum02;
			d02_p++;
			sum03 += *s03_p2 - *s03_p1;
			*d03_p = sum03;
			d03_p++;
			sum04 += (*s00_p2 * *s00_p2) - (*s00_p1 * *s00_p1);
			*d04_p = sum04;
			d04_p++;
			sum05 += (*s00_p2 * *s01_p2) - (*s00_p1 * *s01_p1);
			*d05_p = sum05;
			d05_p++;
			sum06 += (*s00_p2 * *s02_p2) - (*s00_p1 * *s02_p1);
			*d06_p = sum06;
			d06_p++;
			sum07 += (*s01_p2 * *s01_p2) - (*s01_p1 * *s01_p1);
			*d07_p = sum07;
			d07_p++;
			sum08 += (*s01_p2 * *s02_p2) - (*s01_p1 * *s02_p1);
			*d08_p = sum08;
			d08_p++;
			sum09 += (*s02_p2 * *s02_p2) - (*s02_p1 * *s02_p1);
			*d09_p = sum09;
			d09_p++;
			sum10 += (*s03_p2 * *s00_p2) - (*s03_p1 * *s00_p1);
			s00_p1++;
			*d10_p = sum10;
			d10_p++;
			sum11 += (*s03_p2 * *s01_p2) - (*s03_p1 * *s01_p1);
			s01_p1++;
			*d11_p = sum11;
			d11_p++;
			sum12 += (*s03_p2 * *s02_p2) - (*s03_p1 * *s02_p1);
			s02_p1++;
			s03_p1++;
			*d12_p = sum12;
			d12_p++;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_nonVec::filter_naive_impl()
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

		float c0 = gg * rr - gr * gr;
		float c1 = br * gr - bg * rr;
		float c2 = bg * gr - br * gg;
		float c4 = bb * rr - br * br;
		float c5 = br * bg - bb * gr;
		float c8 = bb * gg - bg * bg;

		*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
		*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
		*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_nonVec::filter_omp_impl()
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

		float c0 = gg * rr - gr * gr;
		float c1 = br * gr - bg * rr;
		float c2 = bg * gr - br * gg;
		float c4 = bb * rr - br * br;
		float c5 = br * bg - bb * gr;
		float c8 = bb * gg - bg * bg;

		*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
		*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
		*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_nonVec::operator()(const cv::Range& range) const
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

		float c0 = gg * rr - gr * gr;
		float c1 = br * gr - bg * rr;
		float c2 = bg * gr - br * gg;
		float c4 = bb * rr - br * br;
		float c5 = br * bg - bb * gr;
		float c8 = bb * gg - bg * bg;

		*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
		*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
		*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
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

			float c0 = gg * rr - gr * gr;
			float c1 = br * gr - bg * rr;
			float c2 = bg * gr - br * gg;
			float c4 = bb * rr - br * br;
			float c5 = br * bg - bb * gr;
			float c8 = bb * gg - bg * bg;

			*a_b_p = id * (covb*c0 + covg * c1 + covr * c2);
			*a_g_p = id * (covb*c1 + covg * c4 + covr * c5);
			*a_r_p = id * (covb*c2 + covg * c5 + covr * c8);
			*b_p = tmp03 - (*a_b_p * tmp00 + *a_g_p * tmp01 + *a_r_p * tmp02);
			a_b_p += step;
			a_g_p += step;
			a_r_p += step;
			b_p += step;
		}
	}
}



void ColumnSumFilter_Ip2ab_Guide3_SSE::filter_naive_impl()
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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

void ColumnSumFilter_Ip2ab_Guide3_SSE::filter_omp_impl()
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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

void ColumnSumFilter_Ip2ab_Guide3_SSE::operator()(const cv::Range& range) const
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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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

			mTmp00 = _mm_mul_ps(mSum00, mDiv);
			mTmp01 = _mm_mul_ps(mSum01, mDiv);
			mTmp02 = _mm_mul_ps(mSum02, mDiv);
			mTmp03 = _mm_mul_ps(mSum03, mDiv);
			mTmp04 = _mm_mul_ps(mSum04, mDiv);
			mTmp05 = _mm_mul_ps(mSum05, mDiv);
			mTmp06 = _mm_mul_ps(mSum06, mDiv);
			mTmp07 = _mm_mul_ps(mSum07, mDiv);
			mTmp08 = _mm_mul_ps(mSum08, mDiv);
			mTmp09 = _mm_mul_ps(mSum09, mDiv);
			mTmp10 = _mm_mul_ps(mSum10, mDiv);
			mTmp11 = _mm_mul_ps(mSum11, mDiv);
			mTmp12 = _mm_mul_ps(mSum12, mDiv);

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



void ColumnSumFilter_Ip2ab_Guide3_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 8;
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 8;
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 8;
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 8;

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

		mSum00 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s00_p1), _mm256_loadu_ps(s00_p2));
		mSum01 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s01_p1), _mm256_loadu_ps(s01_p2));
		mSum02 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s02_p1), _mm256_loadu_ps(s02_p2));
		mSum03 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s03_p1), _mm256_loadu_ps(s03_p2));
		mSum04 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s04_p1), _mm256_loadu_ps(s04_p2));
		mSum05 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s05_p1), _mm256_loadu_ps(s05_p2));
		mSum06 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s06_p1), _mm256_loadu_ps(s06_p2));
		mSum07 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s07_p1), _mm256_loadu_ps(s07_p2));
		mSum08 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s08_p1), _mm256_loadu_ps(s08_p2));
		mSum09 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s09_p1), _mm256_loadu_ps(s09_p2));
		mSum10 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s10_p1), _mm256_loadu_ps(s10_p2));
		mSum11 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s11_p1), _mm256_loadu_ps(s11_p2));
		mSum12 = _mm256_fmadd_ps(mBorder, _mm256_load_ps(s12_p1), _mm256_loadu_ps(s12_p2));
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

		for (int j = 2; j <= r; j++)
		{
			mSum00 = _mm256_add_ps(mSum00, _mm256_loadu_ps(s00_p2));
			s00_p2 += step;
			//_mm_prefetch((char *)&s01_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			//_mm_prefetch((char *)&s02_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			//_mm_prefetch((char *)&s03_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			//_mm_prefetch((char *)&s04_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			//_mm_prefetch((char *)&s05_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			//_mm_prefetch((char *)&s06_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			//_mm_prefetch((char *)&s07_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			//_mm_prefetch((char *)&s08_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			//_mm_prefetch((char *)&s09_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			//_mm_prefetch((char *)&s10_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			//_mm_prefetch((char *)&s11_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			//_mm_prefetch((char *)&s12_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			//_mm_prefetch((char *)&s00_p[(j+0)*step + i * 8], _MM_HINT_NTA);
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);
		mTmp09 = _mm256_mul_ps(mSum09, mDiv);
		mTmp10 = _mm256_mul_ps(mSum10, mDiv);
		mTmp11 = _mm256_mul_ps(mSum11, mDiv);
		mTmp12 = _mm256_mul_ps(mSum12, mDiv);
		/*
		mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));	// bb
		mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));	// bg
		mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));	// br
		mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));	// gg
		mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));	// gr
		mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));	// rr
		mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));
		*/
		mVar00 = _mm256_fnmadd_ps(mTmp00, mTmp00, mTmp04);	// bb
		mVar01 = _mm256_fnmadd_ps(mTmp00, mTmp01, mTmp05);	// bg
		mVar02 = _mm256_fnmadd_ps(mTmp00, mTmp02, mTmp06);	// br
		mVar03 = _mm256_fnmadd_ps(mTmp01, mTmp01, mTmp07);	// gg
		mVar04 = _mm256_fnmadd_ps(mTmp01, mTmp02, mTmp08);	// gr
		mVar05 = _mm256_fnmadd_ps(mTmp02, mTmp02, mTmp09);	// rr
		mCov00 = _mm256_fnmadd_ps(mTmp00, mTmp03, mTmp10);
		mCov01 = _mm256_fnmadd_ps(mTmp01, mTmp03, mTmp11);
		mCov02 = _mm256_fnmadd_ps(mTmp02, mTmp03, mTmp12);

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

		/*
		mDet = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));// *bg * *gr * *br
		mDet = _mm256_add_ps(mDet, mDet);
		mDet = _mm256_fmadd_ps(mVar00, _mm256_mul_ps(mVar03, mVar05), mDet);// *bb * *gg * *rr
		mDet = _mm256_fmadd_ps(mVar01, _mm256_mul_ps(mVar02, mVar04), mDet);// *bg * *gr * *br
		mDet = _mm256_fmsub_ps(mVar00, _mm256_mul_ps(mVar04, mVar04), mDet);// *bb * *gr * *gr
		mDet = _mm256_fmsub_ps(mVar03, _mm256_mul_ps(mVar02, mVar02), mDet);// *bg * *bg * *rr
		mDet = _mm256_fmsub_ps(mVar05, _mm256_mul_ps(mVar01, mVar01), mDet);// *br * *gg * *br
		*/
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

		/*
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;


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

		mTmp03 = _mm256_fnmadd_ps(mTmp10, mTmp00, mTmp03);
		mTmp03 = _mm256_fnmadd_ps(mTmp11, mTmp01, mTmp03);
		mTmp03 = _mm256_fnmadd_ps(mTmp12, mTmp02, mTmp03);
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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

			mVar00 = _mm256_fnmadd_ps(mTmp00, mTmp00, mTmp04);
			mVar01 = _mm256_fnmadd_ps(mTmp00, mTmp01, mTmp05);
			mVar02 = _mm256_fnmadd_ps(mTmp00, mTmp02, mTmp06);
			mVar03 = _mm256_fnmadd_ps(mTmp01, mTmp01, mTmp07);
			mVar04 = _mm256_fnmadd_ps(mTmp01, mTmp02, mTmp08);
			mVar05 = _mm256_fnmadd_ps(mTmp02, mTmp02, mTmp09);
			mCov00 = _mm256_fnmadd_ps(mTmp00, mTmp03, mTmp10);
			mCov01 = _mm256_fnmadd_ps(mTmp01, mTmp03, mTmp11);
			mCov02 = _mm256_fnmadd_ps(mTmp02, mTmp03, mTmp12);

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

			/*
			mDet = _mm256_mul_ps(mVar01, _mm256_mul_ps(mVar02, mVar04));// *bg * *gr * *br
			mDet = _mm256_add_ps(mDet, mDet);
			mDet = _mm256_fmadd_ps(mVar00, _mm256_mul_ps(mVar03, mVar05), mDet);// *bb * *gg * *rr
			mDet = _mm256_fmadd_ps(mVar01, _mm256_mul_ps(mVar02, mVar04), mDet);// *bg * *gr * *br
			mDet = _mm256_fmsub_ps(mVar00, _mm256_mul_ps(mVar04, mVar04), mDet);// *bb * *gr * *gr
			mDet = _mm256_fmsub_ps(mVar03, _mm256_mul_ps(mVar02, mVar02), mDet);// *bg * *bg * *rr
			mDet = _mm256_fmsub_ps(mVar05, _mm256_mul_ps(mVar01, mVar01), mDet);// *br * *gg * *br
			*/
			mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);


			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

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

			mTmp03 = _mm256_fnmadd_ps(mTmp10, mTmp00, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp11, mTmp01, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp12, mTmp02, mTmp03);
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			s09_p1 += step;
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			s10_p1 += step;
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			s11_p1 += step;
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

			mVar00 = _mm256_fnmadd_ps(mTmp00, mTmp00, mTmp04);
			mVar01 = _mm256_fnmadd_ps(mTmp00, mTmp01, mTmp05);
			mVar02 = _mm256_fnmadd_ps(mTmp00, mTmp02, mTmp06);
			mVar03 = _mm256_fnmadd_ps(mTmp01, mTmp01, mTmp07);
			mVar04 = _mm256_fnmadd_ps(mTmp01, mTmp02, mTmp08);
			mVar05 = _mm256_fnmadd_ps(mTmp02, mTmp02, mTmp09);
			mCov00 = _mm256_fnmadd_ps(mTmp00, mTmp03, mTmp10);
			mCov01 = _mm256_fnmadd_ps(mTmp01, mTmp03, mTmp11);
			mCov02 = _mm256_fnmadd_ps(mTmp02, mTmp03, mTmp12);

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

			mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); //c0
			mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); //c1
			mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); //c2
			mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); //c4
			mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); //c5
			mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); //c8

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

			mTmp03 = _mm256_fnmadd_ps(mTmp10, mTmp00, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp11, mTmp01, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp12, mTmp02, mTmp03);
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			s09_p1 += step;
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			s10_p1 += step;
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			s11_p1 += step;
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

			mVar00 = _mm256_fnmadd_ps(mTmp00, mTmp00, mTmp04);
			mVar01 = _mm256_fnmadd_ps(mTmp00, mTmp01, mTmp05);
			mVar02 = _mm256_fnmadd_ps(mTmp00, mTmp02, mTmp06);
			mVar03 = _mm256_fnmadd_ps(mTmp01, mTmp01, mTmp07);
			mVar04 = _mm256_fnmadd_ps(mTmp01, mTmp02, mTmp08);
			mVar05 = _mm256_fnmadd_ps(mTmp02, mTmp02, mTmp09);
			mCov00 = _mm256_fnmadd_ps(mTmp00, mTmp03, mTmp10);
			mCov01 = _mm256_fnmadd_ps(mTmp01, mTmp03, mTmp11);
			mCov02 = _mm256_fnmadd_ps(mTmp02, mTmp03, mTmp12);

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

			mTmp03 = _mm256_fnmadd_ps(mTmp10, mTmp00, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp11, mTmp01, mTmp03);
			mTmp03 = _mm256_fnmadd_ps(mTmp12, mTmp02, mTmp03);
			_mm256_storeu_ps(b_p, mTmp03);
			b_p += step;
		}
	}
}

void ColumnSumFilter_Ip2ab_Guide3_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 8;
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 8;
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 8;
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 8;

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
			s00_p2 += step;
			//_mm_prefetch((char *)&s01_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			//_mm_prefetch((char *)&s02_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			//_mm_prefetch((char *)&s03_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			//_mm_prefetch((char *)&s04_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			//_mm_prefetch((char *)&s05_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			//_mm_prefetch((char *)&s06_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			//_mm_prefetch((char *)&s07_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			//_mm_prefetch((char *)&s08_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			//_mm_prefetch((char *)&s09_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			//_mm_prefetch((char *)&s10_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			//_mm_prefetch((char *)&s11_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			//_mm_prefetch((char *)&s12_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			//_mm_prefetch((char *)&s00_p[(j+0)*step + i * 8], _MM_HINT_NTA);
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);
		mTmp09 = _mm256_mul_ps(mSum09, mDiv);
		mTmp10 = _mm256_mul_ps(mSum10, mDiv);
		mTmp11 = _mm256_mul_ps(mSum11, mDiv);
		mTmp12 = _mm256_mul_ps(mSum12, mDiv);

		mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));	// bb
		mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));	// bg
		mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));	// br
		mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));	// gg
		mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));	// gr
		mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));	// rr
		mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

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
		mDet = _mm256_sub_ps(mDet, mTmp08);
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

		/*
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;


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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			s09_p1 += step;
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			s10_p1 += step;
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			s11_p1 += step;
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			s09_p1 += step;
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			s10_p1 += step;
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			s11_p1 += step;
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

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

void ColumnSumFilter_Ip2ab_Guide3_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s00_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* s01_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* s02_p1 = tempVec[2].ptr<float>(0) + i * 8;
		float* s03_p1 = tempVec[3].ptr<float>(0) + i * 8;
		float* s04_p1 = tempVec[4].ptr<float>(0) + i * 8;
		float* s05_p1 = tempVec[5].ptr<float>(0) + i * 8;
		float* s06_p1 = tempVec[6].ptr<float>(0) + i * 8;
		float* s07_p1 = tempVec[7].ptr<float>(0) + i * 8;
		float* s08_p1 = tempVec[8].ptr<float>(0) + i * 8;
		float* s09_p1 = tempVec[9].ptr<float>(0) + i * 8;
		float* s10_p1 = tempVec[10].ptr<float>(0) + i * 8;
		float* s11_p1 = tempVec[11].ptr<float>(0) + i * 8;
		float* s12_p1 = tempVec[12].ptr<float>(0) + i * 8;

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
			s00_p2 += step;
			//_mm_prefetch((char *)&s01_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(s01_p2));
			s01_p2 += step;
			//_mm_prefetch((char *)&s02_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(s02_p2));
			s02_p2 += step;
			//_mm_prefetch((char *)&s03_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(s03_p2));
			s03_p2 += step;
			//_mm_prefetch((char *)&s04_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum04 = _mm256_add_ps(mSum04, _mm256_loadu_ps(s04_p2));
			s04_p2 += step;
			//_mm_prefetch((char *)&s05_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum05 = _mm256_add_ps(mSum05, _mm256_loadu_ps(s05_p2));
			s05_p2 += step;
			//_mm_prefetch((char *)&s06_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum06 = _mm256_add_ps(mSum06, _mm256_loadu_ps(s06_p2));
			s06_p2 += step;
			//_mm_prefetch((char *)&s07_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum07 = _mm256_add_ps(mSum07, _mm256_loadu_ps(s07_p2));
			s07_p2 += step;
			//_mm_prefetch((char *)&s08_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum08 = _mm256_add_ps(mSum08, _mm256_loadu_ps(s08_p2));
			s08_p2 += step;
			//_mm_prefetch((char *)&s09_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			//_mm_prefetch((char *)&s10_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			//_mm_prefetch((char *)&s11_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			//_mm_prefetch((char *)&s12_p[(j+0)*step + i * 8], _MM_HINT_NTA);
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			//_mm_prefetch((char *)&s00_p[(j+0)*step + i * 8], _MM_HINT_NTA);
		}
		mTmp00 = _mm256_mul_ps(mSum00, mDiv);
		mTmp01 = _mm256_mul_ps(mSum01, mDiv);
		mTmp02 = _mm256_mul_ps(mSum02, mDiv);
		mTmp03 = _mm256_mul_ps(mSum03, mDiv);
		mTmp04 = _mm256_mul_ps(mSum04, mDiv);
		mTmp05 = _mm256_mul_ps(mSum05, mDiv);
		mTmp06 = _mm256_mul_ps(mSum06, mDiv);
		mTmp07 = _mm256_mul_ps(mSum07, mDiv);
		mTmp08 = _mm256_mul_ps(mSum08, mDiv);
		mTmp09 = _mm256_mul_ps(mSum09, mDiv);
		mTmp10 = _mm256_mul_ps(mSum10, mDiv);
		mTmp11 = _mm256_mul_ps(mSum11, mDiv);
		mTmp12 = _mm256_mul_ps(mSum12, mDiv);

		mVar00 = _mm256_sub_ps(mTmp04, _mm256_mul_ps(mTmp00, mTmp00));	// bb
		mVar01 = _mm256_sub_ps(mTmp05, _mm256_mul_ps(mTmp00, mTmp01));	// bg
		mVar02 = _mm256_sub_ps(mTmp06, _mm256_mul_ps(mTmp00, mTmp02));	// br
		mVar03 = _mm256_sub_ps(mTmp07, _mm256_mul_ps(mTmp01, mTmp01));	// gg
		mVar04 = _mm256_sub_ps(mTmp08, _mm256_mul_ps(mTmp01, mTmp02));	// gr
		mVar05 = _mm256_sub_ps(mTmp09, _mm256_mul_ps(mTmp02, mTmp02));	// rr
		mCov00 = _mm256_sub_ps(mTmp10, _mm256_mul_ps(mTmp00, mTmp03));
		mCov01 = _mm256_sub_ps(mTmp11, _mm256_mul_ps(mTmp01, mTmp03));
		mCov02 = _mm256_sub_ps(mTmp12, _mm256_mul_ps(mTmp02, mTmp03));

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
		mDet = _mm256_sub_ps(mDet, mTmp08);
		mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

		/*
		mTmp04 = _mm256_sub_ps(_mm256_mul_ps(mVar03, mVar05), _mm256_mul_ps(mVar04, mVar04)); //c0
		mTmp05 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar04), _mm256_mul_ps(mVar01, mVar05)); //c1
		mTmp06 = _mm256_sub_ps(_mm256_mul_ps(mVar01, mVar04), _mm256_mul_ps(mVar02, mVar03)); //c2
		mTmp07 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar05), _mm256_mul_ps(mVar02, mVar02)); //c4
		mTmp08 = _mm256_sub_ps(_mm256_mul_ps(mVar02, mVar01), _mm256_mul_ps(mVar00, mVar04)); //c5
		mTmp09 = _mm256_sub_ps(_mm256_mul_ps(mVar00, mVar03), _mm256_mul_ps(mVar01, mVar01)); //c8
		*/
		mTmp04 = _mm256_fmsub_ps(mVar03, mVar05, _mm256_mul_ps(mVar04, mVar04)); // c0 = *gg * *rr - *gr * *gr;
		mTmp05 = _mm256_fmsub_ps(mVar02, mVar04, _mm256_mul_ps(mVar01, mVar05)); // c1 = *gr * *br - *bg * *rr;
		mTmp06 = _mm256_fmsub_ps(mVar01, mVar04, _mm256_mul_ps(mVar02, mVar03)); // c2 = *bg * *gr - *br * *gg;
		mTmp07 = _mm256_fmsub_ps(mVar00, mVar05, _mm256_mul_ps(mVar02, mVar02)); // c4 = *bb * *rr - *br * *br;
		mTmp08 = _mm256_fmsub_ps(mVar02, mVar01, _mm256_mul_ps(mVar00, mVar04)); // c5 = *bg * *br - *bb * *gr;
		mTmp09 = _mm256_fmsub_ps(mVar00, mVar03, _mm256_mul_ps(mVar01, mVar01)); // c8 = *bb * *gg - *bg * *bg;


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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			s09_p2 += step;
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			s09_p1 += step;
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			s10_p2 += step;
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			s10_p1 += step;
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			s11_p2 += step;
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			s11_p1 += step;
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			s12_p2 += step;
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

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
			mSum09 = _mm256_add_ps(mSum09, _mm256_loadu_ps(s09_p2));
			mSum09 = _mm256_sub_ps(mSum09, _mm256_load_ps(s09_p1));
			s09_p1 += step;
			mSum10 = _mm256_add_ps(mSum10, _mm256_loadu_ps(s10_p2));
			mSum10 = _mm256_sub_ps(mSum10, _mm256_load_ps(s10_p1));
			s10_p1 += step;
			mSum11 = _mm256_add_ps(mSum11, _mm256_loadu_ps(s11_p2));
			mSum11 = _mm256_sub_ps(mSum11, _mm256_load_ps(s11_p1));
			s11_p1 += step;
			mSum12 = _mm256_add_ps(mSum12, _mm256_loadu_ps(s12_p2));
			mSum12 = _mm256_sub_ps(mSum12, _mm256_load_ps(s12_p1));
			s12_p1 += step;

			mTmp00 = _mm256_mul_ps(mSum00, mDiv);
			mTmp01 = _mm256_mul_ps(mSum01, mDiv);
			mTmp02 = _mm256_mul_ps(mSum02, mDiv);
			mTmp03 = _mm256_mul_ps(mSum03, mDiv);
			mTmp04 = _mm256_mul_ps(mSum04, mDiv);
			mTmp05 = _mm256_mul_ps(mSum05, mDiv);
			mTmp06 = _mm256_mul_ps(mSum06, mDiv);
			mTmp07 = _mm256_mul_ps(mSum07, mDiv);
			mTmp08 = _mm256_mul_ps(mSum08, mDiv);
			mTmp09 = _mm256_mul_ps(mSum09, mDiv);
			mTmp10 = _mm256_mul_ps(mSum10, mDiv);
			mTmp11 = _mm256_mul_ps(mSum11, mDiv);
			mTmp12 = _mm256_mul_ps(mSum12, mDiv);

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



void RowSumFilter_ab2q_Guide3::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* s0_p1 = va[0].ptr<float>(j);
		float* s1_p1 = va[1].ptr<float>(j);
		float* s2_p1 = va[2].ptr<float>(j);
		float* s3_p1 = b.ptr<float>(j);
		float* s0_p2 = va[0].ptr<float>(j) + 1;
		float* s1_p2 = va[1].ptr<float>(j) + 1;
		float* s2_p2 = va[2].ptr<float>(j) + 1;
		float* s3_p2 = b.ptr<float>(j) + 1;

		float* d0_p = tempVec[0].ptr<float>(j);
		float* d1_p = tempVec[1].ptr<float>(j);
		float* d2_p = tempVec[2].ptr<float>(j);
		float* d3_p = tempVec[3].ptr<float>(j);

		float sum0, sum1, sum2, sum3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		sum0 += *s0_p1 * (r + 1);
		sum1 += *s1_p1 * (r + 1);
		sum2 += *s2_p1 * (r + 1);
		sum3 += *s3_p1 * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *s0_p2;
			s0_p2++;
			sum1 += *s1_p2;
			s1_p2++;
			sum2 += *s2_p2;
			s2_p2++;
			sum3 += *s3_p2;
			s3_p2++;
		}
		*d0_p = sum0;
		d0_p++;
		*d1_p = sum1;
		d1_p++;
		*d2_p = sum2;
		d2_p++;
		*d3_p = sum3;
		d3_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p2++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p2++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p2++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p2++;
			*d3_p = sum3;
			d3_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1++;
			s0_p2++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1++;
			s1_p2++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1++;
			s2_p2++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1++;
			s3_p2++;
			*d3_p = sum3;
			d3_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1++;
			*d3_p = sum3;
			d3_p++;
		}
	}
}

void RowSumFilter_ab2q_Guide3::filter_omp_impl()
{
#pragma omp parallel for
	for (int j = 0; j < img_row; j++)
	{
		float* s0_p1 = va[0].ptr<float>(j);
		float* s1_p1 = va[1].ptr<float>(j);
		float* s2_p1 = va[2].ptr<float>(j);
		float* s3_p1 = b.ptr<float>(j);
		float* s0_p2 = va[0].ptr<float>(j) + 1;
		float* s1_p2 = va[1].ptr<float>(j) + 1;
		float* s2_p2 = va[2].ptr<float>(j) + 1;
		float* s3_p2 = b.ptr<float>(j) + 1;

		float* d0_p = tempVec[0].ptr<float>(j);
		float* d1_p = tempVec[1].ptr<float>(j);
		float* d2_p = tempVec[2].ptr<float>(j);
		float* d3_p = tempVec[3].ptr<float>(j);

		float sum0, sum1, sum2, sum3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		sum0 += *s0_p1 * (r + 1);
		sum1 += *s1_p1 * (r + 1);
		sum2 += *s2_p1 * (r + 1);
		sum3 += *s3_p1 * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *s0_p2;
			s0_p2++;
			sum1 += *s1_p2;
			s1_p2++;
			sum2 += *s2_p2;
			s2_p2++;
			sum3 += *s3_p2;
			s3_p2++;
		}
		*d0_p = sum0;
		d0_p++;
		*d1_p = sum1;
		d1_p++;
		*d2_p = sum2;
		d2_p++;
		*d3_p = sum3;
		d3_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p2++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p2++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p2++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p2++;
			*d3_p = sum3;
			d3_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1++;
			s0_p2++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1++;
			s1_p2++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1++;
			s2_p2++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1++;
			s3_p2++;
			*d3_p = sum3;
			d3_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1++;
			*d3_p = sum3;
			d3_p++;
		}
	}
}

void RowSumFilter_ab2q_Guide3::operator()(const cv::Range& range) const
{
	for (int j = range.start; j < range.end; j++)
	{
		float* s0_p1 = va[0].ptr<float>(j);
		float* s1_p1 = va[1].ptr<float>(j);
		float* s2_p1 = va[2].ptr<float>(j);
		float* s3_p1 = b.ptr<float>(j);
		float* s0_p2 = va[0].ptr<float>(j) + 1;
		float* s1_p2 = va[1].ptr<float>(j) + 1;
		float* s2_p2 = va[2].ptr<float>(j) + 1;
		float* s3_p2 = b.ptr<float>(j) + 1;

		float* d0_p = tempVec[0].ptr<float>(j);
		float* d1_p = tempVec[1].ptr<float>(j);
		float* d2_p = tempVec[2].ptr<float>(j);
		float* d3_p = tempVec[3].ptr<float>(j);

		float sum0, sum1, sum2, sum3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		sum0 += *s0_p1 * (r + 1);
		sum1 += *s1_p1 * (r + 1);
		sum2 += *s2_p1 * (r + 1);
		sum3 += *s3_p1 * (r + 1);
		for (int i = 1; i <= r; i++)
		{
			sum0 += *s0_p2;
			s0_p2++;
			sum1 += *s1_p2;
			s1_p2++;
			sum2 += *s2_p2;
			s2_p2++;
			sum3 += *s3_p2;
			s3_p2++;
		}
		*d0_p = sum0;
		d0_p++;
		*d1_p = sum1;
		d1_p++;
		*d2_p = sum2;
		d2_p++;
		*d3_p = sum3;
		d3_p++;

		for (int i = 1; i <= r; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p2++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p2++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p2++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p2++;
			*d3_p = sum3;
			d3_p++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1++;
			s0_p2++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1++;
			s1_p2++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1++;
			s2_p2++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1++;
			s3_p2++;
			*d3_p = sum3;
			d3_p++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1++;
			*d0_p = sum0;
			d0_p++;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1++;
			*d1_p = sum1;
			d1_p++;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1++;
			*d2_p = sum2;
			d2_p++;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1++;
			*d3_p = sum3;
			d3_p++;
		}
	}
}



void ColumnSumFilter_ab2q_Guide3_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i;

		float* I_b_p = vI[0].ptr<float>(0) + i;
		float* I_g_p = vI[1].ptr<float>(0) + i;
		float* I_r_p = vI[2].ptr<float>(0) + i;

		float* q_p = q.ptr<float>(0) + i;

		float sum0, sum1, sum2, sum3;
		float tmp0, tmp1, tmp2, tmp3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		sum0 += *s0_p1 * (r + 1);
		sum1 += *s1_p1 * (r + 1);
		sum2 += *s2_p1 * (r + 1);
		sum3 += *s3_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s0_p2;
			s0_p2 += step;
			sum1 += *s1_p2;
			s1_p2 += step;
			sum2 += *s2_p2;
			s2_p2 += step;
			sum3 += *s3_p2;
			s3_p2 += step;
		}
		tmp0 = sum0 * div;
		tmp1 = sum1 * div;
		tmp2 = sum2 * div;
		tmp3 = sum3 * div;
		*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;
		I_b_p += step;
		I_g_p += step;
		I_r_p += step;
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p2 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p2 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p2 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p2 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1 += step;
			s0_p2 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1 += step;
			s1_p2 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1 += step;
			s2_p2 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1 += step;
			s3_p2 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide3_nonVec::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i;

		float* I_b_p = vI[0].ptr<float>(0) + i;
		float* I_g_p = vI[1].ptr<float>(0) + i;
		float* I_r_p = vI[2].ptr<float>(0) + i;

		float* q_p = q.ptr<float>(0) + i;

		float sum0, sum1, sum2, sum3;
		float tmp0, tmp1, tmp2, tmp3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		sum0 += *s0_p1 * (r + 1);
		sum1 += *s1_p1 * (r + 1);
		sum2 += *s2_p1 * (r + 1);
		sum3 += *s3_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s0_p2;
			s0_p2 += step;
			sum1 += *s1_p2;
			s1_p2 += step;
			sum2 += *s2_p2;
			s2_p2 += step;
			sum3 += *s3_p2;
			s3_p2 += step;
		}
		tmp0 = sum0 * div;
		tmp1 = sum1 * div;
		tmp2 = sum2 * div;
		tmp3 = sum3 * div;
		*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;
		I_b_p += step;
		I_g_p += step;
		I_r_p += step;
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p2 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p2 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p2 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p2 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1 += step;
			s0_p2 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1 += step;
			s1_p2 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1 += step;
			s2_p2 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1 += step;
			s3_p2 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide3_nonVec::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i;

		float* I_b_p = vI[0].ptr<float>(0) + i;
		float* I_g_p = vI[1].ptr<float>(0) + i;
		float* I_r_p = vI[2].ptr<float>(0) + i;

		float* q_p = q.ptr<float>(0) + i;

		float sum0, sum1, sum2, sum3;
		float tmp0, tmp1, tmp2, tmp3;
		sum0 = sum1 = sum2 = sum3 = 0.f;

		sum0 += *s0_p1 * (r + 1);
		sum1 += *s1_p1 * (r + 1);
		sum2 += *s2_p1 * (r + 1);
		sum3 += *s3_p1 * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum0 += *s0_p2;
			s0_p2 += step;
			sum1 += *s1_p2;
			s1_p2 += step;
			sum2 += *s2_p2;
			s2_p2 += step;
			sum3 += *s3_p2;
			s3_p2 += step;
		}
		tmp0 = sum0 * div;
		tmp1 = sum1 * div;
		tmp2 = sum2 * div;
		tmp3 = sum3 * div;
		*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;
		I_b_p += step;
		I_g_p += step;
		I_r_p += step;
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p2 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p2 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p2 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p2 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1 += step;
			s0_p2 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1 += step;
			s1_p2 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1 += step;
			s2_p2 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1 += step;
			s3_p2 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum0 += *s0_p2 - *s0_p1;
			s0_p1 += step;
			sum1 += *s1_p2 - *s1_p1;
			s1_p1 += step;
			sum2 += *s2_p2 - *s2_p1;
			s2_p1 += step;
			sum3 += *s3_p2 - *s3_p1;
			s3_p1 += step;

			tmp0 = sum0 * div;
			tmp1 = sum1 * div;
			tmp2 = sum2 * div;
			tmp3 = sum3 * div;
			*q_p = (tmp0 * *I_b_p + tmp1 * *I_g_p + tmp2 * *I_r_p) + tmp3;

			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			q_p += step;
		}
	}
}



void ColumnSumFilter_ab2q_Guide3_SSE::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i * 4;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i * 4;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i * 4;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i * 4;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i * 4;

		float* I_b_p = vI[0].ptr<float>(0) + i * 4;
		float* I_g_p = vI[1].ptr<float>(0) + i * 4;
		float* I_r_p = vI[2].ptr<float>(0) + i * 4;

		float* q_p = q.ptr<float>(0) + i * 4;

		__m128 mSum0, mSum1, mSum2, mSum3;
		__m128 mTmp0, mTmp1, mTmp2, mTmp3;

		mSum0 = _mm_setzero_ps();
		mSum1 = _mm_setzero_ps();
		mSum2 = _mm_setzero_ps();
		mSum3 = _mm_setzero_ps();

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(s0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(s1_p1));
		mSum2 = _mm_mul_ps(mBorder, _mm_load_ps(s2_p1));
		mSum3 = _mm_mul_ps(mBorder, _mm_load_ps(s3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;
		}
		mTmp0 = _mm_mul_ps(mSum0, _mm_load_ps(I_b_p));
		mTmp1 = _mm_mul_ps(mSum1, _mm_load_ps(I_g_p));
		mTmp2 = _mm_mul_ps(mSum2, _mm_load_ps(I_r_p));
		I_b_p += step;
		I_g_p += step;
		I_r_p += step;
		mTmp3 = _mm_add_ps(mTmp0, mTmp1);
		mTmp3 = _mm_add_ps(mTmp3, mTmp2);
		mTmp3 = _mm_add_ps(mTmp3, mSum3);
		_mm_store_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p1 += step;
			s1_p1 += step;
			s2_p1 += step;
			s3_p1 += step;
			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p1 += step;
			s1_p1 += step;
			s2_p1 += step;
			s3_p1 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide3_SSE::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i * 4;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i * 4;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i * 4;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i * 4;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i * 4;

		float* I_b_p = vI[0].ptr<float>(0) + i * 4;
		float* I_g_p = vI[1].ptr<float>(0) + i * 4;
		float* I_r_p = vI[2].ptr<float>(0) + i * 4;

		float* q_p = q.ptr<float>(0) + i * 4;

		__m128 mSum0, mSum1, mSum2, mSum3;
		__m128 mTmp0, mTmp1, mTmp2, mTmp3;

		mSum0 = _mm_setzero_ps();
		mSum1 = _mm_setzero_ps();
		mSum2 = _mm_setzero_ps();
		mSum3 = _mm_setzero_ps();

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(s0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(s1_p1));
		mSum2 = _mm_mul_ps(mBorder, _mm_load_ps(s2_p1));
		mSum3 = _mm_mul_ps(mBorder, _mm_load_ps(s3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;
		}
		mTmp0 = _mm_mul_ps(mSum0, _mm_load_ps(I_b_p));
		mTmp1 = _mm_mul_ps(mSum1, _mm_load_ps(I_g_p));
		mTmp2 = _mm_mul_ps(mSum2, _mm_load_ps(I_r_p));
		I_b_p += step;
		I_g_p += step;
		I_r_p += step;
		mTmp3 = _mm_add_ps(mTmp0, mTmp1);
		mTmp3 = _mm_add_ps(mTmp3, mTmp2);
		mTmp3 = _mm_add_ps(mTmp3, mSum3);
		_mm_store_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p1 += step;
			s1_p1 += step;
			s2_p1 += step;
			s3_p1 += step;
			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p1 += step;
			s1_p1 += step;
			s2_p1 += step;
			s3_p1 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide3_SSE::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i * 4;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i * 4;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i * 4;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i * 4;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i * 4;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i * 4;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i * 4;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i * 4;

		float* I_b_p = vI[0].ptr<float>(0) + i * 4;
		float* I_g_p = vI[1].ptr<float>(0) + i * 4;
		float* I_r_p = vI[2].ptr<float>(0) + i * 4;

		float* q_p = q.ptr<float>(0) + i * 4;

		__m128 mSum0, mSum1, mSum2, mSum3;
		__m128 mTmp0, mTmp1, mTmp2, mTmp3;

		mSum0 = _mm_setzero_ps();
		mSum1 = _mm_setzero_ps();
		mSum2 = _mm_setzero_ps();
		mSum3 = _mm_setzero_ps();

		mSum0 = _mm_mul_ps(mBorder, _mm_load_ps(s0_p1));
		mSum1 = _mm_mul_ps(mBorder, _mm_load_ps(s1_p1));
		mSum2 = _mm_mul_ps(mBorder, _mm_load_ps(s2_p1));
		mSum3 = _mm_mul_ps(mBorder, _mm_load_ps(s3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;
		}
		mTmp0 = _mm_mul_ps(mSum0, _mm_load_ps(I_b_p));
		mTmp1 = _mm_mul_ps(mSum1, _mm_load_ps(I_g_p));
		mTmp2 = _mm_mul_ps(mSum2, _mm_load_ps(I_r_p));
		I_b_p += step;
		I_g_p += step;
		I_r_p += step;
		mTmp3 = _mm_add_ps(mTmp0, mTmp1);
		mTmp3 = _mm_add_ps(mTmp3, mTmp2);
		mTmp3 = _mm_add_ps(mTmp3, mSum3);
		_mm_store_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p1 += step;
			s1_p1 += step;
			s2_p1 += step;
			s3_p1 += step;
			s0_p2 += step;
			s1_p2 += step;
			s2_p2 += step;
			s3_p2 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm_add_ps(mSum0, _mm_loadu_ps(s0_p2));
			mSum0 = _mm_sub_ps(mSum0, _mm_load_ps(s0_p1));
			mSum1 = _mm_add_ps(mSum1, _mm_loadu_ps(s1_p2));
			mSum1 = _mm_sub_ps(mSum1, _mm_load_ps(s1_p1));
			mSum2 = _mm_add_ps(mSum2, _mm_loadu_ps(s2_p2));
			mSum2 = _mm_sub_ps(mSum2, _mm_load_ps(s2_p1));
			mSum3 = _mm_add_ps(mSum3, _mm_loadu_ps(s3_p2));
			mSum3 = _mm_sub_ps(mSum3, _mm_load_ps(s3_p1));

			s0_p1 += step;
			s1_p1 += step;
			s2_p1 += step;
			s3_p1 += step;

			mTmp0 = _mm_mul_ps(mSum0, _mm_loadu_ps(I_b_p));
			mTmp1 = _mm_mul_ps(mSum1, _mm_loadu_ps(I_g_p));
			mTmp2 = _mm_mul_ps(mSum2, _mm_loadu_ps(I_r_p));
			I_b_p += step;
			I_g_p += step;
			I_r_p += step;
			mTmp3 = _mm_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm_add_ps(mTmp3, mSum3);
			_mm_storeu_ps(q_p, _mm_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
	}
}



void ColumnSumFilter_ab2q_Guide3_AVX::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i * 8;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i * 8;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i * 8;

		float* I_b_p = vI[0].ptr<float>(0) + i * 8;
		float* I_g_p = vI[1].ptr<float>(0) + i * 8;
		float* I_r_p = vI[2].ptr<float>(0) + i * 8;

		float* q_p = q.ptr<float>(0) + i * 8;

		__m256 mSum0, mSum1, mSum2, mSum3;
		__m256 mTmp0, mTmp1, mTmp2, mTmp3;

		mSum0 = _mm256_setzero_ps();
		mSum1 = _mm256_setzero_ps();
		mSum2 = _mm256_setzero_ps();
		mSum3 = _mm256_setzero_ps();

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(s0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(s1_p1));
		mSum2 = _mm256_mul_ps(mBorder, _mm256_load_ps(s2_p1));
		mSum3 = _mm256_mul_ps(mBorder, _mm256_load_ps(s3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
		}
		mTmp0 = _mm256_mul_ps(mSum0, _mm256_load_ps(I_b_p));
		I_b_p += step;
		mTmp1 = _mm256_mul_ps(mSum1, _mm256_load_ps(I_g_p));
		I_g_p += step;
		mTmp2 = _mm256_mul_ps(mSum2, _mm256_load_ps(I_r_p));
		I_r_p += step;
		mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
		mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
		mTmp3 = _mm256_add_ps(mTmp3, mSum3);
		_mm256_store_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;

		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			s0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			s1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			s2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));
			s3_p1 += step;

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			s0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			s1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			s2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));
			s3_p1 += step;

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide3_AVX::filter_omp_impl()
{
#pragma omp parallel for
	for (int i = 0; i < img_col; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i * 8;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i * 8;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i * 8;

		float* I_b_p = vI[0].ptr<float>(0) + i * 8;
		float* I_g_p = vI[1].ptr<float>(0) + i * 8;
		float* I_r_p = vI[2].ptr<float>(0) + i * 8;

		float* q_p = q.ptr<float>(0) + i * 8;

		__m256 mSum0, mSum1, mSum2, mSum3;
		__m256 mTmp0, mTmp1, mTmp2, mTmp3;

		mSum0 = _mm256_setzero_ps();
		mSum1 = _mm256_setzero_ps();
		mSum2 = _mm256_setzero_ps();
		mSum3 = _mm256_setzero_ps();

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(s0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(s1_p1));
		mSum2 = _mm256_mul_ps(mBorder, _mm256_load_ps(s2_p1));
		mSum3 = _mm256_mul_ps(mBorder, _mm256_load_ps(s3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
		}
		mTmp0 = _mm256_mul_ps(mSum0, _mm256_load_ps(I_b_p));
		I_b_p += step;
		mTmp1 = _mm256_mul_ps(mSum1, _mm256_load_ps(I_g_p));
		I_g_p += step;
		mTmp2 = _mm256_mul_ps(mSum2, _mm256_load_ps(I_r_p));
		I_r_p += step;
		mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
		mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
		mTmp3 = _mm256_add_ps(mTmp3, mSum3);
		_mm256_store_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;

		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			s0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			s1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			s2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));
			s3_p1 += step;

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			s0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			s1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			s2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));
			s3_p1 += step;

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
	}
}

void ColumnSumFilter_ab2q_Guide3_AVX::operator()(const cv::Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		float* s0_p1 = tempVec[0].ptr<float>(0) + i * 8;
		float* s1_p1 = tempVec[1].ptr<float>(0) + i * 8;
		float* s2_p1 = tempVec[2].ptr<float>(0) + i * 8;
		float* s3_p1 = tempVec[3].ptr<float>(0) + i * 8;
		float* s0_p2 = tempVec[0].ptr<float>(1) + i * 8;
		float* s1_p2 = tempVec[1].ptr<float>(1) + i * 8;
		float* s2_p2 = tempVec[2].ptr<float>(1) + i * 8;
		float* s3_p2 = tempVec[3].ptr<float>(1) + i * 8;

		float* I_b_p = vI[0].ptr<float>(0) + i * 8;
		float* I_g_p = vI[1].ptr<float>(0) + i * 8;
		float* I_r_p = vI[2].ptr<float>(0) + i * 8;

		float* q_p = q.ptr<float>(0) + i * 8;

		__m256 mSum0, mSum1, mSum2, mSum3;
		__m256 mTmp0, mTmp1, mTmp2, mTmp3;

		mSum0 = _mm256_setzero_ps();
		mSum1 = _mm256_setzero_ps();
		mSum2 = _mm256_setzero_ps();
		mSum3 = _mm256_setzero_ps();

		mSum0 = _mm256_mul_ps(mBorder, _mm256_load_ps(s0_p1));
		mSum1 = _mm256_mul_ps(mBorder, _mm256_load_ps(s1_p1));
		mSum2 = _mm256_mul_ps(mBorder, _mm256_load_ps(s2_p1));
		mSum3 = _mm256_mul_ps(mBorder, _mm256_load_ps(s3_p1));
		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
		}
		mTmp0 = _mm256_mul_ps(mSum0, _mm256_load_ps(I_b_p));
		I_b_p += step;
		mTmp1 = _mm256_mul_ps(mSum1, _mm256_load_ps(I_g_p));
		I_g_p += step;
		mTmp2 = _mm256_mul_ps(mSum2, _mm256_load_ps(I_r_p));
		I_r_p += step;
		mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
		mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
		mTmp3 = _mm256_add_ps(mTmp3, mSum3);
		_mm256_store_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
		q_p += step;

		for (int j = 1; j <= r; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;

		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			s0_p2 += step;
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			s0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			s1_p2 += step;
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			s1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			s2_p2 += step;
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			s2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			s3_p2 += step;
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));
			s3_p1 += step;

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			mSum0 = _mm256_add_ps(mSum0, _mm256_loadu_ps(s0_p2));
			mSum0 = _mm256_sub_ps(mSum0, _mm256_load_ps(s0_p1));
			s0_p1 += step;
			mSum1 = _mm256_add_ps(mSum1, _mm256_loadu_ps(s1_p2));
			mSum1 = _mm256_sub_ps(mSum1, _mm256_load_ps(s1_p1));
			s1_p1 += step;
			mSum2 = _mm256_add_ps(mSum2, _mm256_loadu_ps(s2_p2));
			mSum2 = _mm256_sub_ps(mSum2, _mm256_load_ps(s2_p1));
			s2_p1 += step;
			mSum3 = _mm256_add_ps(mSum3, _mm256_loadu_ps(s3_p2));
			mSum3 = _mm256_sub_ps(mSum3, _mm256_load_ps(s3_p1));
			s3_p1 += step;

			mTmp0 = _mm256_mul_ps(mSum0, _mm256_loadu_ps(I_b_p));
			I_b_p += step;
			mTmp1 = _mm256_mul_ps(mSum1, _mm256_loadu_ps(I_g_p));
			I_g_p += step;
			mTmp2 = _mm256_mul_ps(mSum2, _mm256_loadu_ps(I_r_p));
			I_r_p += step;
			mTmp3 = _mm256_add_ps(mTmp0, mTmp1);
			mTmp3 = _mm256_add_ps(mTmp3, mTmp2);
			mTmp3 = _mm256_add_ps(mTmp3, mSum3);
			_mm256_storeu_ps(q_p, _mm256_mul_ps(mTmp3, mDiv));
			q_p += step;
		}
	}
}
