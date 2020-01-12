#include "guidedFilter_Merge_nonSplit.h"

using namespace std;
using namespace cv;

void guidedFilter_nonSplit_nonVec(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps, int parallelType)
{
	if (src.channels() == 1)
	{
		guidedFilter_Merge_nonVec gf(src, guide, dest, r, eps, parallelType);
		gf.filter();
	}
	else if (src.channels() == 3)
	{
		if (guide.channels() == 1)
		{
			guidedFilter_nonSplit_Guide1_nonVec gf(src, guide, dest, r, eps, parallelType);
			gf.init();
			gf.filter();
		}
		else if (guide.channels() == 3)
		{
			guidedFilter_nonSplit_Guide3_nonVec gf(src, guide, dest, r, eps, parallelType);
			gf.init();
			gf.filter();
		}
	}
}

void guidedFilter_nonSplit_SSE(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps, int parallelType)
{
	if (src.channels() == 1)
	{
		guidedFilter_Merge_SSE gf(src, guide, dest, r, eps, parallelType);
		gf.filter();
	}
	else if (src.channels() == 3)
	{
		if (guide.channels() == 1)
		{
			guidedFilter_nonSplit_Guide1_SSE gf(src, guide, dest, r, eps, parallelType);
			gf.init();
			gf.filter();
		}
		else if (guide.channels() == 3)
		{
			guidedFilter_nonSplit_Guide3_SSE gf(src, guide, dest, r, eps, parallelType);
			gf.init();
			gf.filter();
		}
	}
}

void guidedFilter_nonSplit_AVX(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps, int parallelType)
{
	if (src.channels() == 1)
	{
		guidedFilter_Merge_AVX gf(src, guide, dest, r, eps, parallelType);
		gf.filter();
	}
	else if (src.channels() == 3)
	{
		if (guide.channels() == 1)
		{
			guidedFilter_nonSplit_Guide1_AVX gf(src, guide, dest, r, eps, parallelType);
			gf.init();
			gf.filter();
		}
		else if (guide.channels() == 3)
		{
			guidedFilter_nonSplit_Guide3_AVX gf(src, guide, dest, r, eps, parallelType);
			gf.init();
			gf.filter();
		}
	}
}



void guidedFilter_nonSplit_Guide1_nonVec::init()
{
	a.resize(3);
	b.resize(3);
	temp.resize(8);

	for (int i = 0; i < a.size(); i++) a[i].create(src.size(), CV_32F);
	for (int i = 0; i < b.size(); i++) b[i].create(src.size(), CV_32F);
	for (int i = 0; i < temp.size(); i++) temp[i].create(src.size(), CV_32F);
}

void guidedFilter_nonSplit_Guide1_nonVec::filter()
{
	RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec    rsf_ip2ab(src, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec csf_ip2ab(temp, a, b, r, eps, parallelType);  csf_ip2ab.filter();
	RowSumFilter_nonSplit_ab2q_Guide1_nonVec     rsf_ab2q(a, b, temp, r, parallelType);        rsf_ab2q.filter();
	ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec  csf_ab2q(temp, guide, dest, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_nonSplit_Guide3_nonVec::init()
{

}

void guidedFilter_nonSplit_Guide3_nonVec::filter()
{
	RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec    rsf_ip2ab(src, guide, temp, r, parallelType);  rsf_ip2ab.filter();
	ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec csf_ip2ab(temp, va, b, r, eps, parallelType);  csf_ip2ab.filter();
	RowSumFilter_nonSplit_ab2q_Guide3_nonVec     rsf_ab2q(va, b, temp, r, parallelType);        rsf_ab2q.filter();
	ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec  csf_ab2q(temp, guide, dest, r, parallelType);  csf_ab2q.filter();
}



void guidedFilter_nonSplit_Guide1_SSE::init()
{

}

void guidedFilter_nonSplit_Guide1_SSE::filter()
{
	RowSumFilter_nonSplit_Ip2ab_Guide1_SSE    rsf_ip2ab(src, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_nonSplit_Ip2ab_Guide1_SSE csf_ip2ab(temp, a, b, r, eps, parallelType);  csf_ip2ab.filter();
	RowSumFilter_nonSplit_ab2q_Guide1_SSE     rsf_ab2q(a, b, temp, r, parallelType);        rsf_ab2q.filter();
	ColumnSumFilter_nonSplit_ab2q_Guide1_SSE  csf_ab2q(temp, guide, dest, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_nonSplit_Guide3_SSE::init()
{

}

void guidedFilter_nonSplit_Guide3_SSE::filter()
{
	RowSumFilter_nonSplit_Ip2ab_Guide3_SSE    rsf_ip2ab(src, guide, temp, r, parallelType);  rsf_ip2ab.filter();
	ColumnSumFilter_nonSplit_Ip2ab_Guide3_SSE csf_ip2ab(temp, va, b, r, eps, parallelType);  csf_ip2ab.filter();
	RowSumFilter_nonSplit_ab2q_Guide3_SSE     rsf_ab2q(va, b, temp, r, parallelType);        rsf_ab2q.filter();
	ColumnSumFilter_nonSplit_ab2q_Guide3_SSE  csf_ab2q(temp, guide, dest, r, parallelType);  csf_ab2q.filter();
}



void guidedFilter_nonSplit_Guide1_AVX::init()
{

}

void guidedFilter_nonSplit_Guide1_AVX::filter()
{
	RowSumFilter_nonSplit_Ip2ab_Guide1_AVX    rsf_ip2ab(src, guide, temp, r, parallelType); rsf_ip2ab.filter();
	ColumnSumFilter_nonSplit_Ip2ab_Guide1_AVX csf_ip2ab(temp, a, b, r, eps, parallelType);  csf_ip2ab.filter();
	RowSumFilter_nonSplit_ab2q_Guide1_AVX     rsf_ab2q(a, b, temp, r, parallelType);        rsf_ab2q.filter();
	ColumnSumFilter_nonSplit_ab2q_Guide1_AVX  csf_ab2q(temp, guide, dest, r, parallelType); csf_ab2q.filter();
}

void guidedFilter_nonSplit_Guide3_AVX::init()
{

}

void guidedFilter_nonSplit_Guide3_AVX::filter()
{
	RowSumFilter_nonSplit_Ip2ab_Guide3_AVX    rsf_ip2ab(src, guide, temp, r, parallelType);  rsf_ip2ab.filter();
	ColumnSumFilter_nonSplit_Ip2ab_Guide3_AVX csf_ip2ab(temp, va, b, r, eps, parallelType);  csf_ip2ab.filter();
	RowSumFilter_nonSplit_ab2q_Guide3_AVX     rsf_ab2q(va, b, temp, r, parallelType);        rsf_ab2q.filter();
	ColumnSumFilter_nonSplit_ab2q_Guide3_AVX  csf_ab2q(temp, guide, dest, r, parallelType);  csf_ab2q.filter();
}



/*********************
 * Guide1 nonVec
 *********************/
void RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* ptr_I_prev = I.ptr<float>(j);
		float* ptr_I_next = I.ptr<float>(j) + 1;
		
		float* ptr_p_b_prev = p.ptr<float>(j);
		float* ptr_p_g_prev = p.ptr<float>(j) + 1;
		float* ptr_p_r_prev = p.ptr<float>(j) + 2;

		float* ptr_p_b_next = p.ptr<float>(j) + 3;
		float* ptr_p_g_next = p.ptr<float>(j) + 4;
		float* ptr_p_r_next = p.ptr<float>(j) + 5;

		float* ptr_mean_I    = tempVec[0].ptr<float>(j);
		float* ptr_mean_p_b  = tempVec[1].ptr<float>(j);
		float* ptr_mean_p_g  = tempVec[2].ptr<float>(j);
		float* ptr_mean_p_r  = tempVec[3].ptr<float>(j);
		float* ptr_corr_I    = tempVec[4].ptr<float>(j);
		float* ptr_corr_Ip_b = tempVec[5].ptr<float>(j);
		float* ptr_corr_Ip_g = tempVec[6].ptr<float>(j);
		float* ptr_corr_Ip_r = tempVec[7].ptr<float>(j);

		float sum_mean_I    = *ptr_I_prev   * (r + 1);
		float sum_mean_p_b  = *ptr_p_b_prev * (r + 1);
		float sum_mean_p_g  = *ptr_p_g_prev * (r + 1);
		float sum_mean_p_r  = *ptr_p_r_prev * (r + 1);
		float sum_corr_I    = *ptr_I_prev   * *ptr_I_prev   * (r + 1);
		float sum_corr_Ip_b = *ptr_I_prev   * *ptr_p_b_prev * (r + 1);
		float sum_corr_Ip_g = *ptr_I_prev   * *ptr_p_g_prev * (r + 1);
		float sum_corr_Ip_r = *ptr_I_prev   * *ptr_p_r_prev * (r + 1);

		for (int i = 1; i <= r; i++)
		{
			sum_mean_I    += *ptr_I_next;
			sum_mean_p_b  += *ptr_p_b_next;
			sum_mean_p_g  += *ptr_p_g_next;
			sum_mean_p_r  += *ptr_p_r_next;
			sum_corr_I    += *ptr_I_next * *ptr_I_next;
			sum_corr_Ip_b += *ptr_I_next * *ptr_p_b_next;
			sum_corr_Ip_g += *ptr_I_next * *ptr_p_g_next;
			sum_corr_Ip_r += *ptr_I_next * *ptr_p_r_next;

			ptr_I_next++;
			ptr_p_b_next += 3;
			ptr_p_g_next += 3;
			ptr_p_r_next += 3;
		}
		*ptr_mean_I = sum_mean_I;
		*ptr_mean_p_b = sum_mean_p_b;
		*ptr_mean_p_g = sum_mean_p_g;
		*ptr_mean_p_r = sum_mean_p_r;
		*ptr_corr_I = sum_corr_I;
		*ptr_corr_Ip_b = sum_corr_Ip_b;
		*ptr_corr_Ip_g = sum_corr_Ip_g;
		*ptr_corr_Ip_r = sum_corr_Ip_r;

		ptr_mean_I++;
		ptr_mean_p_b++;
		ptr_mean_p_g++;
		ptr_mean_p_r++;
		ptr_corr_I++;
		ptr_corr_Ip_b++;
		ptr_corr_Ip_g++; 
		ptr_corr_Ip_r++;

		for (int i = 1; i <= r; i++)
		{
			sum_mean_I    += *ptr_I_next   - *ptr_I_prev;
			sum_mean_p_b  += *ptr_p_b_next - *ptr_p_b_prev;
			sum_mean_p_g  += *ptr_p_g_next - *ptr_p_g_prev;
			sum_mean_p_r  += *ptr_p_r_next - *ptr_p_r_prev;
			sum_corr_I    += (*ptr_I_next * *ptr_I_next)   - (*ptr_I_prev * *ptr_I_prev);
			sum_corr_Ip_b += (*ptr_I_next * *ptr_p_b_next) - (*ptr_I_prev * *ptr_p_b_prev);
			sum_corr_Ip_g += (*ptr_I_next * *ptr_p_g_next) - (*ptr_I_prev * *ptr_p_g_prev);
			sum_corr_Ip_r += (*ptr_I_next * *ptr_p_r_next) - (*ptr_I_prev * *ptr_p_r_prev);

			ptr_I_next++;
			ptr_p_b_next += 3;
			ptr_p_g_next += 3;
			ptr_p_r_next += 3;

			*ptr_mean_I = sum_mean_I;
			*ptr_mean_p_b = sum_mean_p_b;
			*ptr_mean_p_g = sum_mean_p_g;
			*ptr_mean_p_r = sum_mean_p_r;
			*ptr_corr_I = sum_corr_I;
			*ptr_corr_Ip_b = sum_corr_Ip_b;
			*ptr_corr_Ip_g = sum_corr_Ip_g;
			*ptr_corr_Ip_r = sum_corr_Ip_r;

			ptr_mean_I++;
			ptr_mean_p_b++;
			ptr_mean_p_g++;
			ptr_mean_p_r++;
			ptr_corr_I++;
			ptr_corr_Ip_b++;
			ptr_corr_Ip_g++;
			ptr_corr_Ip_r++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum_mean_I    += *ptr_I_next   - *ptr_I_prev;
			sum_mean_p_b  += *ptr_p_b_next - *ptr_p_b_prev;
			sum_mean_p_g  += *ptr_p_g_next - *ptr_p_g_prev;
			sum_mean_p_r  += *ptr_p_r_next - *ptr_p_r_prev;
			sum_corr_I    += (*ptr_I_next * *ptr_I_next)   - (*ptr_I_prev * *ptr_I_prev);
			sum_corr_Ip_b += (*ptr_I_next * *ptr_p_b_next) - (*ptr_I_prev * *ptr_p_b_prev);
			sum_corr_Ip_g += (*ptr_I_next * *ptr_p_g_next) - (*ptr_I_prev * *ptr_p_g_prev);
			sum_corr_Ip_r += (*ptr_I_next * *ptr_p_r_next) - (*ptr_I_prev * *ptr_p_r_prev);

			ptr_I_prev++;
			ptr_p_b_prev += 3;
			ptr_p_g_prev += 3;
			ptr_p_r_prev += 3;

			ptr_I_next++;
			ptr_p_b_next += 3;
			ptr_p_g_next += 3; 
			ptr_p_r_next += 3;

			*ptr_mean_I = sum_mean_I;
			*ptr_mean_p_b = sum_mean_p_b;
			*ptr_mean_p_g = sum_mean_p_g;
			*ptr_mean_p_r = sum_mean_p_r;
			*ptr_corr_I = sum_corr_I;
			*ptr_corr_Ip_b = sum_corr_Ip_b;
			*ptr_corr_Ip_g = sum_corr_Ip_g;
			*ptr_corr_Ip_r = sum_corr_Ip_r;
			
			ptr_mean_I++;
			ptr_mean_p_b++;
			ptr_mean_p_g++;
			ptr_mean_p_r++;
			ptr_corr_I++;
			ptr_corr_Ip_b++;
			ptr_corr_Ip_g++;
			ptr_corr_Ip_r++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum_mean_I    += *ptr_I_next   - *ptr_I_prev;
			sum_mean_p_b  += *ptr_p_b_next - *ptr_p_b_prev;
			sum_mean_p_g  += *ptr_p_g_next - *ptr_p_g_prev;
			sum_mean_p_r  += *ptr_p_r_next - *ptr_p_r_prev;
			sum_corr_I    += (*ptr_I_next * *ptr_I_next)   - (*ptr_I_prev * *ptr_I_prev);
			sum_corr_Ip_b += (*ptr_I_next * *ptr_p_b_next) - (*ptr_I_prev * *ptr_p_b_prev);
			sum_corr_Ip_g += (*ptr_I_next * *ptr_p_g_next) - (*ptr_I_prev * *ptr_p_g_prev);
			sum_corr_Ip_r += (*ptr_I_next * *ptr_p_r_next) - (*ptr_I_prev * *ptr_p_r_prev);

			ptr_I_prev++;
			ptr_p_b_prev += 3;
			ptr_p_g_prev += 3;
			ptr_p_r_prev += 3;

			*ptr_mean_I = sum_mean_I;
			*ptr_mean_p_b = sum_mean_p_b;
			*ptr_mean_p_g = sum_mean_p_g;
			*ptr_mean_p_r = sum_mean_p_r;
			*ptr_corr_I = sum_corr_I;
			*ptr_corr_Ip_b = sum_corr_Ip_b;
			*ptr_corr_Ip_g = sum_corr_Ip_g;
			*ptr_corr_Ip_r = sum_corr_Ip_r;

			ptr_mean_I++;
			ptr_mean_p_b++;
			ptr_mean_p_g++;
			ptr_mean_p_r++;
			ptr_corr_I++;
			ptr_corr_Ip_b++;
			ptr_corr_Ip_g++;
			ptr_corr_Ip_r++;
		}
	}
}

void RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec::filter_omp_impl()
{
	
}

void RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec::operator()(const cv::Range& range) const
{
	
}



void ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* ptr_mean_I_prev    = tempVec[0].ptr<float>(0) + i;
		float* ptr_mean_p_b_prev  = tempVec[1].ptr<float>(0) + i;
		float* ptr_mean_p_g_prev  = tempVec[2].ptr<float>(0) + i;
		float* ptr_mean_p_r_prev  = tempVec[3].ptr<float>(0) + i;
		float* ptr_corr_I_prev    = tempVec[4].ptr<float>(0) + i;
		float* ptr_corr_Ip_b_prev = tempVec[5].ptr<float>(0) + i;
		float* ptr_corr_Ip_g_prev = tempVec[6].ptr<float>(0) + i;
		float* ptr_corr_Ip_r_prev = tempVec[7].ptr<float>(0) + i;

		float* ptr_mean_I_next    = tempVec[0].ptr<float>(1) + i;
		float* ptr_mean_p_b_next  = tempVec[1].ptr<float>(1) + i;
		float* ptr_mean_p_g_next  = tempVec[2].ptr<float>(1) + i;
		float* ptr_mean_p_r_next  = tempVec[3].ptr<float>(1) + i;
		float* ptr_corr_I_next    = tempVec[4].ptr<float>(1) + i;
		float* ptr_corr_Ip_b_next = tempVec[5].ptr<float>(1) + i;
		float* ptr_corr_Ip_g_next = tempVec[6].ptr<float>(1) + i;
		float* ptr_corr_Ip_r_next = tempVec[7].ptr<float>(1) + i;

		float* ptr_a_b = a[0].ptr<float>(0) + i;
		float* ptr_a_g = a[1].ptr<float>(0) + i;
		float* ptr_a_r = a[2].ptr<float>(0) + i;
		float* ptr_b_b = b[0].ptr<float>(0) + i;
		float* ptr_b_g = b[1].ptr<float>(0) + i;
		float* ptr_b_r = b[2].ptr<float>(0) + i;

		float sum_mean_I    = *ptr_mean_I_prev    * (r + 1);
		float sum_mean_p_b  = *ptr_mean_p_b_prev  * (r + 1);
		float sum_mean_p_g  = *ptr_mean_p_g_prev  * (r + 1);
		float sum_mean_p_r  = *ptr_mean_p_r_prev  * (r + 1);
		float sum_corr_I    = *ptr_corr_I_prev    * (r + 1);
		float sum_corr_Ip_b = *ptr_corr_Ip_b_prev * (r + 1);
		float sum_corr_Ip_g = *ptr_corr_Ip_g_prev * (r + 1);
		float sum_corr_Ip_r = *ptr_corr_Ip_r_prev * (r + 1);

		for (int j = 1; j <= r; j++)
		{
			sum_mean_I += *ptr_mean_I_next;
			sum_mean_p_b += *ptr_mean_p_b_next;
			sum_mean_p_g += *ptr_mean_p_g_next;
			sum_mean_p_r += *ptr_mean_p_r_next;
			sum_corr_I += *ptr_corr_I_next;
			sum_corr_Ip_b += *ptr_corr_Ip_b_next;
			sum_corr_Ip_g += *ptr_corr_Ip_g_next;
			sum_corr_Ip_r += *ptr_corr_Ip_r_next;

			ptr_mean_I_next += step;
			ptr_mean_p_b_next += step;
			ptr_mean_p_g_next += step;
			ptr_mean_p_r_next += step;
			ptr_corr_I_next += step;
			ptr_corr_Ip_b_next += step;
			ptr_corr_Ip_g_next += step;
			ptr_corr_Ip_r_next += step;
		}
		float meanI = sum_mean_I * div;
		float meanp_b = sum_mean_p_b * div;
		float meanp_g = sum_mean_p_g * div;
		float meanp_r = sum_mean_p_r * div;
		float corrI = sum_corr_I * div;
		float corrIp_b = sum_corr_Ip_b * div;
		float corrIp_g = sum_corr_Ip_g * div;
		float corrIp_r = sum_corr_Ip_r * div;

		float varI = corrI - meanI * meanI;
		float covIp_b = corrIp_b - meanI * meanp_b;
		float covIp_g = corrIp_g - meanI * meanp_g;
		float covIp_r = corrIp_r - meanI * meanp_r;

		*ptr_a_b = covIp_b / (varI + eps);
		*ptr_a_g = covIp_g / (varI + eps);
		*ptr_a_r = covIp_r / (varI + eps);
		*ptr_b_b = meanp_b - *ptr_a_b * meanI;
		*ptr_b_g = meanp_g - *ptr_a_g * meanI;
		*ptr_b_r = meanp_r - *ptr_a_r * meanI;

		ptr_a_b += step;
		ptr_a_g += step;
		ptr_a_r += step;
		ptr_b_b += step;
		ptr_b_g += step;
		ptr_b_r += step;

		for (int j = 1; j <= r; j++)
		{
			sum_mean_I += *ptr_mean_I_next - *ptr_mean_I_prev;
			sum_mean_p_b += *ptr_mean_p_b_next - *ptr_mean_p_b_prev;
			sum_mean_p_g += *ptr_mean_p_g_next - *ptr_mean_p_g_prev;
			sum_mean_p_r += *ptr_mean_p_r_next - *ptr_mean_p_r_prev;
			sum_corr_I += *ptr_corr_I_next - *ptr_corr_I_prev;
			sum_corr_Ip_b += *ptr_corr_Ip_b_next - *ptr_corr_Ip_b_prev;
			sum_corr_Ip_g += *ptr_corr_Ip_g_next - *ptr_corr_Ip_g_prev;
			sum_corr_Ip_r += *ptr_corr_Ip_r_next - *ptr_corr_Ip_r_prev;

			ptr_mean_I_next += step;
			ptr_mean_p_b_next += step;
			ptr_mean_p_g_next += step;
			ptr_mean_p_r_next += step;
			ptr_corr_I_next += step;
			ptr_corr_Ip_b_next += step;
			ptr_corr_Ip_g_next += step;
			ptr_corr_Ip_r_next += step;

			meanI = sum_mean_I * div;
			meanp_b = sum_mean_p_b * div;
			meanp_g = sum_mean_p_g * div;
			meanp_r = sum_mean_p_r * div;
			corrI = sum_corr_I * div;
			corrIp_b = sum_corr_Ip_b * div;
			corrIp_g = sum_corr_Ip_g * div;
			corrIp_r = sum_corr_Ip_r * div;

			varI = corrI - meanI * meanI;
			covIp_b = corrIp_b - meanI * meanp_b;
			covIp_g = corrIp_g - meanI * meanp_g;
			covIp_r = corrIp_r - meanI * meanp_r;

			*ptr_a_b = covIp_b / (varI + eps);
			*ptr_a_g = covIp_g / (varI + eps);
			*ptr_a_r = covIp_r / (varI + eps);
			*ptr_b_b = meanp_b - *ptr_a_b * meanI;
			*ptr_b_g = meanp_g - *ptr_a_g * meanI;
			*ptr_b_r = meanp_r - *ptr_a_r * meanI;

			ptr_a_b += step;
			ptr_a_g += step;
			ptr_a_r += step;
			ptr_b_b += step;
			ptr_b_g += step;
			ptr_b_r += step;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum_mean_I += *ptr_mean_I_next - *ptr_mean_I_prev;
			sum_mean_p_b += *ptr_mean_p_b_next - *ptr_mean_p_b_prev;
			sum_mean_p_g += *ptr_mean_p_g_next - *ptr_mean_p_g_prev;
			sum_mean_p_r += *ptr_mean_p_r_next - *ptr_mean_p_r_prev;
			sum_corr_I += *ptr_corr_I_next - *ptr_corr_I_prev;
			sum_corr_Ip_b += *ptr_corr_Ip_b_next - *ptr_corr_Ip_b_prev;
			sum_corr_Ip_g += *ptr_corr_Ip_g_next - *ptr_corr_Ip_g_prev;
			sum_corr_Ip_r += *ptr_corr_Ip_r_next - *ptr_corr_Ip_r_prev;

			ptr_mean_I_prev += step;
			ptr_mean_p_b_prev += step;
			ptr_mean_p_g_prev += step;
			ptr_mean_p_r_prev += step;
			ptr_corr_I_prev += step;
			ptr_corr_Ip_b_prev += step;
			ptr_corr_Ip_g_prev += step;
			ptr_corr_Ip_r_prev += step;

			ptr_mean_I_next += step;
			ptr_mean_p_b_next += step;
			ptr_mean_p_g_next += step;
			ptr_mean_p_r_next += step;
			ptr_corr_I_next += step;
			ptr_corr_Ip_b_next += step;
			ptr_corr_Ip_g_next += step;
			ptr_corr_Ip_r_next += step;

			meanI = sum_mean_I * div;
			meanp_b = sum_mean_p_b * div;
			meanp_g = sum_mean_p_g * div;
			meanp_r = sum_mean_p_r * div;
			corrI = sum_corr_I * div;
			corrIp_b = sum_corr_Ip_b * div;
			corrIp_g = sum_corr_Ip_g * div;
			corrIp_r = sum_corr_Ip_r * div;

			varI = corrI - meanI * meanI;
			covIp_b = corrIp_b - meanI * meanp_b;
			covIp_g = corrIp_g - meanI * meanp_g;
			covIp_r = corrIp_r - meanI * meanp_r;

			*ptr_a_b = covIp_b / (varI + eps);
			*ptr_a_g = covIp_g / (varI + eps);
			*ptr_a_r = covIp_r / (varI + eps);
			*ptr_b_b = meanp_b - *ptr_a_b * meanI;
			*ptr_b_g = meanp_g - *ptr_a_g * meanI;
			*ptr_b_r = meanp_r - *ptr_a_r * meanI;

			ptr_a_b += step;
			ptr_a_g += step;
			ptr_a_r += step;
			ptr_b_b += step;
			ptr_b_g += step;
			ptr_b_r += step;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum_mean_I += *ptr_mean_I_next - *ptr_mean_I_prev;
			sum_mean_p_b += *ptr_mean_p_b_next - *ptr_mean_p_b_prev;
			sum_mean_p_g += *ptr_mean_p_g_next - *ptr_mean_p_g_prev;
			sum_mean_p_r += *ptr_mean_p_r_next - *ptr_mean_p_r_prev;
			sum_corr_I += *ptr_corr_I_next - *ptr_corr_I_prev;
			sum_corr_Ip_b += *ptr_corr_Ip_b_next - *ptr_corr_Ip_b_prev;
			sum_corr_Ip_g += *ptr_corr_Ip_g_next - *ptr_corr_Ip_g_prev;
			sum_corr_Ip_r += *ptr_corr_Ip_r_next - *ptr_corr_Ip_r_prev;

			ptr_mean_I_prev += step;
			ptr_mean_p_b_prev += step;
			ptr_mean_p_g_prev += step;
			ptr_mean_p_r_prev += step;
			ptr_corr_I_prev += step;
			ptr_corr_Ip_b_prev += step;
			ptr_corr_Ip_g_prev += step;
			ptr_corr_Ip_r_prev += step;

			meanI = sum_mean_I * div;
			meanp_b = sum_mean_p_b * div;
			meanp_g = sum_mean_p_g * div;
			meanp_r = sum_mean_p_r * div;
			corrI = sum_corr_I * div;
			corrIp_b = sum_corr_Ip_b * div;
			corrIp_g = sum_corr_Ip_g * div;
			corrIp_r = sum_corr_Ip_r * div;

			varI = corrI - meanI * meanI;
			covIp_b = corrIp_b - meanI * meanp_b;
			covIp_g = corrIp_g - meanI * meanp_g;
			covIp_r = corrIp_r - meanI * meanp_r;

			*ptr_a_b = covIp_b / (varI + eps);
			*ptr_a_g = covIp_g / (varI + eps);
			*ptr_a_r = covIp_r / (varI + eps);
			*ptr_b_b = meanp_b - *ptr_a_b * meanI;
			*ptr_b_g = meanp_g - *ptr_a_g * meanI;
			*ptr_b_r = meanp_r - *ptr_a_r * meanI;

			ptr_a_b += step;
			ptr_a_g += step;
			ptr_a_r += step;
			ptr_b_b += step;
			ptr_b_g += step;
			ptr_b_r += step;
		}
	}
}

void ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec::filter_omp_impl()
{
	
}

void ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec::operator()(const cv::Range& range) const
{
	
}



void RowSumFilter_nonSplit_ab2q_Guide1_nonVec::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		float* ptr_a_b_prev = a[0].ptr<float>(j);
		float* ptr_a_g_prev = a[1].ptr<float>(j);
		float* ptr_a_r_prev = a[2].ptr<float>(j);
		float* ptr_b_b_prev = b[0].ptr<float>(j);
		float* ptr_b_g_prev = b[1].ptr<float>(j);
		float* ptr_b_r_prev = b[2].ptr<float>(j);

		float* ptr_a_b_next = a[0].ptr<float>(j) + 1;
		float* ptr_a_g_next = a[1].ptr<float>(j) + 1;
		float* ptr_a_r_next = a[2].ptr<float>(j) + 1;
		float* ptr_b_b_next = b[0].ptr<float>(j) + 1;
		float* ptr_b_g_next = b[1].ptr<float>(j) + 1;
		float* ptr_b_r_next = b[2].ptr<float>(j) + 1;

		float* ptr_mean_a_b = tempVec[0].ptr<float>(j);
		float* ptr_mean_a_g = tempVec[1].ptr<float>(j);
		float* ptr_mean_a_r = tempVec[2].ptr<float>(j);
		float* ptr_mean_b_b = tempVec[3].ptr<float>(j);
		float* ptr_mean_b_g = tempVec[4].ptr<float>(j);
		float* ptr_mean_b_r = tempVec[5].ptr<float>(j);

		float sum_mean_a_b = *ptr_a_b_prev * (r + 1);
		float sum_mean_a_g = *ptr_a_g_prev * (r + 1);
		float sum_mean_a_r = *ptr_a_r_prev * (r + 1);
		float sum_mean_b_b = *ptr_b_b_prev * (r + 1);
		float sum_mean_b_g = *ptr_b_g_prev * (r + 1);
		float sum_mean_b_r = *ptr_b_r_prev * (r + 1);

		for (int i = 1; i <= r; i++)
		{
			sum_mean_a_b += *ptr_a_b_next;
			sum_mean_a_g += *ptr_a_g_next;
			sum_mean_a_r += *ptr_a_r_next;
			sum_mean_b_b += *ptr_b_b_next;
			sum_mean_b_g += *ptr_b_g_next;
			sum_mean_b_r += *ptr_b_r_next;

			ptr_a_b_next++;
			ptr_a_g_next++;
			ptr_a_r_next++;
			ptr_b_b_next++;
			ptr_b_g_next++;
			ptr_b_r_next++;
		}
		*ptr_mean_a_b = sum_mean_a_b;
		*ptr_mean_a_g = sum_mean_a_g;
		*ptr_mean_a_r = sum_mean_a_r;
		*ptr_mean_b_b = sum_mean_b_b;
		*ptr_mean_b_g = sum_mean_b_g;
		*ptr_mean_b_r = sum_mean_b_r;

		ptr_mean_a_b++;
		ptr_mean_a_g++;
		ptr_mean_a_r++;
		ptr_mean_b_b++;
		ptr_mean_b_g++;
		ptr_mean_b_r++;

		for (int i = 1; i <= r; i++)
		{
			sum_mean_a_b += *ptr_a_b_next - *ptr_a_b_prev;
			sum_mean_a_g += *ptr_a_g_next - *ptr_a_g_prev;
			sum_mean_a_r += *ptr_a_r_next - *ptr_a_r_prev;
			sum_mean_b_b += *ptr_b_b_next - *ptr_b_b_prev;
			sum_mean_b_g += *ptr_b_g_next - *ptr_b_g_prev;
			sum_mean_b_r += *ptr_b_r_next - *ptr_b_r_prev;

			*ptr_a_b_next++;
			*ptr_a_g_next++;
			*ptr_a_r_next++;
			*ptr_b_b_next++;
			*ptr_b_g_next++;
			*ptr_b_r_next++;

			*ptr_mean_a_b = sum_mean_a_b;
			*ptr_mean_a_g = sum_mean_a_g;
			*ptr_mean_a_r = sum_mean_a_r;
			*ptr_mean_b_b = sum_mean_b_b;
			*ptr_mean_b_g = sum_mean_b_g;
			*ptr_mean_b_r = sum_mean_b_r;

			ptr_mean_a_b++;
			ptr_mean_a_g++;
			ptr_mean_a_r++;
			ptr_mean_b_b++;
			ptr_mean_b_g++;
			ptr_mean_b_r++;
		}
		for (int i = r + 1; i < img_col - r - 1; i++)
		{
			sum_mean_a_b += *ptr_a_b_next - *ptr_a_b_prev;
			sum_mean_a_g += *ptr_a_g_next - *ptr_a_g_prev;
			sum_mean_a_r += *ptr_a_r_next - *ptr_a_r_prev;
			sum_mean_b_b += *ptr_b_b_next - *ptr_b_b_prev;
			sum_mean_b_g += *ptr_b_g_next - *ptr_b_g_prev;
			sum_mean_b_r += *ptr_b_r_next - *ptr_b_r_prev;

			*ptr_a_b_prev++;
			*ptr_a_g_prev++;
			*ptr_a_r_prev++;
			*ptr_b_b_prev++;
			*ptr_b_g_prev++;
			*ptr_b_r_prev++;

			*ptr_a_b_next++;
			*ptr_a_g_next++;
			*ptr_a_r_next++;
			*ptr_b_b_next++;
			*ptr_b_g_next++;
			*ptr_b_r_next++;

			*ptr_mean_a_b = sum_mean_a_b;
			*ptr_mean_a_g = sum_mean_a_g;
			*ptr_mean_a_r = sum_mean_a_r;
			*ptr_mean_b_b = sum_mean_b_b;
			*ptr_mean_b_g = sum_mean_b_g;
			*ptr_mean_b_r = sum_mean_b_r;

			ptr_mean_a_b++;
			ptr_mean_a_g++;
			ptr_mean_a_r++;
			ptr_mean_b_b++;
			ptr_mean_b_g++;
			ptr_mean_b_r++;
		}
		for (int i = img_col - r - 1; i < img_col; i++)
		{
			sum_mean_a_b += *ptr_a_b_next - *ptr_a_b_prev;
			sum_mean_a_g += *ptr_a_g_next - *ptr_a_g_prev;
			sum_mean_a_r += *ptr_a_r_next - *ptr_a_r_prev;
			sum_mean_b_b += *ptr_b_b_next - *ptr_b_b_prev;
			sum_mean_b_g += *ptr_b_g_next - *ptr_b_g_prev;
			sum_mean_b_r += *ptr_b_r_next - *ptr_b_r_prev;

			*ptr_a_b_prev++;
			*ptr_a_g_prev++;
			*ptr_a_r_prev++;
			*ptr_b_b_prev++;
			*ptr_b_g_prev++;
			*ptr_b_r_prev++;

			*ptr_mean_a_b = sum_mean_a_b;
			*ptr_mean_a_g = sum_mean_a_g;
			*ptr_mean_a_r = sum_mean_a_r;
			*ptr_mean_b_b = sum_mean_b_b;
			*ptr_mean_b_g = sum_mean_b_g;
			*ptr_mean_b_r = sum_mean_b_r;

			ptr_mean_a_b++;
			ptr_mean_a_g++;
			ptr_mean_a_r++;
			ptr_mean_b_b++;
			ptr_mean_b_g++;
			ptr_mean_b_r++;
		}
	}
}

void RowSumFilter_nonSplit_ab2q_Guide1_nonVec::filter_omp_impl()
{
	
}

void RowSumFilter_nonSplit_ab2q_Guide1_nonVec::operator()(const cv::Range& range) const
{
	
}



void ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec::filter_naive_impl()
{
	for (int i = 0; i < img_col; i++)
	{
		float* ptr_mean_a_b_prev = tempVec[0].ptr<float>(0) + i;
		float* ptr_mean_a_g_prev = tempVec[1].ptr<float>(0) + i;
		float* ptr_mean_a_r_prev = tempVec[2].ptr<float>(0) + i;
		float* ptr_mean_b_b_prev = tempVec[3].ptr<float>(0) + i;
		float* ptr_mean_b_g_prev = tempVec[4].ptr<float>(0) + i;
		float* ptr_mean_b_r_prev = tempVec[5].ptr<float>(0) + i;

		float* ptr_mean_a_b_next = tempVec[0].ptr<float>(1) + i;
		float* ptr_mean_a_g_next = tempVec[1].ptr<float>(1) + i;
		float* ptr_mean_a_r_next = tempVec[2].ptr<float>(1) + i;
		float* ptr_mean_b_b_next = tempVec[3].ptr<float>(1) + i;
		float* ptr_mean_b_g_next = tempVec[4].ptr<float>(1) + i;
		float* ptr_mean_b_r_next = tempVec[5].ptr<float>(1) + i;

		float* ptr_I = I.ptr<float>(0) + i;

		float* ptr_q_b = q.ptr<float>(0) + i * 3;
		float* ptr_q_g = q.ptr<float>(0) + i * 3 + 1;
		float* ptr_q_r = q.ptr<float>(0) + i * 3 + 2;

		float sum_mean_a_b = *ptr_mean_a_b_prev * (r + 1);
		float sum_mean_a_g = *ptr_mean_a_g_prev * (r + 1);
		float sum_mean_a_r = *ptr_mean_a_r_prev * (r + 1);
		float sum_mean_b_b = *ptr_mean_b_b_prev * (r + 1);
		float sum_mean_b_g = *ptr_mean_b_g_prev * (r + 1);
		float sum_mean_b_r = *ptr_mean_b_r_prev * (r + 1);
		for (int j = 1; j <= r; j++)
		{
			sum_mean_a_b += *ptr_mean_a_b_next;
			sum_mean_a_g += *ptr_mean_a_g_next;
			sum_mean_a_r += *ptr_mean_a_r_next;
			sum_mean_b_b += *ptr_mean_b_b_next;
			sum_mean_b_g += *ptr_mean_b_g_next;
			sum_mean_b_r += *ptr_mean_b_r_next;

			ptr_mean_a_b_next += step;
			ptr_mean_a_g_next += step;
			ptr_mean_a_r_next += step;
			ptr_mean_b_b_next += step;
			ptr_mean_b_g_next += step;
			ptr_mean_b_r_next += step;
		}
		*ptr_q_b = (sum_mean_a_b * *ptr_I + sum_mean_b_b) * div;
		*ptr_q_g = (sum_mean_a_g * *ptr_I + sum_mean_b_g) * div;
		*ptr_q_r = (sum_mean_a_r * *ptr_I + sum_mean_b_r) * div;

		ptr_I += step;
		ptr_q_b += step * 3;
		ptr_q_g += step * 3;
		ptr_q_r += step * 3;

		for (int j = 1; j <= r; j++)
		{
			sum_mean_a_b += *ptr_mean_a_b_next - *ptr_mean_a_b_prev;
			sum_mean_a_g += *ptr_mean_a_g_next - *ptr_mean_a_g_prev;
			sum_mean_a_r += *ptr_mean_a_r_next - *ptr_mean_a_r_prev;
			sum_mean_b_b += *ptr_mean_b_b_next - *ptr_mean_b_b_prev;
			sum_mean_b_g += *ptr_mean_b_g_next - *ptr_mean_b_g_prev;
			sum_mean_b_r += *ptr_mean_b_r_next - *ptr_mean_b_r_prev;

			ptr_mean_a_b_next += step;
			ptr_mean_a_g_next += step;
			ptr_mean_a_r_next += step;
			ptr_mean_b_b_next += step;
			ptr_mean_b_g_next += step;
			ptr_mean_b_r_next += step;

			*ptr_q_b = (sum_mean_a_b * *ptr_I + sum_mean_b_b) * div;
			*ptr_q_g = (sum_mean_a_g * *ptr_I + sum_mean_b_g) * div;
			*ptr_q_r = (sum_mean_a_r * *ptr_I + sum_mean_b_r) * div;

			ptr_I += step;
			ptr_q_b += step * 3;
			ptr_q_g += step * 3;
			ptr_q_r += step * 3;
		}
		for (int j = r + 1; j < img_row - r - 1; j++)
		{
			sum_mean_a_b += *ptr_mean_a_b_next - *ptr_mean_a_b_prev;
			sum_mean_a_g += *ptr_mean_a_g_next - *ptr_mean_a_g_prev;
			sum_mean_a_r += *ptr_mean_a_r_next - *ptr_mean_a_r_prev;
			sum_mean_b_b += *ptr_mean_b_b_next - *ptr_mean_b_b_prev;
			sum_mean_b_g += *ptr_mean_b_g_next - *ptr_mean_b_g_prev;
			sum_mean_b_r += *ptr_mean_b_r_next - *ptr_mean_b_r_prev;

			ptr_mean_a_b_prev += step;
			ptr_mean_a_g_prev += step;
			ptr_mean_a_r_prev += step;
			ptr_mean_b_b_prev += step;
			ptr_mean_b_g_prev += step;
			ptr_mean_b_r_prev += step;

			ptr_mean_a_b_next += step;
			ptr_mean_a_g_next += step;
			ptr_mean_a_r_next += step;
			ptr_mean_b_b_next += step;
			ptr_mean_b_g_next += step;
			ptr_mean_b_r_next += step;

			*ptr_q_b = (sum_mean_a_b * *ptr_I + sum_mean_b_b) * div;
			*ptr_q_g = (sum_mean_a_g * *ptr_I + sum_mean_b_g) * div;
			*ptr_q_r = (sum_mean_a_r * *ptr_I + sum_mean_b_r) * div;

			ptr_I += step;
			ptr_q_b += step * 3;
			ptr_q_g += step * 3;
			ptr_q_r += step * 3;
		}
		for (int j = img_row - r - 1; j < img_row; j++)
		{
			sum_mean_a_b += *ptr_mean_a_b_next - *ptr_mean_a_b_prev;
			sum_mean_a_g += *ptr_mean_a_g_next - *ptr_mean_a_g_prev;
			sum_mean_a_r += *ptr_mean_a_r_next - *ptr_mean_a_r_prev;
			sum_mean_b_b += *ptr_mean_b_b_next - *ptr_mean_b_b_prev;
			sum_mean_b_g += *ptr_mean_b_g_next - *ptr_mean_b_g_prev;
			sum_mean_b_r += *ptr_mean_b_r_next - *ptr_mean_b_r_prev;

			ptr_mean_a_b_prev += step;
			ptr_mean_a_g_prev += step;
			ptr_mean_a_r_prev += step;
			ptr_mean_b_b_prev += step;
			ptr_mean_b_g_prev += step;
			ptr_mean_b_r_prev += step;

			*ptr_q_b = (sum_mean_a_b * *ptr_I + sum_mean_b_b) * div;
			*ptr_q_g = (sum_mean_a_g * *ptr_I + sum_mean_b_g) * div;
			*ptr_q_r = (sum_mean_a_r * *ptr_I + sum_mean_b_r) * div;

			ptr_I += step;
			ptr_q_b += step * 3;
			ptr_q_g += step * 3;
			ptr_q_r += step * 3;
		}
	}
}

void ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec::filter_omp_impl()
{
	
}

void ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec::operator()(const cv::Range& range) const
{
	
}



/*********************
* Guide1 SSE
*********************/
void RowSumFilter_nonSplit_Ip2ab_Guide1_SSE::filter_naive_impl()
{
	
}

void RowSumFilter_nonSplit_Ip2ab_Guide1_SSE::filter_omp_impl()
{
	
}

void RowSumFilter_nonSplit_Ip2ab_Guide1_SSE::operator()(const cv::Range& range) const
{
	
}



void ColumnSumFilter_nonSplit_Ip2ab_Guide1_SSE::filter_naive_impl()
{
	
}

void ColumnSumFilter_nonSplit_Ip2ab_Guide1_SSE::filter_omp_impl()
{
	
}

void ColumnSumFilter_nonSplit_Ip2ab_Guide1_SSE::operator()(const cv::Range& range) const
{
	
}



void RowSumFilter_nonSplit_ab2q_Guide1_SSE::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide1_SSE::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide1_SSE::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_ab2q_Guide1_SSE::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide1_SSE::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide1_SSE::operator()(const cv::Range& range) const
{

}



/*********************
* Guide1 AVX
*********************/
void RowSumFilter_nonSplit_Ip2ab_Guide1_AVX::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_Ip2ab_Guide1_AVX::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_Ip2ab_Guide1_AVX::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_Ip2ab_Guide1_AVX::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide1_AVX::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide1_AVX::operator()(const cv::Range& range) const
{

}



void RowSumFilter_nonSplit_ab2q_Guide1_AVX::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide1_AVX::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide1_AVX::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_ab2q_Guide1_AVX::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide1_AVX::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide1_AVX::operator()(const cv::Range& range) const
{

}



/*********************
 * Guide3 nonVec
 *********************/
void RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec::filter_naive_impl()
{
	for (int j = 0; j < img_row; j++)
	{
		
	}
}

void RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec::operator()(const cv::Range& range) const
{

}



void RowSumFilter_nonSplit_ab2q_Guide3_nonVec::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide3_nonVec::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide3_nonVec::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec::operator()(const cv::Range& range) const
{

}



/*********************
* Guide3 SSE
*********************/
void RowSumFilter_nonSplit_Ip2ab_Guide3_SSE::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_Ip2ab_Guide3_SSE::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_Ip2ab_Guide3_SSE::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_Ip2ab_Guide3_SSE::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide3_SSE::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide3_SSE::operator()(const cv::Range& range) const
{

}



void RowSumFilter_nonSplit_ab2q_Guide3_SSE::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide3_SSE::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide3_SSE::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_ab2q_Guide3_SSE::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide3_SSE::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide3_SSE::operator()(const cv::Range& range) const
{

}



/*********************
* Guide3 AVX
*********************/
void RowSumFilter_nonSplit_Ip2ab_Guide3_AVX::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_Ip2ab_Guide3_AVX::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_Ip2ab_Guide3_AVX::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_Ip2ab_Guide3_AVX::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide3_AVX::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_Ip2ab_Guide3_AVX::operator()(const cv::Range& range) const
{

}



void RowSumFilter_nonSplit_ab2q_Guide3_AVX::filter_naive_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide3_AVX::filter_omp_impl()
{

}

void RowSumFilter_nonSplit_ab2q_Guide3_AVX::operator()(const cv::Range& range) const
{

}



void ColumnSumFilter_nonSplit_ab2q_Guide3_AVX::filter_naive_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide3_AVX::filter_omp_impl()
{

}

void ColumnSumFilter_nonSplit_ab2q_Guide3_AVX::operator()(const cv::Range& range) const
{

}
