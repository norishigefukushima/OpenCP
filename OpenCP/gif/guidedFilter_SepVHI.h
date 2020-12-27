#pragma once

#include "guidedFilter.hpp"
#include <intrin.h>

inline void copyMakeBorderReplicateForLineBuffers(cv::Mat& buffer, const int R)
{
	const int height = buffer.rows;
	const int width = buffer.cols - 2 * R;
	for (int c = 0; c < height; c++)
	{
		const float vs = buffer.at<float>(c, R);
		const float ve = buffer.at<float>(c, width - 1 + R);

		float* ts = buffer.ptr<float>(c, 0);
		float* te = buffer.ptr<float>(c, width + R);
		for (int k = 0; k < R; k += 8)
		{
			_mm256_store_ps(ts, _mm256_set1_ps(vs));
			_mm256_store_ps(te, _mm256_set1_ps(ve));
			ts += 8;
			te += 8;
		}
	}
}


void Ip2ab_Guide1_sep_VHI_Share_AVX(cv::Mat& I, cv::Mat& p, const int r, float eps, cv::Mat& a, cv::Mat& b);
void Ip2ab_Guide1_sep_VHI_AVX_omp(cv::Mat& I, cv::Mat& p, const int r, float eps, cv::Mat& a, cv::Mat& b);
void ab2q_Guide1_sep_VHI_AVX(cv::Mat& a, cv::Mat& b, cv::Mat& guide, const int r, cv::Mat& dest);
void ab2q_Guide1_sep_VHI_AVX_omp(cv::Mat& a, cv::Mat& b, cv::Mat& guide, const int r, cv::Mat& dest);

void Ip2ab_Guide3_sep_VHI_AVX(cv::Mat& I_b, cv::Mat& I_g, cv::Mat& I_r, cv::Mat& p, const int r, float eps,
	cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b);
void Ip2ab_Guide3_sep_VHI_AVX_omp(cv::Mat& I_b, cv::Mat& I_g, cv::Mat& I_r, cv::Mat& p, const int r, float eps,
	cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b);
void Ip2ab_Guide3_sep_VHI_Unroll2_AVX(cv::Mat& I_b, cv::Mat& I_g, cv::Mat& I_r, cv::Mat& p, const int r, float eps,
	cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b);
void Ip2ab_Guide3_sep_VHI_Unroll2_AVX_omp(cv::Mat& I_b, cv::Mat& I_g, cv::Mat& I_r, cv::Mat& p, const int r, float eps,
	cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b);

void ab2q_Guide3_sep_VHI_AVX(cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b,
	cv::Mat& guide_b, cv::Mat& guide_g, cv::Mat& guide_r, const int r,
	cv::Mat& dest);
void ab2q_Guide3_sep_VHI_AVX_omp(cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b,
	cv::Mat& guide_b, cv::Mat& guide_g, cv::Mat& guide_r, const int r,
	cv::Mat& dest);
void ab2q_Guide3_sep_VHI_Unroll2_AVX(cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b,
	cv::Mat& guide_b, cv::Mat& guide_g, cv::Mat& guide_r, const int r,
	cv::Mat& dest);
void ab2q_guide3_sep_VHI_Unroll2_AVX_omp(cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& b,
	cv::Mat& guide_b, cv::Mat& guide_g, cv::Mat& guide_r, const int r,
	cv::Mat& dest);

//for upsamplingvoid ab2p_fmadd_omp(cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& g_b, cv::Mat& g_g, cv::Mat& g_r, cv::Mat& b, cv::Mat& dest);
void blurSeparableVHI(const cv::Mat& src0, const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& src3, const int r,
	cv::Mat& dest0, cv::Mat& dest1, cv::Mat& dest2, cv::Mat& dest3);
void blurSeparableVHI_omp(const cv::Mat& src0, const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& src3, const int r,
	cv::Mat& dest0, cv::Mat& dest1, cv::Mat& dest2, cv::Mat& dest3);

void blurSeparableVHI(const cv::Mat& src0, const cv::Mat& src1, const int r,
	cv::Mat& dest0, cv::Mat& dest1);
void blurSeparableVHI_omp(const cv::Mat& src0, const cv::Mat& src1, const int r,
	cv::Mat& dest0, cv::Mat& dest1);

void ab2q_fmadd(cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& g_b, cv::Mat& g_g, cv::Mat& g_r, cv::Mat& b, cv::Mat& dest);
void ab2q_fmadd_omp(cv::Mat& a_b, cv::Mat& a_g, cv::Mat& a_r, cv::Mat& g_b, cv::Mat& g_g, cv::Mat& g_r, cv::Mat& b, cv::Mat& dest);

class guidedFilter_SepVHI : public cp::GuidedFilterBase
{
protected:
	int parallelType;

	cv::Mat a;
	cv::Mat b;

	cv::Mat a_b;
	cv::Mat a_g;
	cv::Mat a_r;

	//for upsampling
	cv::Mat mean_a_b;
	cv::Mat mean_a_g;
	cv::Mat mean_a_r;
	cv::Mat mean_b;

	cv::Mat a_high_b;
	cv::Mat a_high_g;
	cv::Mat a_high_r;

	void filter_Guide1(cv::Mat& input, cv::Mat& guide, cv::Mat& output);
	void upsample_Guide1(cv::Mat& input_low, cv::Mat& guide, cv::Mat& guide_low, cv::Mat& output);

	void filter_Guide3(cv::Mat& input, std::vector<cv::Mat>& guide, cv::Mat& output);
	void upsample_Guide3(cv::Mat& input_low, std::vector<cv::Mat>& guide, std::vector<cv::Mat>& guide_low, cv::Mat& output);

public:

	guidedFilter_SepVHI(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: GuidedFilterBase(_src, _guide, _dest, _r, _eps), parallelType(_parallelType)
	{
		implementation = cp::GUIDED_SEP_VHI;
	}

	void filter() override;
	void filterVector() override;
	void upsample() override;
};