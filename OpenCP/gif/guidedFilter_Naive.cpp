#include "boxFilter.hpp"
#include "../box/boxFilter_OPSAT_AoS.h"
#include "guidedFilter_Naive.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include<arithmetic.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void guidedFilter_Naive::filter()
{
	//cout << "Conventional: parallel type" << parallelType << endl;

	if (src.depth() == CV_32F)
	{
		average_method = BOX_32F;
	}
	else
	{
		average_method = BOX_64F;
	}

	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			filter_Guide1(src, guide, dest);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);

			filter_Guide3(src, vguide, dest);
		}
	}
	else if (src.channels() == 3)
	{
		split(src, vsrc);

		const int depth = src.depth();
		vdest.resize(3);
		vdest[0].create(src.size(), depth);
		vdest[1].create(src.size(), depth);
		vdest[2].create(src.size(), depth);

		if (guide.channels() == 1)
		{
			filter_Guide1(vsrc[0], guide, vdest[0]);
			filter_Guide1(vsrc[1], guide, vdest[1]);
			filter_Guide1(vsrc[2], guide, vdest[2]);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);

			filter_Guide3(vsrc[0], vguide, vdest[0]);
			filter_Guide3(vsrc[1], vguide, vdest[1]);
			filter_Guide3(vsrc[2], vguide, vdest[2]);
		}

		merge(vdest, dest);
	}
}

void guidedFilter_Naive::filterVector()
{
	//cout << "Conventional: parallel type" << parallelType << endl;

	if (vsrc[0].depth() == CV_32F)
	{
		average_method = BOX_32F;
	}
	else
	{
		average_method = BOX_64F;
	}

	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			filter_Guide1(vsrc[0], vdest[0]);
		}
		else if (guide.channels() == 3)
		{
			filter_Guide3(vsrc[0], vguide, vdest[0]);
		}
	}
	else if (src.channels() == 3)
	{
		const int depth = src.depth();

		if (guide.channels() == 1)
		{
			filter_Guide1(vsrc[0], vguide[0], vdest[0]);
			filter_Guide1(vsrc[1], vguide[0], vdest[1]);
			filter_Guide1(vsrc[2], vguide[0], vdest[2]);
		}
		else if (guide.channels() == 3)
		{
			filter_Guide3(vsrc[0], vguide, vdest[0]);
			filter_Guide3(vsrc[1], vguide, vdest[1]);
			filter_Guide3(vsrc[2], vguide, vdest[2]);
		}
	}
}

void guidedFilter_Naive::filterFast(const int ratio)
{
	resize(src, src_low, src.size() / ratio, 0, 0, downsample_method);

	r = max(r / ratio, 1);

	upsample();
}

void guidedFilter_Naive::upsample()
{
	//cout << "Naive: parallel type" << parallelType << endl;

	if (guide.size() != src.size())
	{
		src_low = src;
	}

	resize(guide, guide_low, src_low.size(), 0, 0, downsample_method);

	if (src.depth() == CV_32F)
	{
		average_method = BOX_32F;
	}
	else
	{
		average_method = BOX_64F;
	}

	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			filterFast_Guide1(src_low, guide, guide_low, dest);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);
			split(guide_low, vguide_low);

			filterFast_Guide3(src_low, vguide, vguide_low, dest);
		}
	}
	else if (src.channels() == 3)
	{
		split(src_low, vsrc_low);

		const int depth = src.depth();
		vdest.resize(3);
		vdest[0].create(guide.size(), depth);
		vdest[1].create(guide.size(), depth);
		vdest[2].create(guide.size(), depth);

		if (guide.channels() == 1)
		{
			filterFast_Guide1(vsrc_low[0], guide, guide_low, vdest[0]);
			filterFast_Guide1(vsrc_low[1], guide, guide_low, vdest[1]);
			filterFast_Guide1(vsrc_low[2], guide, guide_low, vdest[2]);
		}
		else if (guide.channels() == 3)
		{
			split(guide, vguide);
			split(guide_low, vguide_low);

			filterFast_Guide3(vsrc_low[0], vguide, vguide_low, vdest[0]);
			filterFast_Guide3(vsrc_low[1], vguide, vguide_low, vdest[1]);
			filterFast_Guide3(vsrc_low[2], vguide, vguide_low, vdest[2]);
		}

		merge(vdest, dest);
	}
}

void guidedFilter_Naive::average(Mat& src, Mat& dest, const int r)
{
	switch (average_method)
	{
	case BOX_32F:
	default:
		boxFilter_32f(src, dest, r, boxType, parallelType); break;
	case BOX_64F:
		boxFilter_64f(src, dest, r, boxType, parallelType); break;
	}
}

void guidedFilter_Naive::Ip2ab_Guide1(cv::Mat& input, cv::Mat& guide)
{
	average(guide, mean_I, r);
	average(input, mean_p, r);
	multiply(guide, guide, corr_I);
	average(corr_I, corr_I, r);
	multiply(guide, input, corr_Ip);
	average(corr_Ip, corr_Ip, r);

	//multiply(mean_I, mean_I, var_I);
	//subtract(corr_I, var_I, var_I);
	fnmadd(mean_I, mean_I, corr_I, var_I);

	add(var_I, eps, var_I, noArray(), var_I.depth());//var_I+=eps

	//multiply(mean_I, mean_p, cov_Ip);
	//subtract(corr_Ip, cov_Ip, cov_Ip);
	fnmadd(mean_I, mean_p, corr_Ip, cov_Ip);

	divide(cov_Ip, var_I, a);

	//multiply(a, mean_I, b);
	//subtract(mean_p, b, b);
	fnmadd(a, mean_I, mean_p, b);
}

void guidedFilter_Naive::ab2q_Guide1(cv::Mat& guide, cv::Mat& output)
{
	average(a, a, r);
	average(b, b, r);

	//multiply(mean_a, guide, output);
	//add(output, mean_b, output);
	fmadd(a, guide, b, output);
}

void guidedFilter_Naive::filter_Guide1(cv::Mat& input, cv::Mat& guide, cv::Mat& output)
{
	const Size imsize = input.size();
	const int imtype = input.depth();

	mean_I.create(imsize, imtype);
	mean_p.create(imsize, imtype);
	corr_I.create(imsize, imtype);
	corr_Ip.create(imsize, imtype);
	var_I.create(imsize, imtype);
	cov_Ip.create(imsize, imtype);
	a.create(imsize, imtype);
	b.create(imsize, imtype);

	Ip2ab_Guide1(input, guide);
	ab2q_Guide1(guide, output);
}


void guidedFilter_Naive::computeCovariance(const int depth)
{
	int parallelType = ParallelTypes::NAIVE;
	const int width = a_b.cols;
	const int height = a_b.rows;

	const int NAIVE = 0;
	const int SSE = 1;
	const int AVX = 2;
	int implementation = NAIVE;

	if (boxType == BOX_NAIVE_SSE || boxType == BOX_INTEGRAL_SSE || boxType == BOX_SEPARABLE_HV_SSE
		|| boxType == BOX_SSAT_HV_SSE || boxType == BOX_SSAT_HtH_SSE || boxType == BOX_SSAT_VH_SSE)
	{
		implementation = 1;
	}
	else if (boxType == BOX_OPENCV ||
		boxType == BOX_NAIVE_AVX || boxType == BOX_INTEGRAL_AVX || boxType == BOX_SEPARABLE_HV_AVX
		|| boxType == BOX_SSAT_HV_AVX || boxType == BOX_SSAT_HtH_AVX || boxType == BOX_SSAT_VH_AVX)
	{
		implementation = 2;
	}

	if (depth == CV_32F)
	{
		if (parallelType == ParallelTypes::NAIVE)
		{
			if (implementation == SSE)
			{
				for (int i = 0; i < height; i++)
				{
					float* bb = var_I_bb.ptr<float>(i);
					float* bg = var_I_bg.ptr<float>(i);
					float* br = var_I_br.ptr<float>(i);
					float* gg = var_I_gg.ptr<float>(i);
					float* gr = var_I_gr.ptr<float>(i);
					float* rr = var_I_rr.ptr<float>(i);
					float* covb = cov_Ip_b.ptr<float>(i);
					float* covg = cov_Ip_g.ptr<float>(i);
					float* covr = cov_Ip_r.ptr<float>(i);
					float* ab = a_b.ptr<float>(i);
					float* ag = a_g.ptr<float>(i);
					float* ar = a_r.ptr<float>(i);

					for (int j = 0; j < width; j += 4)
					{
						__m128 mBB = _mm_load_ps(bb);
						__m128 mBG = _mm_load_ps(bg);
						__m128 mBR = _mm_load_ps(br);
						__m128 mGG = _mm_load_ps(gg);
						__m128 mGR = _mm_load_ps(gr);
						__m128 mRR = _mm_load_ps(rr);
						bg += 4;
						bb += 4;
						br += 4;
						gg += 4;
						gr += 4;
						rr += 4;

						__m128 mTmp = _mm_mul_ps(mBG, _mm_mul_ps(mGR, mBR));
						__m128 mDet = _mm_add_ps(mTmp, mTmp);
						mTmp = _mm_mul_ps(mBB, _mm_mul_ps(mGG, mRR));
						mDet = _mm_add_ps(mDet, mTmp);
						mTmp = _mm_mul_ps(mBB, _mm_mul_ps(mGR, mGR));
						mDet = _mm_sub_ps(mDet, mTmp);
						mTmp = _mm_mul_ps(mRR, _mm_mul_ps(mBG, mBG));
						mDet = _mm_sub_ps(mDet, mTmp);
						mTmp = _mm_mul_ps(mGG, _mm_mul_ps(mBR, mBR));
						mDet = _mm_sub_ps(mDet, mTmp);
						mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);

						__m128 mC0 = _mm_fmsub_ps(mGG, mRR, _mm_mul_ps(mGR, mGR));
						__m128 mC1 = _mm_fmsub_ps(mGR, mBR, _mm_mul_ps(mBG, mRR));
						__m128 mC2 = _mm_fmsub_ps(mBG, mGR, _mm_mul_ps(mBR, mGG));
						__m128 mC4 = _mm_fmsub_ps(mBB, mRR, _mm_mul_ps(mBR, mBR));
						__m128 mC5 = _mm_fmsub_ps(mBG, mBR, _mm_mul_ps(mBB, mGR));
						__m128 mC8 = _mm_fmsub_ps(mBB, mGG, _mm_mul_ps(mBG, mBG));

						__m128 mCovB = _mm_load_ps(covb);
						__m128 mCovG = _mm_load_ps(covg);
						__m128 mCovR = _mm_load_ps(covr);
						covb += 4;
						covg += 4;
						covr += 4;

						mTmp = _mm_fmadd_ps(mCovB, mC0, _mm_mul_ps(mCovG, mC1));
						mTmp = _mm_fmadd_ps(mCovR, mC2, mTmp);
						mTmp = _mm_mul_ps(mTmp, mDet);
						_mm_store_ps(ab, mTmp);
						ab += 4;

						mTmp = _mm_fmadd_ps(mCovB, mC1, _mm_mul_ps(mCovG, mC4));
						mTmp = _mm_fmadd_ps(mCovR, mC5, mTmp);
						mTmp = _mm_mul_ps(mTmp, mDet);
						_mm_store_ps(ag, mTmp);
						ag += 4;

						mTmp = _mm_fmadd_ps(mCovB, mC2, _mm_mul_ps(mCovG, mC5));
						mTmp = _mm_fmadd_ps(mCovR, mC8, mTmp);
						mTmp = _mm_mul_ps(mTmp, mDet);
						_mm_store_ps(ar, mTmp);
						ar += 4;
					}
				}
			}
			else if (implementation == AVX)
			{
				for (int i = 0; i < height; i++)
				{
					float* bb = var_I_bb.ptr<float>(i);
					float* bg = var_I_bg.ptr<float>(i);
					float* br = var_I_br.ptr<float>(i);
					float* gg = var_I_gg.ptr<float>(i);
					float* gr = var_I_gr.ptr<float>(i);
					float* rr = var_I_rr.ptr<float>(i);
					float* covb = cov_Ip_b.ptr<float>(i);
					float* covg = cov_Ip_g.ptr<float>(i);
					float* covr = cov_Ip_r.ptr<float>(i);
					float* ab = a_b.ptr<float>(i);
					float* ag = a_g.ptr<float>(i);
					float* ar = a_r.ptr<float>(i);

					for (int j = 0; j < width; j += 8)
					{
						__m256 mBB = _mm256_load_ps(bb);
						__m256 mBG = _mm256_load_ps(bg);
						__m256 mBR = _mm256_load_ps(br);
						__m256 mGG = _mm256_load_ps(gg);
						__m256 mGR = _mm256_load_ps(gr);
						__m256 mRR = _mm256_load_ps(rr);
						bb += 8;
						bg += 8;
						br += 8;
						gg += 8;
						gr += 8;
						rr += 8;

						__m256 mTmp = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
						__m256 mDet = _mm256_add_ps(mTmp, mTmp);
						mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
						mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
						mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
						mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
						mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

						__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
						__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
						__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
						__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
						__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
						__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

						__m256 mCovB = _mm256_load_ps(covb);
						__m256 mCovG = _mm256_load_ps(covg);
						__m256 mCovR = _mm256_load_ps(covr);
						covb += 8;
						covg += 8;
						covr += 8;

						mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
						mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
						mTmp = _mm256_mul_ps(mTmp, mDet);
						_mm256_store_ps(ab, mTmp);
						ab += 8;

						mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
						mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
						mTmp = _mm256_mul_ps(mTmp, mDet);
						_mm256_store_ps(ag, mTmp);
						ag += 8;

						mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
						mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
						mTmp = _mm256_mul_ps(mTmp, mDet);
						_mm256_store_ps(ar, mTmp);
						ar += 8;
					}
				}
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					float* bb = var_I_bb.ptr<float>(i);
					float* bg = var_I_bg.ptr<float>(i);
					float* br = var_I_br.ptr<float>(i);
					float* gg = var_I_gg.ptr<float>(i);
					float* gr = var_I_gr.ptr<float>(i);
					float* rr = var_I_rr.ptr<float>(i);
					float* covb = cov_Ip_b.ptr<float>(i);
					float* covg = cov_Ip_g.ptr<float>(i);
					float* covr = cov_Ip_r.ptr<float>(i);
					float* ab = a_b.ptr<float>(i);
					float* ag = a_g.ptr<float>(i);
					float* ar = a_r.ptr<float>(i);

					for (int j = 0; j < width; j++)
					{
						const float det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
							- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
						const float id = 1.f / det;

						float c0 = *gg * *rr - *gr * *gr;
						float c1 = *gr * *br - *bg * *rr;
						float c2 = *bg * *gr - *br * *gg;
						float c4 = *bb * *rr - *br * *br;
						float c5 = *bg * *br - *bb * *gr;
						float c8 = *bb * *gg - *bg * *bg;

						*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
						*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
						*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);

						bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++;
					}
				}
			}
		}
		else
		{
			if (implementation == SSE)
			{
#pragma omp parallel for
				for (int i = 0; i < height; i++)
				{
					float* bb = var_I_bb.ptr<float>(i);
					float* bg = var_I_bg.ptr<float>(i);
					float* br = var_I_br.ptr<float>(i);
					float* gg = var_I_gg.ptr<float>(i);
					float* gr = var_I_gr.ptr<float>(i);
					float* rr = var_I_rr.ptr<float>(i);
					float* covb = cov_Ip_b.ptr<float>(i);
					float* covg = cov_Ip_g.ptr<float>(i);
					float* covr = cov_Ip_r.ptr<float>(i);
					float* ab = a_b.ptr<float>(i);
					float* ag = a_g.ptr<float>(i);
					float* ar = a_r.ptr<float>(i);

					for (int j = 0; j < width; j += 4)
					{
						__m128 mBB = _mm_load_ps(bb);
						__m128 mBG = _mm_load_ps(bg);
						__m128 mBR = _mm_load_ps(br);
						__m128 mGG = _mm_load_ps(gg);
						__m128 mGR = _mm_load_ps(gr);
						__m128 mRR = _mm_load_ps(rr);
						bg += 4;
						bb += 4;
						br += 4;
						gg += 4;
						gr += 4;
						rr += 4;

						__m128 mTmp = _mm_mul_ps(mBG, _mm_mul_ps(mGR, mBR));
						__m128 mDet = _mm_add_ps(mTmp, mTmp);
						mTmp = _mm_mul_ps(mBB, _mm_mul_ps(mGG, mRR));
						mDet = _mm_add_ps(mDet, mTmp);
						mTmp = _mm_mul_ps(mBB, _mm_mul_ps(mGR, mGR));
						mDet = _mm_sub_ps(mDet, mTmp);
						mTmp = _mm_mul_ps(mRR, _mm_mul_ps(mBG, mBG));
						mDet = _mm_sub_ps(mDet, mTmp);
						mTmp = _mm_mul_ps(mGG, _mm_mul_ps(mBR, mBR));
						mDet = _mm_sub_ps(mDet, mTmp);
						mDet = _mm_div_ps(_mm_set1_ps(1.f), mDet);

						__m128 mC0 = _mm_fmsub_ps(mGG, mRR, _mm_mul_ps(mGR, mGR));
						__m128 mC1 = _mm_fmsub_ps(mGR, mBR, _mm_mul_ps(mBG, mRR));
						__m128 mC2 = _mm_fmsub_ps(mBG, mGR, _mm_mul_ps(mBR, mGG));
						__m128 mC4 = _mm_fmsub_ps(mBB, mRR, _mm_mul_ps(mBR, mBR));
						__m128 mC5 = _mm_fmsub_ps(mBG, mBR, _mm_mul_ps(mBB, mGR));
						__m128 mC8 = _mm_fmsub_ps(mBB, mGG, _mm_mul_ps(mBG, mBG));

						__m128 mCovB = _mm_load_ps(covb);
						__m128 mCovG = _mm_load_ps(covg);
						__m128 mCovR = _mm_load_ps(covr);
						covb += 4;
						covg += 4;
						covr += 4;

						mTmp = _mm_fmadd_ps(mCovB, mC0, _mm_mul_ps(mCovG, mC1));
						mTmp = _mm_fmadd_ps(mCovR, mC2, mTmp);
						mTmp = _mm_mul_ps(mTmp, mDet);
						_mm_store_ps(ab, mTmp);
						ab += 4;

						mTmp = _mm_fmadd_ps(mCovB, mC1, _mm_mul_ps(mCovG, mC4));
						mTmp = _mm_fmadd_ps(mCovR, mC5, mTmp);
						mTmp = _mm_mul_ps(mTmp, mDet);
						_mm_store_ps(ag, mTmp);
						ag += 4;

						mTmp = _mm_fmadd_ps(mCovB, mC2, _mm_mul_ps(mCovG, mC5));
						mTmp = _mm_fmadd_ps(mCovR, mC8, mTmp);
						mTmp = _mm_mul_ps(mTmp, mDet);
						_mm_store_ps(ar, mTmp);
						ar += 4;
					}
				}
			}
			else if (implementation == AVX)
			{
#pragma omp parallel for
				for (int i = 0; i < height; i++)
				{
					float* bb = var_I_bb.ptr<float>(i);
					float* bg = var_I_bg.ptr<float>(i);
					float* br = var_I_br.ptr<float>(i);
					float* gg = var_I_gg.ptr<float>(i);
					float* gr = var_I_gr.ptr<float>(i);
					float* rr = var_I_rr.ptr<float>(i);
					float* covb = cov_Ip_b.ptr<float>(i);
					float* covg = cov_Ip_g.ptr<float>(i);
					float* covr = cov_Ip_r.ptr<float>(i);

					float* ab = a_b.ptr<float>(i);
					float* ag = a_g.ptr<float>(i);
					float* ar = a_r.ptr<float>(i);

					for (int j = 0; j < width; j += 8)
					{
						__m256 mBB = _mm256_load_ps(bb);
						__m256 mBG = _mm256_load_ps(bg);
						__m256 mBR = _mm256_load_ps(br);
						__m256 mGG = _mm256_load_ps(gg);
						__m256 mGR = _mm256_load_ps(gr);
						__m256 mRR = _mm256_load_ps(rr);
						bb += 8;
						bg += 8;
						br += 8;
						gg += 8;
						gr += 8;
						rr += 8;

						__m256 mTmp = _mm256_mul_ps(mBG, _mm256_mul_ps(mGR, mBR));
						__m256 mDet = _mm256_add_ps(mTmp, mTmp);
						mDet = _mm256_fmadd_ps(mBB, _mm256_mul_ps(mGG, mRR), mDet);
						mDet = _mm256_fnmadd_ps(mBB, _mm256_mul_ps(mGR, mGR), mDet);
						mDet = _mm256_fnmadd_ps(mRR, _mm256_mul_ps(mBG, mBG), mDet);
						mDet = _mm256_fnmadd_ps(mGG, _mm256_mul_ps(mBR, mBR), mDet);
						mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

						__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
						__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
						__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
						__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
						__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
						__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

						__m256 mCovB = _mm256_load_ps(covb);
						__m256 mCovG = _mm256_load_ps(covg);
						__m256 mCovR = _mm256_load_ps(covr);
						covb += 8;
						covg += 8;
						covr += 8;

						mTmp = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
						mTmp = _mm256_fmadd_ps(mCovR, mC2, mTmp);
						mTmp = _mm256_mul_ps(mTmp, mDet);
						_mm256_store_ps(ab, mTmp);
						ab += 8;

						mTmp = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
						mTmp = _mm256_fmadd_ps(mCovR, mC5, mTmp);
						mTmp = _mm256_mul_ps(mTmp, mDet);
						_mm256_store_ps(ag, mTmp);
						ag += 8;

						mTmp = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
						mTmp = _mm256_fmadd_ps(mCovR, mC8, mTmp);
						mTmp = _mm256_mul_ps(mTmp, mDet);
						_mm256_store_ps(ar, mTmp);
						ar += 8;
					}
				}
			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < height; i++)
				{
					float* bb = var_I_bb.ptr<float>(i);
					float* bg = var_I_bg.ptr<float>(i);
					float* br = var_I_br.ptr<float>(i);
					float* gg = var_I_gg.ptr<float>(i);
					float* gr = var_I_gr.ptr<float>(i);
					float* rr = var_I_rr.ptr<float>(i);
					float* covb = cov_Ip_b.ptr<float>(i);
					float* covg = cov_Ip_g.ptr<float>(i);
					float* covr = cov_Ip_r.ptr<float>(i);
					float* ab = a_b.ptr<float>(i);
					float* ag = a_g.ptr<float>(i);
					float* ar = a_r.ptr<float>(i);

					for (int j = 0; j < width; j++)
					{
						const float det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
							- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
						const float id = 1.f / det;

						float c0 = *gg * *rr - *gr * *gr;
						float c1 = *gr * *br - *bg * *rr;
						float c2 = *bg * *gr - *br * *gg;
						float c4 = *bb * *rr - *br * *br;
						float c5 = *bg * *br - *bb * *gr;
						float c8 = *bb * *gg - *bg * *bg;

						*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
						*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
						*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);

						bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++;
					}
				}
			}
		}
	}
	else
	{
		if (parallelType == ParallelTypes::NAIVE)
		{
			if (implementation == SSE)
			{
				for (int i = 0; i < height; i++)
				{
					double* bb = var_I_bb.ptr<double>(i);
					double* bg = var_I_bg.ptr<double>(i);
					double* br = var_I_br.ptr<double>(i);
					double* gg = var_I_gg.ptr<double>(i);
					double* gr = var_I_gr.ptr<double>(i);
					double* rr = var_I_rr.ptr<double>(i);
					double* covb = cov_Ip_b.ptr<double>(i);
					double* covg = cov_Ip_g.ptr<double>(i);
					double* covr = cov_Ip_r.ptr<double>(i);
					double* ab = a_b.ptr<double>(i);
					double* ag = a_g.ptr<double>(i);
					double* ar = a_r.ptr<double>(i);

					for (int j = 0; j < width; j += 2)
					{
						__m128d mBB = _mm_load_pd(bb);
						__m128d mBG = _mm_load_pd(bg);
						__m128d mBR = _mm_load_pd(br);
						__m128d mGG = _mm_load_pd(gg);
						__m128d mGR = _mm_load_pd(gr);
						__m128d mRR = _mm_load_pd(rr);
						bg += 2;
						bb += 2;
						br += 2;
						gg += 2;
						gr += 2;
						rr += 2;

						__m128d mTmp = _mm_mul_pd(mBG, _mm_mul_pd(mGR, mBR));
						__m128d mDet = _mm_add_pd(mTmp, mTmp);
						mTmp = _mm_mul_pd(mBB, _mm_mul_pd(mGG, mRR));
						mDet = _mm_add_pd(mDet, mTmp);
						mTmp = _mm_mul_pd(mBB, _mm_mul_pd(mGR, mGR));
						mDet = _mm_sub_pd(mDet, mTmp);
						mTmp = _mm_mul_pd(mRR, _mm_mul_pd(mBG, mBG));
						mDet = _mm_sub_pd(mDet, mTmp);
						mTmp = _mm_mul_pd(mGG, _mm_mul_pd(mBR, mBR));
						mDet = _mm_sub_pd(mDet, mTmp);
						mDet = _mm_div_pd(_mm_set1_pd(1.0), mDet);

						__m128d mC0 = _mm_fmsub_pd(mGG, mRR, _mm_mul_pd(mGR, mGR));
						__m128d mC1 = _mm_fmsub_pd(mGR, mBR, _mm_mul_pd(mBG, mRR));
						__m128d mC2 = _mm_fmsub_pd(mBG, mGR, _mm_mul_pd(mBR, mGG));
						__m128d mC4 = _mm_fmsub_pd(mBB, mRR, _mm_mul_pd(mBR, mBR));
						__m128d mC5 = _mm_fmsub_pd(mBG, mBR, _mm_mul_pd(mBB, mGR));
						__m128d mC8 = _mm_fmsub_pd(mBB, mGG, _mm_mul_pd(mBG, mBG));

						__m128d mCovB = _mm_load_pd(covb);
						__m128d mCovG = _mm_load_pd(covg);
						__m128d mCovR = _mm_load_pd(covr);
						covb += 2;
						covg += 2;
						covr += 2;

						mTmp = _mm_fmadd_pd(mCovB, mC0, _mm_mul_pd(mCovG, mC1));
						mTmp = _mm_fmadd_pd(mCovR, mC2, mTmp);
						mTmp = _mm_mul_pd(mTmp, mDet);
						_mm_store_pd(ab, mTmp);
						ab += 2;

						mTmp = _mm_fmadd_pd(mCovB, mC1, _mm_mul_pd(mCovG, mC4));
						mTmp = _mm_fmadd_pd(mCovR, mC5, mTmp);
						mTmp = _mm_mul_pd(mTmp, mDet);
						_mm_store_pd(ag, mTmp);
						ag += 2;

						mTmp = _mm_fmadd_pd(mCovB, mC2, _mm_mul_pd(mCovG, mC5));
						mTmp = _mm_fmadd_pd(mCovR, mC8, mTmp);
						mTmp = _mm_mul_pd(mTmp, mDet);
						_mm_store_pd(ar, mTmp);
						ar += 2;
					}
				}
			}
			else if (implementation == AVX)
			{
				for (int i = 0; i < height; i++)
				{
					double* bb = var_I_bb.ptr<double>(i);
					double* bg = var_I_bg.ptr<double>(i);
					double* br = var_I_br.ptr<double>(i);
					double* gg = var_I_gg.ptr<double>(i);
					double* gr = var_I_gr.ptr<double>(i);
					double* rr = var_I_rr.ptr<double>(i);
					double* covb = cov_Ip_b.ptr<double>(i);
					double* covg = cov_Ip_g.ptr<double>(i);
					double* covr = cov_Ip_r.ptr<double>(i);
					double* ab = a_b.ptr<double>(i);
					double* ag = a_g.ptr<double>(i);
					double* ar = a_r.ptr<double>(i);

					for (int j = 0; j < width; j += 4)
					{
						__m256d mBB = _mm256_load_pd(bb);
						__m256d mBG = _mm256_load_pd(bg);
						__m256d mBR = _mm256_load_pd(br);
						__m256d mGG = _mm256_load_pd(gg);
						__m256d mGR = _mm256_load_pd(gr);
						__m256d mRR = _mm256_load_pd(rr);
						bb += 4;
						bg += 4;
						br += 4;
						gg += 4;
						gr += 4;
						rr += 4;

						__m256d mTmp = _mm256_mul_pd(mBG, _mm256_mul_pd(mGR, mBR));
						__m256d mDet = _mm256_add_pd(mTmp, mTmp);
						mDet = _mm256_fmadd_pd(mBB, _mm256_mul_pd(mGG, mRR), mDet);
						mDet = _mm256_fnmadd_pd(mBB, _mm256_mul_pd(mGR, mGR), mDet);
						mDet = _mm256_fnmadd_pd(mRR, _mm256_mul_pd(mBG, mBG), mDet);
						mDet = _mm256_fnmadd_pd(mGG, _mm256_mul_pd(mBR, mBR), mDet);
						mDet = _mm256_div_pd(_mm256_set1_pd(1.0), mDet);

						__m256d mC0 = _mm256_fmsub_pd(mGG, mRR, _mm256_mul_pd(mGR, mGR));
						__m256d mC1 = _mm256_fmsub_pd(mGR, mBR, _mm256_mul_pd(mBG, mRR));
						__m256d mC2 = _mm256_fmsub_pd(mBG, mGR, _mm256_mul_pd(mBR, mGG));
						__m256d mC4 = _mm256_fmsub_pd(mBB, mRR, _mm256_mul_pd(mBR, mBR));
						__m256d mC5 = _mm256_fmsub_pd(mBG, mBR, _mm256_mul_pd(mBB, mGR));
						__m256d mC8 = _mm256_fmsub_pd(mBB, mGG, _mm256_mul_pd(mBG, mBG));

						__m256d mCovB = _mm256_load_pd(covb);
						__m256d mCovG = _mm256_load_pd(covg);
						__m256d mCovR = _mm256_load_pd(covr);
						covb += 4;
						covg += 4;
						covr += 4;

						mTmp = _mm256_fmadd_pd(mCovB, mC0, _mm256_mul_pd(mCovG, mC1));
						mTmp = _mm256_fmadd_pd(mCovR, mC2, mTmp);
						mTmp = _mm256_mul_pd(mTmp, mDet);
						_mm256_store_pd(ab, mTmp);
						ab += 4;

						mTmp = _mm256_fmadd_pd(mCovB, mC1, _mm256_mul_pd(mCovG, mC4));
						mTmp = _mm256_fmadd_pd(mCovR, mC5, mTmp);
						mTmp = _mm256_mul_pd(mTmp, mDet);
						_mm256_store_pd(ag, mTmp);
						ag += 4;

						mTmp = _mm256_fmadd_pd(mCovB, mC2, _mm256_mul_pd(mCovG, mC5));
						mTmp = _mm256_fmadd_pd(mCovR, mC8, mTmp);
						mTmp = _mm256_mul_pd(mTmp, mDet);
						_mm256_store_pd(ar, mTmp);
						ar += 4;
					}
				}
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					double* bb = var_I_bb.ptr<double>(i);
					double* bg = var_I_bg.ptr<double>(i);
					double* br = var_I_br.ptr<double>(i);
					double* gg = var_I_gg.ptr<double>(i);
					double* gr = var_I_gr.ptr<double>(i);
					double* rr = var_I_rr.ptr<double>(i);
					double* covb = cov_Ip_b.ptr<double>(i);
					double* covg = cov_Ip_g.ptr<double>(i);
					double* covr = cov_Ip_r.ptr<double>(i);
					double* ab = a_b.ptr<double>(i);
					double* ag = a_g.ptr<double>(i);
					double* ar = a_r.ptr<double>(i);

					for (int j = 0; j < width; j++)
					{
						const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
							- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
						const double id = 1.0 / det;

						double c0 = *gg * *rr - *gr * *gr;
						double c1 = *gr * *br - *bg * *rr;
						double c2 = *bg * *gr - *br * *gg;
						double c4 = *bb * *rr - *br * *br;
						double c5 = *bg * *br - *bb * *gr;
						double c8 = *bb * *gg - *bg * *bg;

						*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
						*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
						*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);

						bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++;
					}
				}
			}
		}
		else
		{
			if (implementation == SSE)
			{
#pragma omp parallel for
				for (int i = 0; i < height; i++)
				{
					double* bb = var_I_bb.ptr<double>(i);
					double* bg = var_I_bg.ptr<double>(i);
					double* br = var_I_br.ptr<double>(i);
					double* gg = var_I_gg.ptr<double>(i);
					double* gr = var_I_gr.ptr<double>(i);
					double* rr = var_I_rr.ptr<double>(i);
					double* covb = cov_Ip_b.ptr<double>(i);
					double* covg = cov_Ip_g.ptr<double>(i);
					double* covr = cov_Ip_r.ptr<double>(i);
					double* ab = a_b.ptr<double>(i);
					double* ag = a_g.ptr<double>(i);
					double* ar = a_r.ptr<double>(i);

					for (int j = 0; j < width; j += 2)
					{
						__m128d mBB = _mm_load_pd(bb);
						__m128d mBG = _mm_load_pd(bg);
						__m128d mBR = _mm_load_pd(br);
						__m128d mGG = _mm_load_pd(gg);
						__m128d mGR = _mm_load_pd(gr);
						__m128d mRR = _mm_load_pd(rr);
						bg += 2;
						bb += 2;
						br += 2;
						gg += 2;
						gr += 2;
						rr += 2;

						__m128d mTmp = _mm_mul_pd(mBG, _mm_mul_pd(mGR, mBR));
						__m128d mDet = _mm_add_pd(mTmp, mTmp);
						mTmp = _mm_mul_pd(mBB, _mm_mul_pd(mGG, mRR));
						mDet = _mm_add_pd(mDet, mTmp);
						mTmp = _mm_mul_pd(mBB, _mm_mul_pd(mGR, mGR));
						mDet = _mm_sub_pd(mDet, mTmp);
						mTmp = _mm_mul_pd(mRR, _mm_mul_pd(mBG, mBG));
						mDet = _mm_sub_pd(mDet, mTmp);
						mTmp = _mm_mul_pd(mGG, _mm_mul_pd(mBR, mBR));
						mDet = _mm_sub_pd(mDet, mTmp);
						mDet = _mm_div_pd(_mm_set1_pd(1.0), mDet);

						__m128d mC0 = _mm_fmsub_pd(mGG, mRR, _mm_mul_pd(mGR, mGR));
						__m128d mC1 = _mm_fmsub_pd(mGR, mBR, _mm_mul_pd(mBG, mRR));
						__m128d mC2 = _mm_fmsub_pd(mBG, mGR, _mm_mul_pd(mBR, mGG));
						__m128d mC4 = _mm_fmsub_pd(mBB, mRR, _mm_mul_pd(mBR, mBR));
						__m128d mC5 = _mm_fmsub_pd(mBG, mBR, _mm_mul_pd(mBB, mGR));
						__m128d mC8 = _mm_fmsub_pd(mBB, mGG, _mm_mul_pd(mBG, mBG));

						__m128d mCovB = _mm_load_pd(covb);
						__m128d mCovG = _mm_load_pd(covg);
						__m128d mCovR = _mm_load_pd(covr);
						covb += 2;
						covg += 2;
						covr += 2;

						mTmp = _mm_fmadd_pd(mCovB, mC0, _mm_mul_pd(mCovG, mC1));
						mTmp = _mm_fmadd_pd(mCovR, mC2, mTmp);
						mTmp = _mm_mul_pd(mTmp, mDet);
						_mm_store_pd(ab, mTmp);
						ab += 2;

						mTmp = _mm_fmadd_pd(mCovB, mC1, _mm_mul_pd(mCovG, mC4));
						mTmp = _mm_fmadd_pd(mCovR, mC5, mTmp);
						mTmp = _mm_mul_pd(mTmp, mDet);
						_mm_store_pd(ag, mTmp);
						ag += 2;

						mTmp = _mm_fmadd_pd(mCovB, mC2, _mm_mul_pd(mCovG, mC5));
						mTmp = _mm_fmadd_pd(mCovR, mC8, mTmp);
						mTmp = _mm_mul_pd(mTmp, mDet);
						_mm_store_pd(ar, mTmp);
						ar += 2;
					}
				}
			}
			else if (implementation == AVX)
			{
#pragma omp parallel for
				for (int i = 0; i < height; i++)
				{
					double* bb = var_I_bb.ptr<double>(i);
					double* bg = var_I_bg.ptr<double>(i);
					double* br = var_I_br.ptr<double>(i);
					double* gg = var_I_gg.ptr<double>(i);
					double* gr = var_I_gr.ptr<double>(i);
					double* rr = var_I_rr.ptr<double>(i);
					double* covb = cov_Ip_b.ptr<double>(i);
					double* covg = cov_Ip_g.ptr<double>(i);
					double* covr = cov_Ip_r.ptr<double>(i);
					double* ab = a_b.ptr<double>(i);
					double* ag = a_g.ptr<double>(i);
					double* ar = a_r.ptr<double>(i);

					for (int j = 0; j < width; j += 4)
					{
						__m256d mBB = _mm256_load_pd(bb);
						__m256d mBG = _mm256_load_pd(bg);
						__m256d mBR = _mm256_load_pd(br);
						__m256d mGG = _mm256_load_pd(gg);
						__m256d mGR = _mm256_load_pd(gr);
						__m256d mRR = _mm256_load_pd(rr);
						bb += 4;
						bg += 4;
						br += 4;
						gg += 4;
						gr += 4;
						rr += 4;

						__m256d mTmp = _mm256_mul_pd(mBG, _mm256_mul_pd(mGR, mBR));
						__m256d mDet = _mm256_add_pd(mTmp, mTmp);
						mDet = _mm256_fmadd_pd(mBB, _mm256_mul_pd(mGG, mRR), mDet);
						mDet = _mm256_fnmadd_pd(mBB, _mm256_mul_pd(mGR, mGR), mDet);
						mDet = _mm256_fnmadd_pd(mRR, _mm256_mul_pd(mBG, mBG), mDet);
						mDet = _mm256_fnmadd_pd(mGG, _mm256_mul_pd(mBR, mBR), mDet);
						mDet = _mm256_div_pd(_mm256_set1_pd(1.0), mDet);

						__m256d mC0 = _mm256_fmsub_pd(mGG, mRR, _mm256_mul_pd(mGR, mGR));
						__m256d mC1 = _mm256_fmsub_pd(mGR, mBR, _mm256_mul_pd(mBG, mRR));
						__m256d mC2 = _mm256_fmsub_pd(mBG, mGR, _mm256_mul_pd(mBR, mGG));
						__m256d mC4 = _mm256_fmsub_pd(mBB, mRR, _mm256_mul_pd(mBR, mBR));
						__m256d mC5 = _mm256_fmsub_pd(mBG, mBR, _mm256_mul_pd(mBB, mGR));
						__m256d mC8 = _mm256_fmsub_pd(mBB, mGG, _mm256_mul_pd(mBG, mBG));

						__m256d mCovB = _mm256_load_pd(covb);
						__m256d mCovG = _mm256_load_pd(covg);
						__m256d mCovR = _mm256_load_pd(covr);
						covb += 4;
						covg += 4;
						covr += 4;

						mTmp = _mm256_fmadd_pd(mCovB, mC0, _mm256_mul_pd(mCovG, mC1));
						mTmp = _mm256_fmadd_pd(mCovR, mC2, mTmp);
						mTmp = _mm256_mul_pd(mTmp, mDet);
						_mm256_store_pd(ab, mTmp);
						ab += 4;

						mTmp = _mm256_fmadd_pd(mCovB, mC1, _mm256_mul_pd(mCovG, mC4));
						mTmp = _mm256_fmadd_pd(mCovR, mC5, mTmp);
						mTmp = _mm256_mul_pd(mTmp, mDet);
						_mm256_store_pd(ag, mTmp);
						ag += 4;

						mTmp = _mm256_fmadd_pd(mCovB, mC2, _mm256_mul_pd(mCovG, mC5));
						mTmp = _mm256_fmadd_pd(mCovR, mC8, mTmp);
						mTmp = _mm256_mul_pd(mTmp, mDet);
						_mm256_store_pd(ar, mTmp);
						ar += 4;
					}
				}
			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < height; i++)
				{
					double* bb = var_I_bb.ptr<double>(i);
					double* bg = var_I_bg.ptr<double>(i);
					double* br = var_I_br.ptr<double>(i);
					double* gg = var_I_gg.ptr<double>(i);
					double* gr = var_I_gr.ptr<double>(i);
					double* rr = var_I_rr.ptr<double>(i);
					double* covb = cov_Ip_b.ptr<double>(i);
					double* covg = cov_Ip_g.ptr<double>(i);
					double* covr = cov_Ip_r.ptr<double>(i);
					double* ab = a_b.ptr<double>(i);
					double* ag = a_g.ptr<double>(i);
					double* ar = a_r.ptr<double>(i);

					for (int j = 0; j < width; j++)
					{
						const double det = (*bb * *gg * *rr) + (*bg * *gr * *br) + (*br * *bg * *gr)
							- (*bb * *gr * *gr) - (*bg * *bg * *rr) - (*br * *gg * *br);
						const double id = 1.0 / det;

						double c0 = *gg * *rr - *gr * *gr;
						double c1 = *gr * *br - *bg * *rr;
						double c2 = *bg * *gr - *br * *gg;
						double c4 = *bb * *rr - *br * *br;
						double c5 = *bg * *br - *bb * *gr;
						double c8 = *bb * *gg - *bg * *bg;

						*ab = id * (*covb * c0 + *covg * c1 + *covr * c2);
						*ag = id * (*covb * c1 + *covg * c4 + *covr * c5);
						*ar = id * (*covb * c2 + *covg * c5 + *covr * c8);

						bb++, bg++, br++, gg++, gr++, rr++, covb++, covg++, covr++, ab++, ag++, ar++;
					}
				}
			}
		}
	}
}

void guidedFilter_Naive::ab_up_2q_Guide1(cv::Mat& guide, cv::Mat& output)
{
	average(a, a, r);
	average(b, b, r);

	resize(a, output, guide.size(), 0, 0, upsample_method);
	resize(b, temp_high, guide.size(), 0, 0, upsample_method);

	//multiply(output, guide, output);
	//add(output, temp_high, output);
	fmadd(output, guide, temp_high, output);
}

void guidedFilter_Naive::filterFast_Guide1(cv::Mat& input_low, cv::Mat& guide, cv::Mat& guide_low, cv::Mat& output)
{
	const Size imsize = input_low.size();
	const int imtype = input_low.depth();

	mean_I.create(imsize, imtype);
	mean_p.create(imsize, imtype);
	corr_I.create(imsize, imtype);
	corr_Ip.create(imsize, imtype);
	var_I.create(imsize, imtype);
	cov_Ip.create(imsize, imtype);
	a.create(imsize, imtype);
	b.create(imsize, imtype);

	Ip2ab_Guide1(input_low, guide_low);
	ab_up_2q_Guide1(guide, output);
}

void guidedFilter_Naive::Ip2ab_Guide3(cv::Mat& input, std::vector<cv::Mat>& guide)
{
	Mat I_b = guide[0];
	Mat I_g = guide[1];
	Mat I_r = guide[2];

	average(I_b, mean_I_b, r);
	average(I_g, mean_I_g, r);
	average(I_r, mean_I_r, r);
	average(input, mean_p, r);

	multiply(I_b, I_b, corr_I_bb);
	average(corr_I_bb, corr_I_bb, r);
	multiply(I_b, I_g, corr_I_bg);
	average(corr_I_bg, corr_I_bg, r);
	multiply(I_b, I_r, corr_I_br);
	average(corr_I_br, corr_I_br, r);
	multiply(I_g, I_g, corr_I_gg);
	average(corr_I_gg, corr_I_gg, r);
	multiply(I_g, I_r, corr_I_gr);
	average(corr_I_gr, corr_I_gr, r);
	multiply(I_r, I_r, corr_I_rr);
	average(corr_I_rr, corr_I_rr, r);

	multiply(I_b, input, corr_Ip_b);
	average(corr_Ip_b, corr_Ip_b, r);
	multiply(I_g, input, corr_Ip_g);
	average(corr_Ip_g, corr_Ip_g, r);
	multiply(I_r, input, corr_Ip_r);
	average(corr_Ip_r, corr_Ip_r, r);

	/*multiply(mean_I_b, mean_I_b, temp);
	var_I_bb = corr_I_bb - temp;
	multiply(mean_I_b, mean_I_g, temp);
	var_I_bg = corr_I_bg - temp;
	multiply(mean_I_b, mean_I_r, temp);
	var_I_br = corr_I_br - temp;
	multiply(mean_I_g, mean_I_g, temp);
	var_I_gg = corr_I_gg - temp;
	multiply(mean_I_g, mean_I_r, temp);
	var_I_gr = corr_I_gr - temp;
	multiply(mean_I_r, mean_I_r, temp);
	var_I_rr = corr_I_rr - temp;*/
	fnmadd(mean_I_b, mean_I_b, corr_I_bb, var_I_bb);
	fnmadd(mean_I_b, mean_I_g, corr_I_bg, var_I_bg);
	fnmadd(mean_I_b, mean_I_r, corr_I_br, var_I_br);
	fnmadd(mean_I_g, mean_I_g, corr_I_gg, var_I_gg);
	fnmadd(mean_I_g, mean_I_r, corr_I_gr, var_I_gr);
	fnmadd(mean_I_r, mean_I_r, corr_I_rr, var_I_rr);

	add(var_I_bb, eps, var_I_bb, noArray(), var_I_bb.depth());//var_I_bb += eps;
	add(var_I_gg, eps, var_I_gg, noArray(), var_I_bb.depth());//var_I_gg += eps;
	add(var_I_rr, eps, var_I_rr, noArray(), var_I_bb.depth());//var_I_rr += eps;

	/*
	multiply(mean_I_b, mean_p, temp);
	cov_Ip_b = corr_Ip_b - temp;
	multiply(mean_I_g, mean_p, temp);
	cov_Ip_g = corr_Ip_g - temp;
	multiply(mean_I_r, mean_p, temp);
	cov_Ip_r = corr_Ip_r - temp;
	*/
	fnmadd(mean_I_b, mean_p, corr_Ip_b, cov_Ip_b);
	fnmadd(mean_I_g, mean_p, corr_Ip_g, cov_Ip_g);
	fnmadd(mean_I_r, mean_p, corr_Ip_r, cov_Ip_r);

	computeCovariance(input.depth());

	/*multiply(a_b, mean_I_b, mean_I_b);
	multiply(a_g, mean_I_g, mean_I_g);
	multiply(a_r, mean_I_r, mean_I_r);
	b = mean_p - (mean_I_b + mean_I_g + mean_I_r);*/
	fnmadd(a_b, mean_I_b, mean_p, b);
	fnmadd(a_g, mean_I_g, b, b);
	fnmadd(a_r, mean_I_r, b, b);
}

void guidedFilter_Naive::ab2q_Guide3(std::vector<cv::Mat>& guide, cv::Mat& output)
{
	average(a_b, a_b, r);
	average(a_g, a_g, r);
	average(a_r, a_r, r);
	average(b, b, r);

	fmadd(a_b, guide[0], b, output);
	fmadd(a_g, guide[1], output, output);
	fmadd(a_r, guide[2], output, output);
}

void guidedFilter_Naive::filter_Guide3(cv::Mat& input, std::vector<cv::Mat>& guide, cv::Mat& output)
{
	const Size imsize = input.size();
	const int imtype = input.depth();

	mean_p.create(imsize, imtype);
	mean_I_b.create(imsize, imtype);
	mean_I_g.create(imsize, imtype);
	mean_I_r.create(imsize, imtype);
	corr_I_bb.create(imsize, imtype);
	corr_I_bg.create(imsize, imtype);
	corr_I_br.create(imsize, imtype);
	corr_I_gg.create(imsize, imtype);
	corr_I_gr.create(imsize, imtype);
	corr_I_rr.create(imsize, imtype);
	corr_Ip_b.create(imsize, imtype);
	corr_Ip_g.create(imsize, imtype);
	corr_Ip_r.create(imsize, imtype);
	var_I_bb.create(imsize, imtype);
	var_I_bg.create(imsize, imtype);
	var_I_br.create(imsize, imtype);
	var_I_gg.create(imsize, imtype);
	var_I_gr.create(imsize, imtype);
	var_I_rr.create(imsize, imtype);
	cov_Ip_b.create(imsize, imtype);
	cov_Ip_g.create(imsize, imtype);
	cov_Ip_r.create(imsize, imtype);
	a_b.create(imsize, imtype);
	a_g.create(imsize, imtype);
	a_r.create(imsize, imtype);
	b.create(imsize, imtype);

	Ip2ab_Guide3(input, guide);
	ab2q_Guide3(guide, output);
}

void guidedFilter_Naive::ab_up_2q_Guide3(std::vector<cv::Mat>& guide, cv::Mat& output)
{
	average(a_b, a_b, r);
	average(a_g, a_g, r);
	average(a_r, a_r, r);
	average(b, b, r);


	/*
	vector<Mat> v(4);
	resize(mean_b, v[3], guide[0].size(), 0, 0, upsample_method);
	resize(mean_a_b, v[0], guide[0].size(), 0, 0, upsample_method);
	resize(mean_a_g, v[1], guide[0].size(), 0, 0, upsample_method);
	resize(mean_a_r, v[2], guide[0].size(), 0, 0, upsample_method);

	multiply(v[0], I_b, v[0]);
	multiply(v[1], I_g, v[1]);
	multiply(v[2], I_r, v[2]);

	output = v[0]+ v[1]+ v[2]+ v[3];
	*/

	resize(b, output, guide[0].size(), 0, 0, upsample_method);

	resize(a_b, temp_high, guide[0].size(), 0, 0, upsample_method);
	fmadd(temp_high, guide[0], output, output);
	resize(a_g, temp_high, guide[0].size(), 0, 0, upsample_method);
	fmadd(temp_high, guide[1], output, output);
	resize(a_r, temp_high, guide[0].size(), 0, 0, upsample_method);
	fmadd(temp_high, guide[2], output, output);
}

void guidedFilter_Naive::filterFast_Guide3(cv::Mat& input_low, std::vector<cv::Mat>& guide, std::vector<cv::Mat>& guide_low, cv::Mat& output)
{
	const Size imsize = input_low.size();
	const int imtype = input_low.depth();

	mean_p.create(imsize, imtype);
	mean_I_b.create(imsize, imtype);
	mean_I_g.create(imsize, imtype);
	mean_I_r.create(imsize, imtype);
	corr_I_bb.create(imsize, imtype);
	corr_I_bg.create(imsize, imtype);
	corr_I_br.create(imsize, imtype);
	corr_I_gg.create(imsize, imtype);
	corr_I_gr.create(imsize, imtype);
	corr_I_rr.create(imsize, imtype);
	corr_Ip_b.create(imsize, imtype);
	corr_Ip_g.create(imsize, imtype);
	corr_Ip_r.create(imsize, imtype);
	var_I_bb.create(imsize, imtype);
	var_I_bg.create(imsize, imtype);
	var_I_br.create(imsize, imtype);
	var_I_gg.create(imsize, imtype);
	var_I_gr.create(imsize, imtype);
	var_I_rr.create(imsize, imtype);
	cov_Ip_b.create(imsize, imtype);
	cov_Ip_g.create(imsize, imtype);
	cov_Ip_r.create(imsize, imtype);

	a_b.create(imsize, imtype);
	a_g.create(imsize, imtype);
	a_r.create(imsize, imtype);
	b.create(imsize, imtype);

	Ip2ab_Guide3(input_low, guide_low);
	ab_up_2q_Guide3(guide, output);
}


void guidedFilter_Naive_OnePass::filter()
{
	if (src.channels() == 1)
	{
		if (guide.channels() == 1)
		{
			filter_Guide1(src, dest);
		}
		else if (guide.channels() == 3)
		{
			filter_Guide3(src, dest);
		}
	}
	else if (src.channels() == 3)
	{
		vSrc.resize(3);
		vDest.resize(3);
		split(src, vSrc);
		vDest[0].create(src.size(), CV_32F);
		vDest[1].create(src.size(), CV_32F);
		vDest[2].create(src.size(), CV_32F);
		if (guide.channels() == 1)
		{
			filter_Guide1(vSrc[0], vDest[0]);
			filter_Guide1(vSrc[1], vDest[1]);
			filter_Guide1(vSrc[2], vDest[2]);
		}
		else if (guide.channels() == 3)
		{
			filter_Guide3(vSrc[0], vDest[0]);
			filter_Guide3(vSrc[1], vDest[1]);
			filter_Guide3(vSrc[2], vDest[2]);
		}
		merge(vDest, dest);
	}
}

void guidedFilter_Naive_OnePass::filter_Guide1(cv::Mat& input, cv::Mat& output)
{
	const Size imsize = input.size();
	const int imtype = CV_32F;
	const int pType = ParallelTypes::NAIVE;

	Mat II(imsize, imtype);
	Mat Ip(imsize, imtype);

	if (parallelType == NAIVE)
	{
		/*   1   */
		for (int i = 0; i < imsize.height; i++)
		{
			float* i_ptr = guide.ptr<float>(i);
			float* p_ptr = input.ptr<float>(i);
			float* ii_ptr = II.ptr<float>(i);
			float* ip_ptr = Ip.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mI = _mm256_load_ps(i_ptr);
				i_ptr += 8;
				__m256 mP = _mm256_load_ps(p_ptr);
				p_ptr += 8;

				_mm256_stream_ps(ii_ptr, _mm256_mul_ps(mI, mI));
				ii_ptr += 8;
				_mm256_stream_ps(ip_ptr, _mm256_mul_ps(mI, mP));
				ip_ptr += 8;
			}
		}

		Mat mean_I(imsize, imtype);
		Mat mean_p(imsize, imtype);
		Mat corr_I(imsize, imtype);
		Mat corr_Ip(imsize, imtype);

		boxFilter_OPSAT_AoS(guide, mean_I, r, pType).filter();
		boxFilter_OPSAT_AoS(input, mean_p, r, pType).filter();
		boxFilter_OPSAT_AoS(II, corr_I, r, pType).filter();
		boxFilter_OPSAT_AoS(Ip, corr_Ip, r, pType).filter();

		Mat a(imsize, imtype);
		Mat b(imsize, imtype);

		/*   2, 3   */
		for (int i = 0; i < imsize.height; i++)
		{
			float* meanI_ptr = mean_I.ptr<float>(i);
			float* meanP_ptr = mean_p.ptr<float>(i);
			float* corrI_ptr = corr_I.ptr<float>(i);
			float* corrIP_ptr = corr_Ip.ptr<float>(i);
			float* a_ptr = a.ptr<float>(i);
			float* b_ptr = b.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mMeanI = _mm256_load_ps(meanI_ptr);
				meanI_ptr += 8;
				__m256 mMeanP = _mm256_load_ps(meanP_ptr);
				meanP_ptr += 8;

				__m256 mB = _mm256_sub_ps(_mm256_load_ps(corrI_ptr), _mm256_mul_ps(mMeanI, mMeanI));	// var_I
				corrI_ptr += 8;
				mB = _mm256_add_ps(mB, _mm256_set1_ps(eps));	// var_I + eps
				__m256 mA = _mm256_sub_ps(_mm256_load_ps(corrIP_ptr), _mm256_mul_ps(mMeanI, mMeanP));	// cov_Ip
				corrIP_ptr += 8;

				mA = _mm256_div_ps(mA, mB);	// a = cov_Ip / (var_I + eps)
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));	// b = mean_p - a * mean_I

				_mm256_stream_ps(a_ptr, mA);
				a_ptr += 8;
				_mm256_stream_ps(b_ptr, mB);
				b_ptr += 8;
			}
		}

		Mat mean_a(imsize, imtype);
		Mat mean_b(imsize, imtype);

		/*   4   */
		boxFilter_OPSAT_AoS(a, mean_a, r, pType).filter();
		boxFilter_OPSAT_AoS(b, mean_b, r, pType).filter();

		/*   5   */
		for (int i = 0; i < imsize.height; i++)
		{
			float* meanA_ptr = mean_a.ptr<float>(i);
			float* meanB_ptr = mean_b.ptr<float>(i);
			float* i_ptr = guide.ptr<float>(i);
			float* q_ptr = output.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mTmp = _mm256_load_ps(meanA_ptr);
				meanA_ptr += 8;
				mTmp = _mm256_mul_ps(mTmp, _mm256_load_ps(i_ptr));
				i_ptr += 8;
				mTmp = _mm256_add_ps(mTmp, _mm256_load_ps(meanB_ptr));
				meanB_ptr += 8;

				_mm256_stream_ps(q_ptr, mTmp);
				q_ptr += 8;
			}
		}
	}
	else if (parallelType == OMP)
	{
		/*   1   */
#pragma omp parallel for
		for (int i = 0; i < imsize.height; i++)
		{
			float* i_ptr = guide.ptr<float>(i);
			float* p_ptr = input.ptr<float>(i);
			float* ii_ptr = II.ptr<float>(i);
			float* ip_ptr = Ip.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mI = _mm256_load_ps(i_ptr);
				i_ptr += 8;
				__m256 mP = _mm256_load_ps(p_ptr);
				p_ptr += 8;

				_mm256_stream_ps(ii_ptr, _mm256_mul_ps(mI, mI));
				ii_ptr += 8;
				_mm256_stream_ps(ip_ptr, _mm256_mul_ps(mI, mP));
				ip_ptr += 8;
			}
		}

		Mat mean_I(imsize, imtype);
		Mat mean_p(imsize, imtype);
		Mat corr_I(imsize, imtype);
		Mat corr_Ip(imsize, imtype);

#pragma omp parallel sections
		{
#pragma omp section
			boxFilter_OPSAT_AoS(guide, mean_I, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(input, mean_p, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(II, corr_I, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(Ip, corr_Ip, r, pType).filter();
		}

		Mat a(imsize, imtype);
		Mat b(imsize, imtype);

		/*   2, 3   */
#pragma omp parallel for
		for (int i = 0; i < imsize.height; i++)
		{
			float* meanI_ptr = mean_I.ptr<float>(i);
			float* meanP_ptr = mean_p.ptr<float>(i);
			float* corrI_ptr = corr_I.ptr<float>(i);
			float* corrIP_ptr = corr_Ip.ptr<float>(i);
			float* a_ptr = a.ptr<float>(i);
			float* b_ptr = b.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mMeanI = _mm256_load_ps(meanI_ptr);
				meanI_ptr += 8;
				__m256 mMeanP = _mm256_load_ps(meanP_ptr);
				meanP_ptr += 8;

				__m256 mB = _mm256_sub_ps(_mm256_load_ps(corrI_ptr), _mm256_mul_ps(mMeanI, mMeanI));	// var_I
				corrI_ptr += 8;
				mB = _mm256_add_ps(mB, _mm256_set1_ps(eps));	// var_I + eps
				__m256 mA = _mm256_sub_ps(_mm256_load_ps(corrIP_ptr), _mm256_mul_ps(mMeanI, mMeanP));	// cov_Ip
				corrIP_ptr += 8;

				mA = _mm256_div_ps(mA, mB);	// a = cov_Ip / (var_I + eps)
				mB = _mm256_sub_ps(mMeanP, _mm256_mul_ps(mA, mMeanI));	// b = mean_p - a * mean_I

				_mm256_stream_ps(a_ptr, mA);
				a_ptr += 8;
				_mm256_stream_ps(b_ptr, mB);
				b_ptr += 8;
			}
		}

		Mat mean_a(imsize, imtype);
		Mat mean_b(imsize, imtype);

		/*   4   */
#pragma omp parallel sections
		{
#pragma omp section
			boxFilter_OPSAT_AoS(a, mean_a, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(b, mean_b, r, pType).filter();
		}

		/*   5   */
#pragma omp parallel for
		for (int i = 0; i < imsize.height; i++)
		{
			float* meanA_ptr = mean_a.ptr<float>(i);
			float* meanB_ptr = mean_b.ptr<float>(i);
			float* i_ptr = guide.ptr<float>(i);
			float* q_ptr = output.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mTmp = _mm256_load_ps(meanA_ptr);
				meanA_ptr += 8;
				mTmp = _mm256_mul_ps(mTmp, _mm256_load_ps(i_ptr));
				i_ptr += 8;
				mTmp = _mm256_add_ps(mTmp, _mm256_load_ps(meanB_ptr));
				meanB_ptr += 8;

				_mm256_stream_ps(q_ptr, mTmp);
				q_ptr += 8;
			}
		}
	}
}

void guidedFilter_Naive_OnePass::filter_Guide3(cv::Mat& input, cv::Mat& output)
{
	const Size imsize = input.size();
	const int imtype = CV_32F;
	const int pType = ParallelTypes::NAIVE;

	vector<Mat> vI(3);
	Mat I_b(imsize, imtype);
	Mat I_g(imsize, imtype);
	Mat I_r(imsize, imtype);
	split(guide, vI);
	I_b = vI[0], I_g = vI[1], I_r = vI[2];

	Mat I_bb(imsize, imtype);
	Mat I_bg(imsize, imtype);
	Mat I_br(imsize, imtype);
	Mat I_gg(imsize, imtype);
	Mat I_gr(imsize, imtype);
	Mat I_rr(imsize, imtype);
	Mat Ip_b(imsize, imtype);
	Mat Ip_g(imsize, imtype);
	Mat Ip_r(imsize, imtype);

	if (parallelType == NAIVE)
	{
		/*   1   */
		for (int i = 0; i < imsize.height; i++)
		{
			float* ib_ptr = I_b.ptr<float>(i);
			float* ig_ptr = I_g.ptr<float>(i);
			float* ir_ptr = I_r.ptr<float>(i);
			float* p_ptr = input.ptr<float>(i);

			float* i_bb_ptr = I_bb.ptr<float>(i);
			float* i_bg_ptr = I_bg.ptr<float>(i);
			float* i_br_ptr = I_br.ptr<float>(i);
			float* i_gg_ptr = I_gg.ptr<float>(i);
			float* i_gr_ptr = I_gr.ptr<float>(i);
			float* i_rr_ptr = I_rr.ptr<float>(i);
			float* ip_b_ptr = Ip_b.ptr<float>(i);
			float* ip_g_ptr = Ip_g.ptr<float>(i);
			float* ip_r_ptr = Ip_r.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mI_b = _mm256_load_ps(ib_ptr);
				ib_ptr += 8;
				__m256 mI_g = _mm256_load_ps(ig_ptr);
				ig_ptr += 8;
				__m256 mI_r = _mm256_load_ps(ir_ptr);
				ir_ptr += 8;
				__m256 mP = _mm256_load_ps(p_ptr);
				p_ptr += 8;

				_mm256_stream_ps(i_bb_ptr, _mm256_mul_ps(mI_b, mI_b));
				i_bb_ptr += 8;
				_mm256_stream_ps(i_bg_ptr, _mm256_mul_ps(mI_b, mI_g));
				i_bg_ptr += 8;
				_mm256_stream_ps(i_br_ptr, _mm256_mul_ps(mI_b, mI_r));
				i_br_ptr += 8;
				_mm256_stream_ps(i_gg_ptr, _mm256_mul_ps(mI_g, mI_g));
				i_gg_ptr += 8;
				_mm256_stream_ps(i_gr_ptr, _mm256_mul_ps(mI_g, mI_r));
				i_gr_ptr += 8;
				_mm256_stream_ps(i_rr_ptr, _mm256_mul_ps(mI_r, mI_r));
				i_rr_ptr += 8;
				_mm256_stream_ps(ip_b_ptr, _mm256_mul_ps(mI_b, mP));
				ip_b_ptr += 8;
				_mm256_stream_ps(ip_g_ptr, _mm256_mul_ps(mI_g, mP));
				ip_g_ptr += 8;
				_mm256_stream_ps(ip_r_ptr, _mm256_mul_ps(mI_r, mP));
				ip_r_ptr += 8;
			}
		}

		Mat mean_p(imsize, imtype);
		Mat mean_I_b(imsize, imtype);
		Mat mean_I_g(imsize, imtype);
		Mat mean_I_r(imsize, imtype);
		Mat corr_I_bb(imsize, imtype);
		Mat corr_I_bg(imsize, imtype);
		Mat corr_I_br(imsize, imtype);
		Mat corr_I_gg(imsize, imtype);
		Mat corr_I_gr(imsize, imtype);
		Mat corr_I_rr(imsize, imtype);
		Mat corr_Ip_b(imsize, imtype);
		Mat corr_Ip_g(imsize, imtype);
		Mat corr_Ip_r(imsize, imtype);

		boxFilter_OPSAT_AoS(I_b, mean_I_b, r, pType).filter();
		boxFilter_OPSAT_AoS(I_g, mean_I_g, r, pType).filter();
		boxFilter_OPSAT_AoS(I_r, mean_I_r, r, pType).filter();
		boxFilter_OPSAT_AoS(input, mean_p, r, pType).filter();
		boxFilter_OPSAT_AoS(I_bb, corr_I_bb, r, pType).filter();
		boxFilter_OPSAT_AoS(I_bg, corr_I_bg, r, pType).filter();
		boxFilter_OPSAT_AoS(I_br, corr_I_br, r, pType).filter();
		boxFilter_OPSAT_AoS(I_gg, corr_I_gg, r, pType).filter();
		boxFilter_OPSAT_AoS(I_gr, corr_I_gr, r, pType).filter();
		boxFilter_OPSAT_AoS(I_rr, corr_I_rr, r, pType).filter();
		boxFilter_OPSAT_AoS(Ip_b, corr_Ip_b, r, pType).filter();
		boxFilter_OPSAT_AoS(Ip_g, corr_Ip_g, r, pType).filter();
		boxFilter_OPSAT_AoS(Ip_r, corr_Ip_r, r, pType).filter();

		Mat a_b(imsize, imtype);
		Mat a_g(imsize, imtype);
		Mat a_r(imsize, imtype);
		Mat b(imsize, imtype);

		/*   2, 3   */
		for (int i = 0; i < imsize.height; i++)
		{
			float* meanI_b_ptr = mean_I_b.ptr<float>(i);
			float* meanI_g_ptr = mean_I_g.ptr<float>(i);
			float* meanI_r_ptr = mean_I_r.ptr<float>(i);
			float* meanP_ptr = mean_p.ptr<float>(i);
			float* corrI_bb_ptr = corr_I_bb.ptr<float>(i);
			float* corrI_bg_ptr = corr_I_bg.ptr<float>(i);
			float* corrI_br_ptr = corr_I_br.ptr<float>(i);
			float* corrI_gg_ptr = corr_I_gg.ptr<float>(i);
			float* corrI_gr_ptr = corr_I_gr.ptr<float>(i);
			float* corrI_rr_ptr = corr_I_rr.ptr<float>(i);
			float* corrIp_b_ptr = corr_Ip_b.ptr<float>(i);
			float* corrIp_g_ptr = corr_Ip_g.ptr<float>(i);
			float* corrIp_r_ptr = corr_Ip_r.ptr<float>(i);

			float* a_b_ptr = a_b.ptr<float>(i);
			float* a_g_ptr = a_g.ptr<float>(i);
			float* a_r_ptr = a_r.ptr<float>(i);
			float* b_ptr = b.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mMeanI_b = _mm256_load_ps(meanI_b_ptr);
				meanI_b_ptr += 8;
				__m256 mMeanI_g = _mm256_load_ps(meanI_g_ptr);
				meanI_g_ptr += 8;
				__m256 mMeanI_r = _mm256_load_ps(meanI_r_ptr);
				meanI_r_ptr += 8;
				__m256 mMeanP = _mm256_load_ps(meanP_ptr);
				meanP_ptr += 8;

				__m256 mBB = _mm256_sub_ps(_mm256_load_ps(corrI_bb_ptr), _mm256_mul_ps(mMeanI_b, mMeanI_b));
				corrI_bb_ptr += 8;
				__m256 mBG = _mm256_sub_ps(_mm256_load_ps(corrI_bg_ptr), _mm256_mul_ps(mMeanI_b, mMeanI_g));
				corrI_bg_ptr += 8;
				__m256 mBR = _mm256_sub_ps(_mm256_load_ps(corrI_br_ptr), _mm256_mul_ps(mMeanI_b, mMeanI_r));
				corrI_br_ptr += 8;
				__m256 mGG = _mm256_sub_ps(_mm256_load_ps(corrI_gg_ptr), _mm256_mul_ps(mMeanI_g, mMeanI_g));
				corrI_gg_ptr += 8;
				__m256 mGR = _mm256_sub_ps(_mm256_load_ps(corrI_gr_ptr), _mm256_mul_ps(mMeanI_g, mMeanI_r));
				corrI_gr_ptr += 8;
				__m256 mRR = _mm256_sub_ps(_mm256_load_ps(corrI_rr_ptr), _mm256_mul_ps(mMeanI_r, mMeanI_r));
				corrI_rr_ptr += 8;
				__m256 mCovB = _mm256_sub_ps(_mm256_load_ps(corrIp_b_ptr), _mm256_mul_ps(mMeanI_b, mMeanP));
				corrIp_b_ptr += 8;
				__m256 mCovG = _mm256_sub_ps(_mm256_load_ps(corrIp_g_ptr), _mm256_mul_ps(mMeanI_g, mMeanP));
				corrIp_g_ptr += 8;
				__m256 mCovR = _mm256_sub_ps(_mm256_load_ps(corrIp_r_ptr), _mm256_mul_ps(mMeanI_r, mMeanP));
				corrIp_r_ptr += 8;

				mBB = _mm256_add_ps(mBB, _mm256_set1_ps(eps));
				mGG = _mm256_add_ps(mGG, _mm256_set1_ps(eps));
				mRR = _mm256_add_ps(mRR, _mm256_set1_ps(eps));

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mDet = _mm256_mul_ps(mBB, mC0);
				mDet = _mm256_fmadd_ps(mBG, mC1, mDet);
				mDet = _mm256_fmadd_ps(mBR, mC2, mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mA = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mA = _mm256_fmadd_ps(mCovR, mC2, mA);
				mA = _mm256_mul_ps(mA, mDet);
				mBB = _mm256_mul_ps(mA, mMeanI_b);
				_mm256_stream_ps(a_b_ptr, mA);
				a_b_ptr += 8;

				mA = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mA = _mm256_fmadd_ps(mCovR, mC5, mA);
				mA = _mm256_mul_ps(mA, mDet);
				mGG = _mm256_mul_ps(mA, mMeanI_g);
				_mm256_stream_ps(a_g_ptr, mA);
				a_g_ptr += 8;

				mA = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mA = _mm256_fmadd_ps(mCovR, mC8, mA);
				mA = _mm256_mul_ps(mA, mDet);
				mRR = _mm256_mul_ps(mA, mMeanI_r);
				_mm256_stream_ps(a_r_ptr, mA);
				a_r_ptr += 8;

				mMeanP = _mm256_sub_ps(mMeanP, mBB);
				mMeanP = _mm256_sub_ps(mMeanP, mGG);
				mMeanP = _mm256_sub_ps(mMeanP, mRR);
				_mm256_stream_ps(b_ptr, mMeanP);
				b_ptr += 8;
			}
		}

		Mat mean_a_b(imsize, imtype);
		Mat mean_a_g(imsize, imtype);
		Mat mean_a_r(imsize, imtype);
		Mat mean_b(imsize, imtype);

		/*   4   */
		boxFilter_OPSAT_AoS(a_b, mean_a_b, r, pType).filter();
		boxFilter_OPSAT_AoS(a_g, mean_a_g, r, pType).filter();
		boxFilter_OPSAT_AoS(a_r, mean_a_r, r, pType).filter();
		boxFilter_OPSAT_AoS(b, mean_b, r, pType).filter();

		/*   5   */
		for (int i = 0; i < imsize.height; i++)
		{
			float* ib_ptr = I_b.ptr<float>(i);
			float* ig_ptr = I_g.ptr<float>(i);
			float* ir_ptr = I_r.ptr<float>(i);
			float* meanA_b_ptr = mean_a_b.ptr<float>(i);
			float* meanA_g_ptr = mean_a_g.ptr<float>(i);
			float* meanA_r_ptr = mean_a_r.ptr<float>(i);
			float* meanB_ptr = mean_b.ptr<float>(i);

			float* q_ptr = output.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mTmp = _mm256_setzero_ps();
				mTmp = _mm256_fmadd_ps(_mm256_load_ps(meanA_b_ptr), _mm256_load_ps(ib_ptr), mTmp);
				meanA_b_ptr += 8;
				ib_ptr += 8;
				mTmp = _mm256_fmadd_ps(_mm256_load_ps(meanA_g_ptr), _mm256_load_ps(ig_ptr), mTmp);
				meanA_g_ptr += 8;
				ig_ptr += 8;
				mTmp = _mm256_fmadd_ps(_mm256_load_ps(meanA_r_ptr), _mm256_load_ps(ir_ptr), mTmp);
				meanA_r_ptr += 8;
				ir_ptr += 8;
				mTmp = _mm256_add_ps(mTmp, _mm256_load_ps(meanB_ptr));
				meanB_ptr += 8;
				_mm256_stream_ps(q_ptr, mTmp);
				q_ptr += 8;
			}
		}
	}
	else if (parallelType == OMP)
	{
		/*   1   */
#pragma omp parallel for
		for (int i = 0; i < imsize.height; i++)
		{
			float* ib_ptr = I_b.ptr<float>(i);
			float* ig_ptr = I_g.ptr<float>(i);
			float* ir_ptr = I_r.ptr<float>(i);
			float* p_ptr = input.ptr<float>(i);

			float* i_bb_ptr = I_bb.ptr<float>(i);
			float* i_bg_ptr = I_bg.ptr<float>(i);
			float* i_br_ptr = I_br.ptr<float>(i);
			float* i_gg_ptr = I_gg.ptr<float>(i);
			float* i_gr_ptr = I_gr.ptr<float>(i);
			float* i_rr_ptr = I_rr.ptr<float>(i);
			float* ip_b_ptr = Ip_b.ptr<float>(i);
			float* ip_g_ptr = Ip_g.ptr<float>(i);
			float* ip_r_ptr = Ip_r.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mI_b = _mm256_load_ps(ib_ptr);
				ib_ptr += 8;
				__m256 mI_g = _mm256_load_ps(ig_ptr);
				ig_ptr += 8;
				__m256 mI_r = _mm256_load_ps(ir_ptr);
				ir_ptr += 8;
				__m256 mP = _mm256_load_ps(p_ptr);
				p_ptr += 8;

				_mm256_stream_ps(i_bb_ptr, _mm256_mul_ps(mI_b, mI_b));
				i_bb_ptr += 8;
				_mm256_stream_ps(i_bg_ptr, _mm256_mul_ps(mI_b, mI_g));
				i_bg_ptr += 8;
				_mm256_stream_ps(i_br_ptr, _mm256_mul_ps(mI_b, mI_r));
				i_br_ptr += 8;
				_mm256_stream_ps(i_gg_ptr, _mm256_mul_ps(mI_g, mI_g));
				i_gg_ptr += 8;
				_mm256_stream_ps(i_gr_ptr, _mm256_mul_ps(mI_g, mI_r));
				i_gr_ptr += 8;
				_mm256_stream_ps(i_rr_ptr, _mm256_mul_ps(mI_r, mI_r));
				i_rr_ptr += 8;
				_mm256_stream_ps(ip_b_ptr, _mm256_mul_ps(mI_b, mP));
				ip_b_ptr += 8;
				_mm256_stream_ps(ip_g_ptr, _mm256_mul_ps(mI_g, mP));
				ip_g_ptr += 8;
				_mm256_stream_ps(ip_r_ptr, _mm256_mul_ps(mI_r, mP));
				ip_r_ptr += 8;
			}
		}

		Mat mean_p(imsize, imtype);
		Mat mean_I_b(imsize, imtype);
		Mat mean_I_g(imsize, imtype);
		Mat mean_I_r(imsize, imtype);
		Mat corr_I_bb(imsize, imtype);
		Mat corr_I_bg(imsize, imtype);
		Mat corr_I_br(imsize, imtype);
		Mat corr_I_gg(imsize, imtype);
		Mat corr_I_gr(imsize, imtype);
		Mat corr_I_rr(imsize, imtype);
		Mat corr_Ip_b(imsize, imtype);
		Mat corr_Ip_g(imsize, imtype);
		Mat corr_Ip_r(imsize, imtype);

#pragma omp parallel sections
		{
#pragma omp section
			boxFilter_OPSAT_AoS(I_b, mean_I_b, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_g, mean_I_g, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_r, mean_I_r, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(input, mean_p, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_bb, corr_I_bb, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_bg, corr_I_bg, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_br, corr_I_br, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_gg, corr_I_gg, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_gr, corr_I_gr, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(I_rr, corr_I_rr, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(Ip_b, corr_Ip_b, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(Ip_g, corr_Ip_g, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(Ip_r, corr_Ip_r, r, pType).filter();
		}

		Mat a_b(imsize, imtype);
		Mat a_g(imsize, imtype);
		Mat a_r(imsize, imtype);
		Mat b(imsize, imtype);

		/*   2, 3   */
#pragma omp parallel for
		for (int i = 0; i < imsize.height; i++)
		{
			float* meanI_b_ptr = mean_I_b.ptr<float>(i);
			float* meanI_g_ptr = mean_I_g.ptr<float>(i);
			float* meanI_r_ptr = mean_I_r.ptr<float>(i);
			float* meanP_ptr = mean_p.ptr<float>(i);
			float* corrI_bb_ptr = corr_I_bb.ptr<float>(i);
			float* corrI_bg_ptr = corr_I_bg.ptr<float>(i);
			float* corrI_br_ptr = corr_I_br.ptr<float>(i);
			float* corrI_gg_ptr = corr_I_gg.ptr<float>(i);
			float* corrI_gr_ptr = corr_I_gr.ptr<float>(i);
			float* corrI_rr_ptr = corr_I_rr.ptr<float>(i);
			float* corrIp_b_ptr = corr_Ip_b.ptr<float>(i);
			float* corrIp_g_ptr = corr_Ip_g.ptr<float>(i);
			float* corrIp_r_ptr = corr_Ip_r.ptr<float>(i);

			float* a_b_ptr = a_b.ptr<float>(i);
			float* a_g_ptr = a_g.ptr<float>(i);
			float* a_r_ptr = a_r.ptr<float>(i);
			float* b_ptr = b.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mMeanI_b = _mm256_load_ps(meanI_b_ptr);
				meanI_b_ptr += 8;
				__m256 mMeanI_g = _mm256_load_ps(meanI_g_ptr);
				meanI_g_ptr += 8;
				__m256 mMeanI_r = _mm256_load_ps(meanI_r_ptr);
				meanI_r_ptr += 8;
				__m256 mMeanP = _mm256_load_ps(meanP_ptr);
				meanP_ptr += 8;

				__m256 mBB = _mm256_sub_ps(_mm256_load_ps(corrI_bb_ptr), _mm256_mul_ps(mMeanI_b, mMeanI_b));
				corrI_bb_ptr += 8;
				__m256 mBG = _mm256_sub_ps(_mm256_load_ps(corrI_bg_ptr), _mm256_mul_ps(mMeanI_b, mMeanI_g));
				corrI_bg_ptr += 8;
				__m256 mBR = _mm256_sub_ps(_mm256_load_ps(corrI_br_ptr), _mm256_mul_ps(mMeanI_b, mMeanI_r));
				corrI_br_ptr += 8;
				__m256 mGG = _mm256_sub_ps(_mm256_load_ps(corrI_gg_ptr), _mm256_mul_ps(mMeanI_g, mMeanI_g));
				corrI_gg_ptr += 8;
				__m256 mGR = _mm256_sub_ps(_mm256_load_ps(corrI_gr_ptr), _mm256_mul_ps(mMeanI_g, mMeanI_r));
				corrI_gr_ptr += 8;
				__m256 mRR = _mm256_sub_ps(_mm256_load_ps(corrI_rr_ptr), _mm256_mul_ps(mMeanI_r, mMeanI_r));
				corrI_rr_ptr += 8;
				__m256 mCovB = _mm256_sub_ps(_mm256_load_ps(corrIp_b_ptr), _mm256_mul_ps(mMeanI_b, mMeanP));
				corrIp_b_ptr += 8;
				__m256 mCovG = _mm256_sub_ps(_mm256_load_ps(corrIp_g_ptr), _mm256_mul_ps(mMeanI_g, mMeanP));
				corrIp_g_ptr += 8;
				__m256 mCovR = _mm256_sub_ps(_mm256_load_ps(corrIp_r_ptr), _mm256_mul_ps(mMeanI_r, mMeanP));
				corrIp_r_ptr += 8;

				mBB = _mm256_add_ps(mBB, _mm256_set1_ps(eps));
				mGG = _mm256_add_ps(mGG, _mm256_set1_ps(eps));
				mRR = _mm256_add_ps(mRR, _mm256_set1_ps(eps));

				__m256 mC0 = _mm256_fmsub_ps(mGG, mRR, _mm256_mul_ps(mGR, mGR));
				__m256 mC1 = _mm256_fmsub_ps(mGR, mBR, _mm256_mul_ps(mBG, mRR));
				__m256 mC2 = _mm256_fmsub_ps(mBG, mGR, _mm256_mul_ps(mBR, mGG));
				__m256 mC4 = _mm256_fmsub_ps(mBB, mRR, _mm256_mul_ps(mBR, mBR));
				__m256 mC5 = _mm256_fmsub_ps(mBG, mBR, _mm256_mul_ps(mBB, mGR));
				__m256 mC8 = _mm256_fmsub_ps(mBB, mGG, _mm256_mul_ps(mBG, mBG));

				__m256 mDet = _mm256_mul_ps(mBB, mC0);
				mDet = _mm256_fmadd_ps(mBG, mC1, mDet);
				mDet = _mm256_fmadd_ps(mBR, mC2, mDet);
				mDet = _mm256_div_ps(_mm256_set1_ps(1.f), mDet);

				__m256 mA = _mm256_fmadd_ps(mCovB, mC0, _mm256_mul_ps(mCovG, mC1));
				mA = _mm256_fmadd_ps(mCovR, mC2, mA);
				mA = _mm256_mul_ps(mA, mDet);
				mBB = _mm256_mul_ps(mA, mMeanI_b);
				_mm256_stream_ps(a_b_ptr, mA);
				a_b_ptr += 8;

				mA = _mm256_fmadd_ps(mCovB, mC1, _mm256_mul_ps(mCovG, mC4));
				mA = _mm256_fmadd_ps(mCovR, mC5, mA);
				mA = _mm256_mul_ps(mA, mDet);
				mGG = _mm256_mul_ps(mA, mMeanI_g);
				_mm256_stream_ps(a_g_ptr, mA);
				a_g_ptr += 8;

				mA = _mm256_fmadd_ps(mCovB, mC2, _mm256_mul_ps(mCovG, mC5));
				mA = _mm256_fmadd_ps(mCovR, mC8, mA);
				mA = _mm256_mul_ps(mA, mDet);
				mRR = _mm256_mul_ps(mA, mMeanI_r);
				_mm256_stream_ps(a_r_ptr, mA);
				a_r_ptr += 8;

				mMeanP = _mm256_sub_ps(mMeanP, mBB);
				mMeanP = _mm256_sub_ps(mMeanP, mGG);
				mMeanP = _mm256_sub_ps(mMeanP, mRR);
				_mm256_stream_ps(b_ptr, mMeanP);
				b_ptr += 8;
			}
		}

		Mat mean_a_b(imsize, imtype);
		Mat mean_a_g(imsize, imtype);
		Mat mean_a_r(imsize, imtype);
		Mat mean_b(imsize, imtype);

		/*   4   */
#pragma omp parallel sections
		{
#pragma omp section
			boxFilter_OPSAT_AoS(a_b, mean_a_b, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(a_g, mean_a_g, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(a_r, mean_a_r, r, pType).filter();
#pragma omp section
			boxFilter_OPSAT_AoS(b, mean_b, r, pType).filter();
		}

		/*   5   */
#pragma omp parallel for
		for (int i = 0; i < imsize.height; i++)
		{
			float* ib_ptr = I_b.ptr<float>(i);
			float* ig_ptr = I_g.ptr<float>(i);
			float* ir_ptr = I_r.ptr<float>(i);
			float* meanA_b_ptr = mean_a_b.ptr<float>(i);
			float* meanA_g_ptr = mean_a_g.ptr<float>(i);
			float* meanA_r_ptr = mean_a_r.ptr<float>(i);
			float* meanB_ptr = mean_b.ptr<float>(i);

			float* q_ptr = output.ptr<float>(i);

			for (int j = 0; j < imsize.width; j += 8)
			{
				__m256 mTmp = _mm256_setzero_ps();
				mTmp = _mm256_fmadd_ps(_mm256_load_ps(meanA_b_ptr), _mm256_load_ps(ib_ptr), mTmp);
				meanA_b_ptr += 8;
				ib_ptr += 8;
				mTmp = _mm256_fmadd_ps(_mm256_load_ps(meanA_g_ptr), _mm256_load_ps(ig_ptr), mTmp);
				meanA_g_ptr += 8;
				ig_ptr += 8;
				mTmp = _mm256_fmadd_ps(_mm256_load_ps(meanA_r_ptr), _mm256_load_ps(ir_ptr), mTmp);
				meanA_r_ptr += 8;
				ir_ptr += 8;
				mTmp = _mm256_add_ps(mTmp, _mm256_load_ps(meanB_ptr));
				meanB_ptr += 8;
				_mm256_stream_ps(q_ptr, mTmp);
				q_ptr += 8;
			}
		}
	}
}
