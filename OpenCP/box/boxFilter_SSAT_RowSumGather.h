#pragma once

#include "boxFilter_SSAT_HV.h"

#define _USE_GATHER_

/*
 * SSAT horizontal filtering  by SIMD
 */

//TODO: boxFilter_Summed_RowSumGather_SSE
//TODO: RowSumFilter_SSEgather

class boxFilter_SSAT_HV_RowSumGather_AVX
{
private:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;

	int r;
	int parallelType;

public:
	boxFilter_SSAT_HV_RowSumGather_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
};

struct RowSumFilter_HV_AVXgather : public RowSumFilter
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps(static_cast<float>(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_HV_AVXgather(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: RowSumFilter(_src, _dest, _r, _parallelType) {}
};



class boxFilter_SSAT_VH_RowSumGather_SSE
{
protected:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;

	int r;
	int parallelType;

public:
	boxFilter_SSAT_VH_RowSumGather_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
};

struct RowSumFilter_VH_SSEgather : public RowSumFilter
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps(static_cast<float>(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_VH_SSEgather(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: RowSumFilter(_src, _dest, _r, _parallelType) {}
};



class boxFilter_SSAT_VH_RowSumGather_AVX
{
private:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;

	int r;
	int parallelType;

public:
	boxFilter_SSAT_VH_RowSumGather_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
};

struct RowSumFilter_VH_AVXgather : public RowSumFilter
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps(static_cast<float>(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_VH_AVXgather(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: RowSumFilter(_src, _dest, _r, _parallelType) {}
};
