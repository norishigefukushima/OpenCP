#pragma once

#include "boxFilter.hpp"

//integral image based filtering

class boxFilterIntegralScalar : public cp::BoxFilterBase
{
protected:
	int ksize;
	int cn;
	cv::Mat copy;
	cv::Mat sum;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilterIntegralScalar(cv::Mat& src, cv::Mat& dest, int _r, int _parallelType);
};

class boxFilterIntegralSSE : public boxFilterIntegralScalar
{
private:
	const __m128 mDiv = _mm_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilterIntegralSSE(cv::Mat& src, cv::Mat& dest, int _r, int _parallelType)
		: boxFilterIntegralScalar(src, dest, _r, _parallelType) {}
};

class boxFilterIntegralAVX : public boxFilterIntegralScalar
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilterIntegralAVX(cv::Mat& src, cv::Mat& dest, int _r, int _parallelType)
		: boxFilterIntegralScalar(src, dest, _r, _parallelType) {}
};
