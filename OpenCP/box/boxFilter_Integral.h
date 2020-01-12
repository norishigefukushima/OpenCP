#pragma once

#include "boxFilter_Base.h"

//integral image based filtering

class boxFilter_Integral_nonVec : public boxFilter_base
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
	boxFilter_Integral_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_Integral_SSE : public boxFilter_Integral_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Integral_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_Integral_nonVec(_src, _dest, _r, _parallelType) {}
};

class boxFilter_Integral_AVX : public boxFilter_Integral_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Integral_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_Integral_nonVec(_src, _dest, _r, _parallelType) {}
};
