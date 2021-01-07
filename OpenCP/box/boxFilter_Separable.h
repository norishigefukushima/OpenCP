#pragma once

#include "boxFilter.hpp"

//separable box filtering

class boxFilter_Separable_HV_nonVec : public cp::BoxFilterBase
{
protected:
	int ksize;
	cv::Mat copy;
	cv::Mat temp;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Separable_HV_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_Separable_HV_SSE : public boxFilter_Separable_HV_nonVec
{
private:
	__m128 mDiv;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Separable_HV_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_Separable_HV_AVX : public boxFilter_Separable_HV_nonVec
{
private:
	int cn;
	__m256 mDiv;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Separable_HV_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_Separable_VH_AVX : public cp::BoxFilterBase
{
private:
	int padded;
	int ksize;
	cv::Mat copy;
	cv::Mat temp;

	int cn;
	__m256 mDiv;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Separable_VH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};
