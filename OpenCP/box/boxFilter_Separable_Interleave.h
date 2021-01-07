#pragma once

#include "boxFilter.hpp"

class boxFilter_Separable_VHI_nonVec : public cp::BoxFilterBase
{
protected:
	int ksize;
	
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Separable_VHI_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	virtual void init();
};



class boxFilter_Separable_VHI_SSE : public boxFilter_Separable_VHI_nonVec
{
private:
	__m128 mDiv;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Separable_VHI_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void init() override;
};



class boxFilter_Separable_VHI_AVX : public boxFilter_Separable_VHI_nonVec
{
private:
	__m256 mDiv;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Separable_VHI_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void init() override;
};
