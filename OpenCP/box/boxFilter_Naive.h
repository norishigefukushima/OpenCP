#pragma once

#include "boxFilter.hpp"

//Naive implementation

class boxFilter_Naive_nonVec_Gray : public cp::BoxFilterBase
{
protected:
	int ksize;
	cv::Mat copy;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Naive_nonVec_Gray(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_Naive_SSE_Gray : public boxFilter_Naive_nonVec_Gray
{
private:
	const __m128 mDiv = _mm_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Naive_SSE_Gray(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_Naive_nonVec_Gray(_src, _dest, _r, _parallelType) {}
};

class boxFilter_Naive_AVX_Gray : public boxFilter_Naive_nonVec_Gray
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Naive_AVX_Gray(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		:boxFilter_Naive_nonVec_Gray(_src, _dest, _r, _parallelType) {}
};

class boxFilter_Naive_nonVec_Color : public cp::BoxFilterBase
{
protected:
	int ksize;
	cv::Mat copy;
	std::vector<cv::Mat> vCopy;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Naive_nonVec_Color(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_Naive_SSE_Color : public boxFilter_Naive_nonVec_Color
{
private:
	const __m128 mDiv = _mm_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Naive_SSE_Color(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_Naive_nonVec_Color(_src, _dest, _r, _parallelType) {}
};

class boxFilter_Naive_AVX_Color : public boxFilter_Naive_nonVec_Color
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256i mask_b = _mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0);
	const __m256i mask_g = _mm256_set_epi32(2, 7, 4, 1, 6, 3, 0, 5);
	const __m256i mask_r = _mm256_set_epi32(7, 4, 1, 6, 3, 0, 5, 2);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Naive_AVX_Color(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_Naive_nonVec_Color(_src, _dest, _r, _parallelType) {}
};