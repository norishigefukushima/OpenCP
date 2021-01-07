#pragma once

#include "boxFilter.hpp"

//one pass integral image

class boxFilter_Integral_OnePass : public cp::BoxFilterBase
{
protected:
	cv::Mat sum;
	cv::Mat ColSum;
	cv::Mat copy;
	int ksize;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Integral_OnePass(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_Integral_OnePass_8u : public cp::BoxFilterBase
{
protected:
	cv::Mat sum;
	cv::Mat copy;
	int ksize;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Integral_OnePass_8u(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};



class boxFilter_Integral_OnePass_Area : public cp::BoxFilterBase
{
protected:
	cv::Mat copy;
	int ksize;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_Integral_OnePass_Area(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};
