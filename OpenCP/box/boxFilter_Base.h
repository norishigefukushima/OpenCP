#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "parallel_type.hpp"

#define BOX_FILTER_BORDER_TYPE cv::BORDER_REPLICATE

class boxFilter_base : public cv::ParallelLoopBody
{
protected:
	cv::Mat& src;
	cv::Mat& dest;
	int r;
	int parallelType;

	float div;
	int row;
	int col;

	virtual void filter_naive_impl() = 0;
	virtual void filter_omp_impl() = 0;
	void operator()(const cv::Range& range) const override = 0;
public:
	boxFilter_base(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType) : src(_src), dest(_dest), r(_r), parallelType(_parallelType)
	{
		div = 1.f / ((2 * r + 1)*(2 * r + 1));
		row = src.rows;
		col = src.cols;
	};
	virtual void filter();
};

inline void boxFilter_base::filter()
{
	if (parallelType == ParallelTypes::NAIVE)
	{
		filter_naive_impl();
	}
	else if (parallelType == ParallelTypes::OMP)
	{
		filter_omp_impl();
	}
	else if (parallelType == ParallelTypes::PARALLEL_FOR_)
	{
		cv::parallel_for_(cv::Range(0, row), *this, cv::getNumThreads() - 1);
	}
	else
	{

	}
}