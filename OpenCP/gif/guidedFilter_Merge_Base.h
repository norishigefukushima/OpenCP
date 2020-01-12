#pragma once

#include "guidedFilter.hpp"

struct GuidedFilter_Merge_base : public cv::ParallelLoopBody
{
protected:
	int r;
	int img_row = 0;
	int img_col = 0;

	std::vector<cv::Mat>& tempVec;

	int parallelType;

	virtual void filter_naive_impl() = 0;
	virtual void filter_omp_impl() = 0;
	void operator()(const cv::Range& range) const override = 0;
public:
	GuidedFilter_Merge_base(std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: tempVec(_tempVec), r(_r), parallelType(_parallelType) {};
	virtual void filter() = 0;
};

struct RowSumFilter_base : public GuidedFilter_Merge_base
{
protected:

public:
	RowSumFilter_base(std::vector<cv::Mat>& _outputVec, int _r, int _parallelType)
		: GuidedFilter_Merge_base(_outputVec, _r, _parallelType) {};
	void filter() override;
};

//struct RowSumFilter_base : public cv::ParallelLoopBody
//{
//protected:
//	int r;
//	int img_row = 0;
//	int img_col = 0;
//
//	std::vector<cv::Mat>& outputVec;
//
//	int parallelType;
//
//	virtual void filter_naive_impl() = 0;
//	virtual void filter_omp_impl() = 0;
//	void operator()(const cv::Range& range) const override = 0;
//public:
//	RowSumFilter_base(std::vector<cv::Mat>& _outputVec, int _r, int _parallelType) : outputVec(_outputVec), r(_r), parallelType(_parallelType) {};
//	virtual void filter();
//};

inline void RowSumFilter_base::filter()
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
		cv::parallel_for_(cv::Range(0, img_row), *this, cv::getNumThreads() - 1);
	}
	else
	{
		
	}
}

struct ColumnSumFilter_base : public GuidedFilter_Merge_base
{
protected:
	int step = 0;
	float div;
public:
	ColumnSumFilter_base(std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: GuidedFilter_Merge_base(_tempVec, _r, _parallelType)
	{
		div = 1.f / ((2 * r + 1)*(2 * r + 1));
	};
	void filter() override;
};

//struct ColumnSumFilter_base : public cv::ParallelLoopBody
//{
//protected:
//	int r;
//	int img_row = 0;
//	int img_col = 0;
//
//	std::vector<cv::Mat>& inputVec;
//
//	int parallelType;
//
//	virtual void filter_naive_impl() = 0;
//	virtual void filter_omp_impl() = 0;
//	void operator()(const cv::Range& range) const override = 0;
//public:
//	ColumnSumFilter_base(std::vector<cv::Mat>& _inputVec, int _r, int _parallelType) : inputVec(_inputVec), r(_r), parallelType(_parallelType) {};
//	virtual void filter();
//};

inline void ColumnSumFilter_base::filter()
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
		cv::parallel_for_(cv::Range(0, img_col), *this, cv::getNumThreads() - 1);
	}
	else
	{

	}
}
