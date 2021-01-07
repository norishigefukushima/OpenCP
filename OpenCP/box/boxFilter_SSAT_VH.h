#pragma once

#include "boxFilter.hpp"

/*
 * SSATÅiêÇíº -> êÖïΩÅj
 */

class boxFilter_SSAT_VH_nonVec
{
protected:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;
	int r;
	int parallelType;

public:
	boxFilter_SSAT_VH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	virtual void filter();
};

class boxFilter_SSAT_VH_SSE : public boxFilter_SSAT_VH_nonVec
{
private:

public:
	boxFilter_SSAT_VH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_VH_nonVec(_src, _dest, _r, _parallelType) {}
	void filter() override;
};

class boxFilter_SSAT_VH_AVX : public boxFilter_SSAT_VH_nonVec
{
private:

public:
	boxFilter_SSAT_VH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_VH_nonVec(_src, _dest, _r, _parallelType) {}
	void filter() override;
};



class RowSumFilter_VH : public cp::BoxFilterBase
{
private:
	int cn;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_VH(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: BoxFilterBase(_src, _dest, _r, _parallelType)
	{
		cn = src.channels();
	}
};

class ColumnSumFilter_VH_nonVec : public cp::BoxFilterBase
{
protected:
	int cn;
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_VH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: BoxFilterBase(_src, _dest, _r, _parallelType)
	{
		cn = src.channels();
		step = col * cn;
	}
	void filter() override;
};

class ColumnSumFilter_VH_SSE : public ColumnSumFilter_VH_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_VH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: ColumnSumFilter_VH_nonVec(_src, _dest, _r, _parallelType) {}
};

class ColumnSumFilter_VH_AVX : public ColumnSumFilter_VH_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_VH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: ColumnSumFilter_VH_nonVec(_src, _dest, _r, _parallelType) {}
};
