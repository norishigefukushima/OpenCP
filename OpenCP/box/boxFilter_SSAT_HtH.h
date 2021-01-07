#pragma once

#include "boxFilter.hpp"

//SSAT:horizontal ->transpose -> horizontal

class boxFilter_SSAT_HtH_nonVec
{
private:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;
	int r;
	int parallelType;

public:
	boxFilter_SSAT_HtH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
};

class boxFilter_SSAT_HtH_SSE
{
private:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;
	int r;
	int parallelType;

public:
	boxFilter_SSAT_HtH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
};

class boxFilter_SSAT_HtH_AVX
{
private:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;
	int r;
	int parallelType;

public:
	boxFilter_SSAT_HtH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
};



class RowSumFilter_HtH_nonVec : public cp::BoxFilterBase
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_HtH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter() override;
};

class RowSumFilter_HtH_SSE : public RowSumFilter_HtH_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_HtH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class RowSumFilter_HtH_AVX : public RowSumFilter_HtH_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_HtH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class ColumnSumFilter_HtH_nonVec : public RowSumFilter_HtH_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_HtH_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class ColumnSumFilter_HtH_SSE : public RowSumFilter_HtH_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_HtH_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class ColumnSumFilter_HtH_AVX : public RowSumFilter_HtH_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_HtH_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};
