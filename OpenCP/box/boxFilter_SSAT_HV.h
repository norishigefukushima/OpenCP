#pragma once

#include "boxFilter.hpp"

// SSAT: horizontal -> vertical

class boxFilter_SSAT_HV_nonVec
{
protected:
	cv::Mat src;
	cv::Mat dest;
	cv::Mat temp;
	int r;
	int parallelType;

public:
	boxFilter_SSAT_HV_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: src(_src), dest(_dest), r(_r), parallelType(_parallelType)
	{
		temp.create(src.size(), src.type());
	}
	virtual void filter();
};

class boxFilter_SSAT_HV_SSE : public boxFilter_SSAT_HV_nonVec
{
private:

public:
	boxFilter_SSAT_HV_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_HV_nonVec(_src, _dest, _r, _parallelType) {}
	void filter() override;
};

class boxFilter_SSAT_HV_AVX : public boxFilter_SSAT_HV_nonVec
{
private:

public:
	boxFilter_SSAT_HV_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_HV_nonVec(_src, _dest, _r, _parallelType) {}
	void filter() override;
};



struct RowSumFilter : public cp::BoxFilterBase
{
protected:
	int cn;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: BoxFilterBase(_src, _dest, _r, _parallelType)
	{
		cn = src.channels();
	}
};

struct RowSumFilter_SSE : public RowSumFilter
{
private:
	__m128 mBorder = _mm_set1_ps(static_cast<float>(r + 1));
	__m128 mDiv = _mm_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: RowSumFilter(_src, _dest, _r, _parallelType) {}
};

struct RowSumFilter_AVX : public RowSumFilter
{
protected:
	__m256 mBorder = _mm256_set1_ps(static_cast<float>(r + 1));
	__m256 mDiv = _mm256_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: RowSumFilter(_src, _dest, _r, _parallelType) {}
};

struct ColumnSumFilter_nonVec : public cp::BoxFilterBase
{
protected:
	int cn;
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: BoxFilterBase(_src, _dest, _r, _parallelType)
	{
		cn = src.channels();
		step = col * cn;
	}
	void filter() override;
};

struct ColumnSumFilter_SSE : public ColumnSumFilter_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: ColumnSumFilter_nonVec(_src, _dest, _r, _parallelType) {}
};

struct ColumnSumFilter_AVX : public ColumnSumFilter_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: ColumnSumFilter_nonVec(_src, _dest, _r, _parallelType) {}
};





class boxFilter_SSAT_Channel_nonVec : public boxFilter_SSAT_HV_nonVec
{
private:

public:
	boxFilter_SSAT_Channel_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_HV_nonVec(_src, _dest, _r, _parallelType) {}
	void filter() override;
};

class boxFilter_SSAT_Channel_SSE : public boxFilter_SSAT_HV_nonVec
{
private:

public:
	boxFilter_SSAT_Channel_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_HV_nonVec(_src, _dest, _r, _parallelType) {}
	void filter() override;
};

class boxFilter_SSAT_Channel_AVX : public boxFilter_SSAT_HV_nonVec
{
private:

public:
	boxFilter_SSAT_Channel_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_HV_nonVec(_src, _dest, _r, _parallelType) {}
	void filter() override;
};



struct RowSumFilter_CN : public cp::BoxFilterBase
{
protected:
	int cn;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_CN(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: BoxFilterBase(_src, _dest, _r, _parallelType)
	{
		cn = src.channels();
	}
};

struct RowSumFilter_SSE_CN : public RowSumFilter_CN
{
private:
	__m128 mBorder = _mm_set1_ps(static_cast<float>(r + 1));
	__m128 mDiv = _mm_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_SSE_CN(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: RowSumFilter_CN(_src, _dest, _r, _parallelType) {}
};

struct RowSumFilter_AVX_CN : public RowSumFilter_CN
{
protected:
	__m256 mBorder = _mm256_set1_ps(static_cast<float>(r + 1));
	__m256 mDiv = _mm256_set1_ps(div);

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_AVX_CN(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: RowSumFilter_CN(_src, _dest, _r, _parallelType) {}
};



/*
 * uchar implement
 */
class boxFilter_SSAT_8u_nonVec : public boxFilter_SSAT_HV_nonVec
{
protected:

public:
	boxFilter_SSAT_8u_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_HV_nonVec(_src, _dest, _r, _parallelType)
	{
		temp.create(src.size(), CV_32S);
	}
	void filter() override;
};



struct RowSumFilter_8u : public RowSumFilter
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_8u(cv::Mat& _src, cv::Mat& _temp, int _r, int _parallelType)
		: RowSumFilter(_src, _temp, _r, _parallelType) {};
};

struct ColumnSumFilter_8u_nonVec : public ColumnSumFilter_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_8u_nonVec(cv::Mat& _temp, cv::Mat& _dest, int _r, int _parallelType)
		: ColumnSumFilter_nonVec(_temp, _dest, _r, _parallelType) {};
};
