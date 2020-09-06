#pragma once

#include "guidedFilter_Merge_Share_Transpose.h"

class guidedFilter_Merge_Share_Transpose_Inverse_nonVec : public guidedFilter_Merge_Share_Base
{
protected:
	cv::Size t_size;
	std::vector<cv::Mat> temp_ab;

	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_Transpose_Inverse_nonVec(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Base(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE;
		t_size = cv::Size(src.rows, src.cols);
		init();
	}
	void init() override;
};

class guidedFilter_Merge_Share_Transpose_Inverse_SSE : public guidedFilter_Merge_Share_Transpose_Inverse_nonVec
{
private:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_Transpose_Inverse_SSE(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Transpose_Inverse_nonVec(_src, _guide, _dest, _r, _eps, _parallelType)
	{
		implementation = cp::GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_SSE;
		t_size = cv::Size(src.rows * 4, src.cols / 4);
		init();
	}
};

class guidedFilter_Merge_Share_Transpose_Inverse_AVX : public guidedFilter_Merge_Share_Transpose_Inverse_nonVec
{
private:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_Transpose_Inverse_AVX(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Transpose_Inverse_nonVec(_src, _guide, _dest, _r, _eps, _parallelType)
	{
		implementation = cp::GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_AVX;
		t_size = cv::Size(src.rows * 8, src.cols / 8);
		init();
	}
};



/* --- Compute Var --- */
struct RowSumFilter_Var_Transpose_Inverse_nonVec : public RowSumFilter_Var
{
protected:
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Var_Transpose_Inverse_nonVec(cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Var(_I, _tempVec, _r, _parallelType)
	{
		img_row = I.rows;
		img_col = I.cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Var_Transpose_Inverse_SSE : public RowSumFilter_Var_Transpose_Inverse_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Var_Transpose_Inverse_SSE(cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Var_Transpose_Inverse_nonVec(_I, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 3;
	}
};

struct RowSumFilter_Var_Transpose_Inverse_AVX : public RowSumFilter_Var_Transpose_Inverse_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Var_Transpose_Inverse_AVX(cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Var_Transpose_Inverse_nonVec(_I, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 7;
	}
};



/* --- Compute Cov --- */
struct RowSumFilter_Cov_Transpose_Inverse_nonVec : public RowSumFilter_Cov
{
protected:
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Cov_Transpose_Inverse_nonVec(std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Cov(_vI, _tempVec, _r, _parallelType)
	{
		img_row = vI[0].rows;
		img_col = vI[0].cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Cov_Transpose_Inverse_SSE : public RowSumFilter_Cov_Transpose_Inverse_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Cov_Transpose_Inverse_SSE(std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Cov_Transpose_Inverse_nonVec(_vI, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 3;
	}
};

struct RowSumFilter_Cov_Transpose_Inverse_AVX : public RowSumFilter_Cov_Transpose_Inverse_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Cov_Transpose_Inverse_AVX(std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Cov_Transpose_Inverse_nonVec(_vI, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 7;
	}
};



/* --- Guide1 --- */
struct ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec : public ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_SSE : public ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType) {};
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_AVX : public ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec
{
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_Inverse_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType) {};
};



/* --- Guide3 --- */
struct ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec : public ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_SSE : public ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType) {};
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_AVX : public ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_Inverse_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType) {};
};