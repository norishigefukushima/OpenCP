#pragma once

#include "guidedFilter_Merge_Share.h"

class guidedFilter_Merge_Share_Transpose_nonVec : public guidedFilter_Merge_Share_Base
{
protected:
	cv::Size t_size;
	cv::Mat I_t;
	std::vector<cv::Mat> vI_t;

	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_Transpose_nonVec(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType, const bool isInit = true)
		: guidedFilter_Merge_Share_Base(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_SHARE_TRANSPOSE;

		t_size = cv::Size(src.rows, src.cols);
		if (isInit)init();
	}
	void init();
};

class guidedFilter_Merge_Share_Transpose_SSE : public guidedFilter_Merge_Share_Transpose_nonVec
{
private:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_Transpose_SSE(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Transpose_nonVec(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_SHARE_TRANSPOSE_SSE;
		t_size = cv::Size(src.rows * 4, src.cols / 4);
		init();
	}

};

class guidedFilter_Merge_Share_Transpose_AVX : public guidedFilter_Merge_Share_Transpose_nonVec
{
private:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_Transpose_AVX(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Transpose_nonVec(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_SHARE_TRANSPOSE_AVX;
		t_size = cv::Size(src.rows * 8, src.cols / 8);
		init();
	}
};



/* --- Compute Var --- */
struct RowSumFilter_Var_Transpose_nonVec : public RowSumFilter_Var
{
protected:
	int step;
	cv::Mat& I_t;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Var_Transpose_nonVec(cv::Mat& _I, std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, int _r, int _parallelType)
		: I_t(_I_t), RowSumFilter_Var(_I, _tempVec, _r, _parallelType)
	{
		img_row = I.rows;
		img_col = I.cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Var_Transpose_SSE : public RowSumFilter_Var_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Var_Transpose_SSE(cv::Mat& _I, std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, int _r, int _parallelType)
		: RowSumFilter_Var_Transpose_nonVec(_I, _tempVec, _I_t, _r, _parallelType)
	{
		step = I.rows * 4 - 3;
	}
};

struct RowSumFilter_Var_Transpose_AVX : public RowSumFilter_Var_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Var_Transpose_AVX(cv::Mat& _I, std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, int _r, int _parallelType)
		: RowSumFilter_Var_Transpose_nonVec(_I, _tempVec, _I_t, _r, _parallelType)
	{
		step = I.rows * 8 - 7;
	}
};

struct ColumnSumFilter_Var_Transpose_nonVec : public ColumnSumFilter_Var_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Var_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Var_nonVec(_tempVec, _var, _mean_I, _r, _eps, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
	}
};

struct ColumnSumFilter_Var_Transpose_SSE : public ColumnSumFilter_Var_Transpose_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Var_Transpose_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Var_Transpose_nonVec(_tempVec, _var, _mean_I, _r, _eps, _parallelType) {}
};

struct ColumnSumFilter_Var_Transpose_AVX : public ColumnSumFilter_Var_Transpose_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Var_Transpose_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Var_Transpose_nonVec(_tempVec, _var, _mean_I, _r, _eps, _parallelType) {}
};



/* --- Compute Cov --- */
struct RowSumFilter_Cov_Transpose_nonVec : public RowSumFilter_Cov
{
protected:
	int step;
	std::vector<cv::Mat>& vI_t;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Cov_Transpose_nonVec(std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI_t, int _r, int _parallelType)
		: vI_t(_vI_t), RowSumFilter_Cov(_vI, _tempVec, _r, _parallelType)
	{
		img_row = vI[0].rows;
		img_col = vI[0].cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Cov_Transpose_SSE : public RowSumFilter_Cov_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Cov_Transpose_SSE(std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI_t, int _r, int _parallelType)
		: RowSumFilter_Cov_Transpose_nonVec(_vI, _tempVec, _vI_t, _r, _parallelType)
	{
		step = tempVec[0].cols - 3;
	}
};

struct RowSumFilter_Cov_Transpose_AVX : public RowSumFilter_Cov_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Cov_Transpose_AVX(std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI_t, int _r, int _parallelType)
		: RowSumFilter_Cov_Transpose_nonVec(_vI, _tempVec, _vI_t, _r, _parallelType)
	{
		step = tempVec[0].cols - 7;
	}
};

struct ColumnSumFilter_Cov_Transpose_nonVec : public ColumnSumFilter_Cov_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Cov_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Cov_nonVec(_tempVec, _vCov, _det, _vMean_I, _r, _eps, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
	}
};

struct ColumnSumFilter_Cov_Transpose_SSE : public ColumnSumFilter_Cov_Transpose_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Cov_Transpose_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Cov_Transpose_nonVec(_tempVec, _vCov, _det, _vMean_I, _r, _eps, _parallelType) {}
};

struct ColumnSumFilter_Cov_Transpose_AVX : public ColumnSumFilter_Cov_Transpose_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Cov_Transpose_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Cov_Transpose_nonVec(_tempVec, _vCov, _det, _vMean_I, _r, _eps, _parallelType) {}
};



/* --- Guide1 --- */
struct RowSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec : public RowSumFilter_Ip2ab_Guide1_Share
{
protected:
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide1_Share(_p, _I, _tempVec, _r, _parallelType)
	{
		img_row = p.rows;
		img_col = p.cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Ip2ab_Guide1_Share_Transpose_SSE : public RowSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide1_Share_Transpose_SSE(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec(_p, _I, _tempVec, _r, _parallelType)
	{
		step = p.rows * 4 - 3;
	}
};

struct RowSumFilter_Ip2ab_Guide1_Share_Transpose_AVX : public RowSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide1_Share_Transpose_AVX(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec(_p, _I, _tempVec, _r, _parallelType)
	{
		step = p.rows * 8 - 7;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec : ColumnSumFilter_Ip2ab_Guide1_Share_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
		step = a.cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_SSE : ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec
{
protected:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType) {}
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_AVX : ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec
{
protected:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_Transpose_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType) {}
};



/* --- Guide3 --- */
struct RowSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec : public RowSumFilter_Ip2ab_Guide3_Share
{
protected:
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide3_Share(_p, _vI, _tempVec, _r, _parallelType)
	{
		img_row = p.rows;
		img_col = p.cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Ip2ab_Guide3_Share_Transpose_SSE : public RowSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide3_Share_Transpose_SSE(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec(_p, _vI, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 3;
	}
};

struct RowSumFilter_Ip2ab_Guide3_Share_Transpose_AVX : public RowSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide3_Share_Transpose_AVX(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec(_p, _vI, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 7;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec : public ColumnSumFilter_Ip2ab_Guide3_Share_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
		step = va[0].cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_SSE : public ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType) {}
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_AVX : public ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_Transpose_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType) {}
};
