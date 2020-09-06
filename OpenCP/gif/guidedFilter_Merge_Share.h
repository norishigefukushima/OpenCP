#pragma once

#include "guidedFilter_Merge.h"

class guidedFilter_Merge_Share_Base : public guidedImageFilter_Merge_Base
{
protected:
	cv::Mat mean_I;
	cv::Mat var;

	std::vector<cv::Mat> vMean_I;
	std::vector<cv::Mat> vCov;
	cv::Mat det;

	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	virtual void compute_Var();
	virtual void compute_Cov();
public:
	guidedFilter_Merge_Share_Base::guidedFilter_Merge_Share_Base(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType, const bool isInit = true)
		: guidedImageFilter_Merge_Base(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_SHARE;
		if (isInit)init();
	}
	void init() override;
	void filter() override;
	void filterVector() override;
};

class guidedFilter_Merge_Share_SSE : public guidedFilter_Merge_Share_Base
{
protected:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_SSE::guidedFilter_Merge_Share_SSE(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Base(_src, _guide, _dest, _r, _eps, _parallelType, true)
	{
		implementation = cp::GUIDED_MERGE_SHARE_SSE;
	}
};

class guidedFilter_Merge_Share_AVX : public guidedFilter_Merge_Share_Base
{
protected:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

	void compute_Var() override;
	void compute_Cov() override;
public:
	guidedFilter_Merge_Share_AVX::guidedFilter_Merge_Share_AVX(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Base(_src, _guide, _dest, _r, _eps, _parallelType, true)
	{
		implementation = cp::GUIDED_MERGE_SHARE_AVX;
	}
};



/* --- Compute Var --- */
struct RowSumFilter_Var : public RowSumFilter_base
{
protected:
	cv::Mat& I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Var(cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: I(_I), RowSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = I.rows;
		img_col = I.cols;
	}
};

struct ColumnSumFilter_Var_nonVec : public ColumnSumFilter_base
{
protected:
	cv::Mat& var;
	cv::Mat& mean_I;
	float eps;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Var_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, int _r, float _eps, int _parallelType)
		: var(_var), mean_I(_mean_I), eps(_eps), ColumnSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = tempVec[0].rows;
		img_col = tempVec[0].cols;
		step = tempVec[0].cols;
	}
};

struct ColumnSumFilter_Var_SSE : public ColumnSumFilter_Var_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Var_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Var_nonVec(_tempVec, _var, _mean_I, _r, _eps, _parallelType)
	{
		img_col = tempVec[0].cols / 4;
	}
};

struct ColumnSumFilter_Var_AVX : public ColumnSumFilter_Var_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Var_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Var_nonVec(_tempVec, _var, _mean_I, _r, _eps, _parallelType)
	{
		img_col = tempVec[0].cols / 8;
	}
};



/* --- Compute Cov --- */
struct RowSumFilter_Cov : public RowSumFilter_base
{
protected:
	std::vector<cv::Mat>& vI;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Cov(std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: vI(_vI), RowSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = vI[0].rows;
		img_col = vI[0].cols;
	}
};

struct ColumnSumFilter_Cov_nonVec : public ColumnSumFilter_base
{
protected:
	std::vector<cv::Mat>& vCov;
	std::vector<cv::Mat>& vMean_I;
	cv::Mat& det;
	float eps;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Cov_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, int _r, float _eps, int _parallelType)
		: vCov(_vCov), det(_det), vMean_I(_vMean_I), eps(_eps), ColumnSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = tempVec[0].rows;
		img_col = tempVec[0].cols;
		step = tempVec[0].cols;
	}
};

struct ColumnSumFilter_Cov_SSE : public ColumnSumFilter_Cov_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Cov_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Cov_nonVec(_tempVec, _vCov, _det, _vMean_I, _r, _eps, _parallelType)
	{
		img_col = tempVec[0].cols / 4;
	}
};

struct ColumnSumFilter_Cov_AVX : public ColumnSumFilter_Cov_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Cov_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Cov_nonVec(_tempVec, _vCov, _det, _vMean_I, _r, _eps, _parallelType)
	{
		img_col = tempVec[0].cols / 8;
	}
};



/* --- Guide1 --- */
struct RowSumFilter_Ip2ab_Guide1_Share : public RowSumFilter_base
{
protected:
	cv::Mat& p;
	cv::Mat& I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide1_Share(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: p(_p), I(_I), RowSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = p.rows;
		img_col = p.cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_nonVec : public ColumnSumFilter_base
{
protected:
	cv::Mat& a;
	cv::Mat& b;
	cv::Mat& var;
	cv::Mat& mean_I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: a(_a), b(_b), var(_var), mean_I(_mean_I), ColumnSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = a.rows;
		img_col = a.cols;
		step = a.cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_SSE : public ColumnSumFilter_Ip2ab_Guide1_Share_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType)
	{
		img_col = a.cols / 4;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Share_AVX : public ColumnSumFilter_Ip2ab_Guide1_Share_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Share_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Share_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _parallelType)
	{
		img_col = a.cols / 8;
	}
};



/* --- Guide3 --- */
struct RowSumFilter_Ip2ab_Guide3_Share : public RowSumFilter_base
{
protected:
	cv::Mat& p;
	std::vector<cv::Mat>& vI;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide3_Share(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: p(_p), vI(_vI), RowSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = p.rows;
		img_col = p.cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_nonVec : public ColumnSumFilter_base
{
protected:
	std::vector<cv::Mat>& va;
	cv::Mat& b;
	std::vector<cv::Mat>& vCov;
	cv::Mat& det;
	std::vector<cv::Mat>& vMean_I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: va(_va), b(_b), vCov(_vCov), det(_det), vMean_I(_vMean_I), ColumnSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = tempVec[0].rows;
		img_col = tempVec[0].cols;
		step = tempVec[0].cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_SSE : public ColumnSumFilter_Ip2ab_Guide3_Share_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType)
	{
		img_col = tempVec[0].cols / 4;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Share_AVX : public ColumnSumFilter_Ip2ab_Guide3_Share_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Share_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Share_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _parallelType)
	{
		img_col = tempVec[0].cols / 8;
	}
};
