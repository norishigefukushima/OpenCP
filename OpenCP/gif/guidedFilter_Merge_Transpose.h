#pragma once

#include "guidedFilter_Merge.h"

class guidedFilter_Merge_Transpose_nonVec : public guidedImageFilter_Merge_Base
{
protected:
	cv::Size t_size;
	cv::Mat I_t;
	std::vector<cv::Mat> vI_t;

	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;
public:
	guidedFilter_Merge_Transpose_nonVec(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType, const bool isInit=true)
		: guidedImageFilter_Merge_Base(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		t_size = cv::Size(_src.rows, _src.cols);
		implementation = cp::GUIDED_MERGE_TRANSPOSE;
		if(isInit)init();
	}
	void init() override;
};

class guidedFilter_Merge_Transpose_SSE : public guidedFilter_Merge_Transpose_nonVec
{
private:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;
public:
	guidedFilter_Merge_Transpose_SSE(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Transpose_nonVec(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_TRANSPOSE_SSE;
		t_size = cv::Size(src.rows * 4, src.cols / 4);
		init();
	}
};

class guidedFilter_Merge_Transpose_AVX : public guidedFilter_Merge_Transpose_nonVec
{
private:
	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;
public:
	guidedFilter_Merge_Transpose_AVX(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Transpose_nonVec(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_TRANSPOSE_AVX;
		t_size = cv::Size(src.rows * 8, src.cols / 8);
		init();
	}
};



/*
 * Guide1
 */
struct RowSumFilter_Ip2ab_Guide1_Transpose_nonVec : public RowSumFilter_Ip2ab_Guide1
{
protected:
	int step;
	cv::Mat& I_t;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide1_Transpose_nonVec(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, int _r, int _parallelType)
		: I_t(_I_t), RowSumFilter_Ip2ab_Guide1(_p, _I, _tempVec, _r, _parallelType)
	{
		img_row = p.rows;
		img_col = p.cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Ip2ab_Guide1_Transpose_SSE : public RowSumFilter_Ip2ab_Guide1_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide1_Transpose_SSE(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide1_Transpose_nonVec(_p, _I, _tempVec, _I_t, _r, _parallelType)
	{
		step = tempVec[0].cols - 3;
	}
};

struct RowSumFilter_Ip2ab_Guide1_Transpose_AVX : public RowSumFilter_Ip2ab_Guide1_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide1_Transpose_AVX(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide1_Transpose_nonVec(_p, _I, _tempVec, _I_t, _r, _parallelType)
	{
		step = tempVec[0].cols - 7;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Transpose_nonVec : public ColumnSumFilter_Ip2ab_Guide1_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_nonVec(_tempVec, _a, _b, _r, _eps, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
		step = a.cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_Transpose_SSE : public ColumnSumFilter_Ip2ab_Guide1_Transpose_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Transpose_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Transpose_nonVec(_tempVec,_a, _b, _r, _eps, _parallelType) {}
};

struct ColumnSumFilter_Ip2ab_Guide1_Transpose_AVX : public ColumnSumFilter_Ip2ab_Guide1_Transpose_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_Transpose_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_Transpose_nonVec(_tempVec, _a, _b, _r, _eps, _parallelType) {}
};



struct RowSumFilter_ab2q_Guide1_Transpose_nonVec : public RowSumFilter_ab2q_Guide1
{
protected:
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_ab2q_Guide1_Transpose_nonVec(cv::Mat& _a, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_ab2q_Guide1(_a, _b, _tempVec, _r, _parallelType)
	{
		img_row = a.rows;
		img_col = a.cols;
		step = a.rows;
	}
};

struct RowSumFilter_ab2q_Guide1_Transpose_SSE : public RowSumFilter_ab2q_Guide1_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_ab2q_Guide1_Transpose_SSE(cv::Mat& _a, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_ab2q_Guide1_Transpose_nonVec(_a, _b, _tempVec, _r, _parallelType)
	{
		step = a.rows * 4 - 3;
	}
};

struct RowSumFilter_ab2q_Guide1_Transpose_AVX : public RowSumFilter_ab2q_Guide1_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_ab2q_Guide1_Transpose_AVX(cv::Mat& _a, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_ab2q_Guide1_Transpose_nonVec(_a, _b, _tempVec, _r, _parallelType)
	{
		step = a.rows * 8 - 7;
	}
};

struct ColumnSumFilter_ab2q_Guide1_Transpose_nonVec : public ColumnSumFilter_ab2q_Guide1_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_ab2q_Guide1_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, cv::Mat& _q, int _r, int _parallelType)
		: ColumnSumFilter_ab2q_Guide1_nonVec(_tempVec, _I_t, _q, _r, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
		step = q.cols;
	}
};

struct ColumnSumFilter_ab2q_Guide1_Transpose_SSE : public ColumnSumFilter_ab2q_Guide1_Transpose_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_ab2q_Guide1_Transpose_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, cv::Mat& _q, int _r, int _parallelType)
		: ColumnSumFilter_ab2q_Guide1_Transpose_nonVec(_tempVec, _I_t, _q, _r, _parallelType) {}
};

struct ColumnSumFilter_ab2q_Guide1_Transpose_AVX : public ColumnSumFilter_ab2q_Guide1_Transpose_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_ab2q_Guide1_Transpose_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _I_t, cv::Mat& _q, int _r, int _parallelType)
		: ColumnSumFilter_ab2q_Guide1_Transpose_nonVec(_tempVec, _I_t, _q, _r, _parallelType) {}
};



/*
 * Guide3
 */
struct RowSumFilter_Ip2ab_Guide3_Transpose_nonVec : public RowSumFilter_Ip2ab_Guide3
{
protected:
	int step;
	std::vector<cv::Mat>& vI_t;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide3_Transpose_nonVec(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI_t, int _r, int _parallelType)
		: vI_t(_vI_t), RowSumFilter_Ip2ab_Guide3(_p, _vI, _tempVec, _r, _parallelType)
	{
		img_row = p.rows;
		img_col = p.cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_Ip2ab_Guide3_Transpose_SSE : public RowSumFilter_Ip2ab_Guide3_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide3_Transpose_SSE(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI_t, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide3_Transpose_nonVec(_p, _vI, _tempVec, _vI_t, _r, _parallelType)
	{
		step = tempVec[0].cols - 3;
	}
};

struct RowSumFilter_Ip2ab_Guide3_Transpose_AVX : public RowSumFilter_Ip2ab_Guide3_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_Ip2ab_Guide3_Transpose_AVX(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI_t, int _r, int _parallelType)
		: RowSumFilter_Ip2ab_Guide3_Transpose_nonVec(_p, _vI, _tempVec, _vI_t, _r, _parallelType)
	{
		step = tempVec[0].cols - 7;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Transpose_nonVec : public ColumnSumFilter_Ip2ab_Guide3_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_nonVec(_tempVec, _va, _b, _r, _eps, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
		step = va[0].cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_Transpose_SSE : public ColumnSumFilter_Ip2ab_Guide3_Transpose_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Transpose_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Transpose_nonVec(_tempVec, _va, _b, _r, _eps, _parallelType) {}
};

struct ColumnSumFilter_Ip2ab_Guide3_Transpose_AVX : public ColumnSumFilter_Ip2ab_Guide3_Transpose_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_Transpose_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_Transpose_nonVec(_tempVec, _va, _b, _r, _eps, _parallelType) {}
};



struct RowSumFilter_ab2q_Guide3_Transpose_nonVec : public RowSumFilter_ab2q_Guide3
{
protected:
	int step;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_ab2q_Guide3_Transpose_nonVec(std::vector<cv::Mat>& _va, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_ab2q_Guide3(_va, _b, _tempVec, _r, _parallelType)
	{
		img_row = va[0].rows;
		img_col = va[0].cols;
		step = tempVec[0].cols;
	}
};

struct RowSumFilter_ab2q_Guide3_Transpose_SSE : public RowSumFilter_ab2q_Guide3_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_ab2q_Guide3_Transpose_SSE(std::vector<cv::Mat>& _va, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_ab2q_Guide3_Transpose_nonVec(_va, _b, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 3;
	}
};

struct RowSumFilter_ab2q_Guide3_Transpose_AVX : public RowSumFilter_ab2q_Guide3_Transpose_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_ab2q_Guide3_Transpose_AVX(std::vector<cv::Mat>& _va, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
		: RowSumFilter_ab2q_Guide3_Transpose_nonVec(_va, _b, _tempVec, _r, _parallelType)
	{
		step = tempVec[0].cols - 7;
	}
};

struct ColumnSumFilter_ab2q_Guide3_Transpose_nonVec : public ColumnSumFilter_ab2q_Guide3_nonVec
{
protected:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_ab2q_Guide3_Transpose_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI, cv::Mat& _q, int _r, int _parallelType)
		: ColumnSumFilter_ab2q_Guide3_nonVec(_tempVec, _vI, _q, _r, _parallelType)
	{
		img_row = tempVec[0].cols;
		img_col = tempVec[0].rows;
		step = q.cols;
	}
};

struct ColumnSumFilter_ab2q_Guide3_Transpose_SSE : public ColumnSumFilter_ab2q_Guide3_Transpose_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_ab2q_Guide3_Transpose_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI, cv::Mat& _q, int _r, int _parallelType)
		: ColumnSumFilter_ab2q_Guide3_Transpose_nonVec(_tempVec, _vI, _q, _r, _parallelType) {}
};

struct ColumnSumFilter_ab2q_Guide3_Transpose_AVX : public ColumnSumFilter_ab2q_Guide3_Transpose_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_ab2q_Guide3_Transpose_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI, cv::Mat& _q, int _r, int _parallelType)
		: ColumnSumFilter_ab2q_Guide3_Transpose_nonVec(_tempVec, _vI, _q, _r, _parallelType) {}
};
