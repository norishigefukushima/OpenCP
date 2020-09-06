#pragma once

#include "guidedFilter_Merge_Share.h"

class guidedFilter_Merge_Share_Mixed_nonVec : public guidedImageFilter_Merge_Base
{
protected:
	cv::Mat mean_I;
	cv::Mat var;

	std::vector<cv::Mat> vMean_I;
	std::vector<cv::Mat> vCov;
	cv::Mat det;

	std::vector<cv::Mat> vSrc;
	std::vector<cv::Mat> vDest;

	virtual void filter_Src1Guide1(cv::Mat& input, cv::Mat& output);
	virtual void filter_Src1Guide3(cv::Mat& input, cv::Mat& output);

	virtual void filter_Src3Guide1_First(cv::Mat& input, cv::Mat& output);
	virtual void filter_Src3Guide3_First(cv::Mat& input, cv::Mat& output);

	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

public:
	guidedFilter_Merge_Share_Mixed_nonVec(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedImageFilter_Merge_Base(_src, _guide, _dest, _r, _eps, _parallelType, false)
	{
		implementation = cp::GUIDED_MERGE_SHARE_EX;
		init();
	}
	void init() override;
	void filter() override;
	void filterVector() override;
};

class guidedFilter_Merge_Share_Mixed_SSE : public guidedFilter_Merge_Share_Mixed_nonVec
{
private:
	void filter_Src1Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Src1Guide3(cv::Mat& input, cv::Mat& output) override;

	void filter_Src3Guide1_First(cv::Mat& input, cv::Mat& output) override;
	void filter_Src3Guide3_First(cv::Mat& input, cv::Mat& output) override;

	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

public:
	guidedFilter_Merge_Share_Mixed_SSE(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Mixed_nonVec(_src, _guide, _dest, _r, _eps, _parallelType)
	{
		implementation = cp::GUIDED_MERGE_SHARE_EX_SSE;
	}
};

class guidedFilter_Merge_Share_Mixed_AVX : public guidedFilter_Merge_Share_Mixed_nonVec
{
private:
	void filter_Src1Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Src1Guide3(cv::Mat& input, cv::Mat& output) override;

	void filter_Src3Guide1_First(cv::Mat& input, cv::Mat& output) override;
	void filter_Src3Guide3_First(cv::Mat& input, cv::Mat& output) override;

	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;

public:
	guidedFilter_Merge_Share_Mixed_AVX(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_Share_Mixed_nonVec(_src, _guide, _dest, _r, _eps, _parallelType) 
	{
		implementation = cp::GUIDED_MERGE_SHARE_EX_AVX;
	}
};



/*
 * Guide1
 */
struct ColumnSumFilter_Ip2ab_Guide1_First_nonVec : public ColumnSumFilter_base
{
protected:
	float eps;

	cv::Mat& a;
	cv::Mat& b;
	cv::Mat& var;
	cv::Mat& mean_I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_First_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: a(_a), b(_b), var(_var), mean_I(_mean_I), eps(_eps), ColumnSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = a.rows;
		img_col = a.cols;
		step = a.cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_First_SSE : public ColumnSumFilter_Ip2ab_Guide1_First_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_First_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_First_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _eps, _parallelType)
	{
		img_col = a.cols / 4;
	}
};

struct ColumnSumFilter_Ip2ab_Guide1_First_AVX : public ColumnSumFilter_Ip2ab_Guide1_First_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide1_First_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _var, cv::Mat& _mean_I, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide1_First_nonVec(_tempVec, _var, _mean_I, _a, _b, _r, _eps, _parallelType)
	{
		img_col = a.cols / 8;
	}
};



/*
 * Guide3
 */
struct ColumnSumFilter_Ip2ab_Guide3_First_nonVec : public ColumnSumFilter_base
{
protected:
	float eps;

	std::vector<cv::Mat>& va;
	cv::Mat& b;
	std::vector<cv::Mat>& vCov;
	cv::Mat& det;
	std::vector<cv::Mat>& vMean_I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_First_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: va(_va), b(_b), vCov(_vCov), det(_det), vMean_I(_vMean_I), eps(_eps), ColumnSumFilter_base(_tempVec, _r, _parallelType)
	{
		img_row = tempVec[0].rows;
		img_col = tempVec[0].cols;
		step = tempVec[0].cols;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_First_SSE : public ColumnSumFilter_Ip2ab_Guide3_First_nonVec
{
private:
	const __m128 mDiv = _mm_set1_ps(div);
	const __m128 mEps = _mm_set1_ps(eps);
	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_First_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_First_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _eps, _parallelType)
	{
		img_col = tempVec[0].cols / 4;
	}
};

struct ColumnSumFilter_Ip2ab_Guide3_First_AVX : public ColumnSumFilter_Ip2ab_Guide3_First_nonVec
{
private:
	const __m256 mDiv = _mm256_set1_ps(div);
	const __m256 mEps = _mm256_set1_ps(eps);
	const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_Ip2ab_Guide3_First_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vCov, cv::Mat& _det, std::vector<cv::Mat>& _vMean_I, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
		: ColumnSumFilter_Ip2ab_Guide3_First_nonVec(_tempVec, _vCov, _det, _vMean_I, _va, _b, _r, _eps, _parallelType)
	{
		img_col = tempVec[0].cols / 8;
	}
};
