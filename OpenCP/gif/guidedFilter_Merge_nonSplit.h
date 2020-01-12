#pragma once

#include "guidedFilter_Merge.h"

void guidedFilter_nonSplit_nonVec(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps, int parallelType);
void guidedFilter_nonSplit_SSE(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps, int parallelType);
void guidedFilter_nonSplit_AVX(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps, int parallelType);

class guidedFilter_nonSplit_Guide1_nonVec
{
protected:
	cv::Mat src;
	cv::Mat guide;
	cv::Mat dest;
	int r;
	float eps;
	int parallelType;

	std::vector<cv::Mat> a;
	std::vector<cv::Mat> b;
	std::vector<cv::Mat> temp;
public:
	guidedFilter_nonSplit_Guide1_nonVec(cv::Mat& src_, cv::Mat& guide_, cv::Mat& dest_, int r_, float eps_, int parallelType_)
		: src(src_), guide(guide_), dest(dest_), r(r_), eps(eps_), parallelType(parallelType_) {};
	~guidedFilter_nonSplit_Guide1_nonVec() {};
	virtual void init();
	virtual void filter();
};

class guidedFilter_nonSplit_Guide3_nonVec
{
protected:
	cv::Mat src;
	cv::Mat guide;
	cv::Mat dest;
	int r;
	float eps;
	int parallelType;

	std::vector<cv::Mat> va;
	cv::Mat b;
	std::vector<cv::Mat> temp;
public:
	guidedFilter_nonSplit_Guide3_nonVec(cv::Mat src_, cv::Mat guide_, cv::Mat& dest_, int r_, float eps_, int parallelType_)
		: src(src_), guide(guide_), dest(dest_), r(r_), eps(eps_), parallelType(parallelType_) {};
	~guidedFilter_nonSplit_Guide3_nonVec() {};
	virtual void init();
	virtual void filter();
};





class guidedFilter_nonSplit_Guide1_SSE : public guidedFilter_nonSplit_Guide1_nonVec
{
private:

public:
	guidedFilter_nonSplit_Guide1_SSE(cv::Mat src_, cv::Mat guide_, cv::Mat& dest_, int r_, float eps_, int parallelType_)
		: guidedFilter_nonSplit_Guide1_nonVec(src_, guide_, dest_, r_, eps_, parallelType_) {};
	~guidedFilter_nonSplit_Guide1_SSE() {};
	void init() override;
	void filter() override;
};

class guidedFilter_nonSplit_Guide3_SSE : public guidedFilter_nonSplit_Guide3_nonVec
{
private:

public:
	guidedFilter_nonSplit_Guide3_SSE(cv::Mat src_, cv::Mat guide_, cv::Mat& dest_, int r_, float eps_, int parallelType_)
		: guidedFilter_nonSplit_Guide3_nonVec(src_, guide_, dest_, r_, eps_, parallelType_) {};
	~guidedFilter_nonSplit_Guide3_SSE() {};
	void init() override;
	void filter() override;
};





class guidedFilter_nonSplit_Guide1_AVX : public guidedFilter_nonSplit_Guide1_nonVec
{
private:

public:
	guidedFilter_nonSplit_Guide1_AVX(cv::Mat src_, cv::Mat guide_, cv::Mat& dest_, int r_, float eps_, int parallelType_)
		: guidedFilter_nonSplit_Guide1_nonVec(src_, guide_, dest_, r_, eps_, parallelType_) {};
	~guidedFilter_nonSplit_Guide1_AVX() {};
	void init() override;
	void filter() override;
};

class guidedFilter_nonSplit_Guide3_AVX : public guidedFilter_nonSplit_Guide3_nonVec
{
private:

public:
	guidedFilter_nonSplit_Guide3_AVX(cv::Mat src_, cv::Mat guide_, cv::Mat& dest_, int r_, float eps_, int parallelType_)
		: guidedFilter_nonSplit_Guide3_nonVec(src_, guide_, dest_, r_, eps_, parallelType_) {};
	~guidedFilter_nonSplit_Guide3_AVX() {};
	void init() override;
	void filter() override;
};





/*
 * Guide1
 */
struct RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec : public RowSumFilter_base
{
protected:
	cv::Mat& p;
	cv::Mat& I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec(cv::Mat& p_, cv::Mat& I_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: p(p_), I(I_), RowSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = p.rows;
		img_col = p.cols;
	}
};

struct RowSumFilter_nonSplit_Ip2ab_Guide1_SSE : public RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_Ip2ab_Guide1_SSE(cv::Mat& p_, cv::Mat& I_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec(p_, I_, tempVec_, r_, parallelType_) {};
};

struct RowSumFilter_nonSplit_Ip2ab_Guide1_AVX : public RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_Ip2ab_Guide1_AVX(cv::Mat& p_, cv::Mat& I_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_Ip2ab_Guide1_nonVec(p_, I_, tempVec_, r_, parallelType_) {};
};



struct ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec : public ColumnSumFilter_base
{
protected:
	std::vector<cv::Mat>& a;
	std::vector<cv::Mat>& b;
	float eps;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec(std::vector<cv::Mat>& tempVec_, std::vector<cv::Mat>& a_, std::vector<cv::Mat>& b_, int r_, float eps_, int parallelType_)
		: a(a_), b(b_), eps(eps_), ColumnSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = tempVec[0].rows;
		img_col = tempVec[0].cols;
		step = img_col;
	}
};

struct ColumnSumFilter_nonSplit_Ip2ab_Guide1_SSE : public ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_Ip2ab_Guide1_SSE(std::vector<cv::Mat>& tempVec_, std::vector<cv::Mat>& a_, std::vector<cv::Mat>& b_, int r_, float eps_, int parallelType_)
		: ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec(tempVec_, a_, b_, r_, eps_, parallelType_) {};
};

struct ColumnSumFilter_nonSplit_Ip2ab_Guide1_AVX : public ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_Ip2ab_Guide1_AVX(std::vector<cv::Mat>& tempVec_, std::vector<cv::Mat>& a_, std::vector<cv::Mat>& b_, int r_, float eps_, int parallelType_)
		: ColumnSumFilter_nonSplit_Ip2ab_Guide1_nonVec(tempVec_, a_, b_, r_, eps_, parallelType_) {};
};



struct RowSumFilter_nonSplit_ab2q_Guide1_nonVec : public RowSumFilter_base
{
protected:
	std::vector<cv::Mat>& a;
	std::vector<cv::Mat>& b;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_ab2q_Guide1_nonVec(std::vector<cv::Mat>& a_, std::vector<cv::Mat>& b_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: a(a_), b(b_), RowSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = a[0].rows;
		img_col = a[0].cols;
	}
};

struct RowSumFilter_nonSplit_ab2q_Guide1_SSE : public RowSumFilter_nonSplit_ab2q_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_ab2q_Guide1_SSE(std::vector<cv::Mat>& a_, std::vector<cv::Mat>& b_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_ab2q_Guide1_nonVec(a_, b_, tempVec_, r_, parallelType_) {};
};

struct RowSumFilter_nonSplit_ab2q_Guide1_AVX : public RowSumFilter_nonSplit_ab2q_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_ab2q_Guide1_AVX(std::vector<cv::Mat>& a_, std::vector<cv::Mat>& b_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_ab2q_Guide1_nonVec(a_, b_, tempVec_, r_, parallelType_) {};
};



struct ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec : public ColumnSumFilter_base
{
protected:
	cv::Mat& I;
	cv::Mat& q;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec(std::vector<cv::Mat>& tempVec_, cv::Mat& I_, cv::Mat& q_, int r_, int parallelType_)
		: I(I_), q(q_), ColumnSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = q.rows;
		img_col = q.cols;
		step = img_col;
	}
};

struct ColumnSumFilter_nonSplit_ab2q_Guide1_SSE : public ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_ab2q_Guide1_SSE(std::vector<cv::Mat>& tempVec_, cv::Mat& I_, cv::Mat& q_, int r_, int parallelType_)
		: ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec(tempVec_, I_, q_, r_, parallelType_) {};
};

struct ColumnSumFilter_nonSplit_ab2q_Guide1_AVX : public ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_ab2q_Guide1_AVX(std::vector<cv::Mat>& tempVec_, cv::Mat& I_, cv::Mat& q_, int r_, int parallelType_)
		: ColumnSumFilter_nonSplit_ab2q_Guide1_nonVec(tempVec_, I_, q_, r_, parallelType_) {};
};



/*
 * Guide3
 */
struct RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec : public RowSumFilter_base
{
protected:
	cv::Mat& p;
	cv::Mat& I;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec(cv::Mat& p_, cv::Mat& I_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: p(p_), I(I_), RowSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = p.rows;
		img_col = p.cols;
	}
};

struct RowSumFilter_nonSplit_Ip2ab_Guide3_SSE : public RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_Ip2ab_Guide3_SSE(cv::Mat& p_, cv::Mat& I_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec(p_, I_, tempVec_, r_, parallelType_) {};
};

struct RowSumFilter_nonSplit_Ip2ab_Guide3_AVX : public RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_Ip2ab_Guide3_AVX(cv::Mat& p_, cv::Mat& I_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_Ip2ab_Guide3_nonVec(p_, I_, tempVec_, r_, parallelType_) {};
};



struct ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec : public ColumnSumFilter_base
{
protected:
	std::vector<cv::Mat>& va;
	cv::Mat& b;
	float eps;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec(std::vector<cv::Mat>& tempVec_, std::vector<cv::Mat>& va_, cv::Mat& b_, int r_, float eps_, int parallelType_)
		: va(va_), b(b_), eps(eps_), ColumnSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = tempVec[0].rows;
		img_col = tempVec[0].cols;
		step = img_col;
	}
};

struct ColumnSumFilter_nonSplit_Ip2ab_Guide3_SSE : public ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_Ip2ab_Guide3_SSE(std::vector<cv::Mat>& tempVec_, std::vector<cv::Mat>& va_, cv::Mat& b_, int r_, float eps_, int parallelType_)
		: ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec(tempVec_, va_, b_, r_, eps_, parallelType_) {};
};

struct ColumnSumFilter_nonSplit_Ip2ab_Guide3_AVX : public ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_Ip2ab_Guide3_AVX(std::vector<cv::Mat>& tempVec_, std::vector<cv::Mat>& va_, cv::Mat& b_, int r_, float eps_, int parallelType_)
		: ColumnSumFilter_nonSplit_Ip2ab_Guide3_nonVec(tempVec_, va_, b_, r_, eps_, parallelType_) {};
};



struct RowSumFilter_nonSplit_ab2q_Guide3_nonVec : public RowSumFilter_base
{
protected:
	std::vector<cv::Mat>& va;
	cv::Mat& b;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_ab2q_Guide3_nonVec(std::vector<cv::Mat>& va_, cv::Mat& b_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: va(va_), b(b_), RowSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = va[0].rows;
		img_col = va[0].cols;
	}
};

struct RowSumFilter_nonSplit_ab2q_Guide3_SSE : public RowSumFilter_nonSplit_ab2q_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_ab2q_Guide3_SSE(std::vector<cv::Mat>& va_, cv::Mat& b_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_ab2q_Guide3_nonVec(va_, b_, tempVec_, r_, parallelType_) {};
};

struct RowSumFilter_nonSplit_ab2q_Guide3_AVX : public RowSumFilter_nonSplit_ab2q_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	RowSumFilter_nonSplit_ab2q_Guide3_AVX(std::vector<cv::Mat>& va_, cv::Mat& b_, std::vector<cv::Mat>& tempVec_, int r_, int parallelType_)
		: RowSumFilter_nonSplit_ab2q_Guide3_nonVec(va_, b_, tempVec_, r_, parallelType_) {};
};



struct ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec : public ColumnSumFilter_base
{
protected:
	cv::Mat& I;
	cv::Mat& q;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec(std::vector<cv::Mat>& tempVec_, cv::Mat& I_, cv::Mat& q_, int r_, int parallelType_)
		: I(I_), q(q_), ColumnSumFilter_base(tempVec_, r_, parallelType_)
	{
		img_row = q.rows;
		img_col = q.cols;
		step = img_col;
	}
};

struct ColumnSumFilter_nonSplit_ab2q_Guide3_SSE : public ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_ab2q_Guide3_SSE(std::vector<cv::Mat>& tempVec_, cv::Mat& I_, cv::Mat& q_, int r_, int parallelType_)
		: ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec(tempVec_, I_, q_, r_, parallelType_) {};
};

struct ColumnSumFilter_nonSplit_ab2q_Guide3_AVX : public ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec
{
private:
	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	ColumnSumFilter_nonSplit_ab2q_Guide3_AVX(std::vector<cv::Mat>& tempVec_, cv::Mat& I_, cv::Mat& q_, int r_, int parallelType_)
		: ColumnSumFilter_nonSplit_ab2q_Guide3_nonVec(tempVec_, I_, q_, r_, parallelType_) {};
};