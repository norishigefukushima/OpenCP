#pragma once

#include "boxFilter.hpp"

//one pass box filtering with image div

class boxFilter_OPSAT_AoS_2Div
{
protected:
	cv::Mat src;
	cv::Mat dest;
	int r;
	int parallelType;

	float div;
	int row;
	int col;
	int cn;
	int cnNum;

	int divNum;
	int divRow;

	virtual void filter_upper_impl(int cnNum);
	virtual void filter_lower_impl(int cnNum);
public:
	boxFilter_OPSAT_AoS_2Div(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	virtual void init();
	virtual void filter();
};

class boxFilter_OPSAT_AoS_nDiv : public boxFilter_OPSAT_AoS_2Div
{
protected:
	int num;

	void filter_middle_impl(int cnNum, int idx);
public:
	boxFilter_OPSAT_AoS_nDiv(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num);
	void init() override;
	void filter() override;
};



/*
 * SSE
 */
class boxFilter_OPSAT_AoS_2Div_SSE : public boxFilter_OPSAT_AoS_2Div
{
protected:
	__m128 mDiv;
	__m128 mBorder;

	void filter_upper_impl(int cnNum) override;
	void filter_lower_impl(int cnNum) override;
public:
	boxFilter_OPSAT_AoS_2Div_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void init() override;
};

class boxFilter_OPSAT_AoS_nDiv_SSE : public boxFilter_OPSAT_AoS_2Div_SSE
{
private:
	int num;

	void filter_middle_impl(int cnNum, int idx);
public:
	boxFilter_OPSAT_AoS_nDiv_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num);
	void init() override;
	void filter() override;
};



/*
 * AVX
 */
class boxFilter_OPSAT_AoS_2Div_AVX : public boxFilter_OPSAT_AoS_2Div
{
protected:
	__m256 mDiv;
	__m256 mBorder;

	void filter_upper_impl(int cnNum) override;
	void filter_lower_impl(int cnNum) override;
public:
	boxFilter_OPSAT_AoS_2Div_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void init() override;
};

class boxFilter_OPSAT_AoS_nDiv_AVX : public boxFilter_OPSAT_AoS_2Div_AVX
{
private:
	int num;

	void filter_middle_impl(int cnNum, int idx);
public:
	boxFilter_OPSAT_AoS_nDiv_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num);
	void init() override;
	void filter() override;
};
