#pragma once

#include "guidedFilter.hpp"

class guidedFilter_Merge_OnePass_2Div
{
protected:
	cv::Mat src;
	cv::Mat guide;
	cv::Mat dest;
	int r;
	float eps;
	int parallelType;

	float div;
	int row;
	int col;
	int p_cn;
	int I_cn;

	int divNum;
	int divRow;

	void filter_Guide1_upper_impl(int cnNum);
	void filter_Guide1_lower_impl(int cnNum);
	void filter_Guide3_upper_impl(int cnNum);
	void filter_Guide3_lower_impl(int cnNum);

public:
	guidedFilter_Merge_OnePass_2Div(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType);
	virtual void filter();
};

class guidedFilter_Merge_OnePass_nDiv : public guidedFilter_Merge_OnePass_2Div
{
private:
	void filter_Guide1_middle_impl(int cnNum, int rowOffset);
	void filter_Guide3_middle_impl(int cnNum, int rowOffset);
public:
	guidedFilter_Merge_OnePass_nDiv(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType);
	void filter() override;
};

void guidedFilter_Merge_OnePass_Div(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int r, float eps, int parallelType);
