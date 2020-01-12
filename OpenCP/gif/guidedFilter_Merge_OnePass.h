#pragma once

#include "guidedFilter.hpp"

class guidedFilter_Merge_OnePass : public cp::GuidedFilterBase
{
protected:
	int parallelType;

	float div;
	int row;
	int col;
	int p_cn;
	int I_cn;

	//cv::Mat a;
	//cv::Mat b;
	
public:
	virtual void filter_Guide1(int cn);
	virtual void filter_Guide3(int cn);

	guidedFilter_Merge_OnePass(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType);
	void filter();
	void filterVector();
};

class guidedFilter_Merge_OnePass_SIMD : public guidedFilter_Merge_OnePass
{
private:
	//__m128 mDiv;
	//__m128 mBorder;

	void filter_Guide1(int cn) override;
	void filter_Guide3(int cn) override;
public:
	guidedFilter_Merge_OnePass_SIMD(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_OnePass(_src, _guide, _dest, _r, _eps, _parallelType)
	{
		//mDiv = _mm_set1_ps(div);
		//mBorder = _mm_set1_ps(r + 1);
	}
};



class guidedFilter_Merge_OnePass_LoopFusion : public guidedFilter_Merge_OnePass
{
private:
	void filter_Guide1(int cn) override;
	void filter_Guide3(int cn) override;
public:
	guidedFilter_Merge_OnePass_LoopFusion(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: guidedFilter_Merge_OnePass(_src, _guide, _dest, _r, _eps, _parallelType) {};
};
