#pragma once

#include "boxFilter_Base.h"

// SSAT: horizontal->vertical with nxn blocks

class boxFilter_SSAT_HV_CacheBlock_nonVec
{
protected:
	cv::Mat src;
	cv::Mat dest;
	int r;
	int parallelType;

	int row;
	int col;
	int cn;
	float div;
	int step;

	int divRow;

	virtual void upper_impl(const int idxDiv);
	virtual void middle_impl(const int idxDiv);
	virtual void lower_impl(const int idxDiv);

public:
	boxFilter_SSAT_HV_CacheBlock_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	void filter();
};



class boxFilter_SSAT_HV_CacheBlock_SSE : public boxFilter_SSAT_HV_CacheBlock_nonVec
{
private:
	__m128 mBorder;
	__m128 mDiv;

	void upper_impl(int idxDiv) override;
	void middle_impl(int idxDiv) override;
	void lower_impl(int idxDiv) override;

public:
	boxFilter_SSAT_HV_CacheBlock_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};

class boxFilter_SSAT_HV_CacheBlock_AVX : public boxFilter_SSAT_HV_CacheBlock_nonVec
{
private:
	__m256 mBorder;
	__m256 mDiv;

	void upper_impl(int idxDiv) override;
	void middle_impl(int idxDiv) override;
	void lower_impl(int idxDiv) override;

public:
	boxFilter_SSAT_HV_CacheBlock_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};
