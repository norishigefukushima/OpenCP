#pragma once

#include "boxFilter_Base.h"

/*
 * SSATÅiêÇíº -> ì]íu -> êÇíºÅj
 */

class boxFilter_SSAT_VtV_nonVec
{
protected:
	cv::Mat src;
	cv::Mat dest;

	cv::Mat temp;
	cv::Mat temp_t;
	cv::Mat dest_t;

	int r;
	int parallelType;

public:
	boxFilter_SSAT_VtV_nonVec(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	virtual void filter();
};

class boxFilter_SSAT_VtV_SSE : public boxFilter_SSAT_VtV_nonVec
{
private:

public:
	boxFilter_SSAT_VtV_SSE(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_VtV_nonVec(_src, _dest, _r, _parallelType) {};
	void filter() override;
};

class boxFilter_SSAT_VtV_AVX : public boxFilter_SSAT_VtV_nonVec
{
private:

public:
	boxFilter_SSAT_VtV_AVX(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType)
		: boxFilter_SSAT_VtV_nonVec(_src, _dest, _r, _parallelType) {};
	void filter() override;
};
