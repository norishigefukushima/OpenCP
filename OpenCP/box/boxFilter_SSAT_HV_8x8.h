#pragma once

#include "boxFilter.hpp"

// SSAT: horizontal->vertical with 4x4 blocks

class boxFilter_SSAT_HV_8x8 : public cp::BoxFilterBase
{
private:
	int padding;
	cv::Mat copy;
	__m256 mDiv;

	void filter_naive_impl() override;
	void filter_omp_impl() override;
	void operator()(const cv::Range& range) const override;
public:
	boxFilter_SSAT_HV_8x8(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
};
