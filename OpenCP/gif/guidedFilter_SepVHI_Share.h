#pragma once

#include "guidedFilter.hpp"

class guidedFilter_SepVHI_Share : public cp::GuidedFilterBase
{
protected:
	int parallelType;

	std::vector<cv::Mat> ab_p;
	std::vector<cv::Mat> ag_p;
	std::vector<cv::Mat> ar_p;
	std::vector<cv::Mat> b_p;

	//for upsampling
	cv::Mat mean_a_b;
	cv::Mat mean_a_g;
	cv::Mat mean_a_r;
	cv::Mat mean_b;

	cv::Mat a_high_b;
	cv::Mat a_high_g;
	cv::Mat a_high_r;
public:

	guidedFilter_SepVHI_Share(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
		: cp::GuidedFilterBase(_src, _guide, _dest, _r, _eps), parallelType(_parallelType)
	{
		implementation = cp::GUIDED_SEP_VHI_SHARE;
		ab_p.resize(3);
		ag_p.resize(3);
		ar_p.resize(3);
		b_p.resize(3);
	}

	void filter() override;
	void filterVector() override;
	void upsample() override;
};
