#pragma once

#include "guidedFilter.hpp"

class guidedFilter_Naive_Share : public cp::GuidedFilterBase
{
protected:
	cp::BoxFilterMethod boxType;
	int parallelType;

	cv::Mat mean_I;
	cv::Mat mean_p;
	cv::Mat corr_I;
	cv::Mat corr_Ip;
	cv::Mat var_I;
	cv::Mat cov_Ip;
	cv::Mat a;
	cv::Mat b;

	cv::Mat I_b;
	cv::Mat I_g;
	cv::Mat I_r;
	//cv::Mat mean_p;
	cv::Mat mean_I_b;
	cv::Mat mean_I_g;
	cv::Mat mean_I_r;
	cv::Mat corr_I_bb;
	cv::Mat corr_I_bg;
	cv::Mat corr_I_br;
	cv::Mat corr_I_gg;
	cv::Mat corr_I_gr;
	cv::Mat corr_I_rr;
	cv::Mat corr_Ip_b;
	cv::Mat corr_Ip_g;
	cv::Mat corr_Ip_r;
	cv::Mat var_I_bb;
	cv::Mat var_I_bg;
	cv::Mat var_I_br;
	cv::Mat var_I_gg;
	cv::Mat var_I_gr;
	cv::Mat var_I_rr;
	cv::Mat cov_Ip_b;
	cv::Mat cov_Ip_g;
	cv::Mat cov_Ip_r;
	cv::Mat a_b;
	cv::Mat a_g;
	cv::Mat a_r;

	cv::Mat temp_high;

	enum
	{
		BOX_32F,
		BOX_64F
	};
	int average_method = BOX_32F;
	void average(cv::Mat& src, cv::Mat& dest, const int r);

	void computeVar(cv::Mat& guide);
	void Ip2ab_Guide1(cv::Mat& input, cv::Mat& guide);
	void ab2q_Guide1(cv::Mat& guide, cv::Mat& output);
	void filter_Guide1(cv::Mat& input, cv::Mat& guide, cv::Mat& output);

	void ab_up_2q_Guide1(cv::Mat& guide, cv::Mat& output);
	void filterFast_Guide1(cv::Mat& input_low, cv::Mat& guide, cv::Mat& guide_low, cv::Mat& output);

	void computerCov(std::vector<cv::Mat>& guide);
	void computeCovariance(const int depth);
	void Ip2ab_Guide3(cv::Mat& input, std::vector<cv::Mat>& guide);
	void ab2q_Guide3(std::vector<cv::Mat>& guide, cv::Mat& output);
	void filter_Guide3(cv::Mat& input, std::vector<cv::Mat>& guide, cv::Mat& output);

	void ab_up_2q_Guide3(std::vector<cv::Mat>& guide, cv::Mat& output);
	void filterFast_Guide3(cv::Mat& input_low, std::vector<cv::Mat>& guide, std::vector<cv::Mat>& guide_low, cv::Mat& output);

	void computeVarCov()override;
	virtual void filter_Guide1(cv::Mat& input, cv::Mat& output) {};
	virtual void filter_Guide3(cv::Mat& input, cv::Mat& output) {};

public:

	guidedFilter_Naive_Share(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, const cp::BoxFilterMethod _boxType, int _parallelType)
		: cp::GuidedFilterBase(_src, _guide, _dest, _r, _eps), boxType(_boxType), parallelType(_parallelType)
	{
		implementation = cp::GUIDED_NAIVE_SHARE;
	}

	void filter();
	void filterVector();
	void filterGuidePrecomputed() override;
	void upsample() override;
};