#pragma once

#include "boxFilter.hpp"

//one pass box filtering SoA 2Div

class boxFilter_OPSAT_SoA_2Div
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
	int loop;

	int divNum;
	int divRow;

	std::vector<cv::Mat> vSrc;
	std::vector<cv::Mat> vDest;

	void filter_upper_impl(cv::Mat& input, cv::Mat& output);
	void filter_lower_impl(cv::Mat& input, cv::Mat& output);
	virtual void filter_impl(int idx);

public:
	boxFilter_OPSAT_SoA_2Div(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType);
	virtual void init();

	void AoS2SoA();
	void SoA2AoS();

	void filterOnly();
	void filter();
};



class boxFilter_OPSAT_SoA_nDiv : public boxFilter_OPSAT_SoA_2Div
{
private:
	int num;

	void filter_middle_impl(cv::Mat& input, cv::Mat& output, int idx);
	void filter_impl(int idx) override;
public:
	boxFilter_OPSAT_SoA_nDiv(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType, int _num);
	void init() override;
};
