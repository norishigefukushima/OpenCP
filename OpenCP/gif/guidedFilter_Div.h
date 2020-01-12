#pragma once

#include "guidedFilter.hpp"

class guidedFilter_Div
{
private:
	cv::Mat src;
	cv::Mat guide;
	cv::Mat dest;
	int r;
	float eps;
	int parallelType;

	int divType;
	std::vector<cv::Mat> divSrc;
	std::vector<cv::Mat> divGuide;
	std::vector<cv::Mat> divDest;
	int divRow;
	int divCol;
	int padRow;
	int padCol;

	void splitImage();
	void mergeImage();
public:
	guidedFilter_Div(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType, int _divType);

	void init();
	void filter();
};