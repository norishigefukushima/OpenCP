#pragma once

#include "guidedFilter_Merge_OnePass.h"

class guidedFilter_Merge_OnePath_Fast : public guidedFilter_Merge_OnePass
{
private:
	cv::Mat src_resize;
	cv::Mat guide_resize;
	
	int row_resize;
	int col_resize;

	void filter_Guide1(int cn) override;
	void filter_Guide3(int cn) override;
public:
	guidedFilter_Merge_OnePath_Fast(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType);
};
