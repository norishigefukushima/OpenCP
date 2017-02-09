#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT PostFilterSet
	{
		cv::Mat buff, bufff;
	public:
		PostFilterSet();
		~PostFilterSet();
		void filterDisp8U2Depth32F(cv::Mat& src, cv::Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method = FILTER_DEFAULT);
		void filterDisp8U2Depth16U(cv::Mat& src, cv::Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method = FILTER_DEFAULT);
		void filterDisp8U2Disp32F(cv::Mat& src, cv::Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method = FILTER_DEFAULT);
		void operator()(cv::Mat& src, cv::Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, int brange_th, int brange_method = FILTER_DEFAULT);
	};
}
