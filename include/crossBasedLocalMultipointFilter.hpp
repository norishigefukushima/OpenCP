#pragma once

#include "common.hpp"
#include "crossBasedLocalFilter.hpp"

namespace cp
{
	class CP_EXPORT CrossBasedLocalMultipointFilter
	{
		void crossBasedLocalMultipointFilterSrc1Guidance1_(cv::Mat& src, cv::Mat& joint, cv::Mat& dest, const int radius, const float eps);
		void crossBasedLocalMultipointFilterSrc1Guidance3SSE_(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const int thresh, const float eps);
		void crossBasedLocalMultipointFilterSrc1Guidance3_(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const float eps);
		void crossBasedLocalMultipointFilterSrc1Guidance1SSE_(cv::Mat& src, cv::Mat& joint, cv::Mat& dest, const int radius, const int thresh, const float eps);

	public:
		CrossBasedLocalFilter clf;

		void operator()(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const int thresh, const float eps, bool initCLF = true);
	};

	void  CP_EXPORT crossBasedLocalMultipointFilter(cv::Mat& src, cv::Mat& guidance, cv::Mat& dest, const int radius, const int thresh, const float eps);
}