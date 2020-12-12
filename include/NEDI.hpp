#pragma once
#include<opencv2/core.hpp>

//X. Li and M.T.Orchard
//New edge-directed interpolation
//IEEE Transactions on Image Processing, vol. 10, issue 10, 2001.

namespace cp
{
	class CP_EXPORT NewEdgeDirectedInterpolation
	{
		void upsampleGrayDoubleLUOpt(const cv::Mat& sim, cv::Mat& dim, const float threshold, const int window_size, const float eps);
		void upsampleGrayDoubleLU(const cv::Mat& sim, cv::Mat& dim, const float threshold, const int window_size, const float eps);
		void upsampleGrayDoubleQR(const cv::Mat& sim, cv::Mat& dim, const float threshold, const int window_size);

		std::vector<cv::Mat> dest_border;
		std::vector<cv::Mat> image_border;
	public:
		void upsample(cv::InputArray src, cv::OutputArray dest, const int scale, const float threshold, const int WindowSize, int method);
	};
}