#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT OpticalFlowBM
	{
		cv::Mat buffSpeckle;
	public:
		std::vector<cv::Mat>cost;
		std::vector<cv::Mat>ocost;
		OpticalFlowBM();
		void cncheck(cv::Mat& srcx, cv::Mat& srcy, cv::Mat& destx, cv::Mat& desty, int thresh, int invalid);
		void operator()(cv::Mat& curr, cv::Mat& next, cv::Mat& dstx, cv::Mat& dsty, cv::Size ksize, int minx, int maxx, int miny, int maxy, int bd = 30);
	};

	CP_EXPORT void drawOpticalFlow(const cv::Mat_<cv::Point2f>& flow, cv::Mat& dst, float maxmotion = -1);
	CP_EXPORT void mergeFlow(cv::Mat& flow, cv::Mat& xflow, cv::Mat& yflow);
}