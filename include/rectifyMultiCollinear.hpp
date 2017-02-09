#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT float rectifyMultiCollinear(
		const std::vector<cv::Mat>& cameraMatrix,
		const std::vector<cv::Mat>& distCoeffs,
		const int anchorView1,
		const int anchorView2,
		const std::vector<std::vector<std::vector<cv::Point2f>> >& anchorpt,
		cv::Size imageSize, const std::vector<cv::Mat>& relativeR, const std::vector<cv::Mat>& relativeT,
		std::vector<cv::Mat>& R, std::vector<cv::Mat>& P, cv::Mat& Q,
		double alpha, cv::Size newImgSize,
		cv::Rect* anchorROI1, cv::Rect* anchorROI2, int flags);
}