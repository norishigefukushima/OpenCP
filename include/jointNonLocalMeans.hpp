#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void jointNonLocalMeansFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void jointNonLocalMeansFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma, int method = FILTER_DEFAULT, int borderType = cv::BORDER_REPLICATE);
	CP_EXPORT void weightedJointNonLocalMeansFilter(cv::Mat& src, cv::Mat& weightMap, cv::Mat& guide, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma);
}