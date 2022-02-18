#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void highDimensionalGaussianFilterGaussianKDTree(const cv::Mat& src, cv::Mat& dest, const float sigma_color, const float sigma_space);
	CP_EXPORT void highDimensionalGaussianFilterGaussianKDTree(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, const float sigma_color, const float sigma_space);
	CP_EXPORT void highDimensionalGaussianFilterGaussianKDTreeTile(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, const float sigma_color, const float sigma_space, const cv::Size div, const float truncateBoundary = 3.f);
	CP_EXPORT void highDimensionalGaussianFilterGaussianKDTreePCATile(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, const float sigma_color, const float sigma_space, const int dest_pca_ch, const cv::Size div, const float truncateBoundary = 3.f);
}