#pragma once
#pragma once

#include "common.hpp"

namespace cp
{
	//get online image size
	CP_EXPORT cv::Size getSubImageAlignSize(const cv::Size src, const cv::Size div_size, const int r, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1);
	CP_EXPORT cv::Size getSubImageSize(const cv::Size src, const cv::Size div_size, const int r);

	//create a divided sub image
	CP_EXPORT void createSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void createSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void createSubImageAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	//set a divided sub image to a large image
	CP_EXPORT void setSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int top, const int left);
	CP_EXPORT void setSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r);
	CP_EXPORT void setSubImageAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int left_multiple = 1, const int top_multiple = 1);

	//split an image to sub images in std::vector 
	CP_EXPORT void splitSubImage(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void splitSubImageAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int left_multiple = 1, const int top_multiple = 1);

	//merge subimages in std::vector to an image
	CP_EXPORT void mergeSubImage(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r);
	CP_EXPORT void mergeSubImageAlign(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r, const int left_multiple = 1, const int top_multiple = 1);
}