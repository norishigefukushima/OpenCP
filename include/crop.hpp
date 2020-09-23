#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT void cropCenter(cv::InputArray src, cv::OutputArray crop, const int window_size);
	CP_EXPORT void cropZoom(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Rect roi, const int zoom_factor = 1);
	CP_EXPORT void cropZoom(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Point center, const int window_size, const int zoom_factor = 1);
	CP_EXPORT void cropZoomWithBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Rect roi, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);
	CP_EXPORT void cropZoomWithBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Point center, const int window_size, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);
	CP_EXPORT void cropZoomWithSrcMarkAndBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, cv::OutputArray src_mark, const cv::Rect roi, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);
	CP_EXPORT void cropZoomWithSrcMarkAndBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, cv::OutputArray src_mark, const cv::Point center, const int window_size, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);

	CP_EXPORT cv::Mat guiCropZoom(cv::InputArray src, const cv::Scalar color = COLOR_RED, const int thickness = 1, const bool isWait = true, const std::string wname = "crop");
	CP_EXPORT cv::Mat guiCropZoom(cv::InputArray src, cv::Rect& dest_roi, int& dest_zoom_factor, const cv::Scalar color = COLOR_RED, const int thickness = 1, const bool isWait = true, const std::string wname = "crop");
}
