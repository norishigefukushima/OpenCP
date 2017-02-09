#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void disp16S2depth16U(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	CP_EXPORT void disp16S2depth32F(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	CP_EXPORT void depth32F2disp8U(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	CP_EXPORT void depth16U2disp8U(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
	CP_EXPORT void disp8U2depth32F(cv::Mat& src, cv::Mat& dest, const float focal_baseline, float a = 1.f, float b = 0.f);
}