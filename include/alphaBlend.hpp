#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT void alphaBlend(cv::InputArray src1, cv::InputArray src2, const double alpha, cv::OutputArray dest);
	CP_EXPORT void alphaBlend(const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& alpha, cv::Mat& dest);
	CP_EXPORT void alphaBlendApproxmate(cv::InputArray src1, cv::InputArray src2, cv::InputArray alpha, cv::OutputArray dest);
	CP_EXPORT void alphaBlendApproxmate(cv::InputArray src1, cv::InputArray src2, const uchar alpha, cv::OutputArray dest);
	CP_EXPORT void guiAlphaBlend(cv::InputArray src1, cv::InputArray src2, bool isShowImageStats = false);
}