#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void cvtRAWVector2BGR(std::vector<float>& src, cv::OutputArray dest, cv::Size size);
	CP_EXPORT void cvtBGR2RawVector(cv::InputArray src, std::vector<float>& dest);


	//convert a BGR color image into a skipped one channel data: ex BGRBGRBGR... -> BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),BBBB...(cols size), GGGG....(cols size), RRRR....(cols size),...
	CP_EXPORT void splitBGRLineInterleave(cv::InputArray src, cv::OutputArray dest);

	CP_EXPORT void cvtColorBGR2PLANE(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cvtColorPLANE2BGR(cv::InputArray src, cv::OutputArray dest);

	CP_EXPORT void cvtColorBGRA2BGR(const cv::Mat& src, cv::Mat& dest);
	CP_EXPORT void cvtColorBGRA32f2BGR8u(const cv::Mat& src, cv::Mat& dest);

	CP_EXPORT void cvtColorBGR2BGRA(const cv::Mat& src, cv::Mat& dest, const uchar alpha = 255);
	CP_EXPORT void cvtColorBGR8u2BGRA32f(const cv::Mat& src, cv::Mat& dest, const float alpha = 255.f);

	CP_EXPORT void cvtColorOPP2BGR(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cvtColorBGR2OPP(cv::InputArray src, cv::OutputArray dest);
	CP_EXPORT void cvtColorMatrix(cv::InputArray src, cv::OutputArray dest, cv::InputArray C);

	//color correction colorcorrection whilebalance
	CP_EXPORT void findColorMatrixAvgStdDev(cv::InputArray ref_image, cv::InputArray target_image, cv::OutputArray colorMatrix, const double validMin, const double validMax);
}