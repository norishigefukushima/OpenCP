#pragma once
#include "common.hpp"

namespace cp
{
	
	CP_EXPORT void alphaBlend(cv::InputArray src1, cv::InputArray src2, const double alpha, cv::OutputArray dest);//dest = alpha*src1 + (1-alpha)*src2
	CP_EXPORT void alphaBlend(cv::InputArray src1, cv::InputArray src2, cv::InputArray alpha, cv::OutputArray dest);//dest = alpha*src1 + (1-alpha)*src2
	CP_EXPORT void alphaBlendFixedPoint(cv::InputArray src1, cv::InputArray src2, const int alpha/*0-255*/, cv::OutputArray dest);//alpha 0-255 dest = alpha*src1 + (1-alpha)*src2
	CP_EXPORT void alphaBlendFixedPoint(cv::InputArray src1, cv::InputArray src2, cv::InputArray alpha, cv::OutputArray dest);
	CP_EXPORT cv::Mat guiAlphaBlend(cv::InputArray src1, cv::InputArray src2, bool isShowImageStats = false, std::string wname = "alphaBlend");
	
	
	CP_EXPORT void dissolveSlideBlend(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dest, const double ratio = 0.5, const double slant_ratio = 0.4, const int direction = 0, cv::Scalar line_color = cv::Scalar::all(255), const int line_thickness = 2);
	CP_EXPORT cv::Mat guiDissolveSlideBlend(cv::InputArray src1, cv::InputArray src2, std::string wname = "dissolveSlideBlend");
}