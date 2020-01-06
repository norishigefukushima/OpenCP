#pragma once
#include "common.hpp"

namespace cp
{
	CP_EXPORT void diffshow(std::string wname, cv::InputArray src, cv::InputArray ref, const double scale = 1.0);
	CP_EXPORT void guiDiff(cv::InputArray src, cv::InputArray ref, const bool isWait = true, std::string wname = "gui::diff");
	CP_EXPORT void guiCompareDiff(const cv::Mat& before, const cv::Mat& after, const cv::Mat& ref);
	CP_EXPORT void guiAbsDiffCompareGE(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareLE(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareEQ(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareNE(const cv::Mat& src1, const cv::Mat& src2);
}
