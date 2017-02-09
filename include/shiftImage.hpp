#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void warpShiftSubpix(cv::InputArray  src, cv::OutputArray dest, double shiftx, double shifty = 0, const int inter_method = cv::INTER_LANCZOS4);
	CP_EXPORT void warpShiftH(cv::InputArray src, cv::OutputArray dest, const int shiftH);
	CP_EXPORT void warpShift(cv::InputArray src, cv::OutputArray dest, int shiftx, int shifty = 0, int borderType = -1);
	CP_EXPORT cv::Mat guiShift(cv::InputArray fiximg, cv::InputArray moveimg, const int max_move = 200, std::string window_name = "Shift");
	CP_EXPORT void guiShift(cv::InputArray centerimg, cv::InputArray leftimg, cv::InputArray rightimg, int max_move, std::string window_name = "Shift");
}