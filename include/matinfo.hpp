#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void showMatInfo(cv::InputArray src, std::string name = "Mat", const bool isStatInfo = true);
#define print_matinfo(a) cp::showMatInfo(a, #a, false)
#define print_matinfo_detail(a) cp::showMatInfo(a, #a)
}