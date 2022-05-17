#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT int xmeans(cv::InputArray src, cv::OutputArray destLabels, const cv::TermCriteria criteria, const int attempts, const int flags, cv::OutputArray destCenters);
}