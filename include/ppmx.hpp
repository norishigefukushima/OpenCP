#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT cv::Mat imreadPPMX(std::string filename);
	CP_EXPORT cv::Mat imreadPFM(std::string filename);
}
