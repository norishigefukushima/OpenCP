#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT cv::Mat imreadPPMX(std::string filename);
	CP_EXPORT void imwritePPMX(std::string filename, cv::Mat& img);
#define imreadPFM imreadPPMX 
#define imwritePFM imwritePPMX 
}
