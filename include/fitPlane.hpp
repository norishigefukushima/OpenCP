#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void fitPlaneCrossProduct(std::vector<cv::Point3f>& src, cv::Point3f& dest);
	CP_EXPORT void fitPlanePCA(cv::InputArray src, cv::Point3f& dest);
	CP_EXPORT void fitPlaneRANSAC(std::vector<cv::Point3f>& src, cv::Point3f& dest, int numofsample, float threshold, int refineIter = 0);
}