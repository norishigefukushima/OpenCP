#include "common.hpp"

namespace cp
{
	CP_EXPORT void calcMinMax(cv::InputArray src, uchar& minv, uchar& maxv);
	CP_EXPORT void drawMinMaxPoints(cv::InputArray src, cv::OutputArray dest, const uchar minv = 0, const uchar maxv = 255, cv::Scalar minColor = cv::Scalar(0, 0, 255), cv::Scalar maxColor = cv::Scalar(255, 0, 0), const int circle_r = 3);

	CP_EXPORT double average(cv::InputArray src, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0, const bool isNormalize = true);
	CP_EXPORT void average_variance(cv::InputArray src, double& ave, double& var, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0, const bool isNormalize = true);
	CP_EXPORT void weightedAverageVariance(cv::InputArray src, cv::InputArray weight, double& ave, double& var, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0);
}