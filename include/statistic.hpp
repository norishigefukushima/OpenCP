#include "common.hpp"

namespace cp
{
	CP_EXPORT void calcMinMax(cv::InputArray src, uchar& minv, uchar& maxv);
	CP_EXPORT void drawMinMaxPoints(cv::InputArray src, cv::OutputArray dest, const uchar minv = 0, const uchar maxv = 255, cv::Scalar minColor = cv::Scalar(0, 0, 255), cv::Scalar maxColor = cv::Scalar(255, 0, 0), const int circle_r = 3);
}