#pragma once

#include "common.hpp"

namespace cp
{
	//slic
	CP_EXPORT void SLICSegment2Vector3D(cv::InputArray segment, cv::InputArray signal, std::vector<std::vector<cv::Point3f>>& segmentPoint);
	CP_EXPORT void SLICSegment2Vector3D(cv::InputArray segment, cv::InputArray signal, std::vector<std::vector<cv::Point3i>>& segmentPoint);
	CP_EXPORT void SLICVector2Segment(std::vector<std::vector<cv::Point>>& segmentPoint, cv::Size outputImageSize, cv::OutputArray segment);
	CP_EXPORT void SLICVector3D2Signal(std::vector<std::vector<cv::Point3f>>& segmentPoint, cv::Size outputImageSize, cv::OutputArray signal);
	CP_EXPORT void SLICSegment2Vector(cv::InputArray segment, std::vector<std::vector<cv::Point>>& segmentPoint);
	CP_EXPORT void SLIC(cv::InputArray src, cv::OutputArray segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);
	CP_EXPORT void drawSLIC(cv::InputArray src, cv::InputArray segment, cv::OutputArray dst, bool isMean = true, bool isLine = true, cv::Scalar line_color = cv::Scalar(0, 0, 255));
	CP_EXPORT void SLICBase(cv::Mat& src, cv::Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration);//not optimized code for test
}