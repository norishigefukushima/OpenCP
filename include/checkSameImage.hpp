#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT CheckSameImage
	{
	private:
		std::vector<cv::Point> positions;
		std::vector<cv::Scalar> samples;

		bool checkSamplePoints(cv::Mat& src);
		void generateRandomSamplePoints(cv::Mat& src, const int num_check_points);
	public:

		bool isSameImage(cv::Mat& src, const int num_check_points = 10);
		bool isSameImage(cv::Mat& src, cv::Mat& ref, const int num_check_points = 10);
	};

	CP_EXPORT bool checkSameImage(cv::Mat& src, cv::Mat& ref, const int num_check_points = 10);
}