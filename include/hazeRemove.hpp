#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT HazeRemove
	{
	public:
		cv::Size size;
		cv::Mat dark;
		std::vector<cv::Mat> minvalue;
		cv::Mat tmap;
		cv::Scalar A;

		void darkChannel(cv::Mat& src, int r);
		void getAtmosphericLight(cv::Mat& srcImage, double topPercent = 0.1);
		void getTransmissionMap(float omega = 0.95f);
		void removeHaze(cv::Mat& src, cv::Mat& trans, cv::Scalar v, cv::Mat& dest, float clip = 0.3f);
		HazeRemove();
		~HazeRemove();

		void getAtmosphericLightImage(cv::Mat& dest);
		void showTransmissionMap(cv::Mat& dest, bool isPseudoColor = false);
		void showDarkChannel(cv::Mat& dest, bool isPseudoColor = false);
		void operator() (cv::Mat& src, cv::Mat& dest, int r_dark, double toprate, int r_joint, double e_joint);
		void gui(cv::Mat& src, std::string wname = "hazeRemove");
	};
}