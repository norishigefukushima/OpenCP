#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT HazeRemove
	{
	
		cv::Size size;
		cv::Mat dark;
		std::vector<cv::Mat> minvalue;
		cv::Mat tmap;
		cv::Scalar A;

		void darkChannel(cv::Mat& src, int r);
		void getAtmosphericLight(cv::Mat& srcImage, double topPercent = 0.1);
		void getTransmissionMap(float omega = 0.95f);
		void removeHaze(cv::Mat& src, cv::Mat& trans, cv::Scalar v, cv::Mat& dest, float clip = 0.3f);
		
	public:
		HazeRemove();
		~HazeRemove();
		void getAtmosphericLightImage(cv::Mat& dest);
		
		void showTransmissionMap(cv::Mat& dest, bool isPseudoColor = false);
		void showDarkChannel(cv::Mat& dest, bool isPseudoColor = false);
		
		void removeFastGlobalSmootherFilter(cv::Mat& src, cv::Mat& dest, const int r_dark, const double top_rate, const double lambda, const double sigma_color, const double lambda_attenuation, const int iteration);
		void removeGuidedFilter(cv::Mat& src, cv::Mat& dest, const int r_dark, const double toprate, const int r_joint, const double e_joint);
		void operator() (cv::Mat& src, cv::Mat& dest, const int r_dark, const double toprate, const int r_joint, const double e_joint);

		void gui(cv::Mat& src, std::string wname = "hazeRemove");
	};
}