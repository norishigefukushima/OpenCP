#pragma once

#include "common.hpp"
#include "Timer.hpp"

namespace cp
{
	class CP_EXPORT VideoSubtitle
	{
	private:
		cp::Timer tscript;
		double time_dissolve_start = 500.0;
		double time_dissolve_end = 1000.0;
		double time_dissolve = time_dissolve_end - time_dissolve_start;
		std::string font = "Segoe UI";
		//string font = "Consolas";
		int vspace = 20;
		cv::Mat title;

		std::vector<std::string> text;
		std::vector<int> fontSize;
		cv::Rect textROI = cv::Rect(0, 0, 0, 0);
		cv::Point textPoint = cv::Point(0, 0);
		int getAlpha();
		cv::Rect getRectText(std::vector<std::string>& text, std::vector<int>& fontSize);
		void addVText(cv::Mat& image, std::vector<std::string>& text, cv::Point point, std::vector<int>& fontSize, cv::Scalar color);
	public:
		VideoSubtitle();
		void restart();
		void setDisolveTime(const double start, const double end);
		void setFont(std::string font);
		void setVSpace(const int vspace);
		
		void setTitle(const cv::Size size, std::vector<std::string>& text, std::vector<int>& fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor = cv::Scalar::all(0));
		void showTitleDissolve(std::string wname, const cv::Mat& image);
		void showScriptDissolve(std::string wname, cv::Mat& image, const cv::Scalar textColor = cv::Scalar(255, 255, 255));	
		void showTitle(std::string wname, const cv::Size size, std::vector<std::string>& text, std::vector<int>& fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor = cv::Scalar::all(0));
		void showTitle(std::string wname, const cv::Size size, std::string text, const int fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor = cv::Scalar::all(0));
	};
}