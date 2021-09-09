#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT Calibrator
	{
	private:
		std::vector<std::vector<cv::Point3f>> objectPoints;
		std::vector<std::vector<cv::Point2f>> imagePoints;
		std::vector<cv::Point3f> chessboard3D;

		void generatechessboard3D();
		void initRemap();

	public:
		cv::Point2f getImagePoint(const int number_of_chess, const int index = -1);
		cv::Size imageSize;
		cv::Mat intrinsic;
		cv::Mat distortion;

		cv::Size patternSize;
		float lengthofchess;
		int numofchessboards;
		double rep_error;

		std::vector<cv::Mat> rt;
		std::vector<cv::Mat> tv;
		int flag;

		cv::Mat mapu, mapv;

		void init(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_);
		Calibrator(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_);
		Calibrator();
		~Calibrator();

		void setIntrinsic(double focal_length);
		void solvePnP(const int number_of_chess, cv::Mat& r, cv::Mat& t);
		void readParameter(char* name);
		void writeParameter(char* name);
		bool findChess(cv::Mat& im, cv::Mat& dest);
		void pushImagePoint(std::vector<cv::Point2f> point);
		void pushObjectPoint(std::vector<cv::Point3f> point);
		void undistort(cv::Mat& src, cv::Mat& dest);
		void printParameters();
		double operator()();//calibrate camera
		double calibration(const int flag);
	};
}