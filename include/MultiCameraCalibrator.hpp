#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT MultiCameraCalibrator
	{
	private:
		std::vector<std::vector<cv::Point3f>> objectPoints;
		std::vector<std::vector<std::vector<cv::Point2f>>> imagePoints;
		std::vector<cv::Point3f> chessboard3D;
		std::vector<cv::Mat> reR;
		std::vector<cv::Mat> reT;
		cv::Mat E;
		cv::Mat F;
		cv::Mat Q;
		cv::Mat intrinsicRect;

		double reprojectionerr;

		void generatechessboard3D();
		void initRemap();

	public:
		int flag;
		int numofcamera;
		cv::Size imageSize;
		std::vector<cv::Mat> intrinsic;
		std::vector<cv::Mat> distortion;

		cv::Size patternSize;
		float lengthofchess;
		int numofchessboards;

		std::vector<cv::Mat> R;
		std::vector<cv::Mat> P;
		std::vector<cv::Mat> mapu;
		std::vector<cv::Mat> mapv;

		void readParameter(char* name);
		void writeParameter(char* name);

		void init(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_, int numofcamera_);
		MultiCameraCalibrator(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess_, int numofcamera_);

		MultiCameraCalibrator();
		~MultiCameraCalibrator();

		//MultiCameraCalibrator cloneParameters();
		bool findChess(std::vector<cv::Mat>& im);
		bool findChess(std::vector<cv::Mat>& im, std::vector <cv::Mat>& dest);

		void printParameters();

		double getRectificationErrorBetween(int a, int b);
		double getRectificationErrorDisparity();
		double getRectificationErrorDisparityBetween(int ref1, int ref2);
		double getRectificationError();

		//Calibration
		void operator ()(bool isFixIntrinsic = false, int refCamera1 = 0, int refCamera2 = 0);
		void rectifyImageRemap(cv::Mat& src, cv::Mat& dest, int numofcamera);
	};
}