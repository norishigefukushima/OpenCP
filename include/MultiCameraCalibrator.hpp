#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT MultiCameraCalibrator
	{
	private:
		std::vector<cv::Point3f> chessboard3D;
		cv::Mat E;
		cv::Mat F;
		cv::Mat Q;
		cv::Mat intrinsicRect;

		double rep_error = 0.0;
		void generatechessboard3D();
		void initRemap();

	public:
		std::vector<cv::Mat> reR;
		std::vector<cv::Mat> reT;
		std::vector<std::vector<cv::Point3f>> objectPoints;//[pattern][point]
		std::vector<std::vector<std::vector<cv::Point2f>>> imagePoints;//[camera][pattarn][point]
		std::vector<std::vector<cv::Mat>> patternImages;//[camera][pattarn]
		int flag = cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_SAME_FOCAL_LENGTH | cv::CALIB_FIX_ASPECT_RATIO;
		int rect_flag = cv::CALIB_ZERO_DISPARITY;

		int numofcamera = 0;
		cv::Size imageSize;
		std::vector<cv::Mat> intrinsic;//[camera]
		std::vector<cv::Mat> distortion;//[camera]

		cv::Size patternSize;
		cv::Size2f lengthofchess;
		int numofchessboards = 0;

		std::vector<cv::Mat> R;
		std::vector<cv::Mat> P;
		std::vector<cv::Mat> mapu;
		std::vector<cv::Mat> mapv;

		void readParameter(char* name);
		void writeParameter(char* name);

		void init(cv::Size imageSize, cv::Size patternSize, cv::Size2f lengthofchess, int numofcamera);
		MultiCameraCalibrator(cv::Size imageSize, cv::Size patternSize, float lengthofchess, int numofcamera_);
		MultiCameraCalibrator(cv::Size imageSize, cv::Size patternSize, cv::Size2f lengthofchess, int numofcamera_);

		MultiCameraCalibrator();
		~MultiCameraCalibrator();

		//MultiCameraCalibrator cloneParameters();
		bool findChess(std::vector<cv::Mat>& im);
		bool findChess(std::vector<cv::Mat>& im, std::vector <cv::Mat>& dest);
		void pushImage(const std::vector<cv::Mat>& patternImage);
		//input[camera][point], but pushback [camera][patternindex][point]
		void pushImagePoint(const std::vector<std::vector<cv::Point2f>>& point);
		void pushImagePoint(const std::vector<cv::Point2f>& pointL, const std::vector<cv::Point2f>& pointR);//for stereo camera
		void pushObjectPoint(const std::vector<cv::Point3f>& point);
		void clearPatternData();


		double getRectificationErrorBetween(int a, int b);
		double getRectificationErrorDisparity();
		double getRectificationErrorDisparityBetween(int ref1, int ref2);
		double getRectificationError();

		void setIntrinsic(const cv::Mat& intrinsic, const cv::Mat& distortion, const int cameraIndex);
		void setRP(const cv::Mat& R, const cv::Mat& P, const int cameraIndex, bool isInitRemap = false);
		void setQ(const cv::Mat& Q);

		//Calibration
		double calibration(const int flags, int refCamera1 = 0, int refCamera2 = 0, const bool isIndependentCalibration = false);
		double operator ()(const int flags, int refCamera1 = 0, int refCamera2 = 0, const bool isIndependentCalibration = false);
		void rectifyImageRemap(cv::Mat& src, cv::Mat& dest, int numofcamera, const int interpolation = cv::INTER_LINEAR);

		void solvePnP(const int patternIndex, std::vector<cv::Mat>& destR, std::vector<cv::Mat>& destT);
		void printParameters();
		void drawReprojectionError(std::string wname = "error", const bool isInteractive = false, const int plotImageRadius=400);
		//return RMSE between disparityZ and patternZ
		double plotDiffDisparityZPatternZ(const int cam0, const int cam1, const double f, const double l, const double d_offset);
		void guiDisparityTest();
	};
}
