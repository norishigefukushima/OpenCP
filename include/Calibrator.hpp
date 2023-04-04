#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void distortPoints(const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dest, const cv::Mat& intrinsic, const cv::Mat& distortion);
	CP_EXPORT void distortPoints(const std::vector<cv::Point2d>& src, std::vector<cv::Point2d>& dest, const cv::Mat& intrinsic, const cv::Mat& distortion);
	CP_EXPORT void distortOnePoint(const cv::Point2f src, cv::Point2f& dest, const cv::Mat& intrinsic, const cv::Mat& distortion);
	CP_EXPORT void distortOnePoint(const cv::Point2d src, cv::Point2d& dest, const cv::Mat& intrinsic, const cv::Mat& distortion);
	CP_EXPORT void undistortOnePoint(const cv::Point2f src, cv::Point2f& dest, const cv::Mat& intrinsic, const cv::Mat& distortion);
	CP_EXPORT void undistortOnePoint(const cv::Point2d src, cv::Point2d& dest, const cv::Mat& intrinsic, const cv::Mat& distortion);
	CP_EXPORT void drawPatternIndexNumbers(cv::Mat& dest, const std::vector<cv::Point>& points, const double scale, const cv::Scalar color);
	CP_EXPORT void drawPatternIndexNumbers(cv::Mat& dest, const std::vector<cv::Point2f>& points, const double scale, const cv::Scalar color);
	CP_EXPORT void drawDetectedPattern(const cv::Mat& src, cv::Mat& dest, const cv::Size patternSize, const std::vector<cv::Point2f>& points, const bool flag, const double numberFontSize = 0.5, const cv::Scalar numberFontColor = COLOR_ORANGE);
	CP_EXPORT void drawDistortion(cv::Mat& destImage, const cv::Mat& intrinsic, const cv::Mat& distortion, const cv::Size imageSize, const int step, const int thickness, const double amp = 1.0);
	class CP_EXPORT Calibrator
	{
	private:
		std::vector<cv::Mat> patternImages;
		std::vector<std::vector<cv::Point3f>> objectPoints;
		std::vector<std::vector<cv::Point2f>> imagePoints;
		std::vector<cv::Point3f> chessboard3D;

		void generatechessboard3D();
		void initRemap();
		bool isUseInitCameraMatrix = false;
		void drawReprojectionErrorInternal(const std::vector<std::vector<cv::Point2f>>& points,
			const std::vector <std::vector<cv::Point3f>>& objectPoints,
			const std::vector<cv::Mat>& R, const std::vector<cv::Mat>& T,
			const bool isWait = true, const std::string wname = "error", const float scale = 1000.f, const std::vector<cv::Mat>& patternImage = std::vector<cv::Mat>(), const int patternType = 0);
	public:
		cv::TermCriteria tc = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, DBL_EPSILON);
		cv::Point2f getImagePoint(const int number_of_chess, const int index = -1);
		cv::Size imageSize;
		cv::Mat intrinsic;
		cv::Mat distortion;

		cv::Size patternSize;
		cv::Size2f lengthofchess;
		int numofchessboards = 0;
		double rep_error = 0.0;

		std::vector<cv::Mat> rt;
		std::vector<cv::Mat> tv;
		int flag = cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_ASPECT_RATIO;

		cv::Mat mapu, mapv;

		void init(cv::Size imageSize_, cv::Size patternSize_, cv::Size2f lengthofchess);
		Calibrator(cv::Size imageSize_, cv::Size patternSize_, float lengthofchess);
		Calibrator(cv::Size imageSize_, cv::Size patternSize_, cv::Size2f lengthofchess);
		Calibrator();
		~Calibrator();

		void setIntrinsic(double focal_length);
		void setIntrinsic(const cv::Mat& K, const cv::Mat& distortion);
		void setInitCameraMatrix(const bool flag);
		void solvePnP(const int number_of_chess, cv::Mat& r, cv::Mat& t);
		void readParameter(char* name);
		void writeParameter(char* name);
		bool findChess(cv::Mat& im, cv::Mat& dest);
		void pushImage(const cv::Mat& patternImage);
		void pushImagePoint(const std::vector<cv::Point2f>& point);
		void pushObjectPoint(const std::vector<cv::Point3f>& point);
		void clearPatternData();

		void undistort(cv::Mat& src, cv::Mat& dest, const int interpolation = cv::INTER_LINEAR);
		double operator()();//calibrate camera
		double calibration(const int flag);

		void printParameters();
		void drawReprojectionError(std::string wname = "error", const bool isInteractive = false, const float scale = 1000.f);
		void drawReprojectionErrorFromExtraPoints(const std::vector<cv::Point2f>& points, const bool isWait = true, const std::string wname = "error", const float scale = 1000.f, const cv::Mat& patternImage = cv::Mat(), const int patternType = 0, const bool isUseInternalData = false);

		void drawDistortion(std::string wname = "distortion", const bool isInteractive = false);
	};
}