#include "Calibrator.hpp"
#include "draw.hpp"
#include "debugcp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
#pragma region util
	void undistortOnePoint(const Point2f src, Point2f& dest, const Mat& intrinsic, const Mat& distortion)
	{
		vector<Point2f> s;
		vector<Point2f> d;
		s.push_back(src);
		undistortPoints(s, d, intrinsic, distortion, Mat::eye(3, 3, CV_64F), intrinsic,
			cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, DBL_EPSILON));
		dest = d[0];
	}

	void undistortOnePoint(const Point2d src, Point2d& dest, const Mat& intrinsic, const Mat& distortion)
	{
		vector<Point2d> s;
		vector<Point2d> d;
		s.push_back(src);
		undistortPoints(s, d, intrinsic, distortion, Mat::eye(3, 3, CV_64F), intrinsic,
			cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, DBL_EPSILON));
		dest = d[0];
	}

	template <typename FLOAT>
	void computeTiltProjectionMatrix(FLOAT tauX,
		FLOAT tauY,
		Matx<FLOAT, 3, 3>* matTilt)
	{
		FLOAT cTauX = std::cos(tauX);
		FLOAT sTauX = std::sin(tauX);
		FLOAT cTauY = std::cos(tauY);
		FLOAT sTauY = std::sin(tauY);
		Matx<FLOAT, 3, 3> matRotX = Matx<FLOAT, 3, 3>(1, 0, 0, 0, cTauX, sTauX, 0, -sTauX, cTauX);
		Matx<FLOAT, 3, 3> matRotY = Matx<FLOAT, 3, 3>(cTauY, 0, -sTauY, 0, 1, 0, sTauY, 0, cTauY);
		Matx<FLOAT, 3, 3> matRotXY = matRotY * matRotX;
		Matx<FLOAT, 3, 3> matProjZ = Matx<FLOAT, 3, 3>(matRotXY(2, 2), 0, -matRotXY(0, 2), 0, matRotXY(2, 2), -matRotXY(1, 2), 0, 0, 1);
		if (matTilt)
		{
			// Matrix for trapezoidal distortion of tilted image sensor
			*matTilt = matProjZ * matRotXY;
		}
	}

	void distortOnePoint(const Point2f src, Point2f& dest, const Mat& intrinsic, const Mat& distortion)
	{
		const double fx = intrinsic.at<double>(0, 0);
		const double fy = intrinsic.at<double>(1, 1);
		const double cx = intrinsic.at<double>(0, 2);
		const double cy = intrinsic.at<double>(1, 2);
		const double k1 = distortion.at<double>(0);
		const double k2 = distortion.at<double>(1);
		const double k3 = distortion.at<double>(4);
		const double p1 = distortion.at<double>(2);
		const double p2 = distortion.at<double>(3);
		const double xp = (src.x - cx) / fx;
		const double yp = (src.y - cy) / fy;
		const double r2 = xp * xp + yp * yp;
		const double r4 = r2 * r2;
		const double r6 = r2 * r4;
		if (distortion.size().area() == 5)
		{
			const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
			const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;
			dest.x = float(xpp * fx + cx);
			dest.y = float(ypp * fy + cy);
		}
		else if (distortion.size().area() <= 12)
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);

			const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
			const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;
			dest.x = float(xpp * fx + cx);
			dest.y = float(ypp * fy + cy);
		}
		else
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);
			const double tx = distortion.at<double>(12);
			const double ty = distortion.at<double>(13);

			const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
			const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;

			Matx33d matTilt = Matx33d::eye();
			computeTiltProjectionMatrix(tx, ty, &matTilt);

			// additional distortion by projecting onto a tilt plane
			Vec3d vecTilt = matTilt * Vec3d(xpp, ypp, 1);
			double invProj = vecTilt(2) ? 1. / vecTilt(2) : 1;
			const double xppt = invProj * vecTilt(0);
			const double yppt = invProj * vecTilt(1);

			dest.x = saturate_cast<float>(xppt * fx + cx);
			dest.y = saturate_cast<float>(yppt * fy + cy);
		}
	}

	void distortOnePoint(const Point2d src, Point2d& dest, const Mat& intrinsic, const Mat& distortion)
	{
		const double fx = intrinsic.at<double>(0, 0);
		const double fy = intrinsic.at<double>(1, 1);
		const double cx = intrinsic.at<double>(0, 2);
		const double cy = intrinsic.at<double>(1, 2);
		const double k1 = distortion.at<double>(0);
		const double k2 = distortion.at<double>(1);
		const double k3 = distortion.at<double>(4);
		const double p1 = distortion.at<double>(2);
		const double p2 = distortion.at<double>(3);
		const double xp = (src.x - cx) / fx;
		const double yp = (src.y - cy) / fy;
		const double r2 = xp * xp + yp * yp;
		const double r4 = r2 * r2;
		const double r6 = r2 * r4;
		if (distortion.size().area() == 5)
		{
			const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
			const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;
			dest.x = xpp * fx + cx;
			dest.y = ypp * fy + cy;
		}
		else if (distortion.size().area() <= 12)
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);

			const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
			const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;
			dest.x = xpp * fx + cx;
			dest.y = ypp * fy + cy;
		}
		else
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);
			const double tx = distortion.at<double>(12);
			const double ty = distortion.at<double>(13);

			const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
			const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;

			Matx33d matTilt = Matx33d::eye();
			computeTiltProjectionMatrix(tx, ty, &matTilt);

			// additional distortion by projecting onto a tilt plane
			Vec3d vecTilt = matTilt * Vec3d(xpp, ypp, 1);
			double invProj = vecTilt(2) ? 1.0 / vecTilt(2) : 1.0;
			const double xppt = invProj * vecTilt(0);
			const double yppt = invProj * vecTilt(1);

			dest.x = xppt * fx + cx;
			dest.y = yppt * fy + cy;
		}
	}

	void distortPoints(const vector<Point2f>& src, vector<Point2f>& dest, const Mat& intrinsic, const Mat& distortion)
	{
		const double fx = intrinsic.at<double>(0, 0);
		const double fy = intrinsic.at<double>(1, 1);
		const double cx = intrinsic.at<double>(0, 2);
		const double cy = intrinsic.at<double>(1, 2);
		const double k1 = distortion.at<double>(0);
		const double k2 = distortion.at<double>(1);
		const double k3 = distortion.at<double>(4);
		const double p1 = distortion.at<double>(2);
		const double p2 = distortion.at<double>(3);
		dest.resize(src.size());
		if (distortion.size().area() == 5)
		{
			for (int i = 0; i < src.size(); i++)
			{
				const double xp = (src[i].x - cx) / fx;
				const double yp = (src[i].y - cy) / fy;
				const double r2 = xp * xp + yp * yp;
				const double r4 = r2 * r2;
				const double r6 = r2 * r4;
				const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
				const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;
				dest[i].x = saturate_cast<float>(xpp * fx + cx);
				dest[i].y = saturate_cast<float>(ypp * fy + cy);
			}
		}
		else if (distortion.size().area() <= 12)
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);
			for (int i = 0; i < src.size(); i++)
			{
				const double xp = (src[i].x - cx) / fx;
				const double yp = (src[i].y - cy) / fy;
				const double r2 = xp * xp + yp * yp;
				const double r4 = r2 * r2;
				const double r6 = r2 * r4;
				const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
				const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;
				dest[i].x = saturate_cast<float>(xpp * fx + cx);
				dest[i].y = saturate_cast<float>(ypp * fy + cy);
			}
		}
		else
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);
			const double tx = distortion.at<double>(12);
			const double ty = distortion.at<double>(13);
			Matx33d matTilt = Matx33d::eye();
			computeTiltProjectionMatrix(tx, ty, &matTilt);
			for (int i = 0; i < src.size(); i++)
			{
				const double xp = (src[i].x - cx) / fx;
				const double yp = (src[i].y - cy) / fy;
				const double r2 = xp * xp + yp * yp;
				const double r4 = r2 * r2;
				const double r6 = r2 * r4;
				const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
				const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;

				// additional distortion by projecting onto a tilt plane
				Vec3d vecTilt = matTilt * Vec3d(xpp, ypp, 1);
				double invProj = vecTilt(2) ? 1. / vecTilt(2) : 1;
				const double xppt = invProj * vecTilt(0);
				const double yppt = invProj * vecTilt(1);

				dest[i].x = saturate_cast<float>(xppt * fx + cx);
				dest[i].y = saturate_cast<float>(yppt * fy + cy);
			}
		}
	}

	void distortPoints(const vector<Point2d>& src, vector<Point2d>& dest, const Mat& intrinsic, const Mat& distortion)
	{
		const double fx = intrinsic.at<double>(0, 0);
		const double fy = intrinsic.at<double>(1, 1);
		const double cx = intrinsic.at<double>(0, 2);
		const double cy = intrinsic.at<double>(1, 2);
		const double k1 = distortion.at<double>(0);
		const double k2 = distortion.at<double>(1);
		const double k3 = distortion.at<double>(4);
		const double p1 = distortion.at<double>(2);
		const double p2 = distortion.at<double>(3);
		dest.resize(src.size());
		if (distortion.size().area() == 5)
		{
			for (int i = 0; i < src.size(); i++)
			{
				const double xp = (src[i].x - cx) / fx;
				const double yp = (src[i].y - cy) / fy;
				const double r2 = xp * xp + yp * yp;
				const double r4 = r2 * r2;
				const double r6 = r2 * r4;
				const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
				const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;
				dest[i].x = xpp * fx + cx;
				dest[i].y = ypp * fy + cy;
			}
		}
		else if (distortion.size().area() <= 12)
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);
			for (int i = 0; i < src.size(); i++)
			{
				const double xp = (src[i].x - cx) / fx;
				const double yp = (src[i].y - cy) / fy;
				const double r2 = xp * xp + yp * yp;
				const double r4 = r2 * r2;
				const double r6 = r2 * r4;
				const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
				const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;
				dest[i].x = xpp * fx + cx;
				dest[i].y = ypp * fy + cy;
			}
		}
		else
		{
			const double k4 = distortion.at<double>(5);
			const double k5 = distortion.at<double>(6);
			const double k6 = distortion.at<double>(7);
			const double s1 = distortion.at<double>(8);
			const double s2 = distortion.at<double>(9);
			const double s3 = distortion.at<double>(10);
			const double s4 = distortion.at<double>(11);
			const double tx = distortion.at<double>(12);
			const double ty = distortion.at<double>(13);
			Matx33d matTilt = Matx33d::eye();
			computeTiltProjectionMatrix(tx, ty, &matTilt);
			for (int i = 0; i < src.size(); i++)
			{
				const double xp = (src[i].x - cx) / fx;
				const double yp = (src[i].y - cy) / fy;
				const double r2 = xp * xp + yp * yp;
				const double r4 = r2 * r2;
				const double r6 = r2 * r4;
				const double xpp = xp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4;
				const double ypp = yp * (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6) + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp + s3 * r2 + s4 * r4;

				// additional distortion by projecting onto a tilt plane
				Vec3d vecTilt = matTilt * Vec3d(xpp, ypp, 1);
				double invProj = vecTilt(2) ? 1. / vecTilt(2) : 1;
				const double xppt = invProj * vecTilt(0);
				const double yppt = invProj * vecTilt(1);

				dest[i].x = xppt * fx + cx;
				dest[i].y = yppt * fy + cy;
			}
		}
	}

	void drawPatternIndexNumbers(cv::Mat& dest, const std::vector<cv::Point>& points, const double scale, const Scalar color)
	{
		for (int i = 0; i < points.size(); i++)
		{
			drawPlus(dest, points[i], 1);
			cv::putText(dest, format("%d", i), points[i], cv::FONT_HERSHEY_SIMPLEX, scale, color);
		}
	}

	void drawPatternIndexNumbers(cv::Mat& dest, const std::vector<cv::Point2f>& points, const double scale, const Scalar color)
	{
		const int size = (int)points.size();
		vector<Point> p(size);
		for (int i = 0; i < size; i++)
		{
			p[i] = points[i];
		}
		drawPatternIndexNumbers(dest, p, scale, color);
	}

	void drawDetectedPattern(const cv::Mat& src, cv::Mat& dest, const cv::Size patternSize, const std::vector<cv::Point2f>& points, const bool flag, const double numberFontSize, const cv::Scalar numberFontColor)
	{
		if (src.channels() == 3)src.copyTo(dest);
		else cvtColor(src, dest, COLOR_GRAY2BGR);
		cv::drawChessboardCorners(dest, patternSize, points, flag);
		drawPatternIndexNumbers(dest, points, numberFontSize, numberFontColor);
	}

	void drawDistortionLine(const Mat& mapu, const Mat& mapv, Mat& dest, const int step, const int thickness, const float amp = 1.f)
	{
		dest.create(mapu.size(), CV_8UC3);
		dest.setTo(50);
		/*double minvx, maxvx;
		double minvy, maxvy;
		minMaxLoc(mapu, &minvx, &maxvx);
		minMaxLoc(mapv, &minvy, &maxvy);*/

		const int st = max(step, 1);
		int bb = 0;
		const int width = mapu.cols;
		const int height = mapu.rows;
		for (int j = bb; j < height - bb; j += st)
		{
			for (int i = bb; i < width - bb; i += st)
			{
				const Point s = Point(i, j);
				const Point2f diff = Point2f(mapu.at<float>(j, i), mapv.at<float>(j, i)) - Point2f(float(i), float(j));
				const Point d = Point(Point2f(s) + amp * diff);
				//print_debug3(mapu.at<float>(j, i), i, mapu.at<float>(j, i) - i);

				//cv::circle(dest, s, thickness, COLOR_ORANGE, cv::FILLED);
				cv::line(dest, s, d, COLOR_ORANGE, max(thickness, 1));
			}
		}
	}

	void drawDistortion(Mat& destImage, const Mat& intrinsic, const Mat& distortion, const Size imageSize, const int step, const int thickness, const double amp)
	{
		Mat mapu, mapv;
		initUndistortRectifyMap(intrinsic, distortion, Mat::eye(3, 3, CV_64F), intrinsic, imageSize, CV_32FC1, mapu, mapv);
		cp::drawDistortionLine(mapu, mapv, destImage, step, thickness, amp);
	}
#pragma endregion

	void Calibrator::readParameter(char* name)
	{
		FileStorage fs(name, FileStorage::READ);
		if (!fs.isOpened())
		{
			std::cout << "invalid file name: " << name << std::endl;
			return;
		}

		fs["intrinsic"] >> intrinsic;
		fs["distortion"] >> distortion;

		initRemap();
	}

	void Calibrator::writeParameter(char* name)
	{
		FileStorage fs(name, FileStorage::WRITE);

		fs << "intrinsic" << intrinsic;
		fs << "distortion" << distortion;
	}

	void Calibrator::generatechessboard3D()
	{
		for (int j = 0; j < patternSize.height; ++j)
		{
			for (int i = 0; i < patternSize.width; ++i)
			{
				chessboard3D.push_back(Point3f(lengthofchess.width * (float)i, lengthofchess.height * (float)j, 0.0));
			}
		}
	}

	void Calibrator::initRemap()
	{
		Mat P = Mat::eye(3, 3, CV_64F);
		Mat R = Mat::eye(3, 3, CV_64F);
		//print_debug(distortion);
		initUndistortRectifyMap(intrinsic, distortion, R, intrinsic, imageSize, CV_32FC1,
			mapu, mapv);
	}

	void Calibrator::init(Size imageSize_, Size patternSize_, Size2f lengthofchess)
	{
		numofchessboards = 0;
		imageSize = imageSize_;
		patternSize = patternSize_;
		this->lengthofchess = lengthofchess;
		distortion = Mat::zeros(1, 8, CV_64F);
		intrinsic = Mat::eye(3, 3, CV_64F);
		//getDefaultNewCameraMatrix(intrinsic, imageSize, true);
		generatechessboard3D();
	}

	Calibrator::Calibrator(Size imageSize_, Size patternSize_, float lengthofchess)
	{
		init(imageSize_, patternSize_, Size2f(lengthofchess, lengthofchess));
	}

	Calibrator::Calibrator(Size imageSize_, Size patternSize_, Size2f lengthofchess)
	{
		init(imageSize_, patternSize_, lengthofchess);
	}

	Calibrator::Calibrator() { ; }

	Calibrator::~Calibrator() { ; }

	Point2f Calibrator::getImagePoint(const int number_of_chess, const int index)
	{
		if (index<0 || index>patternSize.area())
		{
			float x = 0.f;
			float y = 0.f;
			for (int i = 0; i < patternSize.area(); i++)
			{
				x += imagePoints[number_of_chess][i].x;
				y += imagePoints[number_of_chess][i].y;
			}
			return Point2f((float)x / patternSize.area(), (float)y / patternSize.area());
		}
		else
		{
			return Point2f(imagePoints[number_of_chess][index].x, imagePoints[number_of_chess][index].y);
		}
	}

	void Calibrator::setIntrinsic(double focal_length)
	{
		intrinsic.at<double>(0, 0) = focal_length;
		intrinsic.at<double>(1, 1) = focal_length;
		intrinsic.at<double>(0, 2) = (imageSize.width - 1.0) / 2.0;
		intrinsic.at<double>(1, 2) = (imageSize.height - 1.0) / 2.0;
	}

	void Calibrator::setIntrinsic(const Mat& K, const Mat& distortion)
	{
		K.copyTo(this->intrinsic);
		distortion.copyTo(this->distortion);
		initRemap();
	}

	void Calibrator::setInitCameraMatrix(const bool flag)
	{
		this->isUseInitCameraMatrix = flag;
	}

	void Calibrator::solvePnP(const int number_of_chess, Mat& r, Mat& t)
	{
		cv::solvePnP(objectPoints[number_of_chess], imagePoints[number_of_chess], intrinsic, distortion, r, t);
		//cout<<format(t,"python");
	}

	bool Calibrator::findChess(Mat& im, Mat& dest)
	{
		vector<Point2f> tmp;

		bool ret = findChessboardCorners(im, patternSize, tmp, CALIB_CB_FAST_CHECK);
		if (!ret)return false;
		ret = findChessboardCorners(im, patternSize, tmp);
		if (!ret) ret = findChessboardCorners(im, patternSize, tmp, CALIB_CB_ADAPTIVE_THRESH);

		im.copyTo(dest);

		drawChessboardCorners(dest, patternSize, Mat(tmp), ret);

		if (ret)
		{
			numofchessboards++;
			imagePoints.push_back(tmp);
			objectPoints.push_back(chessboard3D);
		}
		return ret;
	}

	void Calibrator::clearPatternData()
	{
		patternImages.resize(0);
		imagePoints.resize(0);
		objectPoints.resize(0);
		numofchessboards = 0;
	}

	void Calibrator::pushImage(const cv::Mat& patternImage)
	{
		patternImages.push_back(patternImage);
	}

	void Calibrator::pushImagePoint(const vector<Point2f>& point)
	{
		numofchessboards++;
		imagePoints.push_back(point);
	}

	void Calibrator::pushObjectPoint(const vector<Point3f>& point)
	{
		objectPoints.push_back(point);
	}

	void check(InputArrayOfArrays a)
	{
		cout << a.total() << endl;
		Mat b = a.getMat();
	}

	void Calibrator::undistort(Mat& src, Mat& dest, const int interpolation)
	{
		if (mapu.empty())
		{
			cout << "calibration is not ready" << endl;
			return;
		}
		if (mapv.empty())
		{
			cout << "calibration is not ready" << endl;
			return;
		}
		remap(src, dest, mapu, mapv, interpolation);
	}

	double Calibrator::operator()()
	{
		if (numofchessboards < 2)
		{
			std::cout << "input 3 or more chessboards" << std::endl;
			return -1;
		}
		if (isUseInitCameraMatrix)
		{
			Mat intrinsic_local = initCameraMatrix2D(objectPoints, imagePoints, imageSize);
			intrinsic_local.copyTo(intrinsic);
			rep_error = calibrateCamera(objectPoints, imagePoints, imageSize, intrinsic, distortion, rt, tv, flag | CALIB_USE_INTRINSIC_GUESS, tc);
		}
		else
		{
			rep_error = calibrateCamera(objectPoints, imagePoints, imageSize, intrinsic, distortion, rt, tv, flag, tc);
		}
		//vector<vector<Point3f>> n;
		//rep_error = calibrateCameraRO(objectPoints, imagePoints, imageSize, 0, intrinsic, distortion, rt, tv, n, flag);
		initRemap();
		return rep_error;
	}

	double Calibrator::calibration(const int flag)
	{
		this->flag = flag;
		return operator()();
	}

	void Calibrator::printParameters()
	{
		std::cout << "intrinsic" << std::endl;
		cout << intrinsic << endl;
		cout << distortion << endl;
	}

	enum
	{
		All,
		QuadCorner,
		Boundary,
		Center4,
	};

	void extractPatternCorner(const vector<Point2f>& src, const Size patternSize, vector<Point2f>& dest, const int method)
	{
		dest.clear();
		if (method == All)
		{
			for (int i = 0; i < patternSize.area(); i++)
			{
				dest.push_back(src[i]);
			}
		}
		if (method == QuadCorner)
		{
			dest.push_back(src[0]);
			dest.push_back(src[patternSize.width - 1]);
			dest.push_back(src[patternSize.area() - 1]);
			dest.push_back(src[patternSize.area() - patternSize.width]);
		}
		if (method == Boundary)
		{
			for (int j = 0; j < patternSize.height; j++)
			{
				for (int i = 0; i < patternSize.width; i++)
				{
					if (i == 0 || i == patternSize.width - 1 ||
						j == 0 || j == patternSize.height - 1)
					{
						dest.push_back(src[patternSize.width * j + i]);
					}
				}
			}
		}
		if (method == Center4)
		{
			int jc = (patternSize.height - 1) / 2;
			int ic = (patternSize.width - 1) / 2;
			dest.push_back(src[(int)patternSize.width * (jc + 0) + (ic + 0)]);
			dest.push_back(src[(int)patternSize.width * (jc + 0) + (ic + 1)]);
			dest.push_back(src[(int)patternSize.width * (jc + 1) + (ic + 0)]);
			dest.push_back(src[(int)patternSize.width * (jc + 1) + (ic + 1)]);
		}
	}

	void Calibrator::drawReprojectionErrorInternal(const vector<vector<Point2f>>& points,
		const vector<vector<Point3f>>& objectPoints,
		const vector<Mat>& R, const vector<Mat>& T,
		const bool isWait, const std::string wname, const float scale, const std::vector<cv::Mat>& patternImage_, const int patternType)
	{
		string wnameplot = wname + ": Pattern";
		if (isWait)
		{
			if (isWait) destroyWindow(wname);
			if (isWait) destroyWindow(wnameplot);
		}
		int length = 400;
		Size imsize(2 * length + 1, 2 * length + 1);
		Point center(imsize.width / 2, imsize.height / 2);
		Mat repErrorShow(imsize, CV_8UC3);
		repErrorShow.setTo(255);
		Mat repPatternShow(imageSize, CV_8UC3);

		double terror = 0.0;
		Mat_<uchar> graybar = Mat(256, 1, CV_8U);
		Mat colorbar;
		for (int i = 0; i < 256; i++) graybar(i) = i;
		cv::applyColorMap(graybar, colorbar, COLORMAP_JET);

		int step = 256 / (int)chessboard3D.size();

		const int fontType = FONT_HERSHEY_SIMPLEX;

		namedWindow(wnameplot);
		static int escale = 200; createTrackbar("escale", wnameplot, &escale, 1000);//escale
		static int view = 1; createTrackbar("view", wnameplot, &view, 2);
		float pstep = float(round(imageSize.width / (patternSize.width + 2)));
		static int errorCircle = 1; createTrackbar("error circle", wnameplot, &errorCircle, 1);
		static int dist = 1; createTrackbar("undistort", wnameplot, &dist, 1);
		static int grid_r = cvRound(pstep * 0.25); createTrackbar("grid radius", wnameplot, &grid_r, grid_r * 5);
		static int isGrid = 1; createTrackbar("is grid", wnameplot, &isGrid, 1);

		namedWindow(wname);
		const int thmax = 10000;
		static int thresh = thmax; createTrackbar("threshold*0.0001(max 1pixel)", wname, &thresh, thmax);//max 1pixel
		static int indexGUI = (int)objectPoints.size(); createTrackbar("pattern index", wname, &indexGUI, (int)objectPoints.size());
		static int gix = patternSize.width; createTrackbar("grid index X", wname, &gix, patternSize.width);
		static int giy = patternSize.height; createTrackbar("grid index Y", wname, &giy, patternSize.height);
		static int gbb = 0; createTrackbar("grid bb", wname, &gbb, min(patternSize.width, patternSize.height) / 2);
		static int sw = 0; createTrackbar("sw", wname, &sw, 1);
		static int ColorMap = COLORMAP_JET; createTrackbar("ColorMap", wname, &ColorMap, COLORMAP_DEEPGREEN);
		static int scalei = int(scale); createTrackbar("scale", wname, &scalei, scalei * 2);
		static int errorrf = 0; createTrackbar("rep-front", wname, &errorrf, 1);
		static int isFixCornerH = 0; createTrackbar("isFixCornerH", wname, &isFixCornerH, 3);

		const int circle_rad = 9;
#pragma region generate_data

		vector<Point2f> pointsConic;
		vector<vector<Point2f>> pointsWarpConic(patternImage_.size());
		vector<vector<Point2f>> pointsUndistort(patternImage_.size());
		vector<vector<Point2f>> pointsUndistortConic(patternImage_.size());
		vector<vector<Point2f>> reprojectPoints(patternImage_.size());

		const bool isReverse = (points[0][0].x > points[0][patternSize.width - 1].x);
		if (isReverse)
		{
			for (int j = patternSize.height - 1; j >= 0; j--)
			{
				for (int i = patternSize.width - 1; i >= 0; i--)
				{
					pointsConic.push_back(Point2f((i + 1) * pstep, (j + 1) * pstep));
				}
			}
		}
		else
		{
			for (int j = 0; j < patternSize.height; j++)
			{
				for (int i = 0; i < patternSize.width; i++)
				{
					pointsConic.push_back(Point2f((i + 1) * pstep, (j + 1) * pstep));
				}
			}
		}


		vector<Mat> image(patternImage_.size());
		vector<Mat> imageUndist(patternImage_.size());
		vector<Mat> imageCircle(patternImage_.size());

		vector<Mat> imageConic(patternImage_.size());
		vector<Mat> imageUndistConic(patternImage_.size());
		vector<Mat> imageCircleConic(patternImage_.size());
		vector<Mat> imageCircleUndistConic(patternImage_.size());

		if (patternImage_.size() != 0)
		{
#pragma omp parallel for
			for (int pindex = 0; pindex < patternImage_.size(); pindex++)
			{
				if (patternImage_[pindex].channels() == 3)	image[pindex] = patternImage_[pindex];
				else cvtColor(patternImage_[pindex], image[pindex], COLOR_GRAY2BGR);

				imageCircle[pindex] = image[pindex].clone();
				undistort(image[pindex], imageUndist[pindex]);
				//draw circles
				for (int n = 0; n < patternSize.height; n++)
				{
					for (int m = 0; m < patternSize.width; m++)
					{
						cv::circle(imageCircle[pindex], Point(points[pindex][patternSize.width * n + m]), circle_rad, COLOR_WHITE, cv::FILLED);
					}
				}
				cp::drawPatternIndexNumbers(imageCircle[pindex], points[pindex], 2, COLOR_BLUE);

				//warp conic view
				vector<Point2f> cornerQuadConic;
				extractPatternCorner(pointsConic, patternSize, cornerQuadConic, isFixCornerH);
				vector<Point2f> cornerQuad;
				extractPatternCorner(points[pindex], patternSize, cornerQuad, isFixCornerH);
				const Mat H = findHomography(cornerQuad, cornerQuadConic);
				perspectiveTransform(points[pindex], pointsWarpConic[pindex], H);

				undistortPoints(points[pindex], pointsUndistort[pindex], intrinsic, distortion, Mat::eye(3, 3, CV_64F), intrinsic,
					cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.0001));

				extractPatternCorner(pointsUndistort[pindex], patternSize, cornerQuad, isFixCornerH);
				const Mat Hundistort = findHomography(cornerQuad, cornerQuadConic);
				perspectiveTransform(pointsUndistort[pindex], pointsUndistortConic[pindex], Hundistort);

				warpPerspective(image[pindex], imageConic[pindex], H, image[pindex].size());
				warpPerspective(imageCircle[pindex], imageCircleConic[pindex], H, image[pindex].size());
				warpPerspective(imageUndist[pindex], imageUndistConic[pindex], Hundistort, image[pindex].size());
				Mat temp;  undistort(imageCircle[pindex], temp);
				warpPerspective(temp, imageCircleUndistConic[pindex], Hundistort, image[pindex].size());

				projectPoints(objectPoints[pindex], R[pindex], T[pindex], intrinsic, distortion, reprojectPoints[pindex]);
			}
		}
#pragma endregion
		Scalar color = COLOR_RED;
		Scalar colorArrow = COLOR_ORANGE;
		Scalar colorErrorCircle = COLOR_CYAN;
		int key = 0;

		cp::UpdateCheck uc(isFixCornerH);
		while (key != 'q')
		{
#pragma region updateH
			if (uc.isUpdate(isFixCornerH))
			{
				if (patternImage_.size() != 0)
				{
#pragma omp parallel for
					for (int pindex = 0; pindex < patternImage_.size(); pindex++)
					{
						//warp conic view
						vector<Point2f> cornerQuadConic;
						extractPatternCorner(pointsConic, patternSize, cornerQuadConic, isFixCornerH);
						vector<Point2f> cornerQuad;
						extractPatternCorner(points[pindex], patternSize, cornerQuad, isFixCornerH);
						const Mat H = findHomography(cornerQuad, cornerQuadConic);
						perspectiveTransform(points[pindex], pointsWarpConic[pindex], H);

						extractPatternCorner(pointsUndistort[pindex], patternSize, cornerQuad, isFixCornerH);
						const Mat Hundistort = findHomography(cornerQuad, cornerQuadConic);
						perspectiveTransform(pointsUndistort[pindex], pointsUndistortConic[pindex], Hundistort);

						warpPerspective(image[pindex], imageConic[pindex], H, image[pindex].size());
						warpPerspective(imageCircle[pindex], imageCircleConic[pindex], H, image[pindex].size());
						warpPerspective(imageUndist[pindex], imageUndistConic[pindex], Hundistort, image[pindex].size());
						Mat temp;  undistort(imageCircle[pindex], temp);
						warpPerspective(temp, imageCircleUndistConic[pindex], Hundistort, image[pindex].size());
					}
				}
			}
#pragma endregion

			const int pindex = (indexGUI >= (int)objectPoints.size()) ? 0 : indexGUI;
			const float scale = float(scalei);
			const float th2 = (thresh == thmax) ? FLT_MAX : (thresh * 0.0001f) * (thresh * 0.0001f);
			if (!image[pindex].empty())
			{
				if (dist == 0)
				{
					if (view == 0) image[pindex].copyTo(repPatternShow);
					else if (view == 1) imageConic[pindex].copyTo(repPatternShow);
					else imageCircleConic[pindex].copyTo(repPatternShow);
				}
				else
				{
					if (view == 0) imageUndist[pindex].copyTo(repPatternShow);
					else if (view == 1) imageUndistConic[pindex].copyTo(repPatternShow);
					else imageCircleUndistConic[pindex].copyTo(repPatternShow);
				}
			}
			else
			{
				repPatternShow.setTo(255);
			}
			bool isDrawConicalView = isGrid == 1;
			if (isDrawConicalView)
			{
				//h-line
				for (int j = 0; j < patternSize.height; j++)
				{
					const int st = patternSize.width * j + 0;
					const int ed = patternSize.width * j + patternSize.width - 1;
					line(repPatternShow, Point(pointsConic[st]), Point(pointsConic[ed]), COLOR_ORANGE);
					if (j != patternSize.height - 1)
					{
						line(repPatternShow, Point((pointsConic[st] + pointsConic[st + patternSize.width]) * 0.5f), Point((pointsConic[ed + patternSize.width] + pointsConic[ed]) * 0.5f), COLOR_ORANGE * 0.5);
						line(repPatternShow, Point((pointsConic[st] * 3.f + pointsConic[st + patternSize.width] * 1.f) * 0.25f), Point((pointsConic[ed + patternSize.width] * 1.f + pointsConic[ed] * 3.f) * 0.25f), COLOR_ORANGE * 0.5);
						line(repPatternShow, Point((pointsConic[st] * 1.f + pointsConic[st + patternSize.width] * 3.f) * 0.25f), Point((pointsConic[ed + patternSize.width] * 3.f + pointsConic[ed] * 1.f) * 0.25f), COLOR_ORANGE * 0.5);
					}
				}
				//v-line
				for (int i = 0; i < patternSize.width; i++)
				{
					int st = patternSize.width * 0 + i;
					int ed = patternSize.width * (patternSize.height - 1) + i;
					line(repPatternShow, Point(pointsConic[st]), Point(pointsConic[ed]), COLOR_ORANGE);
					if (i != patternSize.width - 1)
					{
						line(repPatternShow, Point((pointsConic[st] + pointsConic[st + 1]) * 0.5f), Point((pointsConic[ed + 1] + pointsConic[ed]) * 0.5f), COLOR_ORANGE * 0.5);
						line(repPatternShow, Point((pointsConic[st] * 3.f + pointsConic[st + 1] * 1.f) * 0.25f), Point((pointsConic[ed + 1] * 1.f + pointsConic[ed] * 3.f) * 0.25f), COLOR_ORANGE * 0.5);
						line(repPatternShow, Point((pointsConic[st] * 1.f + pointsConic[st + 1] * 3.f) * 0.25f), Point((pointsConic[ed + 1] * 3.f + pointsConic[ed] * 1.f) * 0.25f), COLOR_ORANGE * 0.5);
					}
				}
			}

			int gi = patternSize.width * giy + gix;
			gi = (gix == patternSize.width) ? (int)chessboard3D.size() : gi;
			gi = (giy == patternSize.height) ? (int)chessboard3D.size() : gi;

			double terror = 0.0;
			double terrorConic = 0.0;
			cv::applyColorMap(graybar, colorbar, ColorMap);
			repErrorShow.setTo(255);
			AutoBuffer<double> terrorReproPerPos(patternSize.area());
			AutoBuffer<double> terrorConicPerPos(patternSize.area());
			for (int n = 0; n < patternSize.area(); n++)
			{
				terrorReproPerPos[n] = 0.0;
				terrorConicPerPos[n] = 0.0;
			}

			if (indexGUI >= objectPoints.size())//all pattern
			{
				if (view == 0) repPatternShow.setTo(255);

				int count = 0;
				for (int pindex = 0; pindex < objectPoints.size(); pindex++)
				{
					Scalar color = Scalar(colorbar.ptr<uchar>(pindex * step)[0], colorbar.ptr<uchar>(pindex * step)[1], colorbar.ptr<uchar>(pindex * step)[2], 0.0);

					for (int n = 0; n < reprojectPoints[pindex].size(); n++)
					{
						const int x = n % patternSize.width;
						const int y = n / patternSize.width;
						bool flag = (x<gbb || x>patternSize.width - 1 - gbb || y<gbb || y>patternSize.height - 1 - gbb);
						if (flag) continue;

						const float dx = points[pindex][n].x - reprojectPoints[pindex][n].x;
						const float dy = points[pindex][n].y - reprojectPoints[pindex][n].y;

						const float dux = ((dist == 0) ? pointsWarpConic[pindex][n].x : pointsUndistortConic[pindex][n].x) - pointsConic[n].x;
						const float duy = ((dist == 0) ? pointsWarpConic[pindex][n].y : pointsUndistortConic[pindex][n].y) - pointsConic[n].y;

						terror = fma(dx, dx, fma(dy, dy, terror));
						terrorReproPerPos[n] = fma(dx, dx, fma(dy, dy, terrorReproPerPos[n]));//
						terrorConic = fma(dux, dux, fma(duy, duy, terrorConic));
						terrorConicPerPos[n] = fma(dux, dux, fma(duy, duy, terrorConicPerPos[n]));//

						count++;
						if (dx * dx + dy * dy < th2)
						{
							if (view == 0)
							{
								drawPlus(repPatternShow, Point(points[pindex][n]), 2, colorArrow, 2);
								arrowedLine(repPatternShow, Point(points[pindex][n]), Point(int(points[pindex][n].x + dx * escale), int(points[pindex][n].y + dy * escale)), colorArrow, 1);
							}
							else
							{
								drawPlus(repPatternShow, Point(pointsConic[n]), 2, colorArrow, 2);
								cv::circle(repPatternShow, Point(pointsConic[n]), grid_r, colorArrow);
								if (errorrf == 0) arrowedLine(repPatternShow, Point(pointsConic[n]), Point(int(pointsConic[n].x + dx * escale), int(pointsConic[n].y + dy * escale)), colorArrow, 1);
								else arrowedLine(repPatternShow, Point(pointsConic[n]), Point(int(pointsConic[n].x + dux * escale), int(pointsConic[n].y + duy * escale)), colorArrow, 1);
							}
							if (errorrf == 0) drawPlus(repErrorShow, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
							else drawPlus(repErrorShow, center + Point(cvRound(dux * scale), cvRound(duy * scale)), 2, color, 2);
						}
					}
				}
				//draw error circle
				if (view != 0 && errorCircle == 1)
				{
					for (int n = 0; n < patternSize.area(); n++)
					{
						const int x = n % patternSize.width;
						const int y = n / patternSize.width;
						bool flag = (x<gbb || x>patternSize.width - 1 - gbb || y<gbb || y>patternSize.height - 1 - gbb);
						if (flag) continue;

						const double rmse = sqrt(((errorrf == 0) ? terrorReproPerPos[n] : terrorConicPerPos[n]) / objectPoints.size());
						int r = cvRound(rmse * escale);
						cv::circle(repPatternShow, Point(pointsConic[n]), r, colorErrorCircle, 2);
					}
				}
				cv::putText(repErrorShow, format("Rep.Error: %6.4f", sqrt(terror / count)), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
				cv::putText(repErrorShow, format("Frt.Error: %6.4f", sqrt(terrorConic / count)), Point(0, 40), fontType, 0.5, COLOR_GRAY50);
				//putText(show, format("Rep.Error: %6.4f", rep_error), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
			}
			else //one pattern
			{
				if (gi == chessboard3D.size())//all points
				{
					int count = 0;
					Scalar color = Scalar(colorbar.ptr<uchar>(pindex * step)[0], colorbar.ptr<uchar>(pindex * step)[1], colorbar.ptr<uchar>(pindex * step)[2], 0.0);

					for (int n = 0; n < reprojectPoints[pindex].size(); n++)
					{
						const int x = n % patternSize.width;
						const int y = n / patternSize.width;
						bool flag = (x<gbb || x>patternSize.width - 1 - gbb || y<gbb || y>patternSize.height - 1 - gbb);
						if (flag) continue;

						if (sw == 1)
						{
							int step = 256 / (int)reprojectPoints[pindex].size();
							color = Scalar(colorbar.ptr<uchar>(n * step)[0], colorbar.ptr<uchar>(n * step)[1], colorbar.ptr<uchar>(n * step)[2], 0.0);
						}

						const float dx = points[pindex][n].x - reprojectPoints[pindex][n].x;
						const float dy = points[pindex][n].y - reprojectPoints[pindex][n].y;

						const float dux = ((dist == 0) ? pointsWarpConic[pindex][n].x : pointsUndistortConic[pindex][n].x) - pointsConic[n].x;
						const float duy = ((dist == 0) ? pointsWarpConic[pindex][n].y : pointsUndistortConic[pindex][n].y) - pointsConic[n].y;

						terror = fma(dx, dx, fma(dy, dy, terror));
						terrorReproPerPos[n] = fma(dx, dx, fma(dy, dy, terrorReproPerPos[n]));//
						terrorConic = fma(dux, dux, fma(duy, duy, terrorConic));
						terrorConicPerPos[n] = fma(dux, dux, fma(duy, duy, terrorConicPerPos[n]));//
						count++;
						if (dx * dx + dy * dy < th2)
						{
							if (view == 0)
							{
								drawPlus(repPatternShow, Point(points[pindex][n]), 2, colorArrow, 2);
								arrowedLine(repPatternShow, Point(points[pindex][n]), Point(int(points[pindex][n].x + dx * escale), int(points[pindex][n].y + dy * escale)), colorArrow, 1);
							}
							else
							{
								drawPlus(repPatternShow, Point(pointsConic[n]), 2, colorArrow, 2);
								cv::circle(repPatternShow, Point(pointsConic[n]), grid_r, colorArrow);
								//cv::circle(gridposition, Point(pointsWarp[n]), front_r, color2);
								//drawPlus(gridposition, Point(pointsWarp[n]), 2, COLOR_WHITE, 2);
								if (errorrf == 0) arrowedLine(repPatternShow, Point(pointsConic[n]), Point(int(pointsConic[n].x + dx * escale), int(pointsConic[n].y + dy * escale)), colorArrow, 1);
								else arrowedLine(repPatternShow, Point(pointsConic[n]), Point(int(pointsConic[n].x + dux * escale), int(pointsConic[n].y + duy * escale)), colorArrow, 1);
							}
							if (errorrf == 0)drawPlus(repErrorShow, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
							else drawPlus(repErrorShow, center + Point(cvRound(dux * scale), cvRound(duy * scale)), 2, color, 2);
						}
					}
					//draw error circle
					if (view != 0 && errorCircle == 1)
					{
						for (int n = 0; n < patternSize.area(); n++)
						{
							const int x = n % patternSize.width;
							const int y = n / patternSize.width;
							bool flag = (x<gbb || x>patternSize.width - 1 - gbb || y<gbb || y>patternSize.height - 1 - gbb);
							if (flag) continue;

							const double rmse = sqrt(((errorrf == 0) ? terrorReproPerPos[n] : terrorConicPerPos[n]));
							int r = cvRound(rmse * escale);
							cv::circle(repPatternShow, Point(pointsConic[n]), r, colorErrorCircle, 2);
						}
					}
					cv::putText(repErrorShow, format("Rep.Error: %6.4f", sqrt(terror / count)), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
					cv::putText(repErrorShow, format("Frt.Error: %6.4f", sqrt(terrorConic / count)), Point(0, 40), fontType, 0.5, COLOR_GRAY50);
				}
				else //one point
				{
					if (sw == 1)
					{
						int step = 256 / (int)reprojectPoints.size();
						color = Scalar(colorbar.ptr<uchar>(gi * step)[0], colorbar.ptr<uchar>(gi * step)[1], colorbar.ptr<uchar>(gi * step)[2], 0.0);
					}

					const float dx = points[pindex][gi].x - reprojectPoints[pindex][gi].x;
					const float dy = points[pindex][gi].y - reprojectPoints[pindex][gi].y;
					terror = fma(dx, dx, fma(dy, dy, terror));

					if (dx * dx + dy * dy < th2)
					{
						if (view == 0)
						{
							drawPlus(repPatternShow, Point(points[pindex][gi]), 2, colorArrow, 2);
							arrowedLine(repPatternShow, Point(points[pindex][gi]), Point(int(points[pindex][gi].x + dx * escale), int(points[pindex][gi].y + dy * escale)), colorArrow, 1);
						}
						else
						{
							drawPlus(repPatternShow, Point(pointsConic[gi]), 2, colorArrow, 2);
							arrowedLine(repPatternShow, Point(pointsConic[gi]), Point(int(pointsConic[gi].x + dx * escale), int(pointsConic[gi].y + dy * escale)), colorArrow, 1);
						}

						drawPlus(repErrorShow, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
					}
					cv::putText(repErrorShow, format("Rep.Error: %6.4f", sqrt(terror)), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
				}
			}

			cv::putText(repErrorShow, format("%6.4f", length * 0.5 / scale), Point(length / 2 - 30, length / 2), fontType, 0.5, COLOR_GRAY50);
			cv::circle(repErrorShow, center, length / 2, COLOR_GRAY100);
			drawGridMulti(repErrorShow, Size(4, 4), COLOR_GRAY200);
			Mat cres;
			cv::resize(colorbar.t(), cres, Size(repErrorShow.cols, 20));
			cv::vconcat(repErrorShow, cres, cres);
			cv::imshow(wname, cres);

			cv::imshow(wnameplot, repPatternShow);
			key = waitKey(1);

			if (!isWait) break;
		}
		if (isWait) destroyWindow(wname);
		if (isWait) destroyWindow(wnameplot);
	}

	void Calibrator::drawReprojectionErrorFromExtraPoints(const vector<Point2f>& points, const bool isWait, const string wname, const float scale, const Mat& patternImage, const int patternType, const bool isUseInternalData)
	{
		vector<vector<Point2f>> p;
		vector<vector<Point3f>> o;
		vector<Mat> im;
		vector<Mat> R;
		vector<Mat> T;
		Mat r, t;
		p.push_back(points);
		o.push_back(chessboard3D);
		im.push_back(patternImage);
		cv::solvePnP(chessboard3D, points, intrinsic, distortion, r, t);
		R.push_back(r.clone());
		T.push_back(t.clone());
		if (isUseInternalData)
		{
			//cout << "use internal"<<endl;
			for (int i = 0; i < imagePoints.size(); i++)
			{
				p.push_back(imagePoints[i]);
				o.push_back(chessboard3D);
				im.push_back(patternImages[i]);
				cv::solvePnP(chessboard3D, imagePoints[i], intrinsic, distortion, r, t);
				R.push_back(r.clone());
				T.push_back(t.clone());
			}
		}
		drawReprojectionErrorInternal(p, o, R, T, isWait, wname, scale, im, patternType);
	}

	void Calibrator::drawReprojectionError(string wname, const bool isInteractive, const float scale)
	{
		int length = 400;
		Size imsize(2 * length + 1, 2 * length + 1);
		Point center(imsize.width / 2, imsize.height / 2);
		Mat show(imsize, CV_8UC3);
		show.setTo(255);

		double terror = 0.0;
		Mat_<uchar> graybar = Mat(256, 1, CV_8U);
		Mat colorbar;
		for (int i = 0; i < 256; i++) graybar(i) = i;
		cv::applyColorMap(graybar, colorbar, COLORMAP_JET);
		int step = 256 / (int)objectPoints.size();

		if (isInteractive)
		{
			drawReprojectionErrorInternal(imagePoints, objectPoints, rt, tv, true, "error", scale, patternImages);
		}
		else
		{
			for (int i = 0; i < objectPoints.size(); i++)
			{
				Scalar color = Scalar(colorbar.ptr<uchar>(i * step)[0], colorbar.ptr<uchar>(i * step)[1], colorbar.ptr<uchar>(i * step)[2], 0.0);
				vector<Point2f> reprojectPoints;
				projectPoints(objectPoints[i], rt[i], tv[i], intrinsic, distortion, reprojectPoints);
				for (int n = 0; n < reprojectPoints.size(); n++)
				{
					float dx = imagePoints[i][n].x - reprojectPoints[n].x;
					float dy = imagePoints[i][n].y - reprojectPoints[n].y;
					terror = fma(dx, dx, fma(dy, dy, terror));
					drawPlus(show, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
				}
				//imshow("error", show);waitKey();
			}
			putText(show, format("%6.4f", length * 0.5 / scale), Point(length / 2 - 30, length / 2), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, COLOR_GRAY50);
			circle(show, center, length / 2, COLOR_GRAY100);
			drawGridMulti(show, Size(4, 4), COLOR_GRAY200);
			imshow(wname, show);
		}
	}

	void Calibrator::drawDistortion(string wname, const bool isInteractive)
	{
		namedWindow(wname);

		if (isInteractive)
		{
			int key = 0;
			Mat show;
			int step = 20; createTrackbar("step", wname, &step, 100);
			int thickness = 1; createTrackbar("thickness", wname, &thickness, 3);
			int amp = 100; createTrackbar("amp*0.01", wname, &amp, 2000);
			while (key != 'q')
			{
				drawDistortionLine(mapu, mapv, show, step, thickness, amp * 0.01f);
				imshow(wname, show);
				key = waitKey(1);
			}
		}
		destroyWindow(wname);
	}
}