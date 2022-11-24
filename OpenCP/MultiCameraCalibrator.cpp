#include "MultiCameraCalibrator.hpp"
#include "rectifyMultiCollinear.hpp"
#include "consoleImage.hpp"
#include "debugcp.hpp"
#include "draw.hpp"
#include "blend.hpp"
#include "shiftImage.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void undistortOnePoint(const Point2f src, Point2f& dest, const Mat& K, const Mat& dist, const Mat& R, const Mat& P)
	{
		if (K.depth() == CV_64F)
		{
			Mat sm = Mat(2, 1, CV_64F);
			Mat dm = Mat(2, 1, CV_64F);
			sm.at<double>(0) = src.x;
			sm.at<double>(1) = src.y;
			undistortPoints(sm, dm, K, dist, R, P);
			dest.x = dm.at<double>(0);
			dest.y = dm.at<double>(1);
		}
		if (K.depth() == CV_32F)
		{
			Mat sm = Mat(2, 1, CV_32F);
			Mat dm = Mat(2, 1, CV_32F);
			sm.at<float>(0) = src.x;
			sm.at<float>(1) = src.y;
			undistortPoints(sm, dm, K, dist, R, P);
			dest.x = dm.at<float>(0);
			dest.y = dm.at<float>(1);
		}
	}

	void undistortOnePoint(const Point2d src, Point2d& dest, const Mat& K, const Mat& dist, const Mat& R, const Mat& P)
	{
		if (K.depth() == CV_64F)
		{
			Mat sm = Mat(2, 1, CV_64F);
			Mat dm = Mat(2, 1, CV_64F);
			sm.at<double>(0) = src.x;
			sm.at<double>(1) = src.y;
			undistortPoints(sm, dm, K, dist, R, P);
			dest.x = dm.at<double>(0);
			dest.y = dm.at<double>(1);
		}
		if (K.depth() == CV_32F)
		{
			Mat sm = Mat(2, 1, CV_32F);
			Mat dm = Mat(2, 1, CV_32F);
			sm.at<float>(0) = src.x;
			sm.at<float>(1) = src.y;
			undistortPoints(sm, dm, K, dist, R, P);
			dest.x = dm.at<float>(0);
			dest.y = dm.at<float>(1);
		}
	}


	void MultiCameraCalibrator::generatechessboard3D()
	{
		for (int j = 0; j < patternSize.height; ++j)
		{
			for (int i = 0; i < patternSize.width; ++i)
			{
				chessboard3D.push_back(Point3f(lengthofchess * (float)i, lengthofchess * (float)j, 0.0));
			}
		}
	}

	void MultiCameraCalibrator::initRemap()
	{
		for (int i = 0; i < numofcamera; i++)
		{
			Mat u;
			Mat v;
			initUndistortRectifyMap(intrinsic[i], distortion[i], R[i], P[i], imageSize, CV_32FC1,
				u, v);
			mapu.push_back(u);
			mapv.push_back(v);
		}
	}

	/*
	MultiCameraCalibrator MultiCameraCalibrator::cloneParameters()
	{
	MultiCameraCalibrator ret(imageSize, patternSize, lengthofchess);
	ret.intrinsicRect=intrinsicRect.clone();
	for(int i=0;i<3;i++)
	{
	ret.intrinsic.push_back(intrinsic[i]);
	ret.distortion.push_back(distortion[i]);
	ret.R.push_back(R[i]);
	ret.P.push_back(P[i]);

	}
	ret.Q=Q.clone();

	ret.R12=R12.clone();
	ret.T12=T12.clone();
	ret.E=E.clone();
	ret.F=F.clone();

	return ret;
	}*/

	void MultiCameraCalibrator::readParameter(char* name)
	{
		char nn[64];

		FileStorage fs(name, FileStorage::READ);

		fs["numofcamera"] >> numofcamera;
		for (int i = 0; i < numofcamera; i++)
		{
			Mat temp;
			sprintf(nn, "intrinsic%02d", i);

			fs[nn] >> temp;
			temp.copyTo(intrinsic[i]);

			sprintf(nn, "distortion%02d", i);
			fs[nn] >> temp;
			temp.copyTo(distortion[i]);

			sprintf(nn, "rectR%02d", i);
			fs[nn] >> temp;
			temp.copyTo(R[i]);

			sprintf(nn, "rectP%02d", i);
			fs[nn] >> temp;
			temp.copyTo(P[i]);

			sprintf(nn, "rerR%02d", i);
			fs[nn] >> temp;
			temp.copyTo(reR[i]);

			sprintf(nn, "rerT%02d", i);
			fs[nn] >> temp;
			temp.copyTo(reT[i]);
		}
		fs["E"] >> E;
		fs["F"] >> F;
		fs["Q"] >> Q;
		initRemap();
	}

	void MultiCameraCalibrator::writeParameter(char* name)
	{
		FileStorage fs(name, FileStorage::WRITE);
		char nn[64];

		fs << "numofcamera" << numofcamera;

		for (int i = 0; i < numofcamera; i++)
		{
			sprintf(nn, "intrinsic%02d", i);
			fs << nn << intrinsic[i];
			sprintf(nn, "distortion%02d", i);
			fs << nn << distortion[i];

			sprintf(nn, "rectR%02d", i);
			fs << nn << R[i];
			sprintf(nn, "rectP%02d", i);
			fs << nn << P[i];

			sprintf(nn, "rerR%02d", i);
			fs << nn << reR[i];
			sprintf(nn, "rerT%02d", i);
			fs << nn << reT[i];
		}
		fs << "E" << E;
		fs << "F" << F;
		fs << "Q" << Q;
	}

	void MultiCameraCalibrator::init(Size imageSize_, Size patternSize_, float lengthofchess_, int numofcamera_)
	{
		numofcamera = numofcamera_;

		numofchessboards = 0;
		imageSize = imageSize_;
		patternSize = patternSize_;
		lengthofchess = lengthofchess_;

		vector<vector<Point2f>> tmp;
		for (int i = 0; i < numofcamera; i++)
		{
			imagePoints.push_back(tmp);

			Mat temp = Mat::eye(3, 3, CV_64F);
			getDefaultNewCameraMatrix(temp, imageSize, true);
			intrinsic.push_back(temp);
			distortion.push_back(Mat::zeros(1, 8, CV_64F));
			R.push_back(Mat::eye(3, 3, CV_64F));
			P.push_back(Mat::eye(3, 4, CV_64F));
		}
		for (int i = 0; i < numofcamera; i++)
		{
			reT.push_back(Mat::zeros(3, 1, CV_64F));
			reR.push_back(Mat::eye(3, 3, CV_64F));
		}

		patternImages.resize(numofcamera);

		flag = CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_ZERO_TANGENT_DIST | CALIB_SAME_FOCAL_LENGTH | CALIB_FIX_ASPECT_RATIO;
		generatechessboard3D();
	}

	MultiCameraCalibrator::MultiCameraCalibrator(Size imageSize_, Size patternSize_, float lengthofchess_, int numofcamera_)
	{
		init(imageSize_, patternSize_, lengthofchess_, numofcamera_);
	}

	MultiCameraCalibrator::MultiCameraCalibrator() { ; }

	MultiCameraCalibrator::~MultiCameraCalibrator() { ; }


	bool MultiCameraCalibrator::findChess(vector<Mat>& im, vector <Mat>& dest)
	{
		bool ret = true;
		vector<vector<Point2f>> tmp(numofcamera);
		vector<bool> rt(numofcamera);
		for (int i = 0; i < numofcamera; i++)
		{
			rt[i] = findChessboardCorners(im[i], patternSize, tmp[i]);
			im[i].copyTo(dest[i]);
			drawChessboardCorners(dest[i], patternSize, Mat(tmp[i]), rt[i]);
			if (rt[i] == false)ret = false;
		}


		if (ret)
		{
			numofchessboards++;

			for (int i = 0; i < numofcamera; i++)
			{
				imagePoints[i].push_back(tmp[i]);
			}

			objectPoints.push_back(chessboard3D);
		}
		return ret;
	}

	bool MultiCameraCalibrator::findChess(vector<Mat>& im)
	{
		std::vector <cv::Mat> dest = std::vector<cv::Mat>(0);
		return findChess(im, dest);
	}

	void MultiCameraCalibrator::pushImage(const vector<cv::Mat>& patternImage)
	{
		for (int i = 0; i < numofcamera; i++)
		{
			patternImages[i].push_back(patternImage[i]);
		}
	}

	void MultiCameraCalibrator::pushImagePoint(const vector<vector<Point2f>>& point)
	{
		numofchessboards++;
		for (int i = 0; i < imagePoints.size(); i++)
		{
			imagePoints[i].push_back(point[i]);
		}
	}

	void MultiCameraCalibrator::pushObjectPoint(const vector<Point3f>& point)
	{
		objectPoints.push_back(point);
	}


	void MultiCameraCalibrator::printParameters()
	{
		for (int i = 0; i < numofcamera; i++)
		{
			cout << "===========================" << endl;
			std::cout << i << " camera" << std::endl;
			std::cout << "intrinsic" << std::endl;
			cout << intrinsic[i] << endl;
			cout << distortion[i] << endl;
			std::cout << "Rotation" << std::endl;
			cout << R[i] << endl;
			std::cout << "Projection" << std::endl;
			cout << P[i] << endl;

		}
		cout << "Q" << endl;
		cout << Q << endl;
		cout << "===========================" << endl;
	}

	double MultiCameraCalibrator::getRectificationErrorBetween(int a, int b)
	{
		vector<double> error;

		for (int n = 0; n < numofchessboards; n++)
		{
			vector<Point2f> dst1;
			vector<Point2f> dst2;

			undistortPoints(Mat(imagePoints[a][n]), dst1, intrinsic[a], distortion[a], R[a], P[a]);
			undistortPoints(Mat(imagePoints[b][n]), dst2, intrinsic[b], distortion[b], R[b], P[b]);

			double e = 0.0;
			for (int i = 0; i < (int)dst1.size(); i++)
			{
				e += abs(dst1[i].y - dst2[i].y);
			}
			e /= dst1.size();
			error.push_back(e);
		}

		double te = 0.0;

		for (int n = 0; n < numofchessboards; n++)
		{
			te += error[n];
		}
		te /= (double)numofchessboards;
		return (te) / 1.0;
	}

	double MultiCameraCalibrator::getRectificationErrorDisparity()
	{
		vector<double> error;

		for (int n = 0; n < numofchessboards; n++)
		{
			vector<vector<Point2f>> dst(numofcamera);

			for (int i = 0; i < numofcamera; i++)
			{
				undistortPoints(Mat(imagePoints[i][n]), dst[i], intrinsic[i], distortion[i], R[i], P[i]);
			}

			double e = 0.0;
			for (int i = 0; i < (int)dst[0].size(); i++)
			{
				double ave = 0.0;
				for (int nn = 0; nn < numofcamera - 1; nn++)
				{
					ave += abs(dst[nn][i].x - dst[nn + 1][i].x);
				}
				ave /= (double)(numofcamera - 1);
				for (int nn = 0; nn < numofcamera - 1; nn++)
				{
					e += abs(abs(dst[nn][i].x - dst[nn + 1][i].x) - ave);
				}
			}
			e = e / (double)(dst[0].size() * numofcamera);
			error.push_back(e);
		}

		double te = 0.0;
		for (int n = 0; n < numofchessboards; n++)
		{
			te += error[n];
		}
		te /= (double)numofchessboards;
		return te;
	}

	double MultiCameraCalibrator::getRectificationErrorDisparityBetween(int ref1, int ref2)
	{
		vector<double> error;

		for (int n = 0; n < numofchessboards; n++)
		{
			vector<vector<Point2f>> dst(numofcamera);
			const double step = 1.0 / (double)(abs(ref1 - ref2));

			for (int i = 0; i < numofcamera; i++)
			{
				undistortPoints(Mat(imagePoints[i][n]), dst[i], intrinsic[i], distortion[i], R[i], P[i]);
			}

			double e = 0.0;
			for (int i = 0; i < (int)dst[0].size(); i++)
			{
				double ave = 0.0;
				for (int nn = 0; nn < numofcamera - 1; nn++)
				{
					ave = max(ave, (double)abs(dst[nn][i].x - dst[nn + 1][i].x));
				}
				e += abs(dst[ref1][i].x - dst[ref2][i].x) * step / ave;
			}
			e = e / (double)(dst[0].size());
			error.push_back(e);
		}

		double te = 0.0;
		for (int n = 0; n < numofchessboards; n++)
		{
			te += error[n];
		}
		te /= (double)numofchessboards;
		return te;
	}

	double MultiCameraCalibrator::getRectificationError()
	{
		vector<double> error;

		for (int n = 0; n < numofchessboards; n++)
		{
			vector<vector<Point2f>> dst(numofcamera);
			for (int i = 0; i < numofcamera; i++)
			{
				undistortPoints(Mat(imagePoints[i][n]), dst[i], intrinsic[i], distortion[i], R[i], P[i]);
			}

			double e = 0.0;
			for (int i = 0; i < (int)dst[0].size(); i++)
			{
				double ave = 0.0;
				for (int nn = 0; nn < numofcamera; nn++)
				{
					ave += dst[nn][i].y;
				}
				ave /= (double)numofcamera;
				for (int nn = 0; nn < numofcamera; nn++)
				{
					e += abs(dst[nn][i].y - ave);
				}
			}
			e = e / (double)(dst[0].size() * numofcamera);
			error.push_back(e);
		}

		double te = 0.0;
		for (int n = 0; n < numofchessboards; n++)
		{
			te += error[n];
		}
		te /= (double)numofchessboards;
		return te;
	}


	void MultiCameraCalibrator::operator ()(bool isFixIntrinsic, int refCamera1, int refCamera2)
	{
		if (refCamera1 == 0 && refCamera2 == 0)
		{
			refCamera2 = numofcamera - 1;
		}

		if (numofchessboards < 1)
		{
			std::cout << "input 2 or more chessboards" << std::endl;
			return;
		}

		if (isFixIntrinsic)
		{
			Mat r, t;
			cv::calibrateCamera(objectPoints, imagePoints[refCamera1], imageSize, intrinsic[refCamera1], distortion[refCamera1], r, t, flag);
			cv::calibrateCamera(objectPoints, imagePoints[refCamera2], imageSize, intrinsic[refCamera2], distortion[refCamera2], r, t, flag);

			for (int i = 1; i < numofcamera; i++)
			{
				stereoCalibrate(objectPoints, imagePoints[refCamera1], imagePoints[i], intrinsic[refCamera1], distortion[refCamera1], intrinsic[i], distortion[i], imageSize,
					reR[i], reT[i], E, F,
					flag | CALIB_FIX_INTRINSIC,
					TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6)
				);
			}
		}
		else
		{
			for (int i = 1; i < numofcamera; i++)
			{
				stereoCalibrate(objectPoints, imagePoints[refCamera1], imagePoints[i], intrinsic[refCamera1], distortion[refCamera1], intrinsic[i], distortion[i], imageSize,
					reR[i], reT[i], E, F,
					flag,
					TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			}
		}

		rectifyMultiCollinear(intrinsic, distortion, refCamera1, refCamera2, imagePoints, imageSize, reR, reT, R, P, Q, -1.0, imageSize, 0, 0, rect_flag);
		initRemap();
	}

	void MultiCameraCalibrator::calibration(const int flags, int refCamera1, int refCamera2)
	{
		this->flag = flags;
		operator()(false, refCamera1, refCamera2);
		//operator()(true, refCamera1, refCamera2);
	}

	void MultiCameraCalibrator::rectifyImageRemap(Mat& src, Mat& dest, int numofcamera, const int interpolation)
	{
		if (mapu[numofcamera].empty())
		{
			cout << "calibration is not ready" << endl;
			return;
		}
		if (mapv[numofcamera].empty())
		{
			cout << "calibration is not ready" << endl;
			return;
		}
		cv::remap(src, dest, mapu[numofcamera], mapv[numofcamera], interpolation);
	}

	void MultiCameraCalibrator::guiDisparityTest()
	{
		string wname = "disparity";
		namedWindow(wname);
		int pi = 0; createTrackbar("pattern index", wname, &pi, (int)objectPoints.size() - 1);
		int ci = 0; createTrackbar("corner index", wname, &ci, (int)objectPoints[pi].size() - 1);
		int showsw = 1; createTrackbar("show: 0_raw, 1_rect", wname, &showsw, 1);
		const int cami = 0;
		Mat localR, localT;
		int key = 0;
		cp::ConsoleImage con(Size(640, 480), wname);
		const double l = 1.0 / Q.at<double>(3, 2);//norm(reT[cami + 1]);
		const double f = Q.at<double>(2, 3);
		const double d_offset = l * Q.at<double>(3, 3);

		Mat patternL;
		Mat patternR;
		vector<Mat> rectimL(objectPoints.size());
		vector<Mat> rectimR(objectPoints.size());
		for (int i = 0; i < objectPoints.size(); i++)
		{
			rectifyImageRemap(patternImages[cami + 0][i], rectimL[i], 0);
			rectifyImageRemap(patternImages[cami + 1][i], rectimR[i], 1);
		}

		while (key != 'q')
		{
			if (showsw == 1)
			{
				if (rectimL[pi].channels() == 1)cvtColor(rectimL[pi], patternL, COLOR_GRAY2BGR);
				else rectimL[pi].copyTo(patternL);
				if (rectimR[pi].channels() == 1)cvtColor(rectimR[pi], patternR, COLOR_GRAY2BGR);
				else rectimR[pi].copyTo(patternR);
			}
			else
			{
				if (patternImages[cami + 0][pi].channels() == 1)cvtColor(patternImages[cami + 0][pi], patternL, COLOR_GRAY2BGR);
				else patternImages[cami + 0][pi].copyTo(patternL);
				if (patternImages[cami + 1][pi].channels() == 1)cvtColor(patternImages[cami + 1][pi], patternR, COLOR_GRAY2BGR);
				else patternImages[cami + 1][pi].copyTo(patternR);
			}

			Mat a = Mat(objectPoints[pi][ci]);
			a.convertTo(a, CV_64F);
			cv::solvePnP(objectPoints[pi], imagePoints[cami + 0][pi], intrinsic[cami + 0], distortion[cami + 0], localR, localT);
			cv::Rodrigues(localR, localR);
			Mat dL = localR * a + localT;
			Mat TL = localR.t() * localT.clone();
			cv::solvePnP(objectPoints[pi], imagePoints[cami + 1][pi], intrinsic[cami + 1], distortion[cami + 1], localR, localT);
			cv::Rodrigues(localR, localR);
			Mat dR = localR * a + localT;

			Point2f pt0 = imagePoints[cami + 0][pi][ci];
			Point2f pt1 = imagePoints[cami + 1][pi][ci];
			Point2f rpt0, rpt1;
			undistortOnePoint(pt0, rpt0, intrinsic[cami + 0], distortion[cami + 0], R[cami + 0], P[cami + 0]);
			undistortOnePoint(pt1, rpt1, intrinsic[cami + 1], distortion[cami + 1], R[cami + 1], P[cami + 1]);

			if (showsw == 1)
			{
				circle(patternL, Point(rpt0), 10, COLOR_ORANGE, 2);
				cp::drawPlus(patternL, Point(rpt0), 3, COLOR_ORANGE);
				circle(patternR, Point(rpt1), 10, COLOR_ORANGE, 2);
				cp::drawPlus(patternR, Point(rpt1), 3, COLOR_ORANGE);
			}
			else
			{
				circle(patternL, Point(pt0), 10, COLOR_ORANGE, 2);
				cp::drawPlus(patternL, Point(pt0), 3, COLOR_ORANGE);
				circle(patternR, Point(pt1), 10, COLOR_ORANGE, 2);
				cp::drawPlus(patternR, Point(pt1), 3, COLOR_ORANGE);
			}
			con("xL       %f", dL.at<double>(0));
			con("yL       %f", dL.at<double>(1));
			con("zL       %f", dL.at<double>(2));
			con("xR       %f", dR.at<double>(0));
			con("yR       %f", dR.at<double>(1));
			con("zR       %f", dR.at<double>(2));
			con("pt0(x,y) %f %f", pt0.x, pt0.y);
			con("pt1(x,y) %f %f %f", pt1.x, pt1.y, pt1.y - pt0.y);
			con("rpt0(x,y)%f %f", rpt0.x, rpt0.y);
			con("rpt1(x,y)%f %f %f", rpt1.x, rpt1.y, rpt1.y - rpt0.y);

			con("Fx       %f", f);
			con("L        %f %f", l, norm(localR.t() * localT - TL));
			const float disparity = rpt0.x - rpt1.x;
			con("disp     %f", disparity);
			const float flz = f * l / (disparity);
			con("z=fl/d   %f %f %f", flz, flz - dL.at<double>(2), flz - dR.at<double>(2));
			const float flzo = f * l / (disparity + d_offset);
			con("z=fl/d(o)%f %f %f", flzo, flzo - dL.at<double>(2), flzo - dR.at<double>(2));
			con.show();
			imshow(wname + "patternL", patternL);
			imshow(wname + "patternR", patternR);
			key = waitKey(1);
			if (key == 's')
			{
				cp::guiShift(patternImages[cami + 0][pi], patternImages[cami + 1][pi], 500, "shift RAW");
				cp::guiShift(rectimL[pi], rectimR[pi], 500, "shift rectify");

			}
			if (key == 'd')
			{
				cp::guiAlphaBlend(rectimL[pi], patternImages[cami + 0][pi]);
			}
		}
	}
}