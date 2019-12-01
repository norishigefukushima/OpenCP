#include "MultiCameraCalibrator.hpp"
#include "rectifyMultiCollinear.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void MultiCameraCalibrator::generatechessboard3D()
	{
		for (int j = 0; j < patternSize.height; ++j)
			for (int i = 0; i < patternSize.width; ++i)
				chessboard3D.push_back(Point3f(lengthofchess*(float)i, lengthofchess*(float)j, 0.0));

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

		//flag = CALIB_FIX_INTRINSIC;
		flag = CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_ZERO_TANGENT_DIST | CALIB_SAME_FOCAL_LENGTH | CALIB_FIX_ASPECT_RATIO;
		//flag = CALIB_USE_INTRINSIC_GUESS;
		generatechessboard3D();
	}

	MultiCameraCalibrator::MultiCameraCalibrator(Size imageSize_, Size patternSize_, float lengthofchess_, int numofcamera_)
	{
		init(imageSize_, patternSize_, lengthofchess_, numofcamera_);
	}
	MultiCameraCalibrator::MultiCameraCalibrator(){ ; }
	MultiCameraCalibrator::~MultiCameraCalibrator(){ ; }


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
			;
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
	
	void MultiCameraCalibrator::printParameters()
	{
		for (int i = 0; i < numofcamera; i++)
		{
			std::cout << i << " camera" << std::endl;
			std::cout << "intrinsic" << std::endl;
			cout << intrinsic[i] << endl;
			cout << distortion[i] << endl;
			std::cout << "Rotation" << std::endl;
			cout << R[i] << endl;
			std::cout << "Projection" << std::endl;
			cout << P[i] << endl;
		}
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
			e = e / (double)(dst[0].size()*numofcamera);
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
				e += abs(dst[ref1][i].x - dst[ref2][i].x)*step / ave;
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
			e = e / (double)(dst[0].size()*numofcamera);
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

		if (!isFixIntrinsic)
		{
			if (numofchessboards < 2)
			{
				std::cout << "input 3 or more chessboards" << std::endl;
				return;
			}
			for (int i = 1; i < numofcamera; i++)
			{
				stereoCalibrate(objectPoints, imagePoints[refCamera1], imagePoints[i], intrinsic[refCamera1], distortion[refCamera1], intrinsic[i], distortion[i], imageSize, reR[i], reT[i], E, F,
					flag,
					TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			}
		}
		else
		{
			for (int i = 1; i < numofcamera; i++)
			{
				flag = CALIB_USE_INTRINSIC_GUESS;
				//flag = CALIB_FIX_INTRINSIC;
				stereoCalibrate(objectPoints, imagePoints[refCamera1], imagePoints[i], intrinsic[refCamera1], distortion[refCamera1], intrinsic[i], distortion[i], imageSize, reR[i], reT[i], E, F,
					flag,
					TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6)
					);
			}
		}

		rectifyMultiCollinear(intrinsic, distortion, refCamera1, refCamera2, imagePoints, imageSize, reR, reT, R, P, Q, -1.0, imageSize, 0, 0, CALIB_ZERO_DISPARITY);


		initRemap();
	}

	void MultiCameraCalibrator::rectifyImageRemap(Mat& src, Mat& dest, int numofcamera)
	{
		remap(src, dest, mapu[numofcamera], mapv[numofcamera], INTER_LINEAR);

	}
}