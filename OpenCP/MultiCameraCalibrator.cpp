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
			dest.x = float(dm.at<double>(0));
			dest.y = float(dm.at<double>(1));
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
			sm.at<float>(0) = float(src.x);
			sm.at<float>(1) = float(src.y);
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
				chessboard3D.push_back(Point3f(lengthofchess.width * (float)i, lengthofchess.height * (float)j, 0.0));
			}
		}
	}

	void MultiCameraCalibrator::initRemap()
	{
		mapu.resize(0);
		mapv.resize(0);
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

	void MultiCameraCalibrator::init(Size imageSize_, Size patternSize_, Size2f lengthofchess_, int numofcamera_)
	{
		numofcamera = numofcamera_;

		numofchessboards = 0;
		imageSize = imageSize_;
		patternSize = patternSize_;
		this->lengthofchess = lengthofchess_;

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
		init(imageSize_, patternSize_, Size2f(lengthofchess_, lengthofchess_), numofcamera_);
	}

	MultiCameraCalibrator::MultiCameraCalibrator(Size imageSize_, Size patternSize_, Size2f lengthofchess_, int numofcamera_)
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

	void MultiCameraCalibrator::pushImagePoint(const vector<Point2f>& pointL, const vector<Point2f>& pointR)
	{
		numofchessboards++;
		imagePoints[0].push_back(pointL);
		imagePoints[1].push_back(pointR);
	}

	void MultiCameraCalibrator::pushImagePoint(const vector<vector<Point2f>>& point)
	{
		numofchessboards++;
		for (int i = 0; i < numofcamera; i++)
		{
			imagePoints[i].push_back(point[i]);
		}
	}

	void MultiCameraCalibrator::pushObjectPoint(const vector<Point3f>& point)
	{
		objectPoints.push_back(point);
	}

	void MultiCameraCalibrator::clearPatternData()
	{		
		for (int i = 0; i < numofcamera; i++)
		{
			imagePoints[i].resize(0);
			patternImages[i].resize(0);
		}
		objectPoints.resize(0);
		numofchessboards = 0;
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

	void MultiCameraCalibrator::setIntrinsic(const cv::Mat& intrinsic, const cv::Mat& distortion, const int cameraIndex)
	{
		intrinsic.copyTo(this->intrinsic[cameraIndex]);
		distortion.copyTo(this->distortion[cameraIndex]);
	}

	void MultiCameraCalibrator::setRP(const cv::Mat& R, const cv::Mat& P, const int cameraIndex, bool isInitRemap)
	{
		R.copyTo(this->R[cameraIndex]);
		P.copyTo(this->P[cameraIndex]);
		if (isInitRemap)initRemap();
	}

	void MultiCameraCalibrator::setQ(const cv::Mat& Q)
	{
		Q.copyTo(this->Q);
	}


	double MultiCameraCalibrator::operator ()(const int flags, int refCamera1, int refCamera2, const bool isIndependentCalibration)
	{
		if (refCamera1 == 0 && refCamera2 == 0)
		{
			refCamera2 = numofcamera - 1;
		}

		if (numofchessboards < 1)
		{
			std::cout << "input 2 or more chessboards" << std::endl;
			return 0.0;
		}

		if (isIndependentCalibration)
		{
			cout << "call isIndependentCalibration" << endl;
			Mat r, t;
			cv::calibrateCamera(objectPoints, imagePoints[refCamera1], imageSize, intrinsic[refCamera1], distortion[refCamera1], r, t, flag);
			cv::calibrateCamera(objectPoints, imagePoints[refCamera2], imageSize, intrinsic[refCamera2], distortion[refCamera2], r, t, flag);

			for (int i = 1; i < numofcamera; i++)
			{
				rep_error = stereoCalibrate(objectPoints, imagePoints[refCamera1], imagePoints[i], intrinsic[refCamera1], distortion[refCamera1], intrinsic[i], distortion[i], imageSize,
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
				rep_error = stereoCalibrate(objectPoints, imagePoints[refCamera1], imagePoints[i], intrinsic[refCamera1], distortion[refCamera1], intrinsic[i], distortion[i], imageSize,
					reR[i], reT[i], E, F,
					flag,
					TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			}
		}

		rectifyMultiCollinear(intrinsic, distortion, refCamera1, refCamera2, imagePoints, imageSize, reR, reT, R, P, Q, -1.0, imageSize, 0, 0, rect_flag);
		initRemap();
		return rep_error;
	}

	double MultiCameraCalibrator::calibration(const int flags, int refCamera1, int refCamera2, const bool isIndependentCalibration)
	{
		this->flag = flags;
		return operator()(flags, refCamera1, refCamera2, isIndependentCalibration);
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

	void MultiCameraCalibrator::solvePnP(const int patternIndex, std::vector<cv::Mat>& destR, std::vector<cv::Mat>& destT)
	{
		destR.resize(numofcamera);
		destT.resize(numofcamera);
		for (int i = 0; i < numofcamera; i++)
		{
			cv::solvePnP(objectPoints[patternIndex], imagePoints[i][patternIndex], intrinsic[i], distortion[i], destR[i], destT[i]);
		}
	}

	void MultiCameraCalibrator::drawReprojectionError(string wname, const bool isInteractive, const int plotImageRadius)
	{
#if 1
		Size imsize(2 * plotImageRadius + 1, 2 * plotImageRadius + 1);
		Point center(imsize.width / 2, imsize.height / 2);
		Mat show(imsize, CV_8UC3);
		show.setTo(255);

		float scale = 1000.f;

		double terror = 0.0;
		Mat_<uchar> graybar = Mat(256, 1, CV_8U);
		Mat colorbar;
		for (int i = 0; i < 256; i++) graybar(i) = i;
		cv::applyColorMap(graybar, colorbar, COLORMAP_JET);
		int step = 256 / (int)objectPoints.size();

		if (isInteractive)
		{
			const int fontType = FONT_HERSHEY_SIMPLEX;
			namedWindow(wname);
			int index = 0;
			int camera = 0;
			createTrackbar("camera", wname, &camera, (int)imagePoints.size());
			createTrackbar("pattern index", wname, &index, (int)objectPoints.size());
			int gi = (int)objectPoints[0].size();
			createTrackbar("grid index", wname, &gi, (int)objectPoints[0].size());
			int sw = 0; createTrackbar("sw", wname, &sw, 1);
			int ColorMap = COLORMAP_JET; createTrackbar("ColorMap", wname, &ColorMap, COLORMAP_DEEPGREEN);
			int key = 0;

			while (key != 'q')
			{
				double terror = 0.0;
				cv::applyColorMap(graybar, colorbar, ColorMap);
				show.setTo(255);

				int start = 0;
				int end = (int)imagePoints.size();
				if (camera < imagePoints.size())
				{
					start = camera;
					end = camera + 1;
				}
				int count = 0;
				for (int cam = start; cam < end; cam++)
				{
					const bool isLast = cam == end - 1;
					if (index == objectPoints.size())
					{
						for (int i = 0; i < objectPoints.size(); i++)
						{
							Scalar color = Scalar(colorbar.ptr<uchar>(i * step)[0], colorbar.ptr<uchar>(i * step)[1], colorbar.ptr<uchar>(i * step)[2], 0.0);
							vector<Point2f> reprojectPoints;
							Mat local_r;
							Mat local_t;
							cv::solvePnP(objectPoints[i], imagePoints[cam][i], intrinsic[cam], distortion[cam], local_r, local_t);
							projectPoints(objectPoints[i], local_r, local_t, intrinsic[cam], distortion[cam], reprojectPoints);
							for (int n = 0; n < reprojectPoints.size(); n++)
							{
								float dx = imagePoints[cam][i][n].x - reprojectPoints[n].x;
								float dy = imagePoints[cam][i][n].y - reprojectPoints[n].y;
								terror = fma(dx, dx, fma(dy, dy, terror));
								count++;
								drawPlus(show, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
							}
						}
						//putText(show, format("Rep.Error: %6.4f", rep_error), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
					}
					else
					{
						Scalar color = Scalar(colorbar.ptr<uchar>(index * step)[0], colorbar.ptr<uchar>(index * step)[1], colorbar.ptr<uchar>(index * step)[2], 0.0);
						vector<Point2f> reprojectPoints;
						Mat local_r;
						Mat local_t;
						cv::solvePnP(objectPoints[index], imagePoints[cam][index], intrinsic[cam], distortion[cam], local_r, local_t);
						projectPoints(objectPoints[index], local_r, local_t, intrinsic[cam], distortion[cam], reprojectPoints);

						if (gi == objectPoints[0].size())
						{
							for (int n = 0; n < reprojectPoints.size(); n++)
							{
								if (sw == 1)
								{
									int step = 256 / (int)reprojectPoints.size();
									color = Scalar(colorbar.ptr<uchar>(n * step)[0], colorbar.ptr<uchar>(n * step)[1], colorbar.ptr<uchar>(n * step)[2], 0.0);
								}

								float dx = imagePoints[cam][index][n].x - reprojectPoints[n].x;
								float dy = imagePoints[cam][index][n].y - reprojectPoints[n].y;
								terror = fma(dx, dx, fma(dy, dy, terror));
								count++;
								drawPlus(show, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
							}
						}
						else
						{
							if (sw == 1)
							{
								int step = 256 / (int)reprojectPoints.size();
								color = Scalar(colorbar.ptr<uchar>(gi * step)[0], colorbar.ptr<uchar>(gi * step)[1], colorbar.ptr<uchar>(gi * step)[2], 0.0);
							}

							float dx = imagePoints[cam][index][gi].x - reprojectPoints[gi].x;
							float dy = imagePoints[cam][index][gi].y - reprojectPoints[gi].y;
							terror = fma(dx, dx, fma(dy, dy, terror));
							count++;
							drawPlus(show, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
						}
					}

					if (isLast)putText(show, format("Rep.Error: %6.4f", sqrt(terror / count)), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
					//draw image
					if (index < patternImages[cam].size())
					{
						if (!patternImages[cam][index].empty())
						{
							Mat show;
							if (patternImages[cam][index].channels() == 3)show = patternImages[cam][index].clone();
							else cvtColor(patternImages[cam][index], show, COLOR_GRAY2BGR);

							if (sw == 1)
							{
								if (gi == objectPoints[0].size())
								{
									for (int n = 0; n < imagePoints[cam][index].size(); n++)
									{
										int step = 256 / (int)imagePoints[cam][index].size();
										Scalar color = Scalar(colorbar.ptr<uchar>(n * step)[0], colorbar.ptr<uchar>(n * step)[1], colorbar.ptr<uchar>(n * step)[2], 0.0);
										circle(show, Point(imagePoints[cam][index][n]), 5, color, cv::FILLED);
									}
								}
								else
								{
									int step = 256 / (int)imagePoints[cam][index].size();
									Scalar color = Scalar(colorbar.ptr<uchar>(gi * step)[0], colorbar.ptr<uchar>(gi * step)[1], colorbar.ptr<uchar>(gi * step)[2], 0.0);
									circle(show, Point(imagePoints[cam][index][gi]), 5, color, cv::FILLED);
								}
							}
							imshow(wname + format("_image%d", cam), show);
						}
					}
				}

				putText(show, format("Rep.Error(All): %6.4f", rep_error), Point(0, 50), fontType, 0.5, COLOR_GRAY50);
				putText(show, format("%6.4f", plotImageRadius * 0.5 / scale), Point(plotImageRadius / 2 - 30, plotImageRadius / 2), fontType, 0.5, COLOR_GRAY50);
				circle(show, center, plotImageRadius / 2, COLOR_GRAY100);
				drawGridMulti(show, Size(4, 4), COLOR_GRAY200);
				Mat cres;
				resize(colorbar.t(), cres, Size(show.cols, 20));
				vconcat(show, cres, cres);
				imshow(wname, cres);
				key = waitKey(1);
			}
		}
		else
		{
#if 0
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
#endif
		}
		print_debug(rep_error);
		//print_debug(sqrt(terror / (objectPoints.size() * objectPoints[0].size())));
#endif
	}

	double MultiCameraCalibrator::plotDiffDisparityZPatternZ(const int cam0, const int cam1, const double f, const double l, const double d_offset)
	{
		Plot pt;
		pt.setKey(Plot::LEFT_TOP);
		pt.setIsDrawMousePosition(false);
		pt.setGrid(2);
		pt.setPlotTitle(0, "left");
		pt.setPlotTitle(1, "right");
		pt.setPlotLineType(0, cp::Plot::NOLINE);
		pt.setPlotLineType(1, cp::Plot::NOLINE);
		pt.setXLabel("disparity to Z");
		pt.setYLabel("difference");
		Mat localR0, localT0;
		Mat localR1, localT1;
		//vector<double> zL, zR, zD;
		vector<Point2f> dpl, dpr;
		double ret = 0.0;
		int count = 0;
		
		for (int p = 0; p < objectPoints.size(); p++)
		{
			cv::solvePnP(objectPoints[p], imagePoints[cam0][p], intrinsic[cam0], distortion[cam0], localR0, localT0);
			cv::Rodrigues(localR0, localR0);

			cv::solvePnP(objectPoints[p], imagePoints[cam1][p], intrinsic[cam1], distortion[cam1], localR1, localT1);
			cv::Rodrigues(localR1, localR1);
	
			undistortPoints(imagePoints[cam0][p], dpl, intrinsic[cam0], distortion[cam0], R[cam0], P[cam0]);
			undistortPoints(imagePoints[cam1][p], dpr, intrinsic[cam1], distortion[cam1], R[cam1], P[cam1]);

			for (int c = 0; c < objectPoints[0].size(); c++)
			{
				Mat a = Mat(objectPoints[p][c]);
				a.convertTo(a, CV_64F);
				Mat dL = localR0 * a + localT0;
				//Mat TL = localR0.t() * localT0;
				Mat dR = localR1 * a + localT1;

				const double disparity = double(dpl[c].x - dpr[c].x);	
				const double z = (rect_flag == cv::CALIB_ZERO_DISPARITY) ? f * l / (disparity) : f * l / (disparity + d_offset);
				ret += (dL.at<double>(2) - z) * (dL.at<double>(2) - z) + (dR.at<double>(2) - z) * (dR.at<double>(2) - z);
				count += 2;
				pt.push_back(z, dL.at<double>(2) - z, 0);
				pt.push_back(z, dR.at<double>(2) - z, 1);
				//pt.push_back(z, dL.at<double>(2), 0);
				//pt.push_back(z, dR.at<double>(2), 1);

				//zL.push_back(dL.at<double>(2));
				//zR.push_back(dR.at<double>(2));
				//zD.push_back(z);
			}
		}
		double rmse = sqrt(ret / count);

		pt.plot("Z", false, "", format("RMSE=%f",rmse));
		return rmse;
	}

	void MultiCameraCalibrator::guiDisparityTest()
	{
		string wname = "disparity";
		namedWindow(wname);
		int pi = 0; // pattern index
		createTrackbar("pattern index", wname, &pi, (int)objectPoints.size() - 1);
		int ci = 0;// corner index
		createTrackbar("corner index", wname, &ci, (int)objectPoints[pi].size() - 1);
		int showsw = 1; createTrackbar("show: 0_raw, 1_rect", wname, &showsw, 1);
		const int cami = 0;
		Mat localR, localT;
		int key = 0;
		cp::ConsoleImage con(Size(640, 480), wname);
		const double f = Q.at<double>(2, 3);
		const double l = 1.0 / Q.at<double>(3, 2);//norm(reT[cami + 1]);
		const double d_offset = l * Q.at<double>(3, 3);

		//all projection
		double z_error = plotDiffDisparityZPatternZ(0, 1, f, l, d_offset);

		//point-by-point projection
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
			Mat TL = localR.t() * localT;
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
			const float flz = float(f * l / (disparity));
			con("z=fl/d   %f %f %f", flz, flz - dL.at<double>(2), flz - dR.at<double>(2));
			const float flzo = float(f * l / (disparity + d_offset));
			con("z=fl/d(o)%f %f %f", flzo, flzo - dL.at<double>(2), flzo - dR.at<double>(2));
			con("RMSE z   %f", z_error);

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