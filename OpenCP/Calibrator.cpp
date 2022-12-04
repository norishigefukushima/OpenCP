#include "Calibrator.hpp"
#include "draw.hpp"
#include "debugcp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void drawPatternIndexNumbers(cv::Mat& dest, const std::vector<cv::Point>& points, const double scale, const Scalar color)
	{
		for (int i = 0; i < points.size(); i++)
		{
			drawPlus(dest, points[i], 1);
			putText(dest, format("%d", i), points[i], cv::FONT_HERSHEY_SIMPLEX, scale, color);
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
				const Point2f diff = Point2f(mapu.at<float>(j, i), mapv.at<float>(j, i)) - Point2f(i, j);
				const Point d = Point(Point2f(s) + amp * diff);
				//print_debug3(mapu.at<float>(j, i), i, mapu.at<float>(j, i) - i);

				//cv::circle(dest, s, thickness, COLOR_ORANGE, cv::FILLED);
				cv::line(dest, s, d, COLOR_ORANGE, max(thickness, 1));
			}
		}
	}

	void Calibrator::generatechessboard3D()
	{
		for (int j = 0; j < patternSize.height; ++j)
		{
			for (int i = 0; i < patternSize.width; ++i)
			{
				chessboard3D.push_back(Point3f(lengthofchess * (float)i, lengthofchess * (float)j, 0.0));
			}
		}
	}

	void Calibrator::initRemap()
	{
		Mat P = Mat::eye(3, 3, CV_64F);
		Mat R = Mat::eye(3, 3, CV_64F);
		initUndistortRectifyMap(intrinsic, distortion, R, intrinsic, imageSize, CV_32FC1,
			mapu, mapv);
	}

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

	void Calibrator::init(Size imageSize_, Size patternSize_, float lengthofchess_)
	{
		numofchessboards = 0;
		imageSize = imageSize_;
		patternSize = patternSize_;
		lengthofchess = lengthofchess_;
		distortion = Mat::zeros(1, 8, CV_64F);

		intrinsic = Mat::eye(3, 3, CV_64F);
		getDefaultNewCameraMatrix(intrinsic, imageSize, true);
		generatechessboard3D();
	}

	Calibrator::Calibrator(Size imageSize_, Size patternSize_, float lengthofchess_)
	{
		init(imageSize_, patternSize_, lengthofchess_);
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
			rep_error = calibrateCamera(objectPoints, imagePoints, imageSize, intrinsic, distortion, rt, tv, flag | CALIB_USE_INTRINSIC_GUESS);
		}
		else
		{
			rep_error = calibrateCamera(objectPoints, imagePoints, imageSize, intrinsic, distortion, rt, tv, flag);
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

	void Calibrator::drawReprojectionError(string wname, const bool isInteractive)
	{
		int length = 400;
		Size imsize(2 * length + 1, 2 * length + 1);
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

			string wnameplot = "grid position";
			
			Mat gridposition(imageSize, CV_8UC3);

			namedWindow(wname);
			int index = 0;
			const int thmax = 10000;
			int thresh = thmax; createTrackbar("threshold*0.0001(max 1pixel)", wname, &thresh, thmax);//max 1pixel
			createTrackbar("pattern index", wname, &index, (int)objectPoints.size());

			int gix = patternSize.width;
			int giy = patternSize.height;
			createTrackbar("grid index X", wname, &gix, patternSize.width);
			createTrackbar("grid index Y", wname, &giy, patternSize.height);
			int sw = 0; createTrackbar("sw", wname, &sw, 1);
			int ColorMap = COLORMAP_JET; createTrackbar("ColorMap", wname, &ColorMap, COLORMAP_DEEPGREEN);
			int key = 0;
			while (key != 'q')
			{
				const float th2 = (thresh== thmax)?FLT_MAX:(thresh * 0.0001f) * (thresh * 0.0001f);
				gridposition.setTo(255);

				int gi = patternSize.width * giy + gix;
				gi = (gix == patternSize.width) ? (int)objectPoints[0].size() : gi;
				gi = (giy == patternSize.height) ? (int)objectPoints[0].size() : gi;

				double terror = 0.0;
				cv::applyColorMap(graybar, colorbar, ColorMap);
				show.setTo(255);
				if (index == objectPoints.size())
				{
					for (int i = 0; i < objectPoints.size(); i++)
					{
						Scalar color = Scalar(colorbar.ptr<uchar>(i * step)[0], colorbar.ptr<uchar>(i * step)[1], colorbar.ptr<uchar>(i * step)[2], 0.0);
						vector<Point2f> reprojectPoints;
						projectPoints(objectPoints[i], rt[i], tv[i], intrinsic, distortion, reprojectPoints);
						for (int n = 0; n < reprojectPoints.size(); n++)
						{
							const float dx = imagePoints[i][n].x - reprojectPoints[n].x;
							const float dy = imagePoints[i][n].y - reprojectPoints[n].y;
							terror = fma(dx, dx, fma(dy, dy, terror));
							
							if (dx * dx + dy * dy < th2)
							{
								drawPlus(gridposition, Point(imagePoints[i][n]), 2, Scalar(255, 0, 0), 2);
								drawPlus(show, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
							}
						}
					}
					putText(show, format("Rep.Error: %6.4f", rep_error), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
				}
				else
				{
					Scalar color = Scalar(colorbar.ptr<uchar>(index * step)[0], colorbar.ptr<uchar>(index * step)[1], colorbar.ptr<uchar>(index * step)[2], 0.0);
					vector<Point2f> reprojectPoints;
					projectPoints(objectPoints[index], rt[index], tv[index], intrinsic, distortion, reprojectPoints);
					if (gi == objectPoints[0].size())
					{
						for (int n = 0; n < reprojectPoints.size(); n++)
						{
							if (sw == 1)
							{
								int step = 256 / (int)reprojectPoints.size();
								color = Scalar(colorbar.ptr<uchar>(n * step)[0], colorbar.ptr<uchar>(n * step)[1], colorbar.ptr<uchar>(n * step)[2], 0.0);
							}

							const float dx = imagePoints[index][n].x - reprojectPoints[n].x;
							const float dy = imagePoints[index][n].y - reprojectPoints[n].y;
							terror = fma(dx, dx, fma(dy, dy, terror));
							
							if (dx * dx + dy * dy < th2)
							{
								drawPlus(gridposition, Point(imagePoints[index][n]), 2, Scalar(255, 0, 0), 2);
								drawPlus(show, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
							}
						}
						putText(show, format("Rep.Error: %6.4f", sqrt(terror / (objectPoints[0].size()))), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
					}
					else
					{
						if (sw == 1)
						{
							int step = 256 / (int)reprojectPoints.size();
							color = Scalar(colorbar.ptr<uchar>(gi * step)[0], colorbar.ptr<uchar>(gi * step)[1], colorbar.ptr<uchar>(gi * step)[2], 0.0);
						}

						const float dx = imagePoints[index][gi].x - reprojectPoints[gi].x;
						const float dy = imagePoints[index][gi].y - reprojectPoints[gi].y;
						terror = fma(dx, dx, fma(dy, dy, terror));
						
						if (dx * dx + dy * dy < th2)
						{
							drawPlus(gridposition, Point(imagePoints[index][gi]), 2, Scalar(255, 0, 0), 2);
							drawPlus(show, center + Point(cvRound(dx * scale), cvRound(dy * scale)), 2, color, 2);
						}

						putText(show, format("Rep.Error: %6.4f", sqrt(terror)), Point(0, 20), fontType, 0.5, COLOR_GRAY50);
					}
				}

				putText(show, format("%6.4f", length * 0.5 / scale), Point(length / 2 - 30, length / 2), fontType, 0.5, COLOR_GRAY50);
				circle(show, center, length / 2, COLOR_GRAY100);
				drawGridMulti(show, Size(4, 4), COLOR_GRAY200);
				Mat cres;
				resize(colorbar.t(), cres, Size(show.cols, 20));
				vconcat(show, cres, cres);
				imshow(wname, cres);

				imshow(wnameplot, gridposition);

				key = waitKey(1);
				if (index < patternImages.size())
				{
					if (!patternImages[index].empty())
					{
						Mat show;
						if (patternImages[index].channels() == 3)show = patternImages[index].clone();
						else cvtColor(patternImages[index], show, COLOR_GRAY2BGR);

						if (sw == 1)
						{
							if (gi == objectPoints[0].size())
							{
								for (int n = 0; n < imagePoints[index].size(); n++)
								{
									int step = 256 / (int)imagePoints[index].size();
									Scalar color = Scalar(colorbar.ptr<uchar>(n * step)[0], colorbar.ptr<uchar>(n * step)[1], colorbar.ptr<uchar>(n * step)[2], 0.0);
									circle(show, Point(imagePoints[index][n]), 5, color, cv::FILLED);
								}
							}
							else
							{
								int step = 256 / (int)imagePoints[index].size();
								Scalar color = Scalar(colorbar.ptr<uchar>(gi * step)[0], colorbar.ptr<uchar>(gi * step)[1], colorbar.ptr<uchar>(gi * step)[2], 0.0);
								circle(show, Point(imagePoints[index][gi]), 5, color, cv::FILLED);
							}
						}
						imshow(wname + "_image", show);
					}

				}
			}
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
		print_debug(rep_error);
		//print_debug(sqrt(terror / (objectPoints.size() * objectPoints[0].size())));
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
			int amp = 100; createTrackbar("amp", wname, &amp, 2000);
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