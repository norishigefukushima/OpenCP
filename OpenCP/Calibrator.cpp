#include "Calibrator.hpp"

using namespace std;
using namespace cv;

namespace cp
{

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
		initUndistortRectifyMap(intrinsic, distortion, R, P, imageSize, CV_32FC1,
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

		flag = CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_ZERO_TANGENT_DIST | CALIB_FIX_ASPECT_RATIO;

		generatechessboard3D();
	}

	Calibrator::Calibrator(Size imageSize_, Size patternSize_, float lengthofchess_)
	{
		init(imageSize_, patternSize_, lengthofchess_);
	}

	Calibrator::Calibrator(){ ; }

	Calibrator::~Calibrator(){ ; }

	Point2f Calibrator::getImagePoint(const int number_of_chess, const int index)
	{
		if (index<0 || index>patternSize.area())
		{
			float x = 0.0f;
			float y = 0.0f;
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
	void Calibrator::pushImagePoint(vector<Point2f> point)
	{
		numofchessboards++;
		imagePoints.push_back(point);
	}

	void Calibrator::pushObjectPoint(vector<Point3f> point)
	{
		objectPoints.push_back(point);
	}


	void Calibrator::printParameters()
	{
		std::cout << "intrinsic" << std::endl;
		cout << intrinsic << endl;

		cout << distortion << endl;
	}

	void check(InputArrayOfArrays a)
	{
		cout << a.total() << endl;
		Mat b = a.getMat();

	}

	double Calibrator::operator()()
	{
		if (numofchessboards < 2)
		{
			std::cout << "input 3 or more chessboards" << std::endl;
			return -1;
		}
		rep_error = calibrateCamera(objectPoints, imagePoints, imageSize, intrinsic, distortion, rt, tv, flag);
		initRemap();
		return rep_error;
	}

	double Calibrator::calibration(const int flag)
	{
		this->flag = flag;
		return operator()();
	}

	void Calibrator::undistort(Mat& src, Mat& dest)
	{
		remap(src, dest, mapu, mapv, INTER_LINEAR);
	}
}