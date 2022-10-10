#include "stereo_core.hpp"

using namespace std;
using namespace cv;
namespace cp
{

	void rotYaw(cv::InputArray src_, cv::OutputArray dest, const double yaw)
	{
		Mat src = src_.getMat();
		double angle = yaw / 180.0 * CV_PI;
		Mat rot = Mat::eye(3, 3, CV_64F);

		rot.at<double>(1, 1) = cos(angle);
		rot.at<double>(1, 2) = -sin(angle);
		rot.at<double>(2, 1) = sin(angle);
		rot.at<double>(2, 2) = cos(angle);

		Mat a = rot * src;
		a.clone().copyTo(dest);
	}

	void rotPitch(cv::InputArray src_, cv::OutputArray dest, const double pitch)
	{
		Mat src = src_.getMat();
		double angle = pitch / 180.0 * CV_PI;
		Mat rot = Mat::eye(3, 3, CV_64F);

		rot.at<double>(0, 0) = cos(angle);
		rot.at<double>(0, 2) = sin(angle);
		rot.at<double>(2, 0) = -sin(angle);
		rot.at<double>(2, 2) = cos(angle);

		Mat a = rot * src;
		a.clone().copyTo(dest);
	}

	void rotRoll(cv::InputArray src_, cv::OutputArray dest, const double roll)
	{
		Mat src__ = src_.getMat();
		const bool isRod = (min(src__.cols, src__.rows) == 1) ? true : false;

		Mat src;
		if (isRod)
		{
			Rodrigues(src__, src);
			if (src.depth() == CV_32F)src.convertTo(src, CV_64F);
		}
		else
		{
			if (src__.depth() == CV_32F)src__.convertTo(src, CV_64F);
			else src = src__;
		}

		const double angle = roll / 180.0 * CV_PI;
		Mat rot = Mat::eye(3, 3, CV_64F);

		rot.at<double>(0, 0) = cos(angle);
		rot.at<double>(0, 1) = -sin(angle);
		rot.at<double>(1, 0) = sin(angle);
		rot.at<double>(1, 1) = cos(angle);

		Mat a = rot * src;
		if (isRod) Rodrigues(a, a);
		if (src__.depth() == CV_32F)a.convertTo(dest, CV_32F);
		else a.clone().copyTo(dest);
		//cout << "return" << endl;
	}


	void Eular2Rotation(const double pitch, const double roll, const double yaw, cv::OutputArray dest, const int depth)
	{
		dest.create(3, 3, CV_64F);
		
		Mat a = Mat::eye(3, 3, CV_64F); a.copyTo(dest);
		rotYaw(dest, dest, yaw);
		rotPitch(dest, dest, pitch);
		rotRoll(dest, dest, roll);
	}

	void Rotation2Eular(InputArray R_, double& angle_x, double& angle_y, double& angle_z)
	{
		Mat R = R_.getMat();
		double threshold = 0.001;

		if (abs(R.at<double>(2, 1) - 1.0) < threshold) { // R(2,1) = sin(x) = 1
			angle_x = CV_PI / 2;
			angle_y = 0;
			angle_z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
		}
		else if (abs(R.at<double>(2, 1) + 1.0) < threshold) { // R(2,1) = sin(x) = -1
			angle_x = -CV_PI / 2;
			angle_y = 0;
			angle_z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
		}
		else {
			angle_x = asin(R.at<double>(2, 1));
			angle_y = atan2(-R.at<double>(2, 0), R.at<double>(2, 2));
			angle_z = atan2(-R.at<double>(0, 1), R.at<double>(1, 1));
		}
		angle_x = 180.0 * angle_x / CV_PI;
		angle_y = 180.0 * angle_y / CV_PI;
		angle_z = 180.0 * angle_z / CV_PI;
	}

	/*

	void point3d2Mat(const Point3d& src, Mat& dest)
	{
	dest.create(3,1,CV_64F);
	dest.at<double>(0,0)=src.x;
	dest.at<double>(1,0)=src.y;
	dest.at<double>(2,0)=src.z;
	}

	void setXYZ(Mat& in, double&x, double&y, double&z)
	{
	x=in.at<double>(0,0);
	y=in.at<double>(1,0);
	z=in.at<double>(2,0);

	//	cout<<format("set XYZ: %.04f %.04f %.04f\n",x,y,z);
	}

	void lookatBF(const Point3d& from, const Point3d& to, Mat& destR)
	{
	double x,y,z;

	Mat fromMat;
	Mat toMat;
	point3d2Mat(from,fromMat);
	point3d2Mat(to,toMat);

	Mat fromtoMat;
	add(toMat,fromMat,fromtoMat,Mat(),CV_64F);
	double ndiv = 1.0/norm(fromtoMat);
	fromtoMat*=ndiv;

	setXYZ(fromtoMat,x,y,z);
	destR = Mat::eye(3,3,CV_64F);
	double yaw   =-z/abs(z)*asin(y/sqrt(y*y+z*z))/CV_PI*180.0;

	rotYaw(destR,destR,yaw);

	Mat RfromtoMat = destR*fromtoMat;

	setXYZ(RfromtoMat,x,y,z);
	double pitch =z/abs(z)*asin(x/sqrt(x*x+z*z))/CV_PI*180.0;

	rotPitch(destR,destR,pitch);
	}
	*/
	void lookat(const Point3d& from, const Point3d& to, Mat& destR)
	{
		Mat destMat = Mat(Point3d(0.0, 0.0, 1.0));
		Mat srcMat = Mat(from + to);
		srcMat = srcMat / norm(srcMat);

		Mat rotaxis = srcMat.cross(destMat);
		const double angle = acos(srcMat.dot(destMat));
		//normalize cross product and multiply rotation angle
		rotaxis = rotaxis / norm(rotaxis) * angle;
		Rodrigues(rotaxis, destR);
	}

	void lookat(const Point3f& from, const Point3f& to, Mat& destR)
	{
		Mat destMat = Mat(Point3f(0.f, 0.f, 1.f));
		Mat srcMat = Mat(from + to);
		srcMat = srcMat / norm(srcMat);

		Mat rotaxis = srcMat.cross(destMat);
		const float angle = acos(srcMat.dot(destMat));
		//normalize cross product and multiply rotation angle
		rotaxis = rotaxis / norm(rotaxis) * angle;
		Rodrigues(rotaxis, destR);
	}
}