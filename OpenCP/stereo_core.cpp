#include "opencp.hpp"

using namespace std;
using namespace cv;
namespace cp
{

	void rotYaw(Mat& src, Mat& dest, const double yaw)
	{
		double angle = yaw / 180.0*CV_PI;
		Mat rot = Mat::eye(3, 3, CV_64F);

		rot.at<double>(1, 1) = cos(angle);
		rot.at<double>(1, 2) = sin(angle);
		rot.at<double>(2, 1) = -sin(angle);
		rot.at<double>(2, 2) = cos(angle);

		Mat a = rot*src;
		a.copyTo(dest);
	}

	void rotPitch(Mat& src, Mat& dest, const double pitch)
	{
		double angle = pitch / 180.0*CV_PI;
		Mat rot = Mat::eye(3, 3, CV_64F);

		rot.at<double>(0, 0) = cos(angle);
		rot.at<double>(0, 2) = -sin(angle);
		rot.at<double>(2, 0) = sin(angle);
		rot.at<double>(2, 2) = cos(angle);

		Mat a = rot*src;
		a.copyTo(dest);
	}






	void eular2rot(double pitch, double roll, double yaw, Mat& dest)
	{
		dest = Mat::eye(3, 3, CV_64F);
		rotYaw(dest, dest, yaw);
		rotPitch(dest, dest, pitch);
		rotPitch(dest, dest, roll);
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
		double angle = acos(srcMat.dot(destMat));
		//normalize cross product and multiply rotation angle
		rotaxis = rotaxis / norm(rotaxis)*angle;
		Rodrigues(rotaxis, destR);
	}
}