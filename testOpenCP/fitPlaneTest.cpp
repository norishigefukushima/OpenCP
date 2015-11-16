#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void generatePlane(Point3f& abc, OutputArray dest)
{
	CV_Assert(dest.empty() || dest.depth() == CV_32F);
	Mat dst = dest.getMat();
	for (int j = 0; j < dst.rows; ++j)
	{
		for (int i = 0; i < dst.cols; ++i)
		{
			dst.at<float>(j, i) = abc.x*i + abc.y*j + abc.z;
		}
	}
}

void randomSampleMat2Vec3f(Mat& src, vector<Point3f>& dest, int numofsample = 3)
{
	cv::RNG rnd(cv::getTickCount());
	int x, y;
	for (int i = 0; i < numofsample; i++)
	{
		x = rnd.uniform(0, src.cols);
		y = rnd.uniform(0, src.rows);
		dest.push_back(Point3f((float)x, (float)y, src.at<float>(y, x)));
	}
}

void mat2vec3f(Mat& src, vector<Point3f>& dest)
{
	dest.clear();
	dest.resize(src.size().area());

	int n = 0;
	for (int j = 0; j < src.rows; ++j)
	{
		for (int i = 0; i < src.cols; ++i)
		{
			dest[n++] = Point3f((float)i, (float)j, src.at<float>(j, i));
		}
	}
}

void fitPlaneTest()
{
	Point3f abc(5.0f, 2.0f, 3.0f);
	Mat plane = Mat::zeros(Size(512, 512), CV_32F);
	generatePlane(abc, plane);

	Point3f dest;

	vector<Point3f> input3;
	randomSampleMat2Vec3f(plane, input3, 3);

	//crossproduct from 3point
	fitPlaneCrossProduct(input3, dest);
	cout << "cp:" << dest << endl;

	Mat noise;
	addNoise(plane, noise, 2, 0.0);
	{CalcTime t("PCA");
	//PCA from noisy input without solt pepper noise;
	vector<Point3f> input;
	mat2vec3f(noise, input);
	fitPlanePCA(input, dest);
	cout << "pca without solt pepper:" << dest << endl;
	}
	addNoise(plane, noise, 2, 0.1);
	{CalcTime t("PCA");
	//PCA from noisy input without solt pepper noise;
	vector<Point3f> input;
	mat2vec3f(noise, input);
	fitPlanePCA(input, dest);
	cout << "pca with solt pepper:" << dest << endl;
	}
	//RANSAC	
	{CalcTime t("RANSAC");
	vector<Point3f> points;
	mat2vec3f(noise, points);
	fitPlaneRANSAC(points, dest, 50, 5.f, 0);
	cout << "ransac:" << dest << endl;
	}
	//RANSAC
	{CalcTime t("RANSAC+1 iter");
	vector<Point3f> points;
	mat2vec3f(noise, points);
	fitPlaneRANSAC(points, dest, 50, 5.f, 1);
	cout << "ransac+pca1 ref:" << dest << endl;
	}
}