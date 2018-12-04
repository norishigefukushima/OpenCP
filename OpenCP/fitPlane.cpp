#include "fitPlane.hpp"

using namespace std;
using namespace cv;

//todo
//“ü—Í‚Ì”Ä—p«Einput array
//distance mesure:onlyz ->point to plane
//estimation abc: PCA->direct SVD

namespace cp
{
	inline void solveABC(const Point3f& normal, const Point3f& point, Point3f& abc)
	{
		abc.x = normal.x / -normal.z;
		abc.y = normal.y / -normal.z;
		abc.z = (normal.x*point.x + normal.y*point.y + normal.z*point.z) / normal.z;
	}

	void fitPlaneCrossProduct(std::vector<cv::Point3f>& src, Point3f& dest)
	{
		Point3f mean = (src[0] + src[1] + src[2]) / 3;
		Point3f normal = (src[0] - src[1]).cross(src[0] - src[2]);
		solveABC(normal, mean, dest);
	}

	inline void fitPlaneCrossProduct_(std::vector<cv::Point3f>& src, Point3f& dest)
	{
		Point3f mean = (src[0] + src[1] + src[2]) / 3;
		Point3f normal = (src[0] - src[1]).cross(src[0] - src[2]);
		solveABC(normal, mean, dest);
	}

	void fitPlanePCA(InputArray src, Point3f& dest)
	{
		Mat s = src.getMat();
		Mat smat = s.reshape(1, s.cols);
		Mat d; smat.convertTo(d, CV_64F);

		PCA pca(d, Mat(), PCA::DATA_AS_ROW);

		Point3f  normal = pca.eigenvectors.row(2);
		Point3f mean = pca.mean;
		solveABC(normal, mean, dest);
	}

	int countArrowablePoint(Mat& plane, Mat& plane2, float threshold)
	{
		int n = 0;
		int count = 0;
		for (int j = 0; j < plane.rows; ++j)
		{
			for (int i = 0; i < plane.cols; ++i)
			{
				if (abs(plane.at<float>(j, i) - plane2.at<float>(j, i)) < threshold)
				{
					count++;
				}
				n++;
			}
		}
		return count;
	}

	int countArrowablePoint(Mat& plane, std::vector<cv::Point3f>& points, float threshold)
	{
		int n = 0;
		int count = 0;
		for (int j = 0; j < plane.rows; ++j)
		{
			for (int i = 0; i < plane.cols; ++i)
			{
				if (abs(plane.at<float>(j, i) - points[n].x*i + points[n].y*j + points[n].z) < threshold)
				{
					count++;
				}
				n++;
			}
		}
		return count;
	}

	//only consider distance of Z: SVD solution for fixed x,y is sutable
	static int countArrowablePointDistanceZ(std::vector<cv::Point3f>& points, Point3f& abc, float threshold)
	{
		int count = 0;

		for (int n = 0; n < points.size(); ++n)
		{
			if (abs(points[n].z - (points[n].x*abc.x + points[n].y*abc.y + abc.z)) < threshold)
			{
				count++;
			}
		}
		return count;
	}

	//only consider distance of Z: SVD solution for fixed x,y is sutable
	static int filterArrowablePointDistanceZ(const std::vector<cv::Point3f>& points, std::vector<cv::Point3f>& dest, const Point3f& abc, float threshold)
	{
		dest.clear();
		for (int n = 0; n < points.size(); ++n)
		{
			if (abs(points[n].z - (points[n].x*abc.x + points[n].y*abc.y + abc.z)) < threshold)
			{
				dest.push_back(points[n]);
			}
		}

		return (int)dest.size();
	}

	static void randomSampleVector2Vec3f(const vector<Point3f>& src, vector<Point3f>& dest, int numofsample = 3)
	{
		if (!dest.empty())
		{
			dest.clear();
			dest.resize(0);
		}
		cv::RNG rnd(cv::getTickCount());

		int s = (int)src.size();
		for (int i = 0; i < numofsample; i++)
		{
			dest.push_back(src[rnd.uniform(0, s)]);
		}
	}

	void fitPlaneRANSAC(std::vector<cv::Point3f>& src, Point3f& dest, int numofsample, float threshold, int refineIteration)
	{
		int numValidSample = 0;
		Point3f abc = Point3f(0.f, 0.f, 0.f);

		for (int i = 0; i < numofsample; i++)
		{
			vector<Point3f> point3;
			Point3f abctemp;
			randomSampleVector2Vec3f(src, point3, 3);
			fitPlaneCrossProduct_(point3, abctemp);

			if (checkRange(Mat(abctemp)))
			{
				int num = countArrowablePointDistanceZ(src, abctemp, threshold);
				if (numValidSample < num)
				{
					numValidSample = num;
					abc = abctemp;
				}
			}
		}

		vector<Point3f> filtered;
		Point3f mean;
		int samples = filterArrowablePointDistanceZ(src, filtered, abc, threshold);
		
		//cout << abc << endl;
		//cout <<"final:"<< samples << "/"<<src.size()<<endl;
		if (samples>=3) fitPlanePCA(filtered, dest);
		
		for (int i = 0; i < refineIteration; i++)
		{
			vector<Point3f> filtered2;
			int samples = filterArrowablePointDistanceZ(src, filtered2, dest, threshold);
			if (samples>=3)fitPlanePCA(filtered2, dest);
		}	
	}
}