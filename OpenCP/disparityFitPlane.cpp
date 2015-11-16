#include "opencp.hpp"
#include <opencv2/viz.hpp>
#include <fstream>
using namespace std;
using namespace cv;

namespace cp
{
	template <typename T, typename S>
	void _SLICSegment2Vector3D_(cv::InputArray segment, cv::InputArray signal, T invalidValue, std::vector<std::vector<cv::Point3_<S>>>& segmentPoint)
	{
		Mat sig = signal.getMat();
		Mat seg = segment.getMat();

		for (int j = 0; j < seg.rows; ++j)
		{
			int* s = seg.ptr<int>(j);
			T* value = sig.ptr<T>(j);
			for (int i = 0; i < seg.cols; ++i)
			{
				if (value[i] != invalidValue)
				{
						segmentPoint[s[i]].push_back(Point3_<S>((S)i, (S)j, (S)value[i]));
				}
			}
		}
	}

	template <typename S>
	void SLICSegment2Vector3D_(cv::InputArray segment, cv::InputArray signal, double invalidValue, std::vector<std::vector<cv::Point3_<S>>>& segmentPoint)
	{
		double minv, maxv;
		minMaxLoc(segment, &minv, &maxv);
		segmentPoint.clear();
		segmentPoint.resize((int)maxv + 1);
		if (signal.depth() == CV_8U) _SLICSegment2Vector3D_<uchar, S>(segment, signal, (uchar)invalidValue, segmentPoint);
		else if (signal.depth() == CV_16S) _SLICSegment2Vector3D_<short, S>(segment, signal, (short)invalidValue, segmentPoint);
		else if (signal.depth() == CV_16U) _SLICSegment2Vector3D_<ushort, S>(segment, signal, (ushort)invalidValue, segmentPoint);
		else if (signal.depth() == CV_32S) _SLICSegment2Vector3D_<int, S>(segment, signal, (int)invalidValue, segmentPoint);
		else if (signal.depth() == CV_32F) _SLICSegment2Vector3D_<float, S>(segment, signal, (float)invalidValue, segmentPoint);
		else if (signal.depth() == CV_64F) _SLICSegment2Vector3D_<double, S>(segment, signal, (double)invalidValue, segmentPoint);
	}

	static int countArrowablePointDistanceZ(const std::vector<cv::Point3f>& points, Point3f& abc, float threshold)
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
	/*
	void dispalityFitPlane(cv::InputArray disparity, cv::InputArray image, cv::OutputArray dest, int slicRegionSize, float slicRegularization, float slicMinRegionRatio, int slicMaxIteration, int ransacNumofSample, float ransacThreshold)
	{
		Mat segment;

		SLIC(image, segment, slicRegionSize, slicRegularization, slicMinRegionRatio, slicMaxIteration);

		vector<vector<Point3f>> points;
		SLICSegment2Vector3D_<float>(segment, disparity, 0, points);

		Mat disp32f = Mat::zeros(image.size(), CV_32F);

		Mat show = Mat::zeros(image.size(), CV_8U);
		Mat idx = Mat::zeros(image.size(), CV_8U);
		for (int i = 0; i < points.size(); ++i)
		{
			if (points[i].size() < 3)
			{
				if (!points[i].empty())
				{
					for (int j = 0; j < points[i].size(); ++j)
					{
						points[i][j].z = 0.f;
					}
				}
			}
			else
			{
				Point3f abc;
				fitPlaneRANSAC(points[i], abc, ransacNumofSample, ransacThreshold, 1);
				int v = countArrowablePointDistanceZ(points[i], abc, ransacThreshold);
				double rate = (double)v / points[i].size() * 100;
								
				//float error = 0.f; for (int j = 0; j < points[i].size(); ++j) error += abs(points[i][j].z - (points[i][j].x*abc.x + points[i][j].y*abc.y + abc.z));
				//cout << i << "," << error / points[i].size() << endl;

				for (int j = 0; j < points[i].size(); ++j)
				{
					points[i][j].z = points[i][j].x*abc.x + points[i][j].y*abc.y + abc.z;
				}
				
				for (int j = 0; j < points[i].size(); ++j)
				{
					int x = cvRound(points[i][j].x);
					int y = cvRound(points[i][j].y);
					show.at<uchar>(y, x) = points[i][j].z*0.125;
					idx.at<uchar>(y, x) = i%256;
				}

				if (rate < 30)
				{
					cout << "rate0: " << rate << ": " << v << "/" << points[i].size() << endl;

					fitPlaneRANSAC(points[i], abc, ransacNumofSample, ransacThreshold, 1);
					v = countArrowablePointDistanceZ(points[i], abc, ransacThreshold);
					rate = (double)v / points[i].size() * 100;
					cout << "rate1: " << rate << ": " << v << "/" << points[i].size() << endl;

					fitPlaneRANSAC(points[i], abc, ransacNumofSample, ransacThreshold, 1);
					v = countArrowablePointDistanceZ(points[i], abc, ransacThreshold);
					rate = (double)v / points[i].size() * 100;
					cout << "rate2: " << rate << ": " << v << "/" << points[i].size() << endl;

					fitPlaneRANSAC(points[i], abc, ransacNumofSample, ransacThreshold, 1);
					v = countArrowablePointDistanceZ(points[i], abc, ransacThreshold);
					rate = (double)v / points[i].size() * 100;
					cout << "rate3: " << rate << ": " << v << "/" << points[i].size() << endl;

					//cout << abc << endl;
					imshow("debug", show);
					imshow("debug2", idx);
					waitKey();
				}
				
			}
		}

		SLICVector3D2Signal(points, image.size(), disp32f);
		if (disparity.depth() == CV_32F)
		{
			disp32f.copyTo(dest);
		}
		else if (disparity.depth() == CV_8U || disparity.depth() == CV_16U || disparity.depth() == CV_16S || disparity.depth() == CV_32S)
		{
			disp32f.convertTo(dest, disparity.type(), 1.0, 0.5);
		}
		else
		{
			disp32f.convertTo(dest, disparity.type());
		}
	}
	*/
	Point3f getMean(vector<Point3f>& src)
	{
		Point3f ret = Point3f(0.f,0.f,0.f);
		for (int i = 0; i < src.size(); i++)
		{
			ret += src[i];
		}
		return ret / (float)src.size();
	}

	void subMean(vector<Point3f>& src, Point3f mean)
	{
		Point3f ret = Point3f(0.f, 0.f, 0.f);
		for (int i = 0; i < src.size(); i++)
		{
			src[i] -= mean;
		}	
	}

	void addMean(vector<Point3f>& src, Point3f mean)
	{
		Point3f ret = Point3f(0.f, 0.f, 0.f);
		for (int i = 0; i < src.size(); i++)
		{
			src[i] += mean;
		}
	}

	static void randomSampleVector2Vec3f(vector<Point3f>& src, vector<Point3f>& dest, int numofsample = 3)
	{
		if (!dest.empty())dest.clear();
		cv::RNG rnd(cv::getTickCount());

		int s = (int)src.size();
		for (int i = 0; i < numofsample; i++)
		{
			dest.push_back(src[rnd.uniform(0, s)]);
		}
	}
	void disparityFitTestTest(int ransacNumofSample, float ransacThreshold)
	{
		cv::FileStorage pointxml("planePoint.xml", cv::FileStorage::READ);
		namedWindow("test");
		
		gnuplot plot("C:/fukushima/docs/Dropbox/bin/gnuplot/bin/pgnuplot.exe");
		//gnuplot plot("C:/fukushima/docs/Dropbox/bin/gnuplot/bin/wgnuplot_pipes.exe");
	
		int idx = 0;
		for (int i = 0; i < 300; i++)
		{
			int key = 0;
			while (key != 'n')
			{
				cv::RNG rnd(cv::getTickCount());

				vector<Point3f> a;
				std::ofstream raw("raw.dat");
				std::ofstream ref("ref.dat");
				pointxml[format("point%03d", idx)] >> a;

				Point3f abc;
				fitPlaneRANSAC(a, abc, ransacNumofSample, ransacThreshold, 0);

				//vector<Point3f> point3;
				//randomSampleVector2Vec3f(a, point3, 3);
				//fitPlaneCrossProduct(point3, abc);
				//int num = countArrowablePointDistanceZ(a, abc, ransacThreshold);
				//cout << num << endl;
				for (int i = 0; i < a.size(); i++)
				{
					//cout << a << endl;
					raw << format("%f %f %f\n", a[i].x, a[i].y, a[i].z);
					ref << format("%f %f %f\n", a[i].x, a[i].y, abc.x*a[i].x + abc.y*a[i].y + abc.z);
				}
			
				raw.close();
				ref.close();
				plot.cmd("sp 'raw.dat', 'ref.dat'");
				key = waitKey();
			}
			idx++;
		}
	}

	void dispalityFitPlane(cv::InputArray disparity, cv::InputArray image, cv::OutputArray dest, int slicRegionSize, float slicRegularization, float slicMinRegionRatio, int slicMaxIteration, int ransacNumofSample, float ransacThreshold)
	{
		//disparityFitTest(ransacNumofSample, ransacThreshold);
		//cv::FileStorage pointxml("planePoint.xml", cv::FileStorage::WRITE); int err = 0;
		Mat segment;
		
		SLIC(image, segment, slicRegionSize, slicRegularization, slicMinRegionRatio, slicMaxIteration);
		
		vector<vector<Point3f>> points;
		SLICSegment2Vector3D_<float>(segment, disparity, 0, points);

		Mat disp32f = Mat::zeros(dest.size(), CV_32F);

		
		
		for (int i = 0; i < points.size(); ++i)
		{	
			if (points[i].size() < 3)
			{
				if (!points[i].empty())
				{
					for (int j = 0; j < points[i].size(); ++j)
					{
						points[i][j].z = 0.f;
					}
				}
			}
			else
			{
				Point3f abc;

				fitPlaneRANSAC(points[i], abc, ransacNumofSample, ransacThreshold, 1);
				int v = countArrowablePointDistanceZ(points[i], abc, ransacThreshold);
				double rate = (double)v / points[i].size() * 100;

				int itermax = 1;
				for (int n = 0; n < itermax;n++)
				{
					if (rate < 40)
					{
						//pointxml <<format("point%03d",err++)<< points[i];

						fitPlaneRANSAC(points[i], abc, ransacNumofSample, ransacThreshold, 1);
						v = countArrowablePointDistanceZ(points[i], abc, ransacThreshold);
						rate = (double)v / points[i].size() * 100;
					}
				}
	
				for (int j = 0; j < points[i].size(); ++j)
				{
					points[i][j].z = points[i][j].x*abc.x + points[i][j].y*abc.y + abc.z;
				}			
			}
		}

		SLICVector3D2Signal(points, image.size(), disp32f);
		if (disparity.depth() == CV_32F)
		{
			disp32f.copyTo(dest);
		}
		else if (disparity.depth() == CV_8U || disparity.depth() == CV_16U || disparity.depth() == CV_16S || disparity.depth() == CV_32S)
		{
			disp32f.convertTo(dest, disparity.type(), 1.0, 0.5);
		}
		else
		{
			disp32f.convertTo(dest, disparity.type());
		}
	}
	
}