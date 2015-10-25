#include "opencp.hpp"
using namespace cv;
using namespace std;

namespace cp
{

	void diffshow(string wname, InputArray src, InputArray ref, const double scale)
	{
		Mat show;
		Mat diff = Mat::zeros(src.size(), CV_64F);
		subtract(src.getMat(), ref.getMat(), diff, noArray(), CV_64F);
		diff *= scale;
		diff += 128.0;
		diff.convertTo(show, CV_8U, 1.0, 0.5);
		imshow(wname, show);
	}

	void guiCompareDiff(const Mat& before, const Mat& after, const Mat& ref)
	{
		string wname = "comp diff";
		namedWindow(wname);
		static int sw = 0; createTrackbar("switch", wname, &sw, 2);
		static int a = 0; createTrackbar("alpha", wname, &a, 100);
		static int th = 0; createTrackbar("thresh", wname, &th, 255);

		int key = 0;
		Mat show;
		Mat g1, g2, g;
		if (before.channels() == 3)cvtColor(before, g1, CV_BGR2GRAY);
		else  g1 = before;
		if (after.channels() == 3)cvtColor(after, g2, CV_BGR2GRAY);
		else  g2 = after;
		if (ref.channels() == 3)cvtColor(ref, g, CV_BGR2GRAY);
		else  g = ref;

		Mat cmap(before.size(), CV_8UC3);

		Mat db = abs(g - g1);
		Mat da = abs(g - g2);
		Mat d = abs(g1 - g2);
		Mat maskB;
		Mat maskA;
		Mat zmask;


		while (key != 'q')
		{
			cmap.setTo(0);
			compare(d, th, zmask, cv::CMP_LE);

			compare(da, db, maskB, cv::CMP_GT);
			maskB.setTo(0, zmask);
			cmap.setTo(Scalar(0, 0, 255), maskB);

			compare(db, da, maskA, cv::CMP_GT);
			maskA.setTo(0, zmask);
			cmap.setTo(Scalar(0, 255, 0), maskA);

			if (sw == 0)
				alphaBlend(ref, cmap, a / 100.0, show);
			else if (sw == 1)
				alphaBlend(ref, before, a / 100.0, show);
			else if (sw == 2)
				alphaBlend(ref, after, a / 100.0, show);

			imshow(wname, show);
			key = waitKey(1);
			if (key == 'c')
			{
				cout << "red  :" << 100.0*countNonZero(maskB) / (double)maskA.size().area() << "%: " << endl;
				cout << "green:" << 100.0*countNonZero(maskA) / (double)maskA.size().area() << "%: " << endl;

				cout << "before L2       " << norm(db, NORM_L2) << endl;
				cout << "after  L2       " << norm(da, NORM_L2) << endl;

				cout << "before L2 masked" << norm(db, NORM_L2, maskB) << endl;
				cout << "after  L2 masked" << norm(da, NORM_L2, maskA) << endl;
			}
			if (key == 'f')
			{
				a = (a > 0) ? 0 : 100;
				setTrackbarPos("alpha", wname, a);
			}
			if (key == 'h')
			{
				cout << "red: before is better (degraded)" << endl;
				cout << "green: after is better (improved)" << endl;
			}
		}
		destroyWindow(wname);
	}


	void guiAbsDiffCompareGE(const Mat& src1, const Mat& src2)
	{
		string wname = "Diff Threshold GE";
		namedWindow(wname);

		static int alpha = 100; createTrackbar("alpha", wname, &alpha, 100);
		static int th = 1; createTrackbar("thresh", wname, &th, 255);
		int key = 0;
		Mat dest;

		Mat s1, s2;
		if (src1.channels() == 3)
			cvtColor(src1, s1, COLOR_BGR2GRAY);
		else s1 = src1;
		if (src2.channels() == 3)
			cvtColor(src2, s2, COLOR_BGR2GRAY);
		else s2 = src2;

		while (key != 'q')
		{
			Mat diff = abs(s1 - s2);
			compare(diff, th, dest, CMP_GE);
			int num = countNonZero(dest);
			if (src1.depth() != dest.depth())
			{
				Mat s;
				src1.convertTo(s, CV_8U);
				alphaBlend(dest, s, alpha / 100.0, dest);
			}
			else
				alphaBlend(dest, src1, alpha / 100.0, dest);

			imshow(wname, dest);
			key = waitKey(1);
			if (key == 'c')
			{
				cout << num << "pixel:" << 100.0* num / src1.size().area() << "%" << endl;
			}
			if (key == 'f')
			{
				alpha = (alpha > 0) ? 0 : 100;
				setTrackbarPos("alpha", wname, alpha);
			}
		}
		destroyWindow(wname);
	}

	void guiAbsDiffCompareLE(const Mat& src1, const Mat& src2)
	{
		string wname = "Diff Threshold LE";
		namedWindow(wname);
		int th = 1;
		int alpha = 100; createTrackbar("alpha", wname, &alpha, 100);
		createTrackbar("thresh", wname, &th, 255);
		int key = 0;
		Mat dest;
		Mat s1, s2;
		if (src1.channels() == 3)
			cvtColor(src1, s1, COLOR_BGR2GRAY);
		else s1 = src1;
		if (src2.channels() == 3)
			cvtColor(src2, s2, COLOR_BGR2GRAY);
		else s2 = src2;

		while (key != 'q')
		{
			Mat diff = abs(s1 - s2);
			compare(diff, th, dest, CMP_LE);
			int num = countNonZero(dest);
			if (src1.depth() != dest.depth())
			{
				Mat s;
				src1.convertTo(s, CV_8U);
				alphaBlend(dest, s, alpha / 100.0, dest);
			}
			else
				alphaBlend(dest, src1, alpha / 100.0, dest);

			imshow(wname, dest);
			key = waitKey(1);
			if (key == 'c')
			{
				cout << num << "pixel:" << 100.0* num / src1.size().area() << "%" << endl;
			}
			if (key == 'f')
			{
				alpha = (alpha > 0) ? 0 : 100;
				setTrackbarPos("alpha", wname, alpha);
			}
		}
		destroyWindow(wname);
	}

	void guiAbsDiffCompareEQ(const Mat& src1, const Mat& src2)
	{
		string wname = "Diff Threshold EQ";
		namedWindow(wname);
		int th = 1;
		int alpha = 100; createTrackbar("alpha", wname, &alpha, 100);
		createTrackbar("thresh", wname, &th, 255);
		int key = 0;
		Mat dest;
		Mat s1, s2;
		if (src1.channels() == 3)
			cvtColor(src1, s1, COLOR_BGR2GRAY);
		else s1 = src1;
		if (src2.channels() == 3)
			cvtColor(src2, s2, COLOR_BGR2GRAY);
		else s2 = src2;

		while (key != 'q')
		{
			Mat diff = abs(s1 - s2);
			compare(diff, th, dest, CMP_EQ);
			int num = countNonZero(dest);
			if (src1.depth() != dest.depth())
			{
				Mat s;
				src1.convertTo(s, CV_8U);
				alphaBlend(dest, s, alpha / 100.0, dest);
			}
			else
				alphaBlend(dest, src1, alpha / 100.0, dest);

			imshow(wname, dest);
			key = waitKey(1);
			if (key == 'c')
			{
				cout << num << "pixel:" << 100.0* num / src1.size().area() << "%" << endl;
			}
			if (key == 'f')
			{
				alpha = (alpha > 0) ? 0 : 100;
				setTrackbarPos("alpha", wname, alpha);
			}
		}
		destroyWindow(wname);
	}

	void guiAbsDiffCompareNE(const Mat& src1, const Mat& src2)
	{
		string wname = "Diff Threshold NE";
		namedWindow(wname);
		int th = 1;
		int alpha = 100; createTrackbar("alpha", wname, &alpha, 100);
		createTrackbar("thresh", wname, &th, 255);
		int key = 0;
		Mat dest;
		Mat s1, s2;
		if (src1.channels() == 3)
			cvtColor(src1, s1, COLOR_BGR2GRAY);
		else s1 = src1;
		if (src2.channels() == 3)
			cvtColor(src2, s2, COLOR_BGR2GRAY);
		else s2 = src2;

		while (key != 'q')
		{
			Mat diff = abs(s1 - s2);
			compare(diff, th, dest, CMP_NE);

			int num = countNonZero(dest);
			if (src1.depth() != dest.depth())
			{
				Mat s;
				src1.convertTo(s, CV_8U);
				alphaBlend(dest, s, alpha / 100.0, dest);
			}
			else
				alphaBlend(dest, src1, alpha / 100.0, dest);

			imshow(wname, dest);
			key = waitKey(1);
			if (key == 'c')
			{
				cout << num << "pixel:" << 100.0* num / src1.size().area() << "%" << endl;
			}
			if (key == 'f')
			{
				alpha = (alpha > 0) ? 0 : 100;
				setTrackbarPos("alpha", wname, alpha);
			}
		}
		destroyWindow(wname);
	}
}