#include "imagediff.hpp"
#include "blend.hpp"
using namespace cv;
using namespace std;

namespace cp
{

	void diffshow(string wname, InputArray src, InputArray ref, const double scale)
	{
		Mat show;
		Mat diff = Mat::zeros(src.size(), CV_MAKE_TYPE(CV_64F, src.channels()));
		subtract(src.getMat(), ref.getMat(), diff, noArray(), CV_64F);
		diff *= scale;
		diff += Scalar::all(128.0);
		diff.convertTo(show, CV_8U, 1.0, 0.5);
		imshow(wname, show);
	}

	void guiDiff(InputArray src, InputArray ref, const bool isWait, string wname)
	{
		namedWindow(wname);

		static int diff_a = 0; createTrackbar("a", wname, &diff_a, 100);
		static int diff_channel = 3; createTrackbar("channel", wname, &diff_channel, 4);
		static int diff_boost = 10; createTrackbar("boost*0.1", wname, &diff_boost, 1000);
		static int diff_abs_sw = 1; createTrackbar("diff_abs_sw", wname, &diff_abs_sw, 1);

		Mat sf, rf;
		src.getMat().convertTo(sf, CV_32F);
		ref.getMat().convertTo(rf, CV_32F);
		vector<Mat> vsf(3);
		vector<Mat> vrf(3);
		Mat grays;
		Mat grayr;

		if (src.channels() == 3)
		{
			cvtColor(sf, grays, COLOR_BGR2GRAY);
			cvtColor(rf, grayr, COLOR_BGR2GRAY);
			split(sf, vsf);
			split(rf, vrf);
		}
		else
		{
			sf.copyTo(grays);
			rf.copyTo(grayr);
			for (int i = 0; i < 3; i++)
			{
				sf.copyTo(vsf[0]);
				rf.copyTo(vrf[0]);
			}
		}

		static bool isGuiDiffInfo = true;
		int key = 0;
		Mat show;

		while (key != 'q')
		{
			Mat diff;
			string text = "";
			if (diff_abs_sw == 0)
			{
				if (diff_channel < 3)
				{
					if (diff_channel == 0)text = "B";
					if (diff_channel == 1)text = "G";
					if (diff_channel == 2)text = "R";
					diff = 0.1f*diff_boost * (vsf[diff_channel] - vrf[diff_channel]) + 128.f;
				}
				else if (diff_channel == 3)
				{
					text = "Y";
					diff = 0.1f*diff_boost * (grayr - grays) + 128.f;
				}
				else if (diff_channel == 4)
				{
					text = "All";
					subtract(sf, rf, diff);
					diff = 0.1f*diff_boost * diff + Scalar::all(128.f);
				}
			}
			else
			{
				if (diff_channel < 3)
				{
					if (diff_channel == 0)text = "B";
					if (diff_channel == 1)text = "G";
					if (diff_channel == 2)text = "R";
					diff = 0.1*diff_boost * abs(vsf[diff_channel] - vrf[diff_channel]);
				}
				else if (diff_channel == 3)
				{
					text = "Y";
					diff = 0.1*diff_boost * abs(grayr - grays);
				}
				else if (diff_channel == 4)
				{
					text = "All";
					subtract(sf, rf, diff);
					diff = 0.1*diff_boost * abs(diff);
				}
			}

			alphaBlend(sf, diff, 0.01*diff_a, diff);
			diff.convertTo(show, CV_8U);

			if (isGuiDiffInfo)putText(show, text, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2);
			imshow(wname, show);
			

			if (key == 'i')
			{
				isGuiDiffInfo = (isGuiDiffInfo) ? false : true;
			}
			if (key == 'd')
			{
				diff_abs_sw = (diff_abs_sw == 0) ? 1 : 0;
				setTrackbarPos("diff_abs_sw", wname, diff_abs_sw);
			}
			if (key == 'f')
			{
				diff_a = (diff_a != 0) ? 0 : 100;
				setTrackbarPos("a", wname, diff_a);
			}
			if (key == '?')
			{
				cout << "d: swich abs diff or not" << endl;
				cout << "i: show or not info" << endl;
				cout << "f: flip diff or src" << endl;
				cout << "q: quit" << endl;
			}

			if (!isWait)break;
			key = waitKey(1);
		}

		if (isWait) destroyWindow(wname);
	}

	void guiCompareDiff(cv::InputArray before, cv::InputArray after, cv::InputArray ref, std::string name_before, std::string name_after, std::string wname)
	{
		namedWindow(wname);
		static int compare_diff_sw = 0; createTrackbar("switch:compare", wname, &compare_diff_sw, 2);
		static int compare_diff_a = 0; createTrackbar("alpha:compare", wname, &compare_diff_a, 100);
		static int compare_diff_th = 0; createTrackbar("thresh:comapre", wname, &compare_diff_th, 255);

		int key = 0;
		Mat show;
		Mat g1, g2, g;
		if (before.channels() == 3)cvtColor(before, g1, COLOR_BGR2GRAY);
		else  g1 = before.getMat();
		if (after.channels() == 3)cvtColor(after, g2, COLOR_BGR2GRAY);
		else  g2 = after.getMat();
		if (ref.channels() == 3)cvtColor(ref, g, COLOR_BGR2GRAY);
		else  g = ref.getMat();

		Mat cmap(before.size(), CV_8UC3);

		Mat db = abs(g - g1);
		Mat da = abs(g - g2);
		Mat d = abs(g1 - g2);
		Mat maskB;
		Mat maskA;
		Mat zmask;

		int prev_sw = compare_diff_sw;
		while (key != 'q')
		{
			cmap.setTo(0);
			compare(d, compare_diff_th, zmask, cv::CMP_LE);

			compare(da, db, maskB, cv::CMP_GT);
			maskB.setTo(0, zmask);
			cmap.setTo(Scalar(0, 0, 255), maskB);

			compare(db, da, maskA, cv::CMP_GT);
			maskA.setTo(0, zmask);
			cmap.setTo(Scalar(0, 255, 0), maskA);

			if (compare_diff_sw == 0)
			{

				if (prev_sw != compare_diff_sw)displayOverlay(wname, "compare", 1000);
				prev_sw = compare_diff_sw;
				alphaBlend(ref, cmap, compare_diff_a / 100.0, show);
			}
			else if (compare_diff_sw == 1)
			{
				if (prev_sw != compare_diff_sw)displayOverlay(wname, name_before, 1000);
				prev_sw = compare_diff_sw;
				alphaBlend(ref, before, compare_diff_a / 100.0, show);
			}
			else if (compare_diff_sw == 2)
			{
				if (prev_sw != compare_diff_sw)displayOverlay(wname, name_after, 1000);
				prev_sw = compare_diff_sw;
				alphaBlend(ref, after, compare_diff_a / 100.0, show);
			}

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
				compare_diff_a = (compare_diff_a > 0) ? 0 : 100;
				setTrackbarPos("alpha", wname, compare_diff_a);
			}
			if (key == 'h' || key == '?')
			{
				cout << "red: before is better (degraded)" << endl;
				cout << "green: after is better (improved)" << endl;
				cout << "c: compute comparing image stats" << endl;
				cout << "f: flip alpha blend" << endl;
			}
			string mes = "red: " + name_before + " is better" + " green: " + name_after + " is better";
			displayStatusBar(wname, mes);
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