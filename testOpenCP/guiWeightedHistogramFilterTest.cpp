#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void guiWeightedHistogramFilterTest(Mat& src_, Mat& guide_)
{

	Mat src, guide;
	if (src_.channels() == 3)
	{
		//cvtColor(src_, guide, COLOR_BGR2GRAY);
		//cvtColor(src_, src, COLOR_BGR2GRAY);
		//src_.copyTo(src);
		guide = guide_;
		src = src_;
	}
	else
	{
		guide = guide_;
		src = src_;
	}

	Mat dest;

	string wname = "weighted histogram filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 0; createTrackbar("switch", wname, &sw, 5);

	//int r = 10; createTrackbar("r",wname,&r,200);
	int space = 100; createTrackbar("space", wname, &space, 500);
	int color = 300; createTrackbar("color", wname, &color, 5000);
	int tranc = 10; createTrackbar("tranc", wname, &color, 255);
	int r = 1; createTrackbar("r", wname, &r, 20);

	int key = 0;
	Mat show;

	RealtimeO1BilateralFilter rbf;
	Mat ref;
	Mat srcf; src.convertTo(srcf, CV_32F);
	Mat weight = Mat::ones(src.size(), CV_8U);
	while (key != 'q')
	{
		if (sw==0)
		{
			CalcTime t;
			//weightedModeFilter(src, guide, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
			weightedweightedModeFilter(src, guide, weight, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
		}
		else if (sw==1)
		{
			CalcTime t;
			//weightedModeFilter(src, guide, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
			weightedweightedMedianFilter(src, guide, weight, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
		}
		imshow(wname, show);

		key = waitKey(1);
	}
	destroyAllWindows();
}
