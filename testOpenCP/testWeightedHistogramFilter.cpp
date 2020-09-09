#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testWeightedHistogramFilterDisparity()
{
	Mat guide = imread("img/stereo/Dolls/view1.png");
	Mat src = imread("img/stereo/Dolls/disp1_est.png", 0);
	Mat dmap_ = imread("img/stereo/Dolls/disp1.png", 0);
	const int disp_max = get_simd_ceil(100, 16);
	cp::StereoEval eval(dmap_, 2, disp_max);

	Mat dest;

	string wname = "weighted histogram filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 0; createTrackbar("switch", wname, &sw, 5);

	int r = 4; createTrackbar("r", wname, &r, 20);
	int space = 100; createTrackbar("space*0.1", wname, &space, 500);
	int color = 300; createTrackbar("color*0.1", wname, &color, 5000);
	int histogram = 10; createTrackbar("histogram*0.1", wname, &histogram, 255);
	int histogramFunctionType = (int)WHF_HISTOGRAM_WEIGHT::GAUSSIAN; createTrackbar("functionType", wname, &histogramFunctionType, (int)WHF_HISTOGRAM_WEIGHT::SIZE-1);
	

	int key = 0;
	Mat show;
	Mat ref;
	Mat srcf; guide.convertTo(srcf, CV_32F);
	Mat weight = Mat::ones(guide.size(), CV_8U);

	cp::UpdateCheck uc(sw);
	double time;
	while (key != 'q')
	{
		if (uc.isUpdate(sw))
		{
			switch (sw)
			{
			case 0:
				displayOverlay(wname, "weighted mode", 5000); break;
			case 1:
				displayOverlay(wname, "weighted median", 5000); break;
			}
		}
		{
			Timer t("", TIME_MSEC, false);
			if (sw == 0)
			{
				weightedHistogramFilter(src, guide, show, r, color * 0.1, space * 0.1, histogram * 0.1, WHF_HISTOGRAM_WEIGHT(histogramFunctionType), WHF_OPERATION::BILATERAL_MODE);
			}
			else if (sw == 1)
			{
				weightedHistogramFilter(src, guide, show, r, color * 0.1, space * 0.1, histogram * 0.1, WHF_HISTOGRAM_WEIGHT(histogramFunctionType), WHF_OPERATION::BILATERAL_MEDIAN);
				//weightedweightedMedianFilter(src, guide, weight, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
				//weightedMedianFilter(src, guide, show, r, truncate, space / 10.0, color / 10.0, metric, 2);
			}
			time = t.getTime();
		}
		imshow(wname, show);
		ci("Time %5.2f ms", time);
		ci("T 0.5|" + eval(show, 0.5, 2, false));
		ci("T 1.0|" + eval(show, 1.0, 2, false));
		ci("T 2.0|" + eval(show, 2.0, 2, false));
		ci.show();
		key = waitKey(1);
	}
	destroyAllWindows();
}

void testWeightedHistogramFilter(Mat& src_, Mat& guide_)
{
	//const bool isOverwrite = true;
	const bool isOverwrite = src_.empty() || guide_.empty();
	if (isOverwrite)
	{
		string imgPath_src, imgPath_guide;

		//imgPath_src = "img/lenna.png";
		//imgPath_guide = imgPath_src;
		// Flash/no-flash denoising
	/*{
		imgPath_p = "fig/pot2_noflash.png";
		imgPath_I = "fig/pot2_flash.png";
	}*/

		src_ = imread(imgPath_src, 0);
		guide_ = imread(imgPath_guide, 0);

		copyMakeBorder(guide_, guide_, 0, 44, 0, 32, BORDER_REFLECT);

		resize(src_, src_, src_.size() / 2);
		resize(guide_, guide_, guide_.size() / 2);
		guiAlphaBlend(src_, guide_, true);
	}


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

	int sp = 0; createTrackbar("sp noise", wname, &sp, 100);

	int key = 0;
	Mat show;

	RealtimeO1BilateralFilter rbf;
	Mat ref;
	Mat srcf; src.convertTo(srcf, CV_32F);
	Mat weight = Mat::ones(src.size(), CV_8U);

	cp::UpdateCheck uc(sw);
	while (key != 'q')
	{
		Mat img;
		cp::addNoise(src, img, 0, sp * 0.01);
		if (uc.isUpdate(sw))
		{
			switch (sw)
			{
			case 0:
				displayOverlay(wname, "weighted mode", 5000); break;
			case 1:
				displayOverlay(wname, "weighted median", 5000); break;
			}

		}
		if (sw == 0)
		{
			Timer t;
			//weightedModeFilter(img, guide, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
			//weightedweightedModeFilter(src, guide, weight, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
		}
		else if (sw == 1)
		{
			Timer t;
			//weightedModeFilter(src, guide, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
			//weightedweightedMedianFilter(src, guide, weight, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
			//weightedMedianFilter(img, guide, show, r, tranc, space / 10.0, color / 10.0, 2, 2);
		}
		imshow(wname, show);

		key = waitKey(1);
	}
	destroyAllWindows();
}
