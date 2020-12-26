#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testWeightedHistogramFilterDisparity()
{
	Mat guideColor = imread("img/stereo/Dolls/view1.png");
	Mat guideGray; cvtColor(guideColor, guideGray, cv::COLOR_BGR2GRAY);
	Mat src = imread("img/stereo/Dolls/disp1_est.png", 0);
	Mat dmap_ = imread("img/stereo/Dolls/disp1.png", 0);
	const int disp_max = get_simd_ceil(100, 16);
	cp::StereoEval eval(dmap_, 2, disp_max);

	Mat dest;

	string wname = "weighted histogram filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a = 0; createTrackbar("a", wname, &a, 100);
	int whf_option = 2; createTrackbar("WHF_OPERATION", wname, &whf_option, (int)WHF_OPERATION::SIZE - 1);
	int sw = 0; createTrackbar("sw", wname, &sw, 1);

	int r = 4; createTrackbar("r", wname, &r, 20);
	int space = 100; createTrackbar("space*0.1", wname, &space, 500);
	int color = 300; createTrackbar("color*0.1", wname, &color, 5000);
	int histogram = 100; createTrackbar("histogram*0.1", wname, &histogram, 255);
	int histogramFunctionType = (int)WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN; createTrackbar("functionType", wname, &histogramFunctionType, (int)WHF_HISTOGRAM_WEIGHT_FUNCTION::SIZE - 1);
	int isColorGuide = 1; createTrackbar("isColorGuide", wname, &isColorGuide, 1);

	int refinementWeightR = 3; createTrackbar("weightR", wname, &refinementWeightR, 20);
	int refinementWeightSigma = 10; createTrackbar("WeightSigma", wname, &refinementWeightSigma, 255);
	Mat show;
	Mat ref;
	Mat weight = Mat::ones(src.size(), CV_8U);

	cp::UpdateCheck uc(whf_option);
	cp::UpdateCheck uc2(whf_option, sw, r, space, color, histogram, histogramFunctionType, isColorGuide);
	Timer t("", TIME_MSEC, false);

	int key = 0;
	while (key != 'q')
	{
		Mat guide = (isColorGuide == 1) ? guideColor : guideGray;
		if (sw == 0)
		{
			t.start();
			weightedHistogramFilter(src, guide, show, r, color * 0.1, space * 0.1, histogram * 0.1, WHF_HISTOGRAM_WEIGHT_FUNCTION(histogramFunctionType), WHF_OPERATION(whf_option));
			t.getpushLapTime();
		}
		else
		{
			t.start();
			Mat bim;
			Mat weight(src.size(), CV_32F);
			GaussianBlur(src, bim, Size(2 * refinementWeightR + 1, 2 * refinementWeightR + 1), refinementWeightR / 3.0);
			uchar* disp = src.ptr<uchar>();
			uchar* dispb = bim.ptr<uchar>();
			float* s = weight.ptr<float>();
			for (int i = 0; i < weight.size().area(); i++)
			{
				float diff = (disp[i] - dispb[i]) * (disp[i] - dispb[i]) / (-2.f * refinementWeightSigma * refinementWeightSigma);
				s[i] = exp(diff);
			}
			weightedWeightedHistogramFilter(src, weight, guide, show, r, color * 0.1, space * 0.1, histogram * 0.1, WHF_HISTOGRAM_WEIGHT_FUNCTION(histogramFunctionType), WHF_OPERATION(whf_option));
			t.getpushLapTime();
		}
		imshow(wname, show);
		ci("Num %d", t.getStatSize());
		ci("Time %5.2f ms", t.getLapTimeMedian());
		ci("T 0.5|" + eval(show, 0.5, 2, false));
		ci("T 1.0|" + eval(show, 1.0, 2, false));
		ci("T 2.0|" + eval(show, 2.0, 2, false));
		ci.show();
		key = waitKey(1);
		if (key == 'r' || uc2.isUpdate(whf_option, sw, r, space, color, histogram, histogramFunctionType, isColorGuide))
		{
			t.clearStat();
		}
		if (uc.isUpdate(whf_option))
		{
			displayOverlay(wname, cp::getWHFOperationName(WHF_OPERATION(whf_option)), 5000);
		}
	}

	destroyWindow(wname);
}

void testWeightedHistogramFilter(Mat& src_, Mat& guide_)
{
	//const bool isOverwrite = true;
	const bool isOverwrite = src_.empty() || guide_.empty();
	if (isOverwrite)
	{
		src_ = imread("img/flash/cave-flash.png", 1);
		guide_ = imread("img/flash/cave-noflash.png", 0);
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
	int whf_option = 2; createTrackbar("WHF_OPERATION", wname, &whf_option, (int)WHF_OPERATION::SIZE - 1);

	//int r = 10; createTrackbar("r",wname,&r,200);
	int space = 100; createTrackbar("space", wname, &space, 500);
	int color = 300; createTrackbar("color", wname, &color, 5000);
	int histogram = 100; createTrackbar("histogram*0.1", wname, &histogram, 255);
	int histogramFunctionType = (int)WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN; createTrackbar("functionType", wname, &histogramFunctionType, (int)WHF_HISTOGRAM_WEIGHT_FUNCTION::SIZE - 1);
	int isColorGuide = 1; createTrackbar("isColorGuide", wname, &isColorGuide, 1);
	int r = 1; createTrackbar("r", wname, &r, 20);

	int sp = 0; createTrackbar("sp noise", wname, &sp, 100);

	int key = 0;
	Mat show;

	Mat ref;
	Mat srcf; src.convertTo(srcf, CV_32F);
	Mat weight = Mat::ones(src.size(), CV_8U);

	cp::UpdateCheck uc(whf_option);
	while (key != 'q')
	{
		Mat img;
		cp::addNoise(src, img, 0, sp * 0.01);
	
		Timer t;
		weightedHistogramFilter(src, guide, show, r, color * 0.1, space * 0.1, histogram * 0.1, WHF_HISTOGRAM_WEIGHT_FUNCTION(histogramFunctionType), WHF_OPERATION(whf_option));

		imshow(wname, show);

		key = waitKey(1);
		if (uc.isUpdate(whf_option))
		{
			displayOverlay(wname, cp::getWHFOperationName(WHF_OPERATION(whf_option)), 5000);
		}
	}
	//destroyWindow(wname);
}
