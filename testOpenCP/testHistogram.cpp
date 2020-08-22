#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testHistogram2(Mat& src)
{
	Mat gray, hist;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	drawHistogramImageGray(gray, hist, COLOR_GRAY200, COLOR_RED);
	imshow("hist", hist);
	waitKey();

	drawAccumulateHistogramImageGray(gray, hist, COLOR_GRAY150, COLOR_RED);
	imshow("hist", hist);
	waitKey();

	drawHistogramImage(src, hist, COLOR_ORANGE);
	imshow("hist", hist);
	waitKey();

	drawAccumulateHistogramImage(src, hist, COLOR_ORANGE);
	imshow("hist", hist);
	waitKey();
}

void testHistogram()
{
	string wname = "histogram test";
	namedWindow(wname);
	int isShowGrid = 1; createTrackbar("isShowGrid", wname, &isShowGrid, 1);
	int isShowStat = 1; createTrackbar("isShowStat", wname, &isShowStat, 1);
	int isAccumulate = 0; createTrackbar("isAccmulate", wname, &isAccumulate, 1);
	int normalize_value = 0; createTrackbar("normalize_value", wname, &normalize_value, 100);
	int isUniformGauss = 0; createTrackbar("isUniformGauss", wname, &isUniformGauss, 1);
	int sigma = 5; createTrackbar("sigma:Gauss", wname, &sigma, 256);
	Mat img(Size(512, 512), CV_8U);
	Mat hist = Mat::zeros(256, 256, CV_8U);
	int key = 0;
	while (key != 'q')
	{
		if (isUniformGauss == 0)
		{
			randu(img, 0, 256);
		}
		else
		{
			randn(img, 128, sigma);
		}
		cout << cp::getEntropy(img) << endl;
		hist.setTo(0);
		if (isAccumulate)
		{
			drawAccumulateHistogramImage(img, hist, COLOR_BLACK, isShowGrid == 1, isShowStat == 1);
		}
		else
		{
			drawHistogramImage(img, hist, COLOR_BLACK, isShowGrid == 1, isShowStat == 1, normalize_value*0.01*img.size().area());
		}
		//imshow("img", img);
		imshow(wname, hist);

		key = waitKey(1);
	}
}
