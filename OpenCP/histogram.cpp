#include "histogram.hpp"
#include "draw.hpp"

using namespace std;
using namespace cv;

namespace cp
{

void drawHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar color2, bool isGrid)
{
	Scalar mean, stddev;
	meanStdDev(src, mean, stddev);
	int ave = cvRound(mean.val[0]);

	int image_num = 1;      // number of input image
	int channels[] = { 0 }; // index of channels 
	cv::MatND hist;         // output histogram
	int dim_num = 1;        // output histogram dimension
	int bin_num = 256;       // number of bin
	int bin_nums[] = { bin_num };
	float range[] = { 0, 256*20 };        // range
	const float *ranges[] = { range }; // 
	Mat src_ = src.getMat();
	cv::calcHist(&src_, image_num, channels, cv::Mat(), hist, dim_num, bin_nums, ranges);

	double maxVal = 0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	cout << maxVal << endl;
	const int hist_height = 200;
	int shift = 1;
	Mat hist_img = Mat::zeros(Size(bin_num + 2 * shift, hist_height), CV_8UC3);
	hist_img.setTo(Scalar::all(254));
	for (int i = 0; i < bin_num; i++)
	{
		int n = i + 1;
		line(hist_img, Point(n, 0), Point(n, cvRound(hist_height*hist.at<float>(i) / maxVal)), color);
	}
	if (isGrid)
	{
		drawGrid(hist_img, Point(hist_img.cols / 2, hist_height / 2), Scalar::all(200), 2);
		drawGrid(hist_img, Point(hist_img.cols / 4, hist_height / 2), Scalar::all(200), 1);
		drawGrid(hist_img, Point(hist_img.cols * 3 / 4, hist_height / 2), Scalar::all(200), 1);
		drawGrid(hist_img, Point(hist_img.cols / 2, hist_height * 1 / 4), Scalar::all(200), 1);
		drawGrid(hist_img, Point(hist_img.cols / 2, hist_height * 3 / 4), Scalar::all(200), 1);
	}
	line(hist_img, Point(ave + shift, 0), Point(ave + shift, cvRound(hist_height*hist.at<float>(ave) / maxVal)), color2);
	flip(hist_img, hist_img, 0);

	Mat text_img = Mat::zeros(Size(bin_num + 2 * shift, 30), CV_8UC3);

	string text = format("avg%05.1f", mean.val[0]);
	putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
	text = format("sdv%05.1f", stddev.val[0]);
	putText(text_img, text, Point(10 + bin_num / 2, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
	vconcat(hist_img, text_img, histogram);
}

void drawAccumulateHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar color2, bool isGrid)
{
	Scalar mean, stddev;
	meanStdDev(src, mean, stddev);
	int ave = cvRound(mean.val[0]);

	int image_num = 1;      // number of input image
	int channels[] = { 0 }; // index of channels 
	cv::MatND hist;         // output histogram
	int dim_num = 1;        // output histogram dimension
	int bin_num = 256;       // number of bin
	int bin_nums[] = { bin_num };
	float range[] = { 0, 256 };        // range
	const float *ranges[] = { range }; // 
	Mat src_ = src.getMat();
	cv::calcHist(&src_, image_num, channels, cv::Mat(), hist, dim_num, bin_nums, ranges);

	double maxVal = 0.0;
	for (int i = 0; i < bin_num; i++)
		maxVal += (double)hist.at<float>(i);

	const int hist_height = 200;
	int shift = 1;
	Mat hist_img = Mat::zeros(Size(bin_num + 2 * shift, hist_height), CV_8UC3);
	hist_img.setTo(Scalar::all(254));
	double value = 0.0;
	for (int i = 0; i < bin_num; i++)
	{
		int n = i + 1;
		value += hist.at<float>(i);
		if (i == ave)
			line(hist_img, Point(n, 0), Point(n, cvRound(hist_height*value / maxVal)), color2);
		else
			line(hist_img, Point(n, 0), Point(n, cvRound(hist_height*value / maxVal)), color);
	}
	if (isGrid)
	{
		drawGrid(hist_img, Point(hist_img.cols / 2, hist_height / 2), Scalar::all(200), 2);
		drawGrid(hist_img, Point(hist_img.cols / 4, hist_height / 2), Scalar::all(200), 1);
		drawGrid(hist_img, Point(hist_img.cols * 3 / 4, hist_height / 2), Scalar::all(200), 1);
		drawGrid(hist_img, Point(hist_img.cols / 2, hist_height * 1 / 4), Scalar::all(200), 1);
		drawGrid(hist_img, Point(hist_img.cols / 2, hist_height * 3 / 4), Scalar::all(200), 1);
	}
	flip(hist_img, hist_img, 0);

	Mat text_img = Mat::zeros(Size(bin_num + 2 * shift, 30), CV_8UC3);

	string text = format("avg%05.1f", mean.val[0]);
	putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
	text = format("sdv%05.1f", stddev.val[0]);
	putText(text_img, text, Point(10 + bin_num / 2, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
	vconcat(hist_img, text_img, histogram);
}

void drawAccumulateHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, bool isGrid)
{
	vector<Mat> v; split(src, v);
	vector<Mat> hist(3);
	drawAccumulateHistogramImageGray(v[0], hist[0], COLOR_BLUE, meancolor, isGrid);
	drawAccumulateHistogramImageGray(v[1], hist[1], COLOR_GREEN, meancolor, isGrid);
	drawAccumulateHistogramImageGray(v[2], hist[2], COLOR_RED, meancolor, isGrid);

	hconcat(hist, histogram);
}

void drawHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, bool isGrid)
{
	if (src.channels() == 1)
	{
		drawHistogramImageGray(src, histogram, meancolor, isGrid);
		return;
	}
	vector<Mat> v; split(src, v);
	vector<Mat> hist(3);
	drawHistogramImageGray(v[0], hist[0], COLOR_BLUE, meancolor, isGrid);
	drawHistogramImageGray(v[1], hist[1], COLOR_GREEN, meancolor, isGrid);
	drawHistogramImageGray(v[2], hist[2], COLOR_RED, meancolor, isGrid);
	hconcat(hist, histogram);
}
}