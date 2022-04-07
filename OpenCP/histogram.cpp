#include "histogram.hpp"
#include "draw.hpp"
#include "crop.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void drawHistogramImageGray(cv::InputArray src, cv::OutputArray histogramImage, cv::Scalar color, cv::Scalar mean_color, const bool isDrawGrid, const bool isDrawStats, const int normalize_value)
	{
		const int hist_height = 200;
		cv::Scalar back_ground_color = Scalar::all(254);

		Scalar mean, stddev;
		meanStdDev(src, mean, stddev);
		int ave = cvRound(mean.val[0]);

		int image_num = 1;      // number of input image
		int channels[] = { 0 }; // index of channels 
		cv::MatND histogram;    // output histogram
		int dim_num = 1;        // output histogram dimension
		int bin_num = 256;      // number of bin
		int bin_nums[] = { bin_num };
		float range[] = { 0, 256 * 1 };// range
		const float *ranges[] = { range }; // 
		Mat src_ = src.getMat();
		if (src_.depth() == CV_8U || src_.depth() == CV_16U || src_.depth() == CV_32F)
		{
			cv::calcHist(&src_, image_num, channels, cv::Mat(), histogram, dim_num, bin_nums, ranges);
		}
		else
		{
			Mat temp;
			src_.convertTo(temp, CV_32F);
			cv::calcHist(&temp, image_num, channels, cv::Mat(), histogram, dim_num, bin_nums, ranges);
		}

		double minVal;
		double maxVal;
		minMaxLoc(histogram, 0, &maxVal, 0, 0);

		if (normalize_value != 0)maxVal = (double)normalize_value;

		int shift = 1;
		Mat hist_img = Mat::zeros(Size(bin_num + 2 * shift, hist_height), CV_8UC3);
		hist_img.setTo(back_ground_color);
		for (int i = 0; i < bin_num; i++)
		{
			int n = i + 1;
			line(hist_img, Point(n, 0), Point(n, cvRound(hist_height*histogram.at<float>(i) / maxVal)), color);
		}

		if (isDrawGrid)
		{
			drawGrid(hist_img, Point(hist_img.cols / 2, hist_height / 2), Scalar::all(200), 2);
			drawGrid(hist_img, Point(hist_img.cols / 4, hist_height / 2), Scalar::all(200), 1);
			drawGrid(hist_img, Point(hist_img.cols * 3 / 4, hist_height / 2), Scalar::all(200), 1);
			drawGrid(hist_img, Point(hist_img.cols / 2, hist_height * 1 / 4), Scalar::all(200), 1);
			drawGrid(hist_img, Point(hist_img.cols / 2, hist_height * 3 / 4), Scalar::all(200), 1);
		}
		line(hist_img, Point(ave + shift, 0), Point(ave + shift, cvRound(hist_height*histogram.at<float>(ave) / maxVal)), mean_color);
		flip(hist_img, hist_img, 0);

		if (isDrawStats)
		{
			Mat text_img = Mat::zeros(Size(bin_num + 2 * shift, 30), CV_8UC3);

			string text = format("avg%05.1f", mean.val[0]);
			line(text_img, Point(0, 0), Point(text_img.cols - 1, 0), back_ground_color);
			putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
			text = format("sdv%05.1f", stddev.val[0]);
			putText(text_img, text, Point(10 + bin_num / 2, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
			vconcat(hist_img, text_img, histogramImage);

			minMaxLoc(src, &minVal, &maxVal, 0, 0);
			text_img.setTo(0);
			text = format("min%05.1f", minVal);
			putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
			text = format("max%05.1f", maxVal);
			putText(text_img, text, Point(10 + bin_num / 2, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);

			vconcat(histogramImage, text_img, histogramImage);
		}
		else
		{
			hist_img.copyTo(histogramImage);
		}
	}

	void drawHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, const bool isDrawGrid, const bool isDrawStats, const int normalize_value)
	{
		if (src.channels() == 1)
		{
			drawHistogramImageGray(src, histogram, COLOR_BLACK, meancolor, isDrawGrid, isDrawStats, normalize_value);
			return;
		}

		vector<Mat> v; split(src, v);
		vector<Mat> hist(3);
		drawHistogramImageGray(v[0], hist[0], COLOR_BLUE, meancolor, isDrawGrid, isDrawStats, normalize_value);
		drawHistogramImageGray(v[1], hist[1], COLOR_GREEN, meancolor, isDrawGrid, isDrawStats, normalize_value);
		drawHistogramImageGray(v[2], hist[2], COLOR_RED, meancolor, isDrawGrid, isDrawStats, normalize_value);
		hconcat(hist, histogram);
	}

	void drawAccumulateHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar mean_color, const bool isDrawGrid, const bool isDrawStats)
	{
		const int hist_image_height = 200;
		cv::Scalar back_ground_color = Scalar::all(254);

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

		double minVal = 0.0;
		double maxVal = 0.0;
		for (int i = 0; i < bin_num; i++)
		{
			maxVal += (double)hist.at<float>(i);
		}

		int shift = 1;
		Mat hist_img = Mat::zeros(Size(bin_num + 2 * shift, hist_image_height), CV_8UC3);
		hist_img.setTo(back_ground_color);
		double value = 0.0;
		for (int i = 0; i < bin_num; i++)
		{
			int n = i + 1;
			value += hist.at<float>(i);
			if (i == ave)
				line(hist_img, Point(n, 0), Point(n, cvRound(hist_image_height*value / maxVal)), mean_color);
			else
				line(hist_img, Point(n, 0), Point(n, cvRound(hist_image_height*value / maxVal)), color);
		}

		if (isDrawGrid)
		{
			drawGrid(hist_img, Point(hist_img.cols / 2, hist_image_height / 2), Scalar::all(200), 2);
			drawGrid(hist_img, Point(hist_img.cols / 4, hist_image_height / 2), Scalar::all(200), 1);
			drawGrid(hist_img, Point(hist_img.cols * 3 / 4, hist_image_height / 2), Scalar::all(200), 1);
			drawGrid(hist_img, Point(hist_img.cols / 2, hist_image_height * 1 / 4), Scalar::all(200), 1);
			drawGrid(hist_img, Point(hist_img.cols / 2, hist_image_height * 3 / 4), Scalar::all(200), 1);
		}
		flip(hist_img, hist_img, 0);

		if (isDrawStats)
		{
			Mat text_img = Mat::zeros(Size(bin_num + 2 * shift, 30), CV_8UC3);

			string text = format("avg%05.1f", mean.val[0]);
			line(text_img, Point(0, 0), Point(text_img.cols - 1, 0), back_ground_color);
			putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
			text = format("sdv%05.1f", stddev.val[0]);
			putText(text_img, text, Point(10 + bin_num / 2, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
			vconcat(hist_img, text_img, histogram);

			minMaxLoc(src, &minVal, &maxVal, 0, 0);
			text_img.setTo(0);
			text = format("min%05.1f", minVal);
			putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
			text = format("max%05.1f", maxVal);
			putText(text_img, text, Point(10 + bin_num / 2, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);

			vconcat(histogram, text_img, histogram);
		}
		else
		{
			hist_img.copyTo(histogram);
		}
	}

	void drawAccumulateHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, const bool isDrawGrid, const bool isDrawStats)
	{
		if (src.channels() == 1)
		{
			drawAccumulateHistogramImageGray(src, histogram, COLOR_BLACK, meancolor, isDrawGrid, isDrawStats);
			return;
		}

		vector<Mat> v; split(src, v);
		vector<Mat> hist(3);
		drawAccumulateHistogramImageGray(v[0], hist[0], COLOR_BLUE, meancolor, isDrawGrid, isDrawStats);
		drawAccumulateHistogramImageGray(v[1], hist[1], COLOR_GREEN, meancolor, isDrawGrid, isDrawStats);
		drawAccumulateHistogramImageGray(v[2], hist[2], COLOR_RED, meancolor, isDrawGrid, isDrawStats);

		hconcat(hist, histogram);
	}

	struct MouseLocalDiffHistogramParameter
	{
		cv::Rect pt;
		std::string wname;
		MouseLocalDiffHistogramParameter(int x, int y, int width, int height, std::string name)
		{
			pt = cv::Rect(x, y, width, height);
			wname = name;
		}
	};

	static void guiLocalDiffHistogramOnMouse(int event, int x, int y, int flags, void* param)
	{
		MouseLocalDiffHistogramParameter* retp = (MouseLocalDiffHistogramParameter*)param;

		if (flags == EVENT_FLAG_LBUTTON)
		{
			retp->pt.x = max(0, min(retp->pt.width - 1, x));
			retp->pt.y = max(0, min(retp->pt.height - 1, y));

			setTrackbarPos("x", retp->wname, x);
			setTrackbarPos("y", retp->wname, y);
		}
	}

	void guiLocalDiffHistogram(Mat& src, bool isWait, string wname)
	{
		Mat img;
		if (src.channels() == 3)
		{
			cvtColor(src, img, COLOR_BGR2GRAY);
		}
		else
		{
			img = src;
		}


		namedWindow(wname);

		static MouseLocalDiffHistogramParameter param(src.cols / 2, src.rows / 2, src.cols, src.rows, wname);

		setMouseCallback(wname, (MouseCallback)guiLocalDiffHistogramOnMouse, (void*)&param);
		createTrackbar("x", wname, &param.pt.x, src.cols - 1);
		createTrackbar("y", wname, &param.pt.y, src.rows - 1);
		int r = 0;  createTrackbar("r", wname, &r, 50);
		int histmax = 100; createTrackbar("histmax", wname, &histmax, 255);
		int key = 0;
		Mat show;

		int hist[512];

		Plot pt;
		Mat hist_img(Size(512, 512), CV_8UC3);
		while (key != 'q')
		{

			hist_img.setTo(0);
			img.copyTo(show);
			Mat crop;
			Point pt = Point(param.pt.x, param.pt.y);

			const int d = 2 * r + 1;
			cropZoom(img, crop, pt, d);
			rectangle(show, Rect(pt.x - r, pt.y - r, d, d), COLOR_WHITE);
			for (int i = 0; i < 512; i++)hist[i] = 0;

			int b = crop.at<uchar>(r, r);

			int minv = 255;
			int maxv = -255;
			for (int j = 0; j < d; j++)
			{
				for (int i = 0; i < d; i++)
				{
					int val = (int)crop.at<uchar>(j, i) - b + 255;
					hist[val]++;

					maxv = max(maxv, val - 255);
					minv = min(minv, val - 255);
				}
			}

			line(hist_img, Point(255, hist_img.rows - 1), Point(255, 0), COLOR_GRAY50);
			line(hist_img, Point(minv + 255, hist_img.rows - 1), Point(minv + 255, 0), COLOR_GRAY40);
			line(hist_img, Point(maxv + 255, hist_img.rows - 1), Point(maxv + 255, 0), COLOR_GRAY40);
			for (int i = 0; i < 512; i++)
			{
				int v = int((hist_img.rows - 1) * (histmax - min(histmax, hist[i])) / (double)histmax);
				line(hist_img, Point(i, hist_img.rows - 1), Point(i, v), COLOR_WHITE);
			}

			displayOverlay(wname, format("%d %d", minv, maxv));
			imshow("hist", hist_img);
			imshow(wname, show);
			key = waitKey(1);

			if (!isWait)break;
		}

		if (!isWait)destroyWindow(wname);
	}
}