#include "imshowExtension.hpp"
#include "histogram.hpp"
#include "draw.hpp"
#include "concat.hpp"
#include "consoleimage.hpp"
#include "maskoperation.hpp"

#include "debugcp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void imshowSplitScale(string wname, InputArray src, const double alpha, const double beta)
	{
		namedWindow(wname);
		static int imshowSplitScaleColor = 0;
		createTrackbar("channel", wname, &imshowSplitScaleColor, 1);
		setTrackbarMin("channel", wname, -1);
		setTrackbarMax("channel", wname, src.channels() - 1);
		vector<Mat> v;
		split(src, v);
		if (imshowSplitScaleColor < 0)
		{
			Mat dest;
			cp::concat(v, dest, (int)v.size());
			imshowScale(wname, dest, alpha, beta);
		}
		else
		{
			imshowScale(wname, v[imshowSplitScaleColor], alpha, beta);
		}
	}

	void imshowSplit(string wname, InputArray src)
	{
		namedWindow(wname);
		static int imshowSplitColor = 0;
		createTrackbar("channel", wname, &imshowSplitColor, 1);
		setTrackbarMin("channel", wname, -1);
		setTrackbarMax("channel", wname, src.channels() - 1);
		vector<Mat> v;
		split(src, v);
		if (imshowSplitColor < 0)
		{
			Mat dest;
			cp::concat(v, dest, (int)v.size());
			imshow(wname, dest);
		}
		else
		{
			imshow(wname, v[imshowSplitColor]);
		}
	}

	void imshowNormalize(string wname, InputArray src, const int norm_type)
	{
		Mat show;
		normalize(src, show, 255, 0, norm_type, CV_8U);
		imshow(wname, show);
	}

	void imshowScale(string name, InputArray src, const double alpha, const double beta)
	{
		Mat show;
		src.getMat().convertTo(show, CV_8U, alpha, beta);
		imshow(name, show);
	}

	void imshowScaleAbs(string name, InputArray src, const double alpha, const double beta)
	{
		Mat show;
		cv::convertScaleAbs(src.getMat(), show, alpha, beta);
		imshow(name, show);
	}

	void imshowResize(std::string name, cv::InputArray src, const cv::Size dsize, const double fx, const double fy, const int interpolation, const bool isCast8U)
	{
		Mat show;
		if (src.depth() != CV_8U && isCast8U)
		{
			Mat temp;
			src.getMat().convertTo(temp, CV_8U);
			resize(temp, show, dsize, fx, fy, interpolation);
		}
		else
		{
			resize(src, show, dsize, fx, fy, interpolation);
		}
		imshow(name, show);
	}

	void imshowCountDown(string wname, InputArray src, const int waitTime, Scalar color, const int pointSize, std::string fontName)
	{
		Mat s;
		src.copyTo(s);
		addText(s, "3", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);

		src.copyTo(s);
		addText(s, "2", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);

		src.copyTo(s);
		addText(s, "1", Point(s.cols / 2, s.rows / 2), fontName, pointSize, color);
		imshow(wname, s);
		waitKey(waitTime);
	}

#pragma region Analysis
	template <class srcType>
	void getImageLine_(Mat& src, int channel, vector<Point>& v, const int line)
	{
		const int ch = src.channels();

		srcType* s = src.ptr<srcType>(line);
		if (ch == 1)
		{
			for (int i = 0; i < src.cols; i++)
			{
				v[i] = Point(i, (int)(s[i]));
			}

		}
		else if (ch == 3)
		{
			if (channel < 0 || channel>2)
			{
				for (int i = 0; i < src.cols; i++)
				{
					v[i] = Point(i, (int)(0.299 * s[3 * i + 2] + 0.587 * s[3 * i + 1] + 0.114 * s[3 * i + 0]));
				}
			}
			else
			{
				for (int i = 0; i < src.cols; i++)
				{
					v[i] = Point(i, (int)(s[3 * i + channel]));
				}
			}
		}
	}

	static void getImageLine(Mat& src, vector<Point>& v, const int line, int channel)
	{
		if (v.size() == 0)
			v.resize(src.cols);

		if (src.type() == CV_8U || src.type() == CV_8UC3)getImageLine_<uchar>(src, channel, v, line);
		else if (src.type() == CV_8S || src.type() == CV_8SC3)getImageLine_<char>(src, channel, v, line);
		else if (src.type() == CV_16U || src.type() == CV_16UC3)getImageLine_<ushort>(src, channel, v, line);
		else if (src.type() == CV_16S || src.type() == CV_16SC3)getImageLine_<short>(src, channel, v, line);
		else if (src.type() == CV_32S || src.type() == CV_32SC3)getImageLine_<int>(src, channel, v, line);
		else if (src.type() == CV_32F || src.type() == CV_32FC3)getImageLine_<float>(src, channel, v, line);
		else if (src.type() == CV_64F || src.type() == CV_64FC3)getImageLine_<double>(src, channel, v, line);
		else
		{
			cout << "do not support this type(getImageLine)" << endl;
		}
	}

	static void getImageVLine(Mat& src, vector<Point>& v, const int line, int channel)
	{
		if (v.size() == 0)
			v.resize(src.rows);

		Mat srct = src.t();

		if (src.type() == CV_8U || src.type() == CV_8UC3)getImageLine_<uchar>(srct, channel, v, line);
		else if (src.type() == CV_8S || src.type() == CV_8SC3)getImageLine_<char>(srct, channel, v, line);
		else if (src.type() == CV_16U || src.type() == CV_16UC3)getImageLine_<ushort>(srct, channel, v, line);
		else if (src.type() == CV_16S || src.type() == CV_16SC3)getImageLine_<short>(srct, channel, v, line);
		else if (src.type() == CV_32S || src.type() == CV_32SC3)getImageLine_<int>(srct, channel, v, line);
		else if (src.type() == CV_32F || src.type() == CV_32FC3)getImageLine_<float>(srct, channel, v, line);
		else if (src.type() == CV_64F || src.type() == CV_64FC3)getImageLine_<double>(srct, channel, v, line);
	}

	void drawSignalX(InputArrayOfArrays src_, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size outputImageSize, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
	{
		vector<Mat> src;
		if (src_.isMatVector()) src_.getMatVector(src);
		else
		{
			src.resize(1);
			src[0] = src_.getMat();
		}
		Plot p(outputImageSize);
		//p.setKey(cp::Plot::NOKEY);
		p.setKey(cp::Plot::LEFT_TOP);

		p.setPlotProfile(false, false, false);
		p.setPlotSymbolALL(Plot::NOPOINT);
		p.setPlotLineTypeALL(linetype);

		p.setXYRange(shiftx - max(rangex, 1), shiftx + max(rangex, 1), shiftvalue - rangevalue, shiftvalue + rangevalue);
		vector<vector<Point>> v((int)src.size());

		for (int i = 0; i < (int)src.size(); i++)
		{
			getImageLine(src[i], v[i], line_height, color);
			p.push_back(v[i], i);
		}
		bool isWideKey = false;
		p.generateKeyImage((int)src.size(), isWideKey);
		p.plotData();
		p.render.copyTo(dest);
	}

	void drawSignalX(Mat& src1, Mat& src2, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
	{
		vector<Mat> s;
		s.push_back(src1);
		s.push_back(src2);
		drawSignalX(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
	}

	void drawSignalY(InputArrayOfArrays src_, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
	{
		vector<Mat> src;
		src_.getMatVector(src);
		Plot p(size);
		p.setKey(cp::Plot::NOKEY);
		p.setPlotProfile(false, false, false);
		p.setPlotSymbolALL(Plot::NOPOINT);
		p.setPlotLineTypeALL(linetype);
		p.setXYRange(shiftx - max(rangex, 1), shiftx + max(rangex, 1), shiftvalue - rangevalue, shiftvalue + rangevalue);
		vector<vector<Point>> v((int)src.size());
		for (int i = 0; i < (int)src.size(); i++)
		{
			getImageVLine(src[i], v[i], line_height, color);
			p.push_back(v[i], i);
			p.plotData();
		}
		p.render.copyTo(dest);
	}

	void drawSignalY(Mat& src1, Mat& src2, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
	{
		vector<Mat> s;
		s.push_back(src1);
		s.push_back(src2);
		drawSignalY(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
	}

	void guiAnalysisMouse2(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;
		if (flags == EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

#if 0
	void imshowAnalysis(std::string winname, const Mat& src)
	{
		static bool isFirst = true;
		Mat im;
		if (src.channels() == 1)cvtColor(src, im, COLOR_GRAY2BGR);
		else src.copyTo(im);

		namedWindow(winname);
		if (isFirst)moveWindow(winname.c_str(), src.cols * 2, 0);

		static Point pt = Point(src.cols / 2, src.rows / 2);
		static int channel = 0;
		createTrackbar("channel", winname, &channel, 3);
		createTrackbar("x", winname, &pt.x, src.cols - 1);
		createTrackbar("y", winname, &pt.y, src.rows - 1);
		static int step = src.cols / 2;
		createTrackbar("clip x", winname, &step, src.cols / 2);
		static int ystep = src.rows / 2;
		createTrackbar("clip y", winname, &ystep, src.rows / 2);

		string winnameSigx = winname + " Xsignal";
		namedWindow(winnameSigx);

		if (isFirst)moveWindow(winnameSigx.c_str(), 512, src.rows * 2);

		static int shifty = 128;
		createTrackbar("shift y", winnameSigx, &shifty, 128);
		static int stepy = 128;
		createTrackbar("clip y", winnameSigx, &stepy, 255);

		string winnameSigy = winname + " Ysignal";
		namedWindow(winnameSigy);
		if (isFirst)moveWindow(winnameSigy.c_str(), 0, 0);
		static int yshifty = 128;
		createTrackbar("shift y", winnameSigy, &yshifty, 128);
		static int ystepy = 128;
		createTrackbar("clip y", winnameSigy, &ystepy, 255);

		string winnameHist = winname + " Histogram";
		namedWindow(winnameHist);

		Mat dest;
		drawSignalX(src, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.cols, 350), pt.y, pt.x, shifty, step, stepy, 1);
		imshow(winnameSigx, dest);
		drawSignalY(src, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.rows, 350), pt.x, pt.y, yshifty, ystep, ystepy);
		Mat temp;
		flip(dest.t(), temp, 0);
		imshow(winnameSigy, temp);

		rectangle(im, Point(pt.x - step, pt.y - ystep), Point(pt.x + step, pt.y + ystep), COLOR_GREEN);
		drawGrid(im, pt, COLOR_RED);
		imshow(winname, im);

		Mat hist;
		if (src.channels() == 1)
			drawHistogramImageGray(src, hist, COLOR_GRAY200, COLOR_ORANGE);
		else
			drawHistogramImage(src, hist, COLOR_ORANGE);

		imshow(winnameHist, hist);
		isFirst = false;
	}
#endif
	void imshowAnalysis(std::string winname, InputArrayOfArrays s_)
	{
		vector<Mat> s;
		s_.getMatVector(s);
		Mat src = s[0];
		static bool isFirst = true;
		vector<Mat> im(s.size());
		for (int i = 0; i < (int)s.size(); i++)
		{
			if (src.channels() == 1)cvtColor(s[i], im[i], COLOR_GRAY2BGR);
			else s[i].copyTo(im[i]);
		}

		namedWindow(winname);
		if (isFirst)moveWindow(winname.c_str(), src.cols * 2, 0);

		static Point pt = Point(src.cols / 2, src.rows / 2);

		static int amp = 1;
		createTrackbar("amp", winname, &amp, 255);
		static int nov = 0;
		createTrackbar("num of view", winname, &nov, (int)s.size() - 1);

		static int channel = 0;
		createTrackbar("channel", winname, &channel, 3);
		createTrackbar("x", winname, &pt.x, src.cols - 1);
		createTrackbar("y", winname, &pt.y, src.rows - 1);
		static int step = src.cols / 2;
		createTrackbar("clip x", winname, &step, src.cols / 2);
		static int ystep = src.rows / 2;
		createTrackbar("clip y", winname, &ystep, src.rows / 2);

		string winnameSigx = winname + " Xsignal";
		namedWindow(winnameSigx);

		if (isFirst)moveWindow(winnameSigx.c_str(), 512, src.rows);

		static int shifty = 128;
		createTrackbar("shift y", winnameSigx, &shifty, 128);
		static int stepy = 128;
		createTrackbar("clip y", winnameSigx, &stepy, 255);

		string winnameSigy = winname + " Ysignal";
		namedWindow(winnameSigy);
		if (isFirst)moveWindow(winnameSigy.c_str(), 0, 0);
		static int yshifty = 128;
		createTrackbar("shift y", winnameSigy, &yshifty, 128);
		static int ystepy = 128;
		createTrackbar("clip y", winnameSigy, &ystepy, 255);

		string winnameHist = winname + " Histogram";
		namedWindow(winnameHist);

		Mat dest;

		drawSignalX(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.cols, 350), pt.y, pt.x, shifty, step, stepy, 1);
		imshow(winnameSigx, dest);
		drawSignalY(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.rows, 350), pt.x, pt.y, yshifty, ystep, ystepy, 1);
		Mat temp;
		flip(dest.t(), temp, 0);
		imshow(winnameSigy, temp);

		Mat show;
		im[nov].convertTo(show, -1, max(amp, 1));

		//crop
		{
			int x = max(0, pt.x - step);
			int y = max(0, pt.y - ystep);
			int w = min(show.cols, x + 2 * step) - x;
			int h = min(show.rows, y + 2 * ystep) - y;
			Mat rectimage = Mat(show(Rect(x, y, w, h)));
			imshow("crop", rectimage);
		}
		rectangle(show, Point(pt.x - step, pt.y - ystep), Point(pt.x + step, pt.y + ystep), COLOR_GREEN);
		drawGrid(show, pt, COLOR_RED);
		imshowScale(winname, show);

		Mat hist;
		if (src.channels() == 1)
			drawHistogramImageGray(src, hist, COLOR_GRAY200, COLOR_ORANGE);
		else
			drawHistogramImage(src, hist, COLOR_ORANGE);

		imshow(winnameHist, hist);
		isFirst = false;
	}

	void imshowAnalysisCompare(std::string winname, cv::InputArray src1, cv::InputArray src2)
	{
		static bool isFirst = true;

		Mat im1, im2;
		if (src1.channels() == 1) cvtColor(src1, im1, COLOR_GRAY2BGR);
		else src1.copyTo(im1);
		if (src2.channels() == 1) cvtColor(src2, im2, COLOR_GRAY2BGR);
		else src2.copyTo(im2);

		namedWindow(winname);
		if (isFirst)moveWindow(winname.c_str(), im1.cols, 0);
		static Point pt = Point(im1.cols / 2, im1.rows / 2);
		setMouseCallback(winname, (MouseCallback)guiAnalysisMouse2, (void*)&pt);
		static int ALevel = 3;
		static int view_mode = 0;
		static int alpha = 0;
		static int channel = 3;
		static int step = im1.cols / 2;
		static int ystep = im1.rows / 2;

		static int bb = 0;
		static int level = 0;

		static int shifty = 128;
		static int stepy = 128;
		static int yshifty = 128;
		static int ystepy = 128;

		static Point sigxpos = Point(512, im1.rows);
		static Point sigypos = Point(0, 0);
		static Point hisxpos = Point(512, im1.rows);
		static Point shisypos = Point(0, 0);
		createTrackbar("level", winname, &ALevel, 3);
		createTrackbar("view mode", winname, &view_mode, 2);
		createTrackbar("alpha", winname, &alpha, 255);
		createTrackbar("channel", winname, &channel, 3);
		createTrackbar("x", winname, &pt.x, im1.cols - 1);
		createTrackbar("y", winname, &pt.y, im1.rows - 1);
		createTrackbar("clip x", winname, &step, im1.cols / 2);
		createTrackbar("clip y", winname, &ystep, im1.rows / 2);

		Mat show;
		if (view_mode == 0)
		{
			addWeighted(im1, 1.0 - alpha / 255.0, im2, alpha / 255.0, 0.0, show);
		}
		else if (view_mode == 1)
		{
			show.create(im1.size(), CV_8UC3);
			uchar* d = show.ptr<uchar>();
			uchar* s0 = im1.ptr<uchar>();
			uchar* s1 = im2.ptr<uchar>();
			for (int i = 0; i < im1.total(); i++)
			{
				d[i] = saturate_cast<uchar>(s0[i] - s1[i] + 127);
			}
		}
		else if (view_mode == 2)
		{
			Mat tt = abs(im1 - im2);
			if (channel < 0 || channel >2)
			{
				Mat temp, temp2;
				cvtColor(tt, temp, COLOR_BGR2GRAY);
				threshold(temp, temp2, alpha - 1, 255, THRESH_BINARY);
				cvtColor(temp2, show, COLOR_GRAY2BGR);
			}
			else
			{
				Mat temp;
				vector<Mat> vv;
				split(tt, vv);
				threshold(vv[channel], temp, alpha - 1, 255, THRESH_BINARY);
				cvtColor(temp, show, COLOR_GRAY2BGR);
			}
		}
		rectangle(show, Point(pt.x - step, pt.y - ystep), Point(pt.x + step, pt.y + ystep), COLOR_GREEN);
		drawGrid(show, pt, COLOR_RED);
		imshow(winname, show);

#pragma region level_1_signal
		if (ALevel > 1)
		{
			string winnameSigx = winname + " Xsignal";
			namedWindow(winnameSigx);
			if (isFirst)moveWindow(winnameSigx.c_str(), sigxpos.x, sigxpos.y);

			createTrackbar("shift y", winnameSigx, &shifty, 128);
			createTrackbar("clip y", winnameSigx, &stepy, 255);

			string winnameSigy = winname + " Ysignal";
			namedWindow(winnameSigy);
			if (isFirst)moveWindow(winnameSigy.c_str(), sigypos.x, sigypos.y);

			createTrackbar("shift y", winnameSigy, &yshifty, 128);
			createTrackbar("clip y", winnameSigy, &ystepy, 255);
			Mat dest;
			if (view_mode == 0)
			{
				vector<Mat> s;
				s.push_back(im1);
				s.push_back(im2);
				drawSignalX(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(im1.cols, 350), pt.y, pt.x, shifty, step, stepy);
				imshow(winnameSigx, dest);
				drawSignalY(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(im1.rows, 350), pt.x, pt.y, yshifty, ystep, ystepy);
				Mat temp;
				flip(dest.t(), temp, 0);
				imshow(winnameSigy, temp);
			}
			else if (view_mode == 1)
			{
				Mat ss(im1.size(), CV_16SC3);
				//Mat im1s, im2s;
				/*im1.convertTo(im1s, CV_16S);
				im2.convertTo(im2s, CV_16S);*/
				//ss = im1s - im2s;
				subtract(im1, im2, ss, Mat(), CV_16S);
				drawSignalX(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(im1.cols, 350), pt.y, pt.x, shifty - 128, step, stepy);
				imshow(winnameSigx, dest);
				drawSignalY(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(im1.rows, 350), pt.x, pt.y, yshifty - 128, ystep, ystepy);
				Mat temp;
				flip(dest.t(), temp, 0);
				imshow(winnameSigy, temp);
			}
			else if (view_mode == 2)
			{
				Mat ss(im1.size(), CV_16SC3);
				Mat im1s, im2s;
				im1.convertTo(im1s, CV_16SC3);
				im2.convertTo(im2s, CV_16SC3);
				ss = im1s - im2s;
				//subtract(im1,im2,ss,Mat(),CV_16SC3);

				drawSignalX(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(im1.cols, 350), pt.y, pt.x, shifty - 128, step, stepy);
				imshow(winnameSigx, dest);
				drawSignalY(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(im1.rows, 350), pt.x, pt.y, yshifty - 128, ystep, ystepy);
				Mat temp;
				flip(dest.t(), temp, 0);
				imshow(winnameSigy, temp);
			}
		}
		else
		{
			destroyWindow(winname + " Xsignal");
			destroyWindow(winname + " Ysignal");
		}
#pragma endregion
#pragma region level_2_histogram
		if (ALevel > 2)
		{
			string winnameHist1 = winname + " Histogram1";
			string winnameHist2 = winname + " Histogram2";
			namedWindow(winnameHist1);
			namedWindow(winnameHist2);
			if (isFirst)moveWindow(winnameHist1.c_str(), hisxpos.x, hisxpos.y);
			if (isFirst)moveWindow(winnameHist2.c_str(), hisxpos.x, hisxpos.y);

			Mat hist;
			if (src1.channels() == 1)
				drawAccumulateHistogramImageGray(src1, hist, COLOR_GRAY200, COLOR_ORANGE);
			else
				drawAccumulateHistogramImage(src1, hist, COLOR_ORANGE);

			imshow(winnameHist1, hist);

			if (src2.channels() == 1)
				drawAccumulateHistogramImageGray(src2, hist, COLOR_GRAY200, COLOR_ORANGE);
			else
				drawAccumulateHistogramImage(src2, hist, COLOR_ORANGE);

			imshow(winnameHist2, hist);
		}
		else
		{
			destroyWindow(winname + " Histogram1");
			destroyWindow(winname + " Histogram2");
		}
#pragma endregion

#pragma region level_0_base
		if (ALevel > 0)
		{
			string winnameInfo = winname + " Info";
			static cp::ConsoleImage cw(Size(640, 480), winnameInfo);
			static int infoth = 255;
			createTrackbar("BB", winnameInfo, &bb, min(im1.cols, im1.rows) / 2);
			createTrackbar("thresh", winnameInfo, &infoth, 255);
			createTrackbar("level", winnameInfo, &level, 2);
			static int isshowmask = 0;
			createTrackbar("show", winnameInfo, &isshowmask, 1);

			if (src1.channels() == src2.channels())
			{
				Mat tt = abs(im1 - im2);
				Mat threshmask;
				if (channel < 0 || channel >2)
				{
					Mat temp, temp2;
					cvtColor(tt, temp, COLOR_BGR2GRAY);
					threshold(temp, temp2, infoth - 1, 255, THRESH_BINARY);
					threshmask = Mat(~temp2);
				}
				else
				{
					Mat temp;
					vector<Mat> vv;
					split(tt, vv);
					threshold(vv[channel], temp, infoth - 1, 255, THRESH_BINARY);
					threshmask = Mat(~temp);
				}

				addBoxMask(threshmask, bb, bb);
				if (isshowmask == 1)
					imshow("threshmask", threshmask);
				else
					destroyWindow("threshmask");

				cw.clear();
				/*
				int cc = countNonZero(threshmask);
				cw("Pixel  %.03f %% NBP %d", (double)cc / threshmask.size().area()*100.0, threshmask.size().area() - cc);
				cw("PSNR  Y: %.03f", calcPSNR(src1, src2, 0, 82, threshmask));
				cw("MSE   Y: %.03f", calcMSE(src1, src2, 0, 82, threshmask));

				if (level > 0)
				{
				cw("SSIM  Y: %.03f", calcSSIM(src1, src2, 0, 82, threshmask));
				cw("DSSIM Y: %.03f", calcDSSIM(src1, src2, 0, 82, threshmask));
				}
				if (level > 1)
				{

				cw(" ");
				Scalar v1;
				v1.val[2] = calcPSNR(src1, src2, 0, CV_BGR2RGB, threshmask);
				v1.val[1] = calcPSNR(src1, src2, 1, CV_BGR2RGB, threshmask);
				v1.val[0] = calcPSNR(src1, src2, 2, CV_BGR2RGB, threshmask);
				Scalar v2;
				v2.val[2] = calcSSIM(src1, src2, 0, CV_BGR2RGB, threshmask);
				v2.val[1] = calcSSIM(src1, src2, 1, CV_BGR2RGB, threshmask);
				v2.val[0] = calcSSIM(src1, src2, 2, CV_BGR2RGB, threshmask);

				Scalar v3;
				v3.val[2] = calcDSSIM(src1, src2, 0, CV_BGR2RGB, threshmask);
				v3.val[1] = calcDSSIM(src1, src2, 1, CV_BGR2RGB, threshmask);
				v3.val[0] = calcDSSIM(src1, src2, 2, CV_BGR2RGB, threshmask);

				Scalar v4;
				v4.val[2] = calcPSNR(src1, src2, 0, CV_BGR2RGB, threshmask);
				v4.val[1] = calcPSNR(src1, src2, 1, CV_BGR2RGB, threshmask);
				v4.val[0] = calcPSNR(src1, src2, 2, CV_BGR2RGB, threshmask);


				if (src1.channels() == 3)
				{
				cw("PSNR  R: %.03f", v1[2]);
				cw("PSNR  G: %.03f", v1[1]);
				cw("PSNR  B: %.03f", v1[0]);
				cw(" ");
				cw("SSIM  R: %.03f", v2[2]);
				cw("SSIM  G: %.03f", v2[1]);
				cw("SSIM  B: %.03f", v2[0]);
				cw(" ");
				cw("DSSIM R: %.03f", v3[2]);
				cw("DSSIM G: %.03f", v3[1]);
				cw("DSSIM B: %.03f", v3[0]);
				cw(" ");
				cw("MSE   R: %.03f", v4[2]);
				cw("MSE   G: %.03f", v4[1]);
				cw("MSE   B: %.03f", v4[0]);

				//cw(XCV_RED,"PSNR  R: %.03f",v1[2]);
				//cw(XCV_GREEN,"PSNR  G: %.03f",v1[1]);
				//cw(XCV_CYAN,"PSNR  B: %.03f",v1[0]);
				//cw(" ");
				//cw(XCV_RED,"SSIM  R: %.03f",v2[2]);
				//cw(XCV_GREEN,"SSIM  G: %.03f",v2[1]);
				//cw(XCV_CYAN,"SSIM  B: %.03f",v2[0]);
				//cw(" ");
				//cw(XCV_RED,"DSSIM R: %.03f",v3[2]);
				//cw(XCV_GREEN,"DSSIM G: %.03f",v3[1]);
				//cw(XCV_CYAN,"DSSIM B: %.03f",v3[0]);
				//cw(" ");
				//cw(XCV_RED,"MSE   R: %.03f",v4[2]);
				//cw(XCV_GREEN,"MSE   G: %.03f",v4[1]);
				//cw(XCV_CYAN,"MSE   B: %.03f",v4[0]);
				}
				}
				*/
				imshow(winnameInfo, cw.image);
			}
		}
		else
		{
			destroyWindow(winname + " Info");
		}
#pragma endregion
		isFirst = false;
	}

	void guiAnalysisImage(InputArray src_)
	{
		Mat src = src_.getMat();
		string winname = "analysis";

		namedWindow(winname);
		moveWindow(winname.c_str(), src.cols * 2, 0);

		Mat im;
		if (src.channels() == 1)cvtColor(src, im, COLOR_GRAY2BGR);
		else src.copyTo(im);

		Point pt = Point(src.cols / 2, src.rows / 2);
		setMouseCallback(winname, (MouseCallback)guiAnalysisMouse2, (void*)&pt);

		int channel = 3;
		createTrackbar("channel", winname, &channel, 3);
		createTrackbar("x", winname, &pt.x, src.cols - 1);
		createTrackbar("y", winname, &pt.y, src.rows - 1);
		int step = src.cols / 2;
		createTrackbar("clip x", winname, &step, src.cols / 2);
		int ystep = src.rows / 2;
		createTrackbar("clip y", winname, &ystep, src.rows / 2);

		string winnameSigx = winname + " Xsignal";
		namedWindow(winnameSigx);

		moveWindow(winnameSigx.c_str(), 512, src.rows * 2);

		int shifty = 128;
		createTrackbar("shift y", winnameSigx, &shifty, 128);
		int stepy = 128;
		createTrackbar("clip y", winnameSigx, &stepy, 255);

		string winnameSigy = winname + " Ysignal";
		namedWindow(winnameSigy);
		moveWindow(winnameSigy.c_str(), 0, 0);
		int yshifty = 128;
		createTrackbar("shift y", winnameSigy, &yshifty, 128);
		int ystepy = 128;
		createTrackbar("clip y", winnameSigy, &ystepy, 255);

		string winnameHist = winname + " Histogram";
		namedWindow(winnameHist);

		Mat dest;

		Mat hist;
		if (src.channels() == 1)
			drawHistogramImageGray(src, hist, COLOR_GRAY200, COLOR_ORANGE);
		else
			drawHistogramImage(src, hist, COLOR_ORANGE);

		imshow(winnameHist, hist);

		int key = 0;
		Mat transY;
		while (key != 'q')
		{
			if (src.channels() == 1)cvtColor(src, im, COLOR_GRAY2BGR);
			else src.copyTo(im);

			drawSignalX(src, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.cols, 350), pt.y, pt.x, shifty, step, stepy);
			imshow(winnameSigx, dest);

			drawSignalY(src, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.rows, 350), pt.x, pt.y, yshifty, ystep, ystepy);
			flip(dest.t(), transY, 0);
			imshow(winnameSigy, transY);

			rectangle(im, Point(pt.x - step, pt.y - ystep), Point(pt.x + step, pt.y + ystep), COLOR_GREEN);
			drawGrid(im, pt, COLOR_RED);

			imshow(winname, im);
			key = waitKey(1);

			setTrackbarPos("x", winname, pt.x);
			setTrackbarPos("y", winname, pt.y);
		}
	}

	void guiAnalysisCompare(InputArray src1, InputArray src2, string wname)
	{
		int key = 0;
		while (key != 'q')
		{
			imshowAnalysisCompare(wname, src1, src2);
			key = waitKey(1);
		}
	}
#pragma endregion

#pragma region StackImage
	StackImage::StackImage(std::string window_name)
	{
		wname = window_name;
		//namedWindow(wname);
	}

	StackImage::~StackImage()
	{
		if (stack.size() != 0)destroyWindow(wname);
	}

	void StackImage::setWindowName(std::string window_name)
	{
		wname = window_name;
	}

	void StackImage::overwrite(cv::Mat& src)
	{
		if (stack.size() == 0)
		{
			push(src);
			return;
		}

		src.copyTo(stack[stack_max - 1]);
		if (stack_max > 1)
		{
			namedWindow(wname);
			createTrackbar("num", wname, &num_stack, stack_max);
			setTrackbarMax("num", wname, stack_max - 1);
			setTrackbarPos("num", wname, stack_max - 1);
		}
	}

	void StackImage::push(cv::Mat& src)
	{
		stack.push_back(src.clone());
		stack_max = (int)stack.size();

		if (stack_max > 0)
		{
			namedWindow(wname);
			createTrackbar("num", wname, &num_stack, stack_max);
			setTrackbarMax("num", wname, stack_max);
			//setTrackbarPos("num", wname, stack_max);
		}
	}

	void StackImage::push(vector<cv::Mat>& src)
	{
		for (int i = 0; i < src.size(); i++)
		{
			stack.push_back(src[i].clone());
		}
		stack_max = (int)stack.size();

		if (stack_max > 0)
		{
			namedWindow(wname);
			createTrackbar("num", wname, &num_stack, stack_max);
			setTrackbarMax("num", wname, stack_max);
			//setTrackbarPos("num", wname, stack_max);
		}
	}

	void StackImage::clear()
	{
		stack.clear();
		stack_max = (int)stack.size();
	}

	void StackImage::show(cv::Mat& src, bool isMergePages)
	{
		if (stack_max == 0)
		{
			imshow(wname, src);
		}
		else
		{
			if (isMergePages)
			{
				stack.push_back(src);
				Mat show; hconcat(stack, show);
				imshow(wname, show);
			}
			else
			{
				if (stack_max == num_stack) imshow(wname, src);
				else  imshow(wname, stack[num_stack]);
			}
		}
	}

	void StackImage::show()
	{
		if (stack_max > 0) imshow(wname, stack[min(num_stack, stack_max - 1)]);
	}

	void StackImage::gui()
	{
		if (stack_max == 0)return;
		int key = 0;
		while (key != 'q')
		{
			show();
			key = waitKey(1);
			if (key == 'f')
			{
				num_stack++;
				num_stack = (num_stack >= stack_max) ? 0 : num_stack;
				setTrackbarPos("num", wname, num_stack);
			}
			if (key == 'b')
			{
				num_stack--;
				num_stack = (num_stack == -1) ? stack_max - 1 : num_stack;
				setTrackbarPos("num", wname, num_stack);
			}
		}
	}
#pragma endregion
}