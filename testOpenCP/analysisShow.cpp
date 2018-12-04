#include "test.hpp"

using namespace std;
using namespace cv;
using namespace cp;

void guiAnalysisCompare(Mat& src1, Mat& src2)
{
	int key = 0;
	while (key != 'q')
	{
		imshowAnalysisCompare("AnalysisCompare", src1, src2);
		key = waitKey(1);
	}
}

template <class T>
void getImageLine_(Mat& src, int channel, vector<Point>& v, const int line)
{
	const int ch = src.channels();

	T* s = src.ptr<T>(line);
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
				v[i] = Point(i, (int)(0.299*s[3 * i + 2] + 0.587*s[3 * i + 1] + 0.114*s[3 * i + 0]));
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
	else if (src.type() == CV_16S || src.type() == CV_16SC3)getImageLine_<short>(src, channel, v, line);
	else if (src.type() == CV_16U || src.type() == CV_16UC3)getImageLine_<ushort>(src, channel, v, line);
	else if (src.type() == CV_32F || src.type() == CV_32FC3)getImageLine_<float>(src, channel, v, line);
	else if (src.type() == CV_64F || src.type() == CV_64FC3)getImageLine_<double>(src, channel, v, line);
}

static void getImageVLine(Mat& src, vector<Point>& v, const int line, int channel)
{
	if (v.size() == 0)
		v.resize(src.rows);

	Mat srct = src.t();

	if (src.type() == CV_8U || src.type() == CV_8UC3)getImageLine_<uchar>(srct, channel, v, line);
	else if (src.type() == CV_16S || src.type() == CV_16SC3)getImageLine_<short>(srct, channel, v, line);
	else if (src.type() == CV_16U || src.type() == CV_16UC3)getImageLine_<ushort>(srct, channel, v, line);
	else if (src.type() == CV_32F || src.type() == CV_32FC3)getImageLine_<float>(srct, channel, v, line);
	else if (src.type() == CV_64F || src.type() == CV_64FC3)getImageLine_<double>(srct, channel, v, line);
}

void drawSignalX(InputArray src_, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size outputImageSize, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> src;
	src_.getMatVector(src);
	
	Plot p(outputImageSize);
	p.setPlotProfile(false, false, false);
	p.setPlotSymbolALL(Plot::SYMBOL_NOPOINT);
	p.setPlotLineTypeALL(linetype);
	
	p.setXYMinMax(shiftx - max(rangex, 1), shiftx + max(rangex, 1), shiftvalue - rangevalue, shiftvalue + rangevalue);
	vector<vector<Point>> v((int)src.size());

	for (int i = 0; i < (int)src.size(); i++)
	{
		getImageLine(src[i], v[i], line_height, color);
		p.push_back(v[i], i);
		p.plotData();
	}

	p.render.copyTo(dest);
}

void drawSignalX(Mat& src1, Mat& src2, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> s;
	s.push_back(src1);
	s.push_back(src2);
	drawSignalX(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
}

void drawSignalY(vector<Mat>& src, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	Plot p(size);

	p.setPlotProfile(false, false, false);
	p.setPlotSymbolALL(Plot::SYMBOL_NOPOINT);
	p.setPlotLineTypeALL(linetype);
	p.setXYMinMax(shiftx - max(rangex, 1), shiftx + max(rangex, 1), shiftvalue - rangevalue, shiftvalue + rangevalue);
	vector<vector<Point>> v((int)src.size());
	for (int i = 0; i < (int)src.size(); i++)
	{
		getImageVLine(src[i], v[i], line_height, color);
		p.push_back(v[i], i);
		p.plotData();
	}
	p.render.copyTo(dest);
}

void drawSignalY(Mat& src, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> s;
	s.push_back(src);
	drawSignalY(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
}

void drawSignalY(Mat& src1, Mat& src2, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> s;
	s.push_back(src1);
	s.push_back(src2);
	drawSignalY(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
}

void imshowAnalysis(String winname, Mat& src)
{
	static bool isFirst = true;
	Mat im;
	if (src.channels() == 1)cvtColor(src, im, CV_GRAY2BGR);
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

void imshowAnalysis(String winname, vector<Mat>& s)
{
	Mat src = s[0];
	static bool isFirst = true;
	vector<Mat> im(s.size());
	for (int i = 0; i < (int)s.size(); i++)
	{
		if (src.channels() == 1)cvtColor(s[i], im[i], CV_GRAY2BGR);
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
	imshow(winname, show);
	
	Mat hist;
	if (src.channels() == 1)
		drawHistogramImageGray(src, hist, COLOR_GRAY200, COLOR_ORANGE);
	else
		drawHistogramImage(src, hist, COLOR_ORANGE);
	
	imshow(winnameHist, hist);
	isFirst = false;
}

void imshowAnalysisCompare(String winname, Mat& src1, Mat& src2)
{
	static bool isFirst = true;

	Mat im1, im2;
	if (src1.channels() == 1)cvtColor(src1, im1, COLOR_GRAY2BGR);
	else src1.copyTo(im1);
	if (src2.channels() == 1)cvtColor(src2, im2, COLOR_GRAY2BGR);
	else src2.copyTo(im2);

	namedWindow(winname);
	if (isFirst)moveWindow(winname.c_str(), src1.cols, 0);
	static Point pt = Point(src1.cols / 2, src1.rows / 2);

	static int ALevel = 3;
	static int sw = 0;
	static int alpha = 0;
	static int channel = 3;
	static int step = src1.cols / 2;
	static int ystep = src1.rows / 2;

	static int bb = 0;
	static int level = 0;

	static int shifty = 128;
	static int stepy = 128;
	static int yshifty = 128;
	static int ystepy = 128;

	{//sW!=0
		//stepy=256;
		;
	}

	static Point sigxpos = Point(512, src1.rows);
	static Point sigypos = Point(0, 0);
	static Point hisxpos = Point(512, src1.rows);
	static Point shisypos = Point(0, 0);
	createTrackbar("level", winname, &ALevel, 3);
	createTrackbar("sw", winname, &sw, 2);
	createTrackbar("alpha", winname, &alpha, 255);
	createTrackbar("channel", winname, &channel, 3);
	createTrackbar("x", winname, &pt.x, src1.cols - 1);
	createTrackbar("y", winname, &pt.y, src1.rows - 1);
	createTrackbar("clip x", winname, &step, src1.cols / 2);
	createTrackbar("clip y", winname, &ystep, src1.rows / 2);

	Mat show;
	if (sw == 0)
	{
		addWeighted(im1, 1.0 - alpha / 255.0, im2, alpha / 255.0, 0.0, show);
	}
	else if (sw == 1)
	{
		Mat temp2 = im1 - im2 + 127;
		temp2.convertTo(show, CV_8UC3);
	}
	else if (sw == 2)
	{
		Mat tt = abs(im1 - im2);
		if (channel < 0 || channel >2)
		{
			Mat temp, temp2;
			cvtColor(tt, temp, CV_BGR2GRAY);
			threshold(temp, temp2, alpha - 1, 255, THRESH_BINARY);
			cvtColor(temp2, show, CV_GRAY2BGR);
		}
		else
		{
			Mat temp;
			vector<Mat> vv;
			split(tt, vv);
			threshold(vv[channel], temp, alpha - 1, 255, THRESH_BINARY);
			cvtColor(temp, show, CV_GRAY2BGR);
		}
	}
	rectangle(show, Point(pt.x - step, pt.y - ystep), Point(pt.x + step, pt.y + ystep), COLOR_GREEN);
	drawGrid(show, pt, COLOR_RED);
	imshow(winname, show);

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
		if (sw == 0)
		{
			vector<Mat> s;
			s.push_back(im1);
			s.push_back(im2);
			drawSignalX(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src1.cols, 350), pt.y, pt.x, shifty, step, stepy);
			imshow(winnameSigx, dest);
			drawSignalY(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src1.rows, 350), pt.x, pt.y, yshifty, ystep, ystepy);
			Mat temp;
			flip(dest.t(), temp, 0);
			imshow(winnameSigy, temp);
		}
		else if (sw == 1)
		{
			Mat ss(im1.size(), CV_16SC3);
			Mat im1s, im2s;
			im1.convertTo(im1s, CV_16SC3);
			im2.convertTo(im2s, CV_16SC3);
			ss = im1s - im2s;
			//subtract(im1,im2,ss,Mat(),CV_16SC3);

			drawSignalX(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src1.cols, 350), pt.y, pt.x, shifty - 128, step, stepy);
			imshow(winnameSigx, dest);
			drawSignalY(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src1.rows, 350), pt.x, pt.y, yshifty - 128, ystep, ystepy);
			Mat temp;
			flip(dest.t(), temp, 0);
			imshow(winnameSigy, temp);
		}
		else if (sw == 2)
		{
			Mat ss(im1.size(), CV_16SC3);
			Mat im1s, im2s;
			im1.convertTo(im1s, CV_16SC3);
			im2.convertTo(im2s, CV_16SC3);
			ss = im1s - im2s;
			//subtract(im1,im2,ss,Mat(),CV_16SC3);

			drawSignalX(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src1.cols, 350), pt.y, pt.x, shifty - 128, step, stepy);
			imshow(winnameSigx, dest);
			drawSignalY(ss, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src1.rows, 350), pt.x, pt.y, yshifty - 128, ystep, ystepy);
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
	if (ALevel > 0)
	{
		static ConsoleImage cw(Size(640, 480));
		string winnameInfo = winname + " Info";
		namedWindow(winnameInfo);
		static int infoth = 255;
		createTrackbar("BB", winnameInfo, &bb, min(src1.cols, src1.rows) / 2);
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
				cvtColor(tt, temp, CV_BGR2GRAY);
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
	isFirst = false;
}


void guiPreviewMouse2(int event, int x, int y, int flags, void* param)
{
	Point* ret = (Point*)param;
	if (flags == CV_EVENT_FLAG_LBUTTON)
	{
		ret->x = x;
		ret->y = y;
	}
}

void guiAnalysisImage(InputArray src_)
{
	Mat src = src_.getMat();
	string winname = "analysis";

	namedWindow(winname);
	moveWindow(winname.c_str(), src.cols * 2, 0);

	Mat im;
	if (src.channels() == 1)cvtColor(src, im, CV_GRAY2BGR);
	else src.copyTo(im);

	Point pt = Point(src.cols / 2, src.rows / 2);
	setMouseCallback(winname, (MouseCallback)guiPreviewMouse2, (void*)&pt);

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
		if (src.channels() == 1)cvtColor(src, im, CV_GRAY2BGR);
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