#include "opencp.hpp"
#include <iostream>
using namespace std;
using namespace cv;

namespace cp
{

gnuplot::gnuplot(string gnuplotpath)
{

	if ((fp = _popen(gnuplotpath.c_str(), "w")) == NULL)
	{
		fprintf(stderr, "Cannot open gnuplot @ %s\n", gnuplotpath);
		exit(1);
	}
}
void gnuplot::cmd(string name)
{
	fprintf(fp, name.c_str());
	fprintf(fp, "\n");
	fflush(fp);
}
gnuplot::~gnuplot()
{
	//fclose(fp);
	cmd("exit");
	_pclose(fp);
}

void plotGraph(Mat& render, vector<Point2d>& data, double xmin, double xmax, double ymin, double ymax,
	Scalar color, int lt, int isLine, int thickness, int ps)

{
	double x = (double)render.cols / (xmax - xmin);
	double y = (double)render.rows / (ymax - ymin);

	int H = render.rows - 1;
	const int size = (int)data.size();
	for (int i = 0; i < size; i++)
	{
		double src = data[i].x;
		double dest = data[i].y;

		cv::Point p = cvPoint(cvRound(x*(src - xmin)), H - cvRound(y*(dest - ymin)));
		if (isLine == Plot::LINE_LINEAR)
		{
			if (i != size - 1)
			{
				double nsrc = data[i + 1].x;
				double ndest = data[i + 1].y;
				line(render, p, cvPoint(cvRound(x*(nsrc - xmin)), H - cvRound(y*(ndest - ymin))),
					color, thickness);
			}
		}
		else if (isLine == Plot::LINE_H2V)
		{
			if (i != size - 1)
			{
				double nsrc = data[i + 1].x;
				double ndest = data[i + 1].x;
				line(render, p, cvPoint(cvRound(x*(nsrc - xmin)), p.y), color, thickness);
				line(render, cvPoint(cvRound(x*(nsrc - xmin)), p.y), cvPoint(cvRound(x*(nsrc - xmin)), H - cvRound(y*(ndest - ymin))), color, thickness);
			}
		}
		else if (isLine == Plot::LINE_V2H)
		{
			if (i != size - 1)
			{
				double nsrc = data[i + 1].x;
				double ndest = data[i + 1].x;
				line(render, p, cvPoint(p.x, H - cvRound(y*(ndest - ymin))), color, thickness);
				line(render, cvPoint(p.x, H - cvRound(y*(ndest - ymin))), cvPoint(cvRound(x*(nsrc - xmin)), H - cvRound(y*(ndest - ymin))), color, thickness);
			}
		}

		if (lt == Plot::SYMBOL_PLUS)
		{
			drawPlus(render, p, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::SYMBOL_TIMES)
		{
			drawTimes(render, p, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::SYMBOL_ASTERRISK)
		{
			drawAsterisk(render, p, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::SYMBOL_CIRCLE)
		{
			circle(render, p, ps, color, thickness);
		}
		else if (lt == Plot::SYMBOL_RECTANGLE)
		{
			rectangle(render, cvPoint(p.x - ps, p.y - ps), cvPoint(p.x + ps, p.y + ps),
				color, thickness);
		}
		else if (lt == Plot::SYMBOL_CIRCLE_FILL)
		{
			circle(render, p, ps, color, CV_FILLED);
		}
		else if (lt == Plot::SYMBOL_RECTANGLE_FILL)
		{
			rectangle(render, cvPoint(p.x - ps, p.y - ps), cvPoint(p.x + ps, p.y + ps),
				color, CV_FILLED);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE)
		{
			triangle(render, p, 2 * ps, color, thickness);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE_FILL)
		{
			triangle(render, p, 2 * ps, color, CV_FILLED);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE_INV)
		{
			triangleinv(render, p, 2 * ps, color, thickness);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE_INV_FILL)
		{
			triangleinv(render, p, 2 * ps, color, CV_FILLED);
		}
	}
}


Plot::Plot(Size plotsize_)
{
	data_max = 1;
	xlabel = "x";
	ylabel = "y";
	setBackGoundColor(COLOR_WHITE);

	origin = Point(64, 64);//default
	plotImage = NULL;
	render = NULL;
	setPlotImageSize(plotsize_);

	keyImage.create(Size(256, 256), CV_8UC3);
	keyImage.setTo(background_color);

	setXYMinMax(0, plotsize.width, 0, plotsize.height);
	isPosition = true;
	init();
}

Plot::~Plot()
{
	;
}

void Plot::point2val(cv::Point pt, double* valx, double* valy)
{
	double x = (double)plotImage.cols / (xmax - xmin);
	double y = (double)plotImage.rows / (ymax - ymin);
	int H = plotImage.rows - 1;

	*valx = (pt.x - (origin.x) * 2) / x + xmin;
	*valy = (H - (pt.y - origin.y)) / y + ymin;
}

void Plot::init()
{
	const int DefaultPlotInfoSize = 64;
	pinfo.resize(DefaultPlotInfoSize);
	for (int i = 0; i < pinfo.size(); i++)
	{
		pinfo[i].symbolType = Plot::SYMBOL_PLUS;
		pinfo[i].lineType = Plot::LINE_LINEAR;
		pinfo[i].thickness = 1;

		double v = (double)i / DefaultPlotInfoSize*255.0;
		pinfo[i].color = getPseudoColor(cv::saturate_cast<uchar>(v));

		pinfo[i].keyname = format("data %02d", i);
	}

	pinfo[0].color = COLOR_RED;
	pinfo[0].symbolType = SYMBOL_PLUS;

	pinfo[1].color = COLOR_GREEN;
	pinfo[1].symbolType = SYMBOL_TIMES;

	pinfo[2].color = COLOR_BLUE;
	pinfo[2].symbolType = SYMBOL_ASTERRISK;

	pinfo[3].color = COLOR_MAGENDA;
	pinfo[3].symbolType = SYMBOL_RECTANGLE;

	pinfo[4].color = CV_RGB(0, 0, 128);
	pinfo[4].symbolType = SYMBOL_RECTANGLE_FILL;

	pinfo[5].color = CV_RGB(128, 0, 0);
	pinfo[5].symbolType = SYMBOL_CIRCLE;

	pinfo[6].color = CV_RGB(0, 128, 128);
	pinfo[6].symbolType = SYMBOL_CIRCLE_FILL;

	pinfo[7].color = CV_RGB(0, 0, 0);
	pinfo[7].symbolType = SYMBOL_TRIANGLE;

	pinfo[8].color = CV_RGB(128, 128, 128);
	pinfo[8].symbolType = SYMBOL_TRIANGLE_FILL;

	pinfo[9].color = CV_RGB(0, 128, 64);
	pinfo[9].symbolType = SYMBOL_TRIANGLE_INV;

	pinfo[10].color = CV_RGB(128, 128, 0);
	pinfo[10].symbolType = SYMBOL_TRIANGLE_INV_FILL;

	setPlotProfile(false, true, false);
	graphImage = render;
}
void Plot::setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_)
{
	isZeroCross = isZeroCross_;
	isXYMAXMIN = isXYMAXMIN_;
	isXYCenter = isXYCenter_;
}

void Plot::setPlotImageSize(Size s)
{
	plotsize = s;
	plotImage.create(s, CV_8UC3);
	render.create(Size(plotsize.width + 4 * origin.x, plotsize.height + 2 * origin.y), CV_8UC3);
}

void Plot::setXYOriginZERO()
{
	recomputeXYMAXMIN(false);
	xmin = 0;
	ymin = 0;
}
void Plot::setYOriginZERO()
{
	recomputeXYMAXMIN(false);
	ymin = 0;
}
void Plot::setXOriginZERO()
{
	recomputeXYMAXMIN(false);
	xmin = 0;
}

void Plot::recomputeXYMAXMIN(bool isCenter, double marginrate)
{
	if (marginrate<0.0 || marginrate>1.0)marginrate = 1.0;
	xmax = -INT_MAX;
	xmin = INT_MAX;
	ymax = -INT_MAX;
	ymin = INT_MAX;
	for (int i = 0; i < data_max; i++)
	{
		for (int j = 0; j < pinfo[i].data.size(); j++)
		{
			double x = pinfo[i].data[j].x;
			double y = pinfo[i].data[j].y;
			xmax = (xmax < x) ? x : xmax;
			xmin = (xmin > x) ? x : xmin;

			ymax = (ymax < y) ? y : ymax;
			ymin = (ymin > y) ? y : ymin;
		}
	}
	xmax_no_margin = xmax;
	xmin_no_margin = xmin;
	ymax_no_margin = ymax;
	ymin_no_margin = ymin;

	double xmargin = (xmax - xmin)*(1.0 - marginrate)*0.5;
	xmax += xmargin;
	xmin -= xmargin;

	double ymargin = (ymax - ymin)*(1.0 - marginrate)*0.5;
	ymax += ymargin;
	ymin -= ymargin;

	if (isCenter)
	{
		double xxx = abs(xmax);
		double yyy = abs(ymax);
		xxx = (xxx < abs(xmin)) ? abs(xmin) : xxx;
		yyy = (yyy < abs(ymin)) ? abs(ymin) : yyy;

		xmax = xxx;
		xmin = -xxx;
		ymax = yyy;
		ymin = -yyy;

		xxx = abs(xmax_no_margin);
		yyy = abs(ymax_no_margin);
		xxx = (xxx < abs(xmin_no_margin)) ? abs(xmin_no_margin) : xxx;
		yyy = (yyy < abs(ymin_no_margin)) ? abs(ymin_no_margin) : yyy;

		xmax_no_margin = xxx;
		xmin_no_margin = -xxx;
		ymax_no_margin = yyy;
		ymin_no_margin = -yyy;
	}
}

void Plot::setXYMinMax(double xmin_, double xmax_, double ymin_, double ymax_)
{
	xmin = xmin_;
	xmax = xmax_;
	ymin = ymin_;
	ymax = ymax_;

	xmax_no_margin = xmax;
	xmin_no_margin = xmin;
	ymax_no_margin = ymax;
	ymin_no_margin = ymin;
}
void Plot::setXMinMax(double xmin_, double xmax_)
{
	recomputeXYMAXMIN(isXYCenter);
	xmin = xmin_;
	xmax = xmax_;
}
void Plot::setYMinMax(double ymin_, double ymax_)
{
	recomputeXYMAXMIN(isXYCenter);
	ymin = ymin_;
	ymax = ymax_;
}

void Plot::setBackGoundColor(Scalar cl)
{
	background_color = cl;
}

void Plot::setPlotThickness(int plotnum, int thickness_)
{
	pinfo[plotnum].thickness = thickness_;
}

void Plot::setPlotColor(int plotnum, Scalar color_)
{
	pinfo[plotnum].color = color_;
}

void Plot::setPlotLineType(int plotnum, int lineType)
{
	pinfo[plotnum].lineType = lineType;
}

void Plot::setPlotSymbol(int plotnum, int symboltype)
{
	pinfo[plotnum].symbolType = symboltype;
}

void Plot::setPlotKeyName(int plotnum, string name)
{
	pinfo[plotnum].keyname = name;
}

void Plot::setPlot(int plotnum, Scalar color, int symboltype, int linetype, int thickness)
{
	setPlotColor(plotnum, color);
	setPlotSymbol(plotnum, symboltype);
	setPlotLineType(plotnum, linetype);
	setPlotThickness(plotnum, thickness);
}

void Plot::setLinetypeALL(int linetype)
{
	for (int i = 0; i < pinfo.size(); i++)
	{
		pinfo[i].lineType = linetype;
	}
}

void Plot::push_back(double x, double y, int plotIndex)
{
	data_max = max(data_max, plotIndex + 1);
	pinfo[plotIndex].data.push_back(Point2d(x, y));
}
void Plot::push_back(vector<cv::Point> point, int plotIndex)
{
	data_max = max(data_max, plotIndex + 1);
	for (int i = 0; i < (int)point.size(); i++)
	{
		push_back(point[i].x, point[i].y, plotIndex);
	}
}

void Plot::push_back(vector<cv::Point2d> point, int plotIndex)
{
	data_max = max(data_max, plotIndex + 1);
	for (int i = 0; i < (int)point.size() - 1; i++)
	{
		push_back(point[i].x, point[i].y, plotIndex);
	}
}


void Plot::erase(int sampleIndex, int plotIndex)
{
	pinfo[plotIndex].data.erase(pinfo[plotIndex].data.begin() + sampleIndex);
}

void Plot::insert(Point2d v, int sampleIndex, int plotIndex)
{
	pinfo[plotIndex].data.insert(pinfo[plotIndex].data.begin() + sampleIndex, v);
}

void Plot::insert(Point v, int sampleIndex, int plotIndex)
{
	insert(Point2d((double)v.x, (double)v.y), sampleIndex, plotIndex);
}

void Plot::insert(double x, double y, int sampleIndex, int plotIndex)
{
	insert(Point2d(x, y), sampleIndex, plotIndex);
}

void Plot::clear(int datanum)
{
	if (datanum<0)
	{
		for (int i = 0; i < data_max; i++)
			pinfo[i].data.clear();
			
	}
	else
		pinfo[datanum].data.clear();
}


void Plot::swapPlot(int plotIndex1, int plotIndex2)
{
	swap(pinfo[plotIndex1].data, pinfo[plotIndex2].data);
}

void Plot::makeBB(bool isFont)
{
	render.setTo(background_color);
	Mat roi = render(Rect(origin.x * 2, origin.y, plotsize.width, plotsize.height));
	rectangle(plotImage, Point(0, 0), Point(plotImage.cols - 1, plotImage.rows - 1), COLOR_BLACK, 1);
	plotImage.copyTo(roi);

	if (isFont)
	{
		putText(render, xlabel, Point(render.cols / 2, (int)(origin.y*1.85 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);
		putText(render, ylabel, Point(20, (int)(origin.y*0.25 + plotImage.rows / 2)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		string buff;
		//x coordinate
		buff = format("%.2f", xmin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (xmax - xmin)*0.25 + xmin);
		putText(render, buff, Point((int)(origin.x + plotImage.cols*0.25 + 15), (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (xmax - xmin)*0.5 + xmin);
		putText(render, buff, Point((int)(origin.x + plotImage.cols*0.5 + 45), (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (xmax - xmin)*0.75 + xmin);
		putText(render, buff, Point((int)(origin.x + plotImage.cols*0.75 + 35), (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", xmax);
		putText(render, buff, Point(origin.x + plotImage.cols, (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		//y coordinate
		buff = format("%.2f", ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (ymax - ymin)*0.5 + ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows*0.5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (ymax - ymin)*0.25 + ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows*0.75)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (ymax - ymin)*0.75 + ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows*0.25)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", ymax);
		putText(render, buff, Point(origin.x, origin.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);
	}
}
void Plot::plotPoint(Point2d point, Scalar color_, int thickness_, int linetype)
{
	CvMat* temp = cvCreateMat(5, 2, CV_64F);

	vector<Point2d> data;

	data.push_back(Point2d(point.x, ymin));
	data.push_back(Point2d(point.x, ymax));
	data.push_back(Point2d(point.x, point.y));
	data.push_back(Point2d(xmax, point.y));
	data.push_back(Point2d(xmin, point.y));

	plotGraph(plotImage, data, xmin, xmax, ymin, ymax, color_, SYMBOL_NOPOINT, linetype, thickness_);
}


void Plot::plotGrid(int level)
{
	if (level > 0)
	{
		plotPoint(Point2d((xmax - xmin) / 2.0 + xmin, (ymax - ymin) / 2.0 + ymin), COLOR_GRAY150, 1);
	}
	if (level > 1)
	{
		plotPoint(Point2d((xmax - xmin)*1.0 / 4.0 + xmin, (ymax - ymin)*1.0 / 4.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 4.0 + xmin, (ymax - ymin)*1.0 / 4.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*1.0 / 4.0 + xmin, (ymax - ymin)*3.0 / 4.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 4.0 + xmin, (ymax - ymin)*3.0 / 4.0 + ymin), COLOR_GRAY200, 1);
	}
	if (level > 2)
	{
		plotPoint(Point2d((xmax - xmin)*1.0 / 8.0 + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 8.0 + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*1.0 / 8.0 + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 8.0 + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);

		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);

		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);

		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
	}
}

void Plot::makeKey(int num)
{
	int step = 20;
	keyImage.create(Size(256, 20 * (num + 1) + 3), CV_8UC3);
	keyImage.setTo(background_color);

	int height = (int)(0.8*keyImage.rows);
	CvMat* data = cvCreateMat(2, 2, CV_64F);
	for (int i = 0; i < num; i++)
	{
		vector<Point2d> data;
		data.push_back(Point2d(192.0, keyImage.rows - (i + 1) * 20));
		data.push_back(Point2d(keyImage.cols - 20, keyImage.rows - (i + 1) * 20));


		plotGraph(keyImage, data, 0, keyImage.cols, 0, keyImage.rows, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].thickness);
		putText(keyImage, pinfo[i].keyname, Point(0, (i + 1) * 20 + 3), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, pinfo[i].color);
	}
}

void Plot::plotData(int gridlevel, int isKey)
{
	plotImage.setTo(background_color);
	plotGrid(gridlevel);

	if (isZeroCross)	plotPoint(Point2d(0.0, 0.0), COLOR_ORANGE, 1);

	for (int i = 0; i < data_max; i++)
	{
		plotGraph(plotImage, pinfo[i].data, xmin, xmax, ymin, ymax, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].thickness);
	}
	makeBB(true);

	Mat temp = render.clone();
	if (isKey != 0)
	{
		Mat roi;
		if (isKey == 1)
		{
			roi = render(Rect(render.cols - keyImage.cols - 150, 80, keyImage.cols, keyImage.rows));
		}
		else if (isKey == 4)
		{
			roi = render(Rect(render.cols - keyImage.cols - 150, render.rows - keyImage.rows - 150, keyImage.cols, keyImage.rows));
		}
		else if (isKey == 2)
		{
			roi = render(Rect(160, 80, keyImage.cols, keyImage.rows));
		}
		else if (isKey == 3)
		{
			roi = render(Rect(160, render.rows - keyImage.rows - 150, keyImage.cols, keyImage.rows));
		}
		keyImage.copyTo(roi);
	}
	addWeighted(render, 0.8, temp, 0.2, 0.0, render);
}

void Plot::save(string name)
{
	FILE* fp = fopen(name.c_str(), "w");

	int dmax = (int)pinfo[0].data.size();
	for (int i = 1; i < data_max; i++)
	{
		dmax = max((int)pinfo[i].data.size(), dmax);
	}

	for (int n = 0; n < dmax; n++)
	{
		for (int i = 0; i < data_max; i++)
		{
			if (n < pinfo[i].data.size())
			{
				double x = pinfo[i].data[n].x;
				double y = pinfo[i].data[n].y;
				fprintf(fp, "%f %f ", x, y);
			}
			else
			{
				double x = pinfo[i].data[pinfo[i].data.size() - 1].x;
				double y = pinfo[i].data[pinfo[i].data.size() - 1].y;
				fprintf(fp, "%f %f ", x, y);
			}
		}
		fprintf(fp, "\n");
	}
	cout << "p ";
	for (int i = 0; i < data_max; i++)
	{
		cout << "'" << name << "'" << " u " << 2 * i + 1 << ":" << 2 * i + 2 << " w lp" << ",";
	}
	cout << endl;
	fclose(fp);
}

Scalar Plot::getPseudoColor(uchar val)
{
	int i = val;
	double d = 255.0 / 63.0;
	Scalar ret;

	{//g
		uchar lr[256];
		for (int i = 0; i < 64; i++)
			lr[i] = cvRound(d*i);
		for (int i = 64; i < 192; i++)
			lr[i] = 255;
		for (int i = 192; i < 256; i++)
			lr[i] = cvRound(255 - d*(i - 192));

		ret.val[1] = lr[val];
	}
		{//r
			uchar lr[256];
			for (int i = 0; i < 128; i++)
				lr[i] = 0;
			for (int i = 128; i < 192; i++)
				lr[i] = cvRound(d*(i - 128));
			for (int i = 192; i < 256; i++)
				lr[i] = 255;

			ret.val[0] = lr[val];
		}
		{//b
			uchar lr[256];
			for (int i = 0; i < 64; i++)
				lr[i] = 255;
			for (int i = 64; i < 128; i++)
				lr[i] = cvRound(255 - d*(i - 64));
			for (int i = 128; i < 256; i++)
				lr[i] = 0;
			ret.val[2] = lr[val];
		}
		return ret;
}

static void guiPreviewMouse(int event, int x, int y, int flags, void* param)
{
	Point* ret = (Point*)param;

	if (flags == CV_EVENT_FLAG_LBUTTON)
	{
		ret->x = x;
		ret->y = y;
	}
}

void Plot::plot(string wname, bool isWait, string gnuplotpath)
{
	Point pt = Point(0, 0);
	namedWindow(wname);

	plotData(0, false);
	//int ym=ymax;
	//int yn=ymin;
	//createTrackbar("ymax",wname,&ym,ymax*2);
	//createTrackbar("ymin",wname,&yn,ymax*2);
	setMouseCallback(wname, (MouseCallback)guiPreviewMouse, (void*)&pt);
	int key = 0;
	int isKey = 1;
	int gridlevel = 0;
	makeKey(data_max);

	recomputeXYMAXMIN();
	while (key != 'q')
	{
		//ymax=ym+1;
		//ymin=yn;
		plotData(gridlevel, isKey);
		if (isPosition)
		{
			double xx = 0.0;
			double yy = 0.0;
			point2val(pt, &xx, &yy);
			if (pt.x < 0 || pt.y < 0 || pt.x >= render.cols || pt.y >= render.rows)
			{
				pt = Point(0, 0);
				xx = 0.0;
				yy = 0.0;
			}
			string text = format("(%f,%f)", xx, yy);
			putText(render, text, Point(100, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);
		}

		if (isPosition)drawGrid(render, pt, Scalar(180, 180, 255), 1, 4, 0);
		imshow(wname, render);
		key = waitKey(30);

		if (key == '?')
		{
			cout << "*** Help message ***" << endl;
			cout << "m: " << "show mouseover position and grid" << endl;
			cout << "c: " << "(0,0)point must posit center" << endl;
			cout << "g: " << "Show grid" << endl;

			cout << "k: " << "Show key" << endl;

			cout << "x: " << "Set X origin zero " << endl;
			cout << "y: " << "Set Y origin zero " << endl;
			cout << "z: " << "Set XY origin zero " << endl;
			cout << "r: " << "Reset XY max min" << endl;

			cout << "s: " << "Save image (plot.png)" << endl;
			cout << "q: " << "Quit" << endl;

			cout << "********************" << endl;
			cout << endl;
		}
		if (key == 'm')
		{
			isPosition = (isPosition) ? false : true;
		}

		if (key == 'r')
		{
			recomputeXYMAXMIN(false);
		}
		if (key == 'c')
		{
			recomputeXYMAXMIN(true);
		}
		if (key == 'x')
		{
			setXOriginZERO();
		}
		if (key == 'y')
		{
			setYOriginZERO();
		}
		if (key == 'z')
		{
			setXYOriginZERO();
		}
		if (key == 'k')
		{
			isKey++;
			if (isKey == 5)
				isKey = 0;
		}
		if (key == 'g')
		{
			gridlevel++;
			if (gridlevel > 3)gridlevel = 0;
		}
		if (key == 'p')
		{
			save("plot");
			std::string a("plot ");
			gnuplot gplot(gnuplotpath);
			for (int i = 0; i < data_max; i++)
			{
				char name[64];
				if (i != data_max - 1)
					sprintf(name, "'plot' u %d:%d w lp,", 2 * i + 1, 2 * i + 2);
				else
					sprintf(name, "'plot' u %d:%d w lp\n", 2 * i + 1, 2 * i + 2);
				a += name;
			}
			gplot.cmd(a.c_str());
		}
		if (key == 's')
		{
			save("plot");
			imwrite("plotim.png", render);
			//imwrite("plot.png", save);
		}
		if (!isWait) break;
	}
	if (isWait) destroyWindow(wname);
}

void Plot2D::createPlot()
{
	graphBase = Mat::zeros(Size(countx, county), CV_64F);
}
void Plot2D::setMinMaxX(double minv, double maxv, int count)
{
	minx = minv;
	maxx = maxv;
	countx = count;
}
void Plot2D::setMinMaxY(double minv, double maxv, int count)
{
	miny = minv;
	maxy = maxv;
	county = count;
}

void Plot2D::setMinMax(double xmin, double xmax, double xstep, double ymin, double ymax, double ystep)
{
	setMinMaxX(xmin, xmax, (int)xstep);
	setMinMaxY(ymin, ymax, (int)ystep);
	createPlot();
}

Plot2D::Plot2D(Size graph_size, double xmin, double xmax, double xstep, double ymin, double ymax, double ystep)
{
	size = graph_size;
	setMinMax(xmin, xmax, xstep, ymin, ymax, ystep);
}

void Plot2D::add(int x, int y, double val)
{
	//cout<<(x-minx) / stepx<< ","<<X<<","<<(y-miny) / stepy<<","<<Y<<","<<w<<","<<h<<","<<val<<endl;
	graphBase.at<double>(y, x) = val;
	//getchar();
}
void Plot2D::writeGraph(bool isColor, int arg_min_max, double minvalue, double maxvalue, bool isMinMaxSet)
{
	Mat temp;
	resize(graphBase, temp, size, 0, 0, cv::INTER_LINEAR);

	double minv, maxv;
	Point minp;
	Point maxp;

	minMaxLoc(graphBase, &minv, &maxv, &minp, &maxp);
	/*cout<<minv<<","<<maxv<<endl;
	cout<<maxp<<endl;
	cout<<minp<<endl;*/

	minMaxLoc(temp, &minv, &maxv, &minp, &maxp);

	/*cout<<minv<<","<<maxv<<endl;
	cout<<maxp<<endl;
	cout<<minp<<endl;*/

	if (isMinMaxSet)
	{
		minv = minvalue;
		maxv = maxvalue;
	}

	temp -= minv;
	Mat graphG;
	temp.convertTo(graphG, CV_8U, 255.0 / (maxv - minv));

	if (isColor)
	{
		applyColorMap(graphG, graph, 2);
		if (arg_min_max > 0)
			circle(graph, maxp, 5, Scalar(255, 255, 255));
		else
			drawPlus(graph, minp, 8, Scalar(0, 0, 0));
	}
	else
	{
		cvtColor(graphG, graph, CV_GRAY2BGR);
		if (arg_min_max > 0)
			circle(graph, maxp, 5, Scalar(0, 0, 0));
		else
			drawPlus(graph, minp, 8, Scalar(255, 255, 255));
	}

	flip(graph, graph, 0);
}
void Plot2D::setLabel(string namex, string namey)
{
	copyMakeBorder(graph, show, 0, 50, 50, 0, BORDER_CONSTANT, Scalar(255, 255, 255));

	putText(show, namex, Point(show.cols / 2 - 50, graph.rows + 30), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));
	putText(show, format("%.2f", minx), Point(50, graph.rows + 30), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

	double minv, maxv;
	Point minp;
	Point maxp;
	minMaxLoc(graphBase, &minv, &maxv, &minp, &maxp);
	putText(show, format("%.2f", maxv), Point(400, graph.rows + 30), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

	Mat text = ~Mat::zeros(Size(50 + graph.cols, 50), CV_8UC3);

	putText(text, namey, Point(80, 20), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

	Mat a = text.t();
	flip(a, a, 0);

	a.copyTo(show(Rect(0, 0, a.cols, a.rows)));
}

/*
void Plot2D::plot(CSV& result, vector<ExperimentalParameters>& parameters)
{
string wname = "plot2D";
namedWindow(wname);
for (int i = 0; i < result.data.size(); i++)
{
if (result.data[i][2] == 6)
{
int x = parameters[0].getDiscreteValueIndex(result.data[i][0]);
int y = parameters[1].getDiscreteValueIndex(result.data[i][1]);
this->add(x, y, result.data[i][3]);
}
}


int key = 0;

int minvalue = 30; createTrackbar("min", wname, &minvalue, 100);
int maxvalue = 50; createTrackbar("max", wname, &maxvalue, 100);

int xc = 0; createTrackbar("x", wname, &xc, result.width);
int yc = 1; createTrackbar("y", wname, &yc, result.width);
int zc = 3; createTrackbar("z", wname, &zc, result.width);

vector<int> p(parameters.size());

for (int i = 0; i < parameters.size(); i++)
{
p[i] = parameters[i].getDiscreteValue(parameters[i].value);
createTrackbar(parameters[i].name, wname, &p[i], parameters[i].maxvalTrackbar);
}


bool isColor = true;
bool isMinMaxSet = false;

result.initFilter();
while (key != 'q')
{
setMinMax(parameters[xc].discreteValue[xc], parameters[xc].discreteValue[parameters[xc].maxvalTrackbar], parameters[xc].maxvalTrackbar + 1, parameters[yc].discreteValue[yc], parameters[yc].discreteValue[parameters[yc].maxvalTrackbar], parameters[yc].maxvalTrackbar + 1);

result.filterClear();
for (int i = 0; i < parameters.size(); i++)
{
if (i != xc && i != yc)
{
result.makeFilter(i, parameters[i].discreteValue[p[i]]);
//cout<<parameters[i].name<<","<<parameters[i].discreteValue[p[i]]<<endl;
}
}

for (int i = 0; i < result.data.size(); i++)
{
if (result.filter[i])
{
int x = parameters[xc].getDiscreteValueIndex(result.data[i][xc]);
int y = parameters[yc].getDiscreteValueIndex(result.data[i][yc]);
add(x, y, result.data[i][zc]);
}
}

writeGraph(isColor, PLOT_ARG_MAX, minvalue, maxvalue, isMinMaxSet);
setLabel(parameters[xc].name, parameters[yc].name);
imshow(wname, show);
key = waitKey(1);
if (key == 'c')
{
isColor = (isColor) ? false : true;
}
}
}
*/
}