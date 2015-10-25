#include "opencp.hpp"
#include <stdarg.h>

using namespace std;
using namespace cv;

namespace cp
{


void ConsoleImage::init(Size size, string wname)
{
	isLineNumber = false;
	windowName = wname;
	show = Mat::zeros(size, CV_8UC3);
	clear();
}
ConsoleImage::ConsoleImage()
{
	init(Size(640, 480), "console");
}
ConsoleImage::ConsoleImage(Size size, string wname)
{
	init(size, wname);
}
ConsoleImage::~ConsoleImage()
{
	printData();
}
void ConsoleImage::setIsLineNumber(bool isLine)
{
	isLineNumber = isLine;
}

bool ConsoleImage::getIsLineNumber()
{
	return isLineNumber;
}
void ConsoleImage::printData()
{
	for (int i = 0; i < (int)strings.size(); i++)
	{
		cout << strings[i] << endl;
	}
}
void ConsoleImage::clear()
{
	count = 0;
	show.setTo(0);
	strings.clear();
}
void ConsoleImage::flush(bool isClear)
{
	imshow(windowName, show);
	if (isClear)clear();
}
void ConsoleImage::operator()(string src)
{
	if (isLineNumber)strings.push_back(format("%2d ", count) + src);
	else strings.push_back(src);

	cv::putText(show, src, Point(20, 20 + count * 20), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}
void ConsoleImage::operator()(const char *format, ...)
{
	char buff[255];

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;

	if (isLineNumber)strings.push_back(cv::format("%2d ", count) + a);
	else strings.push_back(a);

	cv::putText(show, buff, Point(20, 20 + count * 20), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}

void ConsoleImage::operator()(cv::Scalar color, const char *format, ...)
{
	char buff[255];

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;
	if (isLineNumber)strings.push_back(cv::format("%2d ", count) + a);
	else strings.push_back(a);
	cv::putText(show, buff, Point(20, 20 + count * 20), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}
}