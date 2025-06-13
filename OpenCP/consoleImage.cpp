#include "consoleImage.hpp"
#include <stdarg.h>

using namespace std;
using namespace cv;

namespace cp
{
	void ConsoleImage::setFont(std::string fontName)
	{
		this->fontName = fontName;
	}

	void ConsoleImage::setFontSize(int size)
	{
		fontSize = size;
	}

	void ConsoleImage::setLineSpaceSize(int size)
	{
		lineSpaceSize = size;
	}

	void ConsoleImage::setWindowName(string wname)
	{
		destroyWindow(wname);
		this->windowName = wname;
	}

	void ConsoleImage::setImageSize(Size size)
	{
		image = Mat::zeros(size, CV_8UC3);
	}

	void ConsoleImage::init(Size size, string wname, const bool isNamedWindow)
	{
		if (isNamedWindow) namedWindow(wname);
		isLineNumber = false;
		windowName = wname;
		image = Mat::zeros(size, CV_8UC3);
		fontName = "Consolas";
		fontSize = 20;
		lineSpaceSize = 5;
		si.setWindowName(windowName);
		clear();
	}

	ConsoleImage::ConsoleImage()
	{
		init(Size(640, 480), "console", true);
	}

	ConsoleImage::ConsoleImage(Size size, string wname, const bool isNamedWindow)
	{
		init(size, wname, isNamedWindow);
	}

	ConsoleImage::~ConsoleImage()
	{
		destroyWindow(windowName);
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
		count_page = 0;
		count = 0;
		image.setTo(0);
		strings.clear();
		si.clear();
	}

	void ConsoleImage::push()
	{
		si.push(image);
	}

	void ConsoleImage::show(bool isClear)
	{
		//namedWindow(windowName);
		si.show(image, true);
		if (isClear) clear();
	}

	void ConsoleImage::operator()(string src)
	{
		this->operator()(Scalar(255, 255, 255), src);
	}

	void ConsoleImage::operator()(const char* format, ...)
	{
		char buff[255];
		va_list ap;
		va_start(ap, format);
		vsprintf(buff, format, ap);
		va_end(ap);
		string a = buff;

		this->operator()(Scalar(255, 255, 255), a);
	}

	void ConsoleImage::operator()(cv::Scalar color, const char* format, ...)
	{
		char buff[255];
		va_list ap;
		va_start(ap, format);
		vsprintf(buff, format, ap);
		va_end(ap);
		string a = buff;

		this->operator()(color, a);
	}

	void ConsoleImage::operator()(cv::Scalar color, string src)
	{
		if (isLineNumber)strings.push_back(cv::format("%3d ", count) + src);
		else strings.push_back(src);

		const int skip = fontSize + lineSpaceSize;
		const int y = skip + count_page * skip;
		if (y > image.rows)
		{
			push();
			count_page = 0;
			image.setTo(0);
			const int y = skip + count_page * skip;
			cv::addText(image, strings[count], Point(skip, y), fontName, fontSize, color);
		}
		else
		{
			cv::addText(image, strings[count], Point(skip, y), fontName, fontSize, color);
		}
		//cv::putText(image, strings[count], Point(skip, skip + count * skip), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1);

		count_page++;
		count++;
	}
}