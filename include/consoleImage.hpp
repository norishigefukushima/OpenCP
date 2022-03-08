#pragma once

#include "common.hpp"
#include "imshowExtension.hpp"
namespace cp
{
	class CP_EXPORT ConsoleImage
	{
	private:
		int count;
		std::string windowName;
		StackImage si;
		std::vector<std::string> strings;
		bool isLineNumber;
		std::string fontName;
		int fontSize;
		int lineSpaceSize;
	public:
		void setFont(std::string fontName = "Consolas");
		void setFontSize(int size);
		void setLineSpaceSize(int size);
		void setWindowName(std::string wname);
		void setImageSize(cv::Size size);
		void setIsLineNumber(bool isLine = true);
		bool getIsLineNumber();
		cv::Mat image;

		void init(cv::Size size, std::string wname, const bool isNamedWindow = true);
		ConsoleImage();
		ConsoleImage(cv::Size size, std::string wname = "console", const bool isNamedWindow = true);
		~ConsoleImage();

		void printData();
		void clear();

		void operator()(std::string str);
		void operator()(const char* format, ...);
		void operator()(cv::Scalar color, std::string str);
		void operator()(cv::Scalar color, const char* format, ...);

		void show(bool isClear = true);
		void push();
	};
}