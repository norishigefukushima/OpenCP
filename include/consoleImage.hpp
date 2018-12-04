#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT ConsoleImage
	{
	private:
		int count;
		std::string windowName;
		std::vector<std::string> strings;
		bool isLineNumber;
		int fontSize;
		int lineSpaceSize;
	public:
		void setFontSize(int size);
		void setLineSpaceSize(int size);

		void setIsLineNumber(bool isLine = true);
		bool getIsLineNumber();
		cv::Mat show;

		void init(cv::Size size, std::string wname);
		ConsoleImage();
		ConsoleImage(cv::Size size, std::string wname = "console");
		~ConsoleImage();

		void printData();
		void clear();

		void operator()(std::string src);
		void operator()(const char *format, ...);
		void operator()(cv::Scalar color, const char *format, ...);

		void flush(bool isClear = true);
	};
}