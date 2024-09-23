#pragma once
#include "common.hpp"

namespace cp
{
	constexpr int CSV_READLINE_MAX = USHRT_MAX;
	class CP_EXPORT CSV
	{
		FILE* fp = nullptr;
		bool isTop = true;
		size_t fileSize = 0;
		std::string filename = "";
		bool isCommaSeparator = true;
	public:
		std::vector<double> argMin;
		std::vector<double> argMax;
		std::vector<std::vector<std::string>> datastr;
		std::vector<std::vector<double>> data;
		std::vector<std::string> firstlabel;
		std::vector<bool> filter;
		int width = 0;
		void setSeparator(bool flag);//true:","comma, false, " "space
		void findMinMax(int result_index, bool isUseFilter, double minValue, double maxValue);
		void initFilter();
		void filterClear();
		void makeFilter(int index, double val, double emax = 0.00000001);
		void readHeader();
		void readData();
		cv::Mat getMat();
		int readDataLineByLine(const bool isFirstString = false);//return number of lines
		int readDataLineByLineAsString();//return number of lines

		void init(std::string name, bool isWrite, bool isClear, bool isHeader, bool isBinary);
		CSV();
		CSV(std::string name, bool isWrite = true, bool isClear = true, bool isHeader = true, bool isBinary = false);
		~CSV();
		void write(std::string v);
		void write(double v);
		void write(int v);
		void end();//endl;
		void dumpDataAsBinary(std::string name, int depth = CV_32F);//write data as binary for convert csv to bin
	};

	void CP_EXPORT writeCSV(std::string name, cv::InputArray src);
}