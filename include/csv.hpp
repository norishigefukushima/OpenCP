#pragma once
#include "common.hpp"

namespace cp
{
	class CP_EXPORT CSV
	{
		FILE* fp = NULL;
		bool isTop;
		size_t fileSize;
		std::string filename;
		bool isCommaSeparator = true;
	public:
		std::vector<double> argMin;
		std::vector<double> argMax;
		std::vector<std::vector<double>> data;
		std::vector<bool> filter;
		int width;
		void setSeparator(bool flag);//true:","comma, false, " "space
		void findMinMax(int result_index, bool isUseFilter, double minValue, double maxValue);
		void initFilter();
		void filterClear();
		void makeFilter(int index, double val, double emax = 0.00000001);
		void readHeader();
		void readData();
		cv::Mat getMat();
		void readDataLineByLine();

		void init(std::string name, bool isWrite, bool isClear, bool isHeader);
		CSV();
		CSV(std::string name, bool isWrite = true, bool isClear = true, bool isHeader=true);
		~CSV();
		void write(std::string v);
		void write(double v);
		void write(int v);
		void end();//endl;
	};

	void CP_EXPORT writeCSV(std::string name, cv::InputArray src);
}