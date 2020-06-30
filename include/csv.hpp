#pragma once
#include "common.hpp"

namespace cp
{
	class CP_EXPORT CSV
	{
		FILE* fp = NULL;
		bool isTop;
		long fileSize;
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

		void init(std::string name, bool isWrite, bool isClear);
		CSV();
		CSV(std::string name, bool isWrite = true, bool isClear = true);
		~CSV();
		void write(std::string v);
		void write(double v);
		void write(int v);
		void end();
	};

	void CP_EXPORT writeCSV(std::string name, cv::InputArray src);
}