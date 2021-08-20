#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT Stat
	{
	public:
		std::vector<double> data;
		int getSize();//get data size

		double getSum();
		double getMin();
		double getMax();
		double getMean();
		double getVar();
		double getStd();
		double getMedian();

		void pop_back();
		void push_back(double val);

		void clear();//clear data
		void print();//print all stat
		void drawDistribution(std::string wname = "Stat distribution", int div = 100);
		void drawDistribution(std::string wname, int div, double minv, double maxv);
		void drawPlofilePlot(std::string wname);
	};
}