#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT Stat
	{
	public:
		std::vector<double> data;
		int num_data = 0;

		double getMin();
		double getMax();
		double getMean();
		double getStd();
		double getMedian();

		void push_back(double val);

		void clear();
		void print();
		void drawDistribution(std::string wname = "Stat distribution", int div = 100);
		void drawDistribution(std::string wname, int div, double minv, double maxv);
	};
}