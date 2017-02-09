#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT Stat
	{
	public:
		std::vector<double> data;
		int num_data;
		Stat();
		~Stat();
		double getMin();
		double getMax();
		double getMean();
		double getStd();
		double getMedian();

		void push_back(double val);

		void clear();
		void show();
		void showDistribution();
	};

}