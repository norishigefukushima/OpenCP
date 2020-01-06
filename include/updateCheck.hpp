#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT UpdateCheck
	{
		std::vector<double> parameters;
	public:

		UpdateCheck(double p0);
		UpdateCheck(double p0, double p1);
		UpdateCheck(double p0, double p1, double p2);
		UpdateCheck(double p0, double p1, double p2, double p3);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5);

		bool isUpdate(double p0);
		bool isUpdate(double p0, double p1);
		bool isUpdate(double p0, double p1, double p2);
		bool isUpdate(double p0, double p1, double p2, double p3);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5);
	};
}
