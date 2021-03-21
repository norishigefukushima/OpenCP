#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT UpdateCheck
	{
		bool isSkip = true;
		std::vector<double> parameters;
		bool firstTimeCheck(const bool flag);
	public:
		bool isFourceRetTrueFirstTime = true;
		void setIsFourceRetTrueFirstTime(const bool flag);
		UpdateCheck(double p0);
		UpdateCheck(double p0, double p1);
		UpdateCheck(double p0, double p1, double p2);
		UpdateCheck(double p0, double p1, double p2, double p3);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10);


		bool isUpdate(double p0);
		bool isUpdate(double p0, double p1);
		bool isUpdate(double p0, double p1, double p2);
		bool isUpdate(double p0, double p1, double p2, double p3);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10);
	};
}
