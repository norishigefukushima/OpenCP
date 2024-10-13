#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT UpdateCheck
	{
		bool isSkip = true;
		std::vector<double> prevparameters;
		std::vector<double> parameters;
		bool firstTimeCheck(const bool flag);
	public:
		void push();
		void pop();
		bool isFourceRetTrueFirstTime = true;
		void setIsFourceRetTrueFirstTime(const bool flag);
		UpdateCheck();
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
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16);

		void init(double p0);
		void init(double p0, double p1);
		void init(double p0, double p1, double p2);
		void init(double p0, double p1, double p2, double p3);
		void init(double p0, double p1, double p2, double p3, double p4);
		void init(double p0, double p1, double p2, double p3, double p4, double p5);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15);
		void init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16);

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
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16);

		void get(double& p0);
		void get(double& p0, double& p1);
		void get(double& p0, double& p1, double& p2);
		void get(double& p0, double& p1, double& p2, double& p3);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13, double& p14);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13, double& p14, double& p15);
		void get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13, double& p14, double& p15, double& p16);
		void get(int& p0);
		void get(int& p0, int& p1);
		void get(int& p0, int& p1, int& p2);
		void get(int& p0, int& p1, int& p2, int& p3);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13, int& p14);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13, int& p14, int& p15);
		void get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13, int& p14, int& p15, int& p16);


		void print(int num_args);
	};
}
