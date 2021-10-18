#pragma once
#include <plot.hpp>
#include <cmath>
#include <string>
#include <iostream>

namespace cp
{
	class Search1DInt
	{
		int xmin;
		int xmax;
		int x1;
		int x2;
		const double phi = (1.0 + sqrt(5.0)) * 0.5;//for golden search
		int research = 2;

		virtual double getError(int x) = 0;
		inline int getGSLeft();
		inline int getGSRight();

	public:
		inline void plotError(std::string wname, const int search_min, const int search_max);
		inline int linearSearch(const int search_min, const int search_max);
		inline int goldenSearch(const int search_min, const int search_max);
	};

	class Search1D64F
	{
		double xmin;
		double xmax;
		double x1;
		double x2;
		const double phi = (1.0 + sqrt(5.0)) * 0.5;//for golden search

		virtual double getError(double x) = 0;
		inline double getGSLeft();
		inline double getGSRight();

	public:
		inline double linearSearch(const double search_min, const double search_max, const double step, bool isPlot = false, std::string wname = "search");
		inline double binarySearch(const double search_min, const double search_max, const double eps, const int loop);
		inline double goldenSectionSearch(const double search_min, const double search_max, const double eps, const int loop);
	};

	class Search1D32F
	{
		float xmin;
		float xmax;
		float x1;
		float x2;
		const float phi = (float)((1.0 + sqrt(5.0)) * 0.5);//for golden search

		virtual float getError(float x) = 0;
		inline float getGSLeft();
		inline float getGSRight();

	public:
		inline float linearSearch(const float search_min, const float search_max, const float step, bool isPlot = false, std::string wname = "search");
		inline float binarySearch(const float search_min, const float search_max, const float eps, const int loop);
		inline float goldenSectionSearch(const float search_min, const float search_max, const float eps, const int loop);
	};

#pragma region Search1DInt

	inline int Search1DInt::getGSLeft()
	{
		return (int)floor((xmax - xmin) * 1.0 / (1.0 + phi) + xmin);
	}

	inline int Search1DInt::getGSRight()
	{
		return (int)floor((xmax - xmin) * phi / (1.0 + phi) + xmin);
	}

	inline int Search1DInt::linearSearch(const int search_min, const int search_max)
	{
		int ret = search_min;
		//cp::Plot pt;
		double Emax = DBL_MAX;
		for (int i = search_min; i <= search_max; i++)
		{
			double e = getError(i);
			if (e < Emax)
			{
				Emax = e;
				ret = i;
			}
			//pt.push_back(i, e);
		}
		//pt.plot();
		return ret;
	}

	inline int Search1DInt::goldenSearch(const int search_min, const int search_max)
	{
		if (search_max - search_min <= 2 * research + 1)
		{
			return linearSearch(search_min, search_max);
		}

		xmin = search_min;
		xmax = search_max;
		int ret;
		x1 = getGSLeft();
		x2 = getGSRight();
		double E1 = getError(x1);
		double E2 = getError(x2);

		while (1)
		{
			//print_debug6(x1, x2, E1, E2, xmin, xmax);
			//getchar();
			if (abs(x1 - x2) <= 1 || xmin==xmax)
			{
				if (E2 < E1)
				{
					ret = x2;
				}
				else
				{
					ret = x1;
				}

				ret = linearSearch(std::max(search_min, ret - research), std::min(ret + research, search_max));
				break;
			}

			if (E1 > E2)
			{
				xmin = x1;

				x1 = x2;
				E1 = E2;
				x2 = getGSRight();
				E2 = getError(x2);
			}
			else
			{
				xmax = x2;

				x2 = x1;
				E2 = E1;
				x1 = getGSLeft();
				E1 = getError(x1);
			}
		}

		return ret;
	}

	inline void Search1DInt::plotError(std::string wname, const int search_min, const int search_max)
	{
		cp::Plot pt;
		int ret = search_min;
		double Emax = DBL_MAX;
		for (int i = search_min; i <= search_max; i++)
		{
			const double e = getError(i);
			if (e < Emax)
			{
				Emax = e;
				ret = i;
			}
			std::cout << i << "," << e << std::endl;
			pt.push_back(i, e);
		}
		pt.plot(wname);
	}

#pragma endregion

#pragma region Search1D64F

	inline double Search1D64F::getGSLeft()
	{
		return (xmax - xmin) * 1.0 / (1.0 + phi) + xmin;
	}

	inline double Search1D64F::getGSRight()
	{
		return (xmax - xmin) * phi / (1.0 + phi) + xmin;
	}

	inline double Search1D64F::linearSearch(const double search_min, const double search_max, const double step, bool isPlot, std::string wname)
	{
		double ret = search_min;
		//cp::Plot pt;
		double Emax = DBL_MAX;
		for (double i = search_min; i <= search_max; i += step)
		{
			double e = getError(i);
			if (e < Emax)
			{
				Emax = e;
				ret = i;
			}
			//if (isPlot) pt.push_back(i, e, 0);
		}
		//if (isPlot)pt.plot(wname, false);
		return ret;
	}

	inline double Search1D64F::goldenSectionSearch(const double search_min, const double search_max, const double eps, const int loop)
	{
		xmin = search_min;
		xmax = search_max;
		double ret;
		x1 = getGSLeft();
		x2 = getGSRight();
		double E1 = getError(x1);
		double E2 = getError(x2);

		for (int i = 0; i < loop; i++)
		{
			//cout <<rmin<<","<<rmax<<","<< r1 << "," << r2 << endl; getchar();
			if (abs(x1 - x2) < eps)
			{
				break;
			}

			if (E1 > E2)
			{
				xmin = x1;

				x1 = x2;
				E1 = E2;
				x2 = getGSRight();
				E2 = getError(x2);
			}
			else
			{
				xmax = x2;

				x2 = x1;
				E2 = E1;
				x1 = getGSLeft();
				E1 = getError(x1);
			}
		}

		if (E2 < E1)
		{
			ret = x2;
		}
		else
		{
			ret = x1;
		}

		return ret;
	}

	inline double Search1D64F::binarySearch(const double search_min, const double search_max, const double eps, const int loop)
	{
		xmin = search_min;
		xmax = search_max;
		double ret=0.0;
		double E1 = getError(xmin);
		double E2 = getError(xmax);

		for (int i = 0; i < loop; i++)
		{
			ret = (xmin + xmax) / 2.0;
			double E = getError(ret);

			if (E1 < E2)
			{
				E2 = E;
				xmax = ret;
			}
			else
			{
				E1 = E;
				xmin = ret;
			}
			if (abs(xmax - xmin) < eps)
			{
				break;
			}
		}

		return ret;
	}

#pragma endregion

#pragma region Search1D32F

	inline float Search1D32F::getGSLeft()
	{
		return (xmax - xmin) * 1.f / (1.f + phi) + xmin;
	}

	inline float Search1D32F::getGSRight()
	{
		return (xmax - xmin) * phi / (1.f + phi) + xmin;
	}

	inline float Search1D32F::linearSearch(const float search_min, const float search_max, const float step, bool isPlot, std::string wname)
	{
		float ret = search_min + step;
		//cp::Plot pt;
		float Emax = getError(search_min + step);
		//float ep = Emax;
		//FILE* fp = fopen("hat", "w");
		//FILE* fp = fopen("exp_lp", "w");
		//FILE* fp = fopen("Gauss", "w");
		for (float i = search_min + step; i <= search_max; i += step)
		{
			const float e = getError(i);
			if (e < Emax)
			{
				Emax = e;
				ret = i;
			}
			//fprintf(fp, "%f %f %f\n", i, e, ep - e);
			//ep = e;
			//if (isPlot) pt.push_back(i, e, 0);
		}
		//fclose(fp);
		//std::cout << ret << "," << Emax << std::endl;
		//getchar();
		//if (isPlot)pt.plot(wname, false);
		return ret;
	}

	inline float Search1D32F::goldenSectionSearch(const float search_min, const float search_max, const float eps, const int loop)
	{
		xmin = search_min;
		xmax = search_max;
		float ret;
		x1 = getGSLeft();
		x2 = getGSRight();
		float E1 = getError(x1);
		float E2 = getError(x2);

		for (int i = 0; i < loop; i++)
		{
			//cout <<rmin<<","<<rmax<<","<< r1 << "," << r2 << endl; getchar();
			if (abs(x1 - x2) < eps)
			{
				break;
			}

			if (E1 > E2)
			{
				xmin = x1;

				x1 = x2;
				E1 = E2;
				x2 = getGSRight();
				E2 = getError(x2);
			}
			else
			{
				xmax = x2;

				x2 = x1;
				E2 = E1;
				x1 = getGSLeft();
				E1 = getError(x1);
			}
		}

		if (E2 < E1)
		{
			ret = x2;
		}
		else
		{
			ret = x1;
		}

		return ret;
	}

	inline float Search1D32F::binarySearch(const float search_min, const float search_max, const float eps, const int loop)
	{
		xmin = search_min;
		xmax = search_max;
		float ret = search_min;
		float E1 = getError(xmin);
		float E2 = getError(xmax);

		for (int i = 0; i < loop; i++)
		{
			ret = (xmin + xmax) / 2.f;
			float E = getError(ret);

			if (E1 < E2)
			{
				E2 = E;
				xmax = ret;
			}
			else
			{
				E1 = E;
				xmin = ret;
			}
			if (abs(xmax - xmin) < eps)
			{
				break;
			}
		}

		return ret;
	}

#pragma endregion
}