#pragma once
#include <algorithm>
#include <cmath>
#include "multiscalefilter/MultiScaleFilter.hpp"

class FourierDecomposition
{
	int window_type;
	double T;// period
	double sigma;//parameter sigma
	float beta, alpha, remap_sigma;
	int n;//order

	double omega_n;
	double fcos(double x)
	{
		return getGaussianWindow(abs(x / sigma)) * cos(omega_n * x);
	}

	double fsin(double x)
	{
		switch (window_type)
		{
		case GAUSS:
			//return x * getGaussWeight(x, 0, sigma) * sin(omega_n * x); break;
			return x * getGaussianWindow(abs(x / sigma)) * sin(omega_n * x); break;
		case S_TONE:
			return getSToneCurve<double>(x, 0.0, remap_sigma, beta, alpha) * sin(omega_n * x); break;
		case HAT:
			return x * std::max(0.0, 1.0 - abs(x / sigma)) * sin(omega_n * x); break;
		case SMOOTH_HAT:
			return getSmoothingHat(x, 0.0, sigma, 10) * sin(omega_n * x); break;
		}
		return x * sin(omega_n * x);// getSToneWeight(x, remap_sigma, beta, alpha)* sin(omega_n * x);
	}

	double f(double x)
	{
		switch (window_type)
		{
		case GAUSS:
			return getGaussianWindow(abs(x / sigma)); break;
		case S_TONE:
			return getSToneCurve<double>(x, 0.0, remap_sigma, beta, alpha); break;
		case HAT:
			return std::max(0.0, 1.0 - abs(x / sigma)); break;
		case SMOOTH_HAT:
			return getSmoothingHat(x, 0.0, sigma, 10); break;
		}

		return getSToneWeight(float(x), remap_sigma, beta, alpha);
	}

public:
	FourierDecomposition(double T, double sigma, double beta, double alpha, double remap_sigma, int n, int window_type)
		:T(T), sigma(sigma), n(n), window_type(window_type), beta((float)beta), alpha((float)alpha), remap_sigma((float)remap_sigma)
	{
		omega_n = n * CV_2PI / T;//omega=CV_2PI/T
	}

	//a, b: Integration interval
	//m: number of divisions
	double ct(double a, double b, const int m, bool isKahan = false)//0-T/2
	{
		const double step = (b - a) / m;//interval
		//init
		double x = a;
		double s = 0.0;//result of integral

		if (isKahan)
		{
			double c = 0.0;
			for (int k = 1; k <= m - 1; k++)
			{
				x += step;
				const double y = fcos(x) - c;
				const double t = s + y;
				c = (t - s) - y;
				s = t;
			}
		}
		else
		{
			for (int k = 1; k <= m - 1; k++)
			{
				x += step;
				s += fcos(x);
			}
		}

		s = step * ((fcos(a) + fcos(b)) / 2.0 + s);
		//double sign=0;
		//if (n % 4 == 0)sign = 1.0;
		//if (n % 4 == 2)sign = -1.0;
		//s = step * ((f(0) +sign*f(b)) / 2.0 + s);

		return s;
	}

	//a, b: Integration interval
	//m: number of divisions
	double st(double a, double b, const int m, const bool isKahan = false)
	{
		const double step = (b - a) / m;//interval

		//init
		double x = a;
		double s = 0.0;//result of integral

		if (isKahan)
		{
			double c = 0.0;
			for (int k = 1; k <= m - 1; k++)
			{
				x += step;
				const double y = fsin(x) - c;
				const double t = s + y;
				c = (t - s) - y;
				s = t;
			}
		}
		else
		{
			for (int k = 1; k <= m - 1; k++)
			{
				x += step;
				s += fsin(x);
			}
		}

		s = step * ((fsin(a) + fsin(b)) / 2.0 + s);
		return s;
	}

	//double operator()(double a, double b, const int m)
	double init(double a, double b, const int m, const bool isKahan = false)
	{
		const double step = (b - a) / m;//interval

		//init
		double x = a;
		double s = 0.0;//result of integral

		if (isKahan)
		{
			double c = 0.0;
			for (int k = 1; k <= m - 1; k++)
			{
				x += step;
				const double y = f(x) - c;
				const double t = s + y;
				c = (t - s) - y;
				s = t;
			}
		}
		else
		{
			for (int k = 1; k <= m - 1; k++)
			{
				x += step;
				s += f(x);
			}
		}

		s = step * ((f(a) + f(b)) / 2.0 + s);
		return s;
	}
};