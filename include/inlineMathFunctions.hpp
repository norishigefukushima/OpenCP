#pragma once

namespace cp
{
	inline double cubic(double x, double a)
	{
		double d = abs(x);
		if (0 <= d && d < 1)
		{
			return (a + 2.0) * d * d * d - (a + 3.0) * d * d + 1.0;
		}
		else if (1 <= d && d < 2)
		{
			double v = d * d * d - 5.0 * d * d + 8.0 * d - 4.0;
			v *= a;
			return v;
		}
		else
			return 0.0;
	}

	inline float cubic(const float x, const float a)
	{
		float d = abs(x);
		float ret = 0.f;

		if (0 <= d && d < 1)
		{
			ret = (a + 2.f) * d * d * d - (a + 3.f) * d * d + 1.f;
		}
		else if (1 <= d && d < 2)
		{
			ret = d * d * d - 5.f * d * d + 8.f * d - 4.f;
			ret *= a;
		}

		return ret;
	}

	inline float sinc(float x)
	{
		if (x == 0.f) return 1.f;
		const float x_arg = (float)(CV_PI * x);
		return sin(x_arg) / x_arg;
	}

	inline float lanczos(const float x, const float n)
	{
		if (abs(x) <= n) return sinc(x) * sinc(x / n);
		else return 0.f;
	}

	inline int sign(int x)
	{
		if (x >= 0)return 1;
		else return 0;
	}

	inline int ceilToMultiple(const int x, const int multiple)
	{
		int v = abs(x);
		return (v % multiple == 0) ? sign(x) * v : sign(x) * (v / multiple + 1) * multiple;
	}

	inline int floorToMultiple(const int x, const int multiple)
	{
		int v = abs(x);
		return sign(x) * (v / multiple) * multiple;
	}

	inline int countCircleArea(const int r)
	{
		int count = 0;
		for (int j = -r; j <= r; j++)
		{
			for (int i = -r; i <= r; i++)
			{
				int d = cvRound(sqrt((float)i * i + (float)j * j));
				if (d <= r)count++;
			}
		}
		return count;
	}

	inline cv::Mat convert(cv::Mat& src, const int depth, const double alpha = 1.0, const double beta = 0.0)
	{
		cv::Mat ret;
		src.convertTo(ret, depth, alpha, beta);
		return ret;
	}

	inline double PSNRBB(cv::Mat& src, cv::Mat& dest, int bb)
	{
		cv::Rect roi = cv::Rect(bb, bb, src.cols - 2 * bb, src.rows - 2 * bb);
		return cv::PSNR(src(roi), dest(roi));
	}

	inline void absdiffScale(cv::Mat& src, cv::Mat& ref, cv::Mat& dest, const double scale, const int depth = CV_8U)
	{
		cv::Mat subf;
		cv::subtract(src, ref, subf, cv::noArray(), CV_64F);
		subf = cv::abs(subf) * scale;
		subf.convertTo(dest, depth);
	}

	inline double MSEtoPSNR(const double mse)
	{
		if (mse == 0.0)
		{
			return 0;
		}
		else if (cvIsNaN(mse))
		{
			return -1.0;
		}
		else if (cvIsInf(mse))
		{
			return -2.0;
		}
		else
		{
			return 10.0 * log10(255.0 * 255.0 / mse);
		}
	}

	inline std::string getDepthName(int depth)
	{
		std::string ret;
		switch (depth)
		{
		case CV_8U: ret = "CV_8U"; break;
		case CV_8S:ret = "CV_8S"; break;
		case CV_16U:ret = "CV_16U"; break;
		case CV_16S:ret = "CV_16S"; break;
		case CV_32S:ret = "CV_32S"; break;
		case CV_32F:ret = "CV_32F"; break;
		case CV_64F:ret = "CV_64F"; break;
		case CV_16F:ret = "CV_16F"; break;
		default: ret = "not support this type of depth."; break;
		}
		return ret;
	}
}


#define print_debug(a)              std::cout << #a << ": " << a << std::endl
#define print_debug2(a, b)          std::cout << #a << ": " << a <<", "<< #b << ": " << b << std::endl
#define print_debug3(a, b, c)       std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c << std::endl;
#define print_debug4(a, b, c, d)    std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d << std::endl;
#define print_debug5(a, b, c, d, e) std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e << std::endl;
