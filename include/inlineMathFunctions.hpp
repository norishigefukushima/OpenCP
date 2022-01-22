#pragma once
#include <opencv2/core/cvdef.h>
#include <cmath>
namespace cp
{
	inline void indexToXY(const int index, const int imstep, int& x, int& y)
	{
		x = index % imstep;
		y = index / imstep;
	}

	inline void relativeIndexToXY(const int index, const int imstep, const int r, int& x, int& y)
	{
		x = (index + r * imstep + r) % imstep;
		y = (index + r * imstep + r) / imstep;
	}

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

	inline double sign(double x)
	{
		if (x >= 0.0)return 1.0;
		else return -1.0;
	}

	inline float sign(float x)
	{
		if (x >= 0.f)return 1.f;
		else return -1.f;
	}

	inline int sign(int x)
	{
		if (x >= 0)return 1;
		else return -1;
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

	inline int setzeroCircleOutside(cv::Mat& kernel)
	{
		cv::Size kernelSize = kernel.size();
		int ret = 0;
		const int r = kernelSize.width / 2;
		for (int j = 0; j < kernelSize.height; j++)
		{
			for (int i = 0; i < kernelSize.width; i++)
			{
				if (r < cvRound(sqrt((j - r) * (j - r) + (i - r) * (i - r))))
				{
					if (kernel.at<uchar>(j, i) != 0)
					{
						kernel.at<uchar>(j, i) = 0;
						ret++;
					}
				}
			}
		}
		return ret;
	}

	inline std::vector<cv::Point> getCircleIndex(const int r)
	{
		std::vector<cv::Point> ret;
		const int s = 2 * r + 1;
		for (int j = 0; j < s; j++)
		{
			for (int i = 0; i < s; i++)
			{
				int d = cvRound(sqrt((j - r) * (j - r) + (i - r) * (i - r)));
				if (d <= r)ret.push_back(cv::Point(i, j));
			}
		}
		return ret;
	}

	inline void setCircleMask(cv::Mat& kernel, cv::Size kernelSize, bool isOuterMask = true)
	{
		const int r = kernelSize.width / 2;
		const int rthresh = r;
		if (isOuterMask)
		{
			kernel = cv::Mat::zeros(kernelSize, CV_8U);

			for (int j = 0; j < kernelSize.height; j++)
			{
				for (int i = 0; i < kernelSize.width; i++)
				{
					int d = cvRound(sqrt((j - r) * (j - r) + (i - r) * (i - r)));
					if (d > rthresh)kernel.at<uchar>(j, i) = 255;
				}
			}
		}
		else
		{
			kernel.create(kernelSize, CV_8U);
			kernel.setTo(255);

			for (int j = 0; j < kernelSize.height; j++)
			{
				for (int i = 0; i < kernelSize.width; i++)
				{
					int d = cvRound(sqrt((j - r) * (j - r) + (i - r) * (i - r)));
					if (d > rthresh)kernel.at<uchar>(j, i) = 0;
				}
			}
		}
	}

	inline bool isInCircle(cv::Point pt, cv::Size kernelSize)
	{
		const int r = kernelSize.width / 2;
		const int dist = cvRound(sqrt((pt.x - r) * (pt.x - r) + (pt.y - r) * (pt.y - r)));
		if (dist > r)
		{
			return false;
		}
		return true;
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
}

inline double getHannWindow(const double x)
{
	return (x <= 0.5) ? 0.5 + 0.5 * cos(CV_2PI * x) : 0;
}

inline double getHammingWindow(const double x)
{
	return (x <= 0.5) ? 0.54 + 0.46 * cos(CV_2PI * x) : 0;
}

inline double getBlackmanWindow(const double x)
{
	return (x <= 0.5) ? 0.42 + 0.5 * cos(CV_2PI * x) + 0.08 * cos(CV_2PI * 2.0 * x) : 0;
}

inline double getNuttallWindow(const double x)
{
	return (x <= 0.5) ? 0.355768 + 0.487396 * cos(CV_2PI * x) + 0.144232 * cos(CV_2PI * 2.0 * x) + 0.012604 * cos(CV_2PI * 3.0 * x) : 0;
}

inline double getAkaikeWindow(const double x)
{
	return (x <= 0.5) ? 0.625 + 0.5 * cos(CV_2PI * x) - 0.125 * cos(CV_2PI * 2.0 * x) : 0;
}

inline double getParzenWindow(const double x)
{
	double ret = 0.0;
	if (x <= 1.0)
	{
		ret = 1.0 - 1.5 * x * x + 0.75 * x * x * x;
	}
	else if (x <= 2.0)
	{
		ret = 0.25 * (2.0 - x) * (2.0 - x) * (2.0 - x);
	}
	return ret;
}

inline double getWelchWindow(const double x)
{
	return std::max(0.0, 1.0 - x * x);//1-x*x;	
}

inline double getBartlettWindow(const double x)
{
	return std::max(0.0, 1.0 - x);//Hat; 
}

inline double getFlatTopWindow(const double x)
{
	//return (x <= 0.5) ? 1.0 - 1.93*cos(CV_2PI*x + CV_PI) + 1.29*cos(CV_PI*4.0*x + CV_2PI) - 0.388*cos(CV_PI*6.0*x + 3 * CV_PI) + 0.032*cos(CV_PI*8.0*x + 4 * CV_PI) : 0;
	return (x <= 0.5) ? 0.21557895 - 0.41663158 * cos(CV_2PI * x + CV_PI) + 0.277263158 * cos(CV_PI * 4.0 * x + CV_2PI) - 0.083578947 * cos(CV_PI * 6.0 * x + 3 * CV_PI) + 0.006947368 * cos(CV_PI * 8.0 * x + 4 * CV_PI) : 0;
}

inline double sinc(const double x)
{
	return sin(CV_PI * x) / (CV_PI * x);
}

inline double getLanczosWindow(const double x, const int n)
{
	return (x <= n) ? sinc(x) * sinc(x / n) : 0;
}

inline double getExpLpWindow(const double x, const int n)
{
	return std::exp(-pow(x, n) / n);//Ln-Gaussian
}

inline double getGaussianWindow(const double x)
{
	return std::exp(-0.5 * x * x);
}

inline double getDivSqrtWindow(const double x, double sub)
{
	return 1.0 / pow(x * x + 1.0, 0.5) - sub;
	//return 1.0 / pow(abs(x) + 1.0, 0.35);
}
//https://ja.wikipedia.org/wiki/%E7%AA%93%E9%96%A2%E6%95%B0
//https://en.wikipedia.org/wiki/Window_function

enum
{
	GAUSSIAN_WINDOW,
	EXP_L1_WINDOW,//dual exponential, Laplacian distribution
	EXP_L2_WINDOW, //Gaussian window
	EXP_L3_WINDOW,
	EXP_L4_WINDOW,
	EXP_L5_WINDOW,
	EXP_L6_WINDOW,
	EXP_L7_WINDOW,
	EXP_L8_WINDOW,
	EXP_L9_WINDOW,
	EXP_L10_WINDOW,
	EXP_L20_WINDOW,
	EXP_L40_WINDOW,
	EXP_L80_WINDOW,
	EXP_L160_WINDOW,

	BOX_WINDOW,
	BARTLETT_WINDOW,//triangle window
	WELCH_WINDOW,
	PARZEN_WINDOW,//peachwise approximation of Gaussian with cubic
	DIVSQRT_WINDOW,

	HANN_WINDOW,
	HAMMING_WINDOW,
	BLACKMAN_WINDOW,
	NUTTALL_WINDOW,
	AKAIKE_WINDOW,
	FLATTOP_WINDOW,

	WINDOW_TYPE_SIZE
};

inline std::string getWindowTypeName(const int window_type)
{
	std::string ret;
	switch (window_type)
	{
	case BARTLETT_WINDOW:
		ret = "BARTLETT_WINDOW";
		break;
	case WELCH_WINDOW:
		ret = "WELCH_WINDOW";
		break;
	case PARZEN_WINDOW:
		ret = "PARZEN_WINDOW";
		break;
	case DIVSQRT_WINDOW:
		ret = "DIVSQRT_WINDOW";
		break;

	case HANN_WINDOW:
		ret = "HANN_WINDOW";
		break;
	case HAMMING_WINDOW:
		ret = "HAMMING_WINDOW";
		break;
	case BLACKMAN_WINDOW:
		ret = "BLACKMAN_WINDOW";
		break;
	case NUTTALL_WINDOW:
		ret = "NUTTALL_WINDOW";
		break;
	case AKAIKE_WINDOW:
		ret = "AKAIKE_WINDOW";
		break;
	case FLATTOP_WINDOW:
		ret = "FLATTOP_WINDOW";
		break;

	case EXP_L1_WINDOW:
		ret = "EXP_L1_WINDOW";
		break;
	case EXP_L3_WINDOW:
		ret = "EXP_L3_WINDOW";
		break;
	case EXP_L4_WINDOW:
		ret = "EXP_L4_WINDOW";
		break;
	case EXP_L5_WINDOW:
		ret = "EXP_L5_WINDOW";
		break;
	case EXP_L6_WINDOW:
		ret = "EXP_L6_WINDOW";
		break;
	case EXP_L7_WINDOW:
		ret = "EXP_L7_WINDOW";
		break;
	case EXP_L8_WINDOW:
		ret = "EXP_L8_WINDOW";
		break;
	case EXP_L9_WINDOW:
		ret = "EXP_L9_WINDOW";
		break;
	case EXP_L10_WINDOW:
		ret = "EXP_L10_WINDOW";
		break;
	case EXP_L20_WINDOW:
		ret = "EXP_L20_WINDOW";
		break;
	case EXP_L40_WINDOW:
		ret = "EXP_L40_WINDOW";
		break;
	case EXP_L80_WINDOW:
		ret = "EXP_L80_WINDOW";
		break;
	case EXP_L160_WINDOW:
		ret = "EXP_L160_WINDOW";
		break;
	case BOX_WINDOW:
		ret = "BOX_WINDOW";
		break;
	case GAUSSIAN_WINDOW:
	case EXP_L2_WINDOW:
	default:
		ret = "GAUSSIAN_WINDOW";
		break;
	}

	return ret;
}

inline double getRangeKernelFunction(const double x, const double sigma, const int window_type)
{
	const double gaussRangeCoeff = (x / sigma);

	double ret = 0.0;
	switch (window_type)
	{
	case BARTLETT_WINDOW:
		ret = getBartlettWindow(abs(gaussRangeCoeff));//hat
		break;
	case WELCH_WINDOW:
		ret = getWelchWindow(abs(gaussRangeCoeff));//1-x*x;
		break;
	case PARZEN_WINDOW:
		ret = getParzenWindow(abs(gaussRangeCoeff));
		break;
	case DIVSQRT_WINDOW:
	{
		double sub = getDivSqrtWindow(127.5 / sigma, 0.0);
		ret = getDivSqrtWindow(gaussRangeCoeff, sub) / (1.0 - sub);
		break;
	}

	case HANN_WINDOW:
		ret = getHannWindow(abs(gaussRangeCoeff));
		break;
	case HAMMING_WINDOW:
		ret = getHammingWindow(abs(gaussRangeCoeff));
		break;
	case BLACKMAN_WINDOW:
		ret = getBlackmanWindow(abs(gaussRangeCoeff));
		break;
	case NUTTALL_WINDOW:
		ret = getNuttallWindow(abs(gaussRangeCoeff));
		break;
	case AKAIKE_WINDOW:
		ret = getAkaikeWindow(abs(gaussRangeCoeff));
		break;
	case FLATTOP_WINDOW:
		ret = getFlatTopWindow(abs(gaussRangeCoeff));
		break;

	case EXP_L1_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 1));//Laplacian kernel
		break;
	case EXP_L3_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 3) / 3.0);
		break;
	case EXP_L4_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 4) / 4.0);
		break;
	case EXP_L5_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 5) / 5.0);
		break;
	case EXP_L6_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 6) / 6.0);
		break;
	case EXP_L7_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 7) / 7.0);
		break;
	case EXP_L8_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 8) / 8.0);
		break;
	case EXP_L9_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 9) / 9.0);
		break;
	case EXP_L10_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 10) / 10.0);
		break;
	case EXP_L20_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 20) / 20.0);
		break;
	case EXP_L40_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 40) / 40.0);
		break;
	case EXP_L80_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 80) / 80.0);
		break;
	case EXP_L160_WINDOW:
		ret = std::exp(-pow(abs(gaussRangeCoeff), 160) / 160.0);
		break;
	case BOX_WINDOW:
		ret = (abs(x) <= sigma) ? 1.0 : 0.0;
		break;
	case GAUSSIAN_WINDOW:
	case EXP_L2_WINDOW:
	default:
		ret = getGaussianWindow(abs(gaussRangeCoeff));//Gaussian
		break;
	}

	return ret;
}

inline double getRangeKernelIntegral(const double sigma, const int window_type)
{
	double ret = 0.0;
	for (int i = 0; i < 128; i++)
	{
		ret += getRangeKernelFunction(i, sigma, window_type);
	}

	return ret;
}
