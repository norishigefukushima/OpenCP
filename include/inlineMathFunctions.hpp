#pragma once

namespace cp
{
	inline double cubic(double x, double a)
	{
		double d = abs(x);
		if (0 <= d && d < 1)
		{
			return (a + 2.0)*d*d*d - (a + 3.0)*d*d + 1.0;
		}
		else if (1 <= d && d < 2)
		{
			double v = d * d*d - 5.0*d*d + 8.0*d - 4.0;
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
			ret = (a + 2.f)*d*d*d - (a + 3.f)*d*d + 1.f;
		}
		else if (1 <= d && d < 2)
		{
			ret = d * d*d - 5.f * d*d + 8.f*d - 4.f;
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

	inline cv::Mat convert(cv::Mat& src, const int depth)
	{
		cv::Mat ret;
		src.convertTo(ret, depth);
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
		subf = cv::abs(subf)* scale;
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
			return 10.0 * log10(255.0*255.0 / mse);
		}
	}
}