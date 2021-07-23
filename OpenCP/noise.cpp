#include "noise.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	template <class srcType>
	static void addNoiseSoltPepperMono_(Mat& src, Mat& dest, const double per, const uint64 seed)
	{
		cv::RNG rng;
		if (seed != 0) rng.state = seed;
		else rng.state = cv::getTickCount();
		for (int j = 0; j < src.rows; j++)
		{
			srcType* s = src.ptr<srcType>(j);
			srcType* d = dest.ptr<srcType>(j);
			for (int i = 0; i < src.cols; i++)
			{
				double a1 = rng.uniform((double)0, (double)1);

				if (a1 > per)
					d[i] = s[i];
				else
				{
					double a2 = rng.uniform((double)0, (double)1);
					if (a2 > 0.5)d[i] = (srcType)0.0;
					else d[i] = (srcType)255.0;
				}
			}
		}
	}

	static void addNoiseSoltPepperMono(Mat& src, Mat& dest, double per, const uint64 seed)
	{
		if (src.depth() == CV_8U) addNoiseSoltPepperMono_<uchar>(src, dest, per, seed);
		if (src.depth() == CV_8S) addNoiseSoltPepperMono_<char>(src, dest, per, seed);
		if (src.depth() == CV_16U) addNoiseSoltPepperMono_<ushort>(src, dest, per, seed);
		if (src.depth() == CV_16S) addNoiseSoltPepperMono_<short>(src, dest, per, seed);
		if (src.depth() == CV_32S) addNoiseSoltPepperMono_<int>(src, dest, per, seed);
		if (src.depth() == CV_32F) addNoiseSoltPepperMono_<float>(src, dest, per, seed);
		if (src.depth() == CV_64F) addNoiseSoltPepperMono_<double>(src, dest, per, seed);
	}

	static void addNoiseMono_int(Mat& src, Mat& dest, double sigma)
	{
		Mat s;
		src.convertTo(s, CV_32S);
		Mat n(s.size(), CV_32S);
		randn(n, 0, sigma);
		add(s, n, dest, noArray(), src.depth());
	}

	static void addNoiseMono_double(Mat& src, Mat& dest, const double sigma)
	{
		Mat s; src.convertTo(s, CV_64F);
		Mat n(s.size(), CV_64F);
		randn(n, 0, sigma);
		add(s, n, dest, noArray(), src.depth());
	}

	static void addNoiseMono(Mat& src, Mat& dest, const double sigma)
	{
		if (src.depth() == CV_32F || src.depth() == CV_64F)
		{
			addNoiseMono_double(src, dest, sigma);
		}
		else
		{
			addNoiseMono_int(src, dest, sigma);
		}
	}

	void addNoise(InputArray src_, OutputArray dest_, const double sigma, const double sprate, const uint64 seed)
	{
		CV_Assert(!src_.empty());
		dest_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dest = dest_.getMat();

		if (seed != 0) cv::theRNG().state = seed;
		
		if (src.channels() == 1)
		{
			addNoiseMono(src, dest, sigma);
			if (sprate != 0)addNoiseSoltPepperMono(dest, dest, sprate, seed);
			return;
		}
		else
		{
			Mat s = src.reshape(1);
			dest = dest.reshape(1);
			addNoiseMono(s, dest, sigma);
			if (sprate != 0)addNoiseSoltPepperMono(dest, dest, sprate, seed);
			dest = dest.reshape(3);
		}
	}


	static void addNoisePoissonMono_int(Mat& src, Mat& dest, double lambda)
	{
		Mat s;
		src.convertTo(s, CV_32S);
		Mat n(s.size(), CV_32S);
		randn(n, 0, lambda);
		add(s, n, dest, noArray(), src.depth());
	}

	static void addNoisePoissonMono_double(Mat& src, Mat& dest, const double lambda)
	{
		Mat s; src.convertTo(s, CV_64F);
		Mat n(s.size(), CV_64F);
		randn(n, 0, lambda);
		add(s, n, dest, noArray(), src.depth());
	}

	static void addNoisePoissonMono(Mat& src, Mat& dest, const double lambda)
	{
		if (src.depth() == CV_32F || src.depth() == CV_64F)
		{
			addNoisePoissonMono_double(src, dest, lambda);
		}
		else
		{
			addNoisePoissonMono_int(src, dest, lambda);
		}
	}
	void addNoisePoisson(InputArray src_, OutputArray dest_, const double lambda, const uint64 seed)
	{
		CV_Assert(!src_.empty());
		dest_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dest = dest_.getMat();

		if (seed != 0) cv::theRNG().state = seed;
		if (src.channels() == 1)
		{
			addNoiseMono(src, dest, lambda);
			return;
		}
		else
		{
			Mat s = src.reshape(1);
			dest = dest.reshape(1);
			addNoiseMono(s, dest, lambda);
			dest = dest.reshape(3);
		}
	}
	
	void addJPEGNoise(InputArray src, OutputArray dest, const int quality)
	{
		std::vector<uchar> buff;
		std::vector<int> param(2);
		param[0] = IMWRITE_JPEG_QUALITY;
		param[1] = quality;
		imencode(".jpg", src, buff, param);
		Mat dst = imdecode(buff, IMREAD_ANYCOLOR);
		dst.copyTo(dest);
	}
}