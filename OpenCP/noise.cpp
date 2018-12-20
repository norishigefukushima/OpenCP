#include "noise.hpp"

using namespace std;
using namespace cv;

namespace cp
{

template <class T>
void addNoiseSoltPepperMono_(Mat& src, Mat& dest, const double per, const int seed)
{
	cv::RNG rng;
	if (seed != 0) rng.state = seed;
	else rng.state = cv::getTickCount();
	for (int j = 0; j < src.rows; j++)
	{
		T* s = src.ptr<T>(j);
		T* d = dest.ptr<T>(j);
		for (int i = 0; i < src.cols; i++)
		{
			double a1 = rng.uniform((double)0, (double)1);

			if (a1 > per)
				d[i] = s[i];
			else
			{
				double a2 = rng.uniform((double)0, (double)1);
				if (a2 > 0.5)d[i] = (T)0.0;
				else d[i] = (T)255.0;
			}
		}
	}
}

void addNoiseSoltPepperMono(Mat& src, Mat& dest, double per, int seed=0)
{
	if (src.type() == CV_8U) addNoiseSoltPepperMono_<uchar>(src, dest, per, seed);
	if (src.type() == CV_16U) addNoiseSoltPepperMono_<ushort>(src, dest, per, seed);
	if (src.type() == CV_16S) addNoiseSoltPepperMono_<short>(src, dest, per, seed);
	if (src.type() == CV_32S) addNoiseSoltPepperMono_<int>(src, dest, per, seed);
	if (src.type() == CV_32F) addNoiseSoltPepperMono_<float>(src, dest, per, seed);
	if (src.type() == CV_64F) addNoiseSoltPepperMono_<double>(src, dest, per, seed);
}

void addNoiseMono_nf(Mat& src, Mat& dest, double sigma)
{
	Mat s;
	src.convertTo(s, CV_32S);
	Mat n(s.size(), CV_32S);
	randn(n, 0, sigma);
	Mat temp = s + n;
	temp.convertTo(dest, src.type());
}

void addNoiseMono_f(Mat& src, Mat& dest, double sigma)
{
	Mat s;
	src.convertTo(s, CV_64F);
	Mat n(s.size(), CV_64F);
	randn(n, 0, sigma);
	Mat temp = s + n;
	temp.convertTo(dest, src.type());
}

void addNoiseMono(Mat& src, Mat& dest, double sigma)
{
	if (src.type() == CV_32F || src.type() == CV_64F)
	{
		addNoiseMono_f(src, dest, sigma);
	}
	else
	{
		addNoiseMono_nf(src, dest, sigma);
	}
}

void addNoise(InputArray src_, OutputArray dest_, const double sigma, const double sprate, const int seed)
{
	if(seed!=0) cv::theRNG().state = seed;
	if (dest_.empty() || dest_.size() != src_.size() || dest_.type() != src_.type()) dest_.create(src_.size(), src_.type());
	Mat src = src_.getMat();
	Mat dest = dest_.getMat();
	if (src.channels() == 1)
	{
		addNoiseMono(src, dest, sigma);
		if (sprate != 0)addNoiseSoltPepperMono(dest, dest, sprate, seed);
		return;
	}
	else
	{
		vector<Mat> s(src.channels());
		vector<Mat> d(src.channels());
		split(src, s);
		for (int i = 0; i < src.channels(); i++)
		{
			addNoiseMono(s[i], d[i], sigma);
			if (sprate != 0)addNoiseSoltPepperMono(d[i], d[i], sprate, seed);
		}
		cv::merge(d, dest);
	}
	if (seed != 0) cv::theRNG().state = cv::getTickCount();
}
}