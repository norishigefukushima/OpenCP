#include "noise.hpp"
#include "webp.hpp"
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

	static void addNoiseMono_double(const Mat& src, Mat& dest, const double sigma)
	{
		Mat n(src.size(), CV_64F);
		randn(n, 0, sigma);
		add(src, n, dest);
	}

	static void addNoiseMono_float(const Mat& src, Mat& dest, const double sigma)
	{
		Mat n(src.size(), CV_32F);
		randn(n, 0, sigma);
		add(src, n, dest);
	}

	static void addNoiseMono(Mat& src, Mat& dest, const double sigma)
	{
		if (src.depth() == CV_32F)
		{
			addNoiseMono_float(src, dest, sigma);
		}
		else if (src.depth() == CV_64F)
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

		Mat src = src_.getMat();
		if(src.size()!=dest_.size() || src.type()!=dest_.type()) dest_.create(src_.size(), src_.type());
		Mat dest = dest_.getMat();

		if (seed != 0) cv::theRNG().state = seed;
		
		if (src.channels() == 1)
		{
			addNoiseMono(src, dest, sigma);
			if (sprate != 0.0) addNoiseSoltPepperMono(dest, dest, sprate, seed);
		}
		else
		{
			Mat s = src.reshape(1);
			Mat d = dest.reshape(1);
			addNoiseMono(s, d, sigma);
			if (sprate != 0.0)addNoiseSoltPepperMono(d, d, sprate, seed);
			dest = d.reshape(3);
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
	
#pragma region coding
	double addJPEGNoise(InputArray src, OutputArray dest, const int quality)
	{
		std::vector<uchar> buff;
		std::vector<int> param(2);
		param[0] = IMWRITE_JPEG_QUALITY;
		param[1] = quality;
		imencode(".jpg", src, buff, param);
		Mat dst = imdecode(buff, IMREAD_ANYCOLOR);
		dst.copyTo(dest);
		return 8.0 * buff.size() / src.size().area();
	}

	double addJPEG2000Noise(InputArray src, OutputArray dest, const int quality)
	{
		vector<uchar> buf;
		vector<int> param;
		param.push_back(IMWRITE_JPEG2000_COMPRESSION_X1000);
		param.push_back(quality * 10);
		imencode(".jp2", src, buf, param);
		Mat dst = imdecode(buf, 1);
		dst.copyTo(dest);
		return 8.0 * buf.size() / src.size().area();
	}

	double addWebPNoise(InputArray src, OutputArray dest, const int quality, const int method, const int colorSpace)
	{
		vector<uchar> buf;
		vector<int> param;
		param.push_back(IMWRITE_WEBP_QUALITY);
		param.push_back(quality);
		param.push_back(IMWRITE_WEBP_METHOD);
		param.push_back(method);
		param.push_back(IMWRITE_WEBP_COLORSPACE);
		param.push_back(colorSpace);
		if (colorSpace < 0)
		{
			imencode(".webp", src, buf, param);
		}
		else
		{
			cp::imencodeWebP(src.getMat(), buf, param);
		}
		Mat dst = imdecode(buf, 1);
		dst.copyTo(dest);
		return 8.0 * buf.size() / src.size().area();
	}

#ifdef USE_OPENCP_AVIF
	double addAVIFNoise(InputArray src, OutputArray dest, const int quality, const int method)
	{
		vector<uchar> buf;
		vector<int> param;
		param.push_back(IMWRITE_AVIF_QUALITY);
		param.push_back(quality);
		param.push_back(IMWRITE_AVIF_SPEED);
		param.push_back(method);
		imencode(".avif", src, buf, param);
		Mat dst = imdecode(buf, 1);
		dst.copyTo(dest);
		return 8.0 * buf.size() / src.size().area();
	}
#endif
	
#pragma endregion
}