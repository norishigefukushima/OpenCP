#include "guidedFilter.hpp"
const int BORDER_TYPE = cv::BORDER_REPLICATE;

using namespace cv;
namespace cp
{
	void guidedImageGaussianFilterGray(const Mat& src, Mat& dest, const int radius, const float sigma, const float eps)
	{
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;
		Mat sf; src.convertTo(sf, CV_32F);

		Mat mSrc(imsize, CV_32F);//mean_p
		GaussianBlur(sf, mSrc, ksize, sigma, sigma, BORDER_TYPE);//meanImSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiply(sf, sf, x1);//sf*sf
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//corrI:m*sf*sf

		multiply(mSrc, mSrc, x1);//;msf*msf
		x3 -= x1;//x3: m*sf*sf-msf*msf;
		x1 = x3 + e;
		divide(x3, x1, x3);
		multiply(x3, mSrc, x1);
		x1 -= mSrc;
		GaussianBlur(x3, x2, ksize, sigma, sigma, BORDER_TYPE);//x2*k
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//x3*k
		multiply(x2, sf, x1);//x1*K
		x2 = x1 - x3;//
		x2.convertTo(dest, src.type());
	}

	void guidedImageGaussianFilterGrayEnhance(const Mat& src, Mat& dest, const int radius, const float sigma, const float eps, const float m)
	{
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;
		Mat sf; src.convertTo(sf, CV_32F);

		Mat mSrc(imsize, CV_32F);//mean_p
		GaussianBlur(sf, mSrc, ksize, sigma, sigma, BORDER_TYPE);//meanImSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiply(sf, sf, x1);//sf*sf
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//corrI:m*sf*sf

		multiply(mSrc, mSrc, x1);//;msf*msf
		x3 -= x1;//x3: m*sf*sf-msf*msf;
		x1 = x3 + e;
		divide(x3, x1, x3);
		multiply(x3, mSrc, x1);
		x1 -= mSrc;

		x3 = (m*x3 + 1.f - m);
		GaussianBlur(x3, x2, ksize, sigma, sigma, BORDER_TYPE);//x2*k
		x1 = x1 * m;
		GaussianBlur(x1, x3, ksize, sigma, sigma, BORDER_TYPE);//x3*k
		multiply(x2, sf, x1);//x1*K
		x2 = x1 - x3;//
		x2.convertTo(dest, src.type());
	}
}