#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;


void alphablend_naive8u(Mat& src1, Mat& src2, Mat& alpha, Mat& dest)
{
	if (dest.empty())dest.create(src1.size(), CV_8U);
	const int imsize = (src1.size().area());
	uchar* s1 = src1.data;
	uchar* s2 = src2.data;
	uchar* a = alpha.data;
	uchar* d = dest.data;
	const double div = 1.0 / 255;
	for (int i = 0; i < imsize; i++)
	{
		d[i] = (uchar)((a[i] * s1[i] + (255 - a[i]) * s2[i]) * div + 0.5);
	}
}

void alphablend2(Mat& src1, Mat& src2, Mat& alpha, Mat& dest)
{
	if (dest.empty())dest.create(src1.size(), CV_8U);
	const int imsize = (src1.size().area());
	uchar* s1 = src1.data;
	uchar* s2 = src2.data;
	uchar* a = alpha.data;
	uchar* d = dest.data;
	const double div = 1.0 / 255;
	for (int i = 0; i < imsize; i++)
	{
		d[i] = (a[i] * s1[i] + (255 - a[i]) * s2[i]) >> 8;
	}
}

void setHorizontalGradationMask(Mat& src, const int depth)
{
	const float inv = 1.f / src.cols;
	if (depth == CV_8U)
	{
		for (int j = 0; j < src.rows; j++)
		{
			uchar* s = src.ptr<uchar>(j);
			for (int i = 0; i < src.cols; i++)
			{
				s[i] = saturate_cast<uchar>(i * inv * 255.f);
			}
		}
	}
	else if (depth == CV_32F)
	{
		for (int j = 0; j < src.rows; j++)
		{
			float* s = src.ptr<float>(j);
			for (int i = 0; i < src.cols; i++)
			{
				s[i] = saturate_cast<float>(i * inv);
			}
		}
	}
	else if (depth == CV_64F)
	{
		for (int j = 0; j < src.rows; j++)
		{
			double* s = src.ptr<double>(j);
			for (int i = 0; i < src.cols; i++)
			{
				s[i] = saturate_cast<double>(i * inv);
			}
		}
	}
	else
	{
		cout << "depth GradationMask should be 8U, 32F, or 64F" << endl;
	}
}

Mat generateHorizontalGradationMask(Size size, const int depth)
{
	Mat ret(size, depth);
	setHorizontalGradationMask(ret, depth);
	return ret;
}

template<class T>
void alphaBlendMask8U_Naive_(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
{
	int size = src1.size().area();
	T* s1 = src1.ptr<T>();
	T* s2 = src2.ptr<T>();
	T* d = dst.ptr<T>();
	uchar* a = alpha.ptr<uchar>();
	const float inv = 1.f / 255.0;
	if (src1.channels() == 1)
	{
		for (int i = 0; i < size; i++)
		{
			const float aa = inv * a[i];
			d[i] = saturate_cast<T>(aa * s1[i] + (1.f - aa) * s2[i]);
		}
	}
	else if (src1.channels() == 3)
	{
		for (int i = 0; i < size; i++)
		{
			const float aa = inv * a[i];
			d[3 * i + 0] = saturate_cast<T>(aa * s1[3 * i + 0] + (1.f - aa) * s2[3 * i + 0]);
			d[3 * i + 1] = saturate_cast<T>(aa * s1[3 * i + 1] + (1.f - aa) * s2[3 * i + 1]);
			d[3 * i + 2] = saturate_cast<T>(aa * s1[3 * i + 2] + (1.f - aa) * s2[3 * i + 2]);
		}
	}
}

template<class T>
void alphaBlendMask32F_Naive_(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
{
	int size = src1.size().area();
	T* s1 = src1.ptr<T>();
	T* s2 = src2.ptr<T>();
	T* d = dst.ptr<T>();
	float* a = alpha.ptr<float>();
	if (src1.channels() == 1)
	{
		for (int i = 0; i < size; i++)
		{
			const float aa = a[i];
			d[i] = saturate_cast<T>(aa * s1[i] + (1.f - aa) * s2[i]);
		}
	}
	else if (src1.channels() == 3)
	{
		for (int i = 0; i < size; i++)
		{
			const float aa = a[i];
			d[3 * i + 0] = saturate_cast<T>(aa * s1[3 * i + 0] + (1.f - aa) * s2[3 * i + 0]);
			d[3 * i + 1] = saturate_cast<T>(aa * s1[3 * i + 1] + (1.f - aa) * s2[3 * i + 1]);
			d[3 * i + 2] = saturate_cast<T>(aa * s1[3 * i + 2] + (1.f - aa) * s2[3 * i + 2]);
		}
	}
}

void alphaBlendMask_Naive(Mat& src1, Mat& src2, Mat& alpha, Mat& dst)
{
	dst.create(src1.size(), src1.type());

	if (alpha.depth() == CV_8U)
	{
		if (src1.depth() == CV_8U)
		{
			alphaBlendMask8U_Naive_<uchar>(src1, src2, alpha, dst);
		}
		else if (src1.depth() == CV_32F)
		{
			alphaBlendMask8U_Naive_<float>(src1, src2, alpha, dst);
		}
		else if (src1.depth() == CV_64F)
		{
			alphaBlendMask8U_Naive_<double>(src1, src2, alpha, dst);
		}
	}
	else if (alpha.depth() == CV_32F)
	{
		if (src1.depth() == CV_8U)
		{
			alphaBlendMask32F_Naive_<uchar>(src1, src2, alpha, dst);
		}
		else if (src1.depth() == CV_32F)
		{
			alphaBlendMask32F_Naive_<float>(src1, src2, alpha, dst);
		}
		else if (src1.depth() == CV_64F)
		{
			alphaBlendMask32F_Naive_<double>(src1, src2, alpha, dst);
		}
	}
	else
	{
		cout << "alpha mask should be 8U or 32F" << endl;
	}
}

void testAlphaBlend(Mat& src1, Mat& src2)
{
	ConsoleImage ci(Size(640, 480));
	namedWindow("alphaB");
	int a = 0;
	createTrackbar("a", "alphaB", &a, 255);
	int key = 0;
	Mat alpha(src1.size(), CV_8U);
	Mat s1, s2;
	s1 = src1;
	s2 = src2;
	if (false)//color check
	{
		if (src1.channels() == 3)cvtColor(src1, s1, COLOR_BGR2GRAY);
		else s1 = src1;
		if (src2.channels() == 3)cvtColor(src2, s2, COLOR_BGR2GRAY);
		else s2 = src2;
	}
	if (false)//32F check
	{
		s1.convertTo(s1, CV_32F);
		s2.convertTo(s2, CV_32F);
	}

	Mat destocv(s1.size(),s1.type());
	Mat destocp(s1.size(),s1.type());
	Mat destocpfix(s1.size(), s1.type());
	Mat show;

	int iter = 50; createTrackbar("iter", "alphaB", &iter, 200);
	while (key != 'q')
	{
		ci.clear();
		alpha.setTo(a);

		{
			Timer t("alpha ocv", 0, false);
			for (int i = 0; i < iter; i++)
				cv::addWeighted(s1, a / 255.0, s2, 1.0 - a / 255.0, 0.0, destocv);

			ci("OPENCV %f ms", t.getTime());
		}

		{
			Timer t("alpha ocp", 0, false);
			for (int i = 0; i < iter; i++)
			{
				alphaBlend(s1, s2, a / 255.0, destocp);
			}

			ci("OPENCP %f ms", t.getTime());
		}

		{
			Timer t("alpha ocp fix", 0, false);
			for (int i = 0; i < iter; i++)
			{
				alphaBlendFixedPoint(s1, s2, a, destocpfix);
			}

			ci("OCPFIX %f ms", t.getTime());
		}

		ci("OCP->OCV:  %f dB", PSNR(destocp, destocv));
		ci("FIX->OCV:  %f dB", PSNR(destocpfix, destocv));

		imshow("console", ci.image);
		destocp.convertTo(show, CV_8U);
		imshow("alphaB", show);
		key = waitKey(1);
	}
}

void testAlphaBlendMask(Mat& src1, Mat& src2)
{
	ConsoleImage ci(Size(640, 480));
	namedWindow("alphaB");
	int a = 0;
	createTrackbar("a", "alphaB", &a, 255);
	int key = 0;
	Mat alpha(src1.size(), CV_8U);
	Mat s1, s2;
	s1 = src1;
	s2 = src2;

	//if (false)//color check
	{
		if (src1.channels() == 3)cvtColor(src1, s1, COLOR_BGR2GRAY);
		else s1 = src1;
		if (src2.channels() == 3)cvtColor(src2, s2, COLOR_BGR2GRAY);
		else s2 = src2;
	}
	if (false)//32F check
	{
		s1.convertTo(s1, CV_32F);
		s2.convertTo(s2, CV_32F);	
	}

	Mat destnaive;
	Mat destocp, destocpfx;
	Mat show;
	int iter = 50; createTrackbar("iter", "alphaB", &iter, 200);
	while (key != 'q')
	{
		ci.clear();
		alpha.setTo(a);

		Mat mask = generateHorizontalGradationMask(s1.size(), CV_32F);

		{
			Timer t("alpha naive", 0, false);
			for (int i = 0; i < iter; i++)
				alphaBlendMask_Naive(s1, s2, mask, destnaive);

			ci("Naive %f ms", t.getTime());
		}

		{
			Timer t("alpha ocp", 0, false);
			for (int i = 0; i < iter; i++)
				alphaBlend(s1, s2, mask, destocp);

			ci("OCP   %f ms", t.getTime());
		}

		{
			Mat mask8u,s1u,s2u;
			mask.convertTo(mask8u, CV_8U, 255);
			s1.convertTo(s1u, CV_8U);
			s2.convertTo(s2u, CV_8U);
			Timer t("alpha ocpfx", 0, false);
			for (int i = 0; i < iter; i++)
				alphaBlendFixedPoint(s1u, s2u, mask8u, destocpfx);

			ci("OCPfx %f ms", t.getTime());
		}

		if (key == 't')guiAlphaBlend(destnaive, destocp);

		ci("OCP->OCV  :  %f dB", PSNR(destocp, destnaive));
		ci("OCPfx->OCV:  %f dB", PSNR(destocpfx, destnaive));

		imshow("console", ci.image);
		destnaive.convertTo(show, CV_8U);
		imshow("alphaB", show);
		key = waitKey(1);
	}
}