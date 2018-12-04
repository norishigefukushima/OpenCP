#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void alphablend1(Mat& src1, Mat& src2, Mat& alpha, Mat& dest)
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
		d[i] = (uchar)((a[i] * s1[i] + (255 - a[i])*s2[i])*div + 0.5);
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
		d[i] = (a[i] * s1[i] + (255 - a[i])*s2[i]) >> 8;
	}
}

void alphaBtest(Mat& src1, Mat& src2)
{
	ConsoleImage ci(Size(640, 480));
	namedWindow("alphaB");
	int a = 0;
	createTrackbar("a", "alphaB", &a, 255);
	int key = 0;
	Mat alpha(src1.size(), CV_8U);
	Mat s1, s2;
	if (src1.channels() == 3)cvtColor(src1, s1, COLOR_BGR2GRAY);
	else s1 = src1;
	if (src2.channels() == 3)cvtColor(src2, s2, COLOR_BGR2GRAY);
	else s2 = src2;

	Mat dest;
	Mat destbf;
	Mat destshift;

	int iter = 50;
	createTrackbar("iter", "alphaB", &iter, 200);
	while (key != 'q')
	{
		ci.clear();
		alpha.setTo(a);
		{
			CalcTime t("alpha sse");
			for (int i = 0; i < iter; i++)
				alphaBlend(s1, s2, alpha, dest);
			ci("SSE %f ms", t.getTime());

		}
		{
			CalcTime t("alpha bf");
			for (int i = 0; i < iter; i++)
				alphablend1(s1, s2, alpha, destbf);
			ci("BF %f ms", t.getTime());
		}
		{
			CalcTime t("alpha shift");
			for (int i = 0; i < iter; i++)
				alphablend2(s1, s2, alpha, destshift);
			//alphaBlend(s1,s2,alpha,destshift);
			ci("SHIFT %f ms", t.getTime());
		}
		ci("bf->sse:   %f dB", PSNR(dest, destbf));
		ci("bf->shift  %f dB", PSNR(destshift, destbf));
		ci("shift->sse %f dB", PSNR(destshift, dest));
		imshow("console", ci.image);
		imshow("alphaB", destbf);
		key = waitKey(1);
	}
}