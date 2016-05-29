#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

void alphaBlend(InputArray src1, InputArray src2, const double alpha, OutputArray dest)
{
	CV_Assert(src1.size() == src2.size());
	Mat s1, s2;
	if (src1.depth() == src2.depth())
	{
		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
		}
		else if (src2.channels() == 3)
		{
			cvtColor(src1, s1, CV_GRAY2BGR);
			s2 = src2.getMat();
		}
		else
		{
			cvtColor(src2, s2, CV_GRAY2BGR);
			s1 = src1.getMat();
		}
	}
	else if (src1.depth() != src2.depth())
	{
		int depth = max(src1.depth(), src2.depth());

		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
		}
		else if (src2.channels() == 3)
		{
			cvtColor(src1, s1, CV_GRAY2BGR);
			s2 = src2.getMat();
		}
		else
		{
			cvtColor(src2, s2, CV_GRAY2BGR);
			s1 = src1.getMat();
		}
		s1.convertTo(s1, depth);
		s1.convertTo(s2, depth);
	}

	cv::addWeighted(s1, alpha, s2, 1.0 - alpha, 0.0, dest);
}

void alphaBlendSSE_8u(const Mat& src1, const Mat& src2, const uchar alpha, Mat& dest)
{
	if (dest.empty())dest.create(src1.size(), CV_8U);

	const int imsize = (src1.size().area() / 16);
	uchar* s1 = (uchar*)src1.data;
	uchar* s2 = (uchar*)src2.data;
	uchar* d = dest.data;

	const __m128i zero = _mm_setzero_si128();
	const __m128i amax = _mm_set1_epi8(char(255));
	const __m128i ma = _mm_set1_epi16(alpha);
	const __m128i mia = _mm_sub_epi16(amax, ma);
	int i = 0;
	if (s1 == d)
	{
		
		for (; i < imsize; ++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			
			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			ms1l = _mm_mullo_epi16(ms1l, ma);
			ms2l = _mm_mullo_epi16(ms2l, mia);
			ms1l = _mm_add_epi16(ms1l, ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l, 8);

			ms1h = _mm_mullo_epi16(ms1h, ma);
			ms2h = _mm_mullo_epi16(ms2h, mia);
			ms1h = _mm_add_epi16(ms1h, ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h, 8);

			_mm_stream_si128((__m128i*)s1, _mm_packs_epi16(ms1l, ms1h));

			s1 += 16;
			s2 += 16;
		}
	}
	else
	{
		for (; i < imsize; ++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			ms1l = _mm_mullo_epi16(ms1l, ma);
			ms2l = _mm_mullo_epi16(ms2l, mia);
			ms1l = _mm_add_epi16(ms1l, ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l, 8);

			ms1h = _mm_mullo_epi16(ms1h, ma);
			ms2h = _mm_mullo_epi16(ms2h, mia);
			ms1h = _mm_add_epi16(ms1h, ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h, 8);

			_mm_store_si128((__m128i*)d, _mm_packs_epi16(ms1l, ms1h));

			s1 += 16;
			s2 += 16;
			d += 16;
		}
	}

	{
		uchar* s1 = (uchar*)src1.data;
		uchar* s2 = (uchar*)src2.data;
		uchar* d = dest.data;
		for (int n = i * 16; n < src1.size().area(); n++)
		{
			d[n] = (alpha * s1[n] + (255 - alpha)*s2[n]) >> 8;
		}
	}
}

void alphaBlendSSE_8u(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest)
{
	if (dest.empty())dest.create(src1.size(), CV_8U);

	const int imsize = (src1.size().area() / 16);
	uchar* s1 = (uchar*)src1.data;
	uchar* s2 = (uchar*)src2.data;
	uchar* a = (uchar*)alpha.data;
	uchar* d = dest.data;

	const __m128i zero = _mm_setzero_si128();
	const __m128i amax = _mm_set1_epi8(char(255));
	int i = 0;
	if (s1 == d)
	{
		for (; i < imsize; ++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax, mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l, mal);
			ms2l = _mm_mullo_epi16(ms2l, imal);
			ms1l = _mm_add_epi16(ms1l, ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l, 8);

			ms1h = _mm_mullo_epi16(ms1h, mah);
			ms2h = _mm_mullo_epi16(ms2h, imah);
			ms1h = _mm_add_epi16(ms1h, ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h, 8);

			_mm_stream_si128((__m128i*)s1, _mm_packs_epi16(ms1l, ms1h));

			s1 += 16;
			s2 += 16;
			a += 16;
		}
	}
	else
	{
		for (; i < imsize; ++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax, mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l, mal);
			ms2l = _mm_mullo_epi16(ms2l, imal);
			ms1l = _mm_add_epi16(ms1l, ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l, 8);

			ms1h = _mm_mullo_epi16(ms1h, mah);
			ms2h = _mm_mullo_epi16(ms2h, imah);
			ms1h = _mm_add_epi16(ms1h, ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h, 8);

			_mm_store_si128((__m128i*)d, _mm_packs_epi16(ms1l, ms1h));

			s1 += 16;
			s2 += 16;
			a += 16;
			d += 16;
		}
	}

	{
		uchar* s1 = (uchar*)src1.data;
		uchar* s2 = (uchar*)src2.data;
		uchar* a = (uchar*)alpha.data;
		uchar* d = dest.data;
		for (int n = i * 16; n < src1.size().area(); n++)
		{
			d[n] = (a[n] * s1[n] + (255 - a[n])*s2[n]) >> 8;
		}
	}
}

void alphaBlendApproxmate(InputArray src1, InputArray src2, const uchar alpha, OutputArray dest)
{
	if (dest.empty() || dest.size() != src1.size() || dest.type() != src1.type()) dest.create(src1.size(), src1.type());

	alphaBlendApproxmate(src1.getMat(), src2.getMat(), alpha, dest.getMat());
}

void alphaBlendApproxmate(InputArray src1, InputArray src2, InputArray alpha, OutputArray dest)
{
	CV_Assert(alpha.depth() == CV_8U);
	if (dest.empty() || dest.size() != src1.size() || dest.type() != src1.type()) dest.create(src1.size(), src1.type());

	alphaBlendApproxmate(src1.getMat(), src2.getMat(), alpha.getMat(), dest.getMat());
}

void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest)
{
	int T;
	Mat s1, s2;
	if (src1.channels() <= src2.channels())T = src2.type();
	else T = src1.type();
	if (dest.empty()) dest = Mat::zeros(src1.size(), T);
	if (dest.type() != T)dest = Mat::zeros(src1.size(), T);
	if (src1.channels() == src2.channels())
	{
		s1 = src1;
		s2 = src2;
	}
	else if (src2.channels() == 3)
	{
		cvtColor(src1, s1, CV_GRAY2BGR);
		s2 = src2;
	}
	else
	{
		cvtColor(src2, s2, CV_GRAY2BGR);
		s1 = src1;
	}
	Mat a;
	if (alpha.depth() == CV_8U && s1.channels() == 1)
	{
		//alpha.convertTo(a,CV_32F,1.0/255.0);
		alphaBlendSSE_8u(src1, src2, alpha, dest);
		return;
	}
	else if (alpha.depth() == CV_8U)
	{
		alpha.convertTo(a, CV_32F, 1.0 / 255);
	}
	else
	{
		alpha.copyTo(a);
	}

	if (dest.channels() == 3)
	{
		vector<Mat> ss1(3), ss2(3);
		vector<Mat> ss1f(3), ss2f(3);
		split(s1, ss1);
		split(s2, ss2);
		for (int c = 0; c < 3; c++)
		{
			ss1[c].convertTo(ss1f[c], CV_32F);
			ss2[c].convertTo(ss2f[c], CV_32F);
		}
		{
			float* s1r = ss1f[0].ptr<float>(0);
			float* s2r = ss2f[0].ptr<float>(0);

			float* s1g = ss1f[1].ptr<float>(0);
			float* s2g = ss2f[1].ptr<float>(0);

			float* s1b = ss1f[2].ptr<float>(0);
			float* s2b = ss2f[2].ptr<float>(0);


			float* al = a.ptr<float>(0);
			const int size = src1.size().area() / 4;
			const int sizeRem = src1.size().area() - size * 4;

			const __m128 ones = _mm_set1_ps(1.0f);

			for (int i = size; i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones, msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1, msa);
				ms2 = _mm_mul_ps(ms2, imsa);
				ms1 = _mm_add_ps(ms1, ms2);
				_mm_store_ps(s1r, ms1);//store ss1f

				ms1 = _mm_load_ps(s1g);
				ms2 = _mm_load_ps(s2g);
				ms1 = _mm_mul_ps(ms1, msa);
				ms2 = _mm_mul_ps(ms2, imsa);
				ms1 = _mm_add_ps(ms1, ms2);
				_mm_store_ps(s1g, ms1);//store ss1f

				ms1 = _mm_load_ps(s1b);
				ms2 = _mm_load_ps(s2b);
				ms1 = _mm_mul_ps(ms1, msa);
				ms2 = _mm_mul_ps(ms2, imsa);
				ms1 = _mm_add_ps(ms1, ms2);
				_mm_store_ps(s1b, ms1);//store ss1f

				al += 4, s1r += 4, s2r += 4, s1g += 4, s2g += 4, s1b += 4, s2b += 4;
			}
			for (int i = 0; i < sizeRem; i++)
			{
				*s1r = *al * *s1r + (1.f - *al) * *s2r;
				*s1g = *al * *s1g + (1.f - *al) * *s2g;
				*s1b = *al * *s1b + (1.f - *al) * *s2b;

				al++, s1r++, s2r++, s1g++, s2g++, s1b++, s2b++;
			}
			for (int c = 0; c < 3; c++)
			{
				ss1f[c].convertTo(ss1[c], CV_8U);
			}
			merge(ss1, dest);
		}
	}
	else if (dest.channels() == 1)
	{
		Mat ss1f, ss2f;
		s1.convertTo(ss1f, CV_32F);
		s2.convertTo(ss2f, CV_32F);
		{
			float* s1r = ss1f.ptr<float>(0);
			float* s2r = ss2f.ptr<float>(0);
			float* al = a.ptr<float>(0);
			const int size = src1.size().area() / 4;
			const int nn = src1.size().area() - size * 4;
			const __m128 ones = _mm_set1_ps(1.0f);
			for (int i = size; i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones, msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1, msa);
				ms2 = _mm_mul_ps(ms2, imsa);
				ms1 = _mm_add_ps(ms1, ms2);
				_mm_store_ps(s1r, ms1);//store ss1f

				al += 4, s1r += 4, s2r += 4;
			}
			for (int i = nn; i--;)
			{
				*s1r = *al * *s1r + (1.0f - *al)* *s2r;
				al++, s1r++, s2r++;
			}
			if (src1.depth() == CV_32F)
				ss1f.copyTo(dest);
			else
				ss1f.convertTo(dest, src1.depth());
		}
	}
}

/*
static void alphablend1(Mat& src1, Mat& src2,Mat& alpha, Mat& dest)
{
if(dest.empty())dest.create(src1.size(),CV_8U);
const int imsize = (src1.size().area());
uchar* s1 = src1.data;
uchar* s2 = src2.data;
uchar* a = alpha.data;
uchar* d = dest.data;
const double div = 1.0/255;
for(int i=0;i<imsize;i++)
{
d[i]=(uchar)((a[i]*s1[i]+(255-a[i])*s2[i])*div + 0.5);
}
}
static void alphablend2(Mat& src1, Mat& src2,Mat& alpha, Mat& dest)
{
if(dest.empty())dest.create(src1.size(),CV_8U);
const int imsize = (src1.size().area());
uchar* s1 = src1.data;
uchar* s2 = src2.data;
uchar* a = alpha.data;
uchar* d = dest.data;
const double div = 1.0/255;
for(int i=0;i<imsize;i++)
{
d[i]=(a[i]*s1[i]+(255-a[i])*s2[i])>>8;
}
}


static void alphaBtest(Mat& src1, Mat& src2)
{
//	ConsoleImage ci(Size(640,480));
namedWindow("alphaB");
int a=0;
createTrackbar("a","alphaB",&a,255);
int key = 0;
Mat alpha(src1.size(),CV_8U);
Mat s1,s2;
if(src1.channels()==3)cvtColor(src1,s1,CV_BGR2GRAY);
else s1 = src1;
if(src2.channels()==3)cvtColor(src2,s2,CV_BGR2GRAY);
else s2 = src2;

Mat dest;
Mat destbf;
Mat destshift;

int iter = 50;
createTrackbar("iter","alphaB",&iter,200);
while(key!='q')
{
//		ci.clear();
alpha.setTo(a);
{
CalcTime t("alpha sse");
for(int i=0;i<iter;i++)
alphaBlendSSE_8u(s1,s2,alpha,dest);
//		ci("SSE %f ms", t.getTime());

}
{
CalcTime t("alpha bf");
for(int i=0;i<iter;i++)
alphablend1(s1,s2,alpha,destbf);
//		ci("BF %f ms", t.getTime());
}
{
CalcTime t("alpha shift");
for(int i=0;i<iter;i++)
alphablend2(s1,s2,alpha,destshift);
//alphaBlend(s1,s2,alpha,destshift);
//		ci("SHIFT %f ms", t.getTime());
}
//		ci("bf->sse:   %f dB",calcPSNR(dest,destbf));
//		ci("bf->shift  %f dB",calcPSNR(destshift,destbf));
//	ci("shift->sse %f dB",calcPSNR(destshift,dest));
//		imshow("console",ci.show);
imshow("alphaB",destbf);
key = waitKey(1);
}
}
*/


void guiAlphaBlend(InputArray src1, InputArray src2, bool isShowImageStats)
{
	if (isShowImageStats)
	{
		showMatInfo(src1, "src1");
		cout << endl;
		showMatInfo(src2, "src2");
	}
	double minv, maxv;
	minMaxLoc(src1, &minv, &maxv);
	bool isNormirized = (maxv <= 1.0 &&minv >= 0.0) ? true : false;
	Mat s1, s2;

	if (src1.depth() == CV_8U || src1.depth() == CV_32F)
	{
		if (src1.channels() == 1)cvtColor(src1, s1, CV_GRAY2BGR);
		else s1 = src1.getMat();
		if (src2.channels() == 1)cvtColor(src2, s2, CV_GRAY2BGR);
		else s2 = src2.getMat();
	}
	else
	{
		Mat ss1, ss2;
		src1.getMat().convertTo(ss1, CV_32F);
		src2.getMat().convertTo(ss2, CV_32F);

		if (src1.channels() == 1)cvtColor(ss1, s1, CV_GRAY2BGR);
		else s1 = ss1.clone();
		if (src2.channels() == 1)cvtColor(ss2, s2, CV_GRAY2BGR);
		else s2 = ss2.clone();
	}
	namedWindow("alphaBlend");
	int a = 0;
	cv::createTrackbar("a", "alphaBlend", &a, 100);
	int key = 0;
	Mat show;
	while (key != 'q')
	{
		addWeighted(s1, 1.0 - a / 100.0, s2, a / 100.0, 0.0, show);

		if (show.depth() == CV_8U)
		{
			cv::imshow("alphaBlend", show);
		}
		else
		{
			if (isNormirized)
			{
				cv::imshow("alphaBlend", show);
			}
			else
			{
				minMaxLoc(show, &minv, &maxv);

				Mat s;
				if (maxv <= 255)
					show.convertTo(s, CV_8U);
				else
					show.convertTo(s, CV_8U, 255 / maxv);

				imshow("alphaBlend", s);
			}
		}
		key = waitKey(1);
		if (key == 'f')
		{
			a = (a > 0) ? 0 : 100;
			setTrackbarPos("a", "alphaBlend", a);
		}
		if (key == 'p')
		{
			cout << "PSNR: " << PSNR(src1, src2) << "dB" << endl;
			cout << "MSE: " << MSE(src1, src2) << endl;
		}
	}
	destroyWindow("alphaBlend");
}
}
