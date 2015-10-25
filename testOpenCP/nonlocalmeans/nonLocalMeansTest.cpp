#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiSeparableNLMTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "non local means SP";
	namedWindow(wname);
	ConsoleImage ci;
	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 2);

	int tr = 1; createTrackbar("tr",wname,&tr,20);
	int r = 4; createTrackbar("r",wname,&r,100);
	int h10 = 100; createTrackbar("h",wname,&h10,2550);
	int hrate = 100; createTrackbar("h rate",wname,&hrate,100);
	int color = 100; createTrackbar("color",wname,&color,2550);

	namedWindow("diff");
	int scale = 1; createTrackbar("scale","diff",&scale,50);
	
	int noise_s = 0; createTrackbar("noise",wname,&noise_s,2550);
	Mat noise; addNoise(src,noise,noise_s/10.0);

	Mat ref;
	{
		int d = 2*r+1;
		int td = 2*tr+1;
		float sigma = (float)color/10.f;
		float h = (float)h10/10.f;
		nonLocalMeansFilter(noise,ref,td,d,h,sigma,0);
	}

	Mat show;
	int key = 0;
	while(key!='q')
	{
		float sigma = (float)color/10.f;
		float h = (float)h10/10.f;
		int d = 2*r+1;
		int td = 2*tr+1;

		if(key=='r')
		{
			addNoise(src,noise,noise_s/10.0);
			nonLocalMeansFilter(noise, ref,td,d,h,sigma,0);
		}

		if(sw==0)
		{
			CalcTime t("opencv");
			nonLocalMeansFilter(noise, dest,td,d,h,sigma,0);
		}
		if(sw==1)
		{
			CalcTime t("non local means");
			nonLocalMeansFilter(noise, dest,td,d,h,sigma, FILTER_SEPARABLE);
		}
		if(sw==2)
		{
			CalcTime t("non local means");
			jointNonLocalMeansFilter(noise, noise, dest, Size(td,td), Size(d,1), h, sigma);
			jointNonLocalMeansFilter(dest, noise, dest, Size(td,td), Size(1,d), hrate/100.0f*h, sigma);
		}

		ci("src noise %f dB",PSNR(src,noise));
		ci("denoise %f dB",PSNR(src,dest));
		ci("SP Acc %f dB",PSNR(ref,dest));
		
		ci.flush();
		diffshow("diff", dest,ref, (float)scale);
		alphaBlend(ref, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}

void guiNonLocalMeansTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "non local means";
	namedWindow(wname);
	ConsoleImage ci;
	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 2);

	int tr = 1; createTrackbar("tr",wname,&tr,20);
	int r = 4; createTrackbar("r",wname,&r,100);
	int h10 = 100; createTrackbar("h",wname,&h10,2550);
	int color = 100; createTrackbar("color",wname,&color,2550);

	int noise_s = 100; createTrackbar("noise",wname,&noise_s,2550);
	int key = 0;
	Mat show;

	while(key!='q')
	{
		float sigma = (float)color/10.f;
		float h = (float)h10/10.f;
		int d = 2*r+1;
		int td = 2*tr+1;

		Mat noise;
		addNoise(src,noise,noise_s/10.0);
		if(sw==0)
		{
			CalcTime t("opencv");
			fastNlMeansDenoisingColored(noise,dest,h,h,td,d);
		}
		if(sw==1)
		{
			CalcTime t("non local means");
			nonLocalMeansFilter(noise,dest,td,d,h,sigma,0);
		}
		if(sw==2)
		{
			CalcTime t("non local means");
			nonLocalMeansFilter(noise,dest,td,d,h,sigma, FILTER_SEPARABLE);
		}
		ci("before %f dB",PSNR(src,noise));
		ci("filter %f dB",PSNR(src,dest));
		ci.flush();

		patchBlendImage(src,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}