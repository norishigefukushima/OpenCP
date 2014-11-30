#include "../opencp.hpp"
#include <fstream>
using namespace std;


void GaussianBlur2(Mat& src, Mat& dest, float sigma, int clip = 3, int depth=CV_32F)
{	
	Mat srcf;
	src.convertTo(srcf,depth);

	GaussianBlur(srcf, srcf, Size(cvRound(clip*sigma)*2 +1,cvRound(clip*sigma)*2 +1),sigma, 0.0, BORDER_REPLICATE);
	srcf.convertTo(dest,src.type(),1.0,0.5);
}

void guiGausianFilterTest(Mat& src__)
{
	int w = (4-src__.cols%4)%4;
	int h = (4-src__.rows%4)%4;
	Mat src_; copyMakeBorder(src__,src_,0,h,0,w,BORDER_REPLICATE);

	Mat src;
	if(src_.channels()==3)cvtColor(src_,src,COLOR_BGR2GRAY);
	else src = src_;

	Mat dest;

	string wname = "Gaussian filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 4);

	//int r = 10; createTrackbar("r",wname,&r,200);
	int space = 50; createTrackbar("space",wname,&space,2000);
	int step = 4; createTrackbar("step",wname,&step,100);

	int scale = 1; createTrackbar("scale",wname,&scale,50);
	int key = 0;
	Mat show;

	Mat ref;

	GaussianBlur2(src, ref, space/10.f, 6, CV_64F);

	
	while(key!='q')
	{
		double tim;
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_space = space/10.f;
		int d = cvRound(sigma_space*3.0)*2+1;

		{
			CalcTime t("realtime bilateral filter",0,false);
			GaussianBlur2(src, ref, sigma_space,6,CV_64F);
			tim = t.getTime();
			ci("time: %f",tim);
		}

		if(sw==0)
		{
			CalcTime t("Alvarez-Mazorra",0,false);
			GaussianBlurAM(src, dest, sigma_space, step);
			tim = t.getTime();
		}
		else if(sw==1)
		{
			CalcTime t("Deriche",0,false);
			GaussianBlurDeriche(src, dest, sigma_space, min(max(2, step),4));
			tim = t.getTime();	
		}

		else if(sw==2)
		{
			CalcTime t("sugimoto",0,false);
			GaussianBlurSR(src, dest, sigma_space);
			tim = t.getTime();

		}
		else if(sw==3)
		{

			CalcTime t("float 3 sigma",0,false);
			GaussianBlur2(src, dest, sigma_space,3,CV_32F);
			tim = t.getTime();

		}
		else if(sw==4)
		{
			CalcTime t("float 6 sigma",0,false);
			GaussianBlur2(src, dest, sigma_space,6,CV_32F);
			tim = t.getTime();
		}
		

		ci("time: %f",tim);
		ci("d: %d",d);
		ci("PSNR: %f",PSNR(dest,ref));
		
		ci.flush();
		diffshow("diff", dest,ref, (float)scale);

		alphaBlend(ref, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
	destroyAllWindows();
}