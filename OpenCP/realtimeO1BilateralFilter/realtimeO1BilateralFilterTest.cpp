#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiRealtimeO1BilateralFilterTest(Mat& src_)
{
	Mat src;
	if(src_.channels()==3)cvtColor(src_,src,COLOR_BGR2GRAY);
	else src = src_;

	Mat dest;

	string wname = "realtime O1 bilateral filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 2; createTrackbar("switch",wname,&sw, 4);

	//int r = 10; createTrackbar("r",wname,&r,200);
	int space = 100; createTrackbar("space",wname,&space,2000);
	int color = 500; createTrackbar("color",wname,&color,2550);
	int bin = 16; createTrackbar("bin",wname,&bin,100);
	int iter = 3;createTrackbar("iter",wname,&iter,100);

	namedWindow("diff");
	int scale = 1;createTrackbar("scale","diff",&scale,50);

	int key = 0;
	Mat show;

	RealtimeO1BilateralFilter rbf;
	Mat ref;

	//GaussianBlur(src, ref, Size(cvRound(3*space/10.f)*2 +1,cvRound(3*space/10.f)*2 +1),space/10.f);

	
	bilateralFilter(src,ref,cvRound(3.f*space/10.f)*2 +1, color/10.f,space/10.0f);

	while(key!='q')
	{
		double tim;
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int d = cvRound(sigma_space*3.0)*2+1;

		if(key=='r')
		{
			bilateralFilter(src,ref,Size(d,d), color/10.f,space/10.0f);
		}
		if(sw==0)
		{
			CalcTime t("realtime bilateral filter",0,false);
			rbf.gauss(src, dest, d/2,sigma_color, sigma_space, bin);
			tim = t.getTime();
		}

		if(sw==1)
		{
			CalcTime t("realtime bilateral filter",0,false);
			rbf.gauss_iir(src, dest, sigma_color, sigma_space, bin,iter);
			tim = t.getTime();
		}
		if(sw==2)
		{
			CalcTime t("realtime bilateral filter",0,false);
			rbf.gauss_sr(src, dest, sigma_color, sigma_space, bin);
			tim = t.getTime();
		}
		/*
		else if(sw==1)
		{
			CalcTime t("birateral filter: fastest opencp implimentation");
			bilateralFilterSP_test3_8u(src, dest, Size(d,d), sigma_color,sigma_color*rate/100.0, sigma_space,BORDER_REPLICATE);
			//bilateralFilterSP2_8u(src, dest, Size(d,d), sigma_color, sigma_space,BORDER_REPLICATE);
			//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,1);

			//bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_CIRCLE);
		}
		else if(sw==2)
		{
			CalcTime t("birateral filter: fastest opencp implimentation with rectangle kernel");
			//rbf(src, dest, sigma_color, sigma_space);
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
			//recursiveBilateralFilter(src, dest2, sigma_color, sigma_space,0);
			//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,1);
			//cout<<norm(dest,dest2)<<endl;
			//bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
			
		}
		else if(sw==3)
		{
			CalcTime t("birateral filter: fastest: sepalable approximation of opencp");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_SEPARABLE);

		}
		else if(sw==4)
		{
			CalcTime t("birateral filter: slowest: non-parallel and inefficient implimentation");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, FILTER_SLOWEST);
		}
		*/

		ci("d: %d",d);
		ci("PSNR: %f",PSNR(dest,ref));
		ci("time: %f",tim);
		ci.flush();
		alphaBlend(ref, dest,a/100.0, show);

		diffshow("diff", dest,ref, scale);
		imshow(wname,show);
		key = waitKey(1);
	}
	destroyAllWindows();
}