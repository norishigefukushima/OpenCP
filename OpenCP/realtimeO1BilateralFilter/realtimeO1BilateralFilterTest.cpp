#include "../opencp.hpp"
#include <fstream>
using namespace std;


void guiMaxMinFilter(Mat& src_)
{
	Mat src;
	if(src_.channels()==3)cvtColor(src_,src,COLOR_BGR2GRAY);
	else src = src_;

	Mat dest;

	string wname = "max min filter";
	namedWindow(wname);
	
	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 1);

	int r = 1; createTrackbar("r",wname,&r,10);

	int key = 0;
	Mat show;

	while(key!='q')
	{
		double tim;

		if(sw==0)
		{
			CalcTime t("max filter",0,false);
			maxFilter(src, dest, Size(2*r+1,2*r+1),MORPH_ELLIPSE);
			tim = t.getTime();
		}

		if(sw==1)
		{
			CalcTime t("min filter",0,false);
			minFilter(src, dest, Size(2*r+1,2*r+1),MORPH_ELLIPSE);
			tim = t.getTime();
		}
		
		GaussianBlurAM(dest,dest, 3*r,3);
		alphaBlend(src, dest,a/100.0, show);

		imshow(wname,show);
		key = waitKey(1);
	}
	destroyWindow(wname);
}


double PSNRBB(InputArray src, InputArray ref, int boundingX, int boundingY=0)
{
	if(boundingY==0)boundingY=boundingX;
	Mat a=src.getMat();
	Mat b=ref.getMat();

	Rect roi = Rect(Point(boundingX,boundingY),Point(a.cols-boundingX,a.rows-boundingY));
	
	return PSNR(a(roi),b(roi));
}

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
	int sw = 1; createTrackbar("switch",wname,&sw, 4);

	//int r = 10; createTrackbar("r",wname,&r,200);
	int space = 100; createTrackbar("space",wname,&space,2000);
	int color = 500; createTrackbar("color",wname,&color,2550);
	int bin = 16; createTrackbar("bin",wname,&bin,256);
	int iter = 3;createTrackbar("iter",wname,&iter,100);
	int rsize = 1;createTrackbar("resize",wname,&rsize,32);

	namedWindow("diff");
	int scale = 10;createTrackbar("scale","diff",&scale,50);

	int key = 0;
	Mat show;

	RealtimeO1BilateralFilter rbf;
	Mat ref;

	//GaussianBlur(src, ref, Size(cvRound(3*space/10.f)*2 +1,cvRound(3*space/10.f)*2 +1),space/10.f);

	
	bilateralFilter(src,ref,cvRound(6.f*space/10.f)*2 +1, color/10.f,space/10.0f, BORDER_REPLICATE);

	while(key!='q')
	{
		double tim;
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int d = cvRound(sigma_space*4.0)*2+1;

		rbf.downsample_size = rsize;
		if(key=='r')
		{
			bilateralFilter(src,ref,Size(d,d), color/10.f,space/10.0f, BORDER_REPLICATE);
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
		if(sw==3)
		{
			CalcTime t("realtime bilateral filter",0,false);
			rbf.box(src, dest, d/2, sigma_color, bin, iter);
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
		ci("PSNR: %f",PSNRBB(dest,ref,100,100));
		ci("MSE:  %f",norm(dest,ref,NORM_L2SQR)/(double)dest.size().area());
		ci("time: %f",tim);
		ci.flush();
		alphaBlend(ref, dest,a/100.0, show);

		if(key=='t')guiMaxMinFilter(src);
		diffshow("diff", dest,ref, (float)scale);
		if(key=='d')
		{
			guiAbsDiffCompareGE(ref,dest);
		}

		imshow(wname,show);
		key = waitKey(1);
	}
	destroyAllWindows();
}