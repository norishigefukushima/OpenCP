#include "../opencp.hpp"
#include <fstream>
using namespace std;

#include <opencv2/core/internal.hpp>

void guiBilateralFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "bilateral filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 4);
	int r = 20; createTrackbar("r",wname,&r,200);
	int space = 300; createTrackbar("space",wname,&space,2000);
	int color = 500; createTrackbar("color",wname,&color,2550);
	int rate = 100; createTrackbar("rate",wname,&rate,100);
	int key = 0;
	Mat show;

	RecursiveBilateralFilter rbf(src.size());
	while(key!='q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int d = 2*r+1;

		
		if(sw==0)
		{
			CalcTime t("birateral filter: opencv");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_SEPARABLE);
			//bilateralFilter(src, dest, d, sigma_color, sigma_space);
			//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,0);
		}
		else if(sw==1)
		{
			CalcTime t("birateral filter: fastest opencp implimentation");
//			bilateralFilterSP_test3_8u(src, dest, Size(d,d), sigma_color,sigma_color*rate/100.0, sigma_space,BORDER_REPLICATE);
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

		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}

void guiSeparableBilateralFilterTest(Mat& src)
{
	Mat dest;

	string wname = "bilateral filter SP";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 6);
	//int r = 20; createTrackbar("r",wname,&r,200);
	int space = 300; createTrackbar("space",wname,&space,2000);
	int color = 500; createTrackbar("color",wname,&color,2550);
	int rate = 100; createTrackbar("color2 rate",wname,&rate,100);
	//int rate_s = 100; createTrackbar("space2 rate",wname,&rate_s,100);

	//int rate1 = 100; createTrackbar("c3 rate",wname,&rate1,100);
	//int rate2 = 100; createTrackbar("c4 rate",wname,&rate2,100);

	int s = 15; createTrackbar("ssim",wname,&s,200);
	int skip = 10; createTrackbar("skip",wname,&skip,40);

	Mat ref;
	{
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int r = cvRound(sigma_space*3.0)/2;
		int d = 2*r+1;
		bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
	}
	ConsoleImage ci;

	Mat show;
	int key = 0;
	while(key!='q')
	{
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int r = cvRound(sigma_space*3.0)/2;
		int d = 2*r+1;

		double ssims = s/10.0;
		
		if(key=='r')
		{
			bilateralFilter(src, ref, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
		}
		
		if(sw==0)
		{
			CalcTime t("bilateral filter: opencv");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
		}
		else if(sw==1)
		{
			CalcTime t("bilateral filter: opencv sp");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_SEPARABLE);
		}
		else if(sw==2)
		{
			CalcTime t("bilateral filter: opencv sp HV");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_HV);
		}
		else if(sw==3)
		{
			CalcTime t("bilateral filter: opencv sp HVVH");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_HVVH);
		}
		else if(sw==4)
		{
			CalcTime t("bilateral filter: opencv sp HVVH");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_CROSS);
		}
		else if(sw==5)
		{
			CalcTime t("bilateral filter: opencv sp HVVH");
			separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate/100.0,DUAL_KERNEL_CROSSCROSS);
		}
		else if(sw==6)
		{
			//CalcTime t("bilateral filter: opencv sp HVVH");
			//separableBilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,rate1/100.0,DUAL_KERNEL_CROSSCROSS);
		}

		if(key=='f')
		{
			a = (a==0) ? 100 : 0;
			setTrackbarPos("a",wname,a);
		}
		ci(format("%f dB",PSNR(ref,dest)));
		ci(format("%f dB",SSIM(ref,dest,ssims)));
//		ci(format("%f %f",getTV(dest),getTV(ref)));
		ci.flush();

		alphaBlend(ref, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}

void timeBilateralTest(Mat& src)
{
	int iteration = 10;
	Mat dest;

	ofstream out("birateraltime.csv");

	double time_opencv;
	double time_opencp;
	double time_opencp_sp;
	double time_opencp_sl;
	for(int r=0;r<200;r++)
	{
		cout<<"r:"<<r<<" :";
		int d = 2*r+1;
		double sigma_color = 30.0;
		double sigma_space = d/3.0;

		{
			Stat st;
			for(int i=0;i<iteration;i++)
			{
				CalcTime t("time",TIME_SEC,false);
				bilateralFilter(src, dest, d, sigma_color, sigma_space);
				st.push_back(t.getTime());
			}
			time_opencv=st.getMedian();
		}

		{
			Stat st;
			for(int i=0;i<iteration;i++)
			{
				CalcTime t("time",TIME_SEC,false);
				bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, FILTER_DEFAULT);
				st.push_back(t.getTime());
			}
			time_opencp=st.getMedian();
		}

		{
			Stat st;
			for(int i=0;i<iteration;i++)
			{
				CalcTime t("time",TIME_SEC,false);
				bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, FILTER_SEPARABLE);
				st.push_back(t.getTime());
			}
			time_opencp_sp=st.getMedian();
		}

		{
			Stat st;
			for(int i=0;i<iteration;i++)
			{
				CalcTime t("time",TIME_SEC,false);
				bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, FILTER_SLOWEST);
				st.push_back(t.getTime());
			}
			time_opencp_sl=st.getMedian();
		}

		out<<time_opencv<<","<<time_opencp<<","<<time_opencp_sp<<","<<time_opencp_sl<<endl;
		cout<<time_opencv<<","<<time_opencp<<","<<time_opencp_sp<<","<<time_opencp_sl<<","<<endl;
	}
}


void guiRecursiveBilateralFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "recursive birateral filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 4);
	int r = 20; createTrackbar("r",wname,&r,200);
	int space = 300; createTrackbar("space",wname,&space,2000);
	int color = 500; createTrackbar("color",wname,&color,2550);
	int key = 0;
	Mat show;

	RecursiveBilateralFilter rbf(src.size());
	while(key!='q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int d = 2*r+1;
		
		if(sw==0)
		{
			CalcTime t("birateral filter: separable");
			rbf(src, dest, sigma_color, sigma_space);				
		}
		else if(sw==1)
		{
			CalcTime t("recursiveBilateralFilter: fastest opencp implimentation");
			bilateralFilter(src,dest,(int)(sigma_space*3.f),sigma_color,sigma_space,FILTER_SEPARABLE);
			//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,0);
			//recursiveBilateralFilter(src, dest, sigma_color, sigma_space,1);
		}
		else if(sw==2)
		{
			CalcTime t("birateral filter: fastest opencp implimentation with rectangle kernel");
			
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

		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}