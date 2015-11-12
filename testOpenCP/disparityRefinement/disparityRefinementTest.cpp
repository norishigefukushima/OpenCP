#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;
/*
void guiDisparityRefinementTest(Mat& leftim, Mat& rightim, Mat& leftdisp, Mat& rightdisp, Mat& leftGT)
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

	while(key!='q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int d = 2*r+1;

		
		if(sw==0)
		{
			CalcTime t("bilateral filter: opencv");
			bilateralFilter(src, dest, d, sigma_color, sigma_space);
		}
		else if(sw==1)
		{
			CalcTime t("bilateral filter: fastest opencp implimentation");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_CIRCLE);
		}
		else if(sw==2)
		{
			CalcTime t("bi;ateral filter: fastest opencp implimentation with rectangle kernel");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,FILTER_RECTANGLE);
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
*/