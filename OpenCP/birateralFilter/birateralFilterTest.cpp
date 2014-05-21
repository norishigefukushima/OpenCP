#include "../opencp.hpp"
using namespace std;

void guiBirateralFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "SLIC";
	namedWindow(wname);

	int a=50;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 3);
	int r = 5; createTrackbar("r",wname,&r,200);
	int space = 10; createTrackbar("space",wname,&space,2000);
	int color = 30; createTrackbar("color",wname,&color,2550);
	int key = 0;
	Mat show;

	while(key!='q')
	{
		cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		double sigma_color = color/10.0;
		double sigma_space = space/10.0;
		int d = 2*r+1;

		
		if(sw==0)
		{
			CalcTime t("birateral filter: opencv");
			bilateralFilter(src, dest, d, sigma_color, sigma_space);
		}
		if(sw==1)
		{
			CalcTime t("birateral filter: fastest opencp implimentation");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,BILATERAL_DEFAULT);
		}
		if(sw==2)
		{
			CalcTime t("birateral filter: fastest: sepalable approximation of opencp");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,BILATERAL_SEPARABLE);
		}
		if(sw==3)
		{
			CalcTime t("birateral filter: fastest: sepalable approximation of opencp");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space,BILATERAL_SLOWEST);
		}

		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}