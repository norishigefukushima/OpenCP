#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiJointBirateralFilterTest(Mat& src, Mat& guide)
{
	Mat dest,dest2;

	string wname = "joint birateral filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 2);
	int r = 20; createTrackbar("r",wname,&r,200);
	int space = 200; createTrackbar("space",wname,&space,2000);
	int color = 300; createTrackbar("color",wname,&color,2550);
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
			CalcTime t("birateral filter");
			bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space);
		}
		if(sw==1)
		{
			CalcTime t("joint birateral filter");
			jointBilateralFilter(src, guide, dest, Size(d,d), sigma_color, sigma_space);
		}
		if(sw==2)
		{
			CalcTime t("invalid joint");
			jointBilateralFilter(guide, src, dest, Size(d,d), sigma_color, sigma_space);
		}

		patchBlendImage(src,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}