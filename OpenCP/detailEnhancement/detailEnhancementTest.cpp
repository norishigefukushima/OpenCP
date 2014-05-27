#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiDetailEnhancement(Mat& src)
{
	Mat dest,dest2;

	string wname = "Detail Enhancement";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 1);

	int sigma_s = 20; createTrackbar("space_sigma",wname,&sigma_s,2000);
	int sigma_c = 80; createTrackbar("color_sigma",wname,&sigma_c,2550);

	int r = 4; createTrackbar("r",wname,&r,100);
	int boost = 10; createTrackbar("lambda",wname,&boost,100);
	int color = PROCESS_LAB; createTrackbar("color domain",wname,&color,1);
		
	int key = 0;
	Mat show;

	Mat srcf; src.convertTo(srcf,CV_32F);
	Mat blurred;
	while(key!='q')
	{
		double color_sigma = sigma_c/10.0;
		double space_sigma = sigma_s/10.0;
		int d = 2*r+1;
		double b = boost/10.0;

		if(sw==0)
		{
			CalcTime t("enhancement birateral");
			detailEnhancementBilateral(src,dest,d,color_sigma,space_sigma,b,color);
		}
		if(sw==1)
		{
			CalcTime t("ibp bilateral");
			//iterativeBackProjectionDeblurBilateral(blurred, dest, Size(d,d), color_sigma, r_sigma/10.0,lambda, iter);
		}

		patchBlendImage(src,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}