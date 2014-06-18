#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiNonLocalMeansTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "non local means";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 1);

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

		cout<<"before:"<<PSNR(src,noise)<<endl;
		cout<<"filter:"<<PSNR(src,dest)<<endl<<endl;;

		patchBlendImage(src,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}