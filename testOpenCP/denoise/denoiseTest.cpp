#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiDenoiseTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "non local means";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 4);

	int sigma_color10 = 100; createTrackbar("sigma_color",wname,&sigma_color10,2550);
	int sigma_space10 = 120; createTrackbar("sigma_space",wname,&sigma_color10,2550);
	int r = 4; createTrackbar("r",wname,&r,100);

	int tr = 1; createTrackbar("tr",wname,&tr,20);
	
	int noise_s10 = 100; createTrackbar("noise",wname,&noise_s10,2550);
	int key = 0;
	Mat show;

	RecursiveBilateralFilter recbf(src.size());

	while(key!='q')
	{
		float sigma_color = sigma_color10/10.f;
		float sigma_space = sigma_space10/10.f;
		int d = 2*r+1;
		int td = 2*tr+1;

		Mat noise;
		addNoise(src,noise,noise_s10/10.0);
		if(sw==0)
		{
			CalcTime t("bilateral filter");
			bilateralFilter(noise,dest,Size(d,d),sigma_color*1.5,sigma_space, FILTER_SEPARABLE);
		}
		else if(sw==1)
		{
			CalcTime t("bilateral filter: separable ");
			//GaussianBlur(noise,dest,Size(d,d),sigma_space);
			guidedFilter(noise,dest,r,sigma_color);
			//bilateralFilter(noise,dest,Size(d,d),sigma_color,sigma_space,FILTER_SEPARABLE);
		}
		else if(sw==2)
		{
			CalcTime t("binary weighted range filter");
			binalyWeightedRangeFilter(noise,dest,Size(d,d),sigma_color);
		}
		else if(sw==3)
		{
			CalcTime t("non local means");
			nonLocalMeansFilter(noise,dest,td,d,sigma_color,sigma_color,0);
		}
		else if(sw==4)
		{
			CalcTime t("recursive birateral filter");
			recbf(noise,dest,sigma_color,sigma_space);
		}

		cout<<"before:"<<PSNR(src,noise)<<endl;
		cout<<"filter:"<<PSNR(src,dest)<<endl<<endl;

		patchBlendImage(noise,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		cv::imshow(wname,show);
		key = waitKey(1);
	}
}
