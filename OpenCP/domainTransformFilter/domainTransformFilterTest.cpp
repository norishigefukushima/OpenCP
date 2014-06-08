#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiDomainTransformFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "domain transform filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 2);

	int sigma_color10 = 100; createTrackbar("sigma_color",wname,&sigma_color10,2550);
	int sigma_space10 = 120; createTrackbar("sigma_space",wname,&sigma_color10,2550);
	int iter = 4; createTrackbar("iter",wname,&iter,10);

	//int core = 1; createTrackbar("core",wname,&core,24);
	
	int noise_s10 = 100; createTrackbar("noise",wname,&noise_s10,2550);
	int key = 0;
	Mat show;

	Mat guide;
	
	ConsoleImage ci(Size(640,480));
	while(key!='q')
	{
		float sigma_color = sigma_color10/10.f;
		float sigma_space = sigma_space10/10.f;

		Mat noise;
		addNoise(src,noise,noise_s10/10.0);

		if(sw==0)
		{
			CalcTime t("domain transform filter: base implimentation");
			//blur(noise,noise,Size(3,3));
			domainTransformFilterBase(noise,  dest, sigma_space,sigma_color, iter);
			//guidedFilter(feather,src,dest,r,sigma_color*sigma_color);
		}
		else if(sw==1)
		{	
			CalcTime t("domain transform filter");
			domainTransformFilter(noise,  dest, sigma_space,sigma_color, iter);
		}
		else if(sw==2)
		{	
			CalcTime t("domain transform filter");
			domainTransformFilterBase(noise, src,dest, sigma_space,sigma_color, iter);
			//bilateralFilter(noise,  dest,Size(2*r+1,2*r+1),sigma_color,0.333*r,FILTER_DEFAULT);
		}


		ci(format("before: %f",PSNR(src,noise)));
		ci(format("filter: %f",PSNR(src,dest)));
		
		ci.flush();
		
		//patchBlendImage(noise,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		cv::imshow(wname,show);
		key = waitKey(1);
	}
}