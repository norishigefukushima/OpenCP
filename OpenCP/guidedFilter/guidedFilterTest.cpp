#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiGuidedFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "guided filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 2);

	int sigma_color10 = 100; createTrackbar("sigma_color",wname,&sigma_color10,2550);
	int sigma_space10 = 120; createTrackbar("sigma_space",wname,&sigma_color10,2550);
	int r = 4; createTrackbar("r",wname,&r,100);

	int core = 1; createTrackbar("core",wname,&core,24);
	
	int noise_s10 = 100; createTrackbar("noise",wname,&noise_s10,2550);
	int key = 0;
	Mat show;

	Mat guide;
	
	ConsoleImage ci(Size(640,480));
	while(key!='q')
	{
		float sigma_color = sigma_color10/10.f;
		float sigma_space = sigma_space10/10.f;
		int d = 2*r+1;

		Mat noise;
		addNoise(src,noise,noise_s10/10.0);

		if(sw==0)
		{
			CalcTime t("guided filter");
			//blur(noise,noise,Size(3,3));
			guidedFilter(noise,dest,r,sigma_color*sigma_color);
			//guidedFilter(feather,src,dest,r,sigma_color*sigma_color);
		}
		else if(sw==1)
		{	
			CalcTime t("guided filter tbb");
			guidedFilterMultiCore(noise, dest,r,sigma_color*sigma_color,core);
		}
		else if(sw==2)
		{	
			CalcTime t("bilateral filter");
//			domainTransformFilter(noise,  dest, 0.333*r,sigma_color,3);
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

void timeGuidedFilterTest(Mat& src)
{
	for(int i=0;i<2000;i+=50)
	{
		Mat src = Mat::zeros(i,i,CV_8U);
		Mat dest;

	
		Stat st;
		for(int n=0;n<5;n++)
		{
			CalcTime t("a",TIME_MSEC,false);
		guidedFilter(src,dest,20,0.1f);
		st.push_back(t.getTime());
		}
		cout<<i<<" "<<st.getMedian()<<endl;
		
	}

}