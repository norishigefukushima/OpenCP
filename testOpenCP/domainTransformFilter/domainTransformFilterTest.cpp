#include "../opencp.hpp"
#include <fstream>

using namespace std;

void checkdiff(Mat& src, Mat& src2)
{
	Mat g1,g2;
	cvtColor(src,g1,COLOR_BGR2GRAY);
	cvtColor(src2,g2,COLOR_BGR2GRAY);

	Mat mask;
	compare(g1,g2,mask,cv::CMP_NE);

	cout<<"non zero"<<countNonZero(mask)<<endl;
	for(int j=0;j<src.rows;j++)
	{
		for(int i=0;i<src.cols;i++)
		{
			if(mask.at<uchar>(j,i)!=0)cout<<Point(i,j)<<endl;
		}
	}
}

void guiDomainTransformFilterTest(Mat& src)
{
	 string wname = "smooth";
	namedWindow(wname);

	int sc = 500;
	int ss = 30;
	int iteration = 2;
	
	createTrackbar("sigma_color",wname,&sc,2550);
	createTrackbar("sigma_space",wname,&ss,100);
	createTrackbar("iteration",wname,&iteration,10);
	int norm = 0;
	createTrackbar("normL1/L2",wname,&norm,1);
	int implimentation=0;
	createTrackbar("impliment",wname,&implimentation,2);
	int sw=0;
	createTrackbar("RF/NC/IC",wname,&sw,2);
	int color = 0;
	createTrackbar("color",wname,&color,1);

	int key = 0;
	while(key!='q' && key!=VK_ESCAPE)
	{
		float scf = (float)sc*0.1f;
		float ssf = (float)ss*1.0f;
		Mat show;
		Mat input;
		
		if(color==0) cvtColor(src,input,COLOR_BGR2GRAY);
		else input = src;
		
		int64 startTime = getTickCount();
		if(sw==0)
		{
			domainTransformFilter(input, show,scf,ssf,iteration,norm+1,DTF_RF,implimentation);
		}
		else if(sw == 1)
		{
			domainTransformFilter(input, show,scf,ssf,iteration,norm+1,DTF_NC,implimentation);
		}
		else if(sw == 2)
		{
			domainTransformFilter(input, show,scf,ssf,iteration,norm+1,DTF_IC,implimentation);
		}

		double time = (getTickCount()-startTime)/(getTickFrequency());
		printf("domain transform filter: %f ms\n",time*1000.0);

		imshow(wname,show);
		key = waitKey(1);
	}

	destroyWindow(wname);
 }

 void guiJointDomainTransformFilterTest(Mat& src, Mat& guide)
 {
	 string wname = "smooth";
	 namedWindow(wname);

	 int sc = 500;
	 int ss = 30;
	 int iteration = 2;

	 createTrackbar("sigma_color",wname,&sc,2550);
	 createTrackbar("sigma_space",wname,&ss,100);
	 createTrackbar("iteration",wname,&iteration,10);
	 int norm = 0;
	 createTrackbar("normL1/L2",wname,&norm,1);
	 int implimentation=0;
	 createTrackbar("impliment",wname,&implimentation,2);
	 int sw=0;
	 createTrackbar("RF/NC/IC",wname,&sw,5);

	 int color = 0;
	 createTrackbar("color",wname,&color,1);

	 int key = 0;
	 while(key!='q' && key!=VK_ESCAPE)
	 {
		 float scf = (float)sc*0.1f;
		 float ssf = (float)ss*1.0f;
		 Mat show;
		 Mat input;

		 if(color==0) cvtColor(src,input,COLOR_BGR2GRAY);
		 else input = src;

		 int64 startTime = getTickCount();
		 if(sw==0)
		 {
			 domainTransformFilter(input,show,scf,ssf,iteration,norm+1,DTF_RF,implimentation);
		 }
		 else if(sw == 2)
		 {
			 domainTransformFilter(input, show,scf,ssf,iteration,norm+1,DTF_NC,implimentation);
		 }
		 else if(sw == 4)
		 {
			 domainTransformFilter(input, show,scf,ssf,iteration,norm+1,DTF_IC,implimentation);
		 }
		 if(sw==1)
		 {
			 domainTransformFilter(input, guide,show,scf,ssf,iteration,norm+1,DTF_RF,implimentation);
		 }
		 else if(sw == 3)
		 {
			 domainTransformFilter(input, guide, show,scf,ssf,iteration,norm+1,DTF_NC,implimentation);
		 }
		 else if(sw == 5)
		 {
			 domainTransformFilter(input, guide, show,scf,ssf,iteration,norm+1,DTF_IC,implimentation);
		 }

		 double time = (getTickCount()-startTime)/(getTickFrequency());
		 printf("domain transform filter: %f ms\n",time*1000.0);

		 imshow(wname,show);
		 key = waitKey(1);
	 }
	 destroyWindow(wname);
 }