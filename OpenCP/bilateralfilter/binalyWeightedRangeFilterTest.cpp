#include "../opencp.hpp"
#include <fstream>
using namespace std;

//depth map range interpolation
void guiBinalyWeightedRangeFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "joint binary weighted range filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	//int sw = 1; createTrackbar("switch",wname,&sw, 2);
	int range_comp=6;createTrackbar("comp",wname,&range_comp,128);
	int r = 30; createTrackbar("r",wname,&r,200);

	int color = 4; createTrackbar("thresh",wname,&color,255);
	int key = 0;
	Mat show;

	if(src.depth()!=CV_8U)
	{
		while(key!='q')
		{
			//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;

			float sigma_color = (float)color;
			double rcomp = max(range_comp,1);
			int d = 2*r+1;

			{
				CalcTime t("binary weighted range filter 32f");
				binalyWeightedRangeFilter(src, dest2, Size(d,d), sigma_color,FILTER_RECTANGLE);
				//binalyWeightedRangeFilter(dest2, dest2, Size(d,d), sigma_color,BILATERAL_SEPARABLE);
			}

			dest2.convertTo(dest,CV_8U,1.0/rcomp,0.5);
			showMatInfo(dest);


			Mat c1,c2;
			//applyColorMap(dest,c1,2);
			//applyColorMap(a,c2,2);

			//patchBlendImage(c1,c2,c2,Scalar(255,255,255),2,2);
			//alphaBlend(src, c2,a/100.0, show);
			//imshow(wname,c2);
			imshow(wname,dest);
			key = waitKey(1);
		}
	}
	else
	{
		while(key!='q')
		{
			//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
			float sigma_color = (float)color;
			double rcomp = max(range_comp,1);
			int d = 2*r+1;

			Mat a,b;
			src.convertTo(a,CV_8U,1.0/rcomp);
			a.convertTo(dest2,CV_32F);
			{
				CalcTime t("binary weighted range filter 32f");
				binalyWeightedRangeFilter(dest2, dest2, Size(d,d), sigma_color,FILTER_RECTANGLE);
				//binalyWeightedRangeFilter(dest2, dest2, Size(d,d), sigma_color,BILATERAL_SEPARABLE);
			}


			dest2.convertTo(dest,CV_8U,rcomp);
			a.convertTo(a,CV_8U,rcomp);

			Mat c1,c2;
			applyColorMap(dest,c1,2);
			applyColorMap(a,c2,2);

			patchBlendImage(c1,c2,c2,Scalar(255,255,255),2,2);
			//alphaBlend(src, c2,a/100.0, show);
			imshow(wname,c2);
			key = waitKey(1);
		}
	}
}
void guiJointBinalyWeightedRangeFilterTest(Mat& src, Mat& guide)
{
	Mat dest,dest2;

	string wname = "joint binary weighted range filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 1; createTrackbar("switch",wname,&sw, 2);
	int r = 20; createTrackbar("r",wname,&r,200);
	int color = 300; createTrackbar("color",wname,&color,2550);
	int key = 0;
	Mat show;

	while(key!='q')
	{
		cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color/10.f;
		int d = 2*r+1;

		if(sw==0)
		{
			CalcTime t("binary weighted range filter");
			binalyWeightedRangeFilter(src, dest, Size(d,d), sigma_color);
		}
		if(sw==1)
		{
			CalcTime t("joint binary weighted range filter");
			jointBinalyWeightedRangeFilter(src, guide, dest, Size(d,d), sigma_color);
		}
		if(sw==2)
		{
			CalcTime t("invalid joint");
			jointBinalyWeightedRangeFilter(guide, src, dest, Size(d,d), sigma_color);
		}

		patchBlendImage(src,dest,dest,Scalar(255,255,255));
		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}

/*
void timeBirateralTest(Mat& src)
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
				bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, BILATERAL_DEFAULT);
				st.push_back(t.getTime());
			}
			time_opencp=st.getMedian();
		}

		{
			Stat st;
			for(int i=0;i<iteration;i++)
			{
				CalcTime t("time",TIME_SEC,false);
				bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, BILATERAL_SEPARABLE);
				st.push_back(t.getTime());
			}
			time_opencp_sp=st.getMedian();
		}

		{
			Stat st;
			for(int i=0;i<iteration;i++)
			{
				CalcTime t("time",TIME_SEC,false);
				bilateralFilter(src, dest, Size(d,d), sigma_color, sigma_space, BILATERAL_SLOWEST);
				st.push_back(t.getTime());
			}
			time_opencp_sl=st.getMedian();
		}

		out<<time_opencv<<","<<time_opencp<<","<<time_opencp_sp<<","<<time_opencp_sl<<endl;
		cout<<time_opencv<<","<<time_opencp<<","<<time_opencp_sp<<","<<time_opencp_sl<<","<<endl;
	}
}
*/