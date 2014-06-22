#include "../opencp.hpp"
#include <fstream>
using namespace std;

void guiJointNearestFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "joint nearest filter";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int sw = 0; createTrackbar("switch",wname,&sw, 4);
	int r = 20; createTrackbar("r",wname,&r,200);
	int space = 300; createTrackbar("space",wname,&space,2000);
	int color = 500; createTrackbar("color",wname,&color,2550);

	int iter = 1; createTrackbar("iter",wname,&iter,10);
	int key = 0;
	Mat show;

	while(key!='q')
	{

		float sigma_color = color/10.f;
		float sigma_space = space/10.f;
		int d = 2*r+1;

		Mat temp = src.clone();
		src.copyTo(dest);
		for(int i=0;i<iter;i++)
		{
			//GaussianBlur(dest, temp, Size(d,d),space*0.1);
			Mat c;
			Mat n;
			resize(dest, c, Size(dest.cols*2,dest.rows*2),0,0,INTER_LANCZOS4);
			resize(dest, n, Size(dest.cols*2,dest.rows*2),0,0,INTER_NEAREST);
			
			dest = Mat::zeros(n.size(),n.type());
			jointNearestFilterBase(c, n, Size(5,5),dest);
		}
		imshow("smooth",temp);
		dest.copyTo(show);
		//alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}