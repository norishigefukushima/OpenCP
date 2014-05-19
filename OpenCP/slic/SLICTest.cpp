#include "../opencp.hpp"

void guiSLICTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "SLIC";
	namedWindow(wname);

	int a=50;createTrackbar("a",wname,&a,100);
	int S = 16; createTrackbar("S",wname,&S,200);
	int m = 30; createTrackbar("m",wname,&m,800);

	int mrs = 10; createTrackbar("ratio of min region size",wname,&mrs,100);
	int iter = 20; createTrackbar("iteration",wname,&iter,1000);
	int key = 0;
	Mat seg;
	Mat lab;
	while(key!='q')
	{
		Mat show;
		{
			CalcTime t("slic all");
			cvtColor(src,lab,COLOR_BGR2Lab);

			SLIC(lab, seg, S, m, mrs/100.0,iter);
		}
		drawSLIC(src,seg,dest,true,Scalar(255,255,0));
		
		patchBlendImage(src,dest, show, Scalar(255,255,255));
		alphaBlend(src, show,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}