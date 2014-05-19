#include "../opencp.hpp"

void guiSLICTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "SLIC";
	namedWindow(wname);

	//int a=0;createTrackbar("a",wname,&a,100);
	int rs = 10; createTrackbar("rs",wname,&rs,200);
	int reg = 100; createTrackbar("reg",wname,&reg,2000);
	int mrs = 10; createTrackbar("ratio of min region size",wname,&mrs,100);
	int iter = 10; createTrackbar("iteration",wname,&iter,1000);
	int key = 0;
	Mat seg;

	while(key!='q')
	{
		Mat show;
		{
			CalcTime t("slic all");
			//SLIC(src, seg, rs, reg/100.0, mrs/100.0,iter);
			SLICBase(src, seg, rs, reg/100.0, mrs/100.0,iter);
			
		}

		drawSLIC(src,seg,dest,true,Scalar(255,255,0));
		patchBlendImage(src,dest, show, Scalar(255,255,255));
		imshow(wname,show);
		key = waitKey(1);
	}
}