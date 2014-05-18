#include "opencp.hpp"
using namespace std;

void guiSLICTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "SLIC";
	namedWindow(wname);

	int a=0;createTrackbar("a",wname,&a,100);
	int rs = 10; createTrackbar("rs",wname,&rs,200);
	int reg = 100; createTrackbar("reg",wname,&reg,200000);
	int mrs = 10; createTrackbar("seg size",wname,&mrs,10000);
	int iter = 10; createTrackbar("iteration",wname,&mrs,1000);
	int key = 0;
	while(key!='q')
	{
		
		Mat show;
		{
			CalcTime t("slic all");
			SLIC(src, dest,rs,reg*255, mrs,iter);
		}
		cout<<endl<<endl;
		{
			CalcTime t("slic base");
			SLICBase(src, dest2,rs,reg*255, mrs,iter);
		}

		//std::cout<<"comp "<<calcPSNR(dest,dest2)<<std::endl;

		//alphaBlend(src,dest,a/100.0,show);
		//alphaBlend(dest2,dest,a/100.0,show);

		patchBlendImage(src,dest, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}

int main(int argc, char** argv)
{
	
	Mat src = imread("lenna.png");
	guiSLICTest(src);
	
	Mat filtered;
	Mat temp;
	src.copyTo(temp);
	for(int i=0;i<15;i++)
	{
		
		bilateralFilter(temp,filtered,21,35,5,BORDER_REPLICATE);
		filtered.copyTo(temp);
	}

	Mat dest;
	patchBlendImage(src,filtered, dest);
	imshow("show",dest);
	waitKey();

	return 0;
}