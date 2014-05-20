#include "../opencp.hpp"
using namespace std;

void guiBirateralFilterTest(Mat& src)
{
	Mat dest,dest2;

	string wname = "SLIC";
	namedWindow(wname);

	int a=50;createTrackbar("a",wname,&a,100);
	int r = 5; createTrackbar("r",wname,&r,200);
	int space = 10; createTrackbar("space",wname,&space,2000);
	int color = 30; createTrackbar("color",wname,&color,2550);
	int key = 0;
	Mat show;

	while(key!='q')
	{
		cout<<"r:"<<r<<endl;
		float sigma_color = color/10.0;
		float sigma_space = space/10.0;
		int d = 2*r+1;

		{
			CalcTime t("birateral slowest");
			bilateralFilterSlowest(src, dest, d, sigma_color, sigma_space);
		}

		alphaBlend(src, dest,a/100.0, show);
		imshow(wname,show);
		key = waitKey(1);
	}
}