#include <opencp.hpp>
#include <opencv2/ximgproc.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiDomainTransformFilterTest(Mat& src)
{
	string wname = "domain transform  filter";
	namedWindow(wname);
	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 2; createTrackbar("switch", wname, &sw, 2);
	
	int space = 500; createTrackbar("space", wname, &space, 2000);
	int color = 700; createTrackbar("color", wname, &color, 2550);
	int iter = 3; createTrackbar("iter", wname, &iter, 10);
	

	Mat dest;
	int key = 0;
	Mat show;

	while (key != 'q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;

		if (sw == 0)
		{
			CalcTime t("epf: domain transform in namespace cv");
			edgePreservingFilter(src, dest, RECURS_FILTER, sigma_space, sigma_color / 255.f);
		}
		else if (sw == 1)
		{
			CalcTime t("domain transform in namespace xphoto");
			ximgproc::dtFilter(src, src, dest, sigma_space, sigma_color, 1, iter);//NC, RF, IC
		}
		else if (sw == 2)
		{
			CalcTime t("domain transform in opencp");
			domainTransformFilter(src, dest, sigma_color, sigma_space, iter, 1, DTF_RF, DTF_BGRA_SSE_PARALLEL);
		}
		
		if (key == 'f')
		{
			a = (a == 0) ? 100 : 0;
			setTrackbarPos("a", wname, a);
		}
		alphaBlend(src, dest, a / 100.0, show);
		imshow(wname, show);
		key = waitKey(1);
	}
}