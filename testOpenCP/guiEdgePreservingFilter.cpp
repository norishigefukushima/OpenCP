#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiEdgePresevingFilterOpenCV(Mat& src)
{
	string wname = "edge preserving filter";
	namedWindow(wname);
	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 4; createTrackbar("switch", wname, &sw, 8);
	int r = 10; createTrackbar("r", wname, &r, 200);
	int space = 300; createTrackbar("space", wname, &space, 2000);
	int color = 500; createTrackbar("color", wname, &color, 2550);
	int tr = 1; createTrackbar("tr", wname, &tr, 10);
	//int param = 100; createTrackbar("param", wname, &param, 100);

	Mat dest;
	int key = 0;
	Mat show;

	while (key != 'q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;
		int d = 2 * r + 1;

		if (sw == 0)
		{
			Timer t("Bilateral filter");
			bilateralFilter(src, dest, d, sigma_color, sigma_space);
		}
		else if (sw == 1)
		{
			Timer t("pyrMeanShift");
			pyrMeanShiftFiltering(src, dest, sigma_space / 4, sigma_color, 2);
		}
		else if (sw == 2)
		{
			Timer t("Non-local means");
			if (src.channels() == 1) fastNlMeansDenoising(src, dest, sigma_color, 2 * tr + 1, 2 * r + 1);
			if (src.channels() == 3) fastNlMeansDenoisingColored(src, dest, sigma_color, sigma_color, 2 * tr + 1, 2 * r + 1);
		}
		else if (sw == 3)
		{
			Timer t("epf: domain transform in namespace cv");
			edgePreservingFilter(src, dest, RECURS_FILTER, sigma_space, sigma_color / 255.f);
		}
		else if (sw == 4)
		{
			Timer t("domain transform in namespace xphoto");
			ximgproc::dtFilter(src, src, dest, sigma_space, sigma_color, 0);//NC, RF, IC
		}
		else if (sw == 5)
		{
			Timer t("adaptive manifold");
			ximgproc::amFilter(src, src, dest, sigma_space, sigma_color / 255.0);
		}
		else if (sw == 6)
		{
			Timer t("guided filter");
			ximgproc::guidedFilter(src, src, dest, r, sigma_color*sigma_color);
		}
		else if (sw == 7)
		{
			Timer t("joint bilateral filter");
			ximgproc::jointBilateralFilter(src, src, dest, 2 * r + 1, sigma_color, sigma_space);
		}
		else if (sw == 8)
		{
			Timer t("fsat GSF");
			ximgproc::fastGlobalSmootherFilter(src, src, dest, sigma_space, sigma_color);
		}
		else
		{
			Timer t("dct denoise");
			xphoto::dctDenoising(src, dest, sigma_color, 8);
			
			//opencv code maybe invalid

			//CalcTime t("denoise TV");
			//vector<Mat> v;
			//vector<Mat> d(3);
			//split(src, v);
			//denoise_TVL1(v[0], d[0], sigma_color);
			//denoise_TVL1(v[1], d[1], sigma_color);
			//denoise_TVL1(v[2], d[2], sigma_color);
			//merge(d, dest);
			
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
