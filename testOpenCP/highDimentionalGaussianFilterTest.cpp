#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void highDimentionalGaussianFilterTest(Mat& src)
{
	Mat srcf = convert(src, CV_32F);
	Mat dest, dest2;

	string wname = "highDimentionalGaussianFilter";
	namedWindow(wname);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 1; createTrackbar("switch", wname, &sw, 1);
	//int r = 20; createTrackbar("r", wname, &r, 200);
	int space = 30; createTrackbar("space", wname, &space, 200);
	int color = 500; createTrackbar("color", wname, &color, 2550);
	int rate = 100; createTrackbar("rate", wname, &rate, 100);
	int key = 0;
	Mat show;
	cp::ConsoleImage ci;

	while (key != 'q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;
		int d = 2 * ceil(sigma_space * 3) + 1;
		double time;
		string method;

		Mat ref;
		cp::bilateralFilterL2(srcf, ref, ceil(sigma_space * 3), sigma_color, sigma_space, BORDER_DEFAULT);
		if (sw == 0)
		{
			method = "cp::highDimensionalGaussianFilter";
			Timer t("bilateral filter: opencv", 0, false);
			cp::highDimensionalGaussianFilter(srcf, srcf, dest, Size(d, d), sigma_color, sigma_space, BORDER_DEFAULT);
			time = t.getTime();
		}
		else if (sw == 1)
		{
			method = "cp::bilateralFilterPermutohedralLattice";
			Timer t("bilateral filter: opencv", 0, false);
			//cp::highDimensionalGaussianFilterPermutohedralLattice(srcf, dest, sigma_color, sigma_space);
			cp::highDimensionalGaussianFilterPermutohedralLatticeTile(srcf, srcf, dest, sigma_color, sigma_space, Size(4, 4));
			time = t.getTime();
		}
		ci(method);
		ci("time %f ms", time);
		ci("PSNR %f dB", getPSNR(dest, ref));
		ci.show();
		if (key == 'd')guiDiff(dest, ref);
		alphaBlend(src, dest, a / 100.0, show);
		imshowScale(wname, show);
		key = waitKey(1);
	}
}