#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void highDimentionalGaussianFilterTest(Mat& src)
{
	//resize(src, src, Size(4000, 3000));
	Mat srcf = convert(src, CV_32F);
	Mat dest, dest2;

	string wname = "highDimentionalGaussianFilter";
	namedWindow(wname);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 1; createTrackbar("switch", wname, &sw, 2);
	//int r = 20; createTrackbar("r", wname, &r, 200);
	int space = 36; createTrackbar("space", wname, &space, 200);
	int color = 500; createTrackbar("color", wname, &color, 2550);
	int clip = 30; createTrackbar("space_clip", wname, &clip, 80);
	//int rate = 100; createTrackbar("rate", wname, &rate, 100);
	int key = 0;
	Mat show;
	cp::ConsoleImage ci;
	cp::UpdateCheck uc(color, space);
	cp::UpdateCheck uc2(color, space, sw);
	Timer t;
	Mat ref;
	while (key != 'q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;
		int d = 2 * (int)ceil(sigma_space * clip*0.1f) + 1;

		string method;
		
		if (uc.isUpdate(color, space))
		{
			cp::bilateralFilterL2(srcf, ref, (int)ceil(sigma_space * 3.f), sigma_color, sigma_space, BORDER_DEFAULT);
		}
		if (uc.isUpdate(color, space, sw) || key=='r')
		{
			t.clearStat();
		}

		dest.setTo(0);
		if (sw == 0)
		{
			method = "cp::highDimensionalGaussianFilter";
			//cp::nonLocalMeansFilter(srcf, dest, Size(6, 6), Size(22, 22), sigma_space, -1.0, 0);
			//cp::nonLocalMeansFilter(srcf, dest, Size(6, 6), Size(22, 22), sigma_color, sigma_space, 0);
			t.start();
			cp::highDimensionalGaussianFilter(srcf, srcf, dest, Size(d, d), sigma_color, sigma_space, BORDER_DEFAULT);
			//cp::highDimensionalGaussianFilterPermutohedralLattice(srcf, dest, sigma_color, sigma_space);
			t.getpushLapTime();
		}
		else if (sw == 1)
		{
			method = "cp::bilateralFilterPermutohedralLattice";
			t.start();
			//cp::highDimensionalGaussianFilterPermutohedralLattice(srcf, dest, sigma_color, sigma_space);
			cp::highDimensionalGaussianFilterPermutohedralLatticeTile(srcf, srcf, dest, sigma_color, sigma_space, Size(4, 4));

			t.getpushLapTime();
		}
		else if (sw == 2)
		{
			method = "cp::bilateralFilterGaussianKDTree";
			t.start();
			//cp::nonLocalMeansFilter(srcf, dest, Size(6, 6), Size(22, 22), sigma_color, sigma_space, 4);
			cp::highDimensionalGaussianFilterGaussianKDTreeTile(srcf, srcf, dest, sigma_color, sigma_space, Size(4, 2), 3.f);
			t.getpushLapTime();
		}

		ci(method);
		ci("time %7.2f ms (%5d)", t.getLapTimeMedian(),t.getStatSize());
		ci("PSNR %f dB", getPSNR(dest, ref));
		ci.show();
		if (key == 'd')guiDiff(dest, ref);
		alphaBlend(src, dest, a / 100.0, show);
		imshowScale(wname, show);
		key = waitKey(1);
	}
}