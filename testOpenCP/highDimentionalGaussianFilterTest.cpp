#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void highDimentionalGaussianFilterHSITest()
{
	vector<string> dir;
	vector<Mat> s;
	cv::glob("img/hsi/braga/*.png", dir);
	const int res = 8;
	for (int i = 0; i < dir.size(); i++)
	{
		Mat a = imread(dir[i], 0);
		//const int w = get_simd_floor(a.cols, 8 * res * division.width);
		//const int h = get_simd_floor(a.rows, 8 * res * division.height);
		const int w = get_simd_floor(a.cols, 8 * res);
		const int h = get_simd_floor(a.rows, 8 * res);
		Mat b;
		resize(a(Rect(0, 0, w, h)), b, Size(), 1.0 / res, 1.0 / res);
		s.push_back(convert(b, CV_32F));
	}
	Mat hsi; merge(s, hsi);
	Mat hsipca;
	Mat dest, dest2;

	string wname = "highDimentionalGaussianFilter";
	namedWindow(wname);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 0; createTrackbar("switch", wname, &sw, 2);
	//int r = 20; createTrackbar("r", wname, &r, 200);
	int space10 = 36; createTrackbar("space10", wname, &space10, 200);
	int color10 = 500; createTrackbar("color10", wname, &color10, 2550);
	int clip = 30; createTrackbar("space_clip", wname, &clip, 80);
	int pcach = 1; createTrackbar("pca_ch", wname, &pcach, 33); setTrackbarMin("pca_ch", wname, 1);
	//int rate = 100; createTrackbar("rate", wname, &rate, 100);
	int key = 0;
	Mat show;
	cp::ConsoleImage ci;
	cp::UpdateCheck uc(color10, space10);
	cp::UpdateCheck uc2(color10, space10, sw, pcach);
	Timer t("", TIME_MSEC);
	Mat ref;

	while (key != 'q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color10 / 10.f;
		float sigma_space = space10 / 10.f;
		int d = 2 * (int)ceil(sigma_space * clip * 0.1f) + 1;

		string method;

		if (uc.isUpdate(color10, space10))
		{
			cout << "compute reference 5 sigma" << endl;
			const int d = 2 * (int)ceil(sigma_space * 5.f) + 1;
			cp::highDimensionalGaussianFilter(hsi, hsi, ref, Size(d, d), sigma_color, sigma_space, BORDER_DEFAULT);
		}
		if (uc.isUpdate(color10, space10, sw, pcach) || key == 'r')
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
			cp::cvtColorPCA(hsi, hsipca, pcach);
			//cp::highDimensionalGaussianFilter(hsi, hsi, dest, Size(d, d), sigma_color, sigma_space);
			cp::highDimensionalGaussianFilter(hsi, hsipca, dest, Size(d,d), sigma_color, sigma_space);
			t.getpushLapTime();
		}
		else if (sw == 1)
		{
			method = "cp::bilateralFilterPermutohedralLattice";
			t.start();
			cp::cvtColorPCA(hsi, hsipca, pcach);
			cp::highDimensionalGaussianFilterPermutohedralLattice(hsi, hsipca, dest, sigma_color, sigma_space);
			//cp::highDimensionalGaussianFilterPermutohedralLatticeTile(srcf, srcf, dest, sigma_color, sigma_space, Size(4, 4));

			t.getpushLapTime();
		}
		else if (sw == 2)
		{
			method = "cp::bilateralFilterGaussianKDTree";
			t.start();
			cp::highDimensionalGaussianFilterGaussianKDTree(hsi, hsipca, dest, sigma_color, sigma_space);
			//cp::highDimensionalGaussianFilterGaussianKDTreeTile(srcf, srcf, dest, sigma_color, sigma_space, Size(4, 2), 3.f);
			t.getpushLapTime();
		}

		/*{
			Timer t;
			//cout << hsi.channels() << endl;
			cp::highDimensionalGaussianFilter(hsi, hsi, dest2, Size(d, d), sigma_color, sigma_space, BORDER_DEFAULT);
		}
		cp::imshowSplitScale("hsi", hsi);*/
		ci(method);
		ci("time %7.2f ms (%5d)", t.getLapTimeMedian(), t.getStatSize());
		ci("PSNR %f dB", getPSNR(dest, ref));
		if (key == 'p')ci.push();
		ci.show();
		if (key == 'd')guiDiff(dest, ref);

		cp::cvtColorHSI2BGR(dest, show);
		//alphaBlend(src, dest, a / 100.0, show);
		imshowScale(wname, show);
		key = waitKey(1);
	}
}

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
	int color = 200; createTrackbar("color", wname, &color, 2550);
	int clip = 30; createTrackbar("space_clip", wname, &clip, 80);
	//int rate = 100; createTrackbar("rate", wname, &rate, 100);
	int key = 0;
	Mat show;
	cp::ConsoleImage ci;
	cp::UpdateCheck uc(color, space);
	cp::UpdateCheck uc2(color, space, sw);
	cp::Stat st;
	Timer t;
	Mat ref;
	Mat src64 = cp::convert(srcf, CV_64F);
	while (key != 'q')
	{
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;
		int d = 2 * (int)ceil(sigma_space * clip * 0.1f) + 1;

		string method;

		if (uc.isUpdate(color, space))
		{
			cp::bilateralFilterL2(srcf, ref, (int)ceil(sigma_space * 3.f), sigma_color, sigma_space, BORDER_DEFAULT);
		}
		if (uc.isUpdate(color, space, sw) || key == 'r')
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
			cp::highDimensionalGaussianFilter(src64, src64, dest, Size(d, d), sigma_color, sigma_space, BORDER_DEFAULT);
			//cp::highDimensionalGaussianFilterPermutohedralLattice(srcf, dest, sigma_color, sigma_space);
			//cp::highDimensionalGaussianFilterPermutohedralLattice(srcf, dest, sigma_color, sigma_space);
			//cp::highDimensionalGaussianFilterPermutohedralLatticeTile(srcf, srcf, dest, sigma_color, sigma_space, Size(4, 4));
			//cp::highDimensionalGaussianFilterPermutohedralLatticePCATile(srcf, srcf, dest, sigma_color, sigma_space, 1,Size(4, 4));

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
		ci("time %7.2f ms (%5d)", t.getLapTimeMedian(), t.getStatSize());
		st.push_back(getPSNR(dest, ref));
		ci("PSNR %f %f %f dB", getPSNR(dest, ref), st.getMin(), st.getMax());
		ci.show();
		if (key == 'd')guiDiff(dest, ref);
		alphaBlend(src, dest, a / 100.0, show);
		imshowScale(wname, show);
		key = waitKey(1);
	}
}