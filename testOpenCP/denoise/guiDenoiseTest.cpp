#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

static enum method
{
	BILATERAL,
	GUIDED,
	BINARYRANGE,
	NONLOCAL,
	NONLOCAL_L1,
	PATCHBF,
	PATCHBF_L1,
	REC_BF,
	BM3D,

	SIZE
};

string getDenoiseMethodName(int method)
{
	string ret = "";
	switch (method)
	{
	case BILATERAL: ret = "BILATERAL"; break;
	case GUIDED: ret = "GUIDED"; break;
	case BINARYRANGE: ret = "BINARYRANGE"; break;
	case NONLOCAL: ret = "NONLOCAL"; break;
	case NONLOCAL_L1: ret = "NONLOCAL_L1"; break;
	case PATCHBF: ret = "PATCHBF"; break;
	case PATCHBF_L1: ret = "PATCHBF_L1"; break;
	case REC_BF: ret = "REC_BF"; break;
	case BM3D: ret = "BM3D"; break;
	default:
		break;
	}
	return ret;
}

void guiDenoiseTest(Mat& src)
{
	Mat dest, dest2;

	string wname = "denoise";
	namedWindow(wname);

	int a = 0; createTrackbar("a", wname, &a, 100);
	
	int sw = NONLOCAL; createTrackbar("switch", wname, &sw, method::SIZE-1);
	int sigma_color10 = 400; createTrackbar("sigma_color", wname, &sigma_color10, 5000);
	int sigma_space10 = 100; createTrackbar("sigma_space", wname, &sigma_space10, 200);
	int r = 4; createTrackbar("r", wname, &r, 100);
	int powexp = 2; createTrackbar("powexp", wname, &powexp, 100);
	int powexp_space = 2; createTrackbar("powexp_space", wname, &powexp_space, 100);
	int tr = 1; createTrackbar("tr", wname, &tr, 20);

	int noise_s10 = 200; createTrackbar("noise", wname, &noise_s10, 2550);
	int key = 0;
	Mat show;

	RecursiveBilateralFilter recbf(src.size());
	cp::ConsoleImage ci;
	cp::UpdateCheck uc(sw);
	cp::UpdateCheck ucnoise(noise_s10);
	Timer t("time", TIME_MSEC, false);
	Mat noise;
	Mat srcf; src.convertTo(srcf, CV_32F);
	bool isFourceUpdateNoise = false;
	
	while (key != 'q')
	{
		float sigma_color = sigma_color10 / 10.f;
		float sigma_space = sigma_space10 / 10.f;
		int d = 2 * r + 1;
		int td = 2 * tr + 1;

		
		if (ucnoise.isUpdate(noise_s10)||isFourceUpdateNoise)
		{
			addNoise(src, noise, noise_s10 / 10.0);
		}

		if (uc.isUpdate(sw))
		{
			t.clearStat();
		}
		if (sw == 0)
		{
			t.start();
			//bilateralFilter(noise, dest, Size(d, d), sigma_color, sigma_space, FILTER_RECTANGLE);
			bilateralFilter(noise, dest, d, sigma_color, sigma_space, FILTER_RECTANGLE);
			t.getpushLapTime();
		}
		else if (sw == 1)
		{
			t.start();
			//GaussianBlur(noise,dest,Size(d,d),sigma_space);
			guidedImageFilter(noise, noise, dest, r, sigma_color * 10.f);
			//bilateralFilter(noise,dest,Size(d,d),sigma_color,sigma_space,FILTER_SEPARABLE);
			t.getpushLapTime();
		}
		else if (sw == 2)
		{
			t.start();
			binalyWeightedRangeFilter(noise, dest, Size(d, d), sigma_color);
			t.getpushLapTime();
		}
		else if (sw == 3)
		{
			t.start();
			nonLocalMeansFilter(noise, dest, td, d, sigma_color, powexp);
			t.getpushLapTime();
		}
		else if (sw == 4)
		{
			t.start();
			nonLocalMeansFilter(noise, dest, td, d, sigma_color, powexp, 1);
			//nonLocalMeansFilterL1PatchDistance(noise, dest, td, d, sigma_color, powexp);
			t.getpushLapTime();
		}
		else if (sw == 5)
		{
			t.start();
			patchBilateralFilter(noise, dest, td, d, sigma_color, powexp, 2, sigma_space, powexp_space);
			t.getpushLapTime();
		}
		else if (sw == 6)
		{
			t.start();
			patchBilateralFilter(noise, dest, td, d, sigma_color, powexp, 1, sigma_space, powexp_space);
			//nonLocalMeansFilterL1PatchDistance(noise, dest, td, d, sigma_color, powexp);
			t.getpushLapTime();
		}
		else if (sw == 7)
		{
			t.start();
			recbf(noise, dest, sigma_color, sigma_space);
			t.getpushLapTime();
		}
		else if (sw == 8)
		{
			/*
			CalcTime t("DCT Denoising");
			Mat temp;
			copyMakeBorder(noise, temp, 8, 8, 8, 8, BORDER_REFLECT);
			xphoto::dctDenoising(temp, temp, sigma_color, 16);
			Mat(temp(Rect(8, 8, noise.cols, noise.rows))).copyTo(dest);
			*/
			t.start();
			xphoto::bm3dDenoising(noise, dest, sigma_color, 8, 16, 2500, 400, 16, 1, 2.f, 4, 1, cv::xphoto::HAAR);
			t.getpushLapTime();
		}

		ci(getDenoiseMethodName(sw));
		if(isFourceUpdateNoise) ci("noise  %6.3f dB, update (swich 'n' key)", PSNR(src, noise));
		else ci("noise  %6.3f dB, const (swich 'n' key)", PSNR(src, noise));
		
		ci("filter %6.3f dB", PSNR(src, dest));
		ci("time   %f ms %d", t.getLapTimeMedian(), t.getStatSize());
		if (key == 'p')ci.push();
		ci.show();

		cp::dissolveSlideBlend(noise, dest, dest);
		cp::alphaBlend(src, dest, a / 100.0, show);
		cv::imshow(wname, show);
		key = waitKey(1);
		if (key == 'n')isFourceUpdateNoise = isFourceUpdateNoise ? false : true;
		
	}
}
