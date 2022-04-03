#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

enum class DENOISE_METHOD
{
	BILATERAL,
	GUIDED,
	UNNORMALIZED_BF,

	NONLOCAL,
	JOINTNONLOCAL,
	NONLOCAL_L1,
	JOINTNONLOCAL_L1,
	PATCHBF,
	PATCHBF_L1,
	JOINTPATCHBF,
	JOINTPATCHBF_L1,
	REC_BF,
	BM3D,

	SIZE
};

string getDenoiseMethodName(DENOISE_METHOD method)
{
	string ret = "";
	switch (method)
	{
	case DENOISE_METHOD::BILATERAL:			ret = "BILATERAL"; break;
	case DENOISE_METHOD::GUIDED:			ret = "GUIDED"; break;
	case DENOISE_METHOD::UNNORMALIZED_BF:	ret = "UNNORMALIZED_BF"; break;
	case DENOISE_METHOD::NONLOCAL:			ret = "NONLOCAL"; break;
	case DENOISE_METHOD::NONLOCAL_L1:		ret = "NONLOCAL_L1"; break;
	case DENOISE_METHOD::PATCHBF:			ret = "PATCHBF"; break;
	case DENOISE_METHOD::PATCHBF_L1:		ret = "PATCHBF_L1"; break;
	case DENOISE_METHOD::JOINTNONLOCAL:		ret = "JOINTNONLOCAL"; break;
	case DENOISE_METHOD::JOINTNONLOCAL_L1:	ret = "JOINTNONLOCAL_L1"; break;
	case DENOISE_METHOD::JOINTPATCHBF:		ret = "JOINTPATCHBF"; break;
	case DENOISE_METHOD::JOINTPATCHBF_L1:	ret = "JOINTPATCHBF_L1"; break;
	case DENOISE_METHOD::REC_BF:			ret = "REC_BF"; break;
	case DENOISE_METHOD::BM3D:				ret = "BM3D"; break;
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
	int type = 0; createTrackbar("type:8u/32f", wname, &type, 1);
	int color = 0; createTrackbar("color", wname, &color, 1);
	int sw = (int)DENOISE_METHOD::NONLOCAL; 
	createTrackbar("switch", wname, &sw, (int)DENOISE_METHOD::SIZE - 1);
	int sigma_color10 = 850; createTrackbar("sigma_color", wname, &sigma_color10, 5000);
	int sigma_space10 = 100; createTrackbar("sigma_space", wname, &sigma_space10, 200);
	int r = 4; createTrackbar("r", wname, &r, 100);
	int powexp = 2; createTrackbar("powexp", wname, &powexp, 100);
	int powexp_space = 2; createTrackbar("powexp_space", wname, &powexp_space, 100);
	int tr = 1; createTrackbar("tr", wname, &tr, 20);
	int isSep = 1; createTrackbar("sep", wname, &isSep, 1);

	int noise_s10 = 200; createTrackbar("noise", wname, &noise_s10, 2550);
	int sigma_space2_ = 100; createTrackbar("sigma_space2", wname, &sigma_space2_, 200);
	int key = 0;
	Mat show;

	RecursiveBilateralFilter recbf(src.size());
	cp::ConsoleImage ci;
	cp::UpdateCheck uc_filter(sw, r, tr, isSep);
	cp::UpdateCheck uc_noise(noise_s10, type, color);
	Timer t("time", TIME_MSEC, false);
	Mat noise;
	Mat srcf; src.convertTo(srcf, CV_32F);
	bool isFourceUpdateNoise = false;
	Mat ref;
	while (key != 'q')
	{
		float sigma_color = sigma_color10 / 10.f;
		float sigma_space = sigma_space10 / 10.f;
		float sigma_space2 = sigma_space2_ / 10.f;
		int d = 2 * r + 1;
		int td = 2 * tr + 1;
		
		const bool noiseUpdate = uc_noise.isUpdate(noise_s10, type, color);
		if (noiseUpdate || isFourceUpdateNoise)
		{
			if (color == 0) cvtColor(src, ref, COLOR_BGR2GRAY);
			else src.copyTo(ref);
			if (type == 1) ref.convertTo(ref, CV_32F);
			addNoise(ref, noise, noise_s10 / 10.0);
		}
		if (uc_filter.isUpdate(sw, r, tr, isSep) || key == 'r' || noiseUpdate)
		{
			t.clearStat();
		}

		DENOISE_METHOD method= DENOISE_METHOD(sw);
		t.start();
		switch (method)
		{
		case DENOISE_METHOD::BILATERAL:
			cv::bilateralFilter(noise, dest, d, sigma_color, sigma_space, BORDER_DEFAULT); break;
		case DENOISE_METHOD::GUIDED:
			cp::guidedImageFilter(noise, noise, dest, r, sigma_color * 10.f);break;
		case DENOISE_METHOD::UNNORMALIZED_BF:
			cp::unnormalizedBilateralFilterCenterBlur(noise, dest, r, sigma_color, sigma_space, sigma_space2, false); break;
			//cp::unnormalizedBilateralFilter(noise, dest, r, sigma_color, sigma_space, false); break;
		case DENOISE_METHOD::NONLOCAL:
			//if(isSep) nonLocalMeansFilterSeparable(noise, dest, td, d, sigma_color, powexp); 
			//else nonLocalMeansFilter(noise, dest, td, d, sigma_color, powexp); break;
		{
			Mat a = convert(noise, CV_16S);
			Mat b;
			binaryWeightedRangeFilter(a, b, d, sigma_color, 2);
			b.convertTo(dest, noise.type());
			break;
		}
		case DENOISE_METHOD::JOINTNONLOCAL:
			if (isSep) jointNonLocalMeansFilterSeparable(noise, ref, dest, td, d, sigma_color, powexp, 2); 
			else jointNonLocalMeansFilter(noise, ref, dest, td, d, sigma_color, powexp, 2); break;
		case DENOISE_METHOD::NONLOCAL_L1:
			if (isSep) nonLocalMeansFilterSeparable(noise, dest, td, d, sigma_color, powexp, 1);
			else nonLocalMeansFilter(noise, dest, td, d, sigma_color, powexp, 1); break;
		case DENOISE_METHOD::JOINTNONLOCAL_L1:
			if (isSep) jointNonLocalMeansFilterSeparable(noise, ref, dest, td, d, sigma_color, powexp, 1);
			else jointNonLocalMeansFilter(noise, ref, dest, td, d, sigma_color, powexp, 1); break;
		case DENOISE_METHOD::PATCHBF:
			if (isSep) patchBilateralFilterSeparable(noise, dest, td, d, sigma_color, powexp, 2, sigma_space, powexp_space);
			else patchBilateralFilter(noise, dest, td, d, sigma_color, powexp, 2, sigma_space, powexp_space); break;
		case DENOISE_METHOD::PATCHBF_L1:
			if (isSep) patchBilateralFilterSeparable(noise, dest, td, d, sigma_color, powexp, 1, sigma_space, powexp_space);
			else patchBilateralFilter(noise, dest, td, d, sigma_color, powexp, 1, sigma_space, powexp_space); break;
		case DENOISE_METHOD::JOINTPATCHBF:
			if (isSep) jointPatchBilateralFilterSeparable(noise, ref, dest, td, d, sigma_color, powexp, 2, sigma_space, powexp_space);
			else jointPatchBilateralFilter(noise, ref, dest, td, d, sigma_color, powexp, 2, sigma_space, powexp_space); break;
		case DENOISE_METHOD::JOINTPATCHBF_L1:
			if (isSep) jointPatchBilateralFilterSeparable(noise, ref, dest, td, d, sigma_color, powexp, 1, sigma_space, powexp_space);
			else jointPatchBilateralFilter(noise, ref, dest, td, d, sigma_color, powexp, 1, sigma_space, powexp_space); break;
		case DENOISE_METHOD::REC_BF:
			recbf(noise, dest, sigma_color, sigma_space); break;
		case DENOISE_METHOD::BM3D:
			xphoto::bm3dDenoising(noise, dest, sigma_color, 8, 16, 2500, 400, 16, 1, 2.f, 4, 1, cv::xphoto::HAAR); break;
		default:
			xphoto::dctDenoising(noise, dest, sigma_color, 16);
			break;
		}
		t.getpushLapTime();

		ci(getDenoiseMethodName(DENOISE_METHOD(sw)));
		if (isFourceUpdateNoise) ci("noise  %6.3f dB, update (swich 'n' key)", PSNR(ref, noise));
		else ci("noise  %6.3f dB, const (swich 'n' key)", PSNR(ref, noise));

		ci("filter %6.3f dB", PSNR(ref, dest));
		ci("time   %f ms %d", t.getLapTimeMedian(), t.getStatSize());
		if (key == 'p')ci.push();
		ci.show();

		cp::dissolveSlideBlend(noise, dest, dest);
		cp::alphaBlend(src, dest, a / 100.0, show);
		cp::imshowScale(wname, show);
		key = waitKey(1);
		if (key == 'n')isFourceUpdateNoise = isFourceUpdateNoise ? false : true;

	}
}
