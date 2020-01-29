#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testLocalPSNR(Mat& ref)
{
	Mat noise;
	addJPEGNoise(ref, noise, 30);
	//addNoise(ref, noise, 50);
	//blur(ref, noise, Size(5, 5));
	
	guiLocalPSNRMap(ref, noise);
}

void testPSNRTime(Mat& src)
{
	CV_Assert(src.channels() == 3);

	//some image processing 
	Mat ref; src.convertTo(ref, CV_64F);
	Mat dst;
	addNoise(ref, dst, 1, 0);
	//addNoise(ref, dst, 0.0000001, 0);//for double
	/*
	GaussianBlur(ref, ref, Size(19, 19), 10);
	ref.convertTo(dst, CV_32F);
	dst.convertTo(dst, CV_64F);
	*/

	Mat ref_8UC3, dst_8UC3;
	ref.convertTo(ref_8UC3, CV_8U);
	dst.convertTo(dst_8UC3, CV_8U);
	Mat ref32FC3, dst32FC3;
	ref.convertTo(ref32FC3, CV_32F);
	dst.convertTo(dst32FC3, CV_32F);
	Mat ref64FC3, dst64FC3;
	ref.convertTo(ref64FC3, CV_64F);
	dst.convertTo(dst64FC3, CV_64F);

	const int iteration = 100;
	{
		Timer t("OpenCV 8U", 0, false);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			PSNR(dst_8UC3, ref_8UC3);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << PSNR(dst_8UC3, ref_8UC3) << endl;
	}
	{
		Timer t("Prop 8U", 0, false);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			getPSNR(dst_8UC3, ref_8UC3, 0, PSNR_8U);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << getPSNR(dst_8UC3, ref_8UC3, 0, PSNR_8U) << endl;
	}
	{
		Timer t("PropC 8U", 0, false);
		PSNRMetrics psnr;
		psnr.setReference(ref_8UC3, 0, PSNR_8U);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			psnr.getPSNRPreset(dst_8UC3, 0, PSNR_8U);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << psnr.getPSNRPreset(dst_8UC3, 0, PSNR_8U) << endl;
	}

	{
		Timer t("OpenCV 32F", 0, false);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			PSNR(dst32FC3, ref32FC3);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << PSNR(dst32FC3, ref32FC3) << endl;
	}
	{
		Timer t("Prop 32F", 0, false);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			getPSNR(dst32FC3, ref32FC3);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << getPSNR(dst32FC3, ref32FC3) << endl;
	}
	{
		Timer t("PropC 32F", 0, false);
		PSNRMetrics psnr;
		psnr.setReference(ref32FC3);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			psnr.getPSNRPreset(dst32FC3);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << psnr.getPSNRPreset(dst32FC3) << endl;
	}

	{
		Timer t("OpenCV 64F", 0, false);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			PSNR(dst64FC3, ref64FC3);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << PSNR(dst64FC3, ref64FC3) << endl;
	}
	{
		Timer t("Prop 64F", 0, false);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			getPSNR(dst64FC3, ref64FC3, 0, PSNR_64F);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << getPSNR(dst64FC3, ref64FC3, 0, PSNR_64F) << endl;
	}
	{
		Timer t("PropC 64F", 0, false);
		PSNRMetrics psnr;
		psnr.setReference(ref64FC3, 0, PSNR_64F);
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			psnr.getPSNRPreset(dst64FC3, 0, PSNR_64F);
			t.getpushLapTime();
		}
		t.getLapTimeMedian(true);
		cout << psnr.getPSNRPreset(dst64FC3, 0, PSNR_64F) << endl;
	}

	return;
}

void testPSNRAccuracy(Mat& src)
{

	CV_Assert(src.channels() == 3);

	//some image processing 
	Mat ref; src.convertTo(ref, CV_64F);
	Mat dst;
	addNoise(ref, dst, 1, 0);
	//addNoise(ref, dst, 0.0000001, 0);//for double
	/*
	GaussianBlur(ref, ref, Size(19, 19), 10);
	ref.convertTo(dst, CV_32F);
	dst.convertTo(dst, CV_64F);
	*/

	Mat ref_8UC3, dst_8UC3;
	ref.convertTo(ref_8UC3, CV_8U);
	dst.convertTo(dst_8UC3, CV_8U);
	Mat ref32FC3, dst32FC3;
	ref.convertTo(ref32FC3, CV_32F);
	dst.convertTo(dst32FC3, CV_32F);
	Mat ref64FC3, dst64FC3;
	ref.convertTo(ref64FC3, CV_64F);
	dst.convertTo(dst64FC3, CV_64F);

	int bb;
	int compare_method;
	cout << "function call test" << endl;
	cout << "==================" << endl;
	cout << "opencv ref function of PSNR()" << endl;
	cout << " 8U  8U: " << PSNR(dst_8UC3, ref_8UC3) << endl;
	cout << "32F 32F: " << PSNR(dst32FC3, ref32FC3) << endl;
	cout << "64F 64F: " << PSNR(dst64FC3, ref64FC3) << endl;
	cout << "OpenCV supports only the same depth type of inputs." << endl;
	cout << endl;

	for (int precision = 0; precision < PSNR_PRECISION_SIZE; precision++)
	{
		//if (precision == PSNR_8U)continue;
		//if (precision == PSNR_32F)continue;
		//if (precision == PSNR_64F)continue;
		//if (precision == PSNR_KAHAN_64F)continue;
		cout << getPSNR_PRECISION(precision) << endl;

		cout << "bounding_box=0, precision=" + getPSNR_PRECISION(precision) + "compare=PSNR_ALL" << endl;
		bb = 0;
		compare_method = PSNR_ALL;
		cout << "dst ref: PSNR [dB]" << endl;
		cout << " 8U  8U: " << getPSNR(dst_8UC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << " 8U 32F: " << getPSNR(dst_8UC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << " 8U 64F: " << getPSNR(dst_8UC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "32F  8U: " << getPSNR(dst32FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "32F 32F: " << getPSNR(dst32FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "32F 64F: " << getPSNR(dst32FC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "64F  8U: " << getPSNR(dst64FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "64F 32F: " << getPSNR(dst64FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "64F 64F: " << getPSNR(dst64FC3, ref64FC3, bb, precision, compare_method) << endl;

		cout << endl;
		cout << "bounding_box=0, precision=" + getPSNR_PRECISION(precision) + "compare=PSNR_Y" << endl;
		bb = 0;
		compare_method = PSNR_Y;
		cout << "dst ref: PSNR [dB]" << endl;
		cout << " 8U  8U: " << getPSNR(dst_8UC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << " 8U 32F: " << getPSNR(dst_8UC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << " 8U 64F: " << getPSNR(dst_8UC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "32F  8U: " << getPSNR(dst32FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "32F 32F: " << getPSNR(dst32FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "32F 64F: " << getPSNR(dst32FC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "64F  8U: " << getPSNR(dst64FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "64F 32F: " << getPSNR(dst64FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "64F 64F: " << getPSNR(dst64FC3, ref64FC3, bb, precision, compare_method) << endl;

		cout << endl;
		cout << "bounding_box=0, precision=" + getPSNR_PRECISION(precision) + "compare=PSNR_B" << endl;
		bb = 0;
		compare_method = PSNR_B;
		cout << "dst ref: PSNR [dB]" << endl;
		cout << " 8U  8U: " << getPSNR(dst_8UC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << " 8U 32F: " << getPSNR(dst_8UC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << " 8U 64F: " << getPSNR(dst_8UC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "32F  8U: " << getPSNR(dst32FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "32F 32F: " << getPSNR(dst32FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "32F 64F: " << getPSNR(dst32FC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "64F  8U: " << getPSNR(dst64FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "64F 32F: " << getPSNR(dst64FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "64F 64F: " << getPSNR(dst64FC3, ref64FC3, bb, precision, compare_method) << endl;

		cout << endl;
		cout << "bounding_box=0, precision=" + getPSNR_PRECISION(precision) + "compare=PSNR_G" << endl;
		bb = 0;
		compare_method = PSNR_G;
		cout << "dst ref: PSNR [dB]" << endl;
		cout << " 8U  8U: " << getPSNR(dst_8UC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << " 8U 32F: " << getPSNR(dst_8UC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << " 8U 64F: " << getPSNR(dst_8UC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "32F  8U: " << getPSNR(dst32FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "32F 32F: " << getPSNR(dst32FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "32F 64F: " << getPSNR(dst32FC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "64F  8U: " << getPSNR(dst64FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "64F 32F: " << getPSNR(dst64FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "64F 64F: " << getPSNR(dst64FC3, ref64FC3, bb, precision, compare_method) << endl;

		cout << endl;
		cout << "bounding_box=0, precision=" + getPSNR_PRECISION(precision) + "compare=PSNR_R" << endl;
		bb = 0;
		compare_method = PSNR_R;
		cout << "dst ref: PSNR [dB]" << endl;
		cout << " 8U  8U: " << getPSNR(dst_8UC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << " 8U 32F: " << getPSNR(dst_8UC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << " 8U 64F: " << getPSNR(dst_8UC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "32F  8U: " << getPSNR(dst32FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "32F 32F: " << getPSNR(dst32FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "32F 64F: " << getPSNR(dst32FC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "64F  8U: " << getPSNR(dst64FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "64F 32F: " << getPSNR(dst64FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "64F 64F: " << getPSNR(dst64FC3, ref64FC3, bb, precision, compare_method) << endl;

		cout << endl;
		cout << "bounding_box=20, precision=" + getPSNR_PRECISION(precision) + "compare=PSNR_ALL" << endl;
		bb = 20;
		compare_method = PSNR_ALL;
		cout << "dst ref: PSNR [dB]" << endl;
		cout << " 8U  8U: " << getPSNR(dst_8UC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << " 8U 32F: " << getPSNR(dst_8UC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << " 8U 64F: " << getPSNR(dst_8UC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "32F  8U: " << getPSNR(dst32FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "32F 32F: " << getPSNR(dst32FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "32F 64F: " << getPSNR(dst32FC3, ref64FC3, bb, precision, compare_method) << endl;
		cout << "64F  8U: " << getPSNR(dst64FC3, ref_8UC3, bb, precision, compare_method) << endl;
		cout << "64F 32F: " << getPSNR(dst64FC3, ref32FC3, bb, precision, compare_method) << endl;
		cout << "64F 64F: " << getPSNR(dst64FC3, ref64FC3, bb, precision, compare_method) << endl;

		cout << "==================" << endl;
	}
}

void testPSNR(Mat& src)
{
	testPSNRAccuracy(src);
	testPSNRTime(src);
}