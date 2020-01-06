#include <opencp.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cp;

void deblurring(Mat& src, Mat& dest, float sigma, int sw, float sigmanoise, const float eps, bool isSobolev);

void ibp_demo(const Mat& src, Mat& dest, const Size ksize, const float sigma, const float backprojection_sigma, const float lambda, const int iteration, Mat& init=Mat())
{
	Mat srcf;
	Mat destf;
	Mat subf;
	src.convertTo(srcf, CV_32FC3);

	if (init.empty()) src.convertTo(destf, CV_32FC3);
	else init.convertTo(destf, CV_32F);
	Mat bdest;
	Mat show;
	int key = 0;
	while (true)
	{
		srcf.copyTo(destf);
		for (int i = 0; i < iteration; i++)
		{
			GaussianBlur(destf, bdest, ksize, sigma);
			subtract(srcf, bdest, subf);
			//double e = norm(subf);

			if (backprojection_sigma > 0.f)
				GaussianBlur(subf, subf, ksize, backprojection_sigma);


			destf += lambda * subf;
			/*if(i==0)fma(subf, 3, destf);
			else if (i==1)fma(subf, 2, destf);
			else	fma(subf, lambda, destf);*/
			//fma(subf, lambda, destf);

			//l *= lambdaamp;
			destf.convertTo(show, CV_8U);
			putText(show, format("iteration %03d", i), Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, Scalar::all(0), 2);
			imshow("ibp demo", show);
			key = waitKey(30);
			if (key == 'q') break;
		}
		if (key == 'q') break;
	}
	destf.convertTo(dest, src.depth());
}

void guiIterativeBackProjectionTest(Mat& src)
{
	Mat dest, dest2;

	string wname = "Iterative Back Projection";
	namedWindow(wname);
	namedWindow("diff");
	int diff = 10; createTrackbar("a", "diff", &diff, 200);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 2; createTrackbar("switch", wname, &sw, 7);

	int r_sigma = 15; createTrackbar("rev_sigma", wname, &r_sigma, 2000);
	//int r = 4; createTrackbar("r", wname, &r, 200);

	int eps = 10; createTrackbar("eps*0.001", wname, &eps, 100);
	int eps2 = 10;  createTrackbar("eps*0.01", wname, &eps2, 100);
	//int r_sigma2 = 10; createTrackbar("r_sigma2", wname, &r_sigma2, 2000);
	//int ks = 1000; createTrackbar("ks", wname, &ks, 500);
	int iter = 10; createTrackbar("iteration", wname, &iter, 800);
	int l = 10; createTrackbar("lambda", wname, &l, 30);
	int bss = r_sigma; createTrackbar("bss", wname, &bss, 255);//for back projection
	int cs = 5; createTrackbar("bcs", wname, &cs, 255);//for bilateral back projection

	//int denormal = 80; createTrackbar("denormal", wname, &denormal, 90);
	int d_sigma = 15; createTrackbar("d_sigma", wname, &d_sigma, 200);
	int noise_s = 0; createTrackbar("d_noise", wname, &noise_s, 2000);
	int noise_th = 100; createTrackbar("th_noise", wname, &noise_th, 2000);

	int ds = 2;  createTrackbar("BFs", wname, &ds, 20);
	int dc = 0;  createTrackbar("BFc", wname, &dc, 50);


	int nth = 100;  createTrackbar("nth", wname, &nth, 2000);
	int psw = 1; createTrackbar("psw", wname, &psw, 4);
	int sovolev = 0; createTrackbar("sovolev", wname, &sovolev, 1);
	int key = 0;
	Mat show;
	Mat srcf; src.convertTo(srcf, CV_32F);
	Mat blurred;
	ConsoleImage ci;

	int border = 10;
	while (key != 'q')
	{
		double e = eps * 0.01;
		double e2 = eps2 * 0.01;
		e = e * e;
		e2 = e2 * e2;
		double noiselevel = noise_s * 0.01;
		double color_sigma = cs / 10.0;
		int r = int((d_sigma / 10.0) * 3.0);
		int d = 2 * r + 1;
		double lambda = l / 10.0;

		//Mat test; GaussianFilterDCT32f(srcf, test, d_sigma / 10.0, 0); test.convertTo(blurred, CV_8U);
		//Mat test; GaussianFilterDCT32f(srcf, blurred, d_sigma / 10.0, 0); 
		GaussianBlur(src, blurred, Size(d, d), d_sigma / 10.0, 0.0);
		//GaussianBlur(src, blurred, Size(d, d), d_sigma / 10.0, 0.0);

		Mat show;
		Mat raw;
		addNoise(blurred, raw, noiselevel);

		if (key == 'd')
		{
			ibp_demo(raw, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, 100);
		}

		float denoiseth = nth * 0.01;
		{
			ci(format("src: %05.2f %0.2f", 0, PSNR64F(raw, src)));
			if (sw == 0) raw.copyTo(show);
		}

		if (dc == 0)
		{
			raw.copyTo(blurred);
		}
		else
		{
			//deblurring(raw, blurred, 0, 0, noise_th*0.01, 0);
			//bilateralFilter(raw, blurred, d, dc, ds);
		}
		if (sw == 1) blurred.copyTo(show);
		//cout << "before:" << PSNR(src, blurred) << endl;
		//ci(format("BF: %4.2f %0.2f", 0, PSNR64F(blurred, src)));

		dest.setTo(0);
		//if(sw==0)
		{
			Timer t("Prop");
			//iterativeBackProjectionDeblurDelta(blurred, dest, Size(d, d), r_sigma / 10.0, lambda, iter);
			//LucyRichardsonGauss(blurred, dest, Size(d, d), r_sigma / 10.0, iter);


			//deblurring(blurred, dest, r_sigma / 10.0, 0, denoiseth, e2, false);
			//LucyRichardsonGaussTikhonov(dest, dest, Size(d, d), r_sigma / 10.0, lambda*0.0001, iter);
			iterativeBackProjectionDeblurGaussian(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, iter);
			//Mat temp;
			//GaussianBlur(blurred, temp, Size(d, d), r_sigma / 10.0);
			//dest = blurred + lambda*(blurred- temp);
			//bool flag = (sovolev == 0) ? false: true;
			//deblurring(blurred, dest, r_sigma / 10.0, psw, denoiseth, e2, flag);
			double time = t.getTime();
			ci(format("IBP: %05.2f %0.2f", time, cp::PSNR64F(dest, src)));
			//ci(format("DCT: %05.2f %0.2f", time, calcImageQualityMetric(dest, src, IQM_PSNR, 10)));
			//ci(format("DCT: %05.2f %0.2f %0.4f", t.getTime(), cp::PSNR64F(dest, src), cp::calcImageQualityMetric(dest, src, IQM_CWSSIM)));
			if (sw == 2) dest.copyTo(show);
		}

		//if(sw==0)
		{
			Timer t("ibp");
			//LucyRichardsonGaussTikhonov(blurred, dest, Size(d, d), r_sigma / 10.0, th*0.0001, iter);
			//iterativeBackProjectionDeblurGaussian(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, iter);


			deblurDCTWiener(blurred, dest, r_sigma / 10.0, e2);
			//deblurDCTWiener(dest, dest, r_sigma / 10.0, e2);
			double time = t.getTime();
			ci(format("IBP: %05.2f %0.2f", time, PSNR64F(dest, src)));
			//ci(format("IBP: %05.2f %0.2f %0.4f", t.getTime(), PSNR64F(dest, src), cp::calcImageQualityMetric(dest, src, IQM_MSSSIM_FAST)));
			if (sw == 3) dest.copyTo(show);
		}

		Scalar mean;
		Scalar std;
		cv::meanStdDev(blurred, mean, std);

		ci(format("FFT: %0.4f %0.4f", e, (noiselevel*noiselevel) / (std[0] * std[0])));
		{
			Timer t("wiener");

			//deblurring(blurred, dest, r_sigma / 10.0, psw, denoiseth, e2);
			//iterativeBackProjectionDeblurGaussian(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, iter, dest);
			//deblurring(blurred, dest, r_sigma / 10.0, 0, denoiseth, e2);
			//deblurDCTTihkonov(dest, dest, r_sigma / 10.0, e2);
			//deblurDCTWiener(blurred, dest, r_sigma / 10.0, e2);
			//wienerDeconvolutionGauss(blurred, dest, Size(d, d), r_sigma / 10.f, e);
			//iterativeBackProjectionDeblurGuidedImageFilter(blurred, dest, Size(d, d), eps*0.1, r_sigma/10.0,  lambda, iter);
			//ci(format("FFT: %0.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			ci(format("FFT: %0.2f %0.2f %0.4f", t.getTime(), PSNR64F(dest, src), cp::calcImageQualityMetric(dest, src, IQM_CWSSIM)));
			if (sw == 4) dest.copyTo(show);
		}

		//if(sw==1)
		{
			Timer t("DCT ");
			//iterativeBackProjectionDeblurGaussianFast(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, iter);
			//iterativeBackProjectionDeblurGaussianTV(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, th,iter);
			//iterativeBackProjectionDeblurBilateral(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, color_sigma, lambda, iter);
			//deblurDCT32f(blurred, dest, r_sigma / 10.f, e2, 1);
//			deblurdenoiseDCT32f(blurred, dest, r_sigma / 10.f, e2, denoiseth);
			ci(format("F-DCT inv: %05.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			//ci(format("F-DCT: %05.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			if (sw == 5) dest.copyTo(show);
		}

		{
			Timer t("DCT ");
			//iterativeBackProjectionDeblurGaussianFast(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, iter);
			//iterativeBackProjectionDeblurGaussianTV(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, th,iter);
			//iterativeBackProjectionDeblurBilateral(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, color_sigma, lambda, iter);
			//deblurDCT32f(blurred, dest, r_sigma / 10.f, e2, 1);
//			deblurdenoiseDCTWiener32f(blurred, dest, r_sigma / 10.f, e2, denoiseth);
			ci(format("F-DCT wie: %05.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			//ci(format("F-DCT: %05.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			if (sw == 6) dest.copyTo(show);
		}

		{
			Timer t("DCT ");
			//iterativeBackProjectionDeblurGaussianFast(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, iter);
			//iterativeBackProjectionDeblurGaussianTV(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, lambda, th,iter);

//			deblurring(blurred, dest, r_sigma / 10.0, psw, denoiseth, e2);
			iterativeBackProjectionDeblurBilateral(blurred, dest, Size(d, d), r_sigma / 10.0, bss*0.1, color_sigma, lambda, iter, dest);
			//deblurDCT32f(blurred, dest, r_sigma / 10.f, e2, 1);
			//deblurdenoiseDCTWiener32f(blurred, dest, r_sigma / 10.f, e2, denoiseth);
			ci(format("BBP wie: %05.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			//ci(format("F-DCT: %05.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			if (sw == 7) dest.copyTo(show);
		}

		//if (sw == 2)
		/*{
			CalcTime t("DCT");
			//fastDeblurringHyperLaplacian(blurred, dest, d, r_sigma / 10.f);
			deblurDCT32f(blurred, dest, r_sigma / 10.f, eps*0.0001, ks);
			ci(format("DCT: %0.2f %0.2f", t.getTime(), PSNR64F(dest, src)));
			if (sw == 5) dest.copyTo(show);
		}*/
		/*
		//if (sw == 3)
		{
			CalcTime t("DCT");

			Mat blurf, destf;
			//GaussianBlur(blurred, blurf, Size(5, 5), 1.0);
			//deblurDCT32f(blurf, destf, r_sigma / 10.f + 0.5, eps*0.0001, ks);
			//destf.convertTo(dest, CV_8U);
			//deblurDCT64f(blurred, dest, r_sigma / 10, eps*0.0001, ks);

			ci(format("DCT: %0.2f %0.2f", t.getTime(), PSNR(dest, src)));
			if (sw == 4) dest.copyTo(show);
		}
		*/
		//patchBlendImage(blurred,dest,dest,Scalar(255,255,255));
		//alphaBlend(src, show, a / 100.0, show);

		Mat aa;
		show.convertTo(aa, CV_8U);
		Mat dmap;
		absdiff(src, show, dmap);
		//absdiff(srcf, show, dmap);
		Mat bb; Mat(dmap*(float)diff).convertTo(bb, CV_8U);
		imshow("diff", bb);
		//alphaBlend(blurred, show, a / 100.0, aa);

		imshow(wname, aa);
		ci.show();
		key = waitKey(1);
	}
}


//----------------------------------------------------------------------------

// forAAN
//const float a1 = sqrt(.5);
//const float a2 = sqrt(2.) * cos(3. / 16. * 2 * CV_PI);
//const float a3 = a1;
//const float a4 = sqrt(2.) * cos(1. / 16. * 2 * CV_PI);
//const float a5 = cos(3. / 16. * 2 * CV_PI);
//
//const float s0 = (cos(0)*sqrt(.5) / 2) / (1);  // 0.353553
//const float s1 = (cos(1.*CV_PI / 16) / 2) / (-a5 + a4 + 1);  // 0.254898
//const float s2 = (cos(2.*CV_PI / 16) / 2) / (a1 + 1);  // 0.270598
//const float s3 = (cos(3.*CV_PI / 16) / 2) / (a5 + 1);  // 0.300672
//const float s4 = s0;  // (cos(4.*CV_PI/16)/2)/(1       );
//const float s5 = (cos(5.*CV_PI / 16) / 2) / (1 - a5);  // 0.449988
//const float s6 = (cos(6.*CV_PI / 16) / 2) / (1 - a1);  // 0.653281
//const float s7 = (cos(7.*CV_PI / 16) / 2) / (a5 - a4 + 1);  // 1.281458


static const float a1 = (float)cos(4.0 * CV_PI / 16.0);
static const float a2 = (float)(cos(2.0 * CV_PI / 16.0) - cos(6.0 * CV_PI / 16.0));
static const float a3 = (float)cos(4.0 * CV_PI / 16.0);
static const float a4 = (float)(cos(6.0 * CV_PI / 16.0) + cos(2.0 * CV_PI / 16.0));
static const float a5 = (float)cos(6.0 * CV_PI / 16.0);

static const float s0 = (float)(1.0 / (2.0 * sqrt(2.0)));  // 0.353553
static const float s1 = (float)(1.0 / (4.0 * cos(1.0 * CV_PI / 16.0)));  // 0.254898
static const float s2 = (float)(1.0 / (4.0 * cos(2.0 * CV_PI / 16.0)));  // 0.270598
static const float s3 = (float)(1.0 / (4.0 * cos(3.0 * CV_PI / 16.0)));  // 0.300672
static const float s4 = (float)(1.0 / (4.0 * cos(4.0 * CV_PI / 16.0)));  // (cos(4.*CV_PI/16)/2)/(1       );
static const float s5 = (float)(1.0 / (4.0 * cos(5.0 * CV_PI / 16.0)));  // 0.449988
static const float s6 = (float)(1.0 / (4.0 * cos(6.0 * CV_PI / 16.0)));  // 0.653281
static const float s7 = (float)(1.0 / (4.0 * cos(7.0 * CV_PI / 16.0)));  // 1.281458

static float scale[8] = { s0, s1, s2, s3, s4, s5, s6, s7 };
static Mat scaleMat(Size(8, 8), CV_32FC1);
static Mat thAAN(Size(8, 8), CV_32FC1);
static Mat gauss(Size(8, 8), CV_32FC1);
static Mat gausswei(Size(8, 8), CV_32FC1);
static float gaussscale = 0;

static void createGaussianDCTI(Mat& src, const float sigma, const float eps)
{
	float temp = sigma * CV_PI / 8;
	float a = -temp * temp / 2.0;
	//int r = min((int)(ks * sigma), src.cols - 1);

	for (int j = 0; j <= 7; j++)
	{
		for (int i = 0; i <= 7; i++)
		{
			float d = i * i + j * j;
			float v = exp(a*d);

			src.at<float>(j, i) = v + eps;
		}
	}
}

static void createGaussianDCTIMax(Mat& src, const float sigma, const float eps, const int ks)
{
	float temp = sigma * CV_PI / 8;
	float a = -temp * temp / 2.0;
	//int r = min((int)(ks * sigma), src.cols - 1);

	for (int j = 0; j <= 7; j++)
	{
		for (int i = 0; i <= 7; i++)
		{
			float d = (float)(i * i + j * j);
			float v = exp(a*d);

			src.at<float>(j, i) = max(v, eps);
		}
	}
}

// (5 mul, 29 add)
static void dctAAN(float* s, float* d)
{
	// stage 1 (8 add)
	const float b0 = s[0] + s[7];
	const float b1 = s[1] + s[6];
	const float b2 = s[2] + s[5];
	const float b3 = s[3] + s[4];
	const float b4 = -s[4] + s[3];
	const float b5 = -s[5] + s[2];
	const float b6 = -s[6] + s[1];
	const float b7 = -s[7] + s[0];

	// stage 2 (7 add)
	const float c0 = b0 + b3;
	const float c1 = b1 + b2;
	const float c2 = -b2 + b1;
	const float c3 = -b3 + b0;
	const float c4 = -b4 - b5;
	const float c5 = b5 + b6;
	const float c6 = b6 + b7;
	//const float c7 = b7;

	// stage 3 (1 mul, 4 add)
	//const float d0 = c0 + c1;
	//const float d1 = -c1 + c0;
	const float d2 = c2 + c3;
	//const float d3 = c3;
	//const float d4 = c4;
	//const float d5 = c5;
	//const float d6 = c6;
	//const float d7 = c7;

	const float d8 = (c4 + c6) * a5;

	// stage 4 (4 mul, 2 add)
	//const float e0 = d0;
	//const float e1 = d1;
	const float e2 = d2 * a1;
	//const float e3 = d3;
	const float e4 = -c4 * a2 - d8;
	const float e5 = c5 * a3;
	const float e6 = c6 * a4 - d8;
	//const float e7 = d7;

	// stage 5 (4 add)
	//const float f0 = e0;
	//const float f1 = e1;
	//const float f2 = e2 + c3;
	//const float f3 = c3 - e2;
	//const float f4 = e4;
	const float f5 = e5 + b7;
	//const  f6 = e6;
	const float f7 = b7 - e5;

	// stage 6 (4 add)
	d[0] = c0 + c1;
	d[4] = -c1 + c0;
	d[2] = e2 + c3;
	d[6] = c3 - e2;
	d[5] = e4 + f7;
	d[1] = f5 + e6;
	d[7] = -e6 + f5;
	d[3] = f7 - e4;
}

// (5 mul, 29 add)
static void idctAAN(float* s, float* d)
{
	// stage 6 (4 add)
	//const float f0 = s[0];
	//const float f1 = s[4];
	//const float f2 = s[2];
	//const float f3 = s[6];
	const float f4 = s[5] - s[3];
	const float f5 = s[1] + s[7];
	const float f6 = s[1] - s[7];
	const float f7 = s[5] + s[3];

	// stage 5 (1 mul, 5 add)
	//const float e0 = f0;
	//const float e1 = f1;
	const float e2 = s[2] - s[6];
	const float e3 = s[2] + s[6];
	//const float e4 = f4;
	const float e5 = f5 - f7;
	//const float e6 = f6;
	const float e7 = f5 + f7;

	const float e8 = (-f4 - f6) * a5;

	// stage 4 (4 mul, 2 add)
	//const float d0 = e0;
	//const float d1 = e1;
	const float d2 = e2 * a1;
	//const float d3 = e3;
	const float d4 = (-f4 * a2) + e8;
	const float d5 = e5 * a3;
	const float d6 = (f6 * a4) + e8;
	//const float d7 = e7;

	// stage 3 (3 add)
	const float c0 = s[0] + s[4];
	const float c1 = s[0] - s[4];
	//const float c2 = d2;
	const float c3 = d2 + e3;
	//const float c4 = d4;
	//const float c5 = d5;
	//const float c6 = d6;
	//const float c7 = d7;

	// stage 2 (7 add)
	const float b0 = c0 + c3;
	const float b1 = c1 + d2;
	const float b2 = c1 - d2;
	const float b3 = c0 - c3;
	const float b4 = -d4;
	const float b5 = -d4 + d5;
	const float b6 = d5 + d6;
	const float b7 = d6 + e7;

	// stage 1 (8 add)
	d[0] = b0 + b7;
	d[1] = b1 + b6;
	d[2] = b2 + b5;
	d[3] = b3 + b4;
	d[4] = b3 - b4;
	d[5] = b2 - b5;
	d[6] = b1 - b6;
	d[7] = b0 - b7;
}

// DCT AAN 8*8 (224 multiples)
static void dctAAN88(Mat& src, Mat& dest, float th, int sw, float eps)
{
	// DCT (AAN) (80 multiples)
	for (int j = 0; j < 8; j++)
	{
		float* s = src.ptr<float>(j, 0);
		float* d = dest.ptr<float>(j, 0);

		dctAAN(s, d);
	}
	dest = dest.t();
	for (int j = 0; j < 8; j++)
	{
		float* d = dest.ptr<float>(j, 0);

		dctAAN(d, d);
	}
	if (sw == 0)
	{
		for (int i = 0; i < 8; i++)
		{
			float* d = dest.ptr<float>(i, 0);
			float* pscale = scaleMat.ptr<float>(i, 0);
			float* pthAAN = thAAN.ptr<float>(i, 0);
			for (int j = 0; j < 8; j++)
			{
				//cout << d[j] << endl;
				if (i == 0 && j == 0)
				{
					d[j] *= pscale[j];
				}
				else
				{
					if (fabs(d[j]) < pthAAN[j])
					{
						d[j] = 0.0f;
					}
					else
					{
						d[j] *= pscale[j];
					}
				}
			}
		}
	}
	else if (sw == 1)
	{
		float* d = dest.ptr<float>(0);
		float* pscale = scaleMat.ptr<float>(0);
		float* pthAAN = thAAN.ptr<float>(0);
		float* gaussptr = gauss.ptr<float>(0);

		d[0] *= pscale[0];
		d[0] /= gaussptr[0];
		for (int i = 1; i < 64; i++)
		{
			if (abs(d[i]) < pthAAN[i])
			{
				d[i] = 0.0f;
			}
			else
			{
				d[i] *= pscale[i];
				d[i] /= gaussptr[i];
			}
		}
	}
	else if (sw == 2)
	{
		for (int i = 0; i < 8; i++)
		{
			float* d = dest.ptr<float>(i, 0);
			float* pscale = scaleMat.ptr<float>(i, 0);
			float* gaussptr = gauss.ptr<float>(i, 0);
			for (int j = 0; j < 8; j++)
			{
				d[j] *= pscale[j];
				d[j] /= gaussptr[j];
			}
		}
	}


	for (int i = 0; i < 8; i++)
	{
		float* d = dest.ptr<float>(i, 0);
		float* pscale = scaleMat.ptr<float>(i, 0);

		for (int j = 0; j < 8; j++)
		{
			d[j] *= pscale[j];
		}
	}

	float* d = dest.ptr<float>(0);

	// IDCT (AAN) (80 multiples)
	for (int j = 0; j < 8; j++)
	{
		float* d = dest.ptr<float>(j, 0);

		idctAAN(d, d);
	}
	dest = dest.t();
	for (int j = 0; j < 8; j++)
	{
		float* d = dest.ptr<float>(j, 0);

		idctAAN(d, d);
	}
}


static void dctDenoisingSingleAAN(Mat& src, Mat& dest, float th, int sw, float eps)
{
	Mat num = Mat(src.size(), CV_32FC1, Scalar(64.f));
	Mat res = Mat::zeros(src.size(), CV_32FC1);

	for (int i = 0; i < 8; i++)
	{
		float* n = num.ptr<float>(i, 0);
		for (int j = 0; j < 8; j++)
		{
			n[j] = (i + 1.0f) * (j + 1.0f);
			n[num.cols - 1 - j] = (i + 1.0f) * (j + 1.0f);
		}
		for (int j = 8; j < num.cols - 8; j++)
		{
			n[j] = (i + 1.0f) * 8.0f;
		}
		n = num.ptr<float>(num.rows - 1 - i, 0);
		for (int j = 0; j < 8; j++)
		{
			n[j] = (i + 1.0f) * (j + 1.0f);
			n[num.cols - 1 - j] = (i + 1.0f) * (j + 1.0f);
		}
		for (int j = 8; j < num.cols - 8; j++)
		{
			n[j] = (i + 1.0f) * 8.0f;
		}
	}
	for (int i = 8; i < num.rows - 8; i++)
	{
		float* n = num.ptr<float>(i, 0);
		for (int j = 0; j < 8; j++)
		{
			n[j] = (j + 1.0f) * 8.0f;
			n[num.cols - 1 - j] = (j + 1.0f) * 8.0f;
		}
		/*for (int j = 8; j < num.cols - 8; j++)
		{
			n[j] = 64.0f;
		}*/
	}

	// dct denoising (AAN)
#pragma omp parallel for
	for (int i = 0; i < src.rows - 8 + 1; i++)
	{
		Mat patch(Size(8, 8), CV_32FC1);
		for (int j = 0; j < src.cols - 8 + 1; j++)
		{
			dctAAN88(src(Rect(j, i, 8, 8)), patch, th, sw, eps);
			res(Rect(j, i, 8, 8)) += patch;
		}
	}

	res /= num;
	res.convertTo(dest, src.type());
}


template<class T>
void createGaussianDCTIWiener_(Mat& src, const T sigma, const T eps)
{
	T temp = sigma * (T)CV_PI / (src.cols);
	T a = -temp * temp / (T)2.0;

	float s = 1.f / ((src.rows - 1)*(src.cols - 1));
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			T d = (T)(i*i + j * j);
			T v = exp(a*d);
			//src.at<T>(j, i) = v / (v + eps);
			src.at<T>(j, i) = v * v / (v*v + eps);
			//src.at<T>(j, i) = sqrt(v)*v / (sqrt(v)*v + eps);
			//src.at<T>(j, i) = v*v*v / (v*v*v + eps);
		}
	}
}

template<class T>
void createGaussianDCTIWienerSobolev_(Mat& src, const T sigma, const T eps)
{
	T temp = sigma * (T)CV_PI / (src.cols);
	T a = -temp * temp / (T)2.0;

	float s = 1.f / ((src.rows - 1)*(src.cols - 1));
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			T d = (T)(i*i + j * j);
			T v = exp(a*d);
			//src.at<T>(j, i) = v / (v + eps);
			src.at<T>(j, i) = v * v / (v*v + eps * d*s);
			//src.at<T>(j, i) = sqrt(v)*v / (sqrt(v)*v + eps);
			//src.at<T>(j, i) = v*v*v / (v*v*v + eps);
		}
	}
}

void deblurring(Mat& src, Mat& dest, float sigma, int sw, float sigmanoise, const float eps, bool isSobolev)
{
	Mat copy;
	copyMakeBorder(src, copy, 8, 8, 8, 8, BORDER_REPLICATE);

	float th = 3 * sigmanoise;

	for (int i = 0; i < 8; i++)
	{
		float* pscale = scaleMat.ptr<float>(i, 0);
		float* pthAAN = thAAN.ptr<float>(i, 0);
		for (int j = 0; j < 8; j++)
		{
			pscale[j] = scale[i] * scale[j];
			pthAAN[j] = th / (scale[i] * scale[j]);
		}
	}

	//createGaussianDCTI(gauss, sigma, eps);
	//createGaussianDCTI(gauss, sigma, eps, 8);
	createGaussianDCTI(gauss, sigma, FLT_EPSILON);

	if (isSobolev)createGaussianDCTIWienerSobolev_<float>(gausswei, sigma, eps);
	else createGaussianDCTIWiener_<float>(gausswei, sigma, eps);
	divide(gauss, gausswei, gauss);

	//thAAN = thAAN / gauss;

	Mat img;
	copy.convertTo(img, CV_32F);

	if (src.channels() == 1)
	{
		dctDenoisingSingleAAN(img, img, th, sw, eps);
	}
	else
	{
		if (src.channels() == 3)
		{
			cv::Matx33f mt(1.f / sqrt(3.0f), 1.f / sqrt(3.0f), 1.f / sqrt(3.0f),
				1.f / sqrt(2.0f), 0.0f, -1.f / sqrt(2.0f),
				1.f / sqrt(6.0f), -2.0f / sqrt(6.0f), 1.f / sqrt(6.0f));

			cv::transform(img, img, mt);

			std::vector <Mat> mv;
			split(img, mv);

			//Mat gauss1(Size(512, 512), CV_32FC1);
			//gauss1.setTo(0);
			//for (int i = 0; i < 512; i++)
			//{
			//	float* kgauss = gauss1.ptr<float>(i);
			//	for (int j = 0; j < 512; j++) {

			//		float x = j - 0.5;
			//		float y = i - 0.5;
			//		kgauss[j] = expf(-(x*x + y*y) / (2 * sigma*sigma));
			//		//gaussscale += kgauss[j];
			//	}
			//}

			//cv::dct(gauss1, gauss1);

			//for (int i = 0; i < 512 * 512; i++)
			//	gauss1.at<float>(i) /= gauss1.at<float>(0, 0);

			for (size_t i = 0; i < mv.size(); ++i)
				dctDenoisingSingleAAN(mv[i], mv[i], th, sw, eps);

			merge(mv, img);

			cv::transform(img, img, mt.inv());
		}
		else
		{
			CV_Error_(cv::Error::StsNotImplemented, ("Unsupported source image format (=%d)", img.type()));
		}
	}
	//imshow("AAN", img);

	if (sw == 0 || isSobolev) img.convertTo(copy, src.type());
	else Mat((1.f + eps)*img).convertTo(copy, src.type());

	copy(Rect(8, 8, src.cols, src.rows)).copyTo(dest);
}