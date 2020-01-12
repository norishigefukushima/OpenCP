#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void guiMaxMinFilter(Mat& src_)
{
	Mat src;
	if (src_.channels() == 3)cvtColor(src_, src, COLOR_BGR2GRAY);
	else src = src_;

	Mat dest;

	string wname = "max min filter";
	namedWindow(wname);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 1; createTrackbar("switch", wname, &sw, 1);

	int r = 1; createTrackbar("r", wname, &r, 10);

	int key = 0;
	Mat show;

	while (key != 'q')
	{
		double tim;

		if (sw == 0)
		{
			Timer t("max filter", 0, false);
			maxFilter(src, dest, Size(2 * r + 1, 2 * r + 1), MORPH_ELLIPSE);
			tim = t.getTime();
		}

		if (sw == 1)
		{
			Timer t("min filter", 0, false);
			minFilter(src, dest, Size(2 * r + 1, 2 * r + 1), MORPH_ELLIPSE);
			tim = t.getTime();
		}

		alphaBlend(src, dest, a / 100.0, show);

		imshow(wname, show);
		key = waitKey(1);
	}
	destroyWindow(wname);
}

double PSNRBB(InputArray src, InputArray ref, int boundingX, int boundingY = 0)
{
	if (boundingY == 0)boundingY = boundingX;
	Mat a = src.getMat();
	Mat b = ref.getMat();

	Rect roi = Rect(Point(boundingX, boundingY), Point(a.cols - boundingX, a.rows - boundingY));

	return PSNR(a(roi), b(roi));
}

double psnrRGBave(Mat& src, Mat& ref)
{
	Mat srcf, reff;
	src.convertTo(srcf, CV_64F);
	ref.convertTo(reff, CV_64F);
	vector<Mat> srcfs;
	vector<Mat> reffs;
	split(srcf, srcfs);
	split(reff, reffs);
	double b = getPSNR(srcfs[0], reffs[0]);
	double g = getPSNR(srcfs[1], reffs[0]);
	double r = getPSNR(srcfs[2], reffs[0]);

	return(b + g + r) / 3;
}

double psnrRGBone(Mat& src, Mat& ref)
{
	Mat srcf,reff;
	Mat srcfs, reffs;
	src.convertTo(srcf, CV_64F);
	ref.convertTo(reff, CV_64F);

	cvtColorBGR2PLANE(srcf, srcfs);
	cvtColorBGR2PLANE(reff, reffs);

	double a = getPSNR(srcfs, reffs);
	return a;
}

static double getPSNRRealtimeO1BilateralFilter(Mat& src, Mat ref, double sigmaSpace, double sigmaColor, int bin)
{
	Mat srcf; src.convertTo(srcf, CV_32F);
	

	Timer t;
	RealtimeO1BilateralFilter rbf;
	rbf.setColorNorm(RealtimeO1BilateralFilter::L2);
	rbf.setBinDepth(CV_64F);

	Mat dest;
	rbf.gaussIIR(srcf, dest, sigmaColor, sigmaSpace, bin, RealtimeO1BilateralFilter::IIR_SR, 0);

	cout << psnrRGBone(dest, ref)<<endl;
	double ret = psnrRGBone(dest, ref);
	return ret;
}

void getPSNRRealtimeO1BilateralFilterKodak()
{
	for (int i = 1; i <= 24; i++)
	{
		Mat src = imread(format("img/Kodak/kodim%02d.png",i));

		double sigmaSpace = 10.0;
		double sigmaColor = 30.0;
		const int r = cvRound(3.f*sigmaSpace);

		Mat srcf; src.convertTo(srcf, CV_32F);
		Mat ref; 
		bilateralFilterL2(srcf, ref, r, 30.0, sigmaSpace, BORDER_REPLICATE);

		cout << i << "," << getPSNRRealtimeO1BilateralFilter(src, ref, sigmaSpace, sigmaColor, 4) << endl;
	}
}

void guiRealtimeO1BilateralFilterTest(Mat& src)
{
	Mat dest;

	string wname = "realtime O1 bilateral filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 2; createTrackbar("filter sw", wname, &sw, 5);

	int space = 100; createTrackbar("space", wname, &space, 2000);
	int color = 300; createTrackbar("color", wname, &color, 2550);
	int bin = 4; createTrackbar("bin", wname, &bin, 256);
	int iter = 3; createTrackbar("iter", wname, &iter, 100);
	int srsize = 1; createTrackbar("sresize", wname, &srsize, 32);
	int rsize = 1; createTrackbar("resize", wname, &rsize, 32);
	int upmethod = 1; createTrackbar("umethod", wname, &upmethod, 4);
	int type = 0; createTrackbar("type32_64", wname, &type, 1);
	int mem = 0; createTrackbar("memory", wname, &mem, 1);

	namedWindow("diff");
	int scale = 10; createTrackbar("scale", "diff", &scale, 50);

	int key = 0;
	Mat show;

	RealtimeO1BilateralFilter rbf;
	Mat ref;

	Mat srcf; src.convertTo(srcf, CV_32F);
	Mat ref2;
#define DEBUG_32F_RTBF 1
#ifdef DEBUG_32F_RTBF
	{
		//bilateralFilter(srcf, ref, cvRound(3.f*space / 10.f) * 2 + 1, color / 10.f, space / 10.0f, BORDER_REPLICATE);
		bilateralFilterL2(srcf, ref, cvRound(3.f*space / 10.f), color / 10.f, space / 10.0f, BORDER_REPLICATE);
		//cout << PSNR64F(ref2, ref) << endl;
		//cout << psnrRGBone(ref, ref) << endl;
	}
#else
	bilateralFilter(src, ref, cvRound(3.f*space / 10.f) * 2 + 1, color / 10.f, space / 10.0f, BORDER_REPLICATE);
#endif

	rbf.setColorNorm(RealtimeO1BilateralFilter::L2);
	while (key != 'q')
	{
		src.convertTo(srcf, CV_32F);
		if (type == 0) rbf.setBinDepth(CV_32F);
		else rbf.setBinDepth(CV_64F);

		if (mem == 0) rbf.isSaveMemory = false;
		else rbf.isSaveMemory = true;

		double tim;
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;
		int d = cvRound(sigma_space*3.f) * 2 + 1;
		rbf.upsampleMethod = upmethod;
		rbf.downsampleSizeBlurring = rsize;
		rbf.downsampleSizeSplatting = srsize;
		if (key == 'r')
		{

			Timer t("reftime");
#ifdef DEBUG_32F_RTBF
			//bilateralFilter(srcf, ref, d, color / 10.f, space / 10.0f, BORDER_REPLICATE);
			bilateralFilterL2(srcf, ref, cvRound(3.f*space / 10.f), color / 10.f, space / 10.0f, BORDER_REPLICATE);
#else
			//cp::bilateralFilter(src, ref, Size(d, d), sigma_color, sigma_space, FILTER_DEFAULT, BORDER_REPLICATE);		
			bilateralFilter(src, ref, d, color / 10.f, space / 10.0f, BORDER_REPLICATE);
#endif
		}
		if (sw == 0)
		{
			Timer t("FIR SP", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussFIR(srcf, dest, d / 2, sigma_color, sigma_space, bin);
#else
			rbf.gauss_fir(src, dest, d / 2, sigma_color, sigma_space, bin);
#endif
			tim = t.getTime();
		}
		if (sw == 1)
		{
			Timer t("IIR AM", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_AM, iter);
#else
			rbf.gauss_iir(src, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_AM, iter);
#endif
			tim = t.getTime();
		}
		if (sw == 2)
		{
			Timer t("IIR SR", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_SR, iter);
#else
			rbf.gauss_iir(src, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_SR, iter);
#endif
			tim = t.getTime();
		}
		if (sw == 3)
		{
			Timer t("IIR Deriche", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_Deriche, iter);
#else
			rbf.gauss_iir(src, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_Deriche, iter);
#endif
			tim = t.getTime();
		}
		if (sw == 4)
		{
			Timer t("IIR YVY", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_YVY, iter);
#else
			rbf.gauss_iir(src, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_YVY, iter);
#endif
			tim = t.getTime();
		}

		ci("d: %d", d);
		//ci("PSNR: %f", PSNRBB(dest, ref, 100, 100));
		//ci("PSNR: %f", PSNR64F(dest, ref));
		
		ci("PSNR: %f", psnrRGBone(dest, ref));
		cout << psnrRGBone(dest, ref) << endl;
		//ci("MSE:  %f", norm(dest, ref, NORM_L2SQR) / (double)dest.size().area());
		ci("time: %f", tim);
		ci.show();
		alphaBlend(ref, dest, a / 100.0, show);

		if (key == 'p') rbf.showBinIndex();
		if (key == 't')guiMaxMinFilter(src);
		diffshow("diff", dest, ref, (float)scale);
		if (key == 'a')
		{
			guiAlphaBlend(ref, dest);
		}
		if (key == 'd')
		{
			guiAbsDiffCompareGE(ref, dest);
		}

		if (key == 's')
		{
			cout << "write" << endl;
			showMatInfo(ref);
			imwrite("test.tiff", ref);
			imwrite("test.png", ref);
		}
#ifdef DEBUG_32F_RTBF
		Mat show8U;
		show.convertTo(show8U, CV_8U);
		imshow(wname, show8U);
#else
		imshow(wname, show);
#endif
		key = waitKey(1);
	}
	destroyAllWindows();
}


void guiJointRealtimeO1BilateralFilterTest(Mat& src_, Mat& guide_)
{

	Mat src, guide;
	if (src_.channels() == 3)
	{
		//cvtColor(src_, guide, COLOR_BGR2GRAY);
		//cvtColor(src_, src, COLOR_BGR2GRAY);
		//src_.copyTo(src);
		guide = guide_;
		src = src_;
	}
	else
	{
		guide = guide_;
		src = src_;
	}

	Mat dest;

	string wname = "realtime O1 bilateral filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 0; createTrackbar("switch", wname, &sw, 5);

	//int r = 10; createTrackbar("r",wname,&r,200);
	int space = 100; createTrackbar("space", wname, &space, 2000);
	int color = 300; createTrackbar("color", wname, &color, 2550);
	int bin = 6; createTrackbar("bin", wname, &bin, 256);
	int iter = 3; createTrackbar("iter", wname, &iter, 100);
	int srsize = 4; createTrackbar("sresize", wname, &srsize, 32);
	int rsize = 1; createTrackbar("resize", wname, &rsize, 32);
	int upmethod = 1; createTrackbar("umethod", wname, &upmethod, 4);
	int type = 0; createTrackbar("type", wname, &type, 1);
	int mem = 0; createTrackbar("memory", wname, &mem, 1);
	int sscale = 10; createTrackbar("sscale", wname, &sscale, 100);
	namedWindow("diff");
	int scale = 10; createTrackbar("scale", "diff", &scale, 50);


	int key = 0;
	Mat show;

	RealtimeO1BilateralFilter rbf;
	Mat ref;
	Mat srcf; src.convertTo(srcf, CV_32F);
	//#define DEBUG_32F_RTBF 1
#ifdef DEBUG_32F_RTBF
	bilateralFilter(srcf, ref, cvRound(6.f*space / 10.f) * 2 + 1, color / 10.f, space / 10.0f, BORDER_REFLECT);
#else
	jointBilateralFilter(src, guide, ref, cvRound(3.f*space / 10.f) * 2 + 1, color / 10.f, space / 10.0f, BORDER_REFLECT);
#endif

	while (key != 'q')
	{
		src.convertTo(srcf, CV_32F, sscale / 10.0);
		if (type == 0) rbf.setBinDepth(CV_32F);
		else rbf.setBinDepth(CV_64F);

		if (mem == 0) rbf.isSaveMemory = false;
		else rbf.isSaveMemory = true;

		double tim;
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_color = color / 10.f;
		float sigma_space = space / 10.f;
		int d = cvRound(sigma_space*3.f) * 2 + 1;
		rbf.upsampleMethod = upmethod;
		rbf.downsampleSizeBlurring = rsize;
		rbf.downsampleSizeSplatting = srsize;
		if (key == 'r')
		{

#ifdef DEBUG_32F_RTBF
			jointBilateralFilter(src, guide, ref, d, color / 10.f, space / 10.0f, BORDER_REFLECT);
#else
			jointBilateralFilter(src, guide, ref, d, color / 10.f, space / 10.0f, BORDER_REFLECT);
			//jointBilateralFilter(src, guide, ref, Size(d, d), color / 10.f, space / 10.0f, FILTER_CIRCLE, BORDER_REFLECT);
#endif
		}

		if (sw == 0)
		{
			Timer t("FIR SP", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussFIR(srcf, guide, dest, d / 2, sigma_color, sigma_space, bin);
#else
			rbf.gauss_fir(src, guide, dest, d / 2, sigma_color, sigma_space, bin);
#endif
			tim = t.getTime();
		}
		if (sw == 1)
		{
			Timer t("IIR AM", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_AM, iter);
#else
			rbf.gauss_iir(src, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_AM, iter);
#endif
			tim = t.getTime();
		}
		if (sw == 2)
		{
			Timer t("IIR SR", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_SR, iter);
#else
			rbf.gauss_iir(src, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_SR, iter);
#endif
			tim = t.getTime();
		}
		if (sw == 3)
		{
			Timer t("IIR Deriche", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_Deriche, iter);
#else
			rbf.gauss_iir(src, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_Deriche, iter);
#endif
			tim = t.getTime();
		}
		if (sw == 4)
		{
			Timer t("IIR YVY", 0, false);

#ifdef DEBUG_32F_RTBF
			rbf.gaussIIR(srcf, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_YVY, iter);
#else
			rbf.gauss_iir(src, guide, dest, sigma_color, sigma_space, bin, RealtimeO1BilateralFilter::IIR_YVY, iter);
#endif
			tim = t.getTime();
		}

		ci("d: %d", d);
		//ci("PSNR: %f", PSNRBB(dest, ref, 100, 100));
		ci("PSNR: %f", getPSNR(dest, ref));
		ci("MSE:  %f", norm(dest, ref, NORM_L2SQR) / (double)dest.size().area());
		ci("time: %f", tim);
		ci.show();
		alphaBlend(ref, dest, a / 100.0, show);

		if (key == 'p') rbf.showBinIndex();
		if (key == 't')guiMaxMinFilter(src);
		diffshow("diff", dest, ref, (float)scale);
		if (key == 'a')
		{
			guiAlphaBlend(ref, dest);
		}
		if (key == 'd')
		{
			guiAbsDiffCompareGE(ref, dest);
		}
#ifdef DEBUG_32F_RTBF
		Mat show8U;
		show.convertTo(show8U, CV_8U);
		imshow(wname, show8U);
#else
		imshow(wname, show);
#endif
		key = waitKey(1);
	}
	destroyAllWindows();
}
