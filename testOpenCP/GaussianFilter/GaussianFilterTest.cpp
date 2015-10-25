#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void GaussianBlur2(Mat& src, Mat& dest, float sigma, int clip = 3, int depth = CV_32F, int borderType = BORDER_REFLECT)
{
	Mat srcf;
	src.convertTo(srcf, depth);

	if (borderType == 3) borderType = 4;
	GaussianBlur(srcf, srcf, Size(cvRound(clip*sigma) * 2 + 1, cvRound(clip*sigma) * 2 + 1), sigma, 0.0, borderType);

	if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
		srcf.convertTo(dest, depth, 1.0, 0.5);
	else
	{
		srcf.convertTo(dest, src.type());
	}
}

void guiGausianFilterTest(Mat& src__)
{
	src__.convertTo(src__, CV_32F);
	int w = (4 - src__.cols % 4) % 4;
	int h = (4 - src__.rows % 4) % 4;
	Mat src_; copyMakeBorder(src__, src_, 0, h, 0, w, BORDER_REPLICATE);

	Mat src;

	if (src_.channels() == 3)
	{
		cvtColor(src_, src, COLOR_BGR2GRAY);
		//src_.copyTo(src);
	}
	else
	{
		src = src_;
	}

	Mat src64F;
	src.convertTo(src64F, CV_64F);
	Mat dest;

	string wname = "Gaussian filter";
	namedWindow(wname);
	ConsoleImage ci;

	int a = 0; createTrackbar("a", wname, &a, 100);
	int sw = 7; createTrackbar("switch", wname, &sw, 9);

	//int r = 10; createTrackbar("r",wname,&r,200);
	int space = 50; createTrackbar("space", wname, &space, 2000);
	int step = 4; createTrackbar("step", wname, &step, 100);

	int scale = 100; createTrackbar("diff scale", wname, &scale, 100);
	int range = 1; createTrackbar("range", wname, &range, 100);
	int type = 0; createTrackbar("type", wname, &type, 1);
	int key = 0;
	Mat show;

	Mat ref;
	Mat base = src.clone();
	Stat st;

	int prev_type = type;
	int prev_sw = sw;
	int prev_step = step;
	while (key != 'q')
	{
		if (type == 0)
		{
			src.convertTo(src, CV_32F);
			ci("32F");
		}
		else
		{
			src.convertTo(src, CV_64F);
			ci("64F");
		}

		dest.setTo(0);
		ref.setTo(0);
		double tim;
		//cout<<"r="<<r<<": "<<"please change 'sw' for changing the type of implimentations."<<endl;
		float sigma_space = space / 10.f;

		int d = cvRound(sigma_space*3.0) * 2 + 1;

		{
			string mes = "DCT Gaussian";
			ci(mes);
			CalcTime t(mes, 0, false);
			GaussianFilter(src64F, ref, sigma_space, GAUSSIAN_FILTER_DCT);
			tim = t.getTime();
			ci("time: %f", tim);
		}

		if (sw == 0)
		{
			string mes = format("OpenCV Gaussian");
			ci(mes);
			CalcTime t(mes, 0, false);
			GaussianBlur2(src, dest, sigma_space, 6, (type == 0) ? CV_32F : CV_64F);// six sigma
			tim = t.getTime();
		}
		else if (sw == 1)
		{
			string mes = format("FIR");
			ci(mes);
			CalcTime t(mes, 0, false);
			GaussianFilter(src, dest, (double)sigma_space, GAUSSIAN_FILTER_FIR, 0, 1.0e-9);
			tim = t.getTime();
		}
		else if (sw == 2)
		{
			string mes = "Alvarez-Mazorra";
			ci(mes);
			CalcTime t(mes, 0, false);

			GaussianFilter(src, dest, sigma_space, GAUSSIAN_FILTER_AM2, step, 1e-2);
			tim = t.getTime();
		}
		else if (sw == 3)
		{
			string mes = "Alvarez-Mazorra Fast";
			ci(mes);
			CalcTime t(mes, 0, false);

			GaussianFilter(src, dest, sigma_space, GAUSSIAN_FILTER_AM, step);
			tim = t.getTime();
		}
		else if (sw == 4)
		{
			string mes = "Box";
			ci(mes);
			CalcTime t(mes, 0, false);

			GaussianFilter(src, dest, sigma_space, GAUSSIAN_FILTER_BOX, step);
			tim = t.getTime();
		}
		else if (sw == 5)
		{
			string mes = "EBox";
			ci(mes);
			CalcTime t(mes, 0, false);
			GaussianFilter(src, dest, sigma_space, GAUSSIAN_FILTER_EBOX, step);
			tim = t.getTime();
		}
		else if (sw == 6)
		{
			string mes = "SII";
			ci(mes);
			CalcTime t(mes, 0, false);
			GaussianFilter(src, dest, (double)sigma_space, GAUSSIAN_FILTER_SII, min(max(step, 3), 5));
			tim = t.getTime();
		}
		else if (sw == 7)
		{
			string mes = "Deriche";
			ci(mes);
			CalcTime t(mes, 0, false);

			GaussianFilter(src, dest, sigma_space, GAUSSIAN_FILTER_DERICHE, min(max(step, 2), 4));
			tim = t.getTime();
		}
		else if (sw == 8)
		{
			string mes = "YVY";
			ci(mes);
			CalcTime t(mes, 0, false);

			GaussianFilter(src, dest, (double)sigma_space, GAUSSIAN_FILTER_VYV, step, 1.0e-6);
			tim = t.getTime();
		}
		else if (sw == 9)
		{
			string mes = "Sugimoto";
			ci(mes);
			CalcTime t(mes, 0, false);

			GaussianFilter(src, dest, sigma_space, GAUSSIAN_FILTER_SR);
			tim = t.getTime();
		}

		if (sw != prev_sw || prev_step != step || prev_type != type)
		{
			st.clear();
			prev_sw = sw;
			prev_step = step;
			prev_type = type;
		}
		st.push_back(tim);

		ci("time: %f", st.getMedian());
		ci("d: %d", d);

		alphaBlend(ref, dest, a / 100.0, show);
		if (src.depth() == CV_32F || src.depth() == CV_64F)
		{
			ci("MSE: %f", MSE(dest, ref));
			ci("PSNR: %f", PSNR64F(dest, ref));
			show.convertTo(show, CV_8U, 1.0, 0.5);
		}
		else
			ci("PSNR: %f", PSNR(dest, ref));

		diffshow("diff", dest, ref, (float)scale);
		ci.flush();
		imshow(wname, show);

		key = waitKey(1);
	}
	destroyAllWindows();
}