#include "hazeRemove.hpp" 
#include "guidedFilter.hpp"
#include "minmaxfilter.hpp"

using namespace std;
using namespace cv;
#include "opencp.hpp"

namespace cp
{

	void HazeRemove::darkChannel(Mat& src, int r)
	{
		if (dark.empty())dark.create(size, CV_8U);

		vector<Mat> srcv;
		split(src, srcv);

		for (int i = 0; i < src.channels(); i++)
		{
			minFilter(srcv[i], minvalue[i], r);
		}

		if (src.channels() == 3)
		{
			//#pragma omp parallel for
			for (int i = 0; i < size.height; i++)
			{
				const uchar* sr = minvalue[0].ptr(i);
				const uchar* sg = minvalue[0].ptr(i);
				const uchar* sb = minvalue[0].ptr(i);
				uchar* dptr = dark.ptr(i);

				for (int j = 0; j < size.width; j++)
				{
					uchar minv = min(sr[j], sg[j]);
					dptr[j] = min(minv, sb[j]);
				}
			}
		}
	}

	void HazeRemove::getAtmosphericLight(Mat& srcImage, double topPercent)
	{
		int hist[256];
		double icount = 1.0 / (double)dark.size().area();
		for (int i = 0; i < 256; i++)hist[i] = 0;
		for (int j = 0; j < srcImage.rows; j++)
		{
			uchar* s = dark.ptr(j);
			for (int i = 0; i < srcImage.cols; i++)
			{
				hist[s[i]]++;
			}
		}

		int thresh = 0;
		int v = 0;
		for (int i = 255; i >= 0; i--)
		{
			v += hist[i];
			if (100.0*v*icount > topPercent)
			{
				thresh = i;
				break;
			}
		}
		Mat map;
		threshold(dark, map, thresh, 255, THRESH_BINARY);
		A = CV_RGB(0, 0, 0);

		int maxv = 0;
		int count = 0;
		for (int j = 0; j < srcImage.rows; j++)
		{
			uchar* m = map.ptr(j);
			uchar* s = srcImage.ptr(j);
			for (int i = 0; i < srcImage.cols; i++)
			{
				if (m[i] == 255)
				{
					if (maxv < s[3 * i] + s[3 * i + 1] + s[3 * i + 2])
					{
						A.val[0] += s[3 * i];
						A.val[1] += s[3 * i + 1];
						A.val[2] += s[3 * i + 2];
						count++;
					}
				}

			}
		}
		if (count != 0)
		{
			A.val[0] /= (double)count;
			A.val[1] /= (double)count;
			A.val[2] /= (double)count;
		}
	}

	void HazeRemove::getTransmissionMap(float omega)
	{
		tmap = Mat::ones(size, CV_32F);
		const float ir = (float)(1.0 / A.val[0]);
		const float ig = (float)(1.0 / A.val[1]);
		const float ib = (float)(1.0 / A.val[2]);

		//#pragma omp parallel for
		for (int i = 0; i < size.height; i++)
		{
			uchar* sr = minvalue[0].ptr<uchar>(i);
			uchar* sg = minvalue[1].ptr<uchar>(i);
			uchar* sb = minvalue[2].ptr<uchar>(i);
			float* dptr = tmap.ptr<float>(i);

			for (int j = 0; j < size.width; j++)
			{
				float minv = min((float)sr[j] * ir, (float)sg[j] * ig);
				dptr[j] -= omega*min(minv, (float)sb[j] * ib);
			}
		}
	}

	void HazeRemove::removeHaze(Mat& src, Mat& trans, Scalar v, Mat& dest, float clip)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		for (int j = 0; j < src.rows; j++)
		{
			float* a = trans.ptr<float>(j);
			uchar* s = src.ptr(j);
			uchar* d = dest.ptr(j);
			for (int i = 0; i < src.cols; i++)
			{
				float t = max(clip, a[i]);

				d[3 * i + 0] = saturate_cast<uchar>((s[3 * i + 0] - v.val[0]) / t + v.val[0]);
				d[3 * i + 1] = saturate_cast<uchar>((s[3 * i + 1] - v.val[1]) / t + v.val[1]);
				d[3 * i + 2] = saturate_cast<uchar>((s[3 * i + 2] - v.val[2]) / t + v.val[2]);
			}
		}
	}

	HazeRemove::HazeRemove()
	{
		minvalue.resize(3);
	}

	HazeRemove::~HazeRemove()
	{
		;
	}

	void HazeRemove::getAtmosphericLightImage(Mat& dest)
	{
		if (dest.empty())dest.create(size, CV_8UC3);
		dest.setTo(A);
	}

	void HazeRemove::showTransmissionMap(Mat& dest, bool isPseudoColor)
	{
		Mat temp;
		tmap.convertTo(temp, CV_8U, 255);
		if (!isPseudoColor)cvtColor(temp, dest, COLOR_GRAY2BGR);
		else applyColorMap(temp, dest, 2);
	}

	void HazeRemove::showDarkChannel(Mat& dest, bool isPseudoColor)
	{
		if (!isPseudoColor)cvtColor(dark, dest, COLOR_GRAY2BGR);
		else applyColorMap(dark, dest, 2);
	}

	void HazeRemove::removeFastGlobalSmootherFilter(Mat& src, Mat& dest, int r_dark, double top_rate, const double lambda, const double sigma_color, const double lambda_attenuation, const int iteration)
	{
		size = src.size();

		darkChannel(src, r_dark);
		getAtmosphericLight(src, top_rate);
		getTransmissionMap();
		Mat srcg;
		cvtColor(src, srcg, COLOR_BGR2GRAY);
		ximgproc::fastGlobalSmootherFilter(srcg, tmap, tmap, lambda, sigma_color, lambda_attenuation, max(iteration, 1));
		
		/*Mat temp;
		cout << lambda << endl;
		ximgproc::fastGlobalSmootherFilter(srcg, srcg, temp, lambda, sigma_color, lambda_attenuation, max(iteration, 1));
		cv::imshow("a", temp);
		*/
		removeHaze(src, tmap, A, dest);		
	}

	void HazeRemove::removeGuidedFilter(Mat& src, Mat& dest, int r_dark, double toprate, int r_joint, double e_joint)
	{
		size = src.size();

		darkChannel(src, r_dark);
		getAtmosphericLight(src, toprate);
		getTransmissionMap();
		Mat srcg;
		cvtColor(src, srcg, COLOR_BGR2GRAY);
		guidedFilter(tmap, srcg, tmap, r_joint, (float)e_joint);
		removeHaze(src, tmap, A, dest);
	}

	void HazeRemove::operator() (Mat& src, Mat& dest, const int r_dark, const double top_rate, const int r_joint, const double e_joint)
	{
		removeGuidedFilter(src, dest, r_dark, top_rate, r_joint, e_joint);
	}

	void HazeRemove::gui(Mat& src, string wname)
	{
		string wnalight = "a light";
		string wnhaze = "dehaze";
		
		namedWindow(wnalight);
		namedWindow(wnhaze);
		namedWindow(wname);

		int mode = 0;
		createTrackbar("mode", wname, &mode, 2);
		int alpha = 0;
		createTrackbar("alpha", wname, &alpha, 100);
		int r_dark = 4;
		createTrackbar("r_dark", wname, &r_dark, 100);
		int hazerate = 10;
		createTrackbar("hazerate*0.01", wname, &hazerate, 100);
		
		int r_joint = 15; createTrackbar("r_joint", wname, &r_joint, 100);
		int e = 6; createTrackbar("e*0.1", wname, &e, 255);

		int lambda = 100; createTrackbar("lambda", wname, &lambda, 50000);
		int sigma_range = 30; createTrackbar("sigma_r", wname, &sigma_range, 500);
		int lambda_a = 25; createTrackbar("lambda_a*0.01", wname, &lambda_a, 100);
		int iteration = 3; createTrackbar("iteration", wname, &iteration, 10);

		int key = 0;
		while (key != 'q')
		{
			if (mode == 0)
			{
				Mat show;
				Mat destC;
				Mat destDark;
				{
					//Timer t("removeGuidedFilter");
					operator()(src, show, r_dark, hazerate / 100.0, r_joint, e / 10.0);

				}
				showTransmissionMap(destC, true);
				Mat a;
				getAtmosphericLightImage(a);
				
				addWeighted(src, alpha / 100.0, destC, 1.0 - alpha / 100.0, 0.0, destC);
				
				imshow(wnalight, a);
				imshow(wnhaze, show);
				imshow(wname, destC);
			}
			else if (mode == 1)
			{
				Mat show;
				Mat destC;
				Mat destDark;
				{
					//Timer t("removeFastGlobalSmootherFilter");
					removeFastGlobalSmootherFilter(src, show, r_dark, hazerate / 100.0, lambda, sigma_range*0.01, lambda_a*0.01, iteration);

				}
				showTransmissionMap(destC, true);
				Mat a;
				getAtmosphericLightImage(a);
				addWeighted(src, alpha / 100.0, destC, 1.0 - alpha / 100.0, 0.0, destC);
				
				imshow(wnalight, a);
				imshow(wnhaze, show);
				imshow(wname, destC);
			}
			else
			{
				double mn, mx;
				Mat srcg;
				cvtColor(src, srcg, COLOR_RGB2YUV);
				minMaxLoc(srcg, &mn, &mx);
				Mat dest = src - CV_RGB(mn, mn, mn);

				dest.convertTo(dest, CV_8UC3, 255 / (mx - mn));
				imshow(wnhaze, dest);
			}

			key = waitKey(1);
		}
	}
}