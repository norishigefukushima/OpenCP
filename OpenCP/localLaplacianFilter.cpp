#include "shiftImage.hpp"
#include <iostream>
#ifdef _OPENMP_LLVM_RUNTIME
#include <omp_llvm.h>
#else
#include <omp.h>
#endif

using namespace std;
using namespace cv;

namespace cp
{



	double fd(double d, double tau, double alpha)
	{
		return tau * pow(d, alpha) + (1.0 - tau) * d;
	}

	double fe(double d)
	{
		return d;
	}

	double sign(double d)
	{
		if (d > 0.0) return 1.0;
		if (d < 0.0) return -1.0;
		return 0.0;
	}

	// remapping function
	void remapping(cv::Mat& R, cv::Vec3f g0, double sigma_r, double tau, double alpha)
	{
		for (int y = 0; y < R.rows; y++)
		{
			for (int x = 0; x < R.cols; x++)
			{
				cv::Vec3f i = R.at<cv::Vec3f>(y, x);
				cv::Vec3f v = i - g0;
				double nrm = cv::norm(v);
				if (nrm != 0.0) v = v / nrm;
				if (nrm <= sigma_r)
				{
					R.at<cv::Vec3f>(y, x) = g0 + v * sigma_r * fd(nrm / sigma_r, tau, alpha);
				}
				else
				{
					R.at<cv::Vec3f>(y, x) = g0 + v * (fe(nrm - sigma_r) + sigma_r);
				}
			}
		}
	}

	// main function
	int llf(int argc, char** argv)
	{
		if (argc <= 1) {
			cout << "usage: LocalLaplacianFilter.exe [input image] ([sigma_r] [max level] [alpha] [tau])" << endl;
			return -1;
		}

		cv::Mat img = cv::imread(argv[1], IMREAD_COLOR);
		if (img.empty()) {
			cout << "Failed to load file \"" << argv[1] << "\"" << endl;
			return -1;
		}

		const int width = img.cols;
		const int height = img.rows;
		const int dim = img.channels();
		img.convertTo(img, CV_32F, 1.0 / 255.0);

		// initial parameter values
		const double sigma_r = argc > 2 ? atof(argv[2]) : 0.2;
		const int    maxLevel = argc > 3 ? atoi(argv[3]) : 3;
		const double alpha = argc > 4 ? atof(argv[4]) : 4.0;
		const double tau = argc > 5 ? atof(argv[5]) : 0.8;

		// stdout info.
		printf("*** Local Laplacian Filter ***\n");
		printf("  sigma_r   = %f\n", sigma_r);
		printf("  max level = %d\n", maxLevel);
		printf("  alpha     = %f\n", alpha);
		printf("  tau       = %f\n", tau);
		printf("\n");

		// building a Gaussian pyramid
		cv::Mat tmp;
		double sigma = 2.0;
		vector<cv::Mat> GaussianPyramid(maxLevel + 1);
		img.convertTo(GaussianPyramid[0], CV_32F);
		for (int level = 1; level <= maxLevel; level++)
		{
			cv::GaussianBlur(GaussianPyramid[level - 1], tmp, cv::Size(), sigma);
			cv::pyrDown(tmp, GaussianPyramid[level]);
		}

		// building a Laplacian pyramid, and remapping the coefficients
		vector<cv::Mat> laplacePyramid(maxLevel);
		for (int level = 0; level < maxLevel; level++)
		{
			printf("  Process Lv. %d ...\n", level);
			laplacePyramid[level] = cv::Mat::zeros(GaussianPyramid[level].size(), CV_MAKETYPE(CV_32F, dim));

#pragma omp parallel for
			for (int y = 0; y < GaussianPyramid[level].rows; y++)
			{
				for (int x = 0; x < GaussianPyramid[level].cols; x++)
				{
					int rx = x;
					int ry = y;
					int K = 4;
					int l = level;
					while (l > 0)
					{
						rx *= 2;
						ry *= 2;
						K *= 2;
						l--;
					}

					K = 3 * (K - 1);
					int cx = max(0, rx - K);
					int cy = max(0, ry - K);
					int bx = min(width - 1, rx + K);
					int by = min(height - 1, ry + K);

					cv::Mat R, tmp;
					tmp = img(cv::Rect(cx, cy, bx - cx + 1, by - cy + 1));
					tmp.convertTo(R, CV_32F);
					cv::Vec3f g0 = GaussianPyramid[level].at<cv::Vec3f>(y, x);
					remapping(R, g0, sigma_r, tau, alpha);

					for (int sublev = 0; sublev < level; sublev++)
					{
						cv::pyrDown(R, tmp);
						tmp.convertTo(R, CV_32F);
					}

					cv::Mat subR;
					cv::pyrDown(R, tmp);
					cv::pyrUp(tmp, subR, R.size());
					cv::Mat L = R - subR;

					for (int sublev = 0; sublev < level; sublev++)
					{
						cv::pyrUp(L, tmp);
						tmp.convertTo(L, CV_32F);
					}

					int dx = cx - (rx - K);
					int dy = cy - (ry - K);
					laplacePyramid[level].at<cv::Vec3f>(y, x) = L.at<cv::Vec3f>(K + 1 - dy, K + 1 - dx);
				}
			}
		}
		printf("  Finish!\n\n");

		// reconstruct resulting image from laplacian pyramid
		cv::Mat res;
		GaussianPyramid[maxLevel].convertTo(res, CV_32F);
		for (int level = maxLevel - 1; level >= 0; level--)
		{
			cv::pyrUp(res, tmp, GaussianPyramid[level].size());
			res = tmp + laplacePyramid[level];
		}

		// show results
		cv::imshow("Input", img);
		cv::imshow("Result", res);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}