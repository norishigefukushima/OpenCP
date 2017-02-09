#include "shockFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	void coherenceEnhancingShockFilter(cv::InputArray src, cv::OutputArray dest_, const int sigma, const int str_sigma_, const double blend, const int iter)
	{
		Mat dest = src.getMat();
		const int str_sigma = min(31, str_sigma_);

		for (int i = 0; i < iter; i++)
		{
			Mat gray;
			if (src.channels() == 3)cvtColor(dest, gray, CV_BGR2GRAY);
			else gray = dest;

			Mat eigen;
			if (gray.type() == CV_8U || gray.type() == CV_32F || gray.type() == CV_64F)
				cornerEigenValsAndVecs(gray, eigen, str_sigma, 3);
			else
			{
				Mat grayf; gray.convertTo(grayf, CV_32F);
				cornerEigenValsAndVecs(grayf, eigen, str_sigma, 3);
			}

			vector<Mat> esplit(6);
			split(eigen, esplit);
			Mat x = esplit[2];
			Mat y = esplit[3];
			Mat gxx;
			Mat gyy;
			Mat gxy;
			Sobel(gray, gxx, CV_32F, 2, 0, sigma);
			Sobel(gray, gyy, CV_32F, 0, 2, sigma);
			Sobel(gray, gxy, CV_32F, 1, 1, sigma);

			Mat gvv = x.mul(x).mul(gxx) + 2 * x.mul(y).mul(gxy) + y.mul(y).mul(gyy);

			Mat mask;
			compare(gvv, 0, mask, cv::CMP_LT);

			Mat di, ero;
			erode(dest, ero, Mat());
			dilate(dest, di, Mat());
			di.copyTo(ero, mask);
			addWeighted(dest, blend, ero, 1.0 - blend, 0.0, dest);
		}
		dest.copyTo(dest_);
	}
}