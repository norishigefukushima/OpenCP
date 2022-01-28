#include "postFilterSet.hpp"
#include "binalyWeightedRangeFilter.hpp"
#include "depth2disparity.hpp"
#include "statisticalFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void smallGaussianBlur(const Mat& src, Mat& dest, const int d, const double sigma)
	{
		if (d == 0)
		{
			src.copyTo(dest);
			return;
		}

		Mat srcf;
		src.convertTo(srcf, CV_32F);
		GaussianBlur(srcf, srcf, Size(d, d), sigma);
		srcf.convertTo(dest, src.type());
	}

	PostFilterSet::PostFilterSet(){ ; }
	PostFilterSet::~PostFilterSet(){ ; }

	void PostFilterSet::filterDisp8U2Depth16U(Mat& src, Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method)
	{
		medianBlur(src, buff, 2 * median_r + 1);
		smallGaussianBlur(buff, buff, 2 * gaussian_r + 1, gaussian_r + 0.5);
		blurRemoveMinMax(buff, buff, minmax_r);

		disp8U2depth32F(buff, bufff, (float)(focus*baseline), (float)amp, 0.f);

		binalyWeightedRangeFilter(bufff, bufff, Size(2 * brange_r + 1, 2 * brange_r + 1), brange_th, brange_method);

		bufff.convertTo(dest, CV_16U);
	}

	void PostFilterSet::filterDisp8U2Depth32F(Mat& src, Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method)
	{
		medianBlur(src, buff, 2 * median_r + 1);
		smallGaussianBlur(buff, buff, 2 * gaussian_r + 1, gaussian_r + 0.5);
		blurRemoveMinMax(buff, buff, minmax_r);

		disp8U2depth32F(buff, bufff, (float)(focus*baseline), (float)amp, 0.f);

		binalyWeightedRangeFilter(bufff, dest, Size(2 * brange_r + 1, 2 * brange_r + 1), brange_th, brange_method);
	}

	void PostFilterSet::filterDisp8U2Disp32F(Mat& src, Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method)
	{
		medianBlur(src, buff, 2 * median_r + 1);
		smallGaussianBlur(buff, buff, 2 * gaussian_r + 1, gaussian_r + 0.5);
		blurRemoveMinMax(buff, buff, minmax_r);

		buff.convertTo(bufff, CV_32F);
		binalyWeightedRangeFilter(bufff, bufff, Size(2 * brange_r + 1, 2 * brange_r + 1), brange_th, brange_method);

		bufff.convertTo(dest, CV_16U);
	}

	void PostFilterSet::operator()(Mat& src, Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, int brange_th, int brange_method)
	{
		medianBlur(src, buff, 2 * median_r + 1);
		smallGaussianBlur(buff, buff, 2 * gaussian_r + 1, gaussian_r + 0.5);
		blurRemoveMinMax(buff, buff, minmax_r);
		binalyWeightedRangeFilter(buff, dest, Size(2 * brange_r + 1, 2 * brange_r + 1), (float)brange_th, brange_method);
	}
}
