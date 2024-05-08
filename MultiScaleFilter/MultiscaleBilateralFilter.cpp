#include "multiscalefilter/MultiScaleFilter.hpp"
using namespace cv;
using namespace std;
namespace cp
{
	void MultiScaleBilateralFilter::buildDoBFStack(const Mat& src, vector<Mat>& DoBFStack, const float sigma_r, const float sigma_s, const int level)
	{
		if (DoBFStack.size() != level + 1) DoBFStack.resize(level + 1);
		for (int l = 0; l <= level; l++) DoBFStack[l].create(src.size(), CV_32F);

		Mat prev = src;
		for (int l = 1; l <= level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_s, l);
			const int r = getGaussianRadius(sigma_l);
			const Size ksize(2 * r + 1, 2 * r + 1);

			cv::bilateralFilter(src, DoBFStack[l], 2 * r + 1, sigma_r, sigma_l, borderType);

			subtract(prev, DoBFStack[l], DoBFStack[l - 1]);
			prev = DoBFStack[l];
		}
	}

	void MultiScaleBilateralFilter::pyramid(const Mat& src, Mat& dest)
	{
		cout << "not support" << endl;
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);
		buildLaplacianPyramid(ImageStack[0], ImageStack, level, sigma_space);


		if (pyramidComputeMethod) cv::buildPyramid(ImageStack[0], ImageStack, level, borderType);
		else buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);

		buildLaplacianPyramid(ImageStack[0], ImageStack, level, 1.f);

		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}

		collapseLaplacianPyramid(ImageStack, ImageStack[0]);//override srcf for saving memory	

		ImageStack[0].convertTo(dest, src.type());
	}

	void MultiScaleBilateralFilter::dog(const Mat& src, Mat& dest)
	{
		//cout << "BF DoG" << endl;
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		buildDoBFStack(src, ImageStack, sigma_range, sigma_space, level);
		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}

		collapseDoGStack(ImageStack, dest, src.depth());
	}

	void MultiScaleBilateralFilter::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);
		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}
}