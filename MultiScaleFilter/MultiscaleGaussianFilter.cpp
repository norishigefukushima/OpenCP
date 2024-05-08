#include "multiscalefilter/MultiScaleFilter.hpp"
using namespace cv;
using namespace std;
namespace cp
{
	void MultiScaleGaussianFilter::pyramid(const Mat& src, Mat& dest)
	{
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		//buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);
		buildLaplacianPyramid(src, ImageStack, level, sigma_space);
		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}
		collapseLaplacianPyramid(ImageStack, ImageStack[0]);//override srcf for saving memory	

		ImageStack[0].convertTo(dest, src.type());
	}

	void MultiScaleGaussianFilter::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		Mat srcf;
		if (src.depth() == CV_8U) src.convertTo(srcf, CV_32F);
		else srcf = src.clone();

		buildDoGStack(srcf, ImageStack, sigma_space, level);
		for (int i = 0; i < ImageStack.size() - 1; i++) // ImageStack.size() - 1: no remap DC
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}
		collapseDoGStack(ImageStack, dest, src.depth());
	}

	void MultiScaleGaussianFilter::filter(const Mat& src, Mat& dest, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);
		//this->pyramidComputeMethod = Fast;

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}
}