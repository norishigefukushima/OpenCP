#include "multiscalefilter/MultiScaleFilter.hpp"
using namespace cv;
using namespace std;
namespace cp
{
	void MultiScaleGuidedImageFilter::buildDoGIFStack(const Mat& src, vector<Mat>& DoGIFStack, const float eps, const float sigma_s, const int level)
	{
		if (DoGIFStack.size() != level + 1) DoGIFStack.resize(level + 1);
		for (int l = 0; l <= level; l++) DoGIFStack[l].create(src.size(), CV_32F);

		Mat prev = src;
		for (int l = 1; l <= level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_s, l);
			const int r = getGaussianRadius(sigma_l);
			const Size ksize(2 * r + 1, 2 * r + 1);
			guidedImageGaussianFilterGray(src, DoGIFStack[l], r, sigma_l, eps);
			subtract(prev, DoGIFStack[l], DoGIFStack[l - 1]);
			prev = DoGIFStack[l];
		}
	}

	void MultiScaleGuidedImageFilter::buildCoGIFStack(const Mat& src, vector<Mat>& CoGIFStack, const float eps, const float sigma_s, const int level)
	{
		if (CoGIFStack.size() != level + 1) CoGIFStack.resize(level + 1);
		for (int l = 0; l <= level; l++) CoGIFStack[l].create(src.size(), CV_32F);

		Mat prev = src;
		for (int l = 1; l <= level; l++)
		{
			const float sigma_l = (float)getPyramidSigma(sigma_s, l);
			const int r = getGaussianRadius(sigma_l);
			const Size ksize(2 * r + 1, 2 * r + 1);
			guidedImageGaussianFilterGray(src, CoGIFStack[l], r, sigma_l, eps);
			divide(prev, CoGIFStack[l] + FLT_EPSILON, CoGIFStack[l - 1]);
			CoGIFStack[l - 1] -= 1.f;
			prev = CoGIFStack[l];
		}
	}
	

	void MultiScaleGuidedImageFilter::pyramid(const Mat& src, Mat& dest)
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
		collapseLaplacianPyramid(ImageStack, dest, src.depth());
	}

	void MultiScaleGuidedImageFilter::dog(const Mat& src, Mat& dest)
	{
		//cout << "GIF DoG" << endl;
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		buildDoGIFStack(src, ImageStack, sigma_range, sigma_space, level);
		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}
		collapseDoGStack(ImageStack, dest, src.depth());
	}

	void MultiScaleGuidedImageFilter::cog(const Mat& src, Mat& dest)
	{
		//cout << "GIF CoG" << endl;
		ImageStack.resize(level + 1);
		initRangeTable(sigma_range, boost);

		if (src.depth() == CV_8U) src.convertTo(ImageStack[0], CV_32F);
		else src.copyTo(ImageStack[0]);

		buildCoGIFStack(src, ImageStack, sigma_range, sigma_space, level);
		for (int i = 0; i < ImageStack.size() - 1; i++)
		{
			remap(ImageStack[i], ImageStack[i], 0.f, sigma_range, boost);
		}
		collapseCoGStack(ImageStack, dest, src.depth());
	}

	void MultiScaleGuidedImageFilter::filter(const Mat& src, Mat& dest, const float eps, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		//allocSpaceWeight(sigma_space);
		this->sigma_range = eps;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		//freeSpaceWeight();
	}
}