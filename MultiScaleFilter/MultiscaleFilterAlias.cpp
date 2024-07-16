#include "multiscalefilter/MultiScaleFilter.hpp"
#include "multiscalefilter/FourierSeriesExpansion.h"
using namespace cv;

namespace cp
{
	void localLaplacianFilterFull(InputArray src, OutputArray dest, const float sigma_range, const float sigma_space, const float boost, const int level)
	{
		LocalMultiScaleFilterFull msf;
		Mat s = src.getMat();
		Mat d = dest.getMat();
		msf.filter(s, d, sigma_range, sigma_space, boost, level, cp::MultiScaleFilter::Pyramid);
	}

	void localLaplacianFilter(InputArray src, OutputArray dest, const float sigma_range, const float sigma_space, const float boost, const int level)
	{
		LocalMultiScaleFilter msf;
		Mat s = src.getMat();
		Mat d = dest.getMat();
		msf.filter(s, d, sigma_range, sigma_space, boost, level, cp::MultiScaleFilter::Pyramid);
	}

	void localDoGFilter(InputArray src, OutputArray dest, const float sigma_range, const float sigma_space, const float boost, const int level)
	{
		LocalMultiScaleFilter msf;
		Mat s = src.getMat();
		Mat d = dest.getMat();
		msf.filter(s, d, sigma_range, sigma_space, boost, level, cp::MultiScaleFilter::DoG);
	}

	void fastLocalLaplacianFilter(InputArray src, OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level)
	{
		LocalMultiScaleFilterInterpolation msf;
		Mat s = src.getMat();
		Mat d = dest.getMat();
		msf.filter(s, d, order, sigma_range, sigma_space, boost, level, cp::MultiScaleFilter::Pyramid);
	}

	void fastLocalDoGFilter(InputArray src, OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level)
	{
		LocalMultiScaleFilterInterpolation msf;
		Mat s = src.getMat();
		Mat d = dest.getMat();
		msf.filter(s, d, order, sigma_range, sigma_space, boost, level, cp::MultiScaleFilter::DoG);
	}

	void FourierLocalLaplacianFilter(InputArray src, OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level)
	{
		LocalMultiScaleFilterFourier msf;
		Mat s = src.getMat();
		Mat d = dest.getMat();
		msf.filter(s, d, order, sigma_range, sigma_space, boost, level, cp::MultiScaleFilter::Pyramid);
	}

	void FourierLocalDoGFilter(InputArray src, OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level)
	{
		LocalMultiScaleFilterFourier msf;
		Mat s = src.getMat();
		Mat d = dest.getMat();
		msf.filter(s, d, order, sigma_range, sigma_space, boost, level, cp::MultiScaleFilter::DoG);
	}
}