#include "detailEnhancement.hpp"
#include "recursiveBilateralFilter.hpp"
#include "bilateralFilter.hpp"
#include "guidedFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void detailEnhancementBox(InputArray src, OutputArray dest, const int r, const float boost)
	{
		const int d = 2 * r + 1;

		Mat smooth;
		boxFilter(src, smooth, src.depth(), Size(d, d));
		addWeighted(src, 1.0 + boost, smooth, -boost, 0, dest);
	}

	void detailEnhancementGauss(InputArray src, OutputArray dest, const int r, const float sigma_space, const float boost)
	{
		const int d = 2 * r + 1;

		Mat smooth;
		if (src.depth() == CV_32F)
		{
			GaussianBlur(src, smooth, Size(d, d), sigma_space);
			addWeighted(src, 1.0 + boost, smooth, -boost, 0, dest);
		}
		else
		{
			Mat src32f = src.getMat();
			src32f.convertTo(src32f, CV_32F);
			GaussianBlur(src32f, smooth, Size(d, d), sigma_space);
			addWeighted(src32f, 1.0 + boost, smooth, -boost, 0, src32f);
			src32f.convertTo(dest, src.type());
		}
	}

	void detailEnhancementBilateral(InputArray src, OutputArray dest, const int r, const float sigma_color, const float sigma_space, const float boost)
	{
		const int d = 2 * r + 1;
		Mat smooth;
		cv::bilateralFilter(src, smooth, d, sigma_color, sigma_space);
		addWeighted(src, 1.0 + boost, smooth, -boost, 0, dest);
	}

	void detailEnhancementGuided(InputArray src_, OutputArray dest, const int r, const float eps, const float boost)
	{
		const int d = 2 * r + 1;

		Mat smooth;
		Mat src = src_.getMat();
		guidedImageFilter(src, src, dest, r, eps);
		addWeighted(src, 1.0 + boost, smooth, -boost, 0, dest);
	}

}
