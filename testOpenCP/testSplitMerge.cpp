#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testSplitMerge(Mat& src)
{
	int iter = 10;
	Mat dest(src.size(), src.type());
	Mat dest32f;
	for (int i = 0; i < iter; i++)
	{
		Timer t("base: split&merge");
		vector<Mat> dst;

		split(src, dst);
		merge(dst, dest);
	}

	for (int i = 0; i < iter; i++)
	{
		Timer t("base: split convert merge");
		vector<Mat> dst;
		
		split(src, dst);
		vector<Mat> dst32f(dst.size());
		for (int i = 0; i < dst.size(); i++)
		{
			dst[i].convertTo(dst32f[i], CV_32F);
		}
		merge(dst32f, dest32f);
	}

	Mat dest2(src.size(), src.type());
	for (int i = 0; i < iter; i++)
	{
		Timer t("my: : split convert merge");
		vector<Mat> dst;

		splitConvert(src, dst, CV_32F);
		//split(src, dst);
		merge(dst, dest2);
		//mergeConvert(dst, dest2, false);
	}
	cp::guiAlphaBlend(dest, dest2);
}