#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testSplitMerge(Mat& src)
{
	int iter = 10;
	Mat dest(src.size(), src.type());
	for (int i = 0; i < iter; i++)
	{
		Timer t("base");
		vector<Mat> dst(3);

		split(src, dst);
		merge(dst, dest);
	}
	Mat dest2(src.size(), src.type());
	for (int i = 0; i < iter; i++)
	{
		Timer t("my");
		vector<Mat> dst;

		splitConvert(src, dst);
		//split(src, dst);
		//merge(dst, dest);
		mergeConvert(dst, dest2, false);
	}
	cp::guiAlphaBlend(dest, dest2);
}