#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testStat()
{
	Mat src = imread("img/lenna.png");

	string wname = "testVideoSubtitle";
	namedWindow(wname);
	int sw = 0; createTrackbar("sw", wname, &sw, 1);
	int sigma = 5; createTrackbar("sigma", wname, &sigma, 100);
	int size = 1; createTrackbar("size", wname, &size, 10000);
	cp::ConsoleImage ci;
	cp::Stat st;
	int key = 0;
	while (key != 'q')
	{
		RNG rng(getTickCount());
		for (int i = 0; i < 10000; i++)
		{
			if (sw == 0) st.push_back(rng.gaussian(sigma));
			else st.push_back(rng.uniform(0.0, (double)sigma));

		}
		for (int i = 0; i < size; i++)
		{
			st.pop_back();
		}

		ci("size %d", st.getSize());
		ci("sum  %8.2f", st.getSum());
		ci("ave  %8.2f", st.getMean());
		ci("med  %8.2f", st.getMedian());
		ci("min  %8.2f", st.getMin());
		ci("max  %8.2f", st.getMax());
		ci("std  %8.2f", st.getStd());
		ci("var  %8.2f", st.getVar());
		ci.show();
		st.drawDistribution(wname);
		st.clear();
		key = waitKey(100);
	}
}