#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testStat()
{
	string wname = "testVideoSubtitle";
	namedWindow(wname);
	int isInt = 1; createTrackbar("isInt", wname, &isInt, 1);
	int sw = 0; createTrackbar("sw", wname, &sw, 1);
	int sigma = 9; createTrackbar("sigma", wname, &sigma, 100);
	const int dataMax = 1000000;
	int size = 1; createTrackbar("size", wname, &size, dataMax);
	int iteration = 2; createTrackbar("iteration", wname, &iteration, 100); setTrackbarMin("iteration", wname, 1);
	cp::ConsoleImage ci;
	cp::Stat st;
	int key = 0;
	
	while (key != 'q')
	{
		RNG rng(getTickCount());
		if (sw == 0)
		{
			if (isInt == 0)
			{
				for (int i = 0; i < dataMax; i++)
				{
					double sum = 0.0;
					for (int n = 0; n < iteration; n++) sum += rng.gaussian(sigma);
					st.push_back(sum/iteration+127.0);
				}
			}
			else
			{
				for (int i = 0; i < dataMax; i++)
				{
					double sum = 0.0;
					for (int n = 0; n < iteration; n++) sum += cvRound(rng.gaussian(sigma));
					st.push_back(sum/iteration + 127.0);
				}
			}
		}
		else
		{
			if (isInt == 0)
			{
				for (int i = 0; i < dataMax; i++)
				{
					st.push_back(rng.uniform(0.0, (double)sigma) + 127.0);
				}
			}
			else
			{
				for (int i = 0; i < dataMax; i++)
				{
					st.push_back(cvRound(rng.uniform(0.0, (double)sigma) + 127.0));
				}
			}

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
		if (isInt == 0)
		{
			st.drawDistribution(wname);
		}
		else
		{
			//st.drawDistribution(wname, 256, 0, 255);
			st.drawDistributionStepSigma(wname, 1.f/iteration, 4.f);
		}
		st.clear();
		key = waitKey(100);
	}
}