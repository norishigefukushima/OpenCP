#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testTimer(Mat& src)
{
	Timer t;
	t.setIgnoringThreshold(10);
	t.setCountMax(1000);

	Mat dest;
	int key = 0;
	while (key != 'q')
	{
		t.start();
		blur(src, dest, Size(7, 7));

		t.getpushLapTime();
		t.getLapTimeMedian(true);
		t.drawDistribution("dist", 300);
		key = waitKey(1);
	}
}

void testDestinationTimePrediction(Mat& src)
{
	const int iteration = 100;

	Mat dest;
	int key = 0;
	while (key != 'q')
	{
		cp::Timer tt("Timerrrrrrrrrr");

		DestinationTimePrediction t(iteration);
		for (int i = 0; i < iteration; i++)
		{
			t.predict();
			for (int j = 0; j < 100; j++)
				blur(src, dest, Size(7, 7));

		}
		key = waitKey(1);
	}
}