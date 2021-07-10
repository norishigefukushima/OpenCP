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
	const int iteration = 15;

	Mat srcf = convert(src, CV_32F);
	Mat dest;
	int key = 0;

	{
		{
			int paramMax = 10;
			DestinationTimePrediction t(paramMax);
			for (int i = 0; i < paramMax; i++)
			{
				for (int j = 0; j < iteration; j++)
				{
					//GaussianBlur(srcf, dest, Size(2 * i + 1, 2 * i + 1), 100);
					//medianBlur(src, dest, 2 * i + 1);
					bilateralFilter(src, dest, 2 * i + 1, 30, 30);
					//blur(src, dest, Size(7, 7));
				}
				t.predict(3, false);
			}
		}
		cout << endl;
		{
			int paramMax = 10;
			DestinationTimePrediction t(iteration);
			for (int j = 0; j < iteration; j++)
			{
				for (int i = 0; i < paramMax; i++)
				{
					//GaussianBlur(srcf, dest, Size(2 * i + 1, 2 * i + 1), 100);
					//medianBlur(src, dest, 2 * i + 1);
					bilateralFilter(src, dest, 2 * i + 1, 30, 30);
					//blur(src, dest, Size(7, 7));
				}
				t.predict(0, false);
			}
		}

	}
}