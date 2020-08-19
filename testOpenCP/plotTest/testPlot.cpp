#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void testPlot()
{
	//if (false)
	{
		Plot test;
		test.setYRange(-1, 1);
		test.setXRange(0, 320);
		const int amp = 100;

		for (int i = 0; i < int(CV_PI*amp); i++)
		{
			test.push_back(i, sin(CV_PI * i * 1.0/amp));
		}

		test.plot();
	}

	if (false)
	{
		Plot test;
		test.setXLabel("x label");
		test.setYLabel("y label");
		for (int i = 0; i < 10; i++)
		{
			test.push_back(i, 2 * i);
			test.push_back(i, 3 * i, 1);
		}

		test.erase(3, 0);
		test.insert(3, 10, 3, 0);

		test.plot();
	}

	if (false)
	{
		Plot test;
		test.setXLabel("x label");
		test.setYLabel("y label");
		for (int i = 0; i < 10; i++)
		{
			test.push_back(i, 2 * i);
			test.push_back(i, 3 * i, 1);
		}

		test.erase(3, 0);
		test.insert(3, 10, 3, 0);

		test.plot();
	}

	if (false)
	{
		Plot test;
		test.setXLabel("log(x) label");
		test.setLogScaleX(true);
		test.setYLabel("y label");

		for (int i = 1; i < 10000; i++)
		{
			test.push_back(i, i, 0);
		}
		test.plot();
	}

	Plot test;
	RNG rng;
	for (int j = 0; j < 100; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			test.push_back(i, rng.uniform(0, i), 0);
			test.push_back(i, rng.uniform(0, i), 1);
		}

		test.plot("loop", false);
	}
}