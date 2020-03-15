#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void testPlot()
{
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

	Plot test;
	RNG rng;
	for (int j = 0; j < 100; j++)
	{
		
		for (int i = 0; i < 10; i++)
		{
			test.push_back(i, rng.uniform(0,i), 0);
			test.push_back(i, rng.uniform(0, i), 1);
		}

		test.plot("loop", false);
	}
}