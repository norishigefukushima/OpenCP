#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiPlotTest()
{
	Plot test;
	for (int i = 0; i < 10; i++)
	{
		test.push_back(i, 2 * i);
		test.push_back(i, 3 * i, 1);
	}

	test.erase(3, 0);
	test.insert(3, 10, 3, 0);

	test.plot();
}