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