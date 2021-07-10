#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testMatInfo()
{
	Mat a = Mat::ones(3, 3, CV_32F);
	showMatInfo(a, "test");
	print_matinfo(a);
}