#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testIsSame()
{
	Mat a(256, 256, CV_8U);
	Mat b(256, 256, CV_8U);
	randu(a, 0, 255);
	randu(b, 0, 255);
	cp::isSame(a, b);
	cp::isSame(a, a);
	cp::isSame(a, b, 10);
	cp::isSame(a, a, 10);
}