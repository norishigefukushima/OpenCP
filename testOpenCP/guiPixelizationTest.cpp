#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void guiPixelizationTest()
{
	Mat src = imread("img/lamp.png");
	guiPixelization("pixel", src);
}
