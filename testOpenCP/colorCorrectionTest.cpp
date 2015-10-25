#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void guiColorCorrectionTest(Mat& src, Mat& ref)
{
	Mat cmat;
	findColorMatrixAvgStdDev(src, ref, cmat, 100, 200);
	Mat dest;
	cout << cmat << endl;
	cvtColorMatrix(src, dest, cmat);
	guiAlphaBlend(src, dest);
}