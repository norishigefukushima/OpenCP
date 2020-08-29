#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testTiling(Mat& src)
{
	Mat srcf = convert(src, CV_32F);	
	//Mat srcf = convert(src, CV_64F);

	vector<Mat> v;
	vector<Mat> sv(3);
	split(srcf, v);
	Mat destf = Mat::zeros(src.size(), srcf.type());

	const Size div = Size(4, 4);
	const int r = 2;

	for (int c = 0; c < 3; c++)
		createSubImage(v[c], sv[c], div, Point(1, 2), 2);

	Mat sub; merge(sv, sub);
	imshowScale("sub", sub);
	setSubImage(sub, destf, div, Point(1, 2), 2);

	imshowScale("tile", destf);
	waitKey();
}