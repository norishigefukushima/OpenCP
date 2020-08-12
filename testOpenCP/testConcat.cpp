#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testConcat()
{
	vector<Mat> kodak24;
	for (int i = 1; i < 25; i++)
	{
		Mat temp = imread(format("img/kodak/kodim%02d.png", i), 1);
		if (temp.cols < temp.rows)transpose(temp, temp);
		//imshow("a", temp); waitKey();
		kodak24.push_back(temp);
	}

	Mat img;
	cp::concat(kodak24, img, 4, 6);
	cp::imshowResize("kodak24", img, Size(), 0.25, 0.25);

	Mat temp;
	cp::concatExtract(img, Size(768, 512), temp, 1, 1);
	imshow("extract (1,1)", temp);

	cp::concatExtract(img, Size(768, 512), temp, 12);
	imshow("extract 12", temp);

	vector<Mat> v;
	cp::concatSplit(img, v, Size(768, 512));
	imshow("concatSplit[3]", v[3]);

	vector<Mat> v2;
	cp::concatSplit(img, v2, 4, 6);
	imshow("concatSplit[3] v2", v2[3]);

	waitKey();
}