#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testVideoSubtitle()
{
	Mat src = imread("img/lenna.png");

	string wname = "testVideoSubtitle";
	namedWindow(wname);
	int sw = 0; createTrackbar("sw", wname, &sw, 1);//subtitle rendering mode
	int pos = 1; createTrackbar("pow", wname, &pos, 2);//subtitle position

	VideoSubtitle vs;
	vector<string> vstring = { "testVideoSubtitle", "press r key to restart" };
	vector<int> vfsize = { 30,20 };
	vs.setFontType("Times New Roman");
	vs.setVSpace(10);

	vs.setDisolveTime(1000, 2000);

	cp::UpdateCheck uc(sw, pos);
	int key = 0;
	while (key != 'q')
	{
		cp::drawGridMulti(src, Size(4,4), COLOR_RED);
			
		if (sw == 0)vs.showScriptDissolve(wname, src);
		if (sw == 1)vs.showTitleDissolve(wname, src);

		if (key == 'r' || uc.isUpdate(sw, pos))
		{
			vs.restart();
			vs.setTitle(src.size(), vstring, vfsize, Scalar(255, 255, 255), Scalar::all(0), VideoSubtitle::POSITION(pos));
		}

		key = waitKey(1);
	}
}