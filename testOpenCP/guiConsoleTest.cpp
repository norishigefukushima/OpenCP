#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void consoleTest()
{
	ConsoleImage ci(Size(640, 480));

	Timer t;
	int count = 0;
	int key = 0;
	while (key != 'q')
	{
		Mat src;
		Mat dest;
		cout << count << endl;
		cout << "===========================" << endl;
		ci("%d", count);
		ci(Scalar(0, 255, 0), "===========================");
		{
			Timer t("image input", TIME_MSEC);
			src = imread("img/lenna.png");
			ci("image input     : %f ms", t.getTime());
		}
		{
			Timer t("box filter", TIME_MSEC);
			boxFilter(src, dest, src.depth(), Size(7, 7));
			ci("box filter      : %f ms", t.getTime());
		}
		{
			Timer t("gauss filter", TIME_MSEC);
			GaussianBlur(dest, dest, Size(7, 7), 2);
			ci("gauss filter    : %f ms", t.getTime());
		}
		{
			Timer t("median filter", TIME_MSEC);
			medianBlur(dest, dest, 5);
			ci("median filter   : %f ms", t.getTime());
		}
		{
			Timer t("bilateralfilter", TIME_MSEC);
			Mat temp = dest.clone();
			bilateralFilter(temp, dest, 7, 2, 2);
			ci("bilateral filter: %f ms", t.getTime());
		}
		cout << endl;
		ci.show();
		imshow("out", dest);
		key = waitKey(1);
		count++;
	}
}