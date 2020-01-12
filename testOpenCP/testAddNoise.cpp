#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testAddNoise(Mat& src)
{
	string wname = "add noise";
	namedWindow(wname);
	int sigma_g = 5; createTrackbar("gauss noise", wname, &sigma_g, 100);
	int spnoise = 0; createTrackbar("salt peppe noise", wname, &spnoise, 100);
	int seed = 0; createTrackbar("random seed", wname, &seed, 100);

	int key = 0;
	Mat dest;
	while (key != 'q')
	{
		addNoise(src, dest, (double)sigma_g, (double)spnoise*0.01, seed);
		cout << getPSNR(src, dest) << endl;
		imshow(wname, dest);
		key = waitKey(1);
	}
}