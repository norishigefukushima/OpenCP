#include <opencp.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void guiHazeRemoveTest()
{
	HazeRemove hz;
	//Mat haze = imread("img/haze/swans.png");
	//Mat haze = imread("img/haze/forest.png");
	Mat haze = imread("img/doumori.jpg");
	//Mat haze = imread("img/haze/haze2.jpg");
	hz.gui(haze, "haze");
}