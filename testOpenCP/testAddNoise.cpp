#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testAddNoise(Mat& src)
{
	string wname = "add noise";
	namedWindow(wname);
	int color = 0; createTrackbar("color", wname, &color, 1);
	int noise_type = 0; createTrackbar("noise_type", wname, &noise_type, 1);
	int sigma_g = 5; createTrackbar("g:gauss sigma", wname, &sigma_g, 100);
	int spnoise = 0; createTrackbar("g:salt peppe noise", wname, &spnoise, 100);
	int seed = 0; createTrackbar("g:random seed", wname, &seed, 100);
	int quality = 80; createTrackbar("j:quality", wname, &quality, 100);

	int key = 0;
	Mat dest;
	Mat show;
	cp::UpdateCheck uc(noise_type);
	while (key != 'q')
	{
		Mat s;
		if (color == 0)
		{
			if(src.channels()==3) cvtColor(src, s, COLOR_BGR2GRAY);
			else s = src;
		}
		else
		{
			s = src;
		}

		if (uc.isUpdate(noise_type))
		{
			if (noise_type == 0) displayOverlay(wname, "Gaussian&SP noise", 3000);
			if (noise_type == 1) displayOverlay(wname, "JPEG noise", 3000);
		}

		if (noise_type == 0) addNoise(s, dest, (double)sigma_g, (double)spnoise*0.01, seed);
		if (noise_type == 1) addJPEGNoise(s, dest, quality);

		if (dest.channels() == 1)cvtColor(dest, show, COLOR_GRAY2BGR);
		else dest.copyTo(show);
		cv::addText(show, format("PSNR %5.2f d", getPSNR(s, dest)), Point(36, 36), "Consolas", 36, Scalar::all(0));
		
		imshow(wname, show);
		key = waitKey(1);
	}
}