#include "ppmx.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	Mat imreadPPMX(string filename)
	{
		using namespace std;
		FILE* fp = fopen(filename.c_str(), "rb");
		if (fp == NULL)cout << " file open error" << filename << endl;

		char buff[256];
		fgets(buff, 256, fp);


		int width = 0;
		int height = 0;

		fgets(buff, 256, fp);
		if (sscanf(buff, "%d %d", &width, &height) != 2)
		{
			printf("couldn't figure out the size of image %s\n", filename);
		}
		fgets(buff, 256, fp);
		Mat ret = Mat::zeros(Size(width, height), CV_32F);
		fread(ret.data, sizeof(float), width*height, fp);

		fclose(fp);

		return ret;
	}

	Mat imreadPFM(string filename)
	{
		return imreadPPMX(filename);
	}

}