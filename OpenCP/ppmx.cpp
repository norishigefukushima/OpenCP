#include "ppmx.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void imwritePPMX(string filename, cv::Mat& img)
	{
		using namespace std;
		FILE* fp = fopen(filename.c_str(), "wb");
		if (fp == NULL)cout << " file open error" << filename << endl;
		else
		{
			fprintf(fp, "FP\n");
			fprintf(fp, "%d %d\n", img.cols, img.rows);
			fwrite(img.data, sizeof(float), img.size().area(), fp);
		}
		fclose(fp);
	}

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
			printf("couldn't figure out the size of image %s\n", filename.c_str());
		}
		fgets(buff, 256, fp);
		Mat ret = Mat::zeros(Size(width, height), CV_32F);
		fread(ret.data, sizeof(float), width * height, fp);

		fclose(fp);

		return ret;
	}
}