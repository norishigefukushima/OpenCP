#include "ppmx.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void imwritePPMX(string filename, cv::Mat& img)
	{
		using namespace std;
		CV_Assert(img.depth() == CV_32F || img.depth() == CV_64F);

		FILE* fp = fopen(filename.c_str(), "wb");
		if (fp == NULL)
		{
			cout << " file open error" << filename << endl;
			fclose(fp);
			return;
		}
		else
		{
			if (img.depth() == CV_32F)
			{
				if (img.channels() == 1) fprintf(fp, "Fp\n");
				if (img.channels() == 3) fprintf(fp, "FP\n");
				fprintf(fp, "%d %d\n", img.cols, img.rows);
				fwrite(img.data, sizeof(float), img.size().area() * img.channels(), fp);
			}
			else
			{
				if (img.channels() == 1) fprintf(fp, "Dp\n");
				if (img.channels() == 3) fprintf(fp, "DP\n");
				fprintf(fp, "%d %d\n", img.cols, img.rows);
				fwrite(img.data, sizeof(double), img.size().area() * img.channels(), fp);
			}
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
		const bool isDouble = buff[0] == 'D' ? true : false;
		const bool isColor = buff[1] == 'P' ? true : false;

		int width = 0;
		int height = 0;
		fgets(buff, 256, fp);
		if (sscanf(buff, "%d %d", &width, &height) != 2)
		{
			printf("couldn't figure out the size of image %s\n", filename.c_str());
		}
		//fgets(buff, 256, fp);

		Mat ret;
		if (isDouble)
		{
			if (isColor)
			{
				ret = Mat::zeros(Size(width, height), CV_64FC3);
				fread(ret.data, sizeof(double), width * height * 3, fp);
			}
			else
			{
				ret = Mat::zeros(Size(width, height), CV_64FC1);
				fread(ret.data, sizeof(double), width * height, fp);
			}
		}
		else
		{
			if (isColor)
			{
				ret = Mat::zeros(Size(width, height), CV_32FC3);
				fread(ret.data, sizeof(float), width * height * 3, fp);
			}
			else
			{
				ret = Mat::zeros(Size(width, height), CV_32FC1);
				fread(ret.data, sizeof(float), width * height, fp);
			}
		}

		fclose(fp);

		return ret;
	}
}