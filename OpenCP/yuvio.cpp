#include "yuvio.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	YUVReader::YUVReader()
	{
		buff = NULL;
	}
	YUVReader::~YUVReader()
	{
		delete[] buff;
		fclose(fp);
	}

	YUVReader::YUVReader(string name, cv::Size size, int frame_max)
	{
		init(name, Size(size.width, size.height), frame_max);
	}

	void YUVReader::init(string name, Size size, int frame_max)
	{
		buff = NULL;
		width = size.width;
		height = size.height;
		imageSize = width * height;
		imageCSize = imageSize * 3;
		yuvSize = size.width * size.height + size.width * size.height / 2;

		fp = NULL;
		fp = fopen(name.c_str(), "rb");
		if (fp == NULL)
		{
			fprintf(stderr, "%s is invalid file name\n", name.c_str());
			return;
		}
		buff = new char[yuvSize];

		isloop = true;
		framemax = frame_max;
		frameCount = 0;
	}


	void YUVReader::readNext(Mat& dest)
	{
		fread(buff, sizeof(char), yuvSize, fp);

		Mat src(Size(width, cvRound(height * 1.5)), CV_8U, buff);
		cvtColor(src, dest, COLOR_YUV420p2BGR);

		frameCount++;
		if (frameCount == framemax)
		{
			if (isloop)
				fseek(fp, 0, 0);
			else
				fseek(fp, -yuvSize, SEEK_CUR);
		}
	}

	bool YUVReader::read(Mat& dest, int frame)
	{
		if (frame < framemax && frame >= 0)
		{
			fseek(fp, (frame - frameCount) * yuvSize, SEEK_CUR);
			fread(buff, sizeof(char), yuvSize, fp);
			Mat src(Size(width, cvRound(height * 1.5)), CV_8U, buff);
			cvtColor(src, dest, COLOR_YUV420p2BGR);

			frameCount = frame + 1;
			if (frameCount == framemax)
			{
				if (isloop)
					fseek(fp, 0, 0);
				else
					fseek(fp, -yuvSize, SEEK_CUR);
			}
			return true;
		}
		else
		{
			return false;
		}
	}

	void readYUVGray(string fname, OutputArray dest, Size size, int frame)
	{
		dest.create(size, CV_8U);
		FILE* fp = fopen(fname.c_str(), "rb");
		if (fp == NULL)cout << fname << " open error\n";
		const int fsize = size.area() + size.area() * 2 / 4;

		fseek(fp, fsize * frame, SEEK_SET);
		Mat data = dest.getMat();
		fread(data.data, sizeof(char), size.area(), fp);
		//cout<<size.area()<<endl;
		fflush(fp);
		fclose(fp);
		//imshow("aa",dest);waitKey();
	}

	void readYUV2BGR(string fname, OutputArray dest, Size size, int frame)
	{
		Mat temp = Mat::zeros(Size(size.width, cvRound(size.height * 1.5)), CV_8U);
		FILE* fp = fopen(fname.c_str(), "rb");
		if (fp == NULL)cout << fname << " open error\n";
		const int fsize = size.area() + size.area() * 2 / 4;

		fseek(fp, fsize * frame, SEEK_SET);

		fread(temp.data, sizeof(char), cvRound(size.area() * 1.5), fp);

		cvtColor(temp, dest, COLOR_YUV420p2RGB);
		fclose(fp);
		//imshow("aa",dest);waitKey();
	}

	void writeYUVBGR(string fname, InputArray src)
	{
		Mat yuv;
		cvtColor(src, yuv, COLOR_BGRA2YUV_YV12);

		Size size = src.size();
		FILE* fp = fopen(fname.c_str(), "wb");
		if (fp == NULL)cout << fname << " open error\n";
		const int fsize = size.area() + size.area() * 2 / 4;

		fwrite(yuv.data, sizeof(char), fsize, fp);
		fflush(fp);
		fclose(fp);
	}

	void writeYUVGray(string fname, InputArray src_)
	{
		Mat src = src_.getMat();
		Size size = src.size();
		FILE* fp = fopen(fname.c_str(), "wb");
		if (fp == NULL)cout << fname << " open error\n";
		const int fsize = size.area() + size.area() * 2 / 4;


		uchar* buff = new uchar[fsize];
		memset(buff, 0, fsize);
		memcpy(buff, src.data, size.area());

		fwrite(buff, sizeof(char), fsize, fp);
		fflush(fp);
		fclose(fp);
		delete[] buff;
		//imshow("ss",src);waitKey();
	}

	void readY16(string fname, OutputArray dest, Size size, int frame)
	{
		dest.create(size, CV_16S);
		FILE* fp = fopen(fname.c_str(), "rb");
		if (fp == NULL)cout << fname << " open error\n";
		const int fsize = size.area();

		fseek(fp, fsize * frame, SEEK_CUR);
		Mat data = dest.getMat();
		fread(data.data, sizeof(short), size.area(), fp);
		//cout<<size.area()<<endl;
		fflush(fp);
		fclose(fp);
		//imshow("aa",dest);waitKey();
	}

	void writeYUV(InputArray src_, string name, int mode)
	{
		Mat src; cvtColor(src_, src, COLOR_BGR2YUV);

		int s = 1;
		if (src.type() == CV_16S) s = 2;
		if (mode == 0 || s == 1)
		{
			FILE* fp = fopen(name.c_str(), "wb");
			fwrite(src.data, sizeof(uchar), src.size().area() * s, fp);
			fclose(fp);
		}
		else
		{
			uchar* buff = new uchar[sizeof(uchar) * src.size().area() * 2];
			int size = 0;
			int s2 = 0;
			short* s = src.ptr<short>(0);
			for (int i = 0; i < src.size().area(); i++)
			{
				int v = s[i] + 128;
				if (v < 255 && v >= 0)
				{
					buff[size++] = v;

				}
				else
				{
					s2++;
					buff[size++] = 255;
					short* a = (short*)buff;
					*a = v - 128;
					buff += 2;
					size += 2;
				}
			}
			FILE* fp = fopen(name.c_str(), "wb");
			fwrite(buff, sizeof(uchar), size, fp);
			fclose(fp);
			delete[] buff;
		}
	}

}