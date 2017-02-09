#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT YUVReader
	{
		FILE* fp;
		int framemax;
		char* buff;
		bool isloop;

		int yuvSize;

	public:
		int width;
		int height;
		int imageSize;
		int imageCSize;
		int frameCount;

		void init(std::string name, cv::Size size, int frame_max);
		YUVReader(std::string name, cv::Size size, int frame_max);
		YUVReader();
		~YUVReader();

		void readNext(cv::Mat& dest);
		bool read(cv::Mat& dest, int frame);
	};

	CP_EXPORT void readYUVGray(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void readYUV2BGR(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void writeYUVBGR(std::string fname, cv::InputArray src);
	CP_EXPORT void writeYUVGray(std::string fname, cv::InputArray src);
	CP_EXPORT void readY16(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void writeYUV(cv::Mat& InputArray, std::string name, int mode = 1);
}