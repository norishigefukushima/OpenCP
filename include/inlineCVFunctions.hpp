#pragma once
#include <opencv2/core/cvdef.h>
#include <opencv2/imgproc.hpp>
#include <string>

#define print_debug(a)						std::cout << #a << ": " << a << std::endl
#define print_debug2(a, b)					std::cout << #a << ": " << a <<", "<< #b << ": " << b << std::endl
#define print_debug3(a, b, c)				std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c << std::endl;
#define print_debug4(a, b, c, d)			std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d << std::endl;
#define print_debug5(a, b, c, d, e)			std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e << std::endl;
#define print_debug6(a, b, c, d, e, f)		std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e <<", "<< #f << ": " << f << std::endl;
#define print_debug7(a, b, c, d, e, f, g)	std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e <<", "<< #f << ": " << f << ", "<< #g << ": " << g <<std::endl;

namespace cp
{
	template<typename Type>
	inline int typeToCVDepth();
	template<> inline int typeToCVDepth<char>() { return CV_8S; }
	template<> inline int typeToCVDepth<uchar>() { return CV_8U; }
	template<> inline int typeToCVDepth<short>() { return CV_16S; }
	template<> inline int typeToCVDepth<ushort>() { return CV_16U; }
	template<> inline int typeToCVDepth<int>() { return CV_32S; }
	template<> inline int typeToCVDepth<float>() { return CV_32F; }
	template<> inline int typeToCVDepth<double>() { return CV_64F; }

	inline std::string getInterpolationName(const int method)
	{
		std::string ret = "no supported";
		switch (method)
		{
		case cv::INTER_NEAREST:		ret = "NEAREST"; break;
		case cv::INTER_LINEAR:		ret = "LINEAR"; break;
		case cv::INTER_CUBIC:		ret = "CUBIC"; break;
		case cv::INTER_AREA:		ret = "AREA"; break;
		case cv::INTER_LANCZOS4:	ret = "LANCZOS4"; break;
		default:
			break;
		}
		return ret;
	}
}