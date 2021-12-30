#pragma once
#include <opencv2/core/cvdef.h>
#include <opencv2/imgproc.hpp>
#include <string>

#define print_debug(a)							std::cout << #a << ": " << a << std::endl
#define print_debug2(a, b)						std::cout << #a << ": " << a <<", "<< #b << ": " << b << std::endl
#define print_debug3(a, b, c)					std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c << std::endl;
#define print_debug4(a, b, c, d)				std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d << std::endl;
#define print_debug5(a, b, c, d, e)				std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e << std::endl;
#define print_debug6(a, b, c, d, e, f)			std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e <<", "<< #f << ": " << f << std::endl;
#define print_debug7(a, b, c, d, e, f, g)		std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e <<", "<< #f << ": " << f << ", "<< #g << ": " << g << std::endl;
#define print_debug8(a, b, c, d, e, f, g, h)	std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e <<", "<< #f << ": " << f << ", "<< #g << ": " << g << #h << ": " << h << std::endl;
#define print_debug9(a, b, c, d, e, f, g, h, i)	std::cout << #a << ": " << a <<", "<< #b << ": " << b <<", "<< #c << ": " << c <<", "<< #d << ": " << d <<", "<< #e << ": " << e <<", "<< #f << ": " << f << ", "<< #g << ": " << g << #h << ": " << h << #i << ": " << i << std::endl;

inline void print_mat_format(cv::Mat& src, std::string mes = "", std::string format = "%8.2f ")
{
	printf("%s:\n", mes.c_str());
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			printf(format.c_str(), src.at<double>(j, i));
		}
		printf("\n");
	}
	printf("\n");
}

#define print_mat(a) print_mat_format(a, #a)

template<int borderType> int ref_lb(int n);
template<int borderType> int ref_rb(int n, int width);
template<int borderType> int ref_tb(int n, int width);
template<int borderType> int ref_bb(int n, int width, int height);

inline int ref_lborder(int n, int borderType)
{
	int ret = 0;
	switch (borderType)
	{
	case cv::BORDER_REPLICATE:
		ret = std::max(n, 0); break;
	case cv::BORDER_REFLECT:
		ret = n < 0 ? -n - 1 : n; break;
	case cv::BORDER_REFLECT101:
	default:
		ret = std::abs(n); break;
	}
	return ret;
}

inline int ref_rborder(int n, int width, int borderType)
{
	int ret = 0;
	switch (borderType)
	{
	case cv::BORDER_REPLICATE:
		ret = std::min(n, width - 1); break;
	case cv::BORDER_REFLECT:
		ret = n < width ? n : 2 * width - n - 1; break;
	case cv::BORDER_REFLECT101:
	default:
		ret = width - 1 - std::abs(width - 1 - n); break;
	}
	return ret;
}

inline int ref_tborder(int n, int width, int borderType)
{
	int ret = 0;
	switch (borderType)
	{
	case cv::BORDER_REPLICATE:
		ret = std::max(n, 0) * width; break;
	case cv::BORDER_REFLECT:
		ret = (n < 0 ? -n - 1 : n) * width; break;
	case cv::BORDER_REFLECT101:
	default:
		ret = std::abs(n) * width; break;
	}
	return ret;
}

inline int ref_bborder(int n, int width, int height, int borderType)
{
	int ret = 0;
	switch (borderType)
	{
	case cv::BORDER_REPLICATE:
		ret = std::min(n, height - 1) * width; break;
	case cv::BORDER_REFLECT:
		ret = (n < height ? n : 2 * height - n - 1) * width; break;
	case cv::BORDER_REFLECT101:
	default:
		ret = (height - 1 - std::abs(height - 1 - n)) * width; break;
	}
	return ret;
}

//cv::BORDER_REPLICATE;
template<>
inline int ref_lb<1>(int n)
{
	return std::max(n, 0);
}

template<>
inline int ref_rb<1>(int n, int width)
{
	return std::min(n, width - 1);
}

template<>
inline int ref_tb<1>(int n, int width)
{
	return std::max(n, 0) * width;
}

template<>
inline int ref_bb<1>(int n, int width, int height)
{
	return std::min(n, height - 1) * width;
}

//cv::BORDER_REFLECT;
template<>
inline int ref_lb<2>(int n)
{
	return n < 0 ? -n - 1 : n;
}

template<>
inline int ref_rb<2>(int n, int width)
{
	return n < width ? n : 2 * width - n - 1;
}

template<>
inline int ref_tb<2>(int n, int width)
{
	return (n < 0 ? -n - 1 : n) * width;
}

template<>
inline int ref_bb<2>(int n, int width, int height)
{
	return (n < height ? n : 2 * height - n - 1) * width;
}

//cv::BORDER_REFLECT101:
template<>
inline int ref_lb<4>(int n)
{
	return std::abs(n);
}

template<>
inline int ref_rb<4>(int n, int width)
{
	return width - 1 - std::abs(width - 1 - n);
}

template<>
inline int ref_tb<4>(int n, int width)
{
	return std::abs(n) * width;
}

template<>
inline int ref_bb<4>(int n, int width, int height)
{
	return (height - 1 - std::abs(height - 1 - n)) * width;
}

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

	inline cv::Mat cvt8UC3(cv::InputArray src)
	{
		if (src.type() == CV_8UC3)return src.getMat();
		cv::Mat ret;
		src.getMat().convertTo(ret, CV_8U);
		if(src.channels()==1) cvtColor(ret, ret, cv::COLOR_GRAY2BGR);
		return ret;
	}

	inline std::string getDepthName(cv::Mat& src)
	{
		const int depth = src.depth();
		std::string ret;
		switch (depth)
		{
		case CV_8U: ret = "CV_8U"; break;
		case CV_8S:ret = "CV_8S"; break;
		case CV_16U:ret = "CV_16U"; break;
		case CV_16S:ret = "CV_16S"; break;
		case CV_32S:ret = "CV_32S"; break;
		case CV_32F:ret = "CV_32F"; break;
		case CV_64F:ret = "CV_64F"; break;
		case CV_16F:ret = "CV_16F"; break;
		default: ret = "not support this type of depth."; break;
		}
		return ret;
	}

	inline std::string getDepthName(int depth)
	{
		std::string ret;
		switch (depth)
		{
		case CV_8U: ret = "CV_8U"; break;
		case CV_8S:ret = "CV_8S"; break;
		case CV_16U:ret = "CV_16U"; break;
		case CV_16S:ret = "CV_16S"; break;
		case CV_32S:ret = "CV_32S"; break;
		case CV_32F:ret = "CV_32F"; break;
		case CV_64F:ret = "CV_64F"; break;
		case CV_16F:ret = "CV_16F"; break;
		default: ret = "not support this type of depth."; break;
		}
		return ret;
	}

	inline int get_avx_element_size(int cv_depth)
	{
		int ret = 0;
		switch (cv_depth)
		{
		case CV_8U:
		case CV_8S:
			ret = 32; break;
		case CV_16U:
		case CV_16S:
			ret = 16; break;
		case CV_32F:
		case CV_32S:
			ret = 8; break;
		case CV_64F:
			ret = 4; break;
		default:
			break;
		}
		return ret;
	}
}
