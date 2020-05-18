```cpp
#include "pch.h"
#include "function.hpp"
#include "upsample_common.hpp"
#include "fmath/fmath.hpp"
#include <inlineSIMDFunctions.hpp>
using namespace cv;
using namespace std;

void vizCubicKernel(const int scale, string wname = "upsample cubic kernel")
{
	Mat weightmap(scale*scale, 16, CV_32F);
	//weightmap.setTo(1);

	namedWindow(wname);
	int ix = 0; createTrackbar("ix", wname, &ix, scale - 1);
	int iy = 0; createTrackbar("iy", wname, &iy, scale - 1);
	int alpha = 100; createTrackbar("(alpha-200)*0.01", wname, &alpha, 400);
	int key = 0;
	Mat show;

	while (key != 'q')
	{
		setCubicWeight4x4(weightmap, (alpha - 200)*0.01);
		displayStatusBar(wname, "(alpha-200)*0.01!!");

		int idx = iy * scale + ix;
		float* w = weightmap.ptr<float>(idx);

		int s = (int)sqrt(weightmap.cols);
		Mat kernelt(s, s, CV_32F, w);
		Mat kernel = kernelt.clone();
		//cout << kernel << endl;
		kernel += 1.f;
		Mat kernel8u; kernel.convertTo(kernel8u, CV_8U, 128);
		resize(kernel8u, show, Size(), 100, 100, INTER_NEAREST);
		imshow(wname, show);
		key = waitKey(1);
	}
}

void vizKernel(Mat& weightmap, string wname = "upsample kernel")
{
	const int scale = (int)sqrt(weightmap.rows);
	namedWindow(wname);
	int ix = 0; createTrackbar("ix", wname, &ix, scale - 1);
	int iy = 0; createTrackbar("iy", wname, &iy, scale - 1);
	//int aaa = 100; createTrackbar("aaa", wname, &aaa, 200);
	int key = 0;
	Mat show;

	while (key != 'q')
	{
		int idx = iy * scale + ix;
		float* w = weightmap.ptr<float>(idx);

		int s = (int)sqrt(weightmap.cols);
		Mat kernelt(s, s, CV_32F, w);
		Mat kernel = kernelt.clone();
		//cout << kernel << endl;
		kernel += 1.f;
		Mat kernel8u; kernel.convertTo(kernel8u, CV_8U, 128);
		resize(kernel8u, show, Size(), 100, 100, INTER_NEAREST);
		imshow(wname, show);
		key = waitKey(1);
	}
}

void NEDI16(const Mat &srcImg, Mat &dstImg_, int n_pow, float threshold, int WindowSize)
{
	CV_Assert(!srcImg.empty());
	//1?{
	if (n_pow == 0)
	{
		dstImg_ = srcImg.clone();
		return;
	}

	CV_Assert(n_pow > 0);
	CV_Assert(srcImg.channels() == 1 || srcImg.channels() == 3);
	CV_Assert(WindowSize % 2 == 0 && WindowSize != 0);

	Mat dstImg;
	/*if (srcImg.channels() == 3)//?J???[
	{
		vector<Mat> planes(3), dst_rgb(3);

		split(srcImg, planes);

		NEDI(planes[0], dst_rgb[0], n_pow);
		NEDI(planes[1], dst_rgb[1], n_pow);
		NEDI(planes[2], dst_rgb[2], n_pow);

		merge(dst_rgb, dstImg);
		return;
	}*/

	int srcType = srcImg.type();
	Mat srcfImg;
	for (int iteration = n_pow; iteration > 0; iteration--)
	{
		if (iteration == n_pow)
		{
			copyMakeBorder(srcImg, srcfImg, WindowSize + 4, WindowSize + 4, WindowSize + 4, WindowSize + 4, BORDER_REPLICATE);
			srcfImg.convertTo(srcfImg, CV_32FC1, 1.0f / 255);

			dstImg = Mat::zeros(srcfImg.size() * 2, CV_32FC1);
		}
		else
		{
			copyMakeBorder(dstImg, srcfImg, WindowSize + 4, WindowSize + 4, WindowSize + 4, WindowSize + 4, BORDER_REPLICATE);
			dstImg = Mat::zeros(srcfImg.size() * 2, CV_32FC1);
		}

		//(2y,2x)?ɓ??͉????u
//#pragma omp parallel for
		for (int y = 0; y < srcfImg.rows; ++y)
		{
			float *src = srcfImg.ptr<float>(y);
			float *dst = dstImg.ptr<float>(y * 2);

			for (int x = 0; x < srcfImg.cols; ++x)
			{
				dst[x * 2] = src[x];
			}
		}

		const int width = dstImg.cols;
		const int b = WindowSize + 4;
		//first step of edge-directed interpolation
#pragma omp parallel for
		for (int y = b; y < dstImg.rows - b; y += 2)//(2y+1,2x+1)
		{
			Mat alpha_coeff = Mat::zeros(Size(16, 1), dstImg.type());
			Mat vectorY = Mat::zeros(Size(WindowSize*WindowSize, 1), dstImg.type());
			Mat matrixC = Mat::zeros(Size(16, WindowSize*WindowSize), dstImg.type());

			float *dst = dstImg.ptr<float>(y + 1, 1);
			float *vecY = vectorY.ptr<float>(0);

			for (int x = b; x < dstImg.cols - b; x += 2)
			{
				int count = 0;

				for (int Y = 0; Y < WindowSize * 2; Y += 2)
				{
					float *window = dstImg.ptr<float>(y - (WindowSize + 0) + Y, x - (WindowSize + 0));

					for (int X = 0; X < WindowSize * 2; X += 2)
					{
						float *matC = matrixC.ptr<float>(count);
						/*
						matC[0] = window[X - 2 - width * 2];
						matC[1] = window[X + 2 - width * 2];
						matC[2] = window[X - 2 + width * 2];
						matC[3] = window[X + 2 + width * 2];
						*/

						matC[0] = window[X - 4 - width * 4];
						matC[1] = window[X - 2 - width * 4];
						matC[2] = window[X + 2 - width * 4];
						matC[3] = window[X + 4 - width * 4];

						matC[4] = window[X - 4 - width * 2];
						matC[5] = window[X - 2 - width * 2];
						matC[6] = window[X + 2 - width * 2];
						matC[7] = window[X + 4 - width * 2];

						matC[8] = window[X - 4 + width * 2];
						matC[9] = window[X - 2 + width * 2];
						matC[10] = window[X + 2 + width * 2];
						matC[11] = window[X + 4 + width * 2];

						matC[12] = window[X - 4 + width * 4];
						matC[13] = window[X - 2 + width * 4];
						matC[14] = window[X + 2 + width * 4];
						matC[15] = window[X + 4 + width * 4];

						vecY[count++] = window[X];
					}
				}

				solve(matrixC.t()*matrixC, matrixC.t()*vectorY.t(), alpha_coeff, DECOMP_LU);//MMSE

				float *a = alpha_coeff.ptr<float>(0);

				if (a[0] == 0.f)
				{
					dst[x] = (
						+1.f*dst[x - 3 - 3 * width] - 9.f*dst[x - 1 - 3 * width] - 9.f*dst[x + 1 - 3 * width] + 1.f*dst[x + 3 - 3 * width]
						- 9.f*dst[x - 3 - 1 * width] + 81.f*dst[x - 1 - 1 * width] + 81.f*dst[x + 1 - 1 * width] - 9.f*dst[x + 3 - 1 * width]
						- 9.f*dst[x - 3 + 1 * width] + 81.f*dst[x - 1 + 1 * width] + 81.f*dst[x + 1 + 1 * width] - 9.f* dst[x + 3 + 1 * width]
						+ 1.f*dst[x - 3 + 3 * width] - 9.f* dst[x - 1 + 3 * width] - 9.f* dst[x + 1 + 3 * width] + 1.f*dst[x + 3 + 3 * width]
						) / 256.0f;

					//dst[x] = (dst[x - 1 - width] + dst[x + 1 - width] + dst[x - 1 + width] + dst[x + 1 + width]) / 4.0f;
				}
				else
				{


					//dst[x] = a[0] * dst[x - 1 - width] + a[1] * dst[x + 1 - width] + a[2] * dst[x - 1 + width] + a[3] * dst[x + 1 + width];

					dst[x] = (
						+a[0] * dst[x - 3 - 3 * width] + a[1] * dst[x - 1 - 3 * width] + a[2] * dst[x + 1 - 3 * width] + a[3] * dst[x + 3 - 3 * width]
						+ a[4] * dst[x - 3 - 1 * width] + a[5] * dst[x - 1 - 1 * width] + a[6] * dst[x + 1 - 1 * width] + a[7] * dst[x + 3 - 1 * width]
						+ a[8] * dst[x - 3 + 1 * width] + a[9] * dst[x - 1 + 1 * width] + a[10] * dst[x + 1 + 1 * width] + a[11] * dst[x + 3 + 1 * width]
						+ a[12] * dst[x - 3 + 3 * width] + a[13] * dst[x - 1 + 3 * width] + a[14] * dst[x + 1 + 3 * width] + a[15] * dst[x + 3 + 3 * width]
						);

				}
			}
		}


		//second step of edge-directed interpolation
		for (int C = 0; C < 2; ++C)// C=0->(2y,2x+1)?CC=1 ->(2y+1,2x)
		{
#pragma omp parallel for
			for (int y = WindowSize * 2 + C; y < dstImg.rows - WindowSize * 2 - C; y += 2)
			{
				Mat alpha_coeff(Size(4, 1), dstImg.type());
				Mat vectorY(Size(WindowSize*WindowSize, 1), dstImg.type());
				Mat matrixC(Size(4, WindowSize*WindowSize), dstImg.type());

				float *vecY = vectorY.ptr<float>(0);
				float *dst = dstImg.ptr<float>(y + 1);

				for (int x = WindowSize * 2 + 4 - C; x < dstImg.cols - (WindowSize * 2 + 4 - C); x += 2)
				{
					int count = 0, point = 0;

					for (int Y = 0; Y < WindowSize; ++Y)
					{
						float *window = dstImg.ptr<float>(y + Y, x);

						for (int X = 0; X < WindowSize; ++X)
						{
							float *matC = matrixC.ptr<float>(count);

							point = -(WindowSize - 4) - width * X + X + Y;//window?ł̈ʒu

							matC[0] = window[point - 2];
							matC[1] = window[point - width * 2];
							matC[2] = window[point + width * 2];
							matC[3] = window[point + 2];

							vecY[count++] = window[point];
						}
					}

					solve(matrixC.t()*matrixC, matrixC.t()*vectorY.t(), alpha_coeff, DECOMP_LU);//MMSE

					float *alpha = alpha_coeff.ptr<float>(0);

					if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0.f)
					{
						dst[x] = (dst[x - 1] + dst[x - width] + dst[x + width] + dst[x + 1]) / 4.0f;
					}
					else
					{
						dst[x] = alpha[0] * dst[x - 1] + alpha[1] * dst[x - width] + alpha[2] * dst[x + width] + alpha[3] * dst[x + 1];
					}
				}
			}
		}

		dstImg = dstImg(Rect((WindowSize + 4) * 2, (WindowSize + 4) * 2, dstImg.cols - (WindowSize + 4) * 4, dstImg.rows - (WindowSize + 4) * 4));

	}

	dstImg.convertTo(dstImg_, srcType, 255);

	return;
}

inline float interpolation(const uchar lt, const uchar rt, const uchar lb, const uchar rb, const float alpha, const float beta)
{
	return beta * alpha*rb +
		beta * (1.f - alpha)*lb +
		(1.f - beta)*alpha * rt +
		(1.f - beta)*(1.f - alpha)*lt;
}


template<typename T>
class MatB_
{
	Mat s;
	int rep_width = 0;
	int rep_height = 0;
public:
	inline int rep_x(const int x)
	{
		return min(max(x, 0), rep_width);
	}
	inline int rep_y(const int y)
	{
		return min(max(y, 0), rep_height);
	}

	T operator()(int j, int i)
	{
		return s.at<T>(rep_y(j), rep_x(i));
	}
	MatB_(Mat& src)
	{
		s = src;
		rep_width = src.cols - 1;
		rep_height = src.rows - 1;
	}
};

typedef MatB_<uchar> MatB1b;
typedef MatB_<float> MatB1f;

void upsample_edge_linear(Mat& src, Mat& dst, int eth)
{
	Mat3b emap = Mat::zeros(dst.size(), CV_8UC3);
	Mat1b s = src;
	Mat1b d = dst;
	MatB1b dd(d);
	float th = eth;

	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			int x = i >> 1;
			int y = j >> 1;
			if (i % 2 == 0 && j % 2 == 0) d(j, i) = s(y, x);
		}
	}

	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			if (i % 2 == 1 && j % 2 == 1)
			{
				uchar e1 = abs(dd(j - 1, i - 1) - dd(j + 1, i + 1));
				uchar e2 = abs(dd(j - 1, i + 1) - dd(j + 1, i - 1));
				if (e1 > th && e2 < th)
				{
					d(j, i) = saturate_cast<uchar>((dd(j - 1, i + 1) + dd(j + 1, i - 1))*0.5f);
					emap(j, i) = Vec3b(0, 255, 255);
				}
				else if (e2 > th && e1 < th)
				{
					d(j, i) = saturate_cast<uchar>((dd(j - 1, i - 1) + dd(j + 1, i + 1))*0.5f);
					emap(j, i) = Vec3b(0, 255, 0);
				}
				else
				{
					d(j, i) = saturate_cast<uchar>((dd(j - 1, i - 1)
						+ dd(j - 1, i + 1)
						+ dd(j + 1, i - 1)
						+ dd(j + 1, i + 1))*0.25f);
				}
			}
		}
	}

	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			if ((i % 2 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 2 == 0))
			{
				uchar e1 = abs(dd(j, i - 1) - dd(j, i + 1));
				uchar e2 = abs(dd(j - 1, i) - dd(j + 1, i));
				//if ((1.f + e1) / (1.f + e2) > th)
				if (e1 > th && e2 < th)
				{
					d(j, i) = saturate_cast<uchar>((dd(j - 1, i) + dd(j + 1, i))*0.5f);
					emap(j, i) = Vec3b(0, 255, 0);
				}
				//else if ((1.f + e2) / (1.f + e1) > th)
				else if (e2 > th && e1 < th)
				{
					d(j, i) = saturate_cast<uchar>((dd(j, i - 1) + dd(j, i + 1))*0.5f);
					emap(j, i) = Vec3b(0, 255, 255);
				}
				else
				{
					d(j, i) = saturate_cast<uchar>((dd(j, i - 1) + dd(j, i + 1) + dd(j - 1, i) + dd(j + 1, i))*0.25f);
				}
			}
		}
	}

	//guiAlphaBlend(dst, emap);
}

void upsample_edge_cubic(Mat& src, Mat& dst)
{
	Mat1b s = src;
	Mat1b d = dst;
	float th = 1.1f;
	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			int x = i >> 1;
			int y = j >> 1;
			if (i % 2 == 0 && j % 2 == 0) d(j, i) = s(y, x);
		}
	}

	MatB1b dd(d);
	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			if (i % 2 == 1 && j % 2 == 1)
			{
				uchar e1 = abs(dd(j - 1, i - 1) - dd(j + 1, i + 1));
				uchar e2 = abs(dd(j - 1, i + 1) - dd(j + 1, i - 1));
				if ((1.f + e1) / (1.f + e2) > th) d(j, i) = saturate_cast<uchar>((-1 * d(j - 3, i + 3) + 9 * d(j - 1, i + 1) + 9 * d(j + 1, i - 1) - 1 * d(j + 3, i - 3))*0.0625f);
				if ((1.f + e2) / (1.f + e1) > th) d(j, i) = saturate_cast<uchar>((-1 * d(j - 3, i - 3) + 9 * d(j - 1, i - 1) + 9 * d(j + 1, i + 1) - 1 * d(j + 3, i + 3))*0.0625f);
				else
				{
					//d(j, i) = (d(j - 1, i + 1) + d(j + 1, i - 1) + d(j - 1, i - 1) + d(j + 1, i + 1)) >> 2;
					d(j, i) = saturate_cast<uchar>((
						(-1 * d(j - 3, i + 3) + 9 * d(j - 1, i + 1) + 9 * d(j + 1, i - 1) - 1 * d(j + 3, i - 3))
						+ (-1 * d(j - 3, i - 3) + 9 * d(j - 1, i - 1) + 9 * d(j + 1, i + 1) - 1 * d(j + 3, i + 3))
						)*0.0625f*0.5f);
				}
			}
		}
	}

	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			if ((i % 2 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 2 == 0))
			{
				uchar e1 = abs(dd(j, i - 1) - dd(j, i + 1));
				uchar e2 = abs(dd(j - 1, i) - dd(j + 1, i));
				if ((1.f + e2) / (1.f + e1) > th) d(j, i) = saturate_cast<uchar>((-1 * dd(j, i - 3) + 9 * dd(j, i - 1) + 9 * d(j, i + 1) - 1 * d(j, i + 3))*0.0625f);
				if ((1.f + e1) / (1.f + e2) > th) d(j, i) = saturate_cast<uchar>((-1 * dd(j - 3, i) + 9 * dd(j - 1, i) + 9 * d(j + 1, i) - 1 * d(j + 3, i))*0.0625f);
				else
				{
					//d(j, i) = (dd(j, i - 1) + dd(j, i + 1) + dd(j - 1, i) + dd(j + 1, i)) >> 2;
					d(j, i) = saturate_cast<uchar>((
						(-1 * dd(j, i - 3) + 9 * dd(j, i - 1) + 9 * d(j, i + 1) - 1 * d(j, i + 3))
						+ (-1 * dd(j - 3, i) + 9 * dd(j - 1, i) + 9 * d(j + 1, i) - 1 * d(j + 3, i))
						)*0.0625f*0.5f);
				}
			}
		}
	}
}

void upsample_edge_cubicp(Mat& src, Mat& dst)
{
	Mat1b s = src;
	Mat1b d = dst;
	float th = 1.1f;
	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			int x = i >> 1;
			int y = j >> 1;
			if (i % 2 == 0 && j % 2 == 0) d(j, i) = s(y, x);
		}
	}

	MatB1b dd(d);
	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			if (i % 2 == 1 && j % 2 == 1)
			{
				uchar e1 = abs(dd(j - 1, i - 1) - dd(j + 1, i + 1));
				uchar e2 = abs(dd(j - 1, i + 1) - dd(j + 1, i - 1));
				if ((1.f + e1) / (1.f + e2) > th) d(j, i) = saturate_cast<uchar>((-1 * d(j - 3, i + 3) + 9 * d(j - 1, i + 1) + 9 * d(j + 1, i - 1) - 1 * d(j + 3, i - 3))*0.0625f);
				if ((1.f + e2) / (1.f + e1) > th) d(j, i) = saturate_cast<uchar>((-1 * d(j - 3, i - 3) + 9 * d(j - 1, i - 1) + 9 * d(j + 1, i + 1) - 1 * d(j + 3, i + 3))*0.0625f);
				else
				{
					//d(j, i) = (d(j - 1, i + 1) + d(j + 1, i - 1) + d(j - 1, i - 1) + d(j + 1, i + 1)) >> 2;
					d(j, i) = saturate_cast<uchar>((
						(1 * d(j - 3, i - 3) - 9 * d(j - 3, i - 1) - 9 * d(j - 3, i + 1) + 1 * d(j - 3, i + 3))
						+ (-9 * d(j - 1, i - 3) + 81 * d(j - 1, i - 1) + 81 * d(j - 1, i + 1) - 9 * d(j - 1, i + 3))
						+ (-9 * d(j + 1, i - 3) + 81 * d(j + 1, i - 1) + 81 * d(j + 1, i + 1) - 9 * d(j + 1, i + 3))
						+ (1 * d(j + 3, i - 3) - 9 * d(j - 3, i - 1) - 9 * d(j + 3, i + 1) + 1 * d(j + 3, i + 3))
						)*0.00390625);
				}
			}
		}
	}

	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			if ((i % 2 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 2 == 0))
			{
				uchar e1 = abs(dd(j, i - 1) - dd(j, i + 1));
				uchar e2 = abs(dd(j - 1, i) - dd(j + 1, i));
				if ((1.f + e2) / (1.f + e1) > th) d(j, i) = saturate_cast<uchar>((-1 * dd(j, i - 3) + 9 * dd(j, i - 1) + 9 * d(j, i + 1) - 1 * d(j, i + 3))*0.0625f);
				if ((1.f + e1) / (1.f + e2) > th) d(j, i) = saturate_cast<uchar>((-1 * dd(j - 3, i) + 9 * dd(j - 1, i) + 9 * d(j + 1, i) - 1 * d(j + 3, i))*0.0625f);
				else
				{
					//d(j, i) = (dd(j, i - 1) + dd(j, i + 1) + dd(j - 1, i) + dd(j + 1, i)) >> 2;
					d(j, i) = saturate_cast<uchar>((
						(-1 * dd(j, i - 3) + 9 * dd(j, i - 1) + 9 * d(j, i + 1) - 1 * d(j, i + 3))
						+ (-1 * dd(j - 3, i) + 9 * dd(j - 1, i) + 9 * d(j + 1, i) - 1 * d(j + 3, i))
						)*0.0625f*0.5f);
				}
			}
		}
	}
}

class UpsampleEdge
{
	int interpolation_method = 0;
	int edge_method = 0;
public:

	void setInterpolationMethod(int method)
	{
		interpolation_method = method;
	}
	void setEdgeMethod(int method)
	{
		edge_method = method;
	}

	void upsampleLinear(Mat& src, Mat& dst, int method)
	{
		Mat1b s(src);
		Mat1b d(dst);
		if (method == 0)
		{
			for (int j = 0; j < dst.rows; j++)
			{
				for (int i = 0; i < dst.cols; i++)
				{
					int x = i >> 1;
					int y = j >> 1;
					const float alpha = (i - (x << 1))*0.5f;
					const float beta = (j - (y << 1))*0.5f;

					d(j, i) = saturate_cast<uchar>(interpolation(s(y, x), s(y, x + 1), s(y + 1, x), s(y + 1, x + 1), alpha, beta));
				}
			}
		}
		else if (method == 1)
		{
			for (int j = 0; j < dst.rows; j += 2)
			{
				for (int i = 0; i < dst.cols; i += 2)
				{
					int x = i >> 1;
					int y = j >> 1;
					d(j, i) = s(y, x);
					d(j, i + 1) = saturate_cast<uchar>((s(y, x) + s(y, x + 1))*0.5);
					d(j + 1, i) = saturate_cast<uchar>((s(y, x) + s(y + 1, x))*0.5);
					d(j + 1, i + 1) = saturate_cast<uchar>((s(y, x) + s(y + 1, x) + s(y, x + 1) + s(y + 1, x + 1))*0.25);
				}
			}
		}
		else if (method == 2)
		{
			Mat3b emap = Mat::zeros(dst.size(), CV_8UC3);

			for (int j = 0; j < dst.rows; j++)
			{
				for (int i = 0; i < dst.cols; i++)
				{
					int x = i >> 1;
					int y = j >> 1;
					if (i % 2 == 0 && j % 2 == 0)
					{
						d(j, i) = s(y, x);
					}
					if (i % 2 == 1 && j % 2 == 1)
					{
						d(j, i) = saturate_cast<uchar>((s(y, x) + s(y + 1, x) + s(y, x + 1) + s(y + 1, x + 1))*0.25);
						emap(j, i) = Vec3b(255, 255, 255);
					}
				}
			}

			for (int j = 0; j < dst.rows; j++)
			{
				for (int i = 0; i < dst.cols; i++)
				{
					if (i % 2 == 0 && j % 2 == 1)
					{
						d(j, i) = saturate_cast<uchar>((d(j, i - 1) + d(j, i + 1) + d(j - 1, i) + d(j + 1, i))*0.25);
						emap(j, i) = Vec3b(0, 255, 0);
					}
					if (i % 2 == 1 && j % 2 == 0)
					{
						d(j, i) = saturate_cast<uchar>((d(j, i - 1) + d(j, i + 1) + d(j - 1, i) + d(j + 1, i))*0.25);
						emap(j, i) = Vec3b(0, 0, 255);
					}
				}
			}
		}
		else if (method == 3)
		{
			for (int j = 0; j < dst.rows; j += 2)
			{
				for (int i = 0; i < dst.cols; i += 2)
				{
					int x = i >> 1;
					int y = j >> 1;
					d(j, i) = s(y, x);
					d(j, i + 1) = saturate_cast<uchar>((s(y, x) + s(y, x + 1))*0.5);
					d(j + 1, i) = saturate_cast<uchar>((s(y, x) + s(y + 1, x))*0.5);
					d(j + 1, i + 1) = saturate_cast<uchar>((s(y, x) + s(y + 1, x) + s(y, x + 1) + s(y + 1, x + 1))*0.25);
				}
			}
		}
	}

	void upsampleCubic(Mat& src, Mat& dst, int method)
	{
		MatB1b s(src);
		Mat1b d(dst);
		if (method == 0)
		{
			for (int j = 0; j < dst.rows; j += 2)
			{
				for (int i = 0; i < dst.cols; i += 2)
				{
					int x = i >> 1;
					int y = j >> 1;

					d(j, i) = s(y, x);

					d(j, i + 1) = saturate_cast<uchar>((-s(y, x - 1) + 9 * s(y, x) + 9 * s(y, x + 1) - s(y, x + 2))*0.0625);

					d(j + 1, i) = saturate_cast<uchar>((-s(y - 1, x) + 9 * s(y, x) + 9 * s(y + 1, x) - s(y + 2, x))*0.0625);

					d(j + 1, i + 1) = saturate_cast<uchar>((
						1 * s(y - 1, x - 1) - 9 * s(y - 1, x) - 9 * s(y - 1, x + 1) + 1 * s(y - 1, x + 2)
						- 9 * s(y + 0, x - 1) + 81 * s(y + 0, x) + 81 * s(y + 0, x + 1) - 9 * s(y + 0, x + 2)
						- 9 * s(y + 1, x - 1) + 81 * s(y + 1, x) + 81 * s(y + 1, x + 1) - 9 * s(y + 1, x + 2)
						+ 1 * s(y + 2, x - 1) - 9 * s(y + 2, x) - 9 * s(y + 2, x + 1) + 1 * s(y + 2, x + 2)
						)*0.00390625);//>>8
				}
			}
		}
		else if (method == 2)
		{
			int w11 = 13;
			int w12 = w11 * 2;
			int w22 = w11 * w11 * 2;
			double invw = 1.0 / (w22 * 2 - w11 * 4 - w12 * 2 + 4);
			for (int j = 0; j < dst.rows; j += 2)
			{
				for (int i = 0; i < dst.cols; i += 2)
				{
					int x = i >> 1;
					int y = j >> 1;

					d(j, i) = s(y, x);

					d(j, i + 1) = saturate_cast<uchar>((
						+1 * s(y - 1, x - 1) - w11 * s(y - 1, x) - w11 * s(y - 1, x + 1) + 1 * s(y - 1, x + 2)
						- w12 * s(y + 0, x - 1) + w22 * s(y + 0, x) + w22 * s(y + 0, x + 1) - w12 * s(y + 0, x + 2)
						+ 1 * s(y + 1, x - 1) - w11 * s(y + 1, x) - w11 * s(y + 1, x + 1) + 1 * s(y + 1, x + 2)
						)*invw);

					d(j + 1, i) = saturate_cast<uchar>((
						+1 * s(y - 1, x - 1) - w11 * s(y, x - 1) - w11 * s(y + 1, x - 1) + 1 * s(y + 2, x - 1)
						- w12 * s(y - 1, x + 0) + w22 * s(y, x + 0) + w22 * s(y + 1, x + 0) - w12 * s(y + 2, x + 0)
						+ 1 * s(y - 1, x + 1) - w11 * s(y, x + 1) - w11 * s(y + 1, x + 1) + 1 * s(y + 2, x + 1)
						)*invw);

					d(j + 1, i + 1) = saturate_cast<uchar>((
						1 * s(y - 1, x - 1) - 9 * s(y - 1, x) - 9 * s(y - 1, x + 1) + 1 * s(y - 1, x + 2)
						- 9 * s(y + 0, x - 1) + 81 * s(y + 0, x) + 81 * s(y + 0, x + 1) - 9 * s(y + 0, x + 2)
						- 9 * s(y + 1, x - 1) + 81 * s(y + 1, x) + 81 * s(y + 1, x + 1) - 9 * s(y + 1, x + 2)
						+ 1 * s(y + 2, x - 1) - 9 * s(y + 2, x) - 9 * s(y + 2, x + 1) + 1 * s(y + 2, x + 2)
						)*0.00390625);//>>8
				}
			}
		}
		else if (method == 1)
		{
			for (int j = 0; j < dst.rows; j += 2)
			{
				for (int i = 0; i < dst.cols; i += 2)
				{
					int x = i >> 1;
					int y = j >> 1;

					d(j, i) = s(y, x);

					d(j, i + 1) = saturate_cast<uchar>((-s(y, x - 1) + 9 * s(y, x) + 9 * s(y, x + 1) - s(y, x + 2))*0.0625);

					d(j + 1, i) = saturate_cast<uchar>((-s(y - 1, x) + 9 * s(y, x) + 9 * s(y + 1, x) - s(y + 2, x))*0.0625);

					d(j + 1, i + 1) = saturate_cast<uchar>((
						1 * s(y - 1, x - 1) - 9 * s(y - 1, x) - 9 * s(y - 1, x + 1) + 1 * s(y - 1, x + 2)
						- 9 * s(y + 0, x - 1) + 81 * s(y + 0, x) + 81 * s(y + 0, x + 1) - 9 * s(y + 0, x + 2)
						- 9 * s(y + 1, x - 1) + 81 * s(y + 1, x) + 81 * s(y + 1, x + 1) - 9 * s(y + 1, x + 2)
						+ 1 * s(y + 2, x - 1) - 9 * s(y + 2, x) - 9 * s(y + 2, x + 1) + 1 * s(y + 2, x + 2)
						)*0.00390625);//>>8

					/*
					d(j + 1, i + 1) = saturate_cast<uchar>((
						  1 * s(y - 1, x + 0) -  9 * s(y - 1, x + 1) -  9 * s(y - 1, x + 2) + 1 * s(y - 1, x + 3)
						- 9 * s(y + 0, x - 1) + 81 * s(y + 0, x + 0) + 81 * s(y + 0, x + 1) - 9 * s(y + 0, x + 2)
						- 9 * s(y + 1, x - 2) + 81 * s(y + 1, x - 1) + 81 * s(y + 1, x + 0) - 9 * s(y + 1, x + 1)
						+ 1 * s(y + 2, x - 3) -  9 * s(y + 2, x - 2) -  9 * s(y + 2, x - 1) + 1 * s(y + 2, x + 0)
						)*0.00390625);//>>8
						*/
						/*
						d(j + 1, i + 1) = saturate_cast<uchar>((
							  1 * s(y - 1, x - 2) -  9 * s(y - 1, x - 1) -  9 * s(y - 1, x - 2) + 1 * s(y - 1, x - 1)
							- 9 * s(y + 0, x - 1) + 81 * s(y + 0, x + 0) + 81 * s(y + 0, x - 1) - 9 * s(y + 0, x + 0)
							- 9 * s(y + 1, x + 0) + 81 * s(y + 1, x + 1) + 81 * s(y + 1, x + 0) - 9 * s(y + 1, x + 1)
							+ 1 * s(y + 2, x + 1) -  9 * s(y + 2, x + 2) -  9 * s(y + 2, x + 1) + 1 * s(y + 2, x + 2)
							)*0.00390625);//>>8
						*/

				}
			}
		}

	}

	void upsampleH264AVC6Tap(Mat& src, Mat& dst, int method)
	{
		MatB1b s(src);
		Mat1b d(dst);
		//if (method == 0)
		{
			for (int j = 0; j < dst.rows; j += 2)
			{
				for (int i = 0; i < dst.cols; i += 2)
				{
					int x = i >> 1;
					int y = j >> 1;

					d(j, i) = s(y, x);

					d(j, i + 1) = saturate_cast<uchar>((s(y, x - 2) - 5 * s(y, x - 1) + 20 * s(y, x) + 20 * s(y, x + 1) - 5 * s(y, x + 2) + 1 * s(y, x + 3))*0.03125);

					d(j + 1, i) = saturate_cast<uchar>((s(y - 2, x) - 5 * s(y - 1, x) + 20 * s(y, x) + 20 * s(y + 1, x) - 5 * s(y + 2, x) + 1 * s(y + 3, x))*0.03125);

					d(j + 1, i + 1) = saturate_cast<uchar>((
						+1 * s(y - 2, x - 2) - 5 * s(y - 2, x - 1) + 20 * s(y - 2, x) + 20 * s(y - 2, x + 1) - 5 * s(y - 2, x + 2) + 1 * s(y - 2, x + 3)
						- 5 * s(y - 1, x - 2) + 25 * s(y - 1, x - 1) - 100 * s(y - 1, x) - 100 * s(y - 1, x + 1) + 25 * s(y - 1, x + 2) - 5 * s(y - 1, x + 3)
						+ 20 * s(y + 0, x - 2) - 100 * s(y + 0, x - 1) + 400 * s(y + 0, x) + 400 * s(y + 0, x + 1) - 100 * s(y + 0, x + 2) + 20 * s(y + 0, x + 3)
						+ 20 * s(y + 1, x - 2) - 100 * s(y + 1, x - 1) + 400 * s(y + 1, x) + 400 * s(y + 1, x + 1) - 100 * s(y + 1, x + 2) + 20 * s(y + 1, x + 3)
						- 5 * s(y + 2, x - 2) + 25 * s(y + 2, x - 1) - 100 * s(y + 2, x) - 100 * s(y + 2, x + 1) + 25 * s(y + 2, x + 2) - 5 * s(y + 2, x + 3)
						+ 1 * s(y + 3, x - 2) - 5 * s(y + 3, x - 1) + 20 * s(y + 3, x) + 20 * s(y + 3, x + 1) - 5 * s(y + 3, x + 2) + 1 * s(y + 3, x + 3)
						)*0.0009765625);//>>8
				}
			}
		}
	}

	void upsampleDirectional(Mat& src, Mat& dst, int eth, int cubic = 9)
	{
		const float w = cubic;
		const float w2 = w * w;
		const float tw = w * 2 - 2;
		const float itw = 1.f / tw;
		const float itw2 = itw * itw;

		const float w61 = 20.f;
		const float w62 = -5.f;
		const float iw6 = 1.f / (2 * w61 + 2 * w62 + 2);
		const float ww61 = w61 * w61;
		const float ww62 = w61 * w62;
		const float ww63 = w62 * w62;
		const float iww6 = iw6 * iw6;

		Mat3b emap = Mat::zeros(dst.size(), CV_8UC3);
		Mat1b s = src;
		Mat1f d(dst.size());
		MatB1f dd(d);
		float th = eth;
		if (edge_method == 0)
		{
			eth *= 1;
		}
		else if (edge_method == 1)
		{
			eth *= 3;
		}
		else if (edge_method == 2)
		{
			eth *= 4;
		}

		for (int j = 0; j < dst.rows; j++)
		{
			for (int i = 0; i < dst.cols; i++)
			{
				int x = i >> 1;
				int y = j >> 1;
				d(j, i) = s(y, x);
			}
		}

		for (int j = 0; j < dst.rows; j++)
		{
			for (int i = 0; i < dst.cols; i++)
			{
				if (i % 2 == 1 && j % 2 == 1)
				{
					float e1, e2;
					if (edge_method == 0)
					{
						e1 = abs(dd(j - 1, i - 1) - dd(j + 1, i + 1));
						e2 = abs(dd(j - 1, i + 1) - dd(j + 1, i - 1));
					}
					else if (edge_method == 1)
					{
						e1 = abs(dd(j - 1, i - 3) - dd(j + 1, i - 1)) + abs(dd(j - 1, i - 1) - dd(j + 1, i + 1)) + abs(dd(j - 1, i + 1) - dd(j + 1, i + 3));
						e2 = abs(dd(j - 1, i - 1) - dd(j + 1, i - 3)) + abs(dd(j - 1, i + 1) - dd(j + 1, i - 1)) + abs(dd(j - 1, i + 3) - dd(j + 1, i + 1));
					}
					else if (edge_method == 2)
					{
						e1 = abs(dd(j - 1, i - 3) - dd(j + 1, i - 1)) + 2 * abs(dd(j - 1, i - 1) - dd(j + 1, i + 1)) + abs(dd(j - 1, i + 1) - dd(j + 1, i + 3));
						e2 = abs(dd(j - 1, i - 1) - dd(j + 1, i - 3)) + 2 * abs(dd(j - 1, i + 1) - dd(j + 1, i - 1)) + abs(dd(j - 1, i + 3) - dd(j + 1, i + 1));
					}

					if (e1 > th && e2 < th)
					{
						if (interpolation_method == 0)
						{
							d(j, i) = (dd(j - 1, i + 1) + dd(j + 1, i - 1))*0.5f;
						}
						else if (interpolation_method == 1 || interpolation_method == 2)
						{
							d(j, i) = (-1 * dd(j - 3, i + 3) + w * dd(j - 1, i + 1) + w * dd(j + 1, i - 1) - 1 * dd(j + 3, i - 3))*itw;
						}
						else
						{
							d(j, i) = (dd(j - 5, i + 5) + w62 * dd(j - 3, i + 3) + w61 * dd(j - 1, i + 1) + w61 * dd(j + 1, i - 1) + w62 * dd(j + 3, i - 3) + dd(j + 5, i - 5))*iw6;
						}

						emap(j, i) = Vec3b(0, 255, 255);
					}
					else if (e2 > th && e1 < th)
					{
						if (interpolation_method == 0)
						{
							d(j, i) = (dd(j - 1, i - 1) + dd(j + 1, i + 1))*0.5f;
						}
						else if (interpolation_method == 1 || interpolation_method == 2)
						{
							d(j, i) = (-dd(j - 3, i - 3) + w * dd(j - 1, i - 1) + w * dd(j + 1, i + 1) - dd(j + 3, i + 3))*itw;
						}
						else
						{
							d(j, i) = (dd(j - 5, i - 5) + w62 * dd(j - 3, i - 3) + w61 * dd(j - 1, i - 1) + w61 * dd(j + 1, i + 1) + w62 * dd(j + 3, i + 3) + dd(j + 5, i + 5))*iw6;
						}

						emap(j, i) = Vec3b(0, 255, 0);
					}
					else
					{
						if (interpolation_method == 0)
						{
							d(j, i) = (dd(j - 1, i - 1)
								+ dd(j - 1, i + 1)
								+ dd(j + 1, i - 1)
								+ dd(j + 1, i + 1))*0.25f;
						}
						else if (interpolation_method == 1)
						{
							d(j, i) = (
								(-1 * dd(j - 3, i + 3) + w * dd(j - 1, i + 1) + w * dd(j + 1, i - 1) - 1 * dd(j + 3, i - 3))
								+ (-1 * dd(j - 3, i - 3) + w * dd(j - 1, i - 1) + w * dd(j + 1, i + 1) - 1 * dd(j + 3, i + 3))
								)*itw*0.5f;
						}
						else if (interpolation_method == 2)
						{
							d(j, i) = (
								(dd(j - 3, i - 3) - w * dd(j - 3, i - 1) - w * dd(j - 3, i + 1) + dd(j - 3, i + 3))
								+ (-w * dd(j - 1, i - 3) + w2 * dd(j - 1, i - 1) + w2 * dd(j - 1, i + 1) - w * dd(j - 1, i + 3))
								+ (-w * dd(j + 1, i - 3) + w2 * dd(j + 1, i - 1) + w2 * dd(j + 1, i + 1) - w * dd(j + 1, i + 3))
								+ (dd(j + 3, i - 3) - w * dd(j - 3, i - 1) - w * dd(j + 3, i + 1) + dd(j + 3, i + 3))
								)*itw2;
						}
						else
						{
							d(j, i) = (
								+(1 * dd(j - 5, i - 5) + w62 * dd(j - 5, i - 3) + w61 * dd(j - 5, i - 1) + w61 * dd(j - 5, i + 1) + w62 * dd(j - 5, i + 3) + 1 * dd(j - 5, i + 5))
								+ (w62 * dd(j - 3, i - 5) + ww63 * dd(j - 3, i - 3) + ww62 * dd(j - 3, i - 1) + ww62 * dd(j - 3, i + 1) + ww63 * dd(j - 3, i + 3) + w62 * dd(j - 3, i + 5))
								+ (w61 * dd(j - 1, i - 5) + ww62 * dd(j - 1, i - 3) + ww61 * dd(j - 1, i - 1) + ww61 * dd(j - 1, i + 1) + ww62 * dd(j - 1, i + 3) + w61 * dd(j - 1, i + 5))
								+ (w61 * dd(j + 1, i - 5) + ww62 * dd(j + 1, i - 3) + ww61 * dd(j + 1, i - 1) + ww61 * dd(j + 1, i + 1) + ww62 * dd(j + 1, i + 3) + w61 * dd(j + 1, i + 5))
								+ (w62 * dd(j + 3, i - 5) + ww63 * dd(j + 3, i - 3) + ww62 * dd(j + 3, i - 1) + ww62 * dd(j + 3, i + 1) + ww63 * dd(j + 3, i + 3) + w62 * dd(j + 3, i + 5))
								+ (1 * dd(j + 5, i - 5) + w62 * dd(j + 5, i - 3) + w61 * dd(j + 5, i - 1) + w61 * dd(j + 5, i + 1) + w62 * dd(j + 5, i + 3) + 1 * dd(j + 5, i + 5))
								)*iww6;
						}
					}
				}
			}
		}

		for (int j = 0; j < dst.rows; j++)
		{
			for (int i = 0; i < dst.cols; i++)
			{
				if ((i % 2 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 2 == 0))
				{
					float e1, e2;
					if (edge_method == 0)
					{
						e1 = abs(dd(j, i - 1) - dd(j, i + 1));
						e2 = abs(dd(j - 1, i) - dd(j + 1, i));
					}
					else if (edge_method == 1)
					{
						e1 = abs(dd(j, i - 3) - dd(j, i - 1)) + abs(dd(j, i - 1) - dd(j, i + 1)) + abs(dd(j, i + 1) - dd(j, i + 3));
						e2 = abs(dd(j - 1, i - 2) - dd(j + 1, i - 2)) + abs(dd(j - 1, i) - dd(j + 1, i)) + abs(dd(j - 1, i + 2) - dd(j + 1, i + 2));
					}
					else if (edge_method == 2)
					{
						e1 = abs(dd(j, i - 3) - dd(j, i - 1)) + 2 * abs(dd(j, i - 1) - dd(j, i + 1)) + abs(dd(j, i + 1) - dd(j, i + 3));
						e2 = abs(dd(j - 1, i - 2) - dd(j + 1, i - 2)) + 2 * abs(dd(j - 1, i) - dd(j + 1, i)) + abs(dd(j - 1, i + 2) - dd(j + 1, i + 2));
					}

					//if ((1.f + e1) / (1.f + e2) > th)
					if (e1 > th && e2 < th)
					{
						if (interpolation_method == 0)
						{
							d(j, i) = (dd(j - 1, i) + dd(j + 1, i))*0.5f;
						}
						else if (interpolation_method == 1 || interpolation_method == 2)
						{
							d(j, i) = (-dd(j - 3, i) + w * dd(j - 1, i) + w * dd(j + 1, i) - dd(j + 3, i))*itw;
						}
						else
						{
							d(j, i) = (dd(j - 5, i) + w62 * dd(j - 3, i) + w61 * dd(j - 1, i) + w61 * dd(j + 1, i) + w62 * dd(j + 3, i) + dd(j + 3, i))*iw6;
						}

						emap(j, i) = Vec3b(0, 255, 0);
					}
					//else if ((1.f + e2) / (1.f + e1) > th)
					else if (e2 > th && e1 < th)
					{
						if (interpolation_method == 0)
						{
							d(j, i) = (dd(j, i - 1) + dd(j, i + 1))*0.5f;
						}
						else if (interpolation_method == 1 || interpolation_method == 2)
						{
							d(j, i) = (-dd(j, i - 3) + w * dd(j, i - 1) + w * dd(j, i + 1) - dd(j, i + 3))*itw;
						}
						else
						{
							d(j, i) = (dd(j, i - 5) + w62 * dd(j, i - 3) + w61 * dd(j, i - 1) + w61 * dd(j, i + 1) + w62 * dd(j, i + 3) + dd(j, i + 3))*iw6;
						}

						emap(j, i) = Vec3b(0, 255, 255);
					}
					else
					{
						if (interpolation_method == 0)
						{
							d(j, i) = (dd(j, i - 1) + dd(j, i + 1) + dd(j - 1, i) + dd(j + 1, i))*0.25f;
						}
						else if (interpolation_method == 1)
						{
							d(j, i) = (
								(-dd(j, i - 3) + w * dd(j, i - 1) + w * dd(j, i + 1) - dd(j, i + 3))
								+ (-dd(j - 3, i) + w * dd(j - 1, i) + w * dd(j + 1, i) - dd(j + 3, i))
								)*itw*0.5f;
						}
						else if (interpolation_method == 2)
						{
							d(j, i) = (
								(dd(j - 3, i - 0) - w * dd(j - 2, i + 1) - w * dd(j - 1, i + 2) + dd(j + 0, i + 3))
								+ (-w * dd(j - 2, i - 1) + w2 * dd(j - 1, i + 0) + w2 * dd(j + 0, i + 1) - w * dd(j + 1, i + 2))
								+ (-w * dd(j - 1, i - 2) + w2 * dd(j + 0, i - 1) + w2 * dd(j + 1, i + 0) - w * dd(j + 2, i + 1))
								+ (dd(j + 0, i - 3) - w * dd(j + 1, i - 2) - w * dd(j + 2, i - 1) + dd(j + 3, i + 0))
								)*itw2;

							/*d(j, i) = (
								(dd(j - 1, i - 2) - w * dd(j - 2, i - 1) - w * dd(j - 3, i + 0) + dd(j - 4, i + 1))
								+ (-w * dd(j + 1, i - 2) + w2 * dd(j + 0, i - 1) + w2 * dd(j - 1, i + 0) - w * dd(j - 2, i + 1))
								+ (-w * dd(j + 2, i - 3) + w2 * dd(j + 1, i + 0) + w2 * dd(j + 0, i + 1) - w * dd(j - 1, i + 2))
								+ (dd(j + 4, i - 1) - w * dd(j + 3, i + 0) - w * dd(j + 2, i + 1) + dd(j + 1, i + 2))
								)*itw2;*/

						}
						else
						{

							d(j, i) = (
								+(1 * dd(j - 5, i - 0) + w62 * dd(j - 4, i + 1) + w61 * dd(j - 3, i + 2) + w61 * dd(j - 2, i + 3) + w62 * dd(j - 1, i + 4) + 1 * dd(j - 0, i + 5))
								+ (w62 * dd(j - 4, i - 1) + ww63 * dd(j - 3, i + 0) + ww62 * dd(j - 2, i + 1) + ww62 * dd(j - 1, i + 2) + ww63 * dd(j - 0, i + 3) + w62 * dd(j + 1, i + 4))
								+ (w61 * dd(j - 3, i - 2) + ww62 * dd(j - 2, i - 1) + ww61 * dd(j - 1, i + 0) + ww61 * dd(j + 0, i + 1) + ww62 * dd(j + 1, i + 2) + w61 * dd(j + 2, i + 3))
								+ (w61 * dd(j - 2, i - 3) + ww62 * dd(j - 1, i - 2) + ww61 * dd(j + 0, i - 1) + ww61 * dd(j + 1, i + 0) + ww62 * dd(j + 2, i + 1) + w61 * dd(j + 3, i + 2))
								+ (w62 * dd(j - 1, i - 4) + ww63 * dd(j + 0, i - 3) + ww62 * dd(j + 1, i - 2) + ww62 * dd(j + 2, i - 1) + ww63 * dd(j + 3, i + 0) + w62 * dd(j + 4, i + 1))
								+ (1 * dd(j - 0, i - 5) + w62 * dd(j + 1, i - 4) + w61 * dd(j + 2, i - 3) + w61 * dd(j + 3, i - 2) + w62 * dd(j + 4, i - 1) + 1 * dd(j + 5, i + 0))
								)*iww6;

						}
					}
				}
			}
		}

		d.convertTo(dst, CV_8U);
		//guiAlphaBlend(dst, emap);
	}

};

/*
{
	int alpha = 0; createTrackbar("alpha", "linear edge", &alpha, 100);
	int swb = 0; createTrackbar("swb", "linear edge", &swb, 6);
	int swm = 3; createTrackbar("swm", "linear edge", &swm, 3);
	int sw = 3; createTrackbar("sw", "linear edge", &sw, 3);
	int ew = 0; createTrackbar("ew", "linear edge", &ew, 2);
	int ss = 33; createTrackbar("ss", "linear edge", &ss, 100);
	int eth = 10; createTrackbar("eth", "linear edge", &eth, 255);
	int c = 9; createTrackbar("c", "linear edge", &c, 20);
	int amp = 5; createTrackbar("diff", "diff", &amp, 20);
	UpsampleEdge ue;

	while (key != 'q')
	{
		Mat temp;
		//GaussianBlur(low_in, temp, Size(3, 3), ss*0.01);
		Mat gkernel = getGaussianKernel(3, ss*0.01);
		sepFilter2D(low_in, temp, CV_8U, gkernel, gkernel);
		resize(temp, temp, Size(), 0.5, 0.5, INTER_NEAREST);
		Mat low_ing; cvtColor(temp, low_ing, COLOR_BGR2GRAY);
		high_outg.setTo(0);

		if (swb == 0)
		{
			if (swm == 0)
			{
				NEDI(low_ing, high_outg, 1, 0.0, 8);
			}
			else
			{
				NEDI16(low_ing, high_outg, 1, 0.0, 8);
			}
		}
		else
		{
			int inter = (swb == 6) ? 0 : swb;
			if (swb == 1)cout << "Linear" << endl;
			if (swb == 2)cout << "Cubic" << endl;
			if (swb == 3)cout << "Area" << endl;
			if (swb == 4)cout << "Lanczos" << endl;
			if (swb == 5)cout << "LinearExact" << endl;
			if (swb == 6)cout << "Nearest" << endl;
			resize(low_ing, high_outg, Size(), 2, 2, inter);
			Mat dest;
			getRectSubPix(high_outg, high_outg.size(), Point2f(high_outg.cols / 2, high_outg.rows / 2), dest);
			dest.copyTo(high_outg);
		}
		//cout << answerg.size() << high_outg.size() << endl;
		cout << "linear eli: " << PSNR2(answerg, high_outg) << " dB" << endl;

		addWeighted(answerg, alpha*0.01, high_outg, 1.0 - alpha * 0.01, 0.0, high_outg);

		imshow("diff", amp*abs(high_outg - answerg));
		imshow("linear edge", high_outg);
		key = waitKey(1);
		if (key == 'n')
		{
			Mat e = high_outg.clone();
			NEDI(low_ing, high_outg, 1, 0.0, 8);
			cout << "NEDI: " << PSNR2(answerg, high_outg) << " dB" << endl;
			guiAlphaBlend(e, high_outg);
			guiAlphaBlend(high_outg, answerg, "2");
		}
	}
}
*/



template <class Type>
void upsampleNN_(Mat& src, Mat& dest, const int scale)
{
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * scale;
		Type* s = src.ptr<Type>(j);

		for (int i = 0, m = 0; i < src.cols; i++, m += scale)
		{
			const Type ltd = s[i];
			for (int l = 0; l < scale; l++)
			{
				Type* d = dest.ptr<Type>(n + l);
				for (int k = 0; k < scale; k++)
				{
					d[m + k] = ltd;
				}
			}
		}
	}
}

template <>
void upsampleNN_<float>(Mat& src, Mat& dest, const int scale)
{
	//__m256 a = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);		
	//printf("%f %f %f %f %f %f %f %f\n", a.m256_f32[0], a.m256_f32[1], a.m256_f32[2], a.m256_f32[3], a.m256_f32[4], a.m256_f32[5], a.m256_f32[6], a.m256_f32[7]);

	if (scale == 2)
	{
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * scale;
			float* s = src.ptr<float>(j);
			float* d = dest.ptr<float>(n);
			for (int i = 0; i < src.cols; i += 4)
			{
				__m256 ms = _mm256_setr_ps(s[i], s[i], s[i + 1], s[i + 1], s[i + 2], s[i + 2], s[i + 3], s[i + 3]);
				//__m256 ms = _mm256_load_ps(s + i);
				//ms = _mm256_shuffle_ps(ms, ms, _MM_SHUFFLE(2, 2, 0, 0));//scale 2
				_mm256_store_ps(d + 2 * i, ms);
				_mm256_store_ps(d + 2 * i + dest.cols, ms);
			}
		}
	}
	else if (scale == 4)
	{
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * scale;
			float* s = src.ptr<float>(j);
			float* d = dest.ptr<float>(n);
			for (int i = 0; i < src.cols; i += 2)
			{
				__m256 ms = _mm256_setr_ps(s[i], s[i], s[i], s[i], s[i + 1], s[i + 1], s[i + 1], s[i + 1]);
				//ms = _mm256_shuffle_ps(ms, ms, 0);//scale 4
				_mm256_store_ps(d + 4 * i, ms);
				_mm256_store_ps(d + 4 * i + dest.cols, ms);
				_mm256_store_ps(d + 4 * i + 2 * dest.cols, ms);
				_mm256_store_ps(d + 4 * i + 3 * dest.cols, ms);
			}
		}
	}
	else if (scale == 8)
	{
		for (int j = 0; j < src.rows; j++)
		{
			float* s = src.ptr<float>(j);
			float* d = dest.ptr<float>(j * scale);
			for (int i = 0; i < src.cols; i++)
			{
				__m256 ms = _mm256_set1_ps(s[i]);
				_mm256_store_ps(d + 8 * i, ms);
				_mm256_store_ps(d + 8 * i + 1 * dest.cols, ms);
				_mm256_store_ps(d + 8 * i + 2 * dest.cols, ms);
				_mm256_store_ps(d + 8 * i + 3 * dest.cols, ms);
				_mm256_store_ps(d + 8 * i + 4 * dest.cols, ms);
				_mm256_store_ps(d + 8 * i + 5 * dest.cols, ms);
				_mm256_store_ps(d + 8 * i + 6 * dest.cols, ms);
				_mm256_store_ps(d + 8 * i + 7 * dest.cols, ms);
			}
		}
	}
	else
	{
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * scale;
			float* s = src.ptr<float>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += scale)
			{
				const float ltd = s[i];
				for (int l = 0; l < scale; l++)
				{
					float* d = dest.ptr<float>(n + l);
					for (int k = 0; k < scale; k++)
					{
						d[m + k] = ltd;
					}
				}
			}
		}
	}
}

template <>
void upsampleNN_<double>(Mat& src, Mat& dest, const int scale)
{
	//__m256 a = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);		
	//printf("%f %f %f %f %f %f %f %f\n", a.m256_f32[0], a.m256_f32[1], a.m256_f32[2], a.m256_f32[3], a.m256_f32[4], a.m256_f32[5], a.m256_f32[6], a.m256_f32[7]);

	if (scale == 2)
	{
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * scale;
			double* s = src.ptr<double>(j);
			double* d = dest.ptr<double>(n);
			for (int i = 0; i < src.cols; i += 2)
			{
				__m256d ms = _mm256_setr_pd(s[i], s[i], s[i + 1], s[i + 1]);
				//__m256 ms = _mm256_load_ps(s + i);
				//ms = _mm256_shuffle_ps(ms, ms, _MM_SHUFFLE(2, 2, 0, 0));//scale 2
				_mm256_store_pd(d + 2 * i, ms);
				_mm256_store_pd(d + 2 * i + dest.cols, ms);
			}
		}
	}
	else if (scale == 4)
	{
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * scale;
			double* s = src.ptr<double>(j);
			double* d = dest.ptr<double>(n);
			for (int i = 0; i < src.cols; ++i)
			{
				__m256d ms = _mm256_setr_pd(s[i], s[i], s[i], s[i]);
				//ms = _mm256_shuffle_ps(ms, ms, 0);//scale 4
				_mm256_store_pd(d + 4 * i, ms);
				_mm256_store_pd(d + 4 * i + dest.cols, ms);
				_mm256_store_pd(d + 4 * i + 2 * dest.cols, ms);
				_mm256_store_pd(d + 4 * i + 3 * dest.cols, ms);
			}
		}
	}
	else if (scale == 8)
	{
		for (int j = 0; j < src.rows; j++)
		{
			double* s = src.ptr<double>(j);
			double* d = dest.ptr<double>(j * scale);
			for (int i = 0; i < src.cols; ++i)
			{
				__m256d ms = _mm256_set1_pd(s[i]);
				_mm256_store_pd(d + 8 * i, ms);
				_mm256_store_pd(d + 8 * i + 1 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 2 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 3 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 4 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 5 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 6 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 7 * dest.cols, ms);

				_mm256_store_pd(d + 8 * i + 4, ms);
				_mm256_store_pd(d + 8 * i + 4 + 1 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 4 + 2 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 4 + 3 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 4 + 4 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 4 + 5 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 4 + 6 * dest.cols, ms);
				_mm256_store_pd(d + 8 * i + 4 + 7 * dest.cols, ms);
			}
		}
	}
	else
	{
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * scale;
			double* s = src.ptr<double>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += scale)
			{
				const double ltd = s[i];
				for (int l = 0; l < scale; l++)
				{
					double* d = dest.ptr<double>(n + l);
					for (int k = 0; k < scale; k++)
					{
						d[m + k] = ltd;
					}
				}
			}
		}
	}
}

void upsampleNN(InputArray src_, OutputArray dest_, const int scale)
{
	if (scale == 1)
	{
		src_.copyTo(dest_);
		return;
	}

	Mat src = src_.getMat();

	if (dest_.empty() || dest_.size() != Size(src.cols*scale, src.rows*scale))
	{
		dest_.create(Size(src.cols*scale, src.rows*scale), src_.type());
	}
	Mat dest = dest_.getMat();


	if (src.depth() == CV_8U) upsampleNN_<uchar>(src, dest, scale);
	else if (src.depth() == CV_16S) upsampleNN_<short>(src, dest, scale);
	else if (src.depth() == CV_16U) upsampleNN_<ushort>(src, dest, scale);
	else if (src.depth() == CV_32S) upsampleNN_<int>(src, dest, scale);
	else if (src.depth() == CV_32F) upsampleNN_<float>(src, dest, scale);
	else if (src.depth() == CV_64F) upsampleNN_<double>(src, dest, scale);
}

inline int linearinterpolate_(int lt, int rt, int lb, int rb, double a, double b)
{
	return (int)((b*a*lt + b * (1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb) + 0.5);
}

template <class Type>
inline double linearinterpolate_(Type lt, Type rt, Type lb, Type rb, double a, double b)
{
	return (b*a*lt + b * (1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb);
}

template <class Type>
void upsampleLinear_(Mat& src, Mat& dest, const int scale)
{
	Mat sim;
	copyMakeBorder(src, sim, 0, 1, 0, 1, BORDER_REPLICATE);
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * scale;
		Type* s = sim.ptr<Type>(j);

		for (int i = 0, m = 0; i < src.cols; i++, m += scale)
		{
			const Type ltd = s[i];
			const Type rtd = s[i + 1];
			const Type lbd = s[i + sim.cols];
			const Type rbd = s[i + 1 + sim.cols];
			for (int l = 0; l < scale; l++)
			{
				double beta = 1.0 - (double)l / scale;
				Type* d = dest.ptr<Type>(n + l);
				for (int k = 0; k < scale; k++)
				{
					double alpha = 1.0 - (double)k / scale;
					d[m + k] = saturate_cast<Type> (linearinterpolate_<Type>(ltd, rtd, lbd, rbd, alpha, beta));
				}
			}
		}
	}
}

template <class Type>
void upsampleCubic_(Mat& src, Mat& dest, const int scale, const double a)
{
	const int dw = dest.cols / (src.cols - 1);
	const int dh = dest.rows / (src.rows - 1);
	const int hdw = dw >> 1;
	const int hdh = dh >> 1;

	vector<vector<double>> weight(dh*dw);
	for (int i = 0; i < weight.size(); i++)weight[i].resize(16);

	int idx = 0;

	for (int l = 0; l < dh; l++)
	{
		const double y = (double)l / (double)dh;
		for (int k = 0; k < dw; k++)
		{
			const double x = (double)k / (double)dw;

			weight[idx][0] = cp::cubic(1.0 + x, a)*cp::cubic(1.0 + y, a);
			weight[idx][1] = cp::cubic(0.0 + x, a)*cp::cubic(1.0 + y, a);
			weight[idx][2] = cp::cubic(1.0 - x, a)*cp::cubic(1.0 + y, a);
			weight[idx][3] = cp::cubic(2.0 - x, a)*cp::cubic(1.0 + y, a);

			weight[idx][4] = cp::cubic(1.0 + x, a)*cp::cubic(0.0 + y, a);
			weight[idx][5] = cp::cubic(0.0 + x, a)*cp::cubic(0.0 + y, a);
			weight[idx][6] = cp::cubic(1.0 - x, a)*cp::cubic(0.0 + y, a);
			weight[idx][7] = cp::cubic(2.0 - x, a)*cp::cubic(0.0 + y, a);

			weight[idx][8] = cp::cubic(1.0 + x, a)*cp::cubic(1.0 - y, a);
			weight[idx][9] = cp::cubic(0.0 + x, a)*cp::cubic(1.0 - y, a);
			weight[idx][10] = cp::cubic(1.0 - x, a)*cp::cubic(1.0 - y, a);
			weight[idx][11] = cp::cubic(2.0 - x, a)*cp::cubic(1.0 - y, a);

			weight[idx][12] = cp::cubic(1.0 + x, a)*cp::cubic(2.0 - y, a);
			weight[idx][13] = cp::cubic(0.0 + x, a)*cp::cubic(2.0 - y, a);
			weight[idx][14] = cp::cubic(1.0 - x, a)*cp::cubic(2.0 - y, a);
			weight[idx][15] = cp::cubic(2.0 - x, a)*cp::cubic(2.0 - y, a);

			double wsum = 0.0;
			for (int i = 0; i < 16; i++)wsum += weight[idx][i];
			for (int i = 0; i < 16; i++)weight[idx][i] /= wsum;

			idx++;
		}
	}

	Mat sim;
	copyMakeBorder(src, sim, 1, 2, 1, 2, BORDER_REPLICATE);
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * dh;
		Type* s = sim.ptr<Type>(j + 1) + 1;

		for (int i = 0, m = 0; i < src.cols; i++, m += dw)
		{
			const Type v00 = s[i - 1 - sim.cols];
			const Type v01 = s[i - 0 - sim.cols];
			const Type v02 = s[i + 1 - sim.cols];
			const Type v03 = s[i + 2 - sim.cols];
			const Type v10 = s[i - 1];
			const Type v11 = s[i - 0];
			const Type v12 = s[i + 1];
			const Type v13 = s[i + 2];
			const Type v20 = s[i - 1 + sim.cols];
			const Type v21 = s[i - 0 + sim.cols];
			const Type v22 = s[i + 1 + sim.cols];
			const Type v23 = s[i + 2 + sim.cols];
			const Type v30 = s[i - 1 + 2 * sim.cols];
			const Type v31 = s[i - 0 + 2 * sim.cols];
			const Type v32 = s[i + 1 + 2 * sim.cols];
			const Type v33 = s[i + 2 + 2 * sim.cols];

			int idx = 0;
			for (int l = 0; l < dh; l++)
			{
				Type* d = dest.ptr<Type>(n + l);
				for (int k = 0; k < dw; k++)
				{
					d[m + k] = saturate_cast<Type>(
						weight[idx][0] * v00 + weight[idx][1] * v01 + weight[idx][2] * v02 + weight[idx][3] * v03
						+ weight[idx][4] * v10 + weight[idx][5] * v11 + weight[idx][6] * v12 + weight[idx][7] * v13
						+ weight[idx][8] * v20 + weight[idx][9] * v21 + weight[idx][10] * v22 + weight[idx][11] * v23
						+ weight[idx][12] * v30 + weight[idx][13] * v31 + weight[idx][14] * v32 + weight[idx][15] * v33
						);

					idx++;
				}
			}
		}
	}
}

template <uchar>
void upsampleCubic_(Mat& src, Mat& dest, const int scale, const double alpha)
{
	const float a = (double)alpha;
	const int dw = dest.cols / (src.cols - 1);
	const int dh = dest.rows / (src.rows - 1);
	const int hdw = dw >> 1;
	const int hdh = dh >> 1;

	vector<vector<float>> weight(dh*dw);
	for (int i = 0; i < weight.size(); i++)weight[i].resize(16);

	int idx = 0;

	for (int l = 0; l < dh; l++)
	{
		const float y = (float)l / (float)dh;
		for (int k = 0; k < dw; k++)
		{
			const float x = (float)k / (float)dw;

			weight[idx][0] = cp::cubic(1.f + x, a)*cp::cubic(1.f + y, a);
			weight[idx][1] = cp::cubic(0.f + x, a)*cp::cubic(1.f + y, a);
			weight[idx][2] = cp::cubic(1.f - x, a)*cp::cubic(1.f + y, a);
			weight[idx][3] = cp::cubic(2.f - x, a)*cp::cubic(1.f + y, a);

			weight[idx][4] = cp::cubic(1.f + x, a)*cp::cubic(0.f + y, a);
			weight[idx][5] = cp::cubic(0.f + x, a)*cp::cubic(0.f + y, a);
			weight[idx][6] = cp::cubic(1.f - x, a)*cp::cubic(0.f + y, a);
			weight[idx][7] = cp::cubic(2.f - x, a)*cp::cubic(0.f + y, a);

			weight[idx][8] = cp::cubic(1.f + x, a)*cp::cubic(1.f - y, a);
			weight[idx][9] = cp::cubic(0.f + x, a)*cp::cubic(1.f - y, a);
			weight[idx][10] = cp::cubic(1.f - x, a)*cp::cubic(1.f - y, a);
			weight[idx][11] = cp::cubic(2.f - x, a)*cp::cubic(1.f - y, a);

			weight[idx][12] = cp::cubic(1.f + x, a)*cp::cubic(2.f - y, a);
			weight[idx][13] = cp::cubic(0.f + x, a)*cp::cubic(2.f - y, a);
			weight[idx][14] = cp::cubic(1.f - x, a)*cp::cubic(2.f - y, a);
			weight[idx][15] = cp::cubic(2.f - x, a)*cp::cubic(2.f - y, a);

			float wsum = 0.f;
			for (int i = 0; i < 16; i++)wsum += weight[idx][i];
			for (int i = 0; i < 16; i++)weight[idx][i] /= wsum;

			idx++;
		}
	}

	Mat sim;
	copyMakeBorder(src, sim, 1, 2, 1, 2, BORDER_REPLICATE);
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * dh;
		uchar* s = sim.ptr<uchar>(j + 1) + 1;

		for (int i = 0, m = 0; i < src.cols; i++, m += dw)
		{
			const uchar v00 = s[i - 1 - sim.cols];
			const uchar v01 = s[i - 0 - sim.cols];
			const uchar v02 = s[i + 1 - sim.cols];
			const uchar v03 = s[i + 2 - sim.cols];
			const uchar v10 = s[i - 1];
			const uchar v11 = s[i - 0];
			const uchar v12 = s[i + 1];
			const uchar v13 = s[i + 2];
			const uchar v20 = s[i - 1 + sim.cols];
			const uchar v21 = s[i - 0 + sim.cols];
			const uchar v22 = s[i + 1 + sim.cols];
			const uchar v23 = s[i + 2 + sim.cols];
			const uchar v30 = s[i - 1 + 2 * sim.cols];
			const uchar v31 = s[i - 0 + 2 * sim.cols];
			const uchar v32 = s[i + 1 + 2 * sim.cols];
			const uchar v33 = s[i + 2 + 2 * sim.cols];

			int idx = 0;
			for (int l = 0; l < dh; l++)
			{
				uchar* d = dest.ptr<uchar>(n + l);
				for (int k = 0; k < dw; k++)
				{
					d[m + k] = saturate_cast<uchar>(
						weight[idx][0] * v00 + weight[idx][1] * v01 + weight[idx][2] * v02 + weight[idx][3] * v03
						+ weight[idx][4] * v10 + weight[idx][5] * v11 + weight[idx][6] * v12 + weight[idx][7] * v13
						+ weight[idx][8] * v20 + weight[idx][9] * v21 + weight[idx][10] * v22 + weight[idx][11] * v23
						+ weight[idx][12] * v30 + weight[idx][13] * v31 + weight[idx][14] * v32 + weight[idx][15] * v33
						);

					idx++;
				}
			}
		}
	}
}

void upsample32fCubicScale2(Mat& src, Mat& dest, double a)
{
	int amp = 2;

	__m256 weight[4][16];
	int width = src.cols;
	int height = src.rows;

	int idx = 0;

	for (int l = 0; l < amp; l++)
	{
		const double y = (double)l / (double)amp;
		for (int k = 0; k < amp; k++)
		{
			const double x = (double)k / (double)amp;

			weight[idx][0] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][1] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][2] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][3] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][4] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][5] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][6] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][7] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][8] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][9] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][10] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][11] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][12] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(2.0 - y, a)));
			weight[idx][13] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(2.0 - y, a)));
			weight[idx][14] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(2.0 - y, a)));
			weight[idx][15] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(2.0 - y, a)));

			double wsum = 0.0;
			for (int i = 0; i < 16; i++)wsum += *(float*)&weight[idx][i];
			for (int i = 0; i < 16; i++)weight[idx][i] = _mm256_div_ps(weight[idx][i], _mm256_set1_ps(float(wsum)));

			idx++;
		}
	}

	__m256 v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33;
	__m256 tmpA[2], tmpB[2];
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * amp;

		/*float* s2 = src.ptr<float>(max(0, j - 1));
		float* s1 = src.ptr<float>(max(0, j + 0));
		float* s2 = src.ptr<float>(min(height - 1, j + 1));
		float* s3 = src.ptr<float>(min(height - 1, j + 2));*/

		float* s0 = src.ptr<float>(max(0, j - 1));
		float* s1 = src.ptr<float>(max(0, j + 0));
		float* s2 = src.ptr<float>(min(height - 1, j + 1));
		float* s3 = src.ptr<float>(min(height - 1, j + 2));

		int sx1 = 0;
		int sx2 = 1;
		int sx3 = 2;

		v00 = _mm256_set_ps(s0[6], s0[5], s0[4], s0[3], s0[2], s0[1], s0[0], s0[0]);
		v01 = *(__m256*)&s0[sx1];
		v02 = *(__m256*)&s0[sx2];
		v03 = *(__m256*)&s0[sx3];
		v10 = _mm256_set_ps(s1[6], s1[5], s1[4], s1[3], s1[2], s1[1], s1[0], s1[0]);
		v11 = *(__m256*)&s1[sx1];
		v12 = *(__m256*)&s1[sx2];
		v13 = *(__m256*)&s1[sx3];
		v20 = _mm256_set_ps(s2[6], s2[5], s2[4], s2[3], s2[2], s2[1], s2[0], s2[0]);
		v21 = *(__m256*)&s2[sx1];
		v22 = *(__m256*)&s2[sx2];
		v23 = *(__m256*)&s2[sx3];
		v30 = _mm256_set_ps(s3[6], s3[5], s3[4], s3[3], s3[2], s3[1], s3[0], s3[0]);
		v31 = *(__m256*)&s3[sx1];
		v32 = *(__m256*)&s3[sx2];
		v33 = *(__m256*)&s3[sx3];

		for (int sx = 0, dx = 0; sx < src.cols - 10; dx += amp * 8)
		{
			int idx = 0;
			for (int l = 0; l < amp; l++)
			{
				float* d = dest.ptr<float>(n + l);
				{
					for (int k = 0; k < amp; k++)
					{
						tmpA[k] =
							_mm256_fmadd_ps(weight[idx][0], v00,
								_mm256_fmadd_ps(weight[idx][1], v01,
									_mm256_fmadd_ps(weight[idx][2], v02,
										_mm256_fmadd_ps(weight[idx][3], v03,
											_mm256_fmadd_ps(weight[idx][4], v10,
												_mm256_fmadd_ps(weight[idx][5], v11,
													_mm256_fmadd_ps(weight[idx][6], v12,
														_mm256_fmadd_ps(weight[idx][7], v13,
															_mm256_fmadd_ps(weight[idx][8], v20,
																_mm256_fmadd_ps(weight[idx][9], v21,
																	_mm256_fmadd_ps(weight[idx][10], v22,
																		_mm256_fmadd_ps(weight[idx][11], v23,
																			_mm256_fmadd_ps(weight[idx][12], v30,
																				_mm256_fmadd_ps(weight[idx][13], v31,
																					_mm256_fmadd_ps(weight[idx][14], v32,
																						_mm256_mul_ps(weight[idx][15], v33))))))))))))))));
						idx++;
					}

					tmpB[0] = _mm256_unpacklo_ps(tmpA[0], tmpA[1]);
					tmpB[1] = _mm256_unpackhi_ps(tmpA[0], tmpA[1]);
					*(__m256*)&d[dx + 0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
					*(__m256*)&d[dx + 8] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);
				}
			}

			sx += 8;

			int sx0 = sx - 1;
			int sx1 = sx + 0;
			int sx2 = sx + 1;
			int sx3 = sx + 2;

			v00 = *(__m256*)&s0[sx0];
			v01 = *(__m256*)&s0[sx1];
			v02 = *(__m256*)&s0[sx2];
			v03 = *(__m256*)&s0[sx3];
			v10 = *(__m256*)&s1[sx0];
			v11 = *(__m256*)&s1[sx1];
			v12 = *(__m256*)&s1[sx2];
			v13 = *(__m256*)&s1[sx3];
			v20 = *(__m256*)&s2[sx0];
			v21 = *(__m256*)&s2[sx1];
			v22 = *(__m256*)&s2[sx2];
			v23 = *(__m256*)&s2[sx3];
			v30 = *(__m256*)&s3[sx0];
			v31 = *(__m256*)&s3[sx1];
			v32 = *(__m256*)&s3[sx2];
			v33 = *(__m256*)&s3[sx3];
		}
		{
			for (int sx = int(src.cols / 8 - 1) * 8, dx; sx < src.cols; sx += 8)
			{
				dx = amp*sx;

				int sx0 = min(src.cols - 1, sx - 1);
				int sx1 = min(src.cols - 1, sx + 0);
				int sx2 = min(src.cols - 1, sx + 1);
				int sx3 = min(src.cols - 1, sx + 2);
				int sx4 = min(src.cols - 1, sx + 3);
				int sx5 = min(src.cols - 1, sx + 4);
				int sx6 = min(src.cols - 1, sx + 5);
				int sx7 = min(src.cols - 1, sx + 6);
				int sx8 = min(src.cols - 1, sx + 7);
				int sx9 = min(src.cols - 1, sx + 8);
				int sx10 = min(src.cols - 1, sx + 10);

				v00 = _mm256_set_ps(s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3], s0[sx2], s0[sx1], s0[sx0]);
				v01 = _mm256_set_ps(s0[sx8], s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3], s0[sx2], s0[sx1]);
				v02 = _mm256_set_ps(s0[sx9], s0[sx8], s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3], s0[sx2]);
				v03 = _mm256_set_ps(s0[sx10], s0[sx9], s0[sx8], s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3]);

				v10 = _mm256_set_ps(s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3], s1[sx2], s1[sx1], s1[sx0]);
				v11 = _mm256_set_ps(s1[sx8], s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3], s1[sx2], s1[sx1]);
				v12 = _mm256_set_ps(s1[sx9], s1[sx8], s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3], s1[sx2]);
				v13 = _mm256_set_ps(s1[sx10], s1[sx9], s1[sx8], s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3]);

				v20 = _mm256_set_ps(s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3], s2[sx2], s2[sx1], s2[sx0]);
				v21 = _mm256_set_ps(s2[sx8], s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3], s2[sx2], s2[sx1]);
				v22 = _mm256_set_ps(s2[sx9], s2[sx8], s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3], s2[sx2]);
				v23 = _mm256_set_ps(s2[sx10], s2[sx9], s2[sx8], s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3]);

				v30 = _mm256_set_ps(s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3], s3[sx2], s3[sx1], s3[sx0]);
				v31 = _mm256_set_ps(s3[sx8], s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3], s3[sx2], s3[sx1]);
				v32 = _mm256_set_ps(s3[sx9], s3[sx8], s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3], s3[sx2]);
				v33 = _mm256_set_ps(s3[sx10], s3[sx9], s3[sx8], s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3]);

				int idx = 0;
				for (int l = 0; l < amp; l++)
				{
					float* d = dest.ptr<float>(n + l);
					{
						for (int k = 0; k < amp; k++)
						{
							tmpA[k] =
								_mm256_fmadd_ps(weight[idx][0], v00,
									_mm256_fmadd_ps(weight[idx][1], v01,
										_mm256_fmadd_ps(weight[idx][2], v02,
											_mm256_fmadd_ps(weight[idx][3], v03,
												_mm256_fmadd_ps(weight[idx][4], v10,
													_mm256_fmadd_ps(weight[idx][5], v11,
														_mm256_fmadd_ps(weight[idx][6], v12,
															_mm256_fmadd_ps(weight[idx][7], v13,
																_mm256_fmadd_ps(weight[idx][8], v20,
																	_mm256_fmadd_ps(weight[idx][9], v21,
																		_mm256_fmadd_ps(weight[idx][10], v22,
																			_mm256_fmadd_ps(weight[idx][11], v23,
																				_mm256_fmadd_ps(weight[idx][12], v30,
																					_mm256_fmadd_ps(weight[idx][13], v31,
																						_mm256_fmadd_ps(weight[idx][14], v32,
																							_mm256_mul_ps(weight[idx][15], v33))))))))))))))));
							idx++;
						}

						tmpB[0] = _mm256_unpacklo_ps(tmpA[0], tmpA[1]);
						tmpB[1] = _mm256_unpackhi_ps(tmpA[0], tmpA[1]);
						tmpA[0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
						tmpA[1] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);

						int iend = min(16, dest.cols - amp*sx);
						if (iend == 16)
						{
							*(__m256*)&d[dx + 0] = tmpA[0];
							*(__m256*)&d[dx + 8] = tmpA[1];
						}
						else
						{
							int i = 0;
							for (int j = 0; j < amp; ++j)
							{
								while (i < min(8 * j, iend))
								{
									d[dx + i] = ((float*)&tmpA[j])[i - 8 * j];
									++i;
								}
							}
						}
					}
				}
			}
		}
	}
}

void upsample32fCubicScale4(Mat& src, Mat& dest, double a)
{
	int amp = 4;

	__m256 weight[16][16];
	int width = src.cols;
	int height = src.rows;

	int idx = 0;
	for (int l = 0; l < amp; l++)
	{
		const double y = (double)l / (double)amp;
		for (int k = 0; k < amp; k++)
		{
			const double x = (double)k / (double)amp;

			weight[idx][0] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][1] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][2] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][3] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(1.0 + y, a)));
			weight[idx][4] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][5] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][6] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][7] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(0.0 + y, a)));
			weight[idx][8] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][9] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][10] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][11] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(1.0 - y, a)));
			weight[idx][12] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a)*cp::cubic(2.0 - y, a)));
			weight[idx][13] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a)*cp::cubic(2.0 - y, a)));
			weight[idx][14] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a)*cp::cubic(2.0 - y, a)));
			weight[idx][15] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a)*cp::cubic(2.0 - y, a)));

			double wsum = 0.0;
			for (int i = 0; i < 16; i++)wsum += *(float*)&weight[idx][i];
			for (int i = 0; i < 16; i++)weight[idx][i] = _mm256_div_ps(weight[idx][i], _mm256_set1_ps(float(wsum)));

			idx++;
		}
	}

	__m256 v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33;
	__m256 tmpA[4], tmpB[4];
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * amp;

		/*float* s2 = src.ptr<float>(max(0, j - 1));
		float* s1 = src.ptr<float>(max(0, j + 0));
		float* s2 = src.ptr<float>(min(height - 1, j + 1));
		float* s3 = src.ptr<float>(min(height - 1, j + 2));*/

		float* s0 = src.ptr<float>(max(0, j - 1));
		float* s1 = src.ptr<float>(max(0, j + 0));
		float* s2 = src.ptr<float>(min(height - 1, j + 1));
		float* s3 = src.ptr<float>(min(height - 1, j + 2));

		int sx1 = 0;
		int sx2 = 1;
		int sx3 = 2;

		v00 = _mm256_set_ps(s0[6], s0[5], s0[4], s0[3], s0[2], s0[1], s0[0], s0[0]);
		v01 = *(__m256*)&s0[sx1];
		v02 = *(__m256*)&s0[sx2];
		v03 = *(__m256*)&s0[sx3];
		v10 = _mm256_set_ps(s1[6], s1[5], s1[4], s1[3], s1[2], s1[1], s1[0], s1[0]);
		v11 = *(__m256*)&s1[sx1];
		v12 = *(__m256*)&s1[sx2];
		v13 = *(__m256*)&s1[sx3];
		v20 = _mm256_set_ps(s2[6], s2[5], s2[4], s2[3], s2[2], s2[1], s2[0], s2[0]);
		v21 = *(__m256*)&s2[sx1];
		v22 = *(__m256*)&s2[sx2];
		v23 = *(__m256*)&s2[sx3];
		v30 = _mm256_set_ps(s3[6], s3[5], s3[4], s3[3], s3[2], s3[1], s3[0], s3[0]);
		v31 = *(__m256*)&s3[sx1];
		v32 = *(__m256*)&s3[sx2];
		v33 = *(__m256*)&s3[sx3];

		for (int sx = 0, dx = 0; sx < src.cols - 10; dx += amp * 8)
		{
			int idx = 0;
			for (int l = 0; l < amp; l++)
			{
				float* d = dest.ptr<float>(n + l);
				{
					for (int k = 0; k < amp; k++)
					{
						tmpA[k] =
							_mm256_fmadd_ps(weight[idx][0], v00,
								_mm256_fmadd_ps(weight[idx][1], v01,
									_mm256_fmadd_ps(weight[idx][2], v02,
										_mm256_fmadd_ps(weight[idx][3], v03,
											_mm256_fmadd_ps(weight[idx][4], v10,
												_mm256_fmadd_ps(weight[idx][5], v11,
													_mm256_fmadd_ps(weight[idx][6], v12,
														_mm256_fmadd_ps(weight[idx][7], v13,
															_mm256_fmadd_ps(weight[idx][8], v20,
																_mm256_fmadd_ps(weight[idx][9], v21,
																	_mm256_fmadd_ps(weight[idx][10], v22,
																		_mm256_fmadd_ps(weight[idx][11], v23,
																			_mm256_fmadd_ps(weight[idx][12], v30,
																				_mm256_fmadd_ps(weight[idx][13], v31,
																					_mm256_fmadd_ps(weight[idx][14], v32,
																						_mm256_mul_ps(weight[idx][15], v33))))))))))))))));
						idx++;
					}

					tmpB[0] = _mm256_unpacklo_ps(tmpA[0], tmpA[2]);
					tmpB[1] = _mm256_unpackhi_ps(tmpA[0], tmpA[2]);
					tmpB[2] = _mm256_unpacklo_ps(tmpA[1], tmpA[3]);
					tmpB[3] = _mm256_unpackhi_ps(tmpA[1], tmpA[3]);

					tmpA[0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
					tmpA[1] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);
					tmpA[2] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00100000);
					tmpA[3] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00110001);


					tmpB[0] = _mm256_unpacklo_ps(tmpA[0], tmpA[2]);
					tmpB[1] = _mm256_unpackhi_ps(tmpA[0], tmpA[2]);
					tmpB[2] = _mm256_unpacklo_ps(tmpA[1], tmpA[3]);
					tmpB[3] = _mm256_unpackhi_ps(tmpA[1], tmpA[3]);

					*(__m256*)&d[dx + 0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
					*(__m256*)&d[dx + 8] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);
					*(__m256*)&d[dx + 16] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00100000);
					*(__m256*)&d[dx + 24] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00110001);
				}
			}

			sx += 8;

			int sx0 = sx - 1;
			int sx1 = sx + 0;
			int sx2 = sx + 1;
			int sx3 = sx + 2;

			v00 = *(__m256*)&s0[sx0];
			v01 = *(__m256*)&s0[sx1];
			v02 = *(__m256*)&s0[sx2];
			v03 = *(__m256*)&s0[sx3];
			v10 = *(__m256*)&s1[sx0];
			v11 = *(__m256*)&s1[sx1];
			v12 = *(__m256*)&s1[sx2];
			v13 = *(__m256*)&s1[sx3];
			v20 = *(__m256*)&s2[sx0];
			v21 = *(__m256*)&s2[sx1];
			v22 = *(__m256*)&s2[sx2];
			v23 = *(__m256*)&s2[sx3];
			v30 = *(__m256*)&s3[sx0];
			v31 = *(__m256*)&s3[sx1];
			v32 = *(__m256*)&s3[sx2];
			v33 = *(__m256*)&s3[sx3];
		}
		{
			for (int sx = int(src.cols / 8 - 1) * 8, dx; sx < src.cols; sx += 8)
			{
				dx = amp*sx;

				int sx0 = min(src.cols - 1, sx - 1);
				int sx1 = min(src.cols - 1, sx + 0);
				int sx2 = min(src.cols - 1, sx + 1);
				int sx3 = min(src.cols - 1, sx + 2);
				int sx4 = min(src.cols - 1, sx + 3);
				int sx5 = min(src.cols - 1, sx + 4);
				int sx6 = min(src.cols - 1, sx + 5);
				int sx7 = min(src.cols - 1, sx + 6);
				int sx8 = min(src.cols - 1, sx + 7);
				int sx9 = min(src.cols - 1, sx + 8);
				int sx10 = min(src.cols - 1, sx + 10);

				v00 = _mm256_set_ps(s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3], s0[sx2], s0[sx1], s0[sx0]);
				v01 = _mm256_set_ps(s0[sx8], s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3], s0[sx2], s0[sx1]);
				v02 = _mm256_set_ps(s0[sx9], s0[sx8], s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3], s0[sx2]);
				v03 = _mm256_set_ps(s0[sx10], s0[sx9], s0[sx8], s0[sx7], s0[sx6], s0[sx5], s0[sx4], s0[sx3]);

				v10 = _mm256_set_ps(s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3], s1[sx2], s1[sx1], s1[sx0]);
				v11 = _mm256_set_ps(s1[sx8], s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3], s1[sx2], s1[sx1]);
				v12 = _mm256_set_ps(s1[sx9], s1[sx8], s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3], s1[sx2]);
				v13 = _mm256_set_ps(s1[sx10], s1[sx9], s1[sx8], s1[sx7], s1[sx6], s1[sx5], s1[sx4], s1[sx3]);

				v20 = _mm256_set_ps(s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3], s2[sx2], s2[sx1], s2[sx0]);
				v21 = _mm256_set_ps(s2[sx8], s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3], s2[sx2], s2[sx1]);
				v22 = _mm256_set_ps(s2[sx9], s2[sx8], s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3], s2[sx2]);
				v23 = _mm256_set_ps(s2[sx10], s2[sx9], s2[sx8], s2[sx7], s2[sx6], s2[sx5], s2[sx4], s2[sx3]);

				v30 = _mm256_set_ps(s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3], s3[sx2], s3[sx1], s3[sx0]);
				v31 = _mm256_set_ps(s3[sx8], s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3], s3[sx2], s3[sx1]);
				v32 = _mm256_set_ps(s3[sx9], s3[sx8], s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3], s3[sx2]);
				v33 = _mm256_set_ps(s3[sx10], s3[sx9], s3[sx8], s3[sx7], s3[sx6], s3[sx5], s3[sx4], s3[sx3]);

				int idx = 0;
				for (int l = 0; l < amp; l++)
				{
					float* d = dest.ptr<float>(n + l);
					{
						for (int k = 0; k < amp; k++)
						{
							tmpA[k] =
								_mm256_fmadd_ps(weight[idx][0], v00,
									_mm256_fmadd_ps(weight[idx][1], v01,
										_mm256_fmadd_ps(weight[idx][2], v02,
											_mm256_fmadd_ps(weight[idx][3], v03,
												_mm256_fmadd_ps(weight[idx][4], v10,
													_mm256_fmadd_ps(weight[idx][5], v11,
														_mm256_fmadd_ps(weight[idx][6], v12,
															_mm256_fmadd_ps(weight[idx][7], v13,
																_mm256_fmadd_ps(weight[idx][8], v20,
																	_mm256_fmadd_ps(weight[idx][9], v21,
																		_mm256_fmadd_ps(weight[idx][10], v22,
																			_mm256_fmadd_ps(weight[idx][11], v23,
																				_mm256_fmadd_ps(weight[idx][12], v30,
																					_mm256_fmadd_ps(weight[idx][13], v31,
																						_mm256_fmadd_ps(weight[idx][14], v32,
																							_mm256_mul_ps(weight[idx][15], v33))))))))))))))));
							idx++;
						}

						tmpB[0] = _mm256_unpacklo_ps(tmpA[0], tmpA[2]);
						tmpB[1] = _mm256_unpackhi_ps(tmpA[0], tmpA[2]);
						tmpB[2] = _mm256_unpacklo_ps(tmpA[1], tmpA[3]);
						tmpB[3] = _mm256_unpackhi_ps(tmpA[1], tmpA[3]);

						tmpA[0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
						tmpA[1] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);
						tmpA[2] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00100000);
						tmpA[3] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00110001);

						tmpB[0] = _mm256_unpacklo_ps(tmpA[0], tmpA[2]);
						tmpB[1] = _mm256_unpackhi_ps(tmpA[0], tmpA[2]);
						tmpB[2] = _mm256_unpacklo_ps(tmpA[1], tmpA[3]);
						tmpB[3] = _mm256_unpackhi_ps(tmpA[1], tmpA[3]);

						tmpA[0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
						tmpA[1] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);
						tmpA[2] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00100000);
						tmpA[3] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00110001);

						int iend = min(32, dest.cols - amp*sx);
						if (iend == 32)
						{
							*(__m256*)&d[dx + 0] = tmpA[0];
							*(__m256*)&d[dx + 8] = tmpA[1];
							*(__m256*)&d[dx + 16] = tmpA[2];
							*(__m256*)&d[dx + 24] = tmpA[3];
						}
						else
						{
							int i = 0;
							for (int j = 0; j < amp; ++j)
							{
								while (i < min(8 * j, iend))
								{
									d[dx + i] = ((float*)&tmpA[j])[i - 8 * j];
									++i;
								}
							}
						}
					}
				}
			}
		}
	}
}

void upsample64fCubicScale2(Mat& src, Mat& dest, double a)
{
	int amp = 2;

	__m256d weight[4][16];
	int width = src.cols;
	int height = src.rows;

	int idx = 0;

	for (int l = 0; l < amp; l++)
	{
		const double y = (double)l / (double)amp;
		for (int k = 0; k < amp; k++)
		{
			const double x = (double)k / (double)amp;

			weight[idx][0] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(1.0 + y, a));
			weight[idx][1] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(1.0 + y, a));
			weight[idx][2] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(1.0 + y, a));
			weight[idx][3] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(1.0 + y, a));
			weight[idx][4] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(0.0 + y, a));
			weight[idx][5] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(0.0 + y, a));
			weight[idx][6] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(0.0 + y, a));
			weight[idx][7] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(0.0 + y, a));
			weight[idx][8] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(1.0 - y, a));
			weight[idx][9] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(1.0 - y, a));
			weight[idx][10] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(1.0 - y, a));
			weight[idx][11] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(1.0 - y, a));
			weight[idx][12] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(2.0 - y, a));
			weight[idx][13] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(2.0 - y, a));
			weight[idx][14] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(2.0 - y, a));
			weight[idx][15] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(2.0 - y, a));

			double wsum = 0.0;
			for (int i = 0; i < 16; i++)wsum += *(double*)&weight[idx][i];
			for (int i = 0; i < 16; i++)weight[idx][i] = _mm256_div_pd(weight[idx][i], _mm256_set1_pd(wsum));

			idx++;
		}
	}

	__m256d v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33;
	__m256d tmpA[2], tmpB[2];
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * amp;

		/*double* s2 = src.ptr<double>(max(0, j - 1));
		double* s1 = src.ptr<double>(max(0, j + 0));
		double* s2 = src.ptr<double>(min(height - 1, j + 1));
		double* s3 = src.ptr<double>(min(height - 1, j + 2));*/

		double* s0 = src.ptr<double>(max(0, j - 1));
		double* s1 = src.ptr<double>(max(0, j + 0));
		double* s2 = src.ptr<double>(min(height - 1, j + 1));
		double* s3 = src.ptr<double>(min(height - 1, j + 2));

		int sx1 = 0;
		int sx2 = 1;
		int sx3 = 2;

		v00 = _mm256_set_pd(s0[2], s0[1], s0[0], s0[0]);
		v01 = *(__m256d*)&s0[sx1];
		v02 = *(__m256d*)&s0[sx2];
		v03 = *(__m256d*)&s0[sx3];
		v10 = _mm256_set_pd(s1[2], s1[1], s1[0], s1[0]);
		v11 = *(__m256d*)&s1[sx1];
		v12 = *(__m256d*)&s1[sx2];
		v13 = *(__m256d*)&s1[sx3];
		v20 = _mm256_set_pd(s2[2], s2[1], s2[0], s2[0]);
		v21 = *(__m256d*)&s2[sx1];
		v22 = *(__m256d*)&s2[sx2];
		v23 = *(__m256d*)&s2[sx3];
		v30 = _mm256_set_pd(s3[2], s3[1], s3[0], s3[0]);
		v31 = *(__m256d*)&s3[sx1];
		v32 = *(__m256d*)&s3[sx2];
		v33 = *(__m256d*)&s3[sx3];

		for (int sx = 0, dx = 0; sx < src.cols - 6; dx += amp * 4)
		{
			int idx = 0;
			for (int l = 0; l < amp; l++)
			{
				double* d = dest.ptr<double>(n + l);
				{
					for (int k = 0; k < amp; k++)
					{
						tmpA[k] =
							_mm256_fmadd_pd(weight[idx][0], v00,
								_mm256_fmadd_pd(weight[idx][1], v01,
									_mm256_fmadd_pd(weight[idx][2], v02,
										_mm256_fmadd_pd(weight[idx][3], v03,
											_mm256_fmadd_pd(weight[idx][4], v10,
												_mm256_fmadd_pd(weight[idx][5], v11,
													_mm256_fmadd_pd(weight[idx][6], v12,
														_mm256_fmadd_pd(weight[idx][7], v13,
															_mm256_fmadd_pd(weight[idx][8], v20,
																_mm256_fmadd_pd(weight[idx][9], v21,
																	_mm256_fmadd_pd(weight[idx][10], v22,
																		_mm256_fmadd_pd(weight[idx][11], v23,
																			_mm256_fmadd_pd(weight[idx][12], v30,
																				_mm256_fmadd_pd(weight[idx][13], v31,
																					_mm256_fmadd_pd(weight[idx][14], v32,
																						_mm256_mul_pd(weight[idx][15], v33))))))))))))))));
						idx++;
					}

					tmpB[0] = _mm256_unpacklo_pd(tmpA[0], tmpA[1]);
					tmpB[1] = _mm256_unpackhi_pd(tmpA[0], tmpA[1]);
					*(__m256d*)&d[dx + 0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
					*(__m256d*)&d[dx + 4] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);
				}
			}

			sx += 4;

			int sx0 = sx - 1;
			int sx1 = sx + 0;
			int sx2 = sx + 1;
			int sx3 = sx + 2;

			v00 = *(__m256d*)&s0[sx0];
			v01 = *(__m256d*)&s0[sx1];
			v02 = *(__m256d*)&s0[sx2];
			v03 = *(__m256d*)&s0[sx3];
			v10 = *(__m256d*)&s1[sx0];
			v11 = *(__m256d*)&s1[sx1];
			v12 = *(__m256d*)&s1[sx2];
			v13 = *(__m256d*)&s1[sx3];
			v20 = *(__m256d*)&s2[sx0];
			v21 = *(__m256d*)&s2[sx1];
			v22 = *(__m256d*)&s2[sx2];
			v23 = *(__m256d*)&s2[sx3];
			v30 = *(__m256d*)&s3[sx0];
			v31 = *(__m256d*)&s3[sx1];
			v32 = *(__m256d*)&s3[sx2];
			v33 = *(__m256d*)&s3[sx3];
		}
		{
			for (int sx = int(src.cols / 4 - 1) * 4, dx; sx < src.cols; sx += 4)
			{
				dx = amp*sx;

				int sx0 = min(src.cols - 1, sx - 1);
				int sx1 = min(src.cols - 1, sx + 0);
				int sx2 = min(src.cols - 1, sx + 1);
				int sx3 = min(src.cols - 1, sx + 2);
				int sx4 = min(src.cols - 1, sx + 3);
				int sx5 = min(src.cols - 1, sx + 4);
				int sx6 = min(src.cols - 1, sx + 5);

				v00 = _mm256_set_pd(s0[sx3], s0[sx2], s0[sx1], s0[sx0]);
				v01 = _mm256_set_pd(s0[sx4], s0[sx3], s0[sx2], s0[sx1]);
				v02 = _mm256_set_pd(s0[sx5], s0[sx4], s0[sx3], s0[sx2]);
				v03 = _mm256_set_pd(s0[sx6], s0[sx5], s0[sx4], s0[sx3]);

				v10 = _mm256_set_pd(s1[sx3], s1[sx2], s1[sx1], s1[sx0]);
				v11 = _mm256_set_pd(s1[sx4], s1[sx3], s1[sx2], s1[sx1]);
				v12 = _mm256_set_pd(s1[sx5], s1[sx4], s1[sx3], s1[sx2]);
				v13 = _mm256_set_pd(s1[sx6], s1[sx5], s1[sx4], s1[sx3]);

				v20 = _mm256_set_pd(s2[sx3], s2[sx2], s2[sx1], s2[sx0]);
				v21 = _mm256_set_pd(s2[sx4], s2[sx3], s2[sx2], s2[sx1]);
				v22 = _mm256_set_pd(s2[sx5], s2[sx4], s2[sx3], s2[sx2]);
				v23 = _mm256_set_pd(s2[sx6], s2[sx5], s2[sx4], s2[sx3]);

				v30 = _mm256_set_pd(s3[sx3], s3[sx2], s3[sx1], s3[sx0]);
				v31 = _mm256_set_pd(s3[sx4], s3[sx3], s3[sx2], s3[sx1]);
				v32 = _mm256_set_pd(s3[sx5], s3[sx4], s3[sx3], s3[sx2]);
				v33 = _mm256_set_pd(s3[sx6], s3[sx5], s3[sx4], s3[sx3]);

				int idx = 0;
				for (int l = 0; l < amp; l++)
				{
					double* d = dest.ptr<double>(n + l);
					{
						for (int k = 0; k < amp; k++)
						{
							tmpA[k] =
								_mm256_fmadd_pd(weight[idx][0], v00,
									_mm256_fmadd_pd(weight[idx][1], v01,
										_mm256_fmadd_pd(weight[idx][2], v02,
											_mm256_fmadd_pd(weight[idx][3], v03,
												_mm256_fmadd_pd(weight[idx][4], v10,
													_mm256_fmadd_pd(weight[idx][5], v11,
														_mm256_fmadd_pd(weight[idx][6], v12,
															_mm256_fmadd_pd(weight[idx][7], v13,
																_mm256_fmadd_pd(weight[idx][8], v20,
																	_mm256_fmadd_pd(weight[idx][9], v21,
																		_mm256_fmadd_pd(weight[idx][10], v22,
																			_mm256_fmadd_pd(weight[idx][11], v23,
																				_mm256_fmadd_pd(weight[idx][12], v30,
																					_mm256_fmadd_pd(weight[idx][13], v31,
																						_mm256_fmadd_pd(weight[idx][14], v32,
																							_mm256_mul_pd(weight[idx][15], v33))))))))))))))));
							idx++;
						}

						tmpB[0] = _mm256_unpacklo_pd(tmpA[0], tmpA[1]);
						tmpB[1] = _mm256_unpackhi_pd(tmpA[0], tmpA[1]);
						tmpA[0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
						tmpA[1] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);

						int iend = min(8, dest.cols - amp*sx);
						if (iend == 8)
						{
							*(__m256d*)&d[dx + 0] = tmpA[0];
							*(__m256d*)&d[dx + 4] = tmpA[1];
						}
						else
						{
							int i = 0;
							for (int j = 0; j < amp; ++j)
							{
								while (i < min(4 * j, iend))
								{
									d[dx + i] = ((double*)&tmpA[j])[i - 4 * j];
									++i;
								}
							}
						}
					}
				}
			}
		}
	}
}

void upsample64fCubicScale4(Mat& src, Mat& dest, double a)
{
	int amp = 4;

	__m256d weight[16][16];
	int width = src.cols;
	int height = src.rows;

	int idx = 0;
	for (int l = 0; l < amp; l++)
	{
		const double y = (double)l / (double)amp;
		for (int k = 0; k < amp; k++)
		{
			const double x = (double)k / (double)amp;

			weight[idx][0] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(1.0 + y, a));
			weight[idx][1] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(1.0 + y, a));
			weight[idx][2] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(1.0 + y, a));
			weight[idx][3] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(1.0 + y, a));
			weight[idx][4] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(0.0 + y, a));
			weight[idx][5] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(0.0 + y, a));
			weight[idx][6] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(0.0 + y, a));
			weight[idx][7] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(0.0 + y, a));
			weight[idx][8] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(1.0 - y, a));
			weight[idx][9] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(1.0 - y, a));
			weight[idx][10] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(1.0 - y, a));
			weight[idx][11] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(1.0 - y, a));
			weight[idx][12] = _mm256_set1_pd(cp::cubic(1.0 + x, a)*cp::cubic(2.0 - y, a));
			weight[idx][13] = _mm256_set1_pd(cp::cubic(0.0 + x, a)*cp::cubic(2.0 - y, a));
			weight[idx][14] = _mm256_set1_pd(cp::cubic(1.0 - x, a)*cp::cubic(2.0 - y, a));
			weight[idx][15] = _mm256_set1_pd(cp::cubic(2.0 - x, a)*cp::cubic(2.0 - y, a));

			double wsum = 0.0;
			for (int i = 0; i < 16; i++)wsum += *(double*)&weight[idx][i];
			for (int i = 0; i < 16; i++)weight[idx][i] = _mm256_div_pd(weight[idx][i], _mm256_set1_pd(wsum));

			idx++;
		}
	}

	__m256d v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33;
	__m256d tmpA[4], tmpB[4];
	for (int j = 0; j < src.rows; j++)
	{
		int n = j * amp;

		double* s0 = src.ptr<double>(max(0, j - 1));
		double* s1 = src.ptr<double>(max(0, j + 0));
		double* s2 = src.ptr<double>(min(height - 1, j + 1));
		double* s3 = src.ptr<double>(min(height - 1, j + 2));

		int sx1 = 0;
		int sx2 = 1;
		int sx3 = 2;

		v00 = _mm256_set_pd(s0[2], s0[1], s0[0], s0[0]);
		v01 = *(__m256d*)&s0[sx1];
		v02 = *(__m256d*)&s0[sx2];
		v03 = *(__m256d*)&s0[sx3];
		v10 = _mm256_set_pd(s1[2], s1[1], s1[0], s1[0]);
		v11 = *(__m256d*)&s1[sx1];
		v12 = *(__m256d*)&s1[sx2];
		v13 = *(__m256d*)&s1[sx3];
		v20 = _mm256_set_pd(s2[2], s2[1], s2[0], s2[0]);
		v21 = *(__m256d*)&s2[sx1];
		v22 = *(__m256d*)&s2[sx2];
		v23 = *(__m256d*)&s2[sx3];
		v30 = _mm256_set_pd(s3[2], s3[1], s3[0], s3[0]);
		v31 = *(__m256d*)&s3[sx1];
		v32 = *(__m256d*)&s3[sx2];
		v33 = *(__m256d*)&s3[sx3];

		for (int sx = 0, dx = 0; sx < src.cols - 6; dx += amp * 4)
		{
			int idx = 0;
			for (int l = 0; l < amp; l++)
			{
				double* d = dest.ptr<double>(n + l);
				{
					for (int k = 0; k < amp; k++)
					{
						tmpA[k] =
							_mm256_fmadd_pd(weight[idx][0], v00,
								_mm256_fmadd_pd(weight[idx][1], v01,
									_mm256_fmadd_pd(weight[idx][2], v02,
										_mm256_fmadd_pd(weight[idx][3], v03,
											_mm256_fmadd_pd(weight[idx][4], v10,
												_mm256_fmadd_pd(weight[idx][5], v11,
													_mm256_fmadd_pd(weight[idx][6], v12,
														_mm256_fmadd_pd(weight[idx][7], v13,
															_mm256_fmadd_pd(weight[idx][8], v20,
																_mm256_fmadd_pd(weight[idx][9], v21,
																	_mm256_fmadd_pd(weight[idx][10], v22,
																		_mm256_fmadd_pd(weight[idx][11], v23,
																			_mm256_fmadd_pd(weight[idx][12], v30,
																				_mm256_fmadd_pd(weight[idx][13], v31,
																					_mm256_fmadd_pd(weight[idx][14], v32,
																						_mm256_mul_pd(weight[idx][15], v33))))))))))))))));
						idx++;
					}

					tmpB[0] = _mm256_unpacklo_pd(tmpA[0], tmpA[2]);
					tmpB[1] = _mm256_unpackhi_pd(tmpA[0], tmpA[2]);
					tmpB[2] = _mm256_unpacklo_pd(tmpA[1], tmpA[3]);
					tmpB[3] = _mm256_unpackhi_pd(tmpA[1], tmpA[3]);

					tmpA[0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
					tmpA[1] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);
					tmpA[2] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00100000);
					tmpA[3] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00110001);


					tmpB[0] = _mm256_unpacklo_pd(tmpA[0], tmpA[2]);
					tmpB[1] = _mm256_unpackhi_pd(tmpA[0], tmpA[2]);
					tmpB[2] = _mm256_unpacklo_pd(tmpA[1], tmpA[3]);
					tmpB[3] = _mm256_unpackhi_pd(tmpA[1], tmpA[3]);

					*(__m256d*)&d[dx + 0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
					*(__m256d*)&d[dx + 4] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);
					*(__m256d*)&d[dx + 8] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00100000);
					*(__m256d*)&d[dx + 12] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00110001);
				}
			}

			sx += 4;

			int sx0 = sx - 1;
			int sx1 = sx + 0;
			int sx2 = sx + 1;
			int sx3 = sx + 2;

			v00 = *(__m256d*)&s0[sx0];
			v01 = *(__m256d*)&s0[sx1];
			v02 = *(__m256d*)&s0[sx2];
			v03 = *(__m256d*)&s0[sx3];
			v10 = *(__m256d*)&s1[sx0];
			v11 = *(__m256d*)&s1[sx1];
			v12 = *(__m256d*)&s1[sx2];
			v13 = *(__m256d*)&s1[sx3];
			v20 = *(__m256d*)&s2[sx0];
			v21 = *(__m256d*)&s2[sx1];
			v22 = *(__m256d*)&s2[sx2];
			v23 = *(__m256d*)&s2[sx3];
			v30 = *(__m256d*)&s3[sx0];
			v31 = *(__m256d*)&s3[sx1];
			v32 = *(__m256d*)&s3[sx2];
			v33 = *(__m256d*)&s3[sx3];
		}
		{
			for (int sx = int(src.cols / 4 - 1) * 4, dx; sx < src.cols; sx += 4)
			{
				dx = amp*sx;

				int sx0 = min(src.cols - 1, sx - 1);
				int sx1 = min(src.cols - 1, sx + 0);
				int sx2 = min(src.cols - 1, sx + 1);
				int sx3 = min(src.cols - 1, sx + 2);
				int sx4 = min(src.cols - 1, sx + 3);
				int sx5 = min(src.cols - 1, sx + 4);
				int sx6 = min(src.cols - 1, sx + 5);

				v00 = _mm256_set_pd(s0[sx3], s0[sx2], s0[sx1], s0[sx0]);
				v01 = _mm256_set_pd(s0[sx4], s0[sx3], s0[sx2], s0[sx1]);
				v02 = _mm256_set_pd(s0[sx5], s0[sx4], s0[sx3], s0[sx2]);
				v03 = _mm256_set_pd(s0[sx6], s0[sx5], s0[sx4], s0[sx3]);

				v10 = _mm256_set_pd(s1[sx3], s1[sx2], s1[sx1], s1[sx0]);
				v11 = _mm256_set_pd(s1[sx4], s1[sx3], s1[sx2], s1[sx1]);
				v12 = _mm256_set_pd(s1[sx5], s1[sx4], s1[sx3], s1[sx2]);
				v13 = _mm256_set_pd(s1[sx6], s1[sx5], s1[sx4], s1[sx3]);

				v20 = _mm256_set_pd(s2[sx3], s2[sx2], s2[sx1], s2[sx0]);
				v21 = _mm256_set_pd(s2[sx4], s2[sx3], s2[sx2], s2[sx1]);
				v22 = _mm256_set_pd(s2[sx5], s2[sx4], s2[sx3], s2[sx2]);
				v23 = _mm256_set_pd(s2[sx6], s2[sx5], s2[sx4], s2[sx3]);

				v30 = _mm256_set_pd(s3[sx3], s3[sx2], s3[sx1], s3[sx0]);
				v31 = _mm256_set_pd(s3[sx4], s3[sx3], s3[sx2], s3[sx1]);
				v32 = _mm256_set_pd(s3[sx5], s3[sx4], s3[sx3], s3[sx2]);
				v33 = _mm256_set_pd(s3[sx6], s3[sx5], s3[sx4], s3[sx3]);

				int idx = 0;
				for (int l = 0; l < amp; l++)
				{
					double *d = dest.ptr<double>(n + l);
					{
						for (int k = 0; k < amp; k++)
						{
							tmpA[k] =
								_mm256_fmadd_pd(weight[idx][0], v00,
									_mm256_fmadd_pd(weight[idx][1], v01,
										_mm256_fmadd_pd(weight[idx][2], v02,
											_mm256_fmadd_pd(weight[idx][3], v03,
												_mm256_fmadd_pd(weight[idx][4], v10,
													_mm256_fmadd_pd(weight[idx][5], v11,
														_mm256_fmadd_pd(weight[idx][6], v12,
															_mm256_fmadd_pd(weight[idx][7], v13,
																_mm256_fmadd_pd(weight[idx][8], v20,
																	_mm256_fmadd_pd(weight[idx][9], v21,
																		_mm256_fmadd_pd(weight[idx][10], v22,
																			_mm256_fmadd_pd(weight[idx][11], v23,
																				_mm256_fmadd_pd(weight[idx][12], v30,
																					_mm256_fmadd_pd(weight[idx][13], v31,
																						_mm256_fmadd_pd(weight[idx][14], v32,
																							_mm256_mul_pd(weight[idx][15], v33))))))))))))))));
							idx++;
						}

						tmpB[0] = _mm256_unpacklo_pd(tmpA[0], tmpA[2]);
						tmpB[1] = _mm256_unpackhi_pd(tmpA[0], tmpA[2]);
						tmpB[2] = _mm256_unpacklo_pd(tmpA[1], tmpA[3]);
						tmpB[3] = _mm256_unpackhi_pd(tmpA[1], tmpA[3]);

						tmpA[0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
						tmpA[1] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);
						tmpA[2] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00100000);
						tmpA[3] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00110001);

						tmpB[0] = _mm256_unpacklo_pd(tmpA[0], tmpA[2]);
						tmpB[1] = _mm256_unpackhi_pd(tmpA[0], tmpA[2]);
						tmpB[2] = _mm256_unpacklo_pd(tmpA[1], tmpA[3]);
						tmpB[3] = _mm256_unpackhi_pd(tmpA[1], tmpA[3]);

						tmpA[0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
						tmpA[1] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);
						tmpA[2] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00100000);
						tmpA[3] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00110001);

						int iend = min(16, dest.cols - amp*sx);
						if (iend == 16)
						{
							*(__m256d*)&d[dx + 0] = tmpA[0];
							*(__m256d*)&d[dx + 4] = tmpA[1];
							*(__m256d*)&d[dx + 8] = tmpA[2];
							*(__m256d*)&d[dx + 12] = tmpA[3];
						}
						else
						{
							int i = 0;
							for (int j = 0; j < amp; ++j)
							{
								while (i < min(4 * j, iend))
								{
									d[dx + i] = ((double*)&tmpA[j])[i - 4 * j];
									++i;
								}
							}
						}
					}
				}
			}
		}
	}
}

void upsampleCubicGray(InputArray src_, OutputArray dest_, const int scale, const double a)
{
	if (scale == 1)
	{
		src_.copyTo(dest_);
		return;
	}

	Mat src = src_.getMat();

	if (dest_.empty() || dest_.size() != Size(src.cols*scale, src.rows*scale))
	{
		dest_.create(Size(src.cols*scale, src.rows*scale), src_.type());
	}

	Mat dest = dest_.getMat();

	if (src.depth() == CV_8U) upsampleCubic_<uchar>(src, dest, scale, a);
	else if (src.depth() == CV_16S) upsampleCubic_<short>(src, dest, scale, a);
	else if (src.depth() == CV_16U) upsampleCubic_<ushort>(src, dest, scale, a);
	else if (src.depth() == CV_32S) upsampleCubic_<int>(src, dest, scale, a);
	else if (src.depth() == CV_32F)
	{
		/*if (scale == 2)
			upsample32fCubicScale2(src, dest, a);
		else if (scale == 4)
			upsample32fCubicScale4(src, dest, a);
		else*/
		upsampleCubic_<float>(src, dest, scale, a);
	}
	else if (src.depth() == CV_64F)
	{
		if (scale == 2)
			upsample64fCubicScale2(src, dest, a);
		else if (scale == 4)
			upsample64fCubicScale4(src, dest, a);
		else
			upsampleCubic_<double>(src, dest, scale, a);
	}
}

void upsampleCubic(InputArray src_, OutputArray dest_, const int scale, const double a)
{
	if (src_.channels() == 1)
		upsampleCubicGray(src_, dest_, scale, a);
	else
	{
		vector<Mat> v;
		vector<Mat> d(3);
		split(src_, v);
		upsampleCubicGray(v[0], d[0], scale, a);
		upsampleCubicGray(v[1], d[1], scale, a);
		upsampleCubicGray(v[2], d[2], scale, a);
		merge(d, dest_);
	}
}

#define UPSAMPLE_USE_SIMD

class UpsampleConv4x4_8U_ParallelBody : public cv::ParallelLoopBody
{
private:
	const cv::Mat* src;
	const cv::Mat* weightmap;
	cv::Mat* dest;

	int scale;
public:
	UpsampleConv4x4_8U_ParallelBody(const cv::Mat& src, const cv::Mat& weightmap, cv::Mat& dst, const int scale)
		: src(&src), weightmap(&weightmap), dest(&dst), scale(scale)
	{
	}

	void operator() (const cv::Range& range) const
	{
		uchar CV_DECL_ALIGNED(AVX_ALIGN) neighbor_b[32];
		uchar CV_DECL_ALIGNED(AVX_ALIGN) neighbor_g[32];
		uchar CV_DECL_ALIGNED(AVX_ALIGN) neighbor_r[32];

		for (int y = range.start; y < range.end; y += scale)
		{
			const int y_ = (int)(y / scale);
			const int y0 = max(0, y_ - 1);
			const int y1 = y_;
			const int y2 = min(y_ + 1, src->rows - 1);
			const int y3 = min(y_ + 2, src->rows - 1);

			for (int x = 0; x < dest->cols; x += scale)
			{
				const int x_ = x / scale;
				const int x0 = max(0, x_ - 1) * 3;
				const int x1 = (x_) * 3;
				const int x2 = min(x_ + 1, src->cols - 1) * 3;
				const int x3 = min(x_ + 2, src->cols - 1) * 3;

				const uchar* s0 = src->ptr<uchar>(y0);
				neighbor_b[0] = s0[x0 + 0];
				neighbor_g[0] = s0[x0 + 1];
				neighbor_r[0] = s0[x0 + 2];
				neighbor_b[1] = s0[x1 + 0];
				neighbor_g[1] = s0[x1 + 1];
				neighbor_r[1] = s0[x1 + 2];
				neighbor_b[2] = s0[x2 + 0];
				neighbor_g[2] = s0[x2 + 1];
				neighbor_r[2] = s0[x2 + 2];
				neighbor_b[3] = s0[x3 + 0];
				neighbor_g[3] = s0[x3 + 1];
				neighbor_r[3] = s0[x3 + 2];
				const uchar* s1 = src->ptr<uchar>(y1);
				neighbor_b[4] = s1[x0 + 0];
				neighbor_g[4] = s1[x0 + 1];
				neighbor_r[4] = s1[x0 + 2];
				neighbor_b[5] = s1[x1 + 0];
				neighbor_g[5] = s1[x1 + 1];
				neighbor_r[5] = s1[x1 + 2];
				neighbor_b[6] = s1[x2 + 0];
				neighbor_g[6] = s1[x2 + 1];
				neighbor_r[6] = s1[x2 + 2];
				neighbor_b[7] = s1[x3 + 0];
				neighbor_g[7] = s1[x3 + 1];
				neighbor_r[7] = s1[x3 + 2];

				const __m256 b0 = _mm256_load_epu8cvtps((__m128i*)(neighbor_b + 0));
				const __m256 g0 = _mm256_load_epu8cvtps((__m128i*)(neighbor_g + 0));
				const __m256 r0 = _mm256_load_epu8cvtps((__m128i*)(neighbor_r + 0));

				const uchar* s2 = src->ptr<uchar>(y2);
				neighbor_b[8] = s2[x0 + 0];
				neighbor_g[8] = s2[x0 + 1];
				neighbor_r[8] = s2[x0 + 2];
				neighbor_b[9] = s2[x1 + 0];
				neighbor_g[9] = s2[x1 + 1];
				neighbor_r[9] = s2[x1 + 2];
				neighbor_b[10] = s2[x2 + 0];
				neighbor_g[10] = s2[x2 + 1];
				neighbor_r[10] = s2[x2 + 2];
				neighbor_b[11] = s2[x3 + 0];
				neighbor_g[11] = s2[x3 + 1];
				neighbor_r[11] = s2[x3 + 2];
				const uchar* s3 = src->ptr<uchar>(y3);
				neighbor_b[12] = s3[x0 + 0];
				neighbor_g[12] = s3[x0 + 1];
				neighbor_r[12] = s3[x0 + 2];
				neighbor_b[13] = s3[x1 + 0];
				neighbor_g[13] = s3[x1 + 1];
				neighbor_r[13] = s3[x1 + 2];
				neighbor_b[14] = s3[x2 + 0];
				neighbor_g[14] = s3[x2 + 1];
				neighbor_r[14] = s3[x2 + 2];
				neighbor_b[15] = s3[x3 + 0];
				neighbor_g[15] = s3[x3 + 1];
				neighbor_r[15] = s3[x3 + 2];

				const __m256 b1 = _mm256_load_epu8cvtps((__m128i*)(neighbor_b + 8));
				const __m256 g1 = _mm256_load_epu8cvtps((__m128i*)(neighbor_g + 8));
				const __m256 r1 = _mm256_load_epu8cvtps((__m128i*)(neighbor_r + 8));

				for (int n = 0; n < scale; n++)
				{
					uchar *dest_ptr = (uchar*)dest->ptr<uchar>(y + n); // output
					for (int m = 0; m < scale; m++)
					{
						int idx = n * scale + m;
						const float *weightmap_ptr = weightmap->ptr<float>(idx);
						const __m256 mw0 = _mm256_load_ps(weightmap_ptr);
						const __m256 mw1 = _mm256_load_ps(weightmap_ptr + 8);

#ifdef UPSAMPLE_USE_SIMD					
						__m256 t0 = _mm256_mul_ps(mw0, b0);
						t0 = _mm256_fmadd_ps(mw1, b1, t0);
						float v0 = _mm256_reduceadd_ps(t0);

						t0 = _mm256_mul_ps(mw0, g0);
						t0 = _mm256_fmadd_ps(mw1, g1, t0);
						float v1 = _mm256_reduceadd_ps(t0);

						t0 = _mm256_mul_ps(mw0, r0);
						t0 = _mm256_fmadd_ps(mw1, r1, t0);
						float v2 = _mm256_reduceadd_ps(t0);
#else
						float v0 = 0.f;
						float v1 = 0.f;
						float v2 = 0.f;
						for (int k = 0; k < 16; k++)
						{
							v0 += weightmap_ptr[k] * neighbor_b[k];
							v1 += weightmap_ptr[k] * neighbor_g[k];
							v2 += weightmap_ptr[k] * neighbor_r[k];
						}
#endif
						dest_ptr[3 * (x + m) + 0] = saturate_cast<uchar>(v0);
						dest_ptr[3 * (x + m) + 1] = saturate_cast<uchar>(v1);
						dest_ptr[3 * (x + m) + 2] = saturate_cast<uchar>(v2);
					}
				}
			}
		}
	}
};

class UpsampleConv4x4_32F_ParallelBody : public cv::ParallelLoopBody
{
private:
	const cv::Mat* src;
	const cv::Mat* weightmap;
	cv::Mat* dest;

	int scale;
public:
	UpsampleConv4x4_32F_ParallelBody(const cv::Mat& src, const cv::Mat& weightmap, cv::Mat& dst, const int scale)
		: src(&src), weightmap(&weightmap), dest(&dst), scale(scale)
	{
	}

	void operator() (const cv::Range& range) const
	{
		float CV_DECL_ALIGNED(AVX_ALIGN) neighbor_b[16];
		float CV_DECL_ALIGNED(AVX_ALIGN) neighbor_g[16];
		float CV_DECL_ALIGNED(AVX_ALIGN) neighbor_r[16];

		if (src->channels() == 3)
		{
			for (int y = range.start; y < range.end; y += scale)
			{
				const int y_ = (int)(y / scale);
				const int y0 = max(0, y_ - 1);
				const int y1 = y_;
				const int y2 = min(y_ + 1, src->rows - 1);
				const int y3 = min(y_ + 2, src->rows - 1);

				for (int x = 0; x < dest->cols; x += scale)
				{
					const int x_ = x / scale;
					const int x0 = max(0, x_ - 1) * 3;
					const int x1 = (x_) * 3;
					const int x2 = min(x_ + 1, src->cols - 1) * 3;
					const int x3 = min(x_ + 2, src->cols - 1) * 3;

					const float* s0 = src->ptr<float>(y0);
					neighbor_b[0] = s0[x0 + 0];
					neighbor_g[0] = s0[x0 + 1];
					neighbor_r[0] = s0[x0 + 2];
					neighbor_b[1] = s0[x1 + 0];
					neighbor_g[1] = s0[x1 + 1];
					neighbor_r[1] = s0[x1 + 2];
					neighbor_b[2] = s0[x2 + 0];
					neighbor_g[2] = s0[x2 + 1];
					neighbor_r[2] = s0[x2 + 2];
					neighbor_b[3] = s0[x3 + 0];
					neighbor_g[3] = s0[x3 + 1];
					neighbor_r[3] = s0[x3 + 2];
					const float* s1 = src->ptr<float>(y1);
					neighbor_b[4] = s1[x0 + 0];
					neighbor_g[4] = s1[x0 + 1];
					neighbor_r[4] = s1[x0 + 2];
					neighbor_b[5] = s1[x1 + 0];
					neighbor_g[5] = s1[x1 + 1];
					neighbor_r[5] = s1[x1 + 2];
					neighbor_b[6] = s1[x2 + 0];
					neighbor_g[6] = s1[x2 + 1];
					neighbor_r[6] = s1[x2 + 2];
					neighbor_b[7] = s1[x3 + 0];
					neighbor_g[7] = s1[x3 + 1];
					neighbor_r[7] = s1[x3 + 2];

					const __m256 b0 = _mm256_load_ps((neighbor_b + 0));
					const __m256 g0 = _mm256_load_ps((neighbor_g + 0));
					const __m256 r0 = _mm256_load_ps((neighbor_r + 0));

					const float* s2 = src->ptr<float>(y2);
					neighbor_b[8] = s2[x0 + 0];
					neighbor_g[8] = s2[x0 + 1];
					neighbor_r[8] = s2[x0 + 2];
					neighbor_b[9] = s2[x1 + 0];
					neighbor_g[9] = s2[x1 + 1];
					neighbor_r[9] = s2[x1 + 2];
					neighbor_b[10] = s2[x2 + 0];
					neighbor_g[10] = s2[x2 + 1];
					neighbor_r[10] = s2[x2 + 2];
					neighbor_b[11] = s2[x3 + 0];
					neighbor_g[11] = s2[x3 + 1];
					neighbor_r[11] = s2[x3 + 2];
					const float* s3 = src->ptr<float>(y3);
					neighbor_b[12] = s3[x0 + 0];
					neighbor_g[12] = s3[x0 + 1];
					neighbor_r[12] = s3[x0 + 2];
					neighbor_b[13] = s3[x1 + 0];
					neighbor_g[13] = s3[x1 + 1];
					neighbor_r[13] = s3[x1 + 2];
					neighbor_b[14] = s3[x2 + 0];
					neighbor_g[14] = s3[x2 + 1];
					neighbor_r[14] = s3[x2 + 2];
					neighbor_b[15] = s3[x3 + 0];
					neighbor_g[15] = s3[x3 + 1];
					neighbor_r[15] = s3[x3 + 2];

					const __m256 b1 = _mm256_load_ps((neighbor_b + 8));
					const __m256 g1 = _mm256_load_ps((neighbor_g + 8));
					const __m256 r1 = _mm256_load_ps((neighbor_r + 8));

					for (int n = 0; n < scale; n++)
					{
						float *dest_ptr = dest->ptr<float>(y + n); // output
						for (int m = 0; m < scale; m++)
						{
							int idx = n * scale + m;
							const float *weightmap_ptr = weightmap->ptr<float>(idx);
							const __m256 mw0 = _mm256_load_ps(weightmap_ptr);
							const __m256 mw1 = _mm256_load_ps(weightmap_ptr + 8);

#ifdef UPSAMPLE_USE_SIMD					
							__m256 t0 = _mm256_mul_ps(mw0, b0);
							t0 = _mm256_fmadd_ps(mw1, b1, t0);
							float v0 = _mm256_reduceadd_ps(t0);

							t0 = _mm256_mul_ps(mw0, g0);
							t0 = _mm256_fmadd_ps(mw1, g1, t0);
							float v1 = _mm256_reduceadd_ps(t0);

							t0 = _mm256_mul_ps(mw0, r0);
							t0 = _mm256_fmadd_ps(mw1, r1, t0);
							float v2 = _mm256_reduceadd_ps(t0);
#else
							float v0 = 0.f;
							float v1 = 0.f;
							float v2 = 0.f;
							for (int k = 0; k < 16; k++)
							{
								v0 += weightmap_ptr[k] * neighbor_b[k];
								v1 += weightmap_ptr[k] * neighbor_g[k];
								v2 += weightmap_ptr[k] * neighbor_r[k];
							}
#endif
							dest_ptr[3 * (x + m) + 0] = v0;
							dest_ptr[3 * (x + m) + 1] = v1;
							dest_ptr[3 * (x + m) + 2] = v2;
						}
					}
				}
			}
		}
		else if (src->channels() == 1)
		{
			for (int y = 0; y < dest->rows; y += scale)
			{
				const int y_ = (int)(y / scale);
				const int y0 = max(0, y_ - 1);
				const int y1 = y_;
				const int y2 = min(y_ + 1, src->rows - 1);
				const int y3 = min(y_ + 2, src->rows - 1);

				for (int x = 0; x < dest->cols; x += scale)
				{
					const int x_ = x / scale;
					const int x0 = max(0, x_ - 1);
					const int x1 = (x_);
					const int x2 = min(x_ + 1, src->cols - 1);
					const int x3 = min(x_ + 2, src->cols - 1);

					const float* s0 = src->ptr<float>(y0);
					neighbor_b[0] = s0[x0 + 0];
					neighbor_b[1] = s0[x1 + 0];
					neighbor_b[2] = s0[x2 + 0];
					neighbor_b[3] = s0[x3 + 0];
					const float* s1 = src->ptr<float>(y1);
					neighbor_b[4] = s1[x0 + 0];
					neighbor_b[5] = s1[x1 + 0];
					neighbor_b[6] = s1[x2 + 0];
					neighbor_b[7] = s1[x3 + 0];

					const __m256 b0 = _mm256_load_ps((neighbor_b + 0));

					const float* s2 = src->ptr<float>(y2);
					neighbor_b[8] = s2[x0 + 0];
					neighbor_b[9] = s2[x1 + 0];
					neighbor_b[10] = s2[x2 + 0];
					neighbor_b[11] = s2[x3 + 0];
					const float* s3 = src->ptr<float>(y3);
					neighbor_b[12] = s3[x0 + 0];
					neighbor_b[13] = s3[x1 + 0];
					neighbor_b[14] = s3[x2 + 0];
					neighbor_b[15] = s3[x3 + 0];

					const __m256 b1 = _mm256_load_ps((neighbor_b + 8));
					for (int n = 0; n < scale; n++)
					{
						float *dest_ptr = dest->ptr<float>(y + n); // output
						for (int m = 0; m < scale; m++)
						{
							int idx = n * scale + m;
							const float *weightmap_ptr = weightmap->ptr<float>(idx);
							const __m256 mw0 = _mm256_load_ps(weightmap_ptr);
							const __m256 mw1 = _mm256_load_ps(weightmap_ptr + 8);

#ifdef UPSAMPLE_USE_SIMD					
							__m256 t0 = _mm256_mul_ps(mw0, b0);
							t0 = _mm256_fmadd_ps(mw1, b1, t0);
							float v0 = _mm256_reduceadd_ps(t0);
#else
							float v0 = 0.f;
							for (int k = 0; k < 16; k++)
							{
								v0 += weightmap_ptr[k] * neighbor_b[k];
							}
#endif
							dest_ptr[x + m] = v0;
						}
					}
				}
			}
		}
	}
};

void upsampleCubic_parallel(const Mat& src, Mat& dest, const int scale, const double a)
{
	dest.create(src.size()*scale, src.type());

	Mat weightmap(scale*scale, 16, CV_32F);
	setCubicWeight4x4(weightmap, a);
	//vizCubicKernel(scale);

	if (src.depth() == CV_32F)
	{
		//upsampleCubic_parallel_32f(src, dest, scale, a);
		cv::parallel_for_
		(
			cv::Range(0, dest.rows),
			UpsampleConv4x4_32F_ParallelBody(src, weightmap, dest, scale),
			8
		);
	}
	else if (src.depth() == CV_8U)
	{	
		CV_Assert(src.channels() == 3);
		cv::parallel_for_
		(
			cv::Range(0, dest.rows),
			UpsampleConv4x4_8U_ParallelBody(src, weightmap, dest, scale),
			8
		);
	}
}

template<typename T>
void upsampleWeightedCubic(Mat& src, Mat& guide, Mat& dest, const int scale, const double a)
{
	Mat weightmap(scale*scale, 16, CV_32F);

	Mat sguide;
	//resize(guide, sguide, src.size(), 0, 0, INTER_NEAREST);
	downsample(guide, sguide, scale, INTER_NEAREST, 4);

	float rweight[256];
	int ss = 30;
	int sr = 30;
	int nrm = 20;
	//static int ss = 30; createTrackbar("sss", "", &ss, 200);
	//static int sr = 30; createTrackbar("ssr", "", &sr, 200);
	//static int nrm = 20; createTrackbar("nrm", "", &nrm, 200);

	/*for (int i = 0; i < 256; i++)
	{
			rweight[i] = (i < sr) ? 1.f: 0.0000001f;
	}*/


	float n = nrm * 0.1;
	for (int i = 0; i < 256; i++)
	{
		rweight[i] = exp(pow(i, n) / (-1.0 / n * pow(sr, n)));
	}
	/*
	cp::Plot pt(Size(512, 512));
	for (int i = -255; i < 256; i++)
	{

		float v = exp(pow(abs(i), n) / (-1.0 / n * pow(sr, n)));

		pt.push_back(i, v);
	}
	pt.plot("pt", false);*/
	//createCubicWeight(weightmap, a);
	//createCubicWeightNonSep(weightmap, a);
	setGaussianWeight4x4(weightmap, (float)ss);

#pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < dest.rows; y += scale)
	{
		/*const int y0 = (int)(y / scale);
		const int y1 = y0 + 1;
		const int y2 = y0 + 2;
		const int y3 = y0 + 3;*/
		const int y0 = max(0, (int)(y / scale) - 1);
		const int y1 = y0 + 1;
		const int y2 = min(y0 + 2, src.rows - 1);
		const int y3 = min(y0 + 3, src.rows - 1);

		float neighbor_b[16];
		float neighbor_g[16];
		float neighbor_r[16];
		float* neighbor;

		float gneighbor_b[16];
		float gneighbor_g[16];
		float gneighbor_r[16];
		float* gneighbor;

		for (int x = 0; x < dest.cols; x += scale)
		{
			/*const int x0 = x / scale;
			const int x1 = x0 + 1;
			const int x2 = x0 + 2;
			const int x3 = x0 + 3;*/
			const int x0 = max(0, x / scale - 1);
			const int x1 = x0 + 1;
			const int x2 = min(x0 + 2, src.cols - 1);
			const int x3 = min(x0 + 3, src.cols - 1);

			for (int c = 0; c < 3; c++)
			{
				if (c == 0)
				{
					neighbor = &neighbor_b[0];
					gneighbor = &gneighbor_b[0];
				}
				if (c == 1)
				{
					neighbor = &neighbor_g[0];
					gneighbor = &gneighbor_g[0];
				}
				if (c == 2)
				{
					neighbor = &neighbor_r[0];
					gneighbor = &gneighbor_r[0];
				}

				neighbor[0] = (float)src.at<T>(y0, 3 * x0 + c);
				neighbor[1] = (float)src.at<T>(y0, 3 * x1 + c);
				neighbor[2] = (float)src.at<T>(y0, 3 * x2 + c);
				neighbor[3] = (float)src.at<T>(y0, 3 * x3 + c);
				neighbor[4] = (float)src.at<T>(y1, 3 * x0 + c);
				neighbor[5] = (float)src.at<T>(y1, 3 * x1 + c);
				neighbor[6] = (float)src.at<T>(y1, 3 * x2 + c);
				neighbor[7] = (float)src.at<T>(y1, 3 * x3 + c);
				neighbor[8] = (float)src.at<T>(y2, 3 * x0 + c);
				neighbor[9] = (float)src.at<T>(y2, 3 * x1 + c);
				neighbor[10] = (float)src.at<T>(y2, 3 * x2 + c);
				neighbor[11] = (float)src.at<T>(y2, 3 * x3 + c);
				neighbor[12] = (float)src.at<T>(y3, 3 * x0 + c);
				neighbor[13] = (float)src.at<T>(y3, 3 * x1 + c);
				neighbor[14] = (float)src.at<T>(y3, 3 * x2 + c);
				neighbor[15] = (float)src.at<T>(y3, 3 * x3 + c);

				gneighbor[0] = (float)sguide.at<T>(y0, 3 * x0 + c);
				gneighbor[1] = (float)sguide.at<T>(y0, 3 * x1 + c);
				gneighbor[2] = (float)sguide.at<T>(y0, 3 * x2 + c);
				gneighbor[3] = (float)sguide.at<T>(y0, 3 * x3 + c);
				gneighbor[4] = (float)sguide.at<T>(y1, 3 * x0 + c);
				gneighbor[5] = (float)sguide.at<T>(y1, 3 * x1 + c);
				gneighbor[6] = (float)sguide.at<T>(y1, 3 * x2 + c);
				gneighbor[7] = (float)sguide.at<T>(y1, 3 * x3 + c);
				gneighbor[8] = (float)sguide.at<T>(y2, 3 * x0 + c);
				gneighbor[9] = (float)sguide.at<T>(y2, 3 * x1 + c);
				gneighbor[10] = (float)sguide.at<T>(y2, 3 * x2 + c);
				gneighbor[11] = (float)sguide.at<T>(y2, 3 * x3 + c);
				gneighbor[12] = (float)sguide.at<T>(y3, 3 * x0 + c);
				gneighbor[13] = (float)sguide.at<T>(y3, 3 * x1 + c);
				gneighbor[14] = (float)sguide.at<T>(y3, 3 * x2 + c);
				gneighbor[15] = (float)sguide.at<T>(y3, 3 * x3 + c);
			}

			for (int n = 0; n < scale; n++)
			{
				uchar *guide_ptr = guide.ptr<uchar>(y + n); // reference
				T *dest_ptr = dest.ptr<T>(y + n); // output

				for (int m = 0; m < scale; m++)
				{
					int idx = n * scale + m;
					float *weightmap_ptr = weightmap.ptr<float>(idx);
					const __m256 mw0 = _mm256_load_ps(weightmap_ptr);
					const __m256 mw1 = _mm256_load_ps(weightmap_ptr + 8);

					for (int c = 0; c < 3; c++)
					{
						if (c == 0)
						{
							neighbor = &neighbor_b[0];
							gneighbor = &gneighbor_b[0];
						}
						if (c == 1)
						{
							neighbor = &neighbor_g[0];
							gneighbor = &gneighbor_g[0];
						}
						if (c == 2)
						{
							neighbor = &neighbor_r[0];
							gneighbor = &gneighbor_r[0];
						}

						uchar g = guide_ptr[3 * (x + m) + c];

#define SIMD
#ifdef SIMD
						const int CV_DECL_ALIGNED(32) v32f_absmask_[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
						const __m256 v32f_absmask = _mm256_load_ps((float*)v32f_absmask_);

						__m256 mg = _mm256_set1_ps(g);

						__m256i midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mg, _mm256_load_ps(gneighbor)), v32f_absmask));
						__m256 mrw0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(rweight, midx, sizeof(float)));
						__m256 t0 = _mm256_load_ps(neighbor);
						t0 = _mm256_mul_ps(mrw0, t0);

						midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mg, _mm256_load_ps(gneighbor + 8)), v32f_absmask));
						__m256 mrw1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(rweight, midx, sizeof(float)));
						__m256 t1 = _mm256_load_ps(neighbor + 8);
						t0 = _mm256_fmadd_ps(mrw1, t1, t0);

						float v = _mm256_reduceadd_ps(t0) / _mm256_reduceadd_ps(_mm256_add_ps(mrw0, mrw1));
						dest_ptr[3 * (x + m) + c] = saturate_cast<uchar>(v);
#else
						float v = 0.f;
						float wsum = 0.f;
						for (int k = 0; k < 16; k++)
						{
							float w = rweight[abs(g - (int)gneighbor[k])] * weightmap_ptr[k];
							v += w * neighbor[k];
							wsum += w;
						}
						dest_ptr[3 * (x + m) + c] = saturate_cast<uchar>(v / wsum);
#endif				
					}
				}
			}
		}
	}
}

void upsampleWeightedCubic(InputArray src_, InputArray guide_, OutputArray dest_, const int scale, const double a)
{
	Mat src = src_.getMat();
	Mat guide = guide_.getMat();
	if (dest_.empty() || dest_.size() != Size(src.cols*scale, src.rows*scale))
	{
		dest_.create(Size(src.cols*scale, src.rows*scale), src_.type());
	}

	Mat dest = dest_.getMat();

	upsampleWeightedCubic<uchar>(src, guide, dest, scale, a);
}

void upSampleCV(InputArray src_, OutputArray dest_, const int scale, const int method, const double sx, const double sy)
{
	Mat src = src_.getMat();

	Mat a = cp::convert(src, CV_32F);
	Mat b;
	resize(a, b, Size(), scale, scale, method);

	cp::warpShiftSubpix(b, b, sx, sy, method);

	b.convertTo(dest_, src_.depth());
}
```
