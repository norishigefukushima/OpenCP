#include "StereoBase.hpp"
#include "depthfilter.hpp"
#include "StereoBM2.hpp"
#include "costVolumeFilter.hpp"
#include "crossBasedLocalMultipointFilter.hpp"
#include "guidedFilter.hpp"
#include "jointBilateralFilter.hpp"
#include "binalyWeightedRangeFilter.hpp"
#include "jointNearestFilter.hpp"
#include "weightedModeFilter.hpp"
#include "minmaxfilter.hpp"
#include "plot.hpp"
#include "timer.hpp"
#include "consoleImage.hpp"
#include "shiftImage.hpp"
#include "blend.hpp"

#include "inlinesimdfunctions.hpp"

#include "debugcp.hpp"

using namespace std;
using namespace cv;

//#define TIMER_STEREO_BASE

namespace cp
{
#pragma region correctDisparityBoundary
	template <class srcType>
	void correctDisparityBoundaryECV(Mat& src, Mat& refimg, const int r, const int edgeth, Mat& dest)
	{
		srcType invalidvalue = 0;

		vector<Mat> ref;
		split(refimg, ref);

		Mat ss;
		Mat sobel;

		Sobel(ref[0], sobel, CV_16S, 1, 0, 3);
		sobel = abs(sobel);
		Sobel(ref[1], ss, CV_16S, 1, 0, 3);
		max(sobel, abs(ss), sobel);
		Sobel(ref[2], ss, CV_16S, 1, 0, 3);
		max(sobel, abs(ss), sobel);

		srcType* s = src.ptr<srcType>(0);
		const int step = src.cols;
		short* sbl = sobel.ptr<short>(0);


		int i, j = 0, k;
		for (; j < src.rows; j++)
		{
			for (i = r + 1; i < src.cols - r - 1; i++)
			{
				if (abs(s[i - 1] - s[i]) < edgeth) continue;

				srcType maxd;
				srcType mind;

				const srcType cd = s[i];
				if (s[i - 1] < s[i])
				{
					const int rl = -(r);
					const int rr = r;
					mind = s[i - 1];
					maxd = s[i];
					const srcType sub = (maxd - mind);

					int maxp;
					int maxval = 0;

					for (k = rl; k <= rr; k++)
					{
						const int rcost = 0;//abs(r);
						if (sbl[i + k] + rcost > maxval)
						{
							if (abs(s[i - 1 + k] - s[i + k]) * 2 >= sub)
							{
								maxp = 0;
								if (k != 0)
									break;

							}
							maxp = k;
							maxval = sbl[i + k] + rcost;
						}
					}
					if (maxp > 0)
					{
						for (int n = 0; n <= maxp; n++, i++)
						{
							s[i] = mind;
						}
						i++;
					}
					else if (maxp < 0)
					{
						i += maxp;
						for (int n = 0; n < -maxp; n++, i++)
						{
							s[i] = maxd;
						}
						i++;
					}
				}
				else
				{
					maxd = s[i - 1];
					mind = s[i];
					const int rl = -r;
					const int rr = (r);
					const srcType sub = (maxd - mind) * 2;

					int maxp;
					int maxval = 0;
					for (k = rl; k <= rr; k++)
					{
						const int rcost = 0;//abs(r);
						if (sbl[i + k] + rcost > maxval)
						{
							if (abs(s[i - 1 + k] - s[i + k]) * 2 >= sub)
							{
								maxp = 0;
								if (k != 0)break;
							}
							maxp = k;
							maxval = sbl[i + k] + rcost;
						}
					}
					if (maxp > 0)
					{
						for (int n = 0; n <= maxp; n++, i++)
						{
							s[i] = maxd;
						}
						i++;
					}
					else if (maxp < 0)
					{
						i += maxp;
						for (int n = 0; n < -maxp; n++, i++)
						{
							s[i] = mind;
						}
						i++;
					}
				}

			}
			s += step;
			sbl += step;
		}
	}
	template <class srcType>
	void correctDisparityBoundaryEC(Mat& src, Mat& refimg, const int r, const int edgeth, Mat& dest)
	{

		srcType invalidvalue = 0;

		vector<Mat> ref;
		split(refimg, ref);

		Mat ss;
		Mat sobel;

		Sobel(ref[0], sobel, CV_16S, 1, 0, 3);
		sobel = abs(sobel);
		Sobel(ref[1], ss, CV_16S, 1, 0, 3);
		max(sobel, abs(ss), sobel);
		Sobel(ref[2], ss, CV_16S, 1, 0, 3);
		max(sobel, abs(ss), sobel);

		srcType* s = src.ptr<srcType>(0);
		const int step = src.cols;
		short* sbl = sobel.ptr<short>(0);


		int i, j = 0, k;
		for (; j < src.rows; j++)
		{
			for (i = r + 1; i < src.cols - r - 1; i++)
			{
				if (abs(s[i - 1] - s[i]) < edgeth) continue;

				srcType maxd;
				srcType mind;

				const srcType cd = s[i];
				if (s[i - 1] < s[i])
				{
					const int rl = -(r >> 1);
					const int rr = r;
					mind = s[i - 1];
					maxd = s[i];
					const srcType sub = (maxd - mind);

					int maxp;
					int maxval = 0;

					for (k = rl; k <= rr; k++)
					{
						const int rcost = 0;//abs(r);
						if (sbl[i + k] + rcost > maxval)
						{
							if (abs(s[i - 1 + k] - s[i + k]) * 2 >= sub)
							{
								maxp = 0;
								if (k != 0)
									break;

							}
							maxp = k;
							maxval = sbl[i + k] + rcost;
						}
					}
					if (maxp > 0)
					{
						for (int n = 0; n <= maxp; n++, i++)
						{
							s[i] = mind;
						}
						i++;
					}
					else if (maxp < 0)
					{
						i += maxp;
						for (int n = 0; n < -maxp; n++, i++)
						{
							s[i] = maxd;
						}
						i++;
					}
				}
				else
				{
					maxd = s[i - 1];
					mind = s[i];
					const int rl = -r;
					const int rr = (r >> 1);
					const srcType sub = (maxd - mind) * 2;

					int maxp;
					int maxval = 0;
					for (k = rl; k <= rr; k++)
					{
						const int rcost = 0;//abs(r);
						if (sbl[i + k] + rcost > maxval)
						{
							if (abs(s[i - 1 + k] - s[i + k]) * 2 >= sub)
							{
								maxp = 0;
								if (k != 0)break;
							}
							maxp = k;
							maxval = sbl[i + k] + rcost;
						}
					}
					if (maxp > 0)
					{
						for (int n = 0; n <= maxp; n++, i++)
						{
							s[i] = maxd;
						}
						i++;
					}
					else if (maxp < 0)
					{
						i += maxp;
						for (int n = 0; n < -maxp; n++, i++)
						{
							s[i] = mind;
						}
						i++;
					}
				}

			}
			s += step;
			sbl += step;
		}
	}

	template <class srcType>
	void correctDisparityBoundaryE(Mat& src, Mat& refimg, const int r, const int edgeth, Mat& dest, const int secondr, const int minedge)
	{

		srcType invalidvalue = 0;

		Mat ref;
		if (refimg.channels() == 3)cvtColor(refimg, ref, COLOR_BGR2GRAY);
		else ref = refimg.clone();

		Mat sobel;
		Sobel(ref, sobel, CV_16S, 1, 0, 3);
		sobel = abs(sobel);

		srcType* s = src.ptr<srcType>(0);
		const int step = src.cols;
		short* sbl = sobel.ptr<short>(0);


		int i, j = 0, k;
		for (; j < src.rows; j++)
		{
			for (i = r + 1; i < src.cols - r - 1; i++)
			{
				if (abs(s[i - 1] - s[i]) < edgeth) continue;

				srcType maxd;
				srcType mind;
				const srcType cd = s[i];
				if (s[i - 1] < s[i])
				{
					const int rl = -(r >> 1);
					const int rr = r;
					mind = s[i - 1];
					maxd = s[i];
					const srcType sub = (maxd - mind);

					int maxp;
					int maxval = 0;

					for (k = rl; k <= rr; k++)
					{
						const int rcost = 0; abs(r);
						if (sbl[i + k] + rcost > maxval && s[i + k] <= maxd + 16 && s[i + k] >= mind - 16)
						{
							maxp = k;
							maxval = sbl[i + k] + rcost;
						}
					}
					/*	if(maxval<minedge)
					{
					for(k=r+1;k<=r+secondr+1;k++)
					{
					if(sbl[i+k]>maxval)
					{
					maxp=k;
					maxval=sbl[i+k];
					}
					}
					}*/
					if (maxp > 0)
					{
						for (int n = 0; n <= maxp; n++, i++)
						{
							s[i] = mind;
						}
						i++;
					}
					else if (maxp < 0)
					{
						i += maxp;
						for (int n = 0; n < -maxp; n++, i++)
						{
							s[i] = maxd;
						}
						i++;
					}
				}
				else
				{
					const int rl = -r;
					const int rr = (r >> 1);
					maxd = s[i - 1];
					mind = s[i];
					const srcType sub = (maxd - mind) * 2;

					int maxp;
					int maxval = 0;
					for (k = rl; k <= rr; k++)
					{
						const int rcost = 0; abs(r);
						if (sbl[i + k] + rcost > maxval && s[i + k] <= maxd + 16 && s[i + k] >= mind - 16)
						{
							maxp = k;
							maxval = sbl[i + k] + rcost;
						}
					}

					/*if(maxval<minedge)
					{
					for(k=-r-secondr;k<-r;k++)
					{
					if(sbl[i+k]>maxval && abs(s[i+k]-cd)<=sub)
					{
					maxp=k;
					maxval=sbl[i+k];
					}
					}
					}*/
					if (maxp > 0)
					{
						for (int n = 0; n <= maxp; n++, i++)
						{
							s[i] = maxd;
						}
						i++;
					}
					else if (maxp < 0)
					{
						i += maxp;
						for (int n = 0; n < -maxp; n++, i++)
						{
							s[i] = mind;
						}
						i++;
					}
				}

			}
			s += step;
			sbl += step;
		}
	}
#pragma endregion

	StereoBase::StereoBase(int blockSize, int minDisp, int disparityRange) :thread_max(omp_get_max_threads())
	{
		border = 0;
		speckleWindowSize = 20;
		speckleRange = 16;
		uniquenessRatio = 0;

		subpixelInterpolationMethod = SUBPIXEL_QUAD;

		subpixelRangeFilterWindow = 2;
		subpixelRangeFilterCap = 16;

		pixelMatchErrorCap = 31;
		aggregationGuidedfilterEps = 0.1;
		aggregationSigmaSpace = 255.0;
		aggregationRadiusH = blockSize;
		aggregationRadiusV = blockSize;
		minDisparity = minDisp;
		numberOfDisparities = disparityRange;

		preFilterCap = 31;
		costAlphaImageSobel = 10;
		sobelBlendMapParam_Size = 0;
		sobelBlendMapParam1 = 50;
		sobelBlendMapParam2 = 20;

		P1 = 0;
		P2 = 0;
		gif = new GuidedImageFilter[thread_max];
	}

	StereoBase::~StereoBase()
	{
		delete[] gif;
	}

	void StereoBase::imshowDisparity(string wname, Mat& disp, int option, OutputArray output)
	{
		//cvtDisparityColor(disp,output,minDisparity,numberOfDisparities,option,16);
		output.create(disp.size(), CV_8UC3);
		Mat dst = output.getMat();
		cvtDisparityColor(disp, dst, 0, 64, option, 16);
		imshow(wname, output);
	}

	void StereoBase::imshowDisparity(string wname, Mat& disp, int option, Mat& output, int mindis, int range)
	{
		cvtDisparityColor(disp, output, mindis, range, option, 16);
		imshow(wname, output);
	}

#pragma region cost computation of pixel matching

#define  CV_CAST_8U(t)  (uchar)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)
	static void prefilterXSobel(Mat& src, Mat& dst, const int preFilterCap)
	{
		if (dst.empty() || dst.depth() != CV_8U)dst.create(src.size(), CV_8U);

		const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;

		Size size = src.size();
		const uchar val0 = preFilterCap;

		const int step = 2 * src.cols;
		uchar* srow1 = src.ptr<uchar>();
		uchar* dptr0 = dst.ptr<uchar>();
		uchar* dptr1 = dptr0 + dst.step;

		const int WIDTH = get_simd_floor(size.width - 1, 8);
		const int e = size.width - 1;

		int y;
		//unrolling y0 and y1
		for (y = 0; y < size.height - 1; y += 2)
		{
			const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
			const uchar* srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
			const uchar* srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;

			dptr0[0] = saturate_cast<uchar>(min(preFilterCap * 2, (srow0[0] - srow0[1]) + 4 * (srow1[0] - srow1[1]) + (srow2[0] - srow2[1]) + preFilterCap));
			dptr1[0] = saturate_cast<uchar>(min(preFilterCap * 2, (srow1[0] - srow1[1]) + 4 * (srow2[0] - srow2[1]) + (srow3[0] - srow3[1]) + preFilterCap));

			__m128i zero = _mm_setzero_si128(), ftz = _mm_set1_epi16((short)preFilterCap),
				ftz2 = _mm_set1_epi8(CV_CAST_8U(preFilterCap * 2));
#if 0
			for (int x = 1; x < WIDTH; x += 8)
			{
				__m128i c0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x - 1)), zero);
				__m128i c1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x - 1)), zero);
				__m128i d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x + 1)), zero);
				__m128i d1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x + 1)), zero);

				d0 = _mm_sub_epi16(d0, c0);
				d1 = _mm_sub_epi16(d1, c1);

				__m128i c2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x - 1)), zero);
				__m128i c3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x - 1)), zero);
				__m128i d2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x + 1)), zero);
				__m128i d3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x + 1)), zero);

				d2 = _mm_sub_epi16(d2, c2);
				d3 = _mm_sub_epi16(d3, c3);

				__m128i v0 = _mm_add_epi16(d0, _mm_add_epi16(d2, _mm_add_epi16(d1, d1)));
				__m128i v1 = _mm_add_epi16(d1, _mm_add_epi16(d3, _mm_add_epi16(d2, d2)));
				v0 = _mm_packus_epi16(_mm_add_epi16(v0, ftz), _mm_add_epi16(v1, ftz));
				v0 = _mm_min_epu8(v0, ftz2);

				_mm_storel_epi64((__m128i*)(dptr0 + x), v0);
				_mm_storel_epi64((__m128i*)(dptr1 + x), _mm_unpackhi_epi64(v0, v0));
			}
#endif
			//for (int x = WIDTH; x < size.width - 1; x++)
			for (int x = 1; x < size.width - 1; x++)
			{
				dptr0[x] = saturate_cast<uchar>(min(preFilterCap * 2, (srow0[x - 1] - srow0[x + 1]) + 4 * (srow1[x - 1] - srow1[x + 1]) + (srow2[x - 1] - srow2[x + 1]) + preFilterCap));
				dptr1[x] = saturate_cast<uchar>(min(preFilterCap * 2, (srow1[x - 1] - srow1[x + 1]) + 4 * (srow2[x - 1] - srow2[x + 1]) + (srow3[x - 1] - srow3[x + 1]) + preFilterCap));
			}
			dptr0[e] = saturate_cast<uchar>(min(preFilterCap * 2, (srow0[e] - srow0[e - 1]) + 4 * (srow1[e] - srow1[e - 1]) + (srow2[e] - srow2[e - 1]) + preFilterCap));
			dptr1[e] = saturate_cast<uchar>(min(preFilterCap * 2, (srow1[e] - srow1[e - 1]) + 4 * (srow2[e] - srow2[e - 1]) + (srow3[e] - srow3[e - 1]) + preFilterCap));
			srow1 += step;
			dptr0 += step;
			dptr1 += step;
		}
		srow1 -= src.cols;
		for (; y < size.height; y++)
		{
			uchar* dptr = dst.ptr<uchar>(y);
			const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
			dptr[0] = saturate_cast<uchar>(min(preFilterCap * 2, 2 * (srow0[0] - srow0[1]) + 3 * (srow1[0] - srow1[1]) + preFilterCap));
			for (int x = 1; x < size.width - 1; x++)
			{
				dptr[x] = saturate_cast<uchar>(min(preFilterCap * 2, 2 * (srow0[x - 1] - srow0[x + 1]) + 3 * (srow1[x - 1] - srow1[x + 1]) + preFilterCap));
			}
			dptr[e] = saturate_cast<uchar>(min(preFilterCap * 2, 2 * (srow0[e] - srow0[e - 1]) + 3 * (srow1[e] - srow1[e - 1]) + preFilterCap));
		}
	}

	static void censusTrans8U_3x3(Mat& src, Mat& dest)
	{
		if (dest.empty() || dest.depth() != CV_8U)dest.create(src.size(), CV_8U);
		const int r = 1;
		const int r2 = 2 * r;
		Mat im; copyMakeBorder(src, im, r, r, r, r, cv::BORDER_REFLECT101);

		uchar* s = im.ptr<uchar>(r); s += r;
		uchar* d = dest.ptr<uchar>(0);
		uchar* sb;

		const int step1 = -r - im.cols;
		const int step2 = -3 + im.cols;
		const int w = src.cols;
		const int h = src.rows;
		for (int j = 0; j < h; j++)
		{
			for (int i = 0; i < w; i++)
			{
				uchar val = 0;//init value
				sb = s + step1;

				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				sb++;//skip r=0
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++;

				*(d++) = val;
				s++;
			}
			s += r2;
		}
	}

	static void censusTrans32S_5x5(Mat& src, Mat& dest)
	{
		if (dest.empty() || dest.depth() != CV_32S)dest.create(src.size(), CV_32S);
		const int r = 2;
		const int r2 = 2 * r;
		const int D = 2 * r + 1;
		Mat im; copyMakeBorder(src, im, r, r, r, r, cv::BORDER_REFLECT101);

		uchar* sb;//around
		uchar* s = im.ptr<uchar>(r); s += r;
		int* d = dest.ptr<int>();
		const int step1 = -r - r * im.cols;
		const int step2 = -D + im.cols;
		const int w = src.cols;
		const int h = src.rows;
		for (int j = 0; j < h; j++)
		{
			for (int i = 0; i < w; i++)
			{
				int val = 0;//init value
				sb = s + step1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				sb++;//skip r=0
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++;

				*(d++) = val;
				s++;
			}
			s += r2;
		}
	}

	static void censusTrans32S_7x5(Mat& src, Mat& dest)
	{
		if (dest.empty() || dest.depth() != CV_32S)dest.create(src.size(), CV_32S);
		const int r = 3;
		const int r2 = 2 * r;
		const int D = 2 * r + 1;
		const int vr = 2;
		Mat im; copyMakeBorder(src, im, vr, vr, r, r, cv::BORDER_REFLECT101);

		uchar* s = im.ptr<uchar>(vr); s += r;
		uchar* sb;
		int* d = dest.ptr<int>();

		const int step1 = -r - vr * im.cols;
		const int step2 = -D + im.cols;
		const int w = src.cols;
		const int h = src.rows;
		for (int j = 0; j < h; j++)
		{
			for (int i = 0; i < w; i++)
			{
				int val = 0;//init value
				sb = s + step1;
				sb++;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				sb++;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				sb++;//skip r=0
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

				sb += step2;
				sb++;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
				val = (*sb < *s) ? val | 1 : val; sb++;
				sb++;

				*(d++) = val;
				s++;
			}
			s += r2;
		}
	}

	static void censusTrans8U_9x1(Mat& src, Mat& dest)
	{
		if (dest.empty() || dest.depth() != CV_8U)dest.create(src.size(), CV_8U);
		const int w = src.cols;
		const int h = src.rows;

		const int r = 4;
		const int r2 = 2 * r;
		Mat im; copyMakeBorder(src, im, 0, 0, r, r, cv::BORDER_REPLICATE);

		uchar* s = im.ptr<uchar>(0); s += r;
		uchar* d = dest.ptr<uchar>(0);
		uchar val;//init value
		uchar* ss;
		for (int j = 0; j < h; j++)
		{
			for (int i = 0; i < w; i++)
			{
				val = 0;//init value
				ss = s - r;
				val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
				val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
				val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
				val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
				ss++;//skip r=0
				val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
				val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
				val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
				val = (*ss < *s) ? val | 1 : val; ss++;

				*d = val;
				d++;
				s++;
			}
			s += r2;
		}
	}

	//0: gray image, 1: sobel/CENSUS image
	void StereoBase::prefilter(Mat& targetImage, Mat& referenceImage)
	{
		const bool isColor = (PixelMatchingMethod % 2 == 1);
		if (isColor)
		{
			if (targetImage.channels() != 3 || referenceImage.channels() != 3)
			{
				std::cout << "input image must have 3 channels" << std::endl;
				CV_Assert(targetImage.channels() == 3);
				CV_Assert(referenceImage.channels() == 3);
			}

			target.resize(6);
			reference.resize(6);
			vector<Mat> temp;
			split(targetImage, temp);
			temp[0].copyTo(target[0]);
			temp[1].copyTo(target[2]);
			temp[2].copyTo(target[4]);
			split(referenceImage, temp);
			temp[0].copyTo(reference[0]);
			temp[1].copyTo(reference[2]);
			temp[2].copyTo(reference[4]);

			if (PixelMatchingMethod == Pixel_Matching_CENSUS3x3Color)
			{
				censusTrans8U_3x3(target[0], target[1]);
				censusTrans8U_3x3(reference[0], reference[1]);
				censusTrans8U_3x3(target[2], target[3]);
				censusTrans8U_3x3(reference[2], reference[3]);
				censusTrans8U_3x3(target[4], target[5]);
				censusTrans8U_3x3(reference[4], reference[5]);
			}
			else if (PixelMatchingMethod == Pixel_Matching_CENSUS9x1Color)
			{
				censusTrans8U_9x1(target[0], target[1]);
				censusTrans8U_9x1(reference[0], reference[1]);
				censusTrans8U_9x1(target[2], target[3]);
				censusTrans8U_9x1(reference[2], reference[3]);
				censusTrans8U_9x1(target[4], target[5]);
				censusTrans8U_9x1(reference[4], reference[5]);
			}
			else if (PixelMatchingMethod == Pixel_Matching_CENSUS5x5Color)
			{
				censusTrans32S_5x5(target[0], target[1]);
				censusTrans32S_5x5(reference[0], reference[1]);
				censusTrans32S_5x5(target[2], target[3]);
				censusTrans32S_5x5(reference[2], reference[3]);
				censusTrans32S_5x5(target[4], target[5]);
				censusTrans32S_5x5(reference[4], reference[5]);
			}
			else if (PixelMatchingMethod == Pixel_Matching_CENSUS7x5Color)
			{
				censusTrans32S_7x5(target[0], target[1]);
				censusTrans32S_7x5(reference[0], reference[1]);
				censusTrans32S_7x5(target[2], target[3]);
				censusTrans32S_7x5(reference[2], reference[3]);
				censusTrans32S_7x5(target[4], target[5]);
				censusTrans32S_7x5(reference[4], reference[5]);
			}
			else
			{
				prefilterXSobel(target[0], target[1], preFilterCap);
				prefilterXSobel(reference[0], reference[1], preFilterCap);
				prefilterXSobel(target[2], target[3], preFilterCap);
				prefilterXSobel(reference[2], reference[3], preFilterCap);
				prefilterXSobel(target[4], target[5], preFilterCap);
				prefilterXSobel(reference[4], reference[5], preFilterCap);
			}
		}
		else
		{
			target.resize(2);
			reference.resize(2);

			if (targetImage.channels() == 3) cvtColor(targetImage, target[0], COLOR_BGR2GRAY);
			else targetImage.copyTo(target[0]);
			if (referenceImage.channels() == 3) cvtColor(referenceImage, reference[0], COLOR_BGR2GRAY);
			else referenceImage.copyTo(reference[0]);

			if (PixelMatchingMethod == Pixel_Matching_CENSUS3x3)
			{
				censusTrans8U_3x3(target[0], target[1]);
				censusTrans8U_3x3(reference[0], reference[1]);
			}
			else if (PixelMatchingMethod == Pixel_Matching_CENSUS9x1)
			{
				censusTrans8U_9x1(target[0], target[1]);
				censusTrans8U_9x1(reference[0], reference[1]);
			}
			else if (PixelMatchingMethod == Pixel_Matching_CENSUS5x5)
			{
				censusTrans32S_5x5(target[0], target[1]);
				censusTrans32S_5x5(reference[0], reference[1]);
			}
			else if (PixelMatchingMethod == Pixel_Matching_CENSUS7x5)
			{
				censusTrans32S_7x5(target[0], target[1]);
				censusTrans32S_7x5(reference[0], reference[1]);
			}
			else
			{
				prefilterXSobel(target[0], target[1], preFilterCap);
				prefilterXSobel(reference[0], reference[1], preFilterCap);
			}
		}
	}

	void StereoBase::textureAlpha(Mat& src, Mat& dest, const int th1, const int th2, const int r)
	{
		if (dest.empty())dest.create(src.size(), CV_8U);
		Mat temp;
		Sobel(src, temp, CV_16S, 1, 0);
		Mat temp2;

		convertScaleAbs(temp, temp2);
		//maxFilter(temp2,temp2,Size(2*r+1,2*r+1));
		//blur(temp2,temp2,Size(2*r+1,2*r+1));
		temp2 -= th1;

		max(temp2, th2, temp2);
		Mat temp33;
		temp2.convertTo(temp33, CV_32F, 1.0 / (double)th2);
		multiply(temp33, temp33, temp33, 1);
		temp33.convertTo(temp2, CV_8U, 255.0 / th2);
		//temp2*=255.0/(double)th2;
		//blur(temp2,temp2,Size(2*r+1,2*r+1));
		//threshold(temp2,dest,th,255,cv::THRESH_BINARY);

		maxFilter(temp2, dest, Size(aggregationRadiusH, 2 * r + 1));

		//blur(temp2,dest,size(2*r+1,2*r+1));
		//dest*=1.5;
		//GaussianBlur(dest,dest,Size(2*r+1,2*r+1),r/1.5);

		//imshow("texture", dest);
	}

	inline __m256i _mm256_squared_distance_epu8(__m256i src1, __m256i src2)
	{
		__m256i s1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(src1));
		__m256i s2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(src2));
		__m256i sub = _mm256_sub_epi16(s1, s2);
		__m128i d1 = _mm256_cvtepi16_epu8(_mm256_mullo_epi16(sub, sub));
		s1 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(src1));
		s2 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(src2));
		sub = _mm256_sub_epi16(s1, s2);
		__m128i d2 = _mm256_cvtepi16_epu8(_mm256_mullo_epi16(sub, sub));
		return _mm256_set_m128i(d2, d1);
	}

	static void SDTruncate_8UC1(const Mat& src1, const Mat& src2, const int disparity, uchar thresh, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);

		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < disparity; i++)
				{
					d[i] = std::min((s1[i] - s2[0]) * (s1[i] - s2[0]), (int)thresh);
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));
					_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_squared_distance_epu8(a, b), mtruncation));
				}
				for (int i = WIDTH; i < src1.cols; i++)
				{
					d[i] = std::min((s1[i] - s2[i - disparity]) * (s1[i] - s2[i - disparity]), (int)thresh);
				}
			}
		}
	}

	static void SDTruncateBlend_8UC1(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const int disparity, const uchar thresh, const float alpha, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);

		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		const int alpha_int = cvRound(255 * alpha);
		const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				const uchar* s3 = src3.ptr<uchar>(j);
				const uchar* s4 = src4.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < disparity; i++)
				{
					d[i] = saturate_cast<uchar>(alpha * std::min((s1[i] - s2[0]) * (s1[i] - s2[0]), (int)thresh)
						+ (1 - alpha) * std::min((s3[i] - s4[0]) * (s3[i] - s4[0]), (int)thresh));
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));
					a = _mm256_min_epu8(_mm256_squared_distance_epu8(a, b), mtruncation);

					b = _mm256_loadu_si256((__m256i*)(s3 + i));
					__m256i c = _mm256_loadu_si256((__m256i*)(s4 - disparity + i));
					b = _mm256_min_epu8(_mm256_squared_distance_epu8(b, c), mtruncation);

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(a, b, ma));
				}
				for (int i = WIDTH; i < src1.cols; i++)
				{
					d[i] = saturate_cast<uchar>(alpha * std::min((s1[i] - s2[i - disparity]) * (s1[i] - s2[i - disparity]), (int)thresh)
						+ (1.f - alpha) * std::min((s3[i] - s4[i - disparity]) * (s3[i] - s4[i - disparity]), (int)thresh));
				}
			}
		}
	}

	static void ADTruncate_8UC1(const Mat& src1, const Mat& src2, const int disparity, const uchar thresh, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);

		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < disparity; i++)
				{
					d[i] = std::min((uchar)abs(s1[i] - s2[0]), thresh);
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_adds_epu8(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a)), mtruncation));
				}
				for (int i = WIDTH; i < src1.cols; i++)
				{
					d[i] = std::min((uchar)abs(s1[i] - s2[i - disparity]), thresh);
				}
			}
		}
	}

	static void ADTruncateBlend_8UC1(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const int disparity, const uchar thresh, const float alpha, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);

		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		const int alpha_int = cvRound(255 * alpha);
		const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				const uchar* s3 = src3.ptr<uchar>(j);
				const uchar* s4 = src4.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < disparity; i++)
				{
					d[i] = saturate_cast<uchar>(alpha * std::min((uchar)abs(s1[i] - s2[0]), thresh)
						+ (1 - alpha) * std::min((uchar)abs(s3[i] - s4[0]), thresh));
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));
					a = _mm256_min_epu8(_mm256_adds_epu8(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a)), mtruncation);

					b = _mm256_loadu_si256((__m256i*)(s3 + i));
					__m256i c = _mm256_loadu_si256((__m256i*)(s4 - disparity + i));
					b = _mm256_min_epu8(_mm256_adds_epu8(_mm256_subs_epu8(b, c), _mm256_subs_epu8(c, b)), mtruncation);

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(a, b, ma));
				}
				for (int i = WIDTH; i < src1.cols; i++)
				{
					d[i] = saturate_cast<uchar>(alpha * std::min((uchar)abs(s1[i] - s2[i - disparity]), thresh)
						+ (1.f - alpha) * std::min((uchar)abs(s3[i] - s4[i - disparity]), thresh));
				}
			}
		}
	}

	static void BTTruncate_8UC1(const Mat& src1, const Mat& src2, const int disparity, const uchar thresh, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);
		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		const int DISPARITY = get_simd_floor(disparity, 32);

		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < DISPARITY; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + 0));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
					__m256i v2x = _mm256_max_epu8(p2, v2);
					__m256i v2n = _mm256_min_epu8(p2, v2);
					_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation));
				}
				for (int i = DISPARITY; i < disparity; i++)
				{
					int v1 = s1[i];
					int v2 = s2[0];
					int p2 = (s2[1] + v2) >> 1;
					int v2x = max(v2, p2);
					int v2n = min(v2, p2);
					int a = max(0, max(v1 - v2x, v2n - v1));
					d[i] = min(a, (int)thresh);
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
					__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
					__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
					__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);
					_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation));
				}
				for (int i = WIDTH; i < src1.cols; i++)
				{
					uchar v1 = s1[i];
					uchar v2 = (s2[i - disparity + 0]);
					uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
					uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
					uchar v2x = max(max(m2, p2), v2);
					uchar v2n = min(min(m2, p2), v2);

					uchar a = max(0, max(v1 - v2x, v2n - v1));
					d[i] = min(a, thresh);
				}
			}
		}
	}

	static void BTTruncateBlend_8UC1(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const int disparity, uchar thresh, const float alpha, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);
		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		const int DISPARITY = get_simd_floor(disparity, 32);
		const int alpha_int = cvRound(255 * alpha);
		const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				const uchar* s3 = src3.ptr<uchar>(j);
				const uchar* s4 = src4.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < DISPARITY; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
					__m256i v2x = _mm256_max_epu8(p2, v2);
					__m256i v2n = _mm256_min_epu8(p2, v2);

					__m256i d1 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

					v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
					v2 = _mm256_loadu_si256((__m256i*)(s4));
					p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + 1)));
					v2x = _mm256_max_epu8(p2, v2);
					v2n = _mm256_min_epu8(p2, v2);

					__m256i d2 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
				}
				//for (int i = 0; i < disparity; i++)
				for (int i = DISPARITY; i < disparity; i++)
				{
					uchar v1 = s1[i];
					uchar v2 = (s2[0]);
					uchar p2 = (s2[1] + v2) >> 1;
					uchar v2x = max(p2, v2);
					uchar v2n = min(p2, v2);

					uchar d1 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);

					v1 = s3[i];
					v2 = (s4[0]);
					p2 = (s4[1] + v2) >> 1;
					v2x = max(p2, v2);
					v2n = min(p2, v2);
					uchar d2 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);
					d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
					__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
					__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
					__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

					__m256i d1 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

					v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
					v2 = _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 0));
					p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 1)));
					m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity - 1)));
					v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
					v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

					__m256i d2 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
				}
				//for (int i = disparity; i < src1.cols; i++)
				for (int i = WIDTH; i < src1.cols; i++)
				{
					uchar v1 = s1[i];
					uchar v2 = (s2[i - disparity + 0]);
					uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
					uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
					uchar v2x = max(max(m2, p2), v2);
					uchar v2n = min(min(m2, p2), v2);

					uchar d1 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);

					v1 = s3[i];
					v2 = (s4[i - disparity + 0]);
					p2 = (s4[i - disparity + 1] + v2) >> 1;
					m2 = (s4[i - disparity - 1] + v2) >> 1;
					v2x = max(max(m2, p2), v2);
					v2n = min(min(m2, p2), v2);
					uchar d2 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);
					d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
				}
			}
		}
	}

	static void BTFullTruncate_8UC1(const Mat& src1, const Mat& src2, const int disparity, const uchar thresh, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);

		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		const int DISPARITY = get_simd_floor(disparity, 32);

		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < DISPARITY; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
					__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + max(i - 1, 0))));
					__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
					__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + 0));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
					__m256i v2x = _mm256_max_epu8(p2, v2);
					__m256i v2n = _mm256_min_epu8(p2, v2);

					__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
					__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation));
				}
				//for (int i = 0; i < disparity; i++)
				for (int i = DISPARITY; i < disparity; i++)
				{
					uchar v1 = s1[i];
					uchar p1 = (s1[i + 1] + v1) >> 1;
					uchar m1 = (s1[max(i - 1, 0)] + v1) >> 1;
					uchar v1x = max(max(m1, p1), v1);
					uchar v1n = min(min(m1, p1), v1);

					uchar v2 = s2[0];
					uchar p2 = (s2[1] + v2) >> 1;
					uchar v2x = max(p2, v2);
					uchar v2n = min(p2, v2);

					uchar a = max(0, max(v1 - v2x, v2n - v1));
					uchar b = max(0, max(v2 - v1x, v1n - v2));

					d[i] = min(min(a, b), thresh);
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
					__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
					__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i - 1)));
					__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
					__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
					__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
					__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
					__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

					__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
					__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation));
				}
				//for (int i = disparity; i < src1.cols; i++)
				for (int i = WIDTH; i < src1.cols; i++)
				{
					uchar v1 = s1[i];
					uchar p1 = (s1[i + 1] + v1) >> 1;
					uchar m1 = (s1[i - 1] + v1) >> 1;
					uchar v1x = max(max(m1, p1), v1);
					uchar v1n = min(min(m1, p1), v1);

					uchar v2 = (s2[i - disparity + 0]);
					uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
					uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
					uchar v2x = max(max(m2, p2), v2);
					uchar v2n = min(min(m2, p2), v2);

					uchar a = max(0, max(v1 - v2x, v2n - v1));
					uchar b = max(0, max(v2 - v1x, v1n - v2));

					d[i] = min(min(a, b), thresh);
				}
			}
		}
	}

	static void BTFullTruncateBlend_8UC1(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const int disparity, uchar thresh, const float alpha, Mat& dest)
	{
		dest.create(src1.size(), CV_8U);
		const __m256i mtruncation = _mm256_set1_epi8(thresh);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		const int DISPARITY = get_simd_floor(disparity, 32);
		const int alpha_int = cvRound(255 * alpha);
		const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

		if (disparity >= 0)
		{
			for (int j = 0; j < src1.rows; j++)
			{
				const uchar* s1 = src1.ptr<uchar>(j);
				const uchar* s2 = src2.ptr<uchar>(j);
				const uchar* s3 = src3.ptr<uchar>(j);
				const uchar* s4 = src4.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < DISPARITY; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i + 0));
					__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
					__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + max(i - 1, 0))));
					__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
					__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + 0));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
					__m256i v2x = _mm256_max_epu8(p2, v2);
					__m256i v2n = _mm256_min_epu8(p2, v2);

					__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
					__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
					__m256i d1 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

					v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
					p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i + 1)));
					m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i - 1)));
					v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
					v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

					v2 = _mm256_loadu_si256((__m256i*)(s4));
					p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + 1)));
					v2x = _mm256_max_epu8(p2, v2);
					v2n = _mm256_min_epu8(p2, v2);

					a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
					b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
					__m256i d2 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
				}
				//for (int i = 0; i < disparity; i++)
				for (int i = DISPARITY; i < disparity; i++)
				{

					uchar v1 = (s1[i + 0]);
					uchar p1 = (s1[i + 1] + v1) >> 1;
					uchar m1 = (s1[max(i - 1, 0)] + v1) >> 1;
					uchar v1x = max(max(m1, p1), v1);
					uchar v1n = min(min(m1, p1), v1);

					uchar v2 = (s2[0]);
					uchar p2 = (s2[1] + v2) >> 1;
					uchar v2x = max(p2, v2);
					uchar v2n = min(p2, v2);

					uchar a = max(0, max(v1 - v2x, v2n - v1));
					uchar b = max(0, max(v2 - v1x, v1n - v2));
					uchar d1 = std::min((uchar)std::min(a, b), thresh);

					v1 = s3[i];
					p1 = (s3[i + 1] + v1) >> 1;
					m1 = (s3[i - 1] + v1) >> 1;
					v1x = max(max(m1, p1), v1);
					v1n = min(min(m1, p1), v1);

					v2 = (s4[0]);
					p2 = (s4[1] + v2) >> 1;
					v2x = max(p2, v2);
					v2n = min(p2, v2);

					a = max(0, max(v1 - v2x, v2n - v1));
					b = max(0, max(v2 - v1x, v1n - v2));
					uchar d2 = std::min((uchar)std::min(a, b), thresh);

					d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
				}
				for (int i = disparity; i < WIDTH; i += 32)
				{
					__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i + 0));
					__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
					__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i - 1)));
					__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
					__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

					__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
					__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
					__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
					__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
					__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

					__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
					__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
					__m256i d1 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

					v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
					p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i + 1)));
					m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i - 1)));
					v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
					v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

					v2 = _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 0));
					p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 1)));
					m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity - 1)));
					v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
					v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

					a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
					b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
					__m256i d2 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

					_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
				}
				//for (int i = disparity; i < src1.cols; i++)
				for (int i = WIDTH; i < src1.cols; i++)
				{
					uchar v1 = (s1[i + 0]);
					uchar p1 = (s1[i + 1] + v1) >> 1;
					uchar m1 = (s1[i - 1] + v1) >> 1;
					uchar v1x = max(max(m1, p1), v1);
					uchar v1n = min(min(m1, p1), v1);

					uchar v2 = (s2[i - disparity + 0]);
					uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
					uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
					uchar v2x = max(max(m2, p2), v2);
					uchar v2n = min(min(m2, p2), v2);

					uchar a = max(0, max(v1 - v2x, v2n - v1));
					uchar b = max(0, max(v2 - v1x, v1n - v2));
					uchar d1 = std::min((uchar)std::min(a, b), thresh);

					v1 = s3[i];
					p1 = (s3[i + 1] + v1) >> 1;
					m1 = (s3[i - 1] + v1) >> 1;
					v1x = max(max(m1, p1), v1);
					v1n = min(min(m1, p1), v1);
					v2 = (s4[i - disparity + 0]);
					p2 = (s4[i - disparity + 1] + v2) >> 1;
					m2 = (s4[i - disparity - 1] + v2) >> 1;
					v2x = max(max(m2, p2), v2);
					v2n = min(min(m2, p2), v2);

					a = max(0, max(v1 - v2x, v2n - v1));
					b = max(0, max(v2 - v1x, v1n - v2));
					uchar d2 = std::min((uchar)std::min(a, b), thresh);

					d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
				}
			}
		}
	}

	//simple stereo matching implementaion
	template<typename srcType>
	static void HammingDistance32S_8UC1(Mat& src1, Mat& src2, const int disparity, Mat& dest)
	{
		if (dest.empty())dest.create(src1.size(), CV_8U);
		const int WIDTH = get_simd_floor(src1.cols, 32);
		const int h = src1.rows;

		for (int j = 0; j < h; j++)
		{
			srcType* s1 = src1.ptr<srcType>(j);
			srcType* s2 = src2.ptr<srcType>(j);
			uchar* d = dest.ptr<uchar>(j);
			for (int i = 0; i < disparity; i++)
			{
				d[i] = _mm_popcnt_u32((s1[i] ^ s2[0]));

			}
			/* for AVX512
						for (int i = disparity; i < WIDTH; i += 32)
						{
							__m256i ms1 = _mm256_loadu_si256((__m256i*)(s1+i));
							__m256i ms2 = _mm256_loadu_si256((__m256i*)(s2+i-disparity));
							__m256i md = _mm256_xor_si256(ms1, ms2);
							_mm256_storeu_si256((__m256i*)(d+i), _mm256_popcnt_epi8(md));
							//_mm256_storeu_si256((__m256i*)(d + i), md);
						}
						for (int i = WIDTH; i < src1.cols; i++)
							*/
			for (int i = disparity; i < src1.cols; i++)
			{
				d[i] = _mm_popcnt_u32((s1[i] ^ s2[i - disparity]));
			}
		}

	}

	void StereoBase::getPixelMatchingCostADAlpha(vector<Mat>& t, vector<Mat>& r, Mat& alpha, const int d, Mat& dest)
	{
		;
	}

	void StereoBase::getPixelMatchingCostBTAlpha(vector<Mat>& target, vector<Mat>& refference, Mat& alpha, const int d, Mat& dest)
	{
		;
	}

	string StereoBase::getPixelMatchingMethodName(int method)
	{
		string mes;
		switch (method)
		{
		case Pixel_Matching_SD:						mes = "SD"; break;
		case Pixel_Matching_SDSobel:				mes = "SDSobel"; break;
		case Pixel_Matching_SDSobelBlend:			mes = "SDSobelBlend"; break;
		case Pixel_Matching_AD:						mes = "AD"; break;
		case Pixel_Matching_ADSobel:				mes = "ADSobel"; break;
		case Pixel_Matching_ADSobelBlend:			mes = "ADSobelBlend"; break;
		case Pixel_Matching_BT:						mes = "BT"; break;
		case Pixel_Matching_BTSobel:				mes = "BTSobel"; break;
		case Pixel_Matching_BTSobelBlend:			mes = "BTSobelBlend"; break;
		case Pixel_Matching_BTFull:					mes = "BTFull"; break;
		case Pixel_Matching_BTFullSobel:			mes = "BTFullSobel"; break;
		case Pixel_Matching_BTFullSobelBlend:		mes = "BTFullSobelBlend"; break;
		case Pixel_Matching_CENSUS3x3:				mes = "CENSUS3x3"; break;
		case Pixel_Matching_CENSUS5x5:				mes = "CENSUS5x5"; break;
		case Pixel_Matching_CENSUS7x5:				mes = "CENSUS7x5"; break;
		case Pixel_Matching_CENSUS9x1:				mes = "CENSUS9x1"; break;

		case Pixel_Matching_SDColor:				mes = "SDColor"; break;
		case Pixel_Matching_SDSobelColor:			mes = "SDSobelColor"; break;
		case Pixel_Matching_SDSobelBlendColor:		mes = "SDSobelBlendColor"; break;
		case Pixel_Matching_ADColor:				mes = "ADColor"; break;
		case Pixel_Matching_ADSobelColor:			mes = "ADSobelColor"; break;
		case Pixel_Matching_ADSobelBlendColor:		mes = "ADSobelBlendColor"; break;
		case Pixel_Matching_BTColor:				mes = "BTColor"; break;
		case Pixel_Matching_BTSobelColor:			mes = "BTSobelColor"; break;
		case Pixel_Matching_BTSobelBlendColor:		mes = "BTSobelBlendColor"; break;
		case Pixel_Matching_BTFullColor:			mes = "BTFullColor"; break;
		case Pixel_Matching_BTFullSobelColor:		mes = "BTFullSobelColor"; break;
		case Pixel_Matching_BTFullSobelBlendColor:	mes = "BTFullSobelBlendColor"; break;
		case Pixel_Matching_CENSUS3x3Color:			mes = "CENSUS3x3Color"; break;
		case Pixel_Matching_CENSUS5x5Color:			mes = "CENSUS5x5Color"; break;
		case Pixel_Matching_CENSUS7x5Color:			mes = "CENSUS7x5Color"; break;
		case Pixel_Matching_CENSUS9x1Color:			mes = "CENSUS9x1Color"; break;

			//case Pixel_Matching_SAD_TextureBlend:	mes = "Cost_Computation_SAD_TextureBlend"; break;
			//case Pixel_Matching_BT_TextureBlend:	mes = "Cost_Computation_BTTextureBlend"; break;
		default:								mes = "This cost computation method is not supported"; break;
		}
		return mes;
	}

	void StereoBase::getPixelMatchingCost(const int d, Mat& dest)
	{
		//gray
		if (PixelMatchingMethod == Pixel_Matching_SD)
		{
			SDTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_SDSobel)
		{
			SDTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_SDSobelBlend)
		{
			SDTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_AD)
		{
			ADTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_ADSobel)
		{
			ADTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_ADSobelBlend)
		{
			ADTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BT)
		{
			BTTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTSobel)
		{
			BTTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTSobelBlend)
		{
			BTTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTFull)
		{
			BTFullTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTFullSobel)
		{
			BTFullTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTFullSobelBlend)
		{
			BTFullTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_CENSUS3x3 || PixelMatchingMethod == Pixel_Matching_CENSUS9x1)
		{
			HammingDistance32S_8UC1<uchar>(target[1], reference[1], d, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_CENSUS5x5 || PixelMatchingMethod == Pixel_Matching_CENSUS7x5)
		{
			HammingDistance32S_8UC1<int>(target[1], reference[1], d, dest);
		}

		//color
		else if (PixelMatchingMethod == Pixel_Matching_SDColor)
		{
			Mat temp;
			SDTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
			SDTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			SDTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_SDSobelColor)
		{
			Mat temp;
			SDTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
			SDTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			SDTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_SDSobelBlendColor)
		{
			Mat temp;
			SDTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
			SDTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
			SDTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_ADColor)
		{
			Mat temp;
			ADTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
			ADTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			ADTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_ADSobelColor)
		{
			Mat temp;
			ADTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
			ADTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			ADTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_ADSobelBlendColor)
		{
			Mat temp;
			ADTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
			ADTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
			ADTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTColor)
		{
			Mat temp;
			BTTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
			BTTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			BTTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTSobelColor)
		{
			Mat temp;
			BTTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
			BTTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			BTTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTSobelBlendColor)
		{
			Mat temp;
			BTTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
			BTTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
			BTTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTFullColor)
		{
			Mat temp;
			BTFullTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
			BTFullTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			BTFullTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTFullSobelColor)
		{
			Mat temp;
			BTFullTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
			BTFullTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
			BTFullTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BTFullSobelBlendColor)
		{
			Mat temp;
			BTFullTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
			BTFullTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
			BTFullTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_CENSUS3x3Color || PixelMatchingMethod == Pixel_Matching_CENSUS9x1Color)
		{
			Mat temp;
			HammingDistance32S_8UC1<uchar>(target[1], reference[1], d, dest);
			HammingDistance32S_8UC1<uchar>(target[3], reference[3], d, temp);
			add(dest, temp, dest);
			HammingDistance32S_8UC1<uchar>(target[5], reference[5], d, temp);
			add(dest, temp, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_CENSUS5x5Color || PixelMatchingMethod == Pixel_Matching_CENSUS7x5Color)
		{
			Mat temp;
			HammingDistance32S_8UC1<int>(target[1], reference[1], d, dest);
			HammingDistance32S_8UC1<int>(target[3], reference[3], d, temp);
			add(dest, temp, dest);
			HammingDistance32S_8UC1<int>(target[5], reference[5], d, temp);
			add(dest, temp, dest);
		}

		/*else if (PixelMatchingMethod == Pixel_Matching_SAD_TextureBlend)
		{
			Mat alpha;
			textureAlpha(target[0], alpha, sobelBlendMapParam2, sobelBlendMapParam1, sobelBlendMapParam_Size);
			getPixelMatchingCostSADAlpha(target, reference, alpha, d, dest);
		}
		else if (PixelMatchingMethod == Pixel_Matching_BT_TextureBlend)
		{
			Mat alpha;
			textureAlpha(target[0], alpha, sobelBlendMapParam2, sobelBlendMapParam1, sobelBlendMapParam_Size);
			getPixelMatchingCostBTAlpha(target, reference, alpha, d, dest);
		}*/
		else
		{
			cout << "This pixel matching method is not supported." << endl;
		}
	}

	inline float distance_functionL1(float diff, float clip)
	{
		return min(abs(diff), clip);
	}
	inline float distance_functionL2(float diff, float clip)
	{
		return min(diff * diff, clip * clip);
	}
	void StereoBase::addCostIterativeFeedback(cv::Mat& cost, const int current_disparity, const cv::Mat& disparity, const int functionType, const int clip, float amp)
	{
		CV_Assert(disparity.depth() == CV_16S);
		const short* dptr = disparity.ptr<short>();
		uchar* cptr = cost.ptr<uchar>();
		for (int i = 0; i < cost.size().area(); i++)
		{
			cptr[i] = saturate_cast<uchar>(cptr[i] + amp * distance_functionL1((float)current_disparity - dptr[i] / 16.f, (float)clip));
		}
	}

#pragma endregion

#pragma region cost aggregation

	string StereoBase::getAggregationMethodName(int method)
	{
		string mes;
		switch (method)
		{
		case Aggregation_Box:				mes = "Box"; break;
		case Aggregation_BoxShiftable:		mes = "BoxShiftable"; break;
		case Aggregation_Gauss:				mes = "Gauss"; break;
		case Aggregation_GaussShiftable:	mes = "GaussShiftable"; break;
		case Aggregation_Guided:			mes = "Guided"; break;
		case Aggregation_CrossBasedBox:		mes = "CrossBasedBox"; break;
		case Aggregation_Bilateral:			mes = "Bilateral"; break;

		default:							mes = "This cost computation method is not supported"; break;
		}
		return mes;
	}

	void StereoBase::getCostAggregation(Mat& src, Mat& dest, cv::InputArray guideImage, const bool isFeedback, cv::Mat& weightMap)
	{
		const float sigma_s_h = (float)aggregationSigmaSpace;
		const float sigma_s_v = (float)aggregationSigmaSpace;

		Mat guide = guideImage.getMat();
		if (aggregationRadiusH != 1)
		{
			//GaussianBlur(dsi,DSI[i],Size(SADWindowSize,SADWindowSize),3);
			Size kernelSize = Size(2 * aggregationRadiusH + 1, 2 * aggregationRadiusV + 1);

			if (AggregationMethod == Aggregation_Box)
			{
				boxFilter(src, dest, -1, kernelSize);
			}
			else if (AggregationMethod == Aggregation_BoxShiftable)
			{
				boxFilter(src, dest, -1, kernelSize);
				minFilter(dest, dest, aggregationShiftableKernel);
			}
			else if (AggregationMethod == Aggregation_Gauss)
			{
				if (isFeedback && !weightMap.empty())
				{
					Mat a, b;
					multiply(src, weightMap, a, 1.0, CV_32F);
					GaussianBlur(a, a, kernelSize, sigma_s_h, sigma_s_v);
					GaussianBlur(weightMap, b, kernelSize, sigma_s_h, sigma_s_v);
					divide(a, b, dest, 1.0, CV_8U);
				}
				else
				{
					GaussianBlur(src, dest, kernelSize, sigma_s_h, sigma_s_v);
				}
			}
			else if (AggregationMethod == Aggregation_GaussShiftable)
			{
				GaussianBlur(src, dest, kernelSize, sigma_s_h, sigma_s_v);
				minFilter(dest, dest, aggregationShiftableKernel);
			}
			else if (AggregationMethod == Aggregation_Guided)
			{
				/*if (isFeedback && !weightMap.empty())
				{
					Mat a, b;
					multiply(src, weightMap, a, 1.0, CV_32F);
					gif[omp_get_thread_num()].filterGuidePrecomputed(a, guide, a, min(aggregationRadiusH, aggregationRadiusV), float(aggregationGuidedfilterEps * aggregationGuidedfilterEps), GuidedTypes::GUIDED_SEP_VHI_SHARE, ParallelTypes::NAIVE);
					gif[omp_get_thread_num()].filterGuidePrecomputed(weightMap, guide, b, min(aggregationRadiusH, aggregationRadiusV), float(aggregationGuidedfilterEps * aggregationGuidedfilterEps), GuidedTypes::GUIDED_SEP_VHI_SHARE, ParallelTypes::NAIVE);
					divide(a, b, dest, 1.0, CV_8U);
				}
				else*/
				{
					//guidedImageFilter(src, guide, dest, max(aggregationRadiusH, aggregationRadiusV), aggregationGuidedfilterEps * aggregationGuidedfilterEps,GuidedTypes::GUIDED_SEP_VHI, BoxTypes::BOX_OPENCV, ParallelTypes::NAIVE);
					//guidedImageFilter(src, guide, dest, max(aggregationRadiusH, aggregationRadiusV), aggregationGuidedfilterEps * aggregationGuidedfilterEps, GuidedTypes::GUIDED_SEP_VHI_SHARE, BoxTypes::BOX_OPENCV, ParallelTypes::NAIVE);
					gif[omp_get_thread_num()].filterGuidePrecomputed(src, guide, dest, min(aggregationRadiusH, aggregationRadiusV), float(aggregationGuidedfilterEps * aggregationGuidedfilterEps), GuidedTypes::GUIDED_SEP_VHI_SHARE, ParallelTypes::NAIVE);
					//guidedImageFilter(src, guide, dest, max(aggregationRadiusH, aggregationRadiusV), aggregationGuidedfilterEps * aggregationGuidedfilterEps, GuidedTypes::GUIDED_SEP_VHI_SHARE, BoxTypes::BOX_OPENCV, ParallelTypes::OMP);
				}
			}
			else if (AggregationMethod == Aggregation_CrossBasedBox)
			{
				clf(src, dest);
			}
			else if (AggregationMethod == Aggregation_Bilateral)
			{
				jointBilateralFilter(src, guide, dest, min(2 * aggregationRadiusH + 1, 2 * aggregationRadiusV + 1), aggregationGuidedfilterEps, sigma_s_h, FILTER_RECTANGLE);
			}
		}
		else
		{
			if (src.data != dest.data) src.copyTo(dest);
		}
	}

	//WTA and optimization
	void StereoBase::getOptScanline()
	{
		cout << "opt scan\n";
		Size size = DSI[0].size();
		Mat disp;

		Mat costMap = Mat::ones(size, CV_8U) * 255;
		const int imsize = size.area();
		//DSI[numberOfDisparities] = Mat::ones(DSI[0].size(),DSI[0].type())*192;
		//int* cost = new int[numberOfDisparities+1];
		//int* vd = new int[size.width+1];
		int cost[100];
		int vd[1000];
		{
			int j = 0;
			int pd = 1;
			for (int i = 0; i < size.width; i++)
			{
				for (int n = 1; n < numberOfDisparities - 1; n++)
				{
					cost[n] = (DSI[n].at<uchar>(j, i) += P2);
				}
				cost[pd] = (DSI[pd].at<uchar>(j, i) -= P2);
				cost[pd + 1] = (DSI[pd + 1].at<uchar>(j, i) -= (P2 - P1));
				cost[pd - 1] = (DSI[pd - 1].at<uchar>(j, i) -= (P2 - P1));

				int maxc = 65535;
				for (int n = 0; n < numberOfDisparities; n++)
				{
					if (cost[n] < maxc)
					{
						maxc = cost[n];
						pd = n;
					}
				}
				pd = max(pd, 1);
				pd = min(pd, numberOfDisparities - 2);
				vd[i] = pd;
			}
		}
		for (int j = 1; j < size.height; j++)
		{
			int pd;;
			{
				int maxc = 65535;
				for (int n = 0; n < numberOfDisparities; n++)
				{
					if (DSI[n].at<uchar>(j, 0) < maxc)
					{
						maxc = DSI[n].at<uchar>(j, 0);
						pd = n;
					}
				}
			}
			for (int i = 1; i < size.width; i++)
			{
				//nt apd = min(max(((pd + vd[i-1] + vd[i] + vd[i+1])>>2),1),numberOfDisparities-1);
				//int apd = min(max(vd[i],1),numberOfDisparities-1);
				//int apd = min(max(((2*pd + 2*vd[i])>>2),1),numberOfDisparities-1);
				int apd = min(max(pd, 1), numberOfDisparities - 2);
				for (int n = 0; n < numberOfDisparities; n++)
				{
					cost[n] = (DSI[n].at<uchar>(j, i) += P2);
				}

				cost[apd] = (DSI[apd].at<uchar>(j, i) -= P2);
				cost[apd + 1] = (DSI[apd + 1].at<uchar>(j, i) -= (P2 - P1));
				cost[apd - 1] = (DSI[apd - 1].at<uchar>(j, i) -= (P2 - P1));

				int maxc = 65535;
				for (int n = 0; n < numberOfDisparities; n++)
				{
					if (cost[n] < maxc)
					{
						maxc = cost[n];
						pd = n;
					}
				}
				vd[i] = pd;
			}
		}
		//delete[] cost;
		//delete[] vd;
	}

	void StereoBase::getWTA(vector<Mat>& dsi, Mat& dest, Mat& minimumCostMap)
	{
		const int imsize = dest.size().area();
		const int simdsize = get_simd_floor(imsize, 32);
#if 0
		for (int i = 0; i < numberOfDisparities; i++)
		{
			const short d = ((minDisparity + i) << 4);

			short* disp = dest.ptr<short>(0);
			uchar* pDSI = dsi[i].data;
			uchar* cost = minimumCostMap.data;

			const __m256i md = _mm256_set1_epi16(d);
			for (int j = simdsize; j -= 32; pDSI += 32, cost += 32, disp += 32)
			{
				__m256i mdsi = _mm256_load_si256((__m256i*) pDSI);
				__m256i mcost = _mm256_load_si256((__m256i*) cost);

				__m256i  mask = _mm256_cmpgt_epu8(mcost, mdsi);
				mcost = _mm256_blendv_epi8(mcost, mdsi, mask);
				_mm256_store_si256((__m256i*)cost, mcost);

				__m256i  mdisp = _mm256_load_si256((__m256i*) disp);
				_mm256_store_si256((__m256i*)disp, _mm256_blendv_epi8(mdisp, md, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(mask))));

				mdisp = _mm256_load_si256((__m256i*) (disp + 16));
				_mm256_store_si256((__m256i*)(disp + 16), _mm256_blendv_epi8(mdisp, md, _mm256_cvtepi8_epi16(_mm256_extractf128_si256(mask, 1))));
			}
		}
#else
		short* disparityMapPtr = dest.ptr<short>();
#pragma omp parallel for
		for (int i = 0; i < simdsize; i += 32)
		{
			__m256i mcost = _mm256_set1_epi8(255);
			__m256i mdisp1 = _mm256_setzero_si256();
			__m256i mdisp2 = _mm256_setzero_si256();
			for (int d = 0; d < numberOfDisparities; d++)
			{
				const short disp_val = ((minDisparity + d) << 4);
				uchar* pDSI = dsi[d].data;
				const __m256i md = _mm256_set1_epi16(disp_val);
				__m256i mdsi = _mm256_load_si256((__m256i*) (pDSI + i));
				__m256i  mask = _mm256_cmpgt_epu8(mcost, mdsi);
				mcost = _mm256_blendv_epi8(mcost, mdsi, mask);
				mdisp1 = _mm256_blendv_epi8(mdisp1, md, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(mask)));
				mdisp2 = _mm256_blendv_epi8(mdisp2, md, _mm256_cvtepi8_epi16(_mm256_extractf128_si256(mask, 1)));
			}
			uchar* cost = minimumCostMap.data;
			_mm256_store_si256((__m256i*)(cost + i), mcost);
			_mm256_store_si256((__m256i*)(disparityMapPtr + i), mdisp1);
			_mm256_store_si256((__m256i*)(disparityMapPtr + i + 16), mdisp2);
		}
		for (int i = simdsize; i < imsize; i++)
		{
			uchar mcost = 255;
			short mind = 0;

			for (int d = 0; d < numberOfDisparities; d++)
			{
				const short disp_val = ((minDisparity + d) << 4);
				uchar* pDSI = dsi[d].data;
				if (pDSI[i] < mcost)
				{
					mcost = pDSI[i];
					mind = d;
				}
			}
			uchar* cost = minimumCostMap.data;
			cost[i] = mcost;
			disparityMapPtr[i] = mind;
		}
#endif
	}
#pragma endregion

#pragma region post filter
	//post filter
	void StereoBase::uniquenessFilter(Mat& minCostMap, Mat& dest)
	{
		if (uniquenessRatio == 0)return;
		if (!isUniquenessFilter) return;

		const int imsize = dest.size().area();
		const int simdsize = get_simd_floor(imsize, 16);

		const float mul = 1.f + uniquenessRatio / 100.f;
		const __m256i mmul = _mm256_set1_epi16((short)((mul - 1.f) * pow(2, 15)));

		uchar* mincostPtr = minCostMap.data;
		short* destPtr = dest.ptr<short>();

		for (int d = 0; d < numberOfDisparities; d++)
		{
			const short disparity = ((minDisparity + d) << 4);
			uchar* DSIPtr = DSI[d].data;

#if 0
			//naive
			for (int i = 0; i < imsize; i++)
			{
				const int v = (mincostPtr[i] * mul);
				if ((*DSIPtr) < v && abs(disparity - destPtr[i]) > 16)
				{
					destPtr[i] = 0;//(minDisparity-1)<<4;
				}
			}
#else
			//simd
			const __m256i md = _mm256_set1_epi16(disparity);
			const __m256i m16 = _mm256_set1_epi16(16);
			for (int i = 0; i < simdsize; i += 16)
			{
				__m256i mc = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i*)(mincostPtr + i)));
				__m256i mv = _mm256_add_epi16(mc, _mm256_mulhrs_epi16(mc, mmul));
				__m256i mdsi = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i*)(DSIPtr + i)));
				__m256i mdest = _mm256_load_si256((__m256i*)(destPtr + i));

				__m256i mask1 = _mm256_cmpgt_epi16(mv, mdsi);
				__m256i mask2 = _mm256_cmpgt_epi16(_mm256_abs_epi16(_mm256_sub_epi16(md, mdest)), m16);
				mask1 = _mm256_and_si256(mask1, mask2);

				_mm256_store_si256((__m256i*)(destPtr + i), _mm256_blendv_epi8(mdest, _mm256_setzero_si256(), mask1));
			}
			for (int i = simdsize; i < imsize; i++)
			{
				int v = saturate_cast<int>(mincostPtr[i] * mul);
				if ((*DSIPtr) < v && abs(disparity - destPtr[i]) > 16)
				{
					destPtr[i] = 0;//(minDisparity-1)<<4;
				}
			}

#endif
		}
	}

	string StereoBase::getSubpixelInterpolationMethodName(const int method)
	{
		string mes;
		switch (method)
		{
		case SUBPIXEL_NONE:		mes = "Subpixel None"; break;
		case SUBPIXEL_QUAD:		mes = "Subpixel Quad"; break;
		case SUBPIXEL_LINEAR:	mes = "Subpixel Linear"; break;

		default:				mes = "This subpixel interpolation method is not supported"; break;
		}
		return mes;
	}

	void StereoBase::subpixelInterpolation(Mat& dest, const int method)
	{
		if (method == SUBPIXEL_NONE)return;

		short* disp = dest.ptr<short>(0);
		const int imsize = dest.size().area();
		if (method == SUBPIXEL_QUAD)
		{
			for (int j = 0; j < imsize; j++)
			{
				short d = disp[j] >> 4;
				int l = d - minDisparity;
				if (l<1 || l>numberOfDisparities - 2)
				{
					;
				}
				else
				{
					int f = DSI[l].data[j];
					int p = DSI[l + 1].data[j];
					int m = DSI[l - 1].data[j];

					int md = ((p + m - (f << 1)) << 1);
					if (md != 0)
					{
						const float dd = (float)d - (float)(p - m) / (float)md;
						disp[j] = (short)(16.f * dd + 0.5f);
						//disp[j] = saturate_cast<short>(16.0 * dd);
					}
				}
			}
		}
		else if (method == SUBPIXEL_LINEAR)
		{
			for (int j = 0; j < imsize; j++)
			{
				short d = disp[j] >> 4;
				int l = d - minDisparity;
				if (l<1 || l>numberOfDisparities - 2)
				{
					;
				}
				else
				{
					const double m1 = (double)DSI[l].data[j];
					const double m3 = (double)DSI[l + 1].data[j];
					const double m2 = (double)DSI[l - 1].data[j];
					const double m31 = m3 - m1;
					const double m21 = m2 - m1;
					double md;

					if (m2 > m3)
					{
						md = 0.5 - 0.25 * ((m31 * m31) / (m21 * m21) + m31 / m21);
					}
					else
					{
						md = -(0.5 - 0.25 * ((m21 * m21) / (m31 * m31) + m21 / m31));

					}

					disp[j] = (short)(16.0 * ((double)d + md) + 0.5);
				}
			}
		}
	}

	void StereoBase::fastLRCheck(Mat& dest)
	{
		Mat dispR = Mat::zeros(dest.size(), CV_16S);
		Mat disp8(dest.size(), CV_16S);
		//dest.convertTo(disp8,CV_16S,1.0/16,0.5);
		const int imsize = dest.size().area();

		short* dddd = dest.ptr<short>(0);
		short* d8 = disp8.ptr<short>(0);
		const double div = 1.0 / 16.0;
		for (int i = 0; i < dest.size().area(); i++)
		{
			d8[i] = (short)(dddd[i] * div + 0.5);
		}

		short* disp = disp8.ptr<short>(0);
		short* dispr = dispR.ptr<short>(0);


		disp += minDisparity + numberOfDisparities;
		dispr += minDisparity + numberOfDisparities;
		for (int j = imsize - (minDisparity + numberOfDisparities); j--; disp++, dispr++)
		{
			const short d = *disp;
			if (d != 0)
			{
				if (*(dispr - d) < *(disp))
				{
					*(dispr - d) = d;
				}
			}
		}
		LRCheckDisparity(disp8, dispR, 0, disp12diff + 1, 0, 1, LR_CHECK_DISPARITY_ONLY_L);
		//LRCheckDisparity(disp8,dispR,0,disp12diff,0,1,xcv::LR_CHECK_DISPARITY_BOTH);
		//Mat dr; dispR.convertTo(dr,CV_8U,4);imshow("dispR",dr);

		Mat mask;
		compare(disp8, 0, mask, cv::CMP_EQ);
		dest.setTo(0, mask);
	}

	void StereoBase::fastLRCheck(Mat& costMap, Mat& dest)
	{
		Mat dispR = Mat::zeros(dest.size(), CV_16S);
		Mat disp8(dest.size(), CV_16S);
		//dest.convertTo(disp8,CV_16S,1.0/16,0.5);
		const int imsize = dest.size().area();

		short* dddd = dest.ptr<short>(0);
		short* d8 = disp8.ptr<short>(0);
		const double div = 1.0 / 16.0;
		for (int i = 0; i < dest.size().area(); i++)
		{
			d8[i] = (short)(dddd[i] * div + 0.5);
		}
		if (isProcessLBorder)
		{
			const int maxval = minDisparity + numberOfDisparities;
			for (int j = 0; j < disp8.rows; j++)
			{
				short* dst = disp8.ptr<short>(j);
				memset(dst, 0, sizeof(short) * minDisparity + 1);
				for (int i = minDisparity + 1; i < maxval; i++)
				{
					dst[i] = (dst[i] >= i) ? 0 : dst[i];
				}
			}
		}

		Mat costMapR(dest.size(), CV_8U);
		costMapR.setTo(255);

		short* disp = disp8.ptr<short>(0);
		short* dispr = dispR.ptr<short>(0);

		uchar* cost = costMap.data;
		uchar* costr = costMapR.data;

		cost += minDisparity + numberOfDisparities;
		costr += minDisparity + numberOfDisparities;
		disp += minDisparity + numberOfDisparities;
		dispr += minDisparity + numberOfDisparities;
		for (int j = imsize - (minDisparity + numberOfDisparities); j--; cost++, costr++, disp++, dispr++)
		{
			const short d = *disp;
			if (d != 0)
			{
				if (*(cost) < *(costr - d))
				{
					*(costr - d) = *cost;
					*(dispr - d) = d;
				}
			}
		}
		LRCheckDisparity(disp8, dispR, 0, disp12diff, 0, 1, LR_CHECK_DISPARITY_ONLY_L);

		if (isProcessLBorder)
		{
			const int maxval = minDisparity + numberOfDisparities;
			for (int j = 0; j < disp8.rows; j++)
			{
				short* dst = disp8.ptr<short>(j);
				for (int i = minDisparity + 1; i < maxval; i++)
				{
					short d = dst[i];
					if (d != 0)
					{
						if (dst[i + d] == 0)dst[i] = 0;
					}
				}
			}
		}
		//LRCheckDisparity(disp8,dispR,0,disp12diff,0,1,xcv::LR_CHECK_DISPARITY_BOTH);
		//Mat dr; dispR.convertTo(dr,CV_8U,4);imshow("dispR",dr);

		Mat mask;
		compare(disp8, 0, mask, cv::CMP_EQ);
		dest.setTo(0, mask);
	}

	void StereoBase::minCostFilter(Mat& minCostMap, Mat& dest)
	{
		if (isMinCostFilter)
		{
			const int imsize = dest.size().area();
			Mat disptemp = dest.clone();
			const int step = dest.cols;
			short* disptempPtr = disptemp.ptr<short>(0);
			short* destPtr = dest.ptr<short>(0);
			uchar* costPtr = minCostMap.ptr<uchar>();
			//for(int j=0;j<imsize-step;j++)
			for (int j = 0; j < imsize; j++)
			{
				/*if( disp[0]!=0 && disp[step]!=0 && abs(disp[step]-disp[0])>=16)
				{
				if(cost[step]<cost[0])
				{
				disp[0]=disp2[step];
				}
				else
				{
				disp[step]=disp2[0];
				}
				}*/
				if (destPtr[0] != 0 && destPtr[-1] != 0 && abs(destPtr[-1] - destPtr[0]) >= 16)
				{
					if (costPtr[-1] < costPtr[0])
					{
						destPtr[0] = disptempPtr[-1];
					}
					else
					{
						destPtr[-1] = disptempPtr[0];
					}
				}

				destPtr++;
				disptempPtr++;
				costPtr++;
			}
		}
	}


	void StereoBase::getWeightUniqness(Mat& disp)
	{
		const int imsize = disp.size().area();
		Mat rmap = Mat::ones(disp.size(), CV_8U) * 255;
		for (int i = 0; i < numberOfDisparities; i++)
		{
			const short d = ((minDisparity + i) << 4);
			short* dis = disp.ptr<short>(0);
			uchar* pDSI = DSI[i].data;
			uchar* cost = costMap.data;
			uchar* r = rmap.data;
			for (int j = imsize; j--; pDSI++, cost++, dis++)
			{
				short dd = (*dis);
				int v = 1000 * (*cost);
				int u = (*pDSI) * (1000 - uniquenessRatio);
				if (u - v < 0 && abs(d - dd)>16)
				{
					//int vv = (abs((double)*cost-*pDSI)/(double)(error_truncate))*255.0;
					//cout<<abs((double)*cost-(double)*pDSI)<<","<<(abs((double)*cost-(double)*pDSI)/(double)(error_truncate))*255.0<<endl;
					if (*cost == *pDSI)*r = 0;
					//*r=min(vv,(int)*r);
				}

				r++;
			}
		}
		rmap.convertTo(weightMap, CV_32F, 1.0 / 255);
		Mat rshow;	applyColorMap(rmap, rshow, 2); imshow("rmap", rshow);
	}

	void StereoBase::refineFromCost(Mat& src, Mat& dest)
	{
		short* disp = src.ptr<short>(0);
		const int imsize = dest.size().area();
		for (int j = 0; j < imsize; j++)
		{
			short d = (short)(disp[j] / 16.0 + 0.5);
			int l = d - minDisparity;
			if (l<1 || l>numberOfDisparities - 2)
			{
				;
			}
			else
			{
				int f = DSI[l].data[j];
				int p = DSI[l + 1].data[j];
				int m = DSI[l - 1].data[j];
				if (f > p || f > m)
				{


					int md = ((p + m - (f << 1)) << 1);
					if (md != 0)
					{
						double dd = (double)d - (double)(p - m) / (double)md;
						disp[j] = (short)(16.0 * dd + 0.5);
					}
					//	disp[j]=0;
				}
			}
		}
	}

	template <class srcType>
	void singleDisparityLRCheck_(Mat& dest, double amp, int thresh, int minDisparity, int numberOfDisparities)
	{
		const int imsize = dest.size().area();
		Mat dispR = Mat::zeros(dest.size(), dest.type());
		Mat disp8(dest.size(), dest.type());


		srcType* dddd = dest.ptr<srcType>(0);
		srcType* d8 = disp8.ptr<srcType>(0);

		const double div = 1.0 / amp;
		if (amp != 1.0)
		{
			for (int i = 0; i < dest.size().area(); i++)
			{
				d8[i] = (srcType)(dddd[i] * div + 0.5);
			}
		}
		else
			dest.copyTo(disp8);

		srcType* disp = disp8.ptr<srcType>(0);
		srcType* dispr = dispR.ptr<srcType>(0);


		disp += minDisparity + numberOfDisparities;
		dispr += minDisparity + numberOfDisparities;
		for (int j = imsize - (minDisparity + numberOfDisparities); j--; disp++, dispr++)
		{
			const short d = *disp;
			if (d != 0)
			{
				if (*(dispr - d) < *(disp))
				{
					*(dispr - d) = d;
				}
			}
		}
		LRCheckDisparity(disp8, dispR, minDisparity + numberOfDisparities, thresh + 1, 0, 1, LR_CHECK_DISPARITY_ONLY_L);
		//LRCheckDisparity(disp8,dispR,0,thresh,0,1,xcv::LR_CHECK_DISPARITY_BOTH);
		//Mat dr; dispR.convertTo(dr,CV_8U,4);imshow("dispR",dr);

		Mat mask;
		compare(disp8, 0, mask, cv::CMP_EQ);
		dest.setTo(0, mask);

	}

	void singleDisparityLRCheck(Mat& dest, double amp, int thresh, int minDisparity, int numberOfDisparities)
	{

		if (dest.type() == CV_8U)
			singleDisparityLRCheck_<uchar>(dest, amp, thresh, minDisparity, numberOfDisparities);
		if (dest.type() == CV_16S)
			singleDisparityLRCheck_<short>(dest, amp, thresh, minDisparity, numberOfDisparities);
		if (dest.type() == CV_16U)
			singleDisparityLRCheck_<unsigned short>(dest, amp, thresh, minDisparity, numberOfDisparities);

	}

	template <class srcType>
	void correctDisparityBoundary(Mat& src, Mat& refimg, const int r, Mat& dest)
	{

		srcType invalidvalue = 0;
		Mat sobel;
		Mat ref;
		if (refimg.channels() == 3)cvtColor(refimg, ref, COLOR_BGR2GRAY);
		else ref = refimg.clone();
		medianBlur(ref, ref, 3);
		//blurRemoveMinMax(ref,ref,1,0);

		Sobel(ref, sobel, CV_16S, 1, 0);
		sobel = abs(sobel);

		int bb = 0;
		const int MAX_LENGTH = (int)(src.cols * 1.0);

		srcType* s = src.ptr<srcType>(0);
		const int step = src.cols;
		short* sbl = sobel.ptr<short>(0);
		Mat sobel2;
		sobel.convertTo(sobel2, CV_8U);
		//imshow("sbl",sobel2);

		for (int j = bb; j < src.rows - bb; j++)
		{
			s[0] = 255 * 16;//�\���̂���ő�l�����
			s[src.cols - 1] = 255 * 16;//�\���̂���ő�l�����
			//��������0��������l�̓����Ă���ߖT�̃s�N�Z���i�G�s�|�[������j�̍ŏ��l�Ŗ��߂�
			for (int i = 1; i < src.cols - 1; i++)
			{
				if (s[i] <= invalidvalue)
				{
					if (s[i + 1] > invalidvalue)
					{
						s[i] = min(s[i + 1], s[i - 1]);
						i++;
					}
					else
					{
						int t = i;
						do
						{
							t++;
							if (t > src.cols - 2)break;
						} while (s[t] <= invalidvalue);

						srcType maxd;
						srcType mind;
						if (s[i - 1] < s[t])
						{
							mind = s[i - 1];
							maxd = s[t];

							int maxp;
							int maxval = 0;
							for (int k = -r; k <= r; k++)
							{
								if (sbl[t + k] > maxval)
								{
									maxp = k;
									maxval = sbl[t + k];
								}
							}
							if (maxp >= 0)
							{
								for (; i < t + maxp; i++)
								{
									s[i] = mind;
								}
							}
							else
							{
								for (; i < t + maxp; i++)
								{
									s[i] = mind;
								}
								for (; i < t; i++)
								{
									s[i] = maxd;
								}
							}
						}
						else
						{
							mind = s[t];
							maxd = s[i - 1];

							int maxp;
							int maxval = 0;
							for (int k = -r; k <= r; k++)
							{
								if (sbl[i - 1 + k] > maxval)
								{
									maxp = k;
									maxval = sbl[i - 1 + k];
								}
							}
							if (maxp >= 0)
							{
								for (; i < maxp; i++)
								{
									s[i] = maxd;
								}
								for (; i < t; i++)
								{
									s[i] = mind;
								}
							}
							else
							{
								i += maxp;
								for (; i < t - maxp; i++)
								{
									s[i] = mind;
								}
							}
						}

						if (t - i > MAX_LENGTH)
						{
							for (int n = 0; n < src.cols; n++)
							{
								s[n] = invalidvalue;
							}
						}
					}
				}
			}
			s[0] = s[1];
			s[src.cols - 1] = s[src.cols - 2];
			s += step;
			sbl += step;
		}
	}

	template <class srcType>
	void correctDisparityBoundary_(Mat& src, Mat& refimg, const int r, Mat& dest)
	{

		srcType invalidvalue = 0;
		Mat sobel;
		Mat ref;
		if (refimg.channels() == 3)cvtColor(refimg, ref, COLOR_BGR2GRAY);
		else ref = refimg.clone();
		medianBlur(ref, ref, 3);
		//blurRemoveMinMax(ref,ref,1,0);

		Sobel(ref, sobel, CV_16S, 1, 0);
		sobel = abs(sobel);

		int bb = 0;
		const int MAX_LENGTH = (int)(src.cols * 1.0);

		srcType* s = src.ptr<srcType>(0);
		const int step = src.cols;
		short* sbl = sobel.ptr<short>(0);
		Mat sobel2;
		sobel.convertTo(sobel2, CV_8U);
		//imshow("sbl",sobel2);

		for (int j = bb; j < src.rows - bb; j++)
		{
			s[0] = saturate_cast<srcType>(255 * 16);
			s[src.cols - 1] = saturate_cast<srcType>(255 * 16);//�\���̂���ő�l�����
			//��������0��������l�̓����Ă���ߖT�̃s�N�Z���i�G�s�|�[������j�̍ŏ��l�Ŗ��߂�
			for (int i = 1; i < src.cols - 1; i++)
			{
				if (s[i] <= invalidvalue)
				{
					if (s[i + 1] > invalidvalue)
					{
						s[i] = min(s[i + 1], s[i - 1]);
						i++;
					}
					else
					{
						int t = i;
						do
						{
							t++;
							if (t > src.cols - 2)break;
						} while (s[t] <= invalidvalue);

						srcType maxd;
						srcType mind;
						if (s[i - 1] < s[t])
						{
							mind = s[i - 1];
							maxd = s[t];

							int maxp;
							int maxval = 0;
							for (int k = -r; k <= r; k++)
							{
								if (sbl[t + k] > maxval)
								{
									maxp = k;
									maxval = sbl[t + k];
								}
							}
							if (maxp >= 0)
							{
								for (; i < t + maxp; i++)
								{
									s[i] = mind;
								}
							}
							else
							{
								for (; i < t + maxp; i++)
								{
									s[i] = mind;
								}
								for (; i < t; i++)
								{
									s[i] = maxd;
								}
							}
						}
						else
						{
							mind = s[t];
							maxd = s[i - 1];

							int maxp;
							int maxval = 0;
							for (int k = -r; k <= r; k++)
							{
								if (sbl[i - 1 + k] > maxval)
								{
									maxp = k;
									maxval = sbl[i - 1 + k];
								}
							}
							if (maxp >= 0)
							{
								for (; i < maxp; i++)
								{
									s[i] = maxd;
								}
								for (; i < t; i++)
								{
									s[i] = mind;
								}
							}
							else
							{
								i += maxp;
								for (; i < t - maxp; i++)
								{
									s[i] = mind;
								}
							}
						}

						if (t - i > MAX_LENGTH)
						{
							for (int n = 0; n < src.cols; n++)
							{
								s[n] = invalidvalue;
							}
						}
					}
				}
			}
			s[0] = s[1];
			s[src.cols - 1] = s[src.cols - 2];
			s += step;
			sbl += step;
		}
	}

	void correctDisparityBoundaryFillOcc(Mat& src, Mat& refimg, const int r, Mat& dest)
	{
		if (src.type() == CV_8U)
			correctDisparityBoundary_<uchar>(src, refimg, r, dest);
		if (src.type() == CV_16U)
			correctDisparityBoundary_<ushort>(src, refimg, r, dest);
		if (src.type() == CV_16S)
			correctDisparityBoundary_<short>(src, refimg, r, dest);
		if (src.type() == CV_32F)
			correctDisparityBoundary_<float>(src, refimg, r, dest);
		if (src.type() == CV_64F)
			correctDisparityBoundary_<double>(src, refimg, r, dest);
	}

	void correctDisparityBoundary(Mat& src, Mat& refimg, const int r, const int edgeth, Mat& dest, const int secondr, const int minedge)
	{
		if (src.type() == CV_8U)
			correctDisparityBoundaryE<uchar>(src, refimg, r, edgeth, dest, 0, 0);
		if (src.type() == CV_16U)
			correctDisparityBoundaryE<ushort>(src, refimg, r, edgeth, dest, 0, 0);
		if (src.type() == CV_16S)
			correctDisparityBoundaryE<short>(src, refimg, r, edgeth, dest, 0, 0);
		if (src.type() == CV_32F)
			correctDisparityBoundaryE<float>(src, refimg, r, edgeth, dest, 0, 0);
		if (src.type() == CV_64F)
			correctDisparityBoundaryE<double>(src, refimg, r, edgeth, dest, 0, 0);

	}

	template <class srcType>
	static void fillOcclusionBox_(Mat& src, const srcType invalidvalue, const srcType maxval)
	{
		int bb = 0;
		const int MAX_LENGTH = (int)(src.cols * 1.0 - 5);

		srcType* s = src.ptr<srcType>(0);
		const int step = src.cols;
		Mat testim = Mat::zeros(src.size(), CV_8U); const int lineth = 30;
		for (int j = bb; j < src.rows - bb; j++)
		{
			s[0] = maxval;//�\���̂���ő�l�����
			s[src.cols - 1] = maxval;//�\���̂���ő�l�����
			//��������0��������l�̓����Ă���ߖT�̃s�N�Z���i�G�s�|�[������j�̍ŏ��l�Ŗ��߂�
			for (int i = 1; i < src.cols - 1; i++)
			{
				if (s[i] <= invalidvalue)
				{
					if (s[i + 1] > invalidvalue)
					{
						s[i] = min(s[i + 1], s[i - 1]);
						i++;
					}
					else
					{
						int t = i;
						do
						{
							t++;
							if (t > src.cols - 2)break;
						} while (s[t] <= invalidvalue);

						if (t - i > lineth)line(testim, Point(i, j), Point(t, j), 255);

						srcType dd;
						//if(s[i-1]<=invalidvalue)dd=s[t];
						//else if(s[t]<=invalidvalue)dd=s[i-1];
						//else dd = min(s[i-1],s[t]);
						dd = min(s[i - 1], s[t]);
						if (t - i > MAX_LENGTH)
						{
							//for(int n=0;n<src.cols;n++)s[n]=invalidvalue;
							memcpy(s, s - step, step * sizeof(srcType));
						}
						else
						{
							for (; i < t; i++)
							{
								s[i] = dd;
							}
						}
					}
				}
			}
			s[0] = s[1];
			s[src.cols - 1] = s[src.cols - 2];
			s += step;
		}
		/*Mat temp;
		boxFilter(src,temp,src.type(),Size(3,7));
		temp.copyTo(src,testim);
		imshow("test",testim);*/
	}
#pragma endregion

	//main function
	void StereoBase::matching(Mat& leftim, Mat& rightim, Mat& destDisparityMap, const bool isFeedback, Mat& weightMap)
	{
		if (destDisparityMap.empty() || leftim.size() != destDisparityMap.size()) destDisparityMap.create(leftim.size(), CV_16S);
		minCostMap.create(leftim.size(), CV_8U);
		minCostMap.setTo(255);
		if ((int)DSI.size() < numberOfDisparities)DSI.resize(numberOfDisparities);

		Mat guideImage;
		cvtColor(leftim, guideImage, COLOR_BGR2GRAY);

		{
#ifdef TIMER_STEREO_BASE
			Timer t("pre filter");
#endif
			prefilter(leftim, rightim);
		}

		{
#ifdef TIMER_STEREO_BASE
			Timer t("Cost computation & aggregation");
#endif
			if (AggregationMethod == Aggregation_CrossBasedBox) clf.makeKernel(guideImage, aggregationRadiusH, aggregationGuidedfilterEps, 0);

#pragma omp parallel for
			for (int i = 0; i < numberOfDisparities; i++)
			{
				const int d = minDisparity + i;
				getPixelMatchingCost(d, DSI[i]);
				if (isFeedback)addCostIterativeFeedback(DSI[i], d, destDisparityMap, 0, 2, 1.f);
				getCostAggregation(DSI[i], DSI[i], guideImage, isFeedback, weightMap);
			}
		}

		{
#ifdef TIMER_STEREO_BASE
			Timer t("Cost Optimization");
#endif
			if (P1 != 0 && P2 != 0)
				getOptScanline();
		}

		{
#ifdef TIMER_STEREO_BASE
			Timer t("DisparityComputation");
#endif
			getWTA(DSI, destDisparityMap, minCostMap);
		}

		{
#ifdef TIMER_STEREO_BASE
			Timer t("===Post Filterings===");
#endif
			//medianBlur(dest,dest,3);
			{
#ifdef TIMER_STEREO_BASE
				Timer t("Post: uniqueness");
#endif
				uniquenessFilter(minCostMap, destDisparityMap);
			}
			//subpix;
			{
#ifdef TIMER_STEREO_BASE
				Timer t("Post: subpix");
#endif
				subpixelInterpolation(destDisparityMap, subpixelInterpolationMethod);
				if (isRangeFilterSubpix) binalyWeightedRangeFilter(destDisparityMap, destDisparityMap, subpixelRangeFilterWindow, subpixelRangeFilterCap);
			}
			//R depth map;

			{
#ifdef TIMER_STEREO_BASE
				Timer t("Post: LR");
#endif
				if (isLRCheck)fastLRCheck(minCostMap, destDisparityMap);
			}
			{
#ifdef TIMER_STEREO_BASE
				Timer t("Post: mincost");
#endif
				minCostFilter(minCostMap, destDisparityMap);
			}
			{
#ifdef TIMER_STEREO_BASE
				Timer t("Post: filterSpeckles");
#endif
				if (isSpeckleFilter)
					filterSpeckles(destDisparityMap, 0, speckleWindowSize, speckleRange, specklebuffer);
			}
		}
#ifdef TIMER_STEREO_BASE
		cout << "=====================" << endl;
#endif
	}

	void StereoBase::operator()(Mat& leftim, Mat& rightim, Mat& dest)
	{
		matching(leftim, rightim, dest);
	}

	static void guiStereoMatchingOnMouse(int events, int x, int y, int flags, void* param)
	{
		Point* pt = (Point*)param;
		//if(events==CV_EVENT_LBUTTONDOWN)
		if (flags & EVENT_FLAG_LBUTTON)
		{
			pt->x = x;
			pt->y = y;
		}
	}

	string getOcclusionMethodName(const int method)
	{
		string ret;
		switch (method)
		{
		case 0: ret = "NO Filling"; break;
		case 1: ret = "Method 1 (simplest)"; break;
		case 2: ret = "Method 2"; break;
		case 3: ret = "Method 3"; break;
		case 4: ret = "Method 4"; break;
		default:
			break;
		}
		return ret;
	}

	void StereoBase::gui(Mat& leftim, Mat& rightim, Mat& destDisparity, StereoEval& eval)
	{
#pragma region setup
		ConsoleImage ci(Size(640, 680));
		//ci.setFontSize(12);
		string wname = "";
		string wname2 = "Disparity Map";

		namedWindow(wname2);
		moveWindow(wname2, 200, 200);
		int display_image_depth_alpha = 0; createTrackbar("disp-image: alpha", wname, &display_image_depth_alpha, 100);
		PixelMatchingMethod = Pixel_Matching_SDSobelBlend;
		//PixelMatchingMethod = Pixel_Matching_CENSUS5x5;
		createTrackbar("pix match method", wname, &PixelMatchingMethod, Pixel_Matching_Method_Size - 1);
		//pre filter
		createTrackbar("pcap", wname, &preFilterCap, 255);

		createTrackbar("Sobel blend a", wname, &costAlphaImageSobel, 100);
		pixelMatchErrorCap = preFilterCap;
		createTrackbar("pixel err cap", wname, &pixelMatchErrorCap, 255);

		//cost computation for texture alpha
		createTrackbar("Soble alpha p size", wname, &sobelBlendMapParam_Size, 10);
		createTrackbar("Soble alpha p 1", wname, &sobelBlendMapParam1, 255);
		createTrackbar("Soble alpha p 2", wname, &sobelBlendMapParam2, 255);

		//AggregationMethod = Aggregation_Gauss;
		AggregationMethod = Aggregation_Guided;
		createTrackbar("agg method", wname, &AggregationMethod, Aggregation_Method_Size - 1);
		createTrackbar("agg r width", wname, &aggregationRadiusH, 20);
		createTrackbar("agg r height", wname, &aggregationRadiusV, 20);
		int aggeps = 1; createTrackbar("agg guide color sigma/eps", wname, &aggeps, 255);
		int aggss = 100; createTrackbar("agg guide space sigma", wname, &aggss, 255);

		createTrackbar("P1", wname, &P1, 20);
		createTrackbar("P2", wname, &P2, 20);

		uniquenessRatio = 10;
		createTrackbar("uniq", wname, &uniquenessRatio, 100);
		createTrackbar("subpixel RF widow size", wname, &subpixelRangeFilterWindow, 10);
		createTrackbar("subpixel RF cap", wname, &subpixelRangeFilterCap, 64);

		createTrackbar("LR check disp12", wname, &disp12diff, 100);
		//int E = (int)(10.0*eps);
		//createTrackbar("eps",wname,&E,1000);

		//int spsize = 300;
		speckleWindowSize = 20;
		createTrackbar("speckleSize", wname, &speckleWindowSize, 1000);
		speckleRange = 16;
		createTrackbar("speckleDiff", wname, &speckleRange, 100);

		int occlusionMethod = 1;
		createTrackbar("occlusionMethod", wname, &occlusionMethod, 3);

		int occsearch2 = 4;
		createTrackbar("occ:s2", wname, &occsearch2, 15);

		int occth = 17;
		createTrackbar("occ:th", wname, &occth, 128);
		int occsearch = 4;
		createTrackbar("occ:s", wname, &occsearch, 15);

		int occsearchh = 2;
		createTrackbar("occ:sH", wname, &occsearchh, 15);
		int occiter = 0;
		createTrackbar("occ:iter", wname, &occiter, 10);

		int key = 0;

		Mat lim, rim;
		Mat bmdisp;
		Plot p(Size(640, 240));
		Plot histp(Size(640, 240));

		Plot signal(Size(640, 240 * 3));
		int vh = 0;
		int diffmode = 0;
		if (eval.isInit)
		{
			namedWindow("signal");
			createTrackbar("vh", "signal", &vh, 1);
			createTrackbar("mode", "signal", &diffmode, 2);
		}

		int bx = 3;
		createTrackbar("AsubboxR", wname, &bx, 10);
		int bran = 31;
		createTrackbar("Asubboxrange", wname, &bran, 64);

		int gr = 3;
		createTrackbar("ref:guide r", wname, &gr, 10);
		int ge = 1;
		createTrackbar("ref:guide e", wname, &ge, 1000);
		int jnr = 2;
		createTrackbar("ref:jn r", wname, &jnr, 10);

		int boxb = 1;
		createTrackbar("boxb", wname, &boxb, 10);

		Point mpt = Point(100, 100);
		createTrackbar("px", wname, &mpt.x, leftim.cols - 1);
		createTrackbar("py", wname, &mpt.y, leftim.rows - 1);

		bool isPlotCostFunction = true;
		bool isPlotSignal = true;

		bool isGrid = false;
		bool isReCost = false;

		bool isStreak = false;
		bool isRefinement = false;
		bool isMedian = false;

		bool isShowGT = false;
		int maskType = 0;
		int maskPrec = 0;
		bool dispColor = false;

		setMouseCallback(wname2, guiStereoMatchingOnMouse, &mpt);

		CostVolumeRefinement cbf(minDisparity, numberOfDisparities);

		const bool isShowSubpixelDisparityDistribution = true;
		const bool isShowZeroMask = true;
		bool isShowDiffFillOcclution = false;

		Mat dispOutput;
		bool isFeedback = false;
		Mat weightMap = Mat::ones(leftim.size(), CV_32F);
		destDisparity.setTo(0);
#pragma endregion

		while (key != 'q')
		{
			//init
			aggregationGuidedfilterEps = aggeps;
			aggregationSigmaSpace = aggss;
			ci.clear();
			ci(getPixelMatchingMethodName(PixelMatchingMethod) + ": (i-u)");
			ci(getAggregationMethodName(AggregationMethod) + ": (@-[)");
			ci(getSubpixelInterpolationMethodName(subpixelInterpolationMethod) + ":  (p)");
			ci(getOcclusionMethodName(occlusionMethod) + ": (o)");
			if (isFeedback) ci(CV_RGB(0, 255, 0), "feedback true (f)");
			else  ci(CV_RGB(255, 0, 0), "feedback false (f)");
			ci("=======================");

			//body
			{
				Timer t("BM", 0, false);
				matching(leftim, rightim, destDisparity, isFeedback, weightMap);
				ci("Total time: %f ms", t.getTime());
			}

			Mat zeromask;
			compare(destDisparity, 0, zeromask, cv::CMP_EQ);
			weightMap.setTo(1.f);
			weightMap.setTo(0.1f, zeromask);
			imshow("wmap", weightMap);
			if (isShowZeroMask)imshow("zero mask before", zeromask);
			ci("A: valid %f %%", 100.0 * countNonZero(destDisparity) / destDisparity.size().area());

			{
				Mat base = destDisparity.clone();
#ifdef TIMER_STEREO_BASE
				Timer t("Post 2");
#else 
				Timer t("Post 2", 0, false);
#endif
				if (occlusionMethod == 1)
				{
#ifdef TIMER_STEREO_BASE
					Timer t("occ");
#endif
					//fillOcclusion(destDisparity, (minDisparity - 1) * 16);
					fillOcclusion(destDisparity);
				}
				else if (occlusionMethod == 2)
				{
					//LRCheckDisparity(destfill, destR, 0, 16, 0, 16);
					/*Mat mask;
					Mat destfill = dest.clone();
					Mat destR,destL;
					Mat dest1,dest2;
					transpose(dest,dest1);

					fillOcclusion(destfill);

					destfill.copyTo(dest);
					shiftDisparity(destfill,destR,1/16.0);
					shiftDisparity(destfill,destL,-1/16.0);
					imshowDisparity("warpl",destR,1);
					imshowDisparity("warpr",destL,1);
					LRCheckDisparity(destfill,destR,0,16,0,16);LRCheckDisparity(destfill,destR,0,16,0,16);
					LRCheckDisparity(destL,dest,0,16,0,16);

					compare(destfill,0,mask,cv::CMP_EQ);
					dest.setTo(0,mask);*/

					fillOcclusion(destDisparity);
					{
						Timer t("border");
						correctDisparityBoundaryE<short>(destDisparity, leftim, occsearch, occth, destDisparity, occsearch2, 32);
						ci("post Border: %f", t.getTime());
					}

					/*for(int i=0;i<occiter;i++)
					{
					//medianBlur(dest,dest,3);
					boxSubpixelRefinement(dest,dest,3,24);
					correctDisparityBoundaryE<short>(dest,leftim,occsearch,17,dest,occsearch2,30);
					}*/
					/*Mat dt;
					transpose(dest,dt);
					Mat lt;transpose(leftim,lt);
					correctDisparityBoundaryE<short>(dt,lt,occsearch,32,dest);
					transpose(dt,dest);*/
					//correctDisparityBoundary<short>(dest,leftim,occsearch,dest);
					//fillOcclusion(dest);

					//fillOcclusion(desttemp);
					//desttemp.copyTo(dest);
					/*fillOcclusion(dest1);
					transpose(dest1,desttemp);
					desttemp.copyTo(dest,~mask);*/

					//medianBlur(dest,dest,3);*/
				}
				else if (occlusionMethod == 3)
				{
					/*Mat mask;
					Mat destfill = dest.clone();
					Mat destR,destL;
					Mat dest1,dest2;
					transpose(dest,dest1);

					fillOcclusion(destfill);

					destfill.copyTo(dest);
					shiftDisparity(destfill,destR,1/16.0);
					shiftDisparity(destfill,destL,-1/16.0);
					imshowDisparity("warpl",destR,1);
					imshowDisparity("warpr",destL,1);
					LRCheckDisparity(destfill,destR,0,16,0,16);
					LRCheckDisparity(destL,dest,0,16,0,16);

					compare(destfill,0,mask,cv::CMP_EQ);
					dest.setTo(0,mask);*/

					//fillOcclusionBox_<short>(dest,0,3000);
					fillOcclusion(destDisparity);
					correctDisparityBoundaryEC<short>(destDisparity, leftim, occsearch, occth, destDisparity);

					/*for(int i=0;i<occiter;i++)
					{
					//medianBlur(dest,dest,3);
					boxSubpixelRefinement(dest,dest,3,24);
					correctDisparityBoundaryE<short>(dest,leftim,occsearch,17,dest,occsearch2,30);
					}*/

					Mat dt;
					transpose(destDisparity, dt);
					Mat lt; transpose(leftim, lt);

					/*Mat sobel;
					Mat ltg;cvtColor(lt,ltg,COLOR_BGR2GRAY);
					Sobel(ltg,sobel,CV_16S,1,0,3);
					Mat disp;dt.convertTo(disp,CV_8U,0.25);
					Mat sobel2;Mat(abs(sobel)).convertTo(sobel2,CV_8U);
					guiAlphaBlend(disp,sobel2);*/


					correctDisparityBoundaryECV<short>(dt, lt, occsearchh, occth, dt);
					Mat dest2;
					transpose(dt, dest2);
					Mat mask = Mat::zeros(destDisparity.size(), CV_8U);
					cv::rectangle(mask, Rect(40, 40, destDisparity.cols - 80, destDisparity.rows - 80), 255, FILLED);
					dest2.copyTo(destDisparity, mask);
				}
				else if (occlusionMethod == 4)
				{
					/*Mat mask;
					Mat destfill = dest.clone();
					Mat destR,destL;
					Mat dest1,dest2;
					transpose(dest,dest1);

					fillOcclusion(destfill);

					destfill.copyTo(dest);
					shiftDisparity(destfill,destR,1/16.0);
					shiftDisparity(destfill,destL,-1/16.0);
					imshowDisparity("warpl",destR,1);
					imshowDisparity("warpr",destL,1);
					LRCheckDisparity(destfill,destR,0,16,0,16);
					LRCheckDisparity(destL,dest,0,16,0,16);

					compare(destfill,0,mask,cv::CMP_EQ);
					dest.setTo(0,mask);*/

					fillOcclusion(destDisparity);


					correctDisparityBoundaryE<short>(destDisparity, leftim, occsearch, 32, destDisparity, occsearch2, 30);

					for (int i = 0; i < occiter; i++)
					{
						Mat dt;
						transpose(destDisparity, dt);
						Mat lt; transpose(leftim, lt);
						correctDisparityBoundaryE<short>(dt, lt, 2, 32, destDisparity, occsearch2, 30);
						transpose(dt, destDisparity);
						correctDisparityBoundaryE<short>(destDisparity, leftim, occsearch, 32, destDisparity, occsearch2, 30);
						filterSpeckles(destDisparity, 0, speckleWindowSize, speckleRange);
						fillOcclusion(destDisparity);
					}


					//correctDisparityBoundary<short>(dest,leftim,occsearch,dest);
					//fillOcclusion(dest);

					//fillOcclusion(desttemp);
					//desttemp.copyTo(dest);
					/*fillOcclusion(dest1);
					transpose(dest1,desttemp);
					desttemp.copyTo(dest,~mask);*/

					//medianBlur(dest,dest,3);*/
				}

				if (isShowDiffFillOcclution)
				{
					fillOcclusion(base);
					absdiff(base, destDisparity, base); Mat mask; compare(base, 0, mask, cv::CMP_NE); imshow("occlusion process diff", mask);
				}

				//medianBlur(dest,dest,3);
				if (isStreak)
				{
					removeStreakingNoise(destDisparity, destDisparity, 16);
					removeStreakingNoiseV(destDisparity, destDisparity, 16);
				}
				if (isMedian)
				{
					medianBlur(destDisparity, destDisparity, 3);
				}
				if (isRefinement)
				{
					Timer t("refinement", 0, false);
					//crossBasedAdaptiveBoxFilter(destDisparity, leftim, destDisparity, Size(2 * gr + 1, 2 * gr + 1), ge);
					Mat temp;

					/*
					Mat a, b;
					multiply(destDisparity, weightMap, a, 1, CV_32F);
					guidedImageFilter(a, leftim, a, 4, 1, GUIDED_SEP_VHI);
					guidedImageFilter(weightMap, leftim, b, 4, 1, GUIDED_SEP_VHI);
					divide(a, b, temp, 1, CV_16S);
					*/
					guidedImageFilter(destDisparity, leftim, temp, gr, ge, GUIDED_SEP_VHI);
					//guidedImageFilter(destDisparity, target[0], temp, 4, 1, GUIDED_SEP_VHI);
					jointNearestFilter(temp, destDisparity, Size(2 * jnr + 1, 2 * jnr + 1), destDisparity);

					//cp::weightedModeFilter(destDisparity, leftim, destDisparity, 4, 2, 10, 500, 0, 2);

					//guidedImageFilter(destDisparity, leftim, temp, 4, 1, GUIDED_SEP_VHI);
					//jointNearestFilter(temp, destDisparity, Size(3, 3), destDisparity);

					ci("refine: %f ms", t.getTime());
				}

				if (isReCost)
				{
					//jointBilateralModeFilter(dest,dest,9,50,255,leftim);
					//refineFromCost(dest,dest);
				}
				ci("a-post time: %f ms", t.getTime());//additional post processing time
			}

			compare(destDisparity, 0, zeromask, cv::CMP_EQ);
			if (isShowZeroMask)imshow("zero mask after", zeromask);
			ci("B: valid %f %%", 100.0 * countNonZero(destDisparity) / destDisparity.size().area());

			if (isShowSubpixelDisparityDistribution)
			{
				histp.clear();
				histp.setYLabel("number of pixels");
				histp.setXLabel("subpixel_16");
				histp.setKey(0);
				histp.setGrid(0);
				histp.setIsDrawMousePosition(false);

				short* d = destDisparity.ptr<short>(0);
				int hist[16];
				for (int i = 0; i < 16; i++)hist[i] = 0;

				for (int i = 0; i < destDisparity.size().area(); i++)
				{
					if (d[i] > minDisparity * 16 && d[i] < (numberOfDisparities + minDisparity - 1) * 16)
					{
						hist[d[i] % 16]++;
					}
				}

				for (int i = 0; i < 16; i++)
				{
					histp.push_back(i, hist[(i + 8) % 16]);
				}

				histp.plot("subpixel disparity distribution", false);
			}

			if (dispColor)cvtDisparityColor(destDisparity, dispOutput, minDisparity, numberOfDisparities - 10, 2, 16);
			else			cvtDisparityColor(destDisparity, dispOutput, 0, numberOfDisparities, 0);

			//plot cost function
			if (isPlotCostFunction)
			{
				p.clear();
				p.setYLabel("error");
				p.setXLabel("disparity");
				p.setPlotTitle(0, "cost");
				p.setPlotTitle(1, "answer");
				p.setPlotTitle(2, "result");
				p.setGrid(0);
				p.setIsDrawMousePosition(false);
				p.setXYRange(0, numberOfDisparities + minDisparity + 1, 0, 64);

				if (eval.isInit)
				{
					const int dd = (int)(eval.ground_truth.at<uchar>(mpt.y, mpt.x) / eval.amp + 0.5);
					const int dd2 = (int)((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0) + 0.5);

					const double ddd = (eval.ground_truth.at<uchar>(mpt.y, mpt.x) / eval.amp);
					const double ddd2 = ((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0));
					for (int i = 0; i < numberOfDisparities; i++)
					{
						p.push_back(i + minDisparity, DSI[i].at<uchar>(mpt.y, mpt.x), 0);

						if (abs(i + minDisparity - dd) <= 1)
							p.push_back(ddd, 0, 1);
						else
							p.push_back(i + minDisparity, 127, 1);

						if (abs(i + minDisparity - dd2) == 0)
							p.push_back(ddd2, 0, 2);
						else
							p.push_back(i + minDisparity, 64, 2);
					}
				}
				else
				{
					const int dd2 = (int)((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0) + 0.5);
					const double ddd2 = ((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0));
					for (int i = 0; i < numberOfDisparities; i++)
					{
						p.push_back(i + minDisparity, DSI[i].at<uchar>(mpt.y, mpt.x), 0);

						if (abs(i + minDisparity - dd2) == 0)
							p.push_back(ddd2, 0, 2);
						else
							p.push_back(i + minDisparity, 64, 2);
					}
				}

				p.plot("cost function", false);

			}

			//plot signal
			if (isPlotSignal)
			{
				signal.clear();
				const int mindisp = 15;
				if (vh == 0)
				{
					signal.setXYRange(0, destDisparity.cols - 1, mindisp, 64);
					signal.setPlot(0, CV_RGB(0, 0, 0), 0, 1, 1);
					signal.setPlot(1, CV_RGB(255, 0, 0), 0, 1, 1);
					for (int i = 0; i < destDisparity.cols; i++)
					{
						double ddd = (eval.ground_truth.at<uchar>(mpt.y, i) / eval.amp);
						double ddd2 = ((double)destDisparity.at<short>(mpt.y, i) / (16.0));

						if (diffmode == 1)
						{
							ddd = abs(ddd - (eval.ground_truth.at<uchar>(mpt.y, i - 1) / eval.amp));
							ddd2 = abs(ddd2 - (((double)destDisparity.at<short>(mpt.y, i - 1) / (16.0))));
						}
						if (diffmode == 2)
						{
							ddd = abs((ddd - (eval.ground_truth.at<uchar>(mpt.y, i - 1) / eval.amp)) - (ddd - (eval.ground_truth.at<uchar>(mpt.y, i + 1) / eval.amp)));
							ddd2 = abs((ddd2 - ((double)destDisparity.at<short>(mpt.y, i - 1) / (16.0))) - (ddd2 - ((double)destDisparity.at<short>(mpt.y, i + 1) / (16.0))));
						}

						signal.push_back(i, ddd2, 0);
						signal.push_back(i, ddd, 1);
					}
					signal.plotData();
					imshow("signal", signal.render);
				}
				else
				{
					signal.setXYRange(0, destDisparity.rows - 1, mindisp, 64);
					signal.setPlot(0, CV_RGB(0, 0, 0), 0, 1, 1);
					signal.setPlot(1, CV_RGB(255, 0, 0), 0, 1, 1);
					for (int i = 0; i < destDisparity.rows; i++)
					{
						double ddd = (eval.ground_truth.at<uchar>(i, mpt.x) / eval.amp);
						double ddd2 = ((double)destDisparity.at<short>(i, mpt.x) / (16.0));

						if (diffmode == 1)
						{
							ddd = abs(ddd - (eval.ground_truth.at<uchar>(i - 1, mpt.x) / eval.amp));
							ddd2 = abs(ddd2 - (((double)destDisparity.at<short>(i - 1, mpt.x) / (16.0))));
						}
						if (diffmode == 2)
						{
							ddd = abs(-(ddd - (eval.ground_truth.at<uchar>(i - 1, mpt.x) / eval.amp)) + (-ddd + (eval.ground_truth.at<uchar>(i + 1, mpt.x) / eval.amp)));
							ddd2 = abs(-(ddd2 - ((double)destDisparity.at<short>(i - 1, mpt.x) / (16.0))) + (-ddd2 + ((double)destDisparity.at<short>(i + 1, mpt.x) / (16.0))));
						}

						signal.push_back(i, ddd2, 0);
						signal.push_back(i, ddd, 1);
					}
					signal.plotData();
					Mat show;
					transpose(signal.render, show);
					flip(show, show, 1);
					imshow("signal", show);
				}
			}

			//compute StereoEval
			if (eval.isInit)
			{
				Mat maskbadpixel = Mat::zeros(destDisparity.size(), CV_8U);
				bool isPrintEval = false;
				;
				ci("th 0.5:" + eval(destDisparity, 0.5, isPrintEval, 16));
				ci("th 1.0:" + eval(destDisparity, 1.0, isPrintEval, 16));
				ci("th 2.0:" + eval(destDisparity, 2.0, isPrintEval, 16));
				ci("MSE   :" + eval.getMSE(destDisparity, false, 16));
				

				if (maskType != 0)
				{
					if (maskPrec == 0)eval(destDisparity, 0.5, isPrintEval, 16);
					if (maskPrec == 1)eval(destDisparity, 1.0, isPrintEval, 16);
					if (maskPrec == 2)eval(destDisparity, 2.0, isPrintEval, 16);

					if (maskType == 1)
						eval.nonocc_th.copyTo(maskbadpixel);
					else if (maskType == 2)
						eval.all_th.copyTo(maskbadpixel);
					else if (maskType == 3)
						eval.disc_th.copyTo(maskbadpixel);
				}
				if (isShowGT)
				{
					if (dispColor) { Mat a; eval.ground_truth.convertTo(a, CV_16S, 4); cvtDisparityColor(a, dispOutput, minDisparity, numberOfDisparities - 10, 2, 16); }
					else eval.ground_truth.copyTo(dispOutput);
				}
				else
				{
					dispOutput.setTo(Scalar(0, 0, 255), maskbadpixel);
				}
			}

			alphaBlend(leftim, dispOutput, display_image_depth_alpha / 100.0, dispOutput);
			if (isGrid)
			{
				line(dispOutput, Point(0, mpt.y), Point(leftim.cols, mpt.y), CV_RGB(0, 255, 0));
				line(dispOutput, Point(mpt.x, 0), Point(mpt.x, leftim.rows), CV_RGB(0, 255, 0));
			}

			if (isUniquenessFilter) ci(CV_RGB(0, 255, 0), "uniqueness filter (1): true");
			else ci(CV_RGB(255, 0, 0), "uniqueness filter (1): false");

			if (isLRCheck) ci(CV_RGB(0, 255, 0), "LR check (2): true");
			else ci(CV_RGB(255, 0, 0), "LR check (2): false");

			if (isProcessLBorder) ci(CV_RGB(0, 255, 0), "LBorder (3) LR check must be true: true");
			else ci(CV_RGB(255, 0, 0), "LBorder (3): false");

			if (isRangeFilterSubpix) ci(CV_RGB(0, 255, 0), "range filter subpixel (4): true");
			else ci(CV_RGB(255, 0, 0), "range filter subpixel (4): false");

			if (isMinCostFilter) ci(CV_RGB(0, 255, 0), "min cost filter (5): true");
			else ci(CV_RGB(255, 0, 0), "min cost filter  (5): false");

			if (isSpeckleFilter) ci(CV_RGB(0, 255, 0), "speckle filter (6): true");
			else ci(CV_RGB(255, 0, 0), "speckle filter  (6): false");

			ci("==== additional post filter =====");
			if (isStreak)ci(CV_RGB(0, 255, 0), "Streak (7): true");
			else ci(CV_RGB(255, 0, 0), "Streak (7): false");

			if (isMedian)ci(CV_RGB(0, 255, 0), "Median (8): true");
			else ci(CV_RGB(255, 0, 0), "Median (8): false");

			if (isRefinement) ci(CV_RGB(0, 255, 0), "Guided (9): true");
			else 	ci(CV_RGB(255, 0, 0), "Guided (9): false");


			if (isShowGT)
			{
				ci(CV_RGB(255, 0, 0), "show ground trueth");
			}
			else
			{
				ci("=======================");
				if (maskType != 0)
				{
					if (maskPrec == 2 && maskType == 1)
						ci(CV_RGB(255, 0, 0), "mask: nonocc, prec: 2.0");
					if (maskPrec == 1 && maskType == 1)
						ci(CV_RGB(255, 0, 0), "mask: nonocc, prec: 1.0");
					if (maskPrec == 0 && maskType == 1)
						ci(CV_RGB(255, 0, 0), "mask: nonocc, prec: 0.5");

					if (maskPrec == 2 && maskType == 2)
						ci(CV_RGB(255, 0, 0), "mask: all, prec: 2.0");
					if (maskPrec == 1 && maskType == 2)
						ci(CV_RGB(255, 0, 0), "mask: all, prec: 1.0");
					if (maskPrec == 0 && maskType == 2)
						ci(CV_RGB(255, 0, 0), "mask: all, prec: 0.5");

					if (maskPrec == 2 && maskType == 3)
						ci(CV_RGB(255, 0, 0), "mask: disc, prec: 2.0");
					if (maskPrec == 1 && maskType == 3)
						ci(CV_RGB(255, 0, 0), "mask: disc, prec: 1.0");
					if (maskPrec == 0 && maskType == 3)
						ci(CV_RGB(255, 0, 0), "mask: disc, prec: 0.5");
				}
				else
				{
					ci(CV_RGB(255, 0, 0), "mask: none");
				}
			}

			imshow(wname2, dispOutput);
			imshow("console", ci.image);
			setTrackbarPos("px", wname, mpt.x);
			setTrackbarPos("py", wname, mpt.y);
			key = waitKey(1);

			if (key == '1') isUniquenessFilter = isUniquenessFilter ? false : true;
			if (key == '2') isLRCheck = isLRCheck ? false : true;
			if (key == '3') isProcessLBorder = (isProcessLBorder) ? false : true;
			if (key == '4')	isRangeFilterSubpix = (isRangeFilterSubpix) ? false : true;
			if (key == '5') isMinCostFilter = isMinCostFilter ? false : true;
			if (key == '6') isSpeckleFilter = isSpeckleFilter ? false : true;

			if (key == '7') isStreak = (isStreak) ? false : true;
			if (key == '8') isMedian = (isMedian) ? false : true;
			if (key == '9') isRefinement = (isRefinement) ? false : true;

			if (key == 'a')	isReCost = (isReCost) ? false : true;
			if (key == 'f') isFeedback = isFeedback ? false : true;
			if (key == 'g') isGrid = (isGrid) ? false : true;
			if (key == 'c') guiAlphaBlend(dispOutput, leftim);
			if (key == 'i') { PixelMatchingMethod++; PixelMatchingMethod = (PixelMatchingMethod > Pixel_Matching_Method_Size - 1) ? 0 : PixelMatchingMethod; }
			if (key == 'u') { PixelMatchingMethod--; PixelMatchingMethod = (PixelMatchingMethod < 0) ? Pixel_Matching_Method_Size - 2 : PixelMatchingMethod; }
			if (key == '@') { AggregationMethod++; AggregationMethod = (AggregationMethod > Aggregation_Method_Size - 1) ? 0 : AggregationMethod; }
			if (key == '[') { AggregationMethod--; AggregationMethod = (AggregationMethod < 0) ? Aggregation_Method_Size - 2 : AggregationMethod; }

			if (key == 'o') { occlusionMethod++; if (occlusionMethod > 4)occlusionMethod = 0; }
			if (key == 'p') { subpixelInterpolationMethod++; subpixelInterpolationMethod = (subpixelInterpolationMethod > 2) ? 0 : subpixelInterpolationMethod; }


			if (key == 'w')dispColor = (dispColor) ? false : true;
			if (key == 'e')isShowGT = (isShowGT) ? false : true;
			//if (key == 'b') guiCrossBasedLocalFilter(leftim);

			if (key == 'r')
			{
				destDisparity.setTo(0);
			}

			
			if (key == 'b')maskType++; maskType = maskType > 4 ? 0 : maskType;
			if (key == 'n')maskType--; maskType = maskType < 0 ? 3 : maskType;
			if (key == 'm')maskPrec++; maskPrec = maskPrec > 3 ? 0 : maskPrec;
			if (key == ',')maskPrec--; maskPrec = maskPrec < 0 ? 2 : maskPrec;
			
		}
	}

	void resizeDown(Mat& src, Mat& dest, int rfact, int method)
	{
		int modH = rfact - src.cols % rfact;
		int modV = rfact - src.rows % rfact;
		Mat sim; copyMakeBorder(src, sim, 0, modV, 0, modH, cv::BORDER_REPLICATE);
		resize(sim, dest, Size(sim.cols / rfact, sim.rows / rfact), 0.0, 0.0, method);
	}

	void resizeUP(Mat& src, Mat& dest, int rfact, int method)
	{
		Mat temp;
		resize(src, temp, Size(src.cols * rfact, src.rows * rfact), 0.0, 0.0, method);
		if (dest.empty())
			temp.copyTo(dest);
		else
		{
			cout << rfact << "," << temp.cols << "," << temp.rows << endl;
			Mat a = temp(Rect(0, 0, dest.cols, dest.rows));
			a.copyTo(dest);
		}
	}
	}