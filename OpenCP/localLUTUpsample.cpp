#include "localLUTUpsample.hpp"
#include "upsample.hpp"

#include <inlineSIMDFunctions.hpp>
#include <arithmetic.hpp>
#include <blend.hpp>
#include <draw.hpp>

#define LOCALLUT_USE_SIMD

using namespace cv;
using namespace std;

namespace cp
{
#pragma region utility
	std::string LocalLUTUpsample::getBuildingLUTMethod(const BUILD_LUT method)
	{
		string ret = "";

		switch (method)
		{
		case BUILD_LUT::LInf_MIN:
		default:
			ret = "LInf_MIN"; break;
		case BUILD_LUT::L1_MIN:
			ret = "L1_MIN"; break;
		case BUILD_LUT::L2_MIN:
			ret = "L2_MIN"; break;
		case BUILD_LUT::FREQUENCY_MAX_WTA:
			ret = "FREQUENCY_MAX_WTA"; break;

		case BUILD_LUT::FREQUENCY_MAX_DP:
			ret = "FREQUENCY_MAX_DP"; break;

		}
		return ret;
	}

	std::string LocalLUTUpsample::getTensorUpsamplingMethod(const UPTENSOR method)
	{
		string ret = "";

		switch (method)
		{
		case UPTENSOR::BOX4:
			ret = "BOX4"; break;
		case UPTENSOR::BOX16:
		default:
			ret = "BOX16"; break;
		case UPTENSOR::BOX64:
			ret = "BOX64"; break;
		case UPTENSOR::GAUSS4:
			ret = "GAUSS4"; break;
		case UPTENSOR::GAUSS16:
			ret = "GAUSS16"; break;
		case UPTENSOR::GAUSS64:
			ret = "GAUSS64"; break;
		case UPTENSOR::LINEAR:
			ret = "LINEAR"; break;
		case UPTENSOR::CUBIC:
			ret = "CUBIC"; break;
		case UPTENSOR::BILATERAL16:
			ret = "BILATERAL16"; break;
		case UPTENSOR::BILATERAL64:
			ret = "BILATERAL64"; break;
		case UPTENSOR::BoxNxN:
			ret = "BoxNxN"; break;
		case UPTENSOR::GaussNxN:
			ret = "GaussNxN"; break;
		case UPTENSOR::LaplaceNxN:
			ret = "LaplaceNxN"; break;

		}
		return ret;
	}

	std::string LocalLUTUpsample::getBoundaryMethod(const BOUNDARY method)
	{
		string ret = "";

		switch (method)
		{
		case BOUNDARY::MINMAX_OUTPUT:
			ret = "MINMAX"; break;
		case BOUNDARY::MINMAX0_255:
			ret = "MINMAX0-255"; break;
		case BOUNDARY::REPLICATE:
			ret = "REPLICATE"; break;
		case BOUNDARY::LINEAR:
			ret = "LINEAR"; break;
		case BOUNDARY::LINEAR_LAST2:
			ret = "LINEAR_LAST2"; break;
		case BOUNDARY::NO_INTERPOLATION:
			ret = "NO_INTERPOLATION"; break;
		case BOUNDARY::EXPERIMENT1:
			ret = "EXPERIMENT 1"; break;
		case BOUNDARY::EXPERIMENT2:
			ret = "EXPERIMENT 2"; break;
		}
		return ret;
	}

#pragma region guiLUT
	static void plotLUTLine(const uchar* src, int lutsize, Mat& dest, Scalar color, int offset)
	{
		if (dest.empty())
		{
			dest.create(256, 256, CV_8UC3);
			dest.setTo(0);
		}

		int step = 256 / lutsize;
		int imax = step * lutsize;
		for (int i = 0; i < imax; i++)
		{
			int idx = i / step;
			int sub = i % step;
			float ia = sub / (float)step;
			float a = 1.f - ia;

			int v = (int)(a * (255 - step * src[idx]) + ia * (255 - step * src[min(idx + 1, lutsize - 1)]) + 0.5) - offset;
			cp::drawPlus(dest, Point(i, v), 2, color);
			//circle(dest, Point(i, v), 1, color, cv::FILLED);
		}
	}

	struct MouseParameterLocalLUT
	{
		cv::Rect pt;
		std::string wname;
	};

	static void guiLUTPreviewMouse(int event, int x, int y, int flags, void* param)
	{
		MouseParameterLocalLUT* retp = (MouseParameterLocalLUT*)param;

		if (flags == EVENT_FLAG_LBUTTON)
		{
			retp->pt.x = max(0, min(retp->pt.width - 1, x));
			retp->pt.y = max(0, min(retp->pt.height - 1, y));

			setTrackbarPos("x", retp->wname, x);
			setTrackbarPos("y", retp->wname, y);
		}
	}

	void LocalLUTUpsample::guiLUT(Mat& lowres_src, Mat& lowres_prc, Mat& highres_src, Mat& highres_groundtruth, bool isWait, string wname)
	{
		const Scalar background_color = Scalar::all(255);
		const Scalar background_gridlinecolor = Scalar::all(200);
		const int scale = highres_src.cols / lowres_src.cols;
		namedWindow(wname);
		moveWindow(wname, 500, 500);
		string wnameShow = wname + +"lowres src image";
		namedWindow(wnameShow);
		int alpha = 0; createTrackbar("0:src 100:prc", wnameShow, &alpha, 100);

		static MouseParameterLocalLUT param
		{
			Rect(lowres_src.cols / 2, lowres_src.rows / 2, lowres_src.cols, lowres_src.rows),
			wname
		};

		setMouseCallback(wnameShow, (MouseCallback)guiLUTPreviewMouse, (void*)&param);
		createTrackbar("x", wname, &param.pt.x, lowres_src.cols - 1);
		createTrackbar("y", wname, &param.pt.y, lowres_src.rows - 1);
		//channel(0: all, 1:B, 2:G, 3:R)
		static int c = 0; createTrackbar("c", wname, &c, 3);
		static int isScatterPlot = 1; createTrackbar("isScatterPlot", wname, &isScatterPlot, 1);
		static int isLUTLine = 1; createTrackbar("isLUTLine", wname, &isLUTLine, 1);
		//if subx == xmax, plot all
		static int subx = 2 * scale; createTrackbar("ex:highsub_x", wname, &subx, 4 * scale);
		//if suby == ymax, no plot
		static int suby = 2 * scale; createTrackbar("ex:highsub_y", wname, &suby, 4 * scale);

		const int lutsize = LUT_TensorAoS_B.channels();
		static bool isGrid = true;
		Mat showImage;
		Mat lutImage(256, 256, CV_8UC3);
		int key = 0;
		while (key != 'q')
		{
			cp::alphaBlend(lowres_src, lowres_prc, 1.0 - alpha * 0.01, showImage);
			Vec3b v = showImage.at<Vec3b>(Point(param.pt.x, param.pt.y));
			line(showImage, Point(0, param.pt.y), Point(lowres_src.cols, param.pt.y), Scalar(0, 0, 255));
			line(showImage, Point(param.pt.x, 0), Point(param.pt.x, lowres_src.rows), Scalar(0, 0, 255));
			showImage.at<Vec3b>(Point(param.pt.x, param.pt.y)) = v;

			lutImage.setTo(background_color);
			if (isGrid)
			{
				line(lutImage, Point(0, 63), Point(255, 63), background_gridlinecolor);
				line(lutImage, Point(0, 127), Point(255, 127), background_gridlinecolor);
				line(lutImage, Point(0, 191), Point(255, 191), background_gridlinecolor);

				line(lutImage, Point(63, 0), Point(63, 255), background_gridlinecolor);
				line(lutImage, Point(127, 0), Point(127, 255), background_gridlinecolor);
				line(lutImage, Point(191, 0), Point(191, 255), background_gridlinecolor);
			}

			//plot from internal src_low_border and dst_low_border
			if (isScatterPlot != 0)
			{
				const int d = 2 * patch_radius + 1;
				const Rect roi = Rect(param.pt.x, param.pt.y, d, d);
				const Mat sb = src_low_border(roi).clone();
				const Mat db = prc_low_border(roi).clone();
				for (int j = 0; j < d; j++)
				{
					for (int i = 0; i < d; i++)
					{
						if (c == 0)
						{
							circle(lutImage, Point(sb.at<uchar>(j, 3 * i + 0), 255 - db.at<uchar>(j, 3 * i + 0)), 3, Scalar(255, 128, 128), 1);
							circle(lutImage, Point(sb.at<uchar>(j, 3 * i + 1), 255 - db.at<uchar>(j, 3 * i + 1)), 3, Scalar(128, 255, 128), 1);
							circle(lutImage, Point(sb.at<uchar>(j, 3 * i + 2), 255 - db.at<uchar>(j, 3 * i + 2)), 3, Scalar(128, 128, 255), 1);
						}
						if (c == 1)
						{
							circle(lutImage, Point(sb.at<uchar>(j, 3 * i + 0), 255 - db.at<uchar>(j, 3 * i + 0)), 3, Scalar(255, 128, 128), 1);
						}
						if (c == 2)
						{
							circle(lutImage, Point(sb.at<uchar>(j, 3 * i + 1), 255 - db.at<uchar>(j, 3 * i + 1)), 3, Scalar(128, 255, 128), 1);
						}
						if (c == 3)
						{
							circle(lutImage, Point(sb.at<uchar>(j, 3 * i + 2), 255 - db.at<uchar>(j, 3 * i + 2)), 3, Scalar(128, 128, 255), 1);
						}
					}
				}
			}

			if (isLUTLine != 0)
			{
				const uchar* lutb = LUT_TensorAoS_B.ptr<uchar>(param.pt.y) + lutsize * param.pt.x;
				const uchar* lutg = LUT_TensorAoS_G.ptr<uchar>(param.pt.y) + lutsize * param.pt.x;
				const uchar* lutr = LUT_TensorAoS_R.ptr<uchar>(param.pt.y) + lutsize * param.pt.x;
				if (c == 0)
				{
					if (lutsize == 256)
					{
						plotLUTLine(lutb, lutsize, lutImage, Scalar(255, 0, 0), 0);
						plotLUTLine(lutg, lutsize, lutImage, Scalar(0, 255, 0), 0);
						plotLUTLine(lutr, lutsize, lutImage, Scalar(0, 0, 255), 0);
					}
					else
					{
						plotLUTLine(lutb, lutsize, lutImage, Scalar(255, 0, 0), offset_map.at<uchar>(param.pt.y, 3 * param.pt.x + 0));
						plotLUTLine(lutg, lutsize, lutImage, Scalar(0, 255, 0), offset_map.at<uchar>(param.pt.y, 3 * param.pt.x + 1));
						plotLUTLine(lutr, lutsize, lutImage, Scalar(0, 0, 255), offset_map.at<uchar>(param.pt.y, 3 * param.pt.x + 2));
					}

				}
				else
				{
					int offset = (lutsize == 256) ? 0 : offset_map.at<uchar>(param.pt.y, 3 * param.pt.x + c - 1);
					if (c == 1) plotLUTLine(lutb, lutsize, lutImage, Scalar(255, 0, 0), offset);
					if (c == 2) plotLUTLine(lutg, lutsize, lutImage, Scalar(0, 255, 0), offset);
					if (c == 3) plotLUTLine(lutr, lutsize, lutImage, Scalar(0, 0, 255), offset);
				}
			}
			//plot extra points from all Mat& highres_src, Mat& highres_groundtruth
			constexpr int explotmode = 0;
			if (subx == 4 * scale) //plot all
			{
				const Scalar b = Scalar(255, 128, 128);
				const Scalar g = Scalar(128, 230, 128);
				const Scalar r = Scalar(128, 128, 255);
				const Scalar gray = Scalar(80, 80, 80);
				for (int j = -2 * scale; j < 2 * scale; j++)
				{
					for (int i = -2 * scale; i < 2 * scale; i++)
					{
						const int bx = scale * param.pt.x + j;
						const int by = scale * param.pt.y + i;
						//int bo = src.at<uchar>(y, 3 * x + 0);
						const int bi = highres_src.at<uchar>(by, 3 * bx + 0);
						const int bo = highres_groundtruth.at<uchar>(by, 3 * bx + 0);
						const int gi = highres_src.at<uchar>(by, 3 * bx + 1);
						const int go = highres_groundtruth.at<uchar>(by, 3 * bx + 1);
						const int ri = highres_src.at<uchar>(by, 3 * bx + 2);
						const int ro = highres_groundtruth.at<uchar>(by, 3 * bx + 2);

						if constexpr (explotmode == 0)
						{
							if (c == 0 || c == 1)cp::drawTimes(lutImage, Point(bi, 255 - bo), 5, gray, 1);
							if (c == 0 || c == 2)cp::drawTimes(lutImage, Point(gi, 255 - go), 5, gray, 1);
							if (c == 0 || c == 3)cp::drawTimes(lutImage, Point(ri, 255 - ro), 5, gray, 1);
						}
						else if constexpr (explotmode == 1)
						{
							if (c == 0 || c == 1)cp::drawTimes(lutImage, Point(bi, 255 - bo), 5, b, 1);
							if (c == 0 || c == 2)cp::drawTimes(lutImage, Point(gi, 255 - go), 5, g, 1);
							if (c == 0 || c == 3)cp::drawTimes(lutImage, Point(ri, 255 - ro), 5, r, 1);
						}
						else
						{
							if (c == 0 || c == 1)circle(lutImage, Point(bi, 255 - bo), 3, Scalar(255, 128, 128), 1);
							if (c == 0 || c == 2)circle(lutImage, Point(gi, 255 - go), 3, Scalar(128, 230, 128), 1);
							if (c == 0 || c == 3)circle(lutImage, Point(ri, 255 - ro), 3, Scalar(128, 128, 255), 1);
						}
					}
				}
			}
			else if (suby == 4 * scale) //no process
			{
				;
			}
			else
			{
				const int bx = scale * param.pt.x + subx - 2 * scale;
				const int by = scale * param.pt.y + suby - 2 * scale;

				const int bi = highres_src.at<uchar>(by, 3 * bx + 0);
				const int bo = highres_groundtruth.at<uchar>(by, 3 * bx + 0);
				const int gi = highres_src.at<uchar>(by, 3 * bx + 1);
				const int go = highres_groundtruth.at<uchar>(by, 3 * bx + 1);
				const int ri = highres_src.at<uchar>(by, 3 * bx + 2);
				const int ro = highres_groundtruth.at<uchar>(by, 3 * bx + 2);

				if (c == 0 || c == 1)circle(lutImage, Point(bi, 255 - bo), 3, Scalar(255, 128, 128), 3);
				if (c == 0 || c == 2)circle(lutImage, Point(gi, 255 - go), 3, Scalar(128, 230, 128), 3);
				if (c == 0 || c == 3)circle(lutImage, Point(ri, 255 - ro), 3, Scalar(128, 128, 255), 3);
				//if (c == 0 || c == 1)cp::drawTimes(show, Point(bi, 255 - bo), 3, Scalar(255, 128, 128), 1);
				//if (c == 0 || c == 2)cp::drawTimes(show, Point(gi, 255 - go), 3, Scalar(128, 230, 128), 1);
				//if (c == 0 || c == 3)cp::drawTimes(show, Point(ri, 255 - ro), 3, Scalar(128, 128, 255), 1);
			}

			cv::imshow(wname, lutImage);
			cv::imshow(wnameShow, showImage);

#pragma region key
			if (isWait) key = waitKey(1);
			else break;

			if (key == 'e') exit(0);
			if (key == 'j')
			{
				param.pt.x--;
				param.pt.x = max(param.pt.x, 0);
				setTrackbarPos("x", wname, param.pt.x);
			}
			if (key == 'i')
			{
				param.pt.y--;
				param.pt.y = max(param.pt.y, 0);
				setTrackbarPos("y", wname, param.pt.y);
			}
			if (key == 'l')
			{
				param.pt.x++;
				param.pt.x = min(param.pt.x, lowres_src.cols - 1);
				setTrackbarPos("x", wname, param.pt.x);
			}
			if (key == 'k')
			{
				param.pt.y++;
				param.pt.y = min(param.pt.y, lowres_src.rows - 1);
				setTrackbarPos("y", wname, param.pt.y);
			}
			if (key == 'g') isGrid = (isGrid) ? false : true;
#pragma endregion
		}
	}
#pragma endregion
#pragma endregion

	// Create LUT (AoS: 8bit width lut_num channels image for each color, SoA: vector<Mat> for each lut_num)
	void LocalLUTUpsample::createLUTTensor(const int width, const int height, const int lut_num)
	{
		if (useSoA)
		{
			LUT_TensorSoA_B.resize(lut_num);
			LUT_TensorSoA_G.resize(lut_num);
			LUT_TensorSoA_R.resize(lut_num);
			for (int i = 0; i < lut_num; i++)
			{
				LUT_TensorSoA_B[i].create(Size(width, height), CV_8U);
				LUT_TensorSoA_G[i].create(Size(width, height), CV_8U);
				LUT_TensorSoA_R[i].create(Size(width, height), CV_8U);
			}
		}
		else
		{
			LUT_TensorAoS_B.create(Size(width, height), CV_MAKETYPE(CV_8U, lut_num));
			LUT_TensorAoS_G.create(Size(width, height), CV_MAKETYPE(CV_8U, lut_num));
			LUT_TensorAoS_R.create(Size(width, height), CV_MAKETYPE(CV_8U, lut_num));
		}
	}

#pragma region lut_boundary_condition
	inline void interpolateLUT(uchar* lut, const int lut_num)
	{
		for (int i = 1; i < lut_num - 1; i++)
		{
			if (lut[i] == 0)
			{
				uchar* l = lut + i;
				int next_d = 1;

				while (l[next_d] == 0)
				{
					next_d++;
					if (i + next_d > lut_num - 2)break;
				}

				const float div = 1.f / (next_d + 1);
				const int pre = l[-1];
				const int nxt = l[+next_d];
				for (int j = 0; j < next_d; j++)
				{
					l[j] = saturate_cast<uchar>(((next_d - j) * pre + (j + 1) * nxt) * div);
				}

				i += next_d;
			}
		}
	}

	inline void interpolateLUT(uchar* lut, const int lut_num, const uchar imin, const uchar imax)
	{
		for (int i = imin + 1; i < imax; i++)
		{
			if (lut[i] == 0)
			{
				uchar* l = lut + i;
				int next_d = 1;

				while (l[next_d] == 0)
				{
					next_d++;
					if (i + next_d > lut_num - 2)break;
				}

				const float div = 1.f / (next_d + 1);
				const int pre = l[-1];
				const int nxt = l[+next_d];
				for (int j = 0; j < next_d; j++)
				{
					l[j] = saturate_cast<uchar>(((next_d - j) * pre + (j + 1) * nxt) * div);
				}

				i += next_d;
			}
		}
	}

	inline void interpolateLUTStep(uchar* lut, const int lut_num, const uchar imin, const uchar imax)
	{
		for (int i = imin + 1; i < imax; i++)
		{
			if (lut[i] == 0)
			{
				uchar* l = lut + i;
				int next_d = 1;

				while (l[next_d] == 0)
				{
					next_d++;
					if (i + next_d > lut_num - 2)break;
				}

				const int h = next_d >> 1;
				const int pre = l[-1];
				const int nxt = l[+next_d];
				for (int j = 0; j < h; j++)
				{
					l[j] = pre;
				}
				for (int j = h; j < next_d; j++)
				{
					l[j] = nxt;
				}

				i += next_d;
			}
		}
	}


	inline void setLUTMinMax(uchar* lut, const int lut_num, const uchar minv, const uchar maxv)
	{
		if (lut[0] == 0) lut[0] = minv;
		if (lut[lut_num - 1] == 0) lut[lut_num - 1] = maxv;
	}

	inline void setLUTMinMax(uchar* lut, const int lut_num, const uchar minv, const uchar maxv, const uchar imin, const uchar imax)
	{
		if (imax - imin == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}

		//left side;
		float d = (float)(lut[imin] - minv) / imin;
		uchar* l = lut;
#ifdef LOCALLUT_USE_SIMD
		int idx = 0;
		const int simd_imin = get_simd_floor(imin, 16);
		const __m256 morder = _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
		__m256 md = _mm256_set1_ps(d);
		__m256 moff = _mm256_set1_ps(minv);
		int simdloop = (simd_imin) / 16;
		for (int i = 0; i < simdloop; ++i)
		{
			__m256i v0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx))), md, moff));
			__m256i v1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx + 8))), md, moff));
			_mm_store_si128((__m128i*)l, _mm256_cvtepi32x2_epu8(v0, v1));

			l += 16;
			idx += 16;
		}
		for (int i = simd_imin; i < imin; i++)
		{
			int v = saturate_cast<int>((idx++) * d + minv);
			*l++ = saturate_cast<uchar>(v);//lut[i] = max(0, min(lut_num - 1, v));
		}
#else
		for (int i = 0; i < imin; i++)
		{
			lut[i] = i * d + minv;
		}
#endif

		//right side
		int pre = lut[imax];
		d = (float)(maxv - pre) / (lut_num - imax);
		idx = 1;
		l = &lut[imax + 1];
#ifdef LOCALLUT_USE_SIMD
		const int simd_imax = min(lut_num, get_simd_ceil(imax, 16));
		for (int i = imax + 1; i < simd_imax; i++)
		{
			//lut[i] = (i - imax) * d + pre;
			float v = (idx++) * d + pre;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}

		md = _mm256_set1_ps(d);
		moff = _mm256_set1_ps(float(pre));
		simdloop = (lut_num - simd_imax) / 16;
		for (int i = 0; i < simdloop; ++i)
		{
			__m256i v0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx))), md, moff));
			__m256i v1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx + 8))), md, moff));
			_mm_storeu_si128((__m128i*)l, _mm256_cvtepi32x2_epu8(v0, v1));

			l += 16;
			idx += 16;
		}
#else
		for (int i = imax + 1; i < lut_num; i++)
		{
			lut[i] = (i - imax) * d + pre;
		}
#endif
	}

	inline void setLUTBoundaryLinearLast2(uchar* lut, const int lut_num, const uchar imin, const uchar imax)
	{
		if (imax - imin == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}

		int iminn = imin;
		int imaxp = imax;
		for (int i = imin + 1; i <= imax; i++)
		{
			if (lut[i] != 0)
			{
				iminn = i;
				break;
			}
		}
		for (int i = imax - 1; i >= imin; i--)
		{
			if (lut[i] != 0)
			{
				imaxp = i;
				break;
			}
		}


		float d0 = (float)(lut[iminn] - lut[imin]) / (float)(iminn - (int)imin);
		float d1 = (float)(lut[imax] - lut[imaxp]) / (float)((int)imax - imaxp);
		/*
		//ネガポジ反転を強引に直す
		const float d = (float)(lut[imax] - lut[imin]) / (float)(imax - imin);
		if (d > 0 && d0 < 0)d0 = d;
		if (d > 0 && d1 < 0)d1 = d;*/

		//left side;
		int offset = lut[imin];

		uchar* l = &lut[0];
		int idx = -imin;
		for (int i = 0; i < imin; i++)
		{
			int v = saturate_cast<int>((idx++) * d0 + offset);
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}

		//right side
		offset = lut[imax];
		l = &lut[imax + 1];
		idx = 1;
		const int end = lut_num - (imax + 1);
		for (int i = 0; i < end; i++)
		{
			float v = (idx++) * d1 + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
	}


	inline void setLUTBoundaryLinearFlatSwich(uchar* lut, const int lut_num, const uchar imin, const uchar imax, const uchar ave)
	{

		if (imax - imin == 0 || lut[imin] - lut[imax] == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}

		int idiff = abs(imax - imin);
		int odiff = abs(lut[imin] - lut[imax]);

		int th = 10;
		if (idiff < th && odiff < th)
		{
			memset(lut, ave, sizeof(uchar) * (lut_num));
			return;
		}

		const float d = (float)(lut[imax] - lut[imin]) / (float)(imax - imin);

		//left side;
		int offset = lut[imin];
		uchar* l = &lut[0];
		int idx = -imin;

#ifdef LOCALLUT_USE_SIMD
		const int simd_imin = get_simd_floor(imin, 16);
		const __m256 md = _mm256_set1_ps(d);
		const __m256 morder = _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
		__m256 moff = _mm256_set1_ps(float(offset));
		int simdloop = (simd_imin) / 16;
		for (int i = 0; i < simdloop; ++i)
		{
			__m256i v0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx))), md, moff));
			__m256i v1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx + 8))), md, moff));
			_mm_store_si128((__m128i*)l, _mm256_cvtepi32x2_epu8(v0, v1));

			l += 16;
			idx += 16;
		}
		for (int i = simd_imin; i < imin; i++)
		{
			float v = (idx++) * d + offset;
			//int v = (i - imin) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
#else
		for (int i = 0; i < imin; i++)
		{
			int v = (idx++) * d + offset;
			//int v = (i - imin) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
#endif

		//right side
		offset = lut[imax];
		l = &lut[imax + 1];
		idx = 1;
#ifdef LOCALLUT_USE_SIMD
		const int simd_imax = min(lut_num, get_simd_ceil(imax, 16));
		for (int i = imax + 1; i < simd_imax; i++)
		{
			float v = (idx++) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}

		moff = _mm256_set1_ps(float(offset));
		simdloop = (lut_num - simd_imax) / 16;
		for (int i = 0; i < simdloop; ++i)
		{
			__m256i v0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx))), md, moff));
			__m256i v1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx + 8))), md, moff));
			_mm_store_si128((__m128i*)l, _mm256_cvtepi32x2_epu8(v0, v1));

			l += 16;
			idx += 16;
		}
#else
		const int end = lut_num - (imax + 1);
		for (int i = 0; i < end; i++)
		{
			int v = (idx++) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
#endif

	}

	inline void setLUTBoundaryLinear(uchar* lut, const int lut_num, const uchar imin, const uchar imax)
	{
		if (imax - imin == 0 || lut[imin] - lut[imax] == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}

		const float d = (float)(lut[imax] - lut[imin]) / (float)(imax - imin);

		//left side;
		int offset = lut[imin];
		uchar* l = &lut[0];
		int idx = -imin;

#ifdef LOCALLUT_USE_SIMD
		const int simd_imin = get_simd_floor(imin, 16);
		const __m256 md = _mm256_set1_ps(d);
		const __m256 morder = _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
		__m256 moff = _mm256_set1_ps(float(offset));
		int simdloop = (simd_imin) / 16;
		for (int i = 0; i < simdloop; ++i)
		{
			__m256i v0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx))), md, moff));
			__m256i v1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx + 8))), md, moff));
			_mm_store_si128((__m128i*)l, _mm256_cvtepi32x2_epu8(v0, v1));

			l += 16;
			idx += 16;
		}
		for (int i = simd_imin; i < imin; i++)
		{
			float v = (idx++) * d + offset;
			//int v = (i - imin) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
#else
		for (int i = 0; i < imin; i++)
		{
			int v = (idx++) * d + offset;
			//int v = (i - imin) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
#endif

		//right side
		offset = lut[imax];
		l = &lut[imax + 1];
		idx = 1;
#ifdef LOCALLUT_USE_SIMD
		const int simd_imax = min(lut_num, get_simd_ceil(imax, 16));
		for (int i = imax + 1; i < simd_imax; i++)
		{
			float v = (idx++) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}

		moff = _mm256_set1_ps(float(offset));
		simdloop = (lut_num - simd_imax) / 16;
		for (int i = 0; i < simdloop; ++i)
		{
			__m256i v0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx))), md, moff));
			__m256i v1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_add_ps(morder, _mm256_set1_ps(float(idx + 8))), md, moff));
			_mm_storeu_si128((__m128i*)l, _mm256_cvtepi32x2_epu8(v0, v1));

			l += 16;
			idx += 16;
		}
#else
		const int end = lut_num - (imax + 1);
		for (int i = 0; i < end; i++)
		{
			int v = (idx++) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
#endif
	}

	inline void setLUTBoundaryLinearScale(uchar* lut, const int lut_num, const uchar imin, const uchar imax, float scale)
	{
		if (imax - imin == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}

		const float d = (float)(lut[imax] - lut[imin]) / (float)(imax - imin) * scale;

		//left side;
		int offset = lut[imin];

		uchar* l = &lut[0];
		int idx = -imin;
		for (int i = 0; i < imin; i++)
		{
			float v = (idx++) * d + offset;
			//int v = (i - imin) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}

		//right side
		offset = lut[imax];
		l = &lut[imax + 1];
		idx = 1;
		const int end = lut_num - (imax + 1);
		//for (int i = imax + 1; i < lut_num; i++)
		for (int i = 0; i < end; i++)
		{
			float v = (idx++) * d + offset;
			//lut[i] = max(0, min(lut_num - 1, v));
			*l++ = saturate_cast<uchar>(v);
		}
	}

	inline void setLUTBoundaryLinearClip(uchar* lut, const int lut_num, const uchar imin, const uchar imax, const int pm_num)
	{
		if (imax - imin == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}

		int pm = pm_num;

		const float d = (float)(lut[imax] - lut[imin]) / (float)(imax - imin);

		//left side;
		int offset = lut[imin];

		uchar* l = &lut[0];
		int idx = -imin;
		for (int i = 0; i < imin; i++)
		{
			int v = saturate_cast<int>((idx++) * d + offset);
			//int v = (i - imin) * d + offset;
			//*l++ = max(0, min(lut_num - 1, v));
			int vv = max(0, lut[imin] - pm);
			*l++ = max(vv, min(lut_num - 1, v));
			//*l++ = saturate_cast<uchar>(v);
		}

		//right side
		offset = lut[imax];
		l = &lut[imax + 1];
		idx = 1;
		const int end = lut_num - (imax + 1);
		//for (int i = imax + 1; i < lut_num; i++)
		for (int i = 0; i < end; i++)
		{
			int v = saturate_cast<int>((idx++) * d + offset);
			int vv = min(lut_num - 1, lut[imax] + pm);
			*l++ = max(0, min(vv, v));
			//*l++ = saturate_cast<uchar>(v);
		}
	}

	inline void searchSetLUTMinMax(uchar* lut, const int lut_num)
	{
		uchar minv = 255;
		uchar maxv = 0;
		for (int i = 0; i < lut_num; i++)
		{
			if (lut[i] != 0)
			{
				minv = min(minv, lut[i]);
				maxv = max(maxv, lut[i]);
			}
		}

		if (lut[0] == 0) lut[0] = minv;
		if (lut[lut_num - 1] == 0) lut[lut_num - 1] = maxv;
	}

	inline void setLUTMinMaxReprecate(uchar* lut, const int lut_num)
	{
		uchar minv = 0;
		int argimin = 0;
		for (int i = 0; i < lut_num; i++)
		{
			if (lut[i] != 0)
			{
				argimin = i;
				minv = lut[i];
				break;
			}
		}
		//if (minv == 0) return;

		uchar maxv = lut_num - 1;
		int argimax = lut_num - 1;
		for (int i = lut_num - 1; i >= 0; i--)
		{
			if (lut[i] != 0)
			{
				argimax = i;
				maxv = lut[i];
				break;
			}
		}

		//memset(lut, minv, sizeof(uchar)*argimin);
		//memset(lut + argimax + 1, maxv, sizeof(uchar)*(lut_num - 1 - argimax));
		lut[0] = minv;
		lut[lut_num - 1] = maxv;
	}

	inline void setLUTMinMaxReprecate(uchar* lut, const int lut_num, const uchar imin, const uchar imax)
	{
		if (imax - imin == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}

		uchar v = lut[imin];
		memset(lut, v, sizeof(uchar) * imin);

		v = lut[imax];
		memset(lut + imax + 1, v, sizeof(uchar) * (lut_num - 1 - imax));
	}

	inline void setLUTMinMaxReprecate_pm_n(uchar* lut, const int lut_num, const int pm_num, const uchar imin, const uchar imax)
	{
		if (imax - imin == 0)
		{
			memset(lut, lut[imin], sizeof(uchar) * (lut_num));
			return;
		}
		uchar v = min(lut_num - 1, max(0, lut[imin] - pm_num));
		memset(lut, v, sizeof(uchar) * imin);

		v = min(lut_num - 1, max(0, lut[imax] + pm_num));
		memset(lut + imax + 1, v, sizeof(uchar) * (lut_num - 1 - imax));
	}

#pragma endregion 

#pragma region blur_lut
	static void blur32f(const uchar* src, uchar* dst, const int r, float div, const int end)
	{
		const __m256 mdiv = _mm256_set1_ps(div);
		//const __m256i mDIV = _mm256_set1_epi16(DIV);
		for (int i = 0; i < end; i++)
		{
			__m256i ms = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			for (int j = 1; j <= 2 * r; j++)
			{
				ms = _mm256_add_epi16(ms, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
			}
			__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms)))));//float2 integer and rounding
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(_mm256_permute2x128_si256(ms, ms, 1))))));
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			//__m256i v0 = _mm256_div_epi16(ms, mDIV);
			//_mm_store_si128((__m128i*)dst, _mm256_cvtepi16_epu8(v0));
			dst += 16;
			src += 16;
		}
	}

	static void blur32f_256unroll1(const uchar* src, uchar* dst, const int r, float div)
	{
		const __m256 mdiv = _mm256_set1_ps(div);
		for (int i = 0; i < 16; i++)
		{
			__m256i ms = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			for (int j = 1; j <= 2 * r; j++)
			{
				ms = _mm256_add_epi16(ms, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
			}
			__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms)))));//float2 integer and rounding
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms)))));
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			dst += 16;
			src += 16;
		}
	}

	static void blur32f_256unroll2(const uchar* src, uchar* dst, const int r, float div)
	{
		const __m256 mdiv = _mm256_set1_ps(div);
		for (int i = 0; i < 8; i++)
		{
			__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
			for (int j = 1; j <= 2 * r; j++)
			{
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
			}
			__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms1)))));//float2 integer and rounding
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms1)))));
			__m256i v2 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms2)))));//float2 integer and rounding
			__m256i v3 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms2)))));

			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v2, v3));
			//__m256i v0 = _mm256_div_epi16(ms, mDIV);
			//_mm_store_si128((__m128i*)dst, _mm256_cvtepi16_epu8(v0));
			dst += 32;
			src += 32;
		}
	}

	static void blur32f_256unroll4(const uchar* src, uchar* dst, const int r, float div)
	{
		const __m256 mdiv = _mm256_set1_ps(div);
		const int R = 2 * r;
		for (int i = 0; i < 4; i++)
		{
			__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
			__m256i ms3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 32)));
			__m256i ms4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 48)));
			for (int j = 1; j <= R; j += 2)
			{
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 16))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 32))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 32))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 48))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 48))));
				/*ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j))));
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j + 1))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j + 16))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j + 1 + 16))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j + 32))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j + 1 + 32))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j + 48))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_lddqu_si128((__m128i*)(src + j + 1 + 48))));*/
			}
			/*__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms1)))));
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms1)))));
			__m256i v2 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms2)))));
			__m256i v3 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms2)))));
			__m256i v4 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms3)))));
			__m256i v5 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms3)))));
			__m256i v6 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms4)))));
			__m256i v7 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms4)))));
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v2, v3));
			_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi32x2_epu8(v4, v5));
			_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi32x2_epu8(v6, v7));*/
			__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms1)))));
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms1)))));
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms2)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms2)))));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms3)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms3)))));
			_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms4)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms4)))));
			_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi32x2_epu8(v0, v1));

			dst += 64;
			src += 64;
		}
	}

	template <int r>
	static void blur32f_256unroll4(const uchar* src, uchar* dst)//can be vnni
	{
		const __m256 mdiv = _mm256_set1_ps(1.f / (2 * r + 1));
		for (int i = 0; i < 4; i++)
		{
			__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
			__m256i ms3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 32)));
			__m256i ms4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 48)));
			for (int j = 1; j <= 2 * r; j += 2)
			{
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 16))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 32))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 32))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 48))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 48))));
			}
			__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms1)))));
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms1)))));
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			//_mm_stream_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms2)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms2)))));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v0, v1));
			//_mm_stream_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms3)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms3)))));
			_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi32x2_epu8(v0, v1));
			//_mm_stream_si128((__m128i*)(dst + 32), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms4)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms4)))));
			_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi32x2_epu8(v0, v1));
			//_mm_stream_si128((__m128i*)(dst + 48), _mm256_cvtepi32x2_epu8(v0, v1));

			dst += 64;
			src += 64;
		}
	}

	static void blur32f_256unroll8(const uchar* src, uchar* dst, const int r, float div)
	{
		const __m256 mdiv = _mm256_set1_ps(div);
		for (int i = 0; i < 2; i++)
		{
			__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
			__m256i ms3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 32)));
			__m256i ms4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 48)));
			__m256i ms5 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 64)));
			__m256i ms6 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 80)));
			__m256i ms7 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 96)));
			__m256i ms8 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 112)));
			for (int j = 1; j <= 2 * r; j++)
			{
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 32))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 48))));
				ms5 = _mm256_add_epi16(ms5, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 64))));
				ms6 = _mm256_add_epi16(ms6, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 80))));
				ms7 = _mm256_add_epi16(ms7, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 96))));
				ms8 = _mm256_add_epi16(ms8, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 112))));
			}
			__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms1)))));
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms1)))));
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms2)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms2)))));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms3)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms3)))));
			_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms4)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms4)))));
			_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms5)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms5)))));
			_mm_store_si128((__m128i*)(dst + 64), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms6)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms6)))));
			_mm_store_si128((__m128i*)(dst + 80), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms7)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms7)))));
			_mm_store_si128((__m128i*)(dst + 96), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms8)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms8)))));
			_mm_store_si128((__m128i*)(dst + 112), _mm256_cvtepi32x2_epu8(v0, v1));

			dst += 128;
			src += 128;
		}
	}

	template <int r>
	static void blur32f_256unroll8(const uchar* src, uchar* dst)
	{
		const __m256 mdiv = _mm256_set1_ps(1.f / (2 * r + 1));
		for (int i = 0; i < 2; i++)
		{
			__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
			__m256i ms3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 32)));
			__m256i ms4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 48)));
			__m256i ms5 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 64)));
			__m256i ms6 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 80)));
			__m256i ms7 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 96)));
			__m256i ms8 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 112)));
			for (int j = 1; j <= 2 * r; j++)
			{
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 32))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 48))));
				ms5 = _mm256_add_epi16(ms5, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 64))));
				ms6 = _mm256_add_epi16(ms6, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 80))));
				ms7 = _mm256_add_epi16(ms7, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 96))));
				ms8 = _mm256_add_epi16(ms8, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 112))));
			}
			__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms1)))));
			__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms1)))));
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms2)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms2)))));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms3)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms3)))));
			_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms4)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms4)))));
			_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms5)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms5)))));
			_mm_store_si128((__m128i*)(dst + 64), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms6)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms6)))));
			_mm_store_si128((__m128i*)(dst + 80), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms7)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms7)))));
			_mm_store_si128((__m128i*)(dst + 96), _mm256_cvtepi32x2_epu8(v0, v1));
			v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms8)))));
			v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms8)))));
			_mm_store_si128((__m128i*)(dst + 112), _mm256_cvtepi32x2_epu8(v0, v1));

			dst += 128;
			src += 128;
		}
	}

	static void blur32f_256unroll16(const uchar* src, uchar* dst, const int r, float div)
	{
		const __m256 mdiv = _mm256_set1_ps(div);
		__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
		__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
		__m256i ms3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 32)));
		__m256i ms4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 48)));
		__m256i ms5 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 64)));
		__m256i ms6 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 80)));
		__m256i ms7 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 96)));
		__m256i ms8 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 112)));

		for (int j = 1; j <= 2 * r; j++)
		{
			ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
			ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
			ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 32))));
			ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 48))));
			ms5 = _mm256_add_epi16(ms5, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 64))));
			ms6 = _mm256_add_epi16(ms6, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 80))));
			ms7 = _mm256_add_epi16(ms7, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 96))));
			ms8 = _mm256_add_epi16(ms8, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 112))));
		}
		__m256i v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms1)))));
		__m256i v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms1)))));
		_mm_store_si128((__m128i*)dst, _mm256_cvtepi32x2_epu8(v0, v1));
		v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms2)))));
		v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms2)))));
		_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi32x2_epu8(v0, v1));
		v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms3)))));
		v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms3)))));
		_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi32x2_epu8(v0, v1));
		v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms4)))));
		v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms4)))));
		_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi32x2_epu8(v0, v1));
		v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms5)))));
		v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms5)))));
		_mm_store_si128((__m128i*)(dst + 64), _mm256_cvtepi32x2_epu8(v0, v1));
		v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms6)))));
		v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms6)))));
		_mm_store_si128((__m128i*)(dst + 80), _mm256_cvtepi32x2_epu8(v0, v1));
		v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms7)))));
		v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms7)))));
		_mm_store_si128((__m128i*)(dst + 96), _mm256_cvtepi32x2_epu8(v0, v1));
		v0 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(ms8)))));
		v1 = _mm256_cvtps_epi32(_mm256_mul_ps(mdiv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256hi_si128(ms8)))));
		_mm_store_si128((__m128i*)(dst + 112), _mm256_cvtepi32x2_epu8(v0, v1));
	}



	static void blur16s_256unroll4(const uchar* src, uchar* dst, const int r, int div)
	{
		const __m256i mdiv = _mm256_set1_epi16(div);
		const int R = 2 * r;
		for (int i = 0; i < 4; i++)
		{
			__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
			__m256i ms3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 32)));
			__m256i ms4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 48)));
			for (int j = 1; j <= R; j += 2)
			{
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 16))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 32))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 32))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 48))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 48))));
			}
			_mm_store_si128((__m128i*)dst, _mm256_cvtepi16_epu8(_mm256_div_epi16(ms1, mdiv)));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi16_epu8(_mm256_div_epi16(ms2, mdiv)));
			_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi16_epu8(_mm256_div_epi16(ms3, mdiv)));
			_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi16_epu8(_mm256_div_epi16(ms4, mdiv)));
			dst += 64;
			src += 64;
		}
	}

	static void blur3_16s_256unroll4(const uchar* src, uchar* dst)
	{
		for (int i = 0; i < 4; i++)
		{
			__m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src)));
			__m256i ms2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 16)));
			__m256i ms3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 32)));
			__m256i ms4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + 48)));
			ms1 = _mm256_add_epi16(ms1, ms1);
			ms2 = _mm256_add_epi16(ms2, ms2);
			ms3 = _mm256_add_epi16(ms3, ms3);
			ms4 = _mm256_add_epi16(ms4, ms4);
			for (int j = 1; j < 7; j += 2)
			{
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j))));
				ms1 = _mm256_add_epi16(ms1, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 16))));
				ms2 = _mm256_add_epi16(ms2, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 16))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 32))));
				ms3 = _mm256_add_epi16(ms3, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 32))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 48))));
				ms4 = _mm256_add_epi16(ms4, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src + j + 1 + 48))));
			}

			_mm_store_si128((__m128i*)(dst + 0), _mm256_cvtepi16_epu8(_mm256_srli_epi16(ms1, 3)));
			_mm_store_si128((__m128i*)(dst + 16), _mm256_cvtepi16_epu8(_mm256_srli_epi16(ms2, 3)));
			_mm_store_si128((__m128i*)(dst + 32), _mm256_cvtepi16_epu8(_mm256_srli_epi16(ms3, 3)));
			_mm_store_si128((__m128i*)(dst + 48), _mm256_cvtepi16_epu8(_mm256_srli_epi16(ms4, 3)));
			dst += 64;
			src += 64;
		}
	}

	void LocalLUTUpsample::boxBlurLUT(uchar* srcdst_lut, uchar* lut_buff, const int lut_num, const int r)
	{
		if (r == 0)return;
		CV_Assert(srcdst_lut != lut_buff);

		const float div = 1.f / (2 * r + 1);
		const int DIV = 2 * r + 1;

		//reprecate
		for (int i = 0; i < r; i++)
		{
			lut_buff[i] = srcdst_lut[0];
		}
		memcpy(lut_buff + r, srcdst_lut, sizeof(uchar) * lut_num);
		//reprecate
		for (int i = 0; i < r; i++)
		{
			lut_buff[r + lut_num + i] = srcdst_lut[lut_num - 1];
		}

#ifdef LOCALLUT_USE_SIMD	
		if (lut_num == 256)
		{
			//blur32f_256unroll2(lut_buff, srcdst_lut, r, div);
			//blur32f_256unroll4(lut_buff, srcdst_lut, r, div);
			//blur32f_256unroll8(lut_buff, srcdst_lut, r, div);
			switch (r)
			{
			case 1:blur32f_256unroll4<1>(lut_buff, srcdst_lut); break;
			case 2:blur32f_256unroll4<2>(lut_buff, srcdst_lut); break;
			case 3:blur32f_256unroll4<3>(lut_buff, srcdst_lut); break;
			case 4:blur32f_256unroll4<4>(lut_buff, srcdst_lut); break;
			case 5:blur32f_256unroll4<5>(lut_buff, srcdst_lut); break;
			case 6:blur32f_256unroll4<6>(lut_buff, srcdst_lut); break;
			case 7:blur32f_256unroll4<7>(lut_buff, srcdst_lut); break;
			default:blur32f_256unroll4(lut_buff, srcdst_lut, r, div); break;
			}
			//blur16s_256unroll4(lut_buff, srcdst_lut, r, DIV);
			//blur3_16s_256unroll4(lut_buff, srcdst_lut);
		}
		else
		{
			blur32f(lut_buff, srcdst_lut, r, div, lut_num / 16);
		}
#else
		//summed area table but it is not efficient for small r
		int sum = 0;
		for (int j = 0; j <= 2 * r; j++)
		{
			sum += lut_buff[j];
		}
		int sump = sum;
		srcdst_lut[0] = cvRound(div * sum);
		for (int i = 1; i < lut_num; i++)
		{
			sum = sump - lut_buff[i - 1] + lut_buff[i + 2 * r];
			srcdst_lut[i] = cvRound(div * sum);
			sump = sum;
		}
		/*
		FIR conv
		for (int i = 0; i < lut_num; i++)
		{
			int sum = 0;
			for (int j = 0; j <= 2 * r; j++)
			{
				sum += lut_buff[i + j];
			}
			srcdst_lut[i] = cvRound(div * sum);
		}
		*/
#endif
	}

#pragma endregion

#pragma region buildLUT
	// WTA approach
	void LocalLUTUpsample::buildLocalLUTTensorFrequencyMaxWTA8U(const int lut_num, const int r, const int range_div, const int lut_filter_radius, const BOUNDARY lut_boundary_method)
	{
		int shift = (int)log2(range_div);

		// L1 norm weighting
		vector<uchar> offset((2 * r + 1) * (2 * r + 1));
		if (r == 0)offset[0] = 1;
		else if (r == 1)
		{
			offset[0] = 1; offset[1] = 2; offset[2] = 1;
			offset[3] = 2; offset[4] = 4; offset[5] = 2;
			offset[6] = 1; offset[7] = 2; offset[8] = 1;
		}
		else if (r == 2)
		{
			offset[0] = 1; offset[1] = 2; offset[2] = 4; offset[3] = 2; offset[4] = 1;
			offset[5] = 2; offset[6] = 4; offset[7] = 8; offset[8] = 4; offset[9] = 2;
			offset[10] = 4; offset[11] = 8; offset[12] = 16; offset[13] = 8; offset[14] = 4;
			offset[15] = 2; offset[16] = 4; offset[17] = 8; offset[18] = 4; offset[19] = 2;
			offset[20] = 1; offset[21] = 2; offset[22] = 4; offset[23] = 2; offset[24] = 1;
		}
		else
		{
			int idx = 0;
			const uchar off = 2 * r + 1;
			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					offset[idx++] = off - abs(j) - abs(i);
				}
			}
		}

		//#pragma omp parallel for schedule (dynamic)
		for (int y = 0; y < lowres_size.height; y++)
		{
			uchar* lut_buff = (uchar*)_mm_malloc(lut_num + 2 * lut_filter_radius, 32);
			const float frac = 1.f / (float)range_div;

			Mat lutmap(lut_num, lut_num, CV_8UC3);
			for (int x = 0; x < lowres_size.width; x++)
			{
				//頻度の集計
				uchar* lutb = LUT_TensorAoS_B.ptr<uchar>(y) + lut_num * x;//LUT_B.channels()=lut_num
				uchar* lutg = LUT_TensorAoS_G.ptr<uchar>(y) + lut_num * x;
				uchar* lutr = LUT_TensorAoS_R.ptr<uchar>(y) + lut_num * x;

				lutmap.setTo(0);

				uchar iminb = lut_num - 1;
				uchar iming = lut_num - 1;
				uchar iminr = lut_num - 1;
				uchar ominb = lut_num - 1;
				uchar oming = lut_num - 1;
				uchar ominr = lut_num - 1;
				uchar imaxb = 0;
				uchar imaxg = 0;
				uchar imaxr = 0;
				uchar omaxb = 0;
				uchar omaxg = 0;
				uchar omaxr = 0;

				int idx = 0;
				for (int i = -r; i <= r; i++)
				{
					uchar* y_src = src_low_border.ptr<uchar>(y + i + r, x);
					uchar* y_res = prc_low_border.ptr<uchar>(y + i + r, x);

					for (int j = -r; j <= r; j++)
					{
						uchar adder = offset[idx++];

						// Quantization
						const uchar inp_b = y_src[0];
						const uchar inp_g = y_src[1];
						const uchar inp_r = y_src[2];
						const uchar oup_b = y_res[0];
						const uchar oup_g = y_res[1];
						const uchar oup_r = y_res[2];

						// 入出力画素の最小値，最大値を記録(入出力値がとる範囲だけで頻度を調べる)
						iminb = min(iminb, inp_b);
						iming = min(iming, inp_g);
						iminr = min(iminr, inp_r);
						imaxb = max(imaxb, inp_b);
						imaxg = max(imaxg, inp_g);
						imaxr = max(imaxr, inp_r);

						ominb = min(ominb, oup_b);
						oming = min(oming, oup_g);
						ominr = min(ominr, oup_r);
						omaxb = max(omaxb, oup_b);
						omaxg = max(omaxg, oup_g);
						omaxr = max(omaxr, oup_r);

						// Count the frequency 
						lutmap.at<uchar>(inp_b, 3 * oup_b + 0) += adder;
						lutmap.at<uchar>(inp_g, 3 * oup_g + 1) += adder;
						lutmap.at<uchar>(inp_r, 3 * oup_r + 2) += adder;

						y_src += 3;
						y_res += 3;
					}
				}

				// Initialize LUT
				memset(lutb, 0, sizeof(uchar) * lut_num);
				memset(lutg, 0, sizeof(uchar) * lut_num);
				memset(lutr, 0, sizeof(uchar) * lut_num);

				// Determine the output by WTA
				for (int j = iminb; j <= imaxb; j++)
				{
					uchar max_b = 0;
					uchar* l = lutmap.ptr<uchar>(j, ominb) + 0;
					for (int i = ominb; i <= omaxb; i++)
					{
						uchar v = *l;
						if (v > max_b)
						{
							max_b = v;
							lutb[j] = i;
						}
						l += 3;
					}
				}
				for (int j = iming; j <= imaxg; j++)
				{
					uchar max_g = 0;
					uchar* l = lutmap.ptr<uchar>(j, oming) + 1;
					for (int i = oming; i <= omaxg; i++)
					{
						uchar v = *l;
						if (v > max_g)
						{
							max_g = v;
							lutg[j] = i;
						}
						l += 3;
					}
				}

				for (int j = iminr; j <= imaxr; j++)
				{
					uchar max_r = 0;
					uchar* l = lutmap.ptr<uchar>(j, ominr) + 2;
					for (int i = ominr; i <= omaxr; i++)
					{
						uchar v = *l;
						if (v > max_r)
						{
							max_r = v;
							lutr[j] = i;
						}
						l += 3;
					}
				}

				if (lut_boundary_method == BOUNDARY::REPLICATE)
				{
					setLUTMinMaxReprecate_pm_n(lutb, lut_num, boundary_replicate_offset, iminb, imaxb);
					setLUTMinMaxReprecate_pm_n(lutg, lut_num, boundary_replicate_offset, iming, imaxg);
					setLUTMinMaxReprecate_pm_n(lutr, lut_num, boundary_replicate_offset, iminr, imaxr);
				}
				else if (lut_boundary_method == BOUNDARY::MINMAX_OUTPUT)
				{
					setLUTMinMax(lutb, lut_num, ominb, omaxb, iminb, imaxb);
					setLUTMinMax(lutg, lut_num, oming, omaxg, iming, imaxg);
					setLUTMinMax(lutr, lut_num, ominr, omaxr, iminr, imaxr);
				}
				else if (lut_boundary_method == BOUNDARY::MINMAX0_255)
				{
					//padding boundary by 0 and 255
					const uchar minv = 0;
					const uchar maxv = lut_num - 1;
					setLUTMinMax(lutb, lut_num, minv, maxv, iminb, imaxb);
					setLUTMinMax(lutg, lut_num, minv, maxv, iming, imaxg);
					setLUTMinMax(lutr, lut_num, minv, maxv, iminr, imaxr);
				}
				else if (lut_boundary_method == BOUNDARY::LINEAR)
				{
					setLUTBoundaryLinear(lutb, lut_num, iminb, imaxb);
					setLUTBoundaryLinear(lutg, lut_num, iming, imaxg);
					setLUTBoundaryLinear(lutr, lut_num, iminr, imaxr);
				}
				else if (lut_boundary_method == BOUNDARY::LINEAR_LAST2)
				{
					setLUTBoundaryLinearLast2(lutb, lut_num, iminb, imaxb);
					setLUTBoundaryLinearLast2(lutg, lut_num, iming, imaxg);
					setLUTBoundaryLinearLast2(lutr, lut_num, iminr, imaxr);
				}
				else if (lut_boundary_method == BOUNDARY::EXPERIMENT1)
				{
					//setLUTBoundaryLinearScale(lutb, lut_num, iminb, imaxb, 1.f + 0.1f*leplicate_offset);
					//setLUTBoundaryLinearScale(lutg, lut_num, iming, imaxg, 1.f + 0.1f*leplicate_offset);
					//setLUTBoundaryLinearScale(lutr, lut_num, iminr, imaxr, 1.f + 0.1f*leplicate_offset);
					setLUTBoundaryLinearClip(lutb, lut_num, iminb, imaxb, boundary_replicate_offset);
					setLUTBoundaryLinearClip(lutg, lut_num, iming, imaxg, boundary_replicate_offset);
					setLUTBoundaryLinearClip(lutr, lut_num, iminr, imaxr, boundary_replicate_offset);

				}
				else if (lut_boundary_method == BOUNDARY::EXPERIMENT2)
				{
					//for compare
					setLUTMinMaxReprecate_pm_n(lutb, lut_num, boundary_replicate_offset, iminb, imaxb);
					setLUTMinMaxReprecate_pm_n(lutg, lut_num, boundary_replicate_offset, iming, imaxg);
					setLUTMinMaxReprecate_pm_n(lutr, lut_num, boundary_replicate_offset, iminr, imaxr);
				}
				else if (lut_boundary_method == BOUNDARY::NO_INTERPOLATION)
				{
					goto LOOP_END;
				}

				// Linearly interpolating in range direction
				interpolateLUT(lutb, lut_num, iminb, imaxb);
				interpolateLUT(lutg, lut_num, iming, imaxg);
				interpolateLUT(lutr, lut_num, iminr, imaxr);

				//filtering LUT
				if (lut_filter_radius != 0)
				{
					boxBlurLUT(lutb, lut_buff, lut_num, lut_filter_radius);
					boxBlurLUT(lutg, lut_buff, lut_num, lut_filter_radius);
					boxBlurLUT(lutr, lut_buff, lut_num, lut_filter_radius);
				}
			LOOP_END:
				;
			}
			_mm_free(lut_buff);
		}
	}

	// WTA approach(16bit)
	void LocalLUTUpsample::buildLocalLUTTensorFrequencyMaxWTA16U(const int lut_num, const int r, const int ratio)
	{
		const int shift = (int)log2(ratio);
		vector<ushort> offset((2 * r + 1) * (2 * r + 1));
		if (r == 0)offset[0] = 1;
		else if (r == 1)
		{
			offset[0] = 1; offset[1] = 2; offset[2] = 1;
			offset[3] = 2; offset[4] = 4; offset[5] = 2;
			offset[6] = 1; offset[7] = 2; offset[8] = 1;
		}
		else if (r == 2)
		{
			offset[0] = 1; offset[1] = 2; offset[2] = 4; offset[3] = 2; offset[4] = 1;
			offset[5] = 2; offset[6] = 4; offset[7] = 8; offset[8] = 4; offset[9] = 2;
			offset[10] = 4; offset[11] = 8; offset[12] = 16; offset[13] = 8; offset[14] = 4;
			offset[15] = 2; offset[16] = 4; offset[17] = 8; offset[18] = 4; offset[19] = 2;
			offset[20] = 1; offset[21] = 2; offset[22] = 4; offset[23] = 2; offset[24] = 1;
		}
		else
		{
			int idx = 0;
			const ushort off = 2 * r + 1;
			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					offset[idx++] = off - abs(j) - abs(i);
				}
			}
		}

#pragma omp parallel for schedule (dynamic)
		for (int y = 0; y < lowres_size.height; y++)
		{
			const float frac = 1.f / (float)ratio;

			Mat lutmap(lut_num, lut_num, CV_16UC3);
			for (int x = 0; x < lowres_size.width; x++)
			{
				uchar* lutb = LUT_TensorAoS_B.ptr<uchar>(y) + LUT_TensorAoS_B.channels() * x;
				uchar* lutg = LUT_TensorAoS_G.ptr<uchar>(y) + LUT_TensorAoS_B.channels() * x;
				uchar* lutr = LUT_TensorAoS_R.ptr<uchar>(y) + LUT_TensorAoS_B.channels() * x;

				lutmap.setTo(0);//clear 2D LUT map (heavy)

				uchar iminb = 255;
				uchar iming = 255;
				uchar iminr = 255;
				uchar ominb = 255;
				uchar oming = 255;
				uchar ominr = 255;
				uchar imaxb = 0;
				uchar imaxg = 0;
				uchar imaxr = 0;
				uchar omaxb = 0;
				uchar omaxg = 0;
				uchar omaxr = 0;

				int idx = 0;

				uchar v = prc_low_border.at<uchar>(y + r, +3 * (x + r) + 0);
				offset_map.at<uchar>(y, 3 * x + 0) = v - ((v >> shift) << shift);
				v = prc_low_border.at<uchar>(y + r, +3 * (x + r) + 1);
				offset_map.at<uchar>(y, 3 * x + 1) = v - ((v >> shift) << shift);
				v = prc_low_border.at<uchar>(y + r, +3 * (x + r) + 2);
				offset_map.at<uchar>(y, 3 * x + 2) = v - ((v >> shift) << shift);

				for (int i = -r; i <= r; i++)
				{
					uchar* y_src = src_low_border.ptr<uchar>(y + i + r) + 3 * (x);
					uchar* y_res = prc_low_border.ptr<uchar>(y + i + r) + 3 * (x);

					for (int j = -r; j <= r; j++)
					{
						ushort adder = offset[idx++];
						uchar inp_b = y_src[0] >> shift;
						uchar inp_g = y_src[1] >> shift;
						uchar inp_r = y_src[2] >> shift;
						uchar oup_b = y_res[0] >> shift;
						uchar oup_g = y_res[1] >> shift;
						uchar oup_r = y_res[2] >> shift;

						y_src += 3;
						y_res += 3;
						iminb = min(iminb, inp_b);
						iming = min(iming, inp_g);
						iminr = min(iminr, inp_r);
						imaxb = max(imaxb, inp_b);
						imaxg = max(imaxg, inp_g);
						imaxr = max(imaxr, inp_r);

						ominb = min(ominb, oup_b);
						oming = min(oming, oup_g);
						ominr = min(ominr, oup_r);
						omaxb = max(omaxb, oup_b);
						omaxg = max(omaxg, oup_g);
						omaxr = max(omaxr, oup_r);
						//lutmap.at<uchar>(inp_b, 3 * oup_b + 0)++;		
						//lutmap.at<uchar>(inp_g, 3 * oup_g + 1)++;
						//lutmap.at<uchar>(inp_r, 3 * oup_r + 2)++;
						lutmap.at<ushort>(inp_b, 3 * oup_b + 0) += adder;//count frequency
						lutmap.at<ushort>(inp_g, 3 * oup_g + 1) += adder;
						lutmap.at<ushort>(inp_r, 3 * oup_r + 2) += adder;
					}
				}

				for (int i = 0; i < lut_num; i++)
				{
					lutb[i] = 0;
					lutg[i] = 0;
					lutr[i] = 0;
				}

				// find most frequent value
				for (int j = iminb; j <= imaxb; j++)
				{
					ushort max_b = 0;
					ushort* l = lutmap.ptr<ushort>(j) + 3 * ominb;
					for (int i = ominb; i <= omaxb; i++)
					{
						ushort v = l[0];
						if (v > max_b)
						{
							max_b = v;
							lutb[j] = i;
						}
						l += 3;
					}
				}
				for (int j = iming; j <= imaxg; j++)
				{
					ushort max_g = 0;
					ushort* l = lutmap.ptr<ushort>(j) + 3 * oming;
					for (int i = oming; i <= omaxg; i++)
					{
						ushort v = l[1];
						if (v > max_g)
						{
							max_g = v;
							lutg[j] = i;
						}
						l += 3;
					}
				}
				for (int j = iminr; j <= imaxr; j++)
				{
					ushort max_r = 0;
					ushort* l = lutmap.ptr<ushort>(j) + 3 * ominr;
					for (int i = ominr; i <= omaxr; i++)
					{
						ushort v = l[2];
						if (v > max_r)
						{
							max_r = v;
							lutr[j] = i;
						}
						l += 3;
					}
				}

				//filling boundary
				const uchar minv = 0;
				const uchar maxv = lut_num - 1;
				//Blue
				if (lutb[0] == 0) lutb[0] = minv;
				if (lutb[lut_num - 1] == 0) lutb[lut_num - 1] = maxv;
				//Green
				if (lutg[0] == 0) lutg[0] = minv;
				if (lutg[lut_num - 1] == 0) lutg[lut_num - 1] = maxv;
				//Red
				if (lutr[0] == 0) lutr[0] = minv;
				if (lutr[lut_num - 1] == 0) lutr[lut_num - 1] = maxv;

				//interlation for range
				//Blue
				for (int i = 1; i < lut_num - 1; i++)
				{
					if (lutb[i] == 0)
					{
						int next_d = 1;

						while (lutb[i + next_d] == 0)
						{
							next_d++;
						}
						for (int j = 0; j < next_d; j++)
						{
							lutb[i + j] = (((next_d - j) * lutb[i - 1]) + ((j + 1) * lutb[i + next_d])) / (next_d + 1);
						}

						i = i + next_d;
					}
				}
				//Green
				for (int i = 1; i < lut_num - 1; i++)
				{
					if (lutg[i] == 0)
					{
						int next_d = 1;
						while (lutg[i + next_d] == 0)
						{
							next_d++;
						}

						for (int j = 0; j < next_d; j++)
						{
							lutg[i + j] = (((next_d - j) * lutg[i - 1]) + ((j + 1) * lutg[i + next_d])) / (next_d + 1);
						}

						i = i + next_d;
					}
				}
				//Red
				for (int i = 1; i < lut_num - 1; i++)
				{
					if (lutr[i] == 0)
					{
						int next_d = 1;

						while (lutr[i + next_d] == 0)
						{
							next_d++;
						}

						for (int j = 0; j < next_d; j++)
						{
							lutr[i + j] = (((next_d - j) * lutr[i - 1]) + ((j + 1) * lutr[i + next_d])) / (next_d + 1);
						}

						i = i + next_d;
					}
				}
			}
		}
	}

	bool comp(const Point3f& l, const Point3f  h)
	{
		return l.z > h.z;
	}

	static void setLInfDistanceOrder2(AutoBuffer<Point>& order, const int r)
	{
		vector<uchar> l1((2 * r + 1) * (2 * r + 1));
		// compute Linf norm
		int k = 0;
		for (int i = -r; i <= r; i++)
		{
			for (int j = -r; j <= r; j++)
			{
				//l1[k] = abs(i) + abs(j);
				l1[k] = max(abs(i), abs(j));
				k++;
			}
		}
		k = 0;
		// Sort descending Linf norm order
		for (int i = 0; i <= 2 * r; i++)
		{
			for (int j = 0; j < (2 * r + 1) * (2 * r + 1); j++)
			{
				if (l1[j] == 2 * r - i)
				{
					order[k].x = j % (2 * r + 1);
					order[k].y = j / (2 * r + 1);
					k++;
				}
			}
		}
	}

	static void setLInfDistanceOrder(AutoBuffer<Point>& order, const int r)
	{
		vector<Point3f> l2((2 * r + 1) * (2 * r + 1));
		// compute L2 norm
		int k = 0;
		for (int i = -r; i <= r; i++)
		{
			for (int j = -r; j <= r; j++)
			{
				l2[k] = Point3f(float(j), float(i), (float)(max(abs(i), abs(j))));
				k++;
			}
		}
		sort(l2.begin(), l2.end(), comp);

		for (int j = 0; j < order.size(); j++)
		{
			order[j].x = (int)l2[j].x + r;
			order[j].y = (int)l2[j].y + r;
		}
	}

	static void setL1DistanceOrder(AutoBuffer<Point>& order, const int r)
	{
		vector<Point3f> l2((2 * r + 1) * (2 * r + 1));
		// compute L2 norm
		int k = 0;
		for (int i = -r; i <= r; i++)
		{
			for (int j = -r; j <= r; j++)
			{
				l2[k] = Point3f(float(j), float(i), (float)(abs(i) + abs(j)));
				k++;
			}
		}
		sort(l2.begin(), l2.end(), comp);

		for (int j = 0; j < order.size(); j++)
		{
			order[j].x = (int)l2[j].x + r;
			order[j].y = (int)l2[j].y + r;
		}
	}

	static void setL2DistanceOrder(AutoBuffer<Point>& order, const int r)
	{
		vector<Point3f> l2((2 * r + 1) * (2 * r + 1));
		// compute L2 norm
		int k = 0;
		for (int i = -r; i <= r; i++)
		{
			for (int j = -r; j <= r; j++)
			{
				l2[k] = Point3f(float(j), float(i), hypot((float)i, (float)j));
				k++;
			}
		}
		sort(l2.begin(), l2.end(), comp);

		for (int j = 0; j < order.size(); j++)
		{
			order[j].x = (int)l2[j].x + r;
			order[j].y = (int)l2[j].y + r;
		}
	}

	static void setL2DistanceOrderEven(AutoBuffer<Point>& order, const int r)
	{
		vector<Point3f> l2((2 * r + 1) * (2 * r + 1));
		// compute L2 norm
		int k = 0;
		for (int i = -r + 1; i <= r; i++)
		{
			for (int j = -r + 1; j <= r; j++)
			{
				l2[k] = Point3f(float(j), float(i), hypot((float)i, (float)j));
				k++;
			}
		}
		sort(l2.begin(), l2.end(), comp);

		for (int j = 0; j < order.size(); j++)
		{
			order[j].x = (int)l2[j].x + r - 1;
			order[j].y = (int)l2[j].y + r - 1;
		}
	}

	template<int window_size>
	static inline void setScatterPlotsRGB(const int x, const int y, const AutoBuffer<Point>& order, const Mat& src, const Mat& dst, uchar* lutb, uchar* lutg, uchar* lutr, __m128i& imin, __m128i& imax)
	{
		for (int i = 0; i < window_size; i++)
		{
			const int n = order[i].y;
			const int m = order[i].x;
			const uchar* y_src = src.ptr<uchar>(y + n, x + m);
			const uchar* y_res = dst.ptr<uchar>(y + n, x + m);

			const uchar inp_b = y_src[0];
			const uchar inp_g = y_src[1];
			const uchar inp_r = y_src[2];

			lutb[inp_b] = y_res[0];
			lutg[inp_g] = y_res[1];
			lutr[inp_r] = y_res[2];
			__m128i v = _mm_setr_epi8(inp_b, inp_g, inp_r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
			imin = _mm_min_epu8(imin, v);
			imax = _mm_max_epu8(imax, v);
		}
	}

	static inline void setScatterPlotsRGB(int window_size, const int x, const int y, const AutoBuffer<Point>& order, const Mat& src, const Mat& dst, uchar* lutb, uchar* lutg, uchar* lutr, __m128i& imin, __m128i& imax)
	{
		for (int i = 0; i < window_size; i++)
		{
			const int n = order[i].y;
			const int m = order[i].x;
			const uchar* y_src = src.ptr<uchar>(y + n, x + m);
			const uchar* y_res = dst.ptr<uchar>(y + n, x + m);

			const uchar inp_b = y_src[0];
			const uchar inp_g = y_src[1];
			const uchar inp_r = y_src[2];

			lutb[inp_b] = y_res[0];
			lutg[inp_g] = y_res[1];
			lutr[inp_r] = y_res[2];
			__m128i v = _mm_setr_epi8(inp_b, inp_g, inp_r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
			imin = _mm_min_epu8(imin, v);
			imax = _mm_max_epu8(imax, v);
		}
	}

	template<int lut_boundary_method, bool isSoA>
	void LocalLUTUpsample::buildLocalLUTTensorDistanceMINInvoker(const int distance, const int lut_num, const int r, const int range_div, const int lut_filter_radius)
	{
		const int shift = (int)log2(range_div);
		const int d = 2 * r + 1;
		const int window_size = d * d;
		AutoBuffer<Point> order(window_size);

		switch (distance)
		{
		case 0: setLInfDistanceOrder(order, r); break;
		case 1: setL1DistanceOrder(order, r); break;
		default:
		case 2: setL2DistanceOrder(order, r); break;
		}

		if (src_low_border.channels() == 1)
		{
			for (int y = 0; y < lowres_size.height; y++)
			{
				uchar* lut_buff = (uchar*)_mm_malloc(lut_num + 2 * lut_filter_radius, 32);
				for (int x = 0; x < lowres_size.width; x++)
				{
					uchar* lutb = LUT_TensorAoS_B.ptr<uchar>(y) + lut_num * x;//LUT_B.channels()=lut_num
					memset(lutb, 0, sizeof(uchar) * lut_num);

					uchar iminb = lut_num - 1;
					uchar imaxb = 0;

					if constexpr (lut_boundary_method == (int)BOUNDARY::MINMAX_OUTPUT)
					{
						uchar ominb = lut_num - 1;
						uchar omaxb = 0;

						for (int i = 0; i < window_size; i++)
						{
							const int n = order[i].y;
							const int m = order[i].x;
							const uchar inp_b = src_low_border.at<uchar>(y + n, x + m);
							const uchar oup_b = prc_low_border.at<uchar>(y + n, x + m);

							lutb[inp_b] = oup_b;

							iminb = min(iminb, inp_b);
							imaxb = max(imaxb, inp_b);
							ominb = min(oup_b, ominb);
							omaxb = max(oup_b, omaxb);
						}
						//padding boundary by min max values
						setLUTMinMax(lutb, lut_num, ominb, omaxb, iminb, imaxb);
					}
					else
					{
						for (int i = 0; i < window_size; i++)
						{
							const int n = order[i].y;
							const int m = order[i].x;
							const uchar inp_b = src_low_border.at<uchar>(y + n, x + m);
							lutb[inp_b] = prc_low_border.at<uchar>(y + n, x + m);

							iminb = min(iminb, inp_b);
							imaxb = max(imaxb, inp_b);
						}

						if constexpr (lut_boundary_method == (int)BOUNDARY::MINMAX0_255)
						{
							//padding boundary by 0 and 255
							const uchar minv = 0;
							const uchar maxv = lut_num - 1;
							setLUTMinMax(lutb, lut_num, minv, maxv, iminb, imaxb);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::REPLICATE)
						{
							setLUTMinMaxReprecate_pm_n(lutb, lut_num, boundary_replicate_offset, iminb, imaxb);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::LINEAR)
						{
							setLUTBoundaryLinear(lutb, lut_num, iminb, imaxb);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::LINEAR_LAST2)
						{
							setLUTBoundaryLinearLast2(lutb, lut_num, iminb, imaxb);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::EXPERIMENT1)
						{

							setLUTBoundaryLinearFlatSwich(lutb, lut_num, iminb, imaxb, (lutb[iminb] + lutb[imaxb]) >> 1);
							/*
							setLUTBoundaryLinearScale(lutb, lut_num, iminb, imaxb, 1.f + 0.1f*leplicate_offset);
							setLUTBoundaryLinearScale(lutg, lut_num, iming, imaxg, 1.f + 0.1f*leplicate_offset);
							setLUTBoundaryLinearScale(lutr, lut_num, iminr, imaxr, 1.f + 0.1f*leplicate_offset);
							*/

							//setLUTBoundaryLinearClip(lutb, lut_num, iminb, imaxb, leplicate_offset);
							//setLUTBoundaryLinearClip(lutg, lut_num, iming, imaxg, leplicate_offset);
							//setLUTBoundaryLinearClip(lutr, lut_num, iminr, imaxr, leplicate_offset);

						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::EXPERIMENT2)
						{
							//for compare
							setLUTBoundaryLinear(lutb, lut_num, iminb, imaxb);
							//setLUTMinMaxReprecate_pm_n(lutb, lut_num, leplicate_offset, iminb, imaxb);
							//setLUTMinMaxReprecate_pm_n(lutg, lut_num, leplicate_offset, iming, imaxg);
							//setLUTMinMaxReprecate_pm_n(lutr, lut_num, leplicate_offset, iminr, imaxr);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::NO_INTERPOLATION)
						{
							continue;//skip interpolation
						}
					}

					// Linearly interpolating in range direction
					interpolateLUT(lutb, lut_num, iminb, imaxb);

					//filtering LUT
					if (lut_filter_radius != 0)
					{
						boxBlurLUT(lutb, lut_buff, lut_num, lut_filter_radius);
					}
				}
				_mm_free(lut_buff);
			}
		}
		else if (src_low_border.channels() == 3)
		{
#pragma omp parallel for schedule (dynamic)
			for (int y = 0; y < lowres_size.height; y++)
			{
				uchar* lut_buff = (uchar*)_mm_malloc(lut_num + 2 * lut_filter_radius, AVX_ALIGN);
				uchar* lutb, * lutg, * lutr;
				uchar** bptr, ** gptr, ** rptr;

				if constexpr (isSoA)
				{
					lutb = (uchar*)_mm_malloc(lut_num, AVX_ALIGN);
					lutg = (uchar*)_mm_malloc(lut_num, AVX_ALIGN);
					lutr = (uchar*)_mm_malloc(lut_num, AVX_ALIGN);

					bptr = (uchar**)_mm_malloc(sizeof(uchar*) * lut_num, AVX_ALIGN);
					gptr = (uchar**)_mm_malloc(sizeof(uchar*) * lut_num, AVX_ALIGN);
					rptr = (uchar**)_mm_malloc(sizeof(uchar*) * lut_num, AVX_ALIGN);
					for (int i = 0; i < lut_num; i++)
					{
						bptr[i] = LUT_TensorSoA_B[i].ptr<uchar>(y);
						gptr[i] = LUT_TensorSoA_G[i].ptr<uchar>(y);
						rptr[i] = LUT_TensorSoA_R[i].ptr<uchar>(y);
					}
				}

				for (int x = 0; x < lowres_size.width; x++)
				{
					if constexpr (!isSoA)
					{
						lutb = LUT_TensorAoS_B.ptr<uchar>(y) + lut_num * x;//LUT_B.channels()=lut_num
						lutg = LUT_TensorAoS_G.ptr<uchar>(y) + lut_num * x;
						lutr = LUT_TensorAoS_R.ptr<uchar>(y) + lut_num * x;
					}

					memset(lutb, 0, sizeof(uchar) * lut_num);
					memset(lutg, 0, sizeof(uchar) * lut_num);
					memset(lutr, 0, sizeof(uchar) * lut_num);

					uchar iminb = lut_num - 1;
					uchar iming = lut_num - 1;
					uchar iminr = lut_num - 1;
					uchar imaxb = 0;
					uchar imaxg = 0;
					uchar imaxr = 0;

					__m128i imin = _mm_set1_epi8(lut_num - 1);
					__m128i imax = _mm_setzero_si128();
					if constexpr (lut_boundary_method == (int)BOUNDARY::MINMAX_OUTPUT)
					{
						uchar ominb = lut_num - 1;
						uchar oming = lut_num - 1;
						uchar ominr = lut_num - 1;
						uchar omaxb = 0;
						uchar omaxg = 0;
						uchar omaxr = 0;

						//int osumb = 0;
						//int osumg = 0;
						//int osumr = 0;
						for (int i = 0; i < window_size; i++)
						{
							int n = order[i].y;
							int m = order[i].x;

							uchar* y_src = src_low_border.ptr<uchar>(y + n, x + m);
							uchar* y_res = prc_low_border.ptr<uchar>(y + n, x + m);
							const uchar inp_b = y_src[0];
							const uchar inp_g = y_src[1];
							const uchar inp_r = y_src[2];
							const uchar oup_b = y_res[0];
							const uchar oup_g = y_res[1];
							const uchar oup_r = y_res[2];

							lutb[inp_b] = oup_b;
							lutg[inp_g] = oup_g;
							lutr[inp_r] = oup_r;

							//osumb += oup_b;
							//osumg += oup_g;
							//osumr += oup_r;

							iminb = min(iminb, inp_b);
							iming = min(iming, inp_g);
							iminr = min(iminr, inp_r);
							imaxb = max(imaxb, inp_b);
							imaxg = max(imaxg, inp_g);
							imaxr = max(imaxr, inp_r);

							ominb = min(oup_b, ominb);
							omaxb = max(oup_b, omaxb);
							oming = min(oup_g, oming);
							omaxg = max(oup_g, omaxg);
							ominr = min(oup_r, ominr);
							omaxr = max(oup_r, omaxr);
						}

						//padding boundary by min max values
						setLUTMinMax(lutb, lut_num, ominb, omaxb, iminb, imaxb);
						setLUTMinMax(lutg, lut_num, oming, omaxg, iming, imaxg);
						setLUTMinMax(lutr, lut_num, ominr, omaxr, iminr, imaxr);
						//setLUTBoundaryLinearFlatSwich(lutb, lut_num, iminb, imaxb, osumb / window_size);
						//setLUTBoundaryLinearFlatSwich(lutg, lut_num, iming, imaxg, osumg / window_size);
						//setLUTBoundaryLinearFlatSwich(lutr, lut_num, iminr, imaxr, osumr / window_size);
					}
					else
					{
						switch (window_size)
						{
						case 1: setScatterPlotsRGB<1>(x, y, order, src_low_border, prc_low_border, lutb, lutg, lutr, imin, imax); break;
						case 9: setScatterPlotsRGB<9>(x, y, order, src_low_border, prc_low_border, lutb, lutg, lutr, imin, imax); break;
						case 25: setScatterPlotsRGB<25>(x, y, order, src_low_border, prc_low_border, lutb, lutg, lutr, imin, imax); break;
						case 49: setScatterPlotsRGB<49>(x, y, order, src_low_border, prc_low_border, lutb, lutg, lutr, imin, imax); break;
						case 81: setScatterPlotsRGB<81>(x, y, order, src_low_border, prc_low_border, lutb, lutg, lutr, imin, imax); break;
						default: setScatterPlotsRGB(window_size, x, y, order, src_low_border, prc_low_border, lutb, lutg, lutr, imin, imax); break;
						}

						iminb = ((uchar*)&imin)[0];
						iming = ((uchar*)&imin)[1];
						iminr = ((uchar*)&imin)[2];
						imaxb = ((uchar*)&imax)[0];
						imaxg = ((uchar*)&imax)[1];
						imaxr = ((uchar*)&imax)[2];

						if constexpr (lut_boundary_method == (int)BOUNDARY::MINMAX0_255)
						{
							//padding boundary by 0 and 255
							const uchar minv = 0;
							const uchar maxv = lut_num - 1;
							setLUTMinMax(lutb, lut_num, minv, maxv, iminb, imaxb);
							setLUTMinMax(lutg, lut_num, minv, maxv, iming, imaxg);
							setLUTMinMax(lutr, lut_num, minv, maxv, iminr, imaxr);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::REPLICATE)
						{
							setLUTMinMaxReprecate_pm_n(lutb, lut_num, boundary_replicate_offset, iminb, imaxb);
							setLUTMinMaxReprecate_pm_n(lutg, lut_num, boundary_replicate_offset, iming, imaxg);
							setLUTMinMaxReprecate_pm_n(lutr, lut_num, boundary_replicate_offset, iminr, imaxr);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::LINEAR)
						{
							setLUTBoundaryLinear(lutb, lut_num, iminb, imaxb);
							setLUTBoundaryLinear(lutg, lut_num, iming, imaxg);
							setLUTBoundaryLinear(lutr, lut_num, iminr, imaxr);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::LINEAR_LAST2)
						{
							setLUTBoundaryLinearLast2(lutb, lut_num, iminb, imaxb);
							setLUTBoundaryLinearLast2(lutg, lut_num, iming, imaxg);
							setLUTBoundaryLinearLast2(lutr, lut_num, iminr, imaxr);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::EXPERIMENT1)
						{

							setLUTBoundaryLinearFlatSwich(lutb, lut_num, iminb, imaxb, (lutb[iminb] + lutb[imaxb]) >> 1);
							setLUTBoundaryLinearFlatSwich(lutg, lut_num, iming, imaxg, (lutg[iming] + lutg[imaxg]) >> 1);
							setLUTBoundaryLinearFlatSwich(lutr, lut_num, iminr, imaxr, (lutr[iminr] + lutr[imaxr]) >> 1);
							/*
							setLUTBoundaryLinearScale(lutb, lut_num, iminb, imaxb, 1.f + 0.1f*leplicate_offset);
							setLUTBoundaryLinearScale(lutg, lut_num, iming, imaxg, 1.f + 0.1f*leplicate_offset);
							setLUTBoundaryLinearScale(lutr, lut_num, iminr, imaxr, 1.f + 0.1f*leplicate_offset);
							*/

							//setLUTBoundaryLinearClip(lutb, lut_num, iminb, imaxb, leplicate_offset);
							//setLUTBoundaryLinearClip(lutg, lut_num, iming, imaxg, leplicate_offset);
							//setLUTBoundaryLinearClip(lutr, lut_num, iminr, imaxr, leplicate_offset);

						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::EXPERIMENT2)
						{
							//for compare
							setLUTBoundaryLinear(lutb, lut_num, iminb, imaxb);
							setLUTBoundaryLinear(lutg, lut_num, iming, imaxg);
							setLUTBoundaryLinear(lutr, lut_num, iminr, imaxr);
							//setLUTMinMaxReprecate_pm_n(lutb, lut_num, leplicate_offset, iminb, imaxb);
							//setLUTMinMaxReprecate_pm_n(lutg, lut_num, leplicate_offset, iming, imaxg);
							//setLUTMinMaxReprecate_pm_n(lutr, lut_num, leplicate_offset, iminr, imaxr);
						}
						else if constexpr (lut_boundary_method == (int)BOUNDARY::NO_INTERPOLATION)
						{
							continue; //skip interpolation
						}
					}

					// Linearly interpolating in range direction
					interpolateLUT(lutb, lut_num, iminb, imaxb);
					interpolateLUT(lutg, lut_num, iming, imaxg);
					interpolateLUT(lutr, lut_num, iminr, imaxr);
					//interpolateLUTStep(lutb, lut_num, iminb, imaxb);
					//interpolateLUTStep(lutg, lut_num, iming, imaxg);
					//interpolateLUTStep(lutr, lut_num, iminr, imaxr);

					//smoothing LUT
					boxBlurLUT(lutb, lut_buff, lut_num, lut_filter_radius);
					boxBlurLUT(lutg, lut_buff, lut_num, lut_filter_radius);
					boxBlurLUT(lutr, lut_buff, lut_num, lut_filter_radius);

					if constexpr (isSoA)
					{
						for (int i = 0; i < lut_num; i++)
						{
							bptr[i][x] = lutb[i];
							gptr[i][x] = lutg[i];
							rptr[i][x] = lutr[i];
						}
					}
				}
				_mm_free(lut_buff);
				if constexpr (isSoA)
				{
					_mm_free(lutb);
					_mm_free(lutg);
					_mm_free(lutr);

					_mm_free(bptr);
					_mm_free(gptr);
					_mm_free(rptr);
				}
			}
		}
	}

	//building tensor from src_low_border and dst_low_border
	void LocalLUTUpsample::buildLocalLUTTensorDistanceMIN(const int distance, const int lut_num, const int r, const int range_div, const int lut_filter_radius, const BOUNDARY lut_boundary_method)
	{
		if (useSoA)
		{
			switch (lut_boundary_method)
			{
			case BOUNDARY::REPLICATE:
				buildLocalLUTTensorDistanceMINInvoker<0, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::MINMAX_OUTPUT:
				buildLocalLUTTensorDistanceMINInvoker<1, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::MINMAX0_255:
				buildLocalLUTTensorDistanceMINInvoker<2, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::LINEAR:
				buildLocalLUTTensorDistanceMINInvoker<3, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::LINEAR_LAST2:
				buildLocalLUTTensorDistanceMINInvoker<4, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::NO_INTERPOLATION:
				buildLocalLUTTensorDistanceMINInvoker<5, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::EXPERIMENT1:
				buildLocalLUTTensorDistanceMINInvoker<6, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::EXPERIMENT2:
				buildLocalLUTTensorDistanceMINInvoker<7, true>(distance, lut_num, r, range_div, lut_filter_radius); break;
			default:
				break;
			}
		}
		else
		{
			switch (lut_boundary_method)
			{
			case BOUNDARY::REPLICATE:
				buildLocalLUTTensorDistanceMINInvoker<0, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::MINMAX_OUTPUT:
				buildLocalLUTTensorDistanceMINInvoker<1, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::MINMAX0_255:
				buildLocalLUTTensorDistanceMINInvoker<2, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::LINEAR:
				buildLocalLUTTensorDistanceMINInvoker<3, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::LINEAR_LAST2:
				buildLocalLUTTensorDistanceMINInvoker<4, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::NO_INTERPOLATION:
				buildLocalLUTTensorDistanceMINInvoker<5, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::EXPERIMENT1:
				buildLocalLUTTensorDistanceMINInvoker<6, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			case BOUNDARY::EXPERIMENT2:
				buildLocalLUTTensorDistanceMINInvoker<7, false>(distance, lut_num, r, range_div, lut_filter_radius); break;
			default:
				break;
			}
		}
	}

	// DP approach
	void LocalLUTUpsample::buildLocalLUTTensorFrequencyMaxDP(const int lut_num, const int r, const int ratio, const short dpcost)
	{
		int shift = (int)log2(ratio);
		vector<ushort> offset((2 * r + 1) * (2 * r + 1));

		if (r == 0)offset[0] = 1;
		else if (r == 1)
		{
			offset[0] = 1; offset[1] = 2; offset[2] = 1;
			offset[3] = 2; offset[4] = 4; offset[5] = 2;
			offset[6] = 1; offset[7] = 2; offset[8] = 1;
		}
		else if (r == 2)
		{
			offset[0] = 1; offset[1] = 2; offset[2] = 4; offset[3] = 2; offset[4] = 1;
			offset[5] = 2; offset[6] = 4; offset[7] = 8; offset[8] = 4; offset[9] = 2;
			offset[10] = 4; offset[11] = 8; offset[12] = 16; offset[13] = 8; offset[14] = 4;
			offset[15] = 2; offset[16] = 4; offset[17] = 8; offset[18] = 4; offset[19] = 2;
			offset[20] = 1; offset[21] = 2; offset[22] = 4; offset[23] = 2; offset[24] = 1;
		}
		else
		{
			int idx = 0;
			const ushort off = 2 * r + 1;
			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					offset[idx++] = off - abs(j) - abs(i);
				}
			}
		}

#pragma omp parallel for schedule (dynamic)
		for (int y = 0; y < lowres_size.height; y++)
		{
			const float frac = 1.f / (float)ratio;

			Mat lutmap(lut_num, lut_num, CV_16UC3);//frequency map
			Mat from(lut_num, lut_num, CV_8UC3);//DP map (left top: 0, left: 1, up: 2)	
			for (int x = 0; x < lowres_size.width; x++)
			{
				lutmap.setTo(0);
				from.setTo(0);

				//Initialize local LUT
				uchar* lutb = LUT_TensorAoS_B.ptr<uchar>(y) + lut_num * x;//LUT_B.channels()=lut_num
				uchar* lutg = LUT_TensorAoS_G.ptr<uchar>(y) + lut_num * x;
				uchar* lutr = LUT_TensorAoS_R.ptr<uchar>(y) + lut_num * x;


				//frequency map computation
				uchar iminb = lut_num - 1;
				uchar iming = lut_num - 1;
				uchar iminr = lut_num - 1;
				uchar ominb = lut_num - 1;
				uchar oming = lut_num - 1;
				uchar ominr = lut_num - 1;
				uchar imaxb = 0;
				uchar imaxg = 0;
				uchar imaxr = 0;
				uchar omaxb = 0;
				uchar omaxg = 0;
				uchar omaxr = 0;

				int idx = 0;
				for (int i = -r; i <= r; i++)
				{
					uchar* y_src = src_low_border.ptr<uchar>(y + i + r) + 3 * (x);
					uchar* y_res = prc_low_border.ptr<uchar>(y + i + r) + 3 * (x);

					for (int j = -r; j <= r; j++)
					{
						ushort adder = offset[idx++];
						uchar inp_b = y_src[0];
						uchar inp_g = y_src[1];
						uchar inp_r = y_src[2];
						uchar oup_b = y_res[0];
						uchar oup_g = y_res[1];
						uchar oup_r = y_res[2];

						y_src += 3;
						y_res += 3;
						iminb = min(iminb, inp_b);
						iming = min(iming, inp_g);
						iminr = min(iminr, inp_r);
						imaxb = max(imaxb, inp_b);
						imaxg = max(imaxg, inp_g);
						imaxr = max(imaxr, inp_r);

						ominb = min(ominb, oup_b);
						oming = min(oming, oup_g);
						ominr = min(ominr, oup_r);
						omaxb = max(omaxb, oup_b);
						omaxg = max(omaxg, oup_g);
						omaxr = max(omaxr, oup_r);
						//for box cost
						//lutmap.at<uchar>(inp_b, 3 * oup_b + 0)++;		
						//lutmap.at<uchar>(inp_g, 3 * oup_g + 1)++;
						//lutmap.at<uchar>(inp_r, 3 * oup_r + 2)++;
						//for Gaussian cost
						lutmap.at<ushort>(oup_b, 3 * inp_b + 0) += adder;
						lutmap.at<ushort>(oup_g, 3 * inp_g + 1) += adder;
						lutmap.at<ushort>(oup_r, 3 * inp_r + 2) += adder;
					}
				}

				// 頻度のヒストグラム
				/*if (x == 140 && y == 136)
				{
					Mat freqMap(lut_num, lut_num, CV_8U);
					Mat freqMaptemp(lut_num, lut_num, CV_8S);
					for (int i = 0; i < lut_num; i++)
					{
						for (int j = 0; j < lut_num; j++)
						{
							freqMap.at<uchar>(lut_num - i, j) = saturate_cast<uchar>(lutmap.at<ushort>(i, 3 * j + 0));
						}
					}
					imshow("freqMap", freqMap*4);	waitKey(0);
				}*/

				/*
				//set boundary condition with wta
				//only left side is implemented, but this is not effective
				short lmax = 0;
				int arg = ominb;
				for (int i = ominb; i <= omaxb; i++)
				{
					short v = lutmap.at<ushort>(i, 3 * iminb + 0);
					if (v > lmax)
					{
						lmax = v;
						arg = i;
					}
				}
				lutmap.at<ushort>(arg, 3 * iminb + 0) = 1000;

				lmax = 0;
				arg = oming;
				for (int i = oming; i <= omaxg; i++)
				{
					short v = lutmap.at<ushort>(i, 3 * iming + 1);
					if (v > lmax)
					{
						lmax = v;
						arg = i;
					}
				}
				lutmap.at<ushort>(arg, 3 * iming + 1) = 1000;

				lmax = 0;
				arg = ominr;
				for (int i = ominr; i <= omaxr; i++)
				{
					short v = lutmap.at<ushort>(i, 3 * iminr + 0);
					if (v > lmax)
					{
						lmax = v;
						arg = i;
					}
				}
				lutmap.at<ushort>(arg, 3 * iminr + 0) = 1000;
				*/

				//init LUT
				memset(lutb, 0, sizeof(uchar) * lut_num);
				memset(lutg, 0, sizeof(uchar) * lut_num);
				memset(lutr, 0, sizeof(uchar) * lut_num);

				//dynamic programming 
				//1. setting boundary conditions
				for (int i = 1; i < lut_num; i++)
				{
					//blue
					lutmap.at<ushort>(0, 3 * i + 0) = lutmap.at<ushort>(0, 3 * (i - 1) + 0) + lutmap.at<ushort>(0, 3 * i + 0);
					from.at<uchar>(0, 3 * i + 0) = 1;
					lutmap.at<ushort>(i, 0) = lutmap.at<ushort>(i - 1, 0) + lutmap.at<ushort>(i, 0) + 1;
					from.at<uchar>(i, 1) = 2;
					//green
					lutmap.at<ushort>(0, 3 * i + 1) = lutmap.at<ushort>(0, 3 * (i - 1) + 1) + lutmap.at<ushort>(0, 3 * i + 1);
					from.at<uchar>(0, 3 * i + 1) = 1;
					lutmap.at<ushort>(i, 1) = lutmap.at<ushort>(i - 1, 1) + lutmap.at<ushort>(i, 1) + 1;
					from.at<uchar>(i, 1) = 2;
					//red
					lutmap.at<ushort>(0, 3 * i + 2) = lutmap.at<ushort>(0, 3 * (i - 1) + 2) + lutmap.at<ushort>(0, 3 * i + 2);
					from.at<uchar>(0, 3 * i + 2) = 1;
					lutmap.at<ushort>(i, 2) = lutmap.at<ushort>(i - 1, 2) + lutmap.at<ushort>(i, 2) + 1;
					from.at<uchar>(i, 2) = 2;
				}

				//2. body 		
				for (int i = 1; i < lut_num; i++)
				{
					ushort* lut_ = lutmap.ptr<ushort>(i);
					uchar* map_ = from.ptr<uchar>(i);
					for (int j = 1; j < lut_num; j++)
					{
						const int j3 = j * 3;
						ushort* lut = lut_ + j3;
						uchar* map = map_ + j3;

						//blue
						int temp1 = lutmap.at<ushort>(i - 1, 3 * (j - 1) + 0) + lut[0] + dpcost; //left top
						int temp2 = lutmap.at<ushort>(i, 3 * (j - 1) + 0) + lut[0];//left
						int temp3 = lutmap.at<ushort>(i - 1, 3 * j + 0) + lut[0];//top

						if (temp1 >= temp2 && temp1 >= temp3)
						{
							lut[0] = temp1;
							map[0] = 0;
						}
						else if (temp2 >= temp3)
						{
							lut[0] = temp2;
							map[0] = 1;
						}
						else
						{
							lut[0] = temp3;
							map[0] = 2;
						}

						//green
						temp1 = lutmap.at<ushort>(i - 1, 3 * (j - 1) + 1) + lut[1] + dpcost;
						temp2 = lutmap.at<ushort>(i, 3 * (j - 1) + 1) + lut[1];
						temp3 = lutmap.at<ushort>(i - 1, 3 * j + 1) + lut[1];

						if (temp1 >= temp2 && temp1 >= temp3)
						{
							lut[1] = temp1;
							map[1] = 0;
						}
						else if (temp2 >= temp3)
						{
							lut[1] = temp2;
							map[1] = 1;
						}
						else
						{
							lut[1] = temp3;
							map[1] = 2;
						}

						//red
						temp1 = lutmap.at<ushort>(i - 1, 3 * (j - 1) + 2) + lut[2] + dpcost;
						temp2 = lutmap.at<ushort>(i, 3 * (j - 1) + 2) + lut[2];
						temp3 = lutmap.at<ushort>(i - 1, 3 * j + 2) + lut[2];

						if (temp1 >= temp2 && temp1 >= temp3)
						{
							lut[2] = temp1;
							map[2] = 0;
						}
						else if (temp2 >= temp3)
						{
							lut[2] = temp2;
							map[2] = 1;
						}
						else
						{
							lut[2] = temp3;
							map[2] = 2;
						}
					}
				}

				// DPマップ
				//if (x == 140 && y == 136)
				//{
				//	Mat freqMap(lut_num, lut_num, CV_8U);
				//	Mat freqMaptemp(lut_num, lut_num, CV_8S);
				//	for (int i = 0; i < lut_num; i++)
				//	{
				//		for (int j = 0; j < lut_num; j++)
				//		{
				//			freqMap.at<uchar>(lut_num - i, j) = saturate_cast<uchar>(lutmap.at<ushort>(i, 3 * j + 0)*(10.0/255.0));
				//		}
				//	}
				//	imshow("freqMap", freqMap);	waitKey(0);
				//}

				//3. tracing back
				//blue
				int i = lut_num - 1;
				int j = lut_num - 1;
				while (i >= 0 && j >= 0)
				{
					lutb[j] = i;

					switch (from.at<uchar>(i, 3 * j + 0))
					{
					case 0:
						i--; j--; break;
					case 1:
						j--; break;
					case 2:
						i--; break;
					}
				}

				//green
				i = lut_num - 1;
				j = lut_num - 1;
				while (i >= 0 && j >= 0)
				{
					lutg[j] = i;

					switch (from.at<uchar>(i, 3 * j + 1))
					{
					case 0:
						i--; j--; break;
					case 1:
						j--; break;
					case 2:
						i--; break;
					}
				}

				//red
				i = lut_num - 1;
				j = lut_num - 1;
				while (i >= 0 && j >= 0)
				{
					lutr[j] = i;

					switch (from.at<uchar>(i, 3 * j + 2))
					{
					case 0:
						i--; j--; break;
					case 1:
						j--; break;
					case 2:
						i--; break;
					}
				}
				// Show the LUT by DP
				//if (x == 140 && y == 136)
				//{
				//	cout << "DP: ";
				//	for (int i = 0; i < lut_num; i++)
				//	{
				//		cout << (int)lutb[i] << " ";
				//	}
				//	cout << endl;
				//}

				/*if (x == 140 && y == 136)
				{
					Mat dp(lut_num, lut_num, CV_8UC3);
					dp.setTo(80);
					for (int i = 0; i < lut_num; i++)
					{
						cout << (int)lutb[i] << " ";
						int out_lum = (int)lutb[i];
						dp.at<uchar>(lut_num - out_lum, 3 * i + 0) = 255;
						dp.at<uchar>(lut_num - out_lum, 3 * i + 1) = 128;
						dp.at<uchar>(lut_num - out_lum, 3 * i + 2) = 128;
					}
					imshow("dp", dp); waitKey(0);
				}*/
			}
		}
	}
#pragma endregion

#pragma region tensorUP

#pragma region general
	inline float linear_interpolation(const float pre_ref_shift, const float pre_interpolation, const float ref_intensity, const float next_ref_shift, const float next_interpolation)
	{
		return((next_ref_shift - ref_intensity) * pre_interpolation + (ref_intensity - pre_ref_shift) * next_interpolation) / (next_ref_shift - pre_ref_shift);
	}

	inline float linear_interpolation_withoutnormalize(const float pre_ref_shift, const float pre_interpolation, const float ref_intensity, const float next_ref_shift, const float next_interpolation)
	{
		return((next_ref_shift - ref_intensity) * pre_interpolation + (ref_intensity - pre_ref_shift) * next_interpolation);
	}
#pragma endregion

#pragma region nxn_naive
	//only support the case bin = 256
	void LocalLUTUpsample::_tensorUpConvNxNLinearNaive(const Mat& src_highres, Mat& dst, const Mat& spaceweight)
	{
		//if (!isOffset) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int lut_num = 256;
		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int rshift = (int)log2(lut_num);

		const int scale = int(up_sampling_ratio_resolution);

		const int d = (int)sqrt(spaceweight.cols);
		const int r = d / 2;
		const int conv_size = d * d;

		//weight.setTo(1.f / conv_size);
#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < src_highres.rows; y += scale)
		{
			int y_ = (int)(y / up_sampling_ratio_resolution);
			Mat* LUT;

			const float normalize = 1.f / range_quantization_ratio;
			for (int x = 0; x < src_highres.cols; x += scale)
			{
				int x_ = (int)(x / up_sampling_ratio_resolution);

				for (int n = 0; n < scale; n++)
				{
					const uchar* src = src_highres.ptr<uchar>(y + n); // reference
					uchar* dest = dst.ptr<uchar>(y + n); // output
					for (int m = 0; m < scale; m++)
					{
						const int idx = scale * n + m;
						const float* weightmap_ptr = spaceweight.ptr<float>(idx);

						//if (range_quantization_ratio == 1)
						{
							for (int c = 0; c < 3; c++)
							{

								if (c == 0)
								{
									LUT = &LUT_TensorAoS_B;
								}
								if (c == 1)
								{
									LUT = &LUT_TensorAoS_G;
								}
								if (c == 2)
								{
									LUT = &LUT_TensorAoS_R;
								}

								const int intensity = src[3 * (x + m) + c];
								int idx = 0;
								float val = 0.f;
								for (int j = 0; j < d; j++)
								{
									for (int i = 0; i < d; i++)
									{
										const int Y = min(sheight - 1, max(0, y_ + j - r + 1));
										const int X = min(swidth - 1, max(0, x_ + i - r + 1));

										val = fma(weightmap_ptr[idx++], (float)LUT->at<uchar>(Y, X * lut_num + intensity), val);
										//val += weightmap_ptr[idx++] * LUT->at<uchar>(Y, X * lut_num + intensity);
										//val += weightmap_ptr[idx++] * LUT->ptr<uchar>(Y, X)[intensity];
									}
								}

								dest[3 * (x + m) + c] = saturate_cast<uchar>(val);
							}
						}
					}
				}
			}
		}
	}

	void LocalLUTUpsample::tensorUpBoxNxNLinear(const Mat& src_highres, Mat& dst, const int d)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, d * d, CV_32F);
		weight.setTo(1.f / (d * d));
		_tensorUpConvNxNLinearNaive(src_highres, dst, weight);
	}

	void LocalLUTUpsample::tensorUpGaussNxNLinear(const Mat& src_highres, Mat& dst, const int d, const float sigma)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, d * d, CV_32F);
		cp::setGaussianLnWeight(weight, sigma, 2);
		_tensorUpConvNxNLinearNaive(src_highres, dst, weight);
	}

	void LocalLUTUpsample::tensorUpLaplaceNxNLinear(const Mat& src_highres, Mat& dst, const int d, const float sigma)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, d * d, CV_32F);
		cp::setGaussianLnWeight(weight, sigma, 1);
		_tensorUpConvNxNLinearNaive(src_highres, dst, weight);
	}
#pragma endregion

#pragma region 1x1
	template<bool quantization>
	void LocalLUTUpsample::_tensorUpNearestLinear(const Mat& src_highres, Mat& dst, const int lut_num, const bool isOffset)
	{
		if (!isOffset && quantization) offset_map.setTo(0);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int scale = int(up_sampling_ratio_resolution);

		if (src_highres.channels() == 1)
		{
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y0 = (int)(y / up_sampling_ratio_resolution);

				uchar* lutbptr = LUT_TensorAoS_B.ptr<uchar>(y0);
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x0 = (int)(x / up_sampling_ratio_resolution);

					uchar* lutb = lutbptr + lut_num * x0;

					if constexpr (quantization)
					{
						Mat* LUT = &LUT_TensorAoS_B;
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output

							for (int m = 0; m < scale; m++)
							{
								const int intensity = src[(x + m)];
								const int i_pre = intensity >> sshift;// range quantization
								const int i_nxt = i_pre + 1;

								const uchar offset = offset_map.at<uchar>(y0, x0);
								float pre_m = float((LUT->at<uchar>(y0, x0 * lut_num + i_pre) << sshift) + offset);

								if (i_pre != lut_num - 1)
								{
									float next_m = float((LUT->at<uchar>(y0, x0 * lut_num + i_pre) << sshift) + offset);
									dest[(x + m)] = saturate_cast<uchar>(linear_interpolation(float(i_pre << sshift), pre_m, float(intensity), float(i_nxt << sshift), next_m));
								}
								else
								{
									dest[(x + m)] = saturate_cast<uchar>(linear_interpolation(float(i_pre << sshift), pre_m, float(intensity), 255.f, 255.f));
								}
							}
						}
					}
					else //no quantization
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output

							for (int m = 0; m < scale; m++)
							{
								const int idx = (x + m);
								dest[idx + 0] = lutb[src[idx + 0]];
							}
						}
					}
				}
			}
		}
		else
		{
#pragma omp parallel for schedule (dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y0 = (int)(y / up_sampling_ratio_resolution);

				const float normalization = 1.f / (1 << sshift);
				uchar* lutbptr = LUT_TensorAoS_B.ptr<uchar>(y0);
				uchar* lutgptr = LUT_TensorAoS_G.ptr<uchar>(y0);
				uchar* lutrptr = LUT_TensorAoS_R.ptr<uchar>(y0);

				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x0 = (int)(x / up_sampling_ratio_resolution);

					uchar* lutb = lutbptr + lut_num * x0;
					uchar* lutg = lutgptr + lut_num * x0;
					uchar* lutr = lutrptr + lut_num * x0;

					if constexpr (quantization)
					{
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						uchar offset[3];
						offset[0] = offset_map.at<uchar>(y0, 3 * x0 + 0);
						offset[1] = offset_map.at<uchar>(y0, 3 * x0 + 1);
						offset[2] = offset_map.at<uchar>(y0, 3 * x0 + 2);
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const int idx = 3 * (x + m);
								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;// range quantization
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								float pre_m = float((lutb[intensity_shift_pre] << sshift) + offset[0]);
								if (intensity_shift_pre != lut_num - 1)
								{
									float next_m = float((lutb[intensity_shift_pre] << sshift) + offset[0]);
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), pre_m, float(intensity), float(intensity_shift_nex << sshift), next_m) * normalization);
								}
								else
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), pre_m, float(intensity), 255.f, 255.f));
								}

								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;// range quantization
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								pre_m = float((lutg[intensity_shift_pre] << sshift) + offset[1]);
								if (intensity_shift_pre != lut_num - 1)
								{
									float next_m = float((lutg[intensity_shift_pre] << sshift) + offset[1]);
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), pre_m, float(intensity), float(intensity_shift_nex << sshift), next_m) * normalization);
								}
								else
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), pre_m, float(intensity), 255.f, 255.f));
								}

								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;// range quantization
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								pre_m = float((lutr[intensity_shift_pre] << sshift) + offset[2]);
								if (intensity_shift_pre != lut_num - 1)
								{
									float next_m = float((lutr[intensity_shift_pre] << sshift) + offset[2]);
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), pre_m, float(intensity), float(intensity_shift_nex << sshift), next_m) * normalization);
								}
								else
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), pre_m, float(intensity), 255.f, 255.f));
								}
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
					else //no quantization
					{
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const int idx = 3 * (x + m);
								dest[idx + 0] = lutb[src[idx + 0]];
								dest[idx + 1] = lutg[src[idx + 1]];
								dest[idx + 2] = lutr[src[idx + 2]];
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
				}
			}
		}
	}

	void LocalLUTUpsample::tensorUpNearestLinear(const Mat& src_highres, Mat& dst, const int lut_num, const bool isOffset)
	{
		if (lut_num == 256)
		{
			_tensorUpNearestLinear<false>(src_highres, dst, lut_num, isOffset);
		}
		else
		{
			_tensorUpNearestLinear<true>(src_highres, dst, lut_num, isOffset);
		}
	}
#pragma endregion

#pragma region 2x2
	template<bool quantization>
	void LocalLUTUpsample::_tensorUpConv4Linear(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const int lut_num, const bool isOffset)
	{
		if (!isOffset && quantization) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int scale = int(up_sampling_ratio_resolution);

		uchar* off = offset_map.ptr<uchar>();
		uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
		uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
		uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();
		const float* sptr = spaceweight.ptr<float>();
		if (src_highres.channels() == 1)
		{
#pragma omp parallel for schedule (dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y0 = (int)(y / up_sampling_ratio_resolution);
				const int y1 = min(y0 + 1, sheight - 1);

				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const float normalization = 1.f / (1 << sshift);

				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x0 = (int)(x / up_sampling_ratio_resolution);
					const int x1 = min(x0 + 1, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const __m128i mlutidx = _mm_setr_epi32(Y0 + X0, Y0 + X1, Y1 + X0, Y1 + X1);

					if constexpr (quantization)
					{
						__m128 a = _mm_setr_ps(
							offset_map.at<uchar>(y0, x0),
							offset_map.at<uchar>(y0, x1),
							offset_map.at<uchar>(y1, x0),
							offset_map.at<uchar>(y1, x1));
						__m256 offset = _mm256_set_m128(a, a);

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);
							for (int m = 0; m < scale; m++)
							{
								__m128 mw_ = _mm_load_ps(wptr + 4 * m);
								__m256 mw = _mm256_set_m128(mw_, mw_);

								float sumpre, sumnex;
								const int idx = (x + m);
								__m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//gray
								int intensity = src[idx];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256i mlutidxpack = _mm256_set_m128i(mlutidx, _mm_add_epi32(mlutidx, _mm_set1_epi32(intensity_shift_nex - intensity_shift_pre)));
								__m256 mv = _mm256_mul_ps(mw,
									_mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidxpack, _mm256_set1_epi32(intensity_shift_pre)))),
										offset));
								_mm256_reduceadd_highlow_ps(mv, sumpre, sumnex);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalization);
								else
									dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255, 255));
							}

						}
					}
					else //no quantization
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);
							for (int m = 0; m < scale; m++)
							{
								__m128 mw = _mm_load_ps(wptr + 4 * m);
								const int idx = (x + m);

								int intensity = src[idx];
								__m128 mv = _mm_mul_ps(mw, _mm_i8gather_ps(lutb, _mm_add_epi32(mlutidx, _mm_set1_epi32(intensity))));
								dest[idx] = saturate_cast<uchar>(_mm_reduceadd_ps(mv));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
#pragma omp parallel for schedule (dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));
				const float normalization = 1.f / (1 << sshift);

				const int y0 = (int)(y / up_sampling_ratio_resolution);
				const int y1 = min(y0 + 1, sheight - 1);

				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;

				const __m128i mlutidxY = _mm_setr_epi32(Y0, Y0, Y1, Y1);

				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x0 = (int)(x / up_sampling_ratio_resolution);
					const int x1 = min(x0 + 1, swidth - 1);
					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const __m128i mlutidxX = _mm_setr_epi32(X0, X1, X0, X1);
					const __m128i mlutidx = _mm_add_epi32(mlutidxX, mlutidxY);

					if constexpr (quantization)
					{
						__m256 offset[3];
						uchar* oy0 = offset_map.ptr<uchar>(y0);
						uchar* oy1 = offset_map.ptr<uchar>(y1);
						for (int c = 0; c < 3; c++)
						{
							const __m128 a = _mm_setr_ps(
								oy0[3 * x0 + c],
								oy0[3 * x1 + c],
								oy1[3 * x0 + c],
								oy1[3 * x1 + c]);

							offset[c] = _mm256_set_m128(a, a);
						}

						const __m128* wptr = (__m128*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								__m128 mw_ = *wptr++;
								__m256 mw = _mm256_set_m128(mw_, mw_);
								const int idx = 3 * (x + m);
								float sumpre, sumnex;//for reduce add

								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;// range quantization
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256i mlutidxpack = _mm256_set_m128i(mlutidx, _mm_add_epi32(mlutidx, _mm_set1_epi32(intensity_shift_nex - intensity_shift_pre)));
								uchar* lut = lutb + intensity_shift_pre;
								__m256 mv = _mm256_mul_ps(mw, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidxpack), offset[0]));
								_mm256_reduceadd_highlow_ps(mv, sumpre, sumnex);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalization);
								}
								else
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mlutidxpack = _mm256_set_m128i(mlutidx, _mm_add_epi32(mlutidx, _mm_set1_epi32(intensity_shift_nex - intensity_shift_pre)));
								lut = lutg + intensity_shift_pre;
								mv = _mm256_mul_ps(mw, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidxpack), offset[1]));
								_mm256_reduceadd_highlow_ps(mv, sumpre, sumnex);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalization);
								}
								else
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mlutidxpack = _mm256_set_m128i(mlutidx, _mm_add_epi32(mlutidx, _mm_set1_epi32(intensity_shift_nex - intensity_shift_pre)));
								lut = lutr + intensity_shift_pre;
								mv = _mm256_mul_ps(mw, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidxpack), offset[2]));
								_mm256_reduceadd_highlow_ps(mv, sumpre, sumnex);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalization);
								}
								else
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
					else //no quantization
					{
						const __m128* wptr = (__m128*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m128 mw = *wptr++;
								const int idx = 3 * (x + m);
								dest[idx + 0] = saturate_cast<uchar>(_mm_reduceadd_ps(_mm_mul_ps(mw, _mm_i8gather_ps(lutb + src[idx + 0], mlutidx))));
								dest[idx + 1] = saturate_cast<uchar>(_mm_reduceadd_ps(_mm_mul_ps(mw, _mm_i8gather_ps(lutg + src[idx + 1], mlutidx))));
								dest[idx + 2] = saturate_cast<uchar>(_mm_reduceadd_ps(_mm_mul_ps(mw, _mm_i8gather_ps(lutr + src[idx + 2], mlutidx))));
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
				}
			}
		}
	}


	void LocalLUTUpsample::tensorUpTriLinear(const Mat& src_highres, Mat& dst, const int lut_num, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 4, CV_32F);
		cp::setLinearWeight2x2(weight);

		if (lut_num == 256) _tensorUpConv4Linear<false>(src_highres, dst, weight, lut_num, isOffset);
		else _tensorUpConv4Linear<true>(src_highres, dst, weight, lut_num, isOffset);
	}

	void LocalLUTUpsample::tensorUpBox4Linear(const Mat& src_highres, Mat& dst, const int lut_num, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 4, CV_32F);
		weight.setTo(1.f / 4.f);

		if (lut_num == 256) _tensorUpConv4Linear<false>(src_highres, dst, weight, lut_num, isOffset);
		else _tensorUpConv4Linear<true>(src_highres, dst, weight, lut_num, isOffset);
	}

	void LocalLUTUpsample::tensorUpGauss4Linear(const Mat& src_highres, Mat& dst, const int lut_num, const float sigma, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 4, CV_32F);
		cp::setGaussianWeight2x2(weight, sigma);

		if (lut_num == 256) _tensorUpConv4Linear<false>(src_highres, dst, weight, lut_num, isOffset);
		else _tensorUpConv4Linear<true>(src_highres, dst, weight, lut_num, isOffset);
	}

#pragma endregion

#pragma region 4x4

#pragma region debug
	template<int scale>
	static void inline _tensorUpConv16LinearRGBNoquantSkip(const int x, const int y, const Mat& src_highres, const float* spaceweight, Mat& dst, const uchar* lutb, const uchar* lutg, const uchar* lutr, const __m256i mlutidx0, const __m256i mlutidx1)
	{
		uchar preib = 0;
		uchar preig = 0;
		uchar preir = 0;
		uchar preob = 0;
		uchar preog = 0;
		uchar preor = 0;
		__m256* wptr = (__m256*)spaceweight;
		for (int n = 0; n < scale; n++)
		{
			const uchar* src = src_highres.ptr<uchar>(y + n); // reference
			uchar* dest = dst.ptr<uchar>(y + n); // output
			for (int m = 0; m < scale; m++)
			{
				const __m256 mw0 = *wptr++;
				const __m256 mw1 = *wptr++;
				const int idx = 3 * (x + m);
				//skip same lut
				__m256 mv;
				int intensity = src[idx + 0];
				if (preib == intensity)dest[idx + 0] = preob;
				else
				{
					preib = intensity;
					mv = _mm256_mul_ps(mw0, _mm256_cvtepi32_ps(_mm256_i32gather_epi32(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
					mv = _mm256_fmadd_ps(mw1, _mm256_cvtepi32_ps(_mm256_i32gather_epi32(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
					preob = dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
				}

				intensity = src[idx + 1];
				if (preig == intensity)dest[idx + 1] = preog;
				else
				{
					preig = intensity;
					mv = _mm256_mul_ps(mw0, _mm256_cvtepi32_ps(_mm256_i32gather_epi32(lutg, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
					mv = _mm256_fmadd_ps(mw1, _mm256_cvtepi32_ps(_mm256_i32gather_epi32(lutg, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
					preog = dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
				}
				intensity = src[idx + 2];
				if (preir == intensity)dest[idx + 2] = preor;
				else
				{
					preir = intensity;
					mv = _mm256_mul_ps(mw0, _mm256_cvtepi32_ps(_mm256_i32gather_epi32(lutr, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
					mv = _mm256_fmadd_ps(mw1, _mm256_cvtepi32_ps(_mm256_i32gather_epi32(lutr, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
					preor = dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
				}
			}
		}
	}
#pragma endregion

	template<bool quantization>
	void LocalLUTUpsample::_tensorUpConv16Linear(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const int lut_num, const bool isOffset)
	{
		if (!isOffset && quantization) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int scale = int(up_sampling_ratio_resolution);

		uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
		uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
		uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();
		const float* sptr = spaceweight.ptr<float>();
		uchar* off = offset_map.ptr<uchar>();

		if (src_highres.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 1);
				const int y1 = y0 + 1;
				const int y2 = min(y_ + 1, sheight - 1);
				const int y3 = min(y_ + 2, sheight - 1);

				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));
				const float normalize = 1.f / range_quantization_ratio;

				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 1);
					const int x1 = x_;
					const int x2 = min(x_ + 1, swidth - 1);
					const int x3 = min(x_ + 2, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3);

					if constexpr (quantization)
					{
						const int OX0 = x0;
						const int OX1 = x1;
						const int OX2 = x2;
						const int OX3 = x3;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3);
						const __m256i moffidx1 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3);

						//offset map
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 16 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 16 * m + 8);
								const int idx = (x + m);

								//b
								int intensity = src[idx];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								float sumnex = _mm256_reduceadd_ps(mv);

								if (intensity_shift_pre != lut_num - 1)
									dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else //no quantization
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 16 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 16 * m + 8);
								const int idx = (x + m);

								int intensity = src[idx];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mw1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								dest[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));
				const float normalize = 1.f / range_quantization_ratio;

				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 1);
				const int y1 = y0 + 1;
				const int y2 = min(y_ + 1, sheight - 1);
				const int y3 = min(y_ + 2, sheight - 1);

				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;

				const __m256i mlutidxY0 = _mm256_setr_m128i(_mm_set1_epi32(Y0), _mm_set1_epi32(Y1));
				const __m256i mlutidxY1 = _mm256_setr_m128i(_mm_set1_epi32(Y2), _mm_set1_epi32(Y3));
				const __m128i mxstep = _mm_setr_epi32(-1, 0, 1, 2);
				const __m128i mxmax = _mm_set1_epi32(swidth - 1);
				const __m128i mlut_num = _mm_set1_epi32(lut_num);

				const __m256i moffidxY0 = _mm256_setr_epi32(OY0, OY0, OY0, OY0, OY1, OY1, OY1, OY1);
				const __m256i moffidxY1 = _mm256_setr_epi32(OY2, OY2, OY2, OY2, OY3, OY3, OY3, OY3);
				const __m256i mone = _mm256_set1_epi32(1);
				const __m256i mtwo = _mm256_set1_epi32(2);
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const __m128i mx = _mm_min_epi32(mxmax, _mm_max_epi32(_mm_setzero_si128(), _mm_add_epi32(_mm_set1_epi32(x_), mxstep)));
					const __m128i mxlut = _mm_mullo_epi32(mlut_num, mx);
					const __m256i mlutidxX = _mm256_setr_m128i(mxlut, mxlut);
					const __m256i mlutidx0 = _mm256_add_epi32(mlutidxY0, mlutidxX);
					const __m256i mlutidx1 = _mm256_add_epi32(mlutidxY1, mlutidxX);

					if constexpr (quantization)
					{
						const __m128i mox = _mm_add_epi32(mx, _mm_slli_epi32(mx, 1));//mx*3
						const __m256i moffidxX = _mm256_setr_m128i(mox, mox);
						const __m256i moffidx0 = _mm256_add_epi32(moffidxY0, moffidxX);
						const __m256i moffidx1 = _mm256_add_epi32(moffidxY1, moffidxX);

						//offset map for rgb
						const __m256 moffb0 = _mm256_i32gather_epu8cvtps(off, moffidx0);
						const __m256 moffb1 = _mm256_i32gather_epu8cvtps(off, moffidx1);
						const __m256 moffg0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mone));
						const __m256 moffg1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mone));
						const __m256 moffr0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mtwo));
						const __m256 moffr1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mtwo));

						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const int idx = 3 * (x + m);
								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								uchar* lut = lutb + intensity_shift_pre;
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								lut = lutb + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutg + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffg1), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutg + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffg1), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutr + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutr + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
					else //no quantization
					{
#if 0
						_tensorUpConv16LinearRGBNoquantSkip<2>(x, y, src_highres, spaceweight.ptr<float>(), dst, lutb, lutg, lutr, mlutidx0, mlutidx1);
#else
						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const int idx = 3 * (x + m);

								//b
								uchar* lut = lutb + src[idx + 0];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//g
								lut = lutg + src[idx + 1];
								mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//r
								lut = lutr + src[idx + 2];
								mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
#endif
					}
				}
			}
		}
	}


	template<bool quantization>
	void LocalLUTUpsample::_tensorUpBilateralConv16Linear(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const float sigma_range, const int lut_num, const bool isOffset)
	{
		const int r = (src_low_border.cols - lowres_size.width) / 2;
		float* expTable = (float*)_mm_malloc(sizeof(float) * 256, AVX_ALIGN);
		const float range_coeff = 1.f / (-2.f * sigma_range * sigma_range);
		for (int i = 0; i < 256; i++)
		{
			expTable[i] = exp((i * i) * range_coeff);
		}

		if (!isOffset) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int rshift = (int)log2(lut_num);

		const int scale = int(up_sampling_ratio_resolution);

		uchar* low = src_low_border.ptr<uchar>();
		uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
		uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
		uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();
		uchar* off = offset_map.ptr<uchar>();
		if (src_highres.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 1);
				const int y1 = y0 + 1;
				const int y2 = min(y_ + 1, sheight - 1);
				const int y3 = min(y_ + 2, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int YL0 = (y0 + r) * src_low_border.cols;
				const int YL1 = (y1 + r) * src_low_border.cols;
				const int YL2 = (y2 + r) * src_low_border.cols;
				const int YL3 = (y3 + r) * src_low_border.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 1);
					const int x1 = x_;
					const int x2 = min(x_ + 1, swidth - 1);
					const int x3 = min(x_ + 2, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int XL0 = (x0 + r);
					const int XL1 = (x1 + r);
					const int XL2 = (x2 + r);
					const int XL3 = (x3 + r);
					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3);

					const __m256i mlowidx0 = _mm256_setr_epi32(YL0 + XL0, YL0 + XL1, YL0 + XL2, YL0 + XL3, YL1 + XL0, YL1 + XL1, YL1 + XL2, YL1 + XL3);
					const __m256i mlowidx1 = _mm256_setr_epi32(YL2 + XL0, YL2 + XL1, YL2 + XL2, YL2 + XL3, YL3 + XL0, YL3 + XL1, YL3 + XL2, YL3 + XL3);

					if constexpr (quantization)
					{
						if constexpr (quantization)
						{
							const int OX0 = x0;
							const int OX1 = x1;
							const int OX2 = x2;
							const int OX3 = x3;
							const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3);
							const __m256i moffidx1 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3);

							//offset map
							__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
							__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));

							for (int n = 0; n < scale; n++)
							{
								const uchar* src = src_highres.ptr<uchar>(y + n); // reference
								uchar* dest = dst.ptr<uchar>(y + n); // output
								const float* wptr = spaceweight.ptr<float>(scale * n);

								for (int m = 0; m < scale; m++)
								{
									const __m256 mw0 = _mm256_load_ps(wptr + 16 * m + 0);
									const __m256 mw1 = _mm256_load_ps(wptr + 16 * m + 8);
									const int idx = (x + m);
									const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

									//gray
									int intensity = src[idx];
									int intensity_shift_pre = intensity >> sshift;
									int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
									__m256i mlows = _mm256_set1_epi32(intensity);
									__m256i mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx0));
									__m256i mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx1));
									__m256 mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
									__m256 mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));

									__m256 mv = _mm256_mul_ps(mwr0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
									mv = _mm256_fmadd_ps(mwr1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
									const float norm = _mm256_reduceadd_ps(_mm256_add_ps(mwr0, mwr1));
									float sumpre = _mm256_reduceadd_ps(mv) / norm;

									mv = _mm256_mul_ps(mwr0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
									mv = _mm256_fmadd_ps(mwr1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
									float sumnex = _mm256_reduceadd_ps(mv) / norm;

									if (intensity_shift_pre != lut_num - 1)
										dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
									else
										dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
							}
						}
					}
					else
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);


							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 16 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 16 * m + 8);
								const int idx = (x + m);

								int intensity = src[idx];

								__m256i mlows = _mm256_set1_epi32(intensity);
								__m256i mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx0, _mm256_set1_epi32(0))));
								__m256i mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx1, _mm256_set1_epi32(0))));
								__m256 mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								__m256 mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								__m256 mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								dest[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(mwr0, mwr1)));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 1);
				const int y1 = y0 + 1;
				const int y2 = min(y_ + 1, sheight - 1);
				const int y3 = min(y_ + 2, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int YL0 = (y0 + r) * 3 * src_low_border.cols;
				const int YL1 = (y1 + r) * 3 * src_low_border.cols;
				const int YL2 = (y2 + r) * 3 * src_low_border.cols;
				const int YL3 = (y3 + r) * 3 * src_low_border.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 1);
					const int x1 = x_;
					const int x2 = min(x_ + 1, swidth - 1);
					const int x3 = min(x_ + 2, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int XL0 = (x0 + r) * 3;
					const int XL1 = (x1 + r) * 3;
					const int XL2 = (x2 + r) * 3;
					const int XL3 = (x3 + r) * 3;
					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3);

					const __m256i mlowidx0 = _mm256_setr_epi32(YL0 + XL0, YL0 + XL1, YL0 + XL2, YL0 + XL3, YL1 + XL0, YL1 + XL1, YL1 + XL2, YL1 + XL3);
					const __m256i mlowidx1 = _mm256_setr_epi32(YL2 + XL0, YL2 + XL1, YL2 + XL2, YL2 + XL3, YL3 + XL0, YL3 + XL1, YL3 + XL2, YL3 + XL3);

					if constexpr (quantization)
					{
						const int OX0 = x0 * 3;
						const int OX1 = x1 * 3;
						const int OX2 = x2 * 3;
						const int OX3 = x3 * 3;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3);
						const __m256i moffidx1 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3);

						//offset map for rgb
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));
						__m256 moffg0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx0, _mm256_set1_epi32(1))));
						__m256 moffg1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx1, _mm256_set1_epi32(1))));
						__m256 moffr0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx0, _mm256_set1_epi32(2))));
						__m256 moffr1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx1, _mm256_set1_epi32(2))));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 16 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 16 * m + 8);

								const int idx = 3 * (x + m);
								const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								uchar inter_pre = saturate_cast<uchar>(sumpre);
								uchar inter_nex = saturate_cast<uchar>(sumnex);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(255.f - range_quantization_ratio, sumpre, float(intensity), 255.f, 255.f));

								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffg1), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffg1), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								inter_pre = saturate_cast<uchar>(sumpre);
								inter_nex = saturate_cast<uchar>(sumnex);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(255.f - range_quantization_ratio, sumpre, float(intensity), 255.f, 255.f));

								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffr1), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffr1), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								inter_pre = saturate_cast<uchar>(sumpre);
								inter_nex = saturate_cast<uchar>(sumnex);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(255.f - range_quantization_ratio, sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 16 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 16 * m + 8);
								const int idx = 3 * (x + m);
								//b
								int intensity = src[idx + 0];
								__m256i mlows = _mm256_set1_epi32(intensity);
								__m256i mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx0, _mm256_set1_epi32(0))));
								__m256i mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx1, _mm256_set1_epi32(0))));
								__m256 mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								__m256 mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								__m256 mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(mwr0, mwr1)));
								//g
								intensity = src[idx + 1];
								mlows = _mm256_set1_epi32(intensity);
								mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx0, _mm256_set1_epi32(1))));
								mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx1, _mm256_set1_epi32(1))));
								mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(mwr0, mwr1)));
								//r
								intensity = src[idx + 2];
								mlows = _mm256_set1_epi32(intensity);
								mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx0, _mm256_set1_epi32(2))));
								mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx1, _mm256_set1_epi32(2))));
								mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(mwr0, mwr1)));
							}
						}
					}
				}
			}
		}
		_mm_free(expTable);
	}


	void LocalLUTUpsample::tensorUpBiCubicLinear(const Mat& src_highres, Mat& dst, const int lut_num, const float alpha, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 16, CV_32F);

		cp::setCubicWeight4x4(weight, alpha);

		if (lut_num == 256)
			_tensorUpConv16Linear<false>(src_highres, dst, weight, lut_num, isOffset);
		else
			_tensorUpConv16Linear<true>(src_highres, dst, weight, lut_num, isOffset);

		//interpolationConv16Linear(src_highres, dst, weight, lut_num, isOffset);
	}

	void LocalLUTUpsample::tensorUpBox16Linear(const Mat& src_highres, Mat& dst, const int lut_num, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 16, CV_32F);
		weight.setTo(1.f / 16.f);

		if (lut_num == 256)
			_tensorUpConv16Linear<false>(src_highres, dst, weight, lut_num, isOffset);
		else
			_tensorUpConv16Linear<true>(src_highres, dst, weight, lut_num, isOffset);
	}

	void LocalLUTUpsample::tensorUpGauss16Linear(const Mat& src_highres, Mat& dst, const int lut_num, const float sigma, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 16, CV_32F);

		cp::setGaussianWeight4x4(weight, sigma);

		//interpolationConv16Linear(src_highres, dst, weight, lut_num, isOffset);
		if (lut_num == 256) _tensorUpConv16Linear<false>(src_highres, dst, weight, lut_num, isOffset);
		else _tensorUpConv16Linear<true>(src_highres, dst, weight, lut_num, isOffset);
	}

	void LocalLUTUpsample::tensorUpBilateral16Linear(const Mat& src_highres, Mat& dst, const int lut_num, const float sigma_space, const float sigma_range, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 16, CV_32F);
		cp::setGaussianWeight4x4(weight, sigma_space);

		if (lut_num == 256)
			_tensorUpBilateralConv16Linear<false>(src_highres, dst, weight, sigma_range, lut_num, isOffset);
		else
			_tensorUpBilateralConv16Linear<true>(src_highres, dst, weight, sigma_range, lut_num, isOffset);
	}
#pragma endregion

#pragma region 8x8
	template<bool quantization>
	void LocalLUTUpsample::_tensorUpConv64Linear(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const int lut_num, const bool isOffset)
	{
		if (!isOffset && quantization) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int scale = int(up_sampling_ratio_resolution);

		uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
		uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
		uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();
		const float* sptr = spaceweight.ptr<float>();
		uchar* off = offset_map.ptr<uchar>();

		if (src_highres.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				const int OY4 = y3 * offset_map.cols;
				const int OY5 = y3 * offset_map.cols;
				const int OY6 = y3 * offset_map.cols;
				const int OY7 = y3 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 3);
					const int x1 = max(0, x_ - 2);
					const int x2 = max(0, x_ - 1);
					const int x3 = x_;
					const int x4 = min(x_ + 1, swidth - 1);
					const int x5 = min(x_ + 2, swidth - 1);
					const int x6 = min(x_ + 3, swidth - 1);
					const int x7 = min(x_ + 4, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int X4 = x4 * lut_num;
					const int X5 = x5 * lut_num;
					const int X6 = x6 * lut_num;
					const int X7 = x7 * lut_num;

					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y0 + X4, Y0 + X5, Y0 + X6, Y0 + X7);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3, Y1 + X4, Y1 + X5, Y1 + X6, Y1 + X7);
					const __m256i mlutidx2 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y2 + X4, Y2 + X5, Y2 + X6, Y2 + X7);
					const __m256i mlutidx3 = _mm256_setr_epi32(Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3, Y3 + X4, Y3 + X5, Y3 + X6, Y3 + X7);
					const __m256i mlutidx4 = _mm256_setr_epi32(Y4 + X0, Y4 + X1, Y4 + X2, Y4 + X3, Y4 + X4, Y4 + X5, Y4 + X6, Y4 + X7);
					const __m256i mlutidx5 = _mm256_setr_epi32(Y5 + X0, Y5 + X1, Y5 + X2, Y5 + X3, Y5 + X4, Y5 + X5, Y5 + X6, Y5 + X7);
					const __m256i mlutidx6 = _mm256_setr_epi32(Y6 + X0, Y6 + X1, Y6 + X2, Y6 + X3, Y6 + X4, Y6 + X5, Y6 + X6, Y6 + X7);
					const __m256i mlutidx7 = _mm256_setr_epi32(Y7 + X0, Y7 + X1, Y7 + X2, Y7 + X3, Y7 + X4, Y7 + X5, Y7 + X6, Y7 + X7);

					if constexpr (quantization)
					{
						const int OX0 = x0;
						const int OX1 = x1;
						const int OX2 = x2;
						const int OX3 = x3;
						const int OX4 = x4;
						const int OX5 = x5;
						const int OX6 = x6;
						const int OX7 = x7;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY0 + OX4, OY0 + OX5, OY0 + OX6, OY0 + OX7);
						const __m256i moffidx1 = _mm256_setr_epi32(OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3, OY1 + OX4, OY1 + OX5, OY1 + OX6, OY1 + OX7);
						const __m256i moffidx2 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY2 + OX4, OY2 + OX5, OY2 + OX6, OY2 + OX7);
						const __m256i moffidx3 = _mm256_setr_epi32(OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3, OY3 + OX4, OY3 + OX5, OY3 + OX6, OY3 + OX7);
						const __m256i moffidx4 = _mm256_setr_epi32(OY4 + OX0, OY4 + OX1, OY4 + OX2, OY4 + OX3, OY4 + OX4, OY4 + OX5, OY4 + OX6, OY4 + OX7);
						const __m256i moffidx5 = _mm256_setr_epi32(OY5 + OX0, OY5 + OX1, OY5 + OX2, OY5 + OX3, OY5 + OX4, OY5 + OX5, OY5 + OX6, OY5 + OX7);
						const __m256i moffidx6 = _mm256_setr_epi32(OY6 + OX0, OY6 + OX1, OY6 + OX2, OY6 + OX3, OY6 + OX4, OY6 + OX5, OY6 + OX6, OY6 + OX7);
						const __m256i moffidx7 = _mm256_setr_epi32(OY7 + OX0, OY7 + OX1, OY7 + OX2, OY7 + OX3, OY7 + OX4, OY7 + OX5, OY7 + OX6, OY7 + OX7);

						//offset map for rgb
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));
						__m256 moffb2 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx2));
						__m256 moffb3 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx3));
						__m256 moffb4 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx4));
						__m256 moffb5 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx5));
						__m256 moffb6 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx6));
						__m256 moffb7 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx7));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);
								const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//gray
								int intensity = src[idx];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_pre)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_pre)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_pre)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_pre)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_pre)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_pre)))), moffb7), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_nex)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_nex)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_nex)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_nex)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_nex)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_nex)))), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);

								//gray
								int intensity = src[idx];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mw1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));
				const float normalize = 1.f / range_quantization_ratio;

				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;

				__m256i moffidxY0;
				__m256i moffidxY1;
				__m256i moffidxY2;
				__m256i moffidxY3;
				__m256i moffidxY4;
				__m256i moffidxY5;
				__m256i moffidxY6;
				__m256i moffidxY7;
				if constexpr (quantization)
				{
					moffidxY0 = _mm256_set1_epi32(y0 * offset_map.cols);
					moffidxY1 = _mm256_set1_epi32(y1 * offset_map.cols);
					moffidxY2 = _mm256_set1_epi32(y2 * offset_map.cols);
					moffidxY3 = _mm256_set1_epi32(y3 * offset_map.cols);
					moffidxY4 = _mm256_set1_epi32(y4 * offset_map.cols);
					moffidxY5 = _mm256_set1_epi32(y5 * offset_map.cols);
					moffidxY6 = _mm256_set1_epi32(y6 * offset_map.cols);
					moffidxY7 = _mm256_set1_epi32(y7 * offset_map.cols);
				}
				const __m256i mlutidxY0 = _mm256_set1_epi32(Y0);
				const __m256i mlutidxY1 = _mm256_set1_epi32(Y1);
				const __m256i mlutidxY2 = _mm256_set1_epi32(Y2);
				const __m256i mlutidxY3 = _mm256_set1_epi32(Y3);
				const __m256i mlutidxY4 = _mm256_set1_epi32(Y4);
				const __m256i mlutidxY5 = _mm256_set1_epi32(Y5);
				const __m256i mlutidxY6 = _mm256_set1_epi32(Y6);
				const __m256i mlutidxY7 = _mm256_set1_epi32(Y7);

				const __m256i mxstep = _mm256_setr_epi32(-3, -2, -1, 0, 1, 2, 3, 4);
				const __m256i mxmax = _mm256_set1_epi32(swidth - 1);
				const __m256i mlut_num = _mm256_set1_epi32(lut_num);
				const __m256i mone = _mm256_set1_epi32(1);
				const __m256i mtwo = _mm256_set1_epi32(2);
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const __m256i mx = _mm256_min_epi32(mxmax, _mm256_max_epi32(_mm256_setzero_si256(), _mm256_add_epi32(_mm256_set1_epi32(x_), mxstep)));
					const __m256i mlutidxX = _mm256_mullo_epi32(mlut_num, mx);
					const __m256i mlutidx0 = _mm256_add_epi32(mlutidxY0, mlutidxX);
					const __m256i mlutidx1 = _mm256_add_epi32(mlutidxY1, mlutidxX);
					const __m256i mlutidx2 = _mm256_add_epi32(mlutidxY2, mlutidxX);
					const __m256i mlutidx3 = _mm256_add_epi32(mlutidxY3, mlutidxX);
					const __m256i mlutidx4 = _mm256_add_epi32(mlutidxY4, mlutidxX);
					const __m256i mlutidx5 = _mm256_add_epi32(mlutidxY5, mlutidxX);
					const __m256i mlutidx6 = _mm256_add_epi32(mlutidxY6, mlutidxX);
					const __m256i mlutidx7 = _mm256_add_epi32(mlutidxY7, mlutidxX);

					if constexpr (quantization)
					{
						const __m256i moffidxX = _mm256_add_epi32(mx, _mm256_slli_epi32(mx, 1));//mx*3
						const __m256i moffidx0 = _mm256_add_epi32(moffidxX, moffidxY0);
						const __m256i moffidx1 = _mm256_add_epi32(moffidxX, moffidxY1);
						const __m256i moffidx2 = _mm256_add_epi32(moffidxX, moffidxY2);
						const __m256i moffidx3 = _mm256_add_epi32(moffidxX, moffidxY3);
						const __m256i moffidx4 = _mm256_add_epi32(moffidxX, moffidxY4);
						const __m256i moffidx5 = _mm256_add_epi32(moffidxX, moffidxY5);
						const __m256i moffidx6 = _mm256_add_epi32(moffidxX, moffidxY6);
						const __m256i moffidx7 = _mm256_add_epi32(moffidxX, moffidxY7);

						//offset map for rgb
						const __m256 moffb0 = _mm256_i32gather_epu8cvtps(off, moffidx0);
						const __m256 moffb1 = _mm256_i32gather_epu8cvtps(off, moffidx1);
						const __m256 moffb2 = _mm256_i32gather_epu8cvtps(off, moffidx2);
						const __m256 moffb3 = _mm256_i32gather_epu8cvtps(off, moffidx3);
						const __m256 moffb4 = _mm256_i32gather_epu8cvtps(off, moffidx4);
						const __m256 moffb5 = _mm256_i32gather_epu8cvtps(off, moffidx5);
						const __m256 moffb6 = _mm256_i32gather_epu8cvtps(off, moffidx6);
						const __m256 moffb7 = _mm256_i32gather_epu8cvtps(off, moffidx7);
						const __m256 moffg0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mone));
						const __m256 moffg1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mone));
						const __m256 moffg2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mone));
						const __m256 moffg3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mone));
						const __m256 moffg4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mone));
						const __m256 moffg5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mone));
						const __m256 moffg6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mone));
						const __m256 moffg7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mone));
						const __m256 moffr0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mtwo));
						const __m256 moffr1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mtwo));
						const __m256 moffr2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mtwo));
						const __m256 moffr3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mtwo));
						const __m256 moffr4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mtwo));
						const __m256 moffr5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mtwo));
						const __m256 moffr6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mtwo));
						const __m256 moffr7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mtwo));

						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);
								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								uchar* lut = lutb + intensity_shift_pre;
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								lut = lutb + intensity_shift_nex;
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutg + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutg + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffg1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffg2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffg3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffg4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffg5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffg6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffg7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutr + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffr7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutr + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffr7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
					else // no quantization
					{
						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output	
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);
#if 1
								//default
								//b
								uchar* lut = lutb + src[idx + 0];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lut, mlutidx2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lut, mlutidx3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lut, mlutidx4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lut, mlutidx5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lut, mlutidx6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lut, mlutidx7), mv);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//g
								lut = lutg + src[idx + 1];
								mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lut, mlutidx2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lut, mlutidx3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lut, mlutidx4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lut, mlutidx5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lut, mlutidx6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lut, mlutidx7), mv);
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//r
								lut = lutr + src[idx + 2];
								mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lut, mlutidx2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lut, mlutidx3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lut, mlutidx4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lut, mlutidx5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lut, mlutidx6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lut, mlutidx7), mv);
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
#else
								//interleave BGR
								uchar* lb = lutb + src[idx + 0];
								uchar* lg = lutg + src[idx + 1];
								uchar* lr = lutr + src[idx + 2];
								__m256 mb = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lb, mlutidx0));
								__m256 mg = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lg, mlutidx0));
								__m256 mr = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lr, mlutidx0));
								mb = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lb, mlutidx1), mb);
								mg = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lg, mlutidx1), mg);
								mr = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lr, mlutidx1), mr);
								mb = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lb, mlutidx2), mb);
								mg = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lg, mlutidx2), mg);
								mr = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lr, mlutidx2), mr);
								mb = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lb, mlutidx3), mb);
								mg = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lg, mlutidx3), mg);
								mr = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lr, mlutidx3), mr);
								mb = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lb, mlutidx4), mb);
								mg = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lg, mlutidx4), mg);
								mr = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lr, mlutidx4), mr);
								mb = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lb, mlutidx5), mb);
								mg = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lg, mlutidx5), mg);
								mr = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lr, mlutidx5), mr);
								mb = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lb, mlutidx6), mb);
								mg = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lg, mlutidx6), mg);
								mr = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lr, mlutidx6), mr);
								mb = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lb, mlutidx7), mb);
								mg = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lg, mlutidx7), mg);
								mr = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lr, mlutidx7), mr);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mb));
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mg));
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mr));
#endif
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
				}
			}
		}
	}

	template<bool quantization, int scale>
	void LocalLUTUpsample::_tensorUpConv64LinearScale(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const int lut_num, const bool isOffset)
	{
		if (!isOffset && quantization) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);

		uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
		uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
		uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();
		const float* sptr = spaceweight.ptr<float>();
		uchar* off = offset_map.ptr<uchar>();

		if (src_highres.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				const int OY4 = y3 * offset_map.cols;
				const int OY5 = y3 * offset_map.cols;
				const int OY6 = y3 * offset_map.cols;
				const int OY7 = y3 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 3);
					const int x1 = max(0, x_ - 2);
					const int x2 = max(0, x_ - 1);
					const int x3 = x_;
					const int x4 = min(x_ + 1, swidth - 1);
					const int x5 = min(x_ + 2, swidth - 1);
					const int x6 = min(x_ + 3, swidth - 1);
					const int x7 = min(x_ + 4, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int X4 = x4 * lut_num;
					const int X5 = x5 * lut_num;
					const int X6 = x6 * lut_num;
					const int X7 = x7 * lut_num;

					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y0 + X4, Y0 + X5, Y0 + X6, Y0 + X7);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3, Y1 + X4, Y1 + X5, Y1 + X6, Y1 + X7);
					const __m256i mlutidx2 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y2 + X4, Y2 + X5, Y2 + X6, Y2 + X7);
					const __m256i mlutidx3 = _mm256_setr_epi32(Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3, Y3 + X4, Y3 + X5, Y3 + X6, Y3 + X7);
					const __m256i mlutidx4 = _mm256_setr_epi32(Y4 + X0, Y4 + X1, Y4 + X2, Y4 + X3, Y4 + X4, Y4 + X5, Y4 + X6, Y4 + X7);
					const __m256i mlutidx5 = _mm256_setr_epi32(Y5 + X0, Y5 + X1, Y5 + X2, Y5 + X3, Y5 + X4, Y5 + X5, Y5 + X6, Y5 + X7);
					const __m256i mlutidx6 = _mm256_setr_epi32(Y6 + X0, Y6 + X1, Y6 + X2, Y6 + X3, Y6 + X4, Y6 + X5, Y6 + X6, Y6 + X7);
					const __m256i mlutidx7 = _mm256_setr_epi32(Y7 + X0, Y7 + X1, Y7 + X2, Y7 + X3, Y7 + X4, Y7 + X5, Y7 + X6, Y7 + X7);

					if constexpr (quantization)
					{
						const int OX0 = x0;
						const int OX1 = x1;
						const int OX2 = x2;
						const int OX3 = x3;
						const int OX4 = x4;
						const int OX5 = x5;
						const int OX6 = x6;
						const int OX7 = x7;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY0 + OX4, OY0 + OX5, OY0 + OX6, OY0 + OX7);
						const __m256i moffidx1 = _mm256_setr_epi32(OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3, OY1 + OX4, OY1 + OX5, OY1 + OX6, OY1 + OX7);
						const __m256i moffidx2 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY2 + OX4, OY2 + OX5, OY2 + OX6, OY2 + OX7);
						const __m256i moffidx3 = _mm256_setr_epi32(OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3, OY3 + OX4, OY3 + OX5, OY3 + OX6, OY3 + OX7);
						const __m256i moffidx4 = _mm256_setr_epi32(OY4 + OX0, OY4 + OX1, OY4 + OX2, OY4 + OX3, OY4 + OX4, OY4 + OX5, OY4 + OX6, OY4 + OX7);
						const __m256i moffidx5 = _mm256_setr_epi32(OY5 + OX0, OY5 + OX1, OY5 + OX2, OY5 + OX3, OY5 + OX4, OY5 + OX5, OY5 + OX6, OY5 + OX7);
						const __m256i moffidx6 = _mm256_setr_epi32(OY6 + OX0, OY6 + OX1, OY6 + OX2, OY6 + OX3, OY6 + OX4, OY6 + OX5, OY6 + OX6, OY6 + OX7);
						const __m256i moffidx7 = _mm256_setr_epi32(OY7 + OX0, OY7 + OX1, OY7 + OX2, OY7 + OX3, OY7 + OX4, OY7 + OX5, OY7 + OX6, OY7 + OX7);

						//offset map for rgb
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));
						__m256 moffb2 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx2));
						__m256 moffb3 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx3));
						__m256 moffb4 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx4));
						__m256 moffb5 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx5));
						__m256 moffb6 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx6));
						__m256 moffb7 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx7));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);
								const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//gray
								int intensity = src[idx];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_pre)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_pre)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_pre)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_pre)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_pre)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_pre)))), moffb7), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_nex)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_nex)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_nex)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_nex)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_nex)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_nex)))), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);

								//gray
								int intensity = src[idx];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mw1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));
				const float normalize = 1.f / range_quantization_ratio;

				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;

				__m256i moffidxY0;
				__m256i moffidxY1;
				__m256i moffidxY2;
				__m256i moffidxY3;
				__m256i moffidxY4;
				__m256i moffidxY5;
				__m256i moffidxY6;
				__m256i moffidxY7;
				if constexpr (quantization)
				{
					moffidxY0 = _mm256_set1_epi32(y0 * offset_map.cols);
					moffidxY1 = _mm256_set1_epi32(y1 * offset_map.cols);
					moffidxY2 = _mm256_set1_epi32(y2 * offset_map.cols);
					moffidxY3 = _mm256_set1_epi32(y3 * offset_map.cols);
					moffidxY4 = _mm256_set1_epi32(y4 * offset_map.cols);
					moffidxY5 = _mm256_set1_epi32(y5 * offset_map.cols);
					moffidxY6 = _mm256_set1_epi32(y6 * offset_map.cols);
					moffidxY7 = _mm256_set1_epi32(y7 * offset_map.cols);
				}
				const __m256i mlutidxY0 = _mm256_set1_epi32(Y0);
				const __m256i mlutidxY1 = _mm256_set1_epi32(Y1);
				const __m256i mlutidxY2 = _mm256_set1_epi32(Y2);
				const __m256i mlutidxY3 = _mm256_set1_epi32(Y3);
				const __m256i mlutidxY4 = _mm256_set1_epi32(Y4);
				const __m256i mlutidxY5 = _mm256_set1_epi32(Y5);
				const __m256i mlutidxY6 = _mm256_set1_epi32(Y6);
				const __m256i mlutidxY7 = _mm256_set1_epi32(Y7);

				const __m256i mxstep = _mm256_setr_epi32(-3, -2, -1, 0, 1, 2, 3, 4);
				const __m256i mxmax = _mm256_set1_epi32(swidth - 1);
				const __m256i mlut_num = _mm256_set1_epi32(lut_num);
				const __m256i mone = _mm256_set1_epi32(1);
				const __m256i mtwo = _mm256_set1_epi32(2);
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const __m256i mx = _mm256_min_epi32(mxmax, _mm256_max_epi32(_mm256_setzero_si256(), _mm256_add_epi32(_mm256_set1_epi32(x_), mxstep)));
					const __m256i mlutidxX = _mm256_mullo_epi32(mlut_num, mx);
					const __m256i mlutidx0 = _mm256_add_epi32(mlutidxY0, mlutidxX);
					const __m256i mlutidx1 = _mm256_add_epi32(mlutidxY1, mlutidxX);
					const __m256i mlutidx2 = _mm256_add_epi32(mlutidxY2, mlutidxX);
					const __m256i mlutidx3 = _mm256_add_epi32(mlutidxY3, mlutidxX);
					const __m256i mlutidx4 = _mm256_add_epi32(mlutidxY4, mlutidxX);
					const __m256i mlutidx5 = _mm256_add_epi32(mlutidxY5, mlutidxX);
					const __m256i mlutidx6 = _mm256_add_epi32(mlutidxY6, mlutidxX);
					const __m256i mlutidx7 = _mm256_add_epi32(mlutidxY7, mlutidxX);

					if constexpr (quantization)
					{
						const __m256i moffidxX = _mm256_add_epi32(mx, _mm256_slli_epi32(mx, 1));//mx*3
						const __m256i moffidx0 = _mm256_add_epi32(moffidxX, moffidxY0);
						const __m256i moffidx1 = _mm256_add_epi32(moffidxX, moffidxY1);
						const __m256i moffidx2 = _mm256_add_epi32(moffidxX, moffidxY2);
						const __m256i moffidx3 = _mm256_add_epi32(moffidxX, moffidxY3);
						const __m256i moffidx4 = _mm256_add_epi32(moffidxX, moffidxY4);
						const __m256i moffidx5 = _mm256_add_epi32(moffidxX, moffidxY5);
						const __m256i moffidx6 = _mm256_add_epi32(moffidxX, moffidxY6);
						const __m256i moffidx7 = _mm256_add_epi32(moffidxX, moffidxY7);

						//offset map for rgb
						const __m256 moffb0 = _mm256_i32gather_epu8cvtps(off, moffidx0);
						const __m256 moffg0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mone));
						const __m256 moffr0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mtwo));
						const __m256 moffb1 = _mm256_i32gather_epu8cvtps(off, moffidx1);
						const __m256 moffg1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mone));
						const __m256 moffr1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mtwo));
						const __m256 moffb2 = _mm256_i32gather_epu8cvtps(off, moffidx2);
						const __m256 moffg2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mone));
						const __m256 moffr2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mtwo));
						const __m256 moffb3 = _mm256_i32gather_epu8cvtps(off, moffidx3);
						const __m256 moffg3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mone));
						const __m256 moffr3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mtwo));
						const __m256 moffb4 = _mm256_i32gather_epu8cvtps(off, moffidx4);
						const __m256 moffg4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mone));
						const __m256 moffr4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mtwo));
						const __m256 moffb5 = _mm256_i32gather_epu8cvtps(off, moffidx5);
						const __m256 moffg5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mone));
						const __m256 moffr5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mtwo));
						const __m256 moffb6 = _mm256_i32gather_epu8cvtps(off, moffidx6);
						const __m256 moffg6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mone));
						const __m256 moffr6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mtwo));
						const __m256 moffb7 = _mm256_i32gather_epu8cvtps(off, moffidx7);
						const __m256 moffg7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mone));
						const __m256 moffr7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mtwo));

						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);
								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								uchar* lut = lutb + intensity_shift_pre;
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								lut = lutb + intensity_shift_nex;
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutg + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffg1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffg2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffg3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffg4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffg5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffg6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffg7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutg + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffg1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffg2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffg3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffg4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffg5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffg6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffg7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutr + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffr7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutr + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffr7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
					else // no quantization
					{
						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output	
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);
#if 1
								//default
								//b
								uchar* lut = lutb + src[idx + 0];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lut, mlutidx2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lut, mlutidx3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lut, mlutidx4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lut, mlutidx5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lut, mlutidx6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lut, mlutidx7), mv);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//g
								lut = lutg + src[idx + 1];
								mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lut, mlutidx2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lut, mlutidx3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lut, mlutidx4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lut, mlutidx5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lut, mlutidx6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lut, mlutidx7), mv);
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//r
								lut = lutr + src[idx + 2];
								mv = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lut, mlutidx0));
								mv = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lut, mlutidx1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lut, mlutidx2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lut, mlutidx3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lut, mlutidx4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lut, mlutidx5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lut, mlutidx6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lut, mlutidx7), mv);
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
#else
								//interleave BGR
								uchar* lb = lutb + src[idx + 0];
								uchar* lg = lutg + src[idx + 1];
								uchar* lr = lutr + src[idx + 2];
								__m256 mb = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lb, mlutidx0));
								__m256 mg = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lg, mlutidx0));
								__m256 mr = _mm256_mul_ps(mw0, _mm256_i32gather_epu8cvtps(lr, mlutidx0));
								mb = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lb, mlutidx1), mb);
								mg = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lg, mlutidx1), mg);
								mr = _mm256_fmadd_ps(mw1, _mm256_i32gather_epu8cvtps(lr, mlutidx1), mr);
								mb = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lb, mlutidx2), mb);
								mg = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lg, mlutidx2), mg);
								mr = _mm256_fmadd_ps(mw2, _mm256_i32gather_epu8cvtps(lr, mlutidx2), mr);
								mb = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lb, mlutidx3), mb);
								mg = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lg, mlutidx3), mg);
								mr = _mm256_fmadd_ps(mw3, _mm256_i32gather_epu8cvtps(lr, mlutidx3), mr);
								mb = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lb, mlutidx4), mb);
								mg = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lg, mlutidx4), mg);
								mr = _mm256_fmadd_ps(mw4, _mm256_i32gather_epu8cvtps(lr, mlutidx4), mr);
								mb = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lb, mlutidx5), mb);
								mg = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lg, mlutidx5), mg);
								mr = _mm256_fmadd_ps(mw5, _mm256_i32gather_epu8cvtps(lr, mlutidx5), mr);
								mb = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lb, mlutidx6), mb);
								mg = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lg, mlutidx6), mg);
								mr = _mm256_fmadd_ps(mw6, _mm256_i32gather_epu8cvtps(lr, mlutidx6), mr);
								mb = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lb, mlutidx7), mb);
								mg = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lg, mlutidx7), mg);
								mr = _mm256_fmadd_ps(mw7, _mm256_i32gather_epu8cvtps(lr, mlutidx7), mr);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mb));
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mg));
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mr));
#endif
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
				}
			}
		}
	}

	template<bool quantization>
	void LocalLUTUpsample::_tensorUpConv64LinearSoA(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const int lut_num, const bool isOffset)
	{
		if (!isOffset && quantization) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int scale = int(up_sampling_ratio_resolution);

		uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
		uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
		uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();
		const float* sptr = spaceweight.ptr<float>();
		uchar* off = offset_map.ptr<uchar>();

		if (src_highres.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				const int OY4 = y3 * offset_map.cols;
				const int OY5 = y3 * offset_map.cols;
				const int OY6 = y3 * offset_map.cols;
				const int OY7 = y3 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 3);
					const int x1 = max(0, x_ - 2);
					const int x2 = max(0, x_ - 1);
					const int x3 = x_;
					const int x4 = min(x_ + 1, swidth - 1);
					const int x5 = min(x_ + 2, swidth - 1);
					const int x6 = min(x_ + 3, swidth - 1);
					const int x7 = min(x_ + 4, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int X4 = x4 * lut_num;
					const int X5 = x5 * lut_num;
					const int X6 = x6 * lut_num;
					const int X7 = x7 * lut_num;

					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y0 + X4, Y0 + X5, Y0 + X6, Y0 + X7);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3, Y1 + X4, Y1 + X5, Y1 + X6, Y1 + X7);
					const __m256i mlutidx2 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y2 + X4, Y2 + X5, Y2 + X6, Y2 + X7);
					const __m256i mlutidx3 = _mm256_setr_epi32(Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3, Y3 + X4, Y3 + X5, Y3 + X6, Y3 + X7);
					const __m256i mlutidx4 = _mm256_setr_epi32(Y4 + X0, Y4 + X1, Y4 + X2, Y4 + X3, Y4 + X4, Y4 + X5, Y4 + X6, Y4 + X7);
					const __m256i mlutidx5 = _mm256_setr_epi32(Y5 + X0, Y5 + X1, Y5 + X2, Y5 + X3, Y5 + X4, Y5 + X5, Y5 + X6, Y5 + X7);
					const __m256i mlutidx6 = _mm256_setr_epi32(Y6 + X0, Y6 + X1, Y6 + X2, Y6 + X3, Y6 + X4, Y6 + X5, Y6 + X6, Y6 + X7);
					const __m256i mlutidx7 = _mm256_setr_epi32(Y7 + X0, Y7 + X1, Y7 + X2, Y7 + X3, Y7 + X4, Y7 + X5, Y7 + X6, Y7 + X7);

					if constexpr (quantization)
					{
						const int OX0 = x0;
						const int OX1 = x1;
						const int OX2 = x2;
						const int OX3 = x3;
						const int OX4 = x4;
						const int OX5 = x5;
						const int OX6 = x6;
						const int OX7 = x7;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY0 + OX4, OY0 + OX5, OY0 + OX6, OY0 + OX7);
						const __m256i moffidx1 = _mm256_setr_epi32(OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3, OY1 + OX4, OY1 + OX5, OY1 + OX6, OY1 + OX7);
						const __m256i moffidx2 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY2 + OX4, OY2 + OX5, OY2 + OX6, OY2 + OX7);
						const __m256i moffidx3 = _mm256_setr_epi32(OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3, OY3 + OX4, OY3 + OX5, OY3 + OX6, OY3 + OX7);
						const __m256i moffidx4 = _mm256_setr_epi32(OY4 + OX0, OY4 + OX1, OY4 + OX2, OY4 + OX3, OY4 + OX4, OY4 + OX5, OY4 + OX6, OY4 + OX7);
						const __m256i moffidx5 = _mm256_setr_epi32(OY5 + OX0, OY5 + OX1, OY5 + OX2, OY5 + OX3, OY5 + OX4, OY5 + OX5, OY5 + OX6, OY5 + OX7);
						const __m256i moffidx6 = _mm256_setr_epi32(OY6 + OX0, OY6 + OX1, OY6 + OX2, OY6 + OX3, OY6 + OX4, OY6 + OX5, OY6 + OX6, OY6 + OX7);
						const __m256i moffidx7 = _mm256_setr_epi32(OY7 + OX0, OY7 + OX1, OY7 + OX2, OY7 + OX3, OY7 + OX4, OY7 + OX5, OY7 + OX6, OY7 + OX7);

						//offset map for rgb
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));
						__m256 moffb2 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx2));
						__m256 moffb3 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx3));
						__m256 moffb4 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx4));
						__m256 moffb5 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx5));
						__m256 moffb6 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx6));
						__m256 moffb7 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx7));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);
								const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//gray
								int intensity = src[idx];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_pre)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_pre)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_pre)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_pre)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_pre)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_pre)))), moffb7), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_nex)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_nex)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_nex)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_nex)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_nex)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_nex)))), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);

								//gray
								int intensity = src[idx];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mw1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
				uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
				uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();

				const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));
				const float normalize = 1.f / range_quantization_ratio;

				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);
				AutoBuffer <uchar*> lutptrB0(lut_num);
				AutoBuffer <uchar*> lutptrG0(lut_num);
				AutoBuffer <uchar*> lutptrR0(lut_num);
				AutoBuffer <uchar*> lutptrB1(lut_num);
				AutoBuffer <uchar*> lutptrG1(lut_num);
				AutoBuffer <uchar*> lutptrR1(lut_num);
				AutoBuffer <uchar*> lutptrB2(lut_num);
				AutoBuffer <uchar*> lutptrG2(lut_num);
				AutoBuffer <uchar*> lutptrR2(lut_num);
				AutoBuffer <uchar*> lutptrB3(lut_num);
				AutoBuffer <uchar*> lutptrG3(lut_num);
				AutoBuffer <uchar*> lutptrR3(lut_num);
				AutoBuffer <uchar*> lutptrB4(lut_num);
				AutoBuffer <uchar*> lutptrG4(lut_num);
				AutoBuffer <uchar*> lutptrR4(lut_num);
				AutoBuffer <uchar*> lutptrB5(lut_num);
				AutoBuffer <uchar*> lutptrG5(lut_num);
				AutoBuffer <uchar*> lutptrR5(lut_num);
				AutoBuffer <uchar*> lutptrB6(lut_num);
				AutoBuffer <uchar*> lutptrG6(lut_num);
				AutoBuffer <uchar*> lutptrR6(lut_num);
				AutoBuffer <uchar*> lutptrB7(lut_num);
				AutoBuffer <uchar*> lutptrG7(lut_num);
				AutoBuffer <uchar*> lutptrR7(lut_num);
				for (int i = 0; i < lut_num; i++)
				{
					lutptrB0[i] = LUT_TensorSoA_B[i].ptr<uchar>(y0);
					lutptrG0[i] = LUT_TensorSoA_G[i].ptr<uchar>(y0);
					lutptrR0[i] = LUT_TensorSoA_R[i].ptr<uchar>(y0);
					lutptrB1[i] = LUT_TensorSoA_B[i].ptr<uchar>(y1);
					lutptrG1[i] = LUT_TensorSoA_G[i].ptr<uchar>(y1);
					lutptrR1[i] = LUT_TensorSoA_R[i].ptr<uchar>(y1);
					lutptrB2[i] = LUT_TensorSoA_B[i].ptr<uchar>(y2);
					lutptrG2[i] = LUT_TensorSoA_G[i].ptr<uchar>(y2);
					lutptrR2[i] = LUT_TensorSoA_R[i].ptr<uchar>(y2);
					lutptrB3[i] = LUT_TensorSoA_B[i].ptr<uchar>(y3);
					lutptrG3[i] = LUT_TensorSoA_G[i].ptr<uchar>(y3);
					lutptrR3[i] = LUT_TensorSoA_R[i].ptr<uchar>(y3);
					lutptrB4[i] = LUT_TensorSoA_B[i].ptr<uchar>(y4);
					lutptrG4[i] = LUT_TensorSoA_G[i].ptr<uchar>(y4);
					lutptrR4[i] = LUT_TensorSoA_R[i].ptr<uchar>(y4);
					lutptrB5[i] = LUT_TensorSoA_B[i].ptr<uchar>(y5);
					lutptrG5[i] = LUT_TensorSoA_G[i].ptr<uchar>(y5);
					lutptrR5[i] = LUT_TensorSoA_R[i].ptr<uchar>(y5);
					lutptrB6[i] = LUT_TensorSoA_B[i].ptr<uchar>(y6);
					lutptrG6[i] = LUT_TensorSoA_G[i].ptr<uchar>(y6);
					lutptrR6[i] = LUT_TensorSoA_R[i].ptr<uchar>(y6);
					lutptrB7[i] = LUT_TensorSoA_B[i].ptr<uchar>(y7);
					lutptrG7[i] = LUT_TensorSoA_G[i].ptr<uchar>(y7);
					lutptrR7[i] = LUT_TensorSoA_R[i].ptr<uchar>(y7);
				}

				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					//const __m256i mx = _mm256_min_epi32(mxmax, _mm256_max_epi32(_mm256_setzero_si256(), _mm256_add_epi32(_mm256_set1_epi32(x_), mxstep)));

					if constexpr (quantization)
					{
#if 0
						const __m256i moffidxX = _mm256_add_epi32(mx, _mm256_slli_epi32(mx, 1));//mx*3
						const __m256i moffidx0 = _mm256_add_epi32(moffidxX, moffidxY0);
						const __m256i moffidx1 = _mm256_add_epi32(moffidxX, moffidxY1);
						const __m256i moffidx2 = _mm256_add_epi32(moffidxX, moffidxY2);
						const __m256i moffidx3 = _mm256_add_epi32(moffidxX, moffidxY3);
						const __m256i moffidx4 = _mm256_add_epi32(moffidxX, moffidxY4);
						const __m256i moffidx5 = _mm256_add_epi32(moffidxX, moffidxY5);
						const __m256i moffidx6 = _mm256_add_epi32(moffidxX, moffidxY6);
						const __m256i moffidx7 = _mm256_add_epi32(moffidxX, moffidxY7);

						//offset map for rgb
						const __m256 moffb0 = _mm256_i32gather_epu8cvtps(off, moffidx0);
						const __m256 moffb1 = _mm256_i32gather_epu8cvtps(off, moffidx1);
						const __m256 moffb2 = _mm256_i32gather_epu8cvtps(off, moffidx2);
						const __m256 moffb3 = _mm256_i32gather_epu8cvtps(off, moffidx3);
						const __m256 moffb4 = _mm256_i32gather_epu8cvtps(off, moffidx4);
						const __m256 moffb5 = _mm256_i32gather_epu8cvtps(off, moffidx5);
						const __m256 moffb6 = _mm256_i32gather_epu8cvtps(off, moffidx6);
						const __m256 moffb7 = _mm256_i32gather_epu8cvtps(off, moffidx7);
						const __m256 moffg0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mone));
						const __m256 moffg1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mone));
						const __m256 moffg2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mone));
						const __m256 moffg3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mone));
						const __m256 moffg4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mone));
						const __m256 moffg5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mone));
						const __m256 moffg6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mone));
						const __m256 moffg7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mone));
						const __m256 moffr0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mtwo));
						const __m256 moffr1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mtwo));
						const __m256 moffr2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mtwo));
						const __m256 moffr3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mtwo));
						const __m256 moffr4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mtwo));
						const __m256 moffr5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mtwo));
						const __m256 moffr6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mtwo));
						const __m256 moffr7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mtwo));

						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);
								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								uchar* lut = lutb + intensity_shift_pre;
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								lut = lutb + intensity_shift_nex;
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutg + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffb7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutg + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffg1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffg2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffg3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffg4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffg5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffg6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffg7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								lut = lutr + intensity_shift_pre;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffr7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								lut = lutr + intensity_shift_nex;
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx0), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx1), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx2), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx3), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx4), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx5), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx6), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lut, mlutidx7), moffr7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								}
								else
								{
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
								}
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
#endif 
					}
					else // no quantization
					{
						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output	
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);
								//b
								int v = src[idx + 0];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_load_epu8cvtps((__m128i*)(lutptrB0[v] + x_ - 3)));
								mv = _mm256_fmadd_ps(mw1, _mm256_load_epu8cvtps((__m128i*)(lutptrB1[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_load_epu8cvtps((__m128i*)(lutptrB2[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_load_epu8cvtps((__m128i*)(lutptrB3[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_load_epu8cvtps((__m128i*)(lutptrB4[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_load_epu8cvtps((__m128i*)(lutptrB5[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_load_epu8cvtps((__m128i*)(lutptrB6[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_load_epu8cvtps((__m128i*)(lutptrB7[v] + x_ - 3)), mv);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//g
								v = src[idx + 1];
								mv = _mm256_mul_ps(mw0, _mm256_load_epu8cvtps((__m128i*)(lutptrG0[v] + x_ - 3)));
								mv = _mm256_fmadd_ps(mw1, _mm256_load_epu8cvtps((__m128i*)(lutptrG1[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_load_epu8cvtps((__m128i*)(lutptrG2[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_load_epu8cvtps((__m128i*)(lutptrG3[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_load_epu8cvtps((__m128i*)(lutptrG4[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_load_epu8cvtps((__m128i*)(lutptrG5[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_load_epu8cvtps((__m128i*)(lutptrG6[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_load_epu8cvtps((__m128i*)(lutptrG7[v] + x_ - 3)), mv);
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
								//r
								v = src[idx + 2];
								mv = _mm256_mul_ps(mw0, _mm256_load_epu8cvtps((__m128i*)(lutptrR0[v] + x_ - 3)));
								mv = _mm256_fmadd_ps(mw1, _mm256_load_epu8cvtps((__m128i*)(lutptrR1[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_load_epu8cvtps((__m128i*)(lutptrR2[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_load_epu8cvtps((__m128i*)(lutptrR3[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_load_epu8cvtps((__m128i*)(lutptrR4[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_load_epu8cvtps((__m128i*)(lutptrR5[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_load_epu8cvtps((__m128i*)(lutptrR6[v] + x_ - 3)), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_load_epu8cvtps((__m128i*)(lutptrR7[v] + x_ - 3)), mv);
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
				}
			}
		}
	}

#pragma region debug
	inline __m256 selectcvt(__m256i* src, const int idx)
	{
		__m128i ret = _mm_insert_epi8(_mm_setzero_si128(), src[0].m256i_u8[idx], 0);
		ret = _mm_insert_epi8(ret, src[1].m256i_u8[idx], 1);
		ret = _mm_insert_epi8(ret, src[2].m256i_u8[idx], 2);
		ret = _mm_insert_epi8(ret, src[3].m256i_u8[idx], 3);
		ret = _mm_insert_epi8(ret, src[4].m256i_u8[idx], 4);
		ret = _mm_insert_epi8(ret, src[5].m256i_u8[idx], 5);
		ret = _mm_insert_epi8(ret, src[6].m256i_u8[idx], 6);
		ret = _mm_insert_epi8(ret, src[7].m256i_u8[idx], 7);
		return _mm256_cvtepu8_ps(ret);
	}

	inline void loadblock64(__m256i* dst, uchar* lut, int offset,
		const __m256i& mlutidx0, const __m256i& mlutidx1, const __m256i& mlutidx2, const __m256i& mlutidx3, const __m256i& mlutidx4, const __m256i& mlutidx5, const __m256i& mlutidx6, const __m256i& mlutidx7)
	{
		const uchar* ptr = lut + offset;
		dst[0] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 0)));
		dst[1] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 1)));
		dst[2] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 2)));
		dst[3] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 3)));
		dst[4] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 4)));
		dst[5] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 5)));
		dst[6] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 6)));
		dst[7] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx0, 7)));

		dst[8] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 0)));
		dst[9] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 1)));
		dst[10] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 2)));
		dst[11] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 3)));
		dst[12] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 4)));
		dst[13] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 5)));
		dst[14] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 6)));
		dst[15] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx1, 7)));

		dst[16] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 0)));
		dst[17] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 1)));
		dst[18] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 2)));
		dst[19] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 3)));
		dst[20] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 4)));
		dst[21] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 5)));
		dst[22] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 6)));
		dst[23] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx2, 7)));

		dst[24] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 0)));
		dst[25] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 1)));
		dst[26] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 2)));
		dst[27] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 3)));
		dst[28] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 4)));
		dst[29] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 5)));
		dst[30] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 6)));
		dst[31] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx3, 7)));

		dst[32] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 0)));
		dst[33] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 1)));
		dst[34] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 2)));
		dst[35] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 3)));
		dst[36] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 4)));
		dst[37] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 5)));
		dst[38] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 6)));
		dst[39] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx4, 7)));

		dst[40] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 0)));
		dst[41] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 1)));
		dst[42] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 2)));
		dst[43] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 3)));
		dst[44] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 4)));
		dst[45] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 5)));
		dst[46] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 6)));
		dst[47] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx5, 7)));

		dst[48] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 0)));
		dst[49] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 1)));
		dst[50] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 2)));
		dst[51] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 3)));
		dst[52] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 4)));
		dst[53] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 5)));
		dst[54] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 6)));
		dst[55] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx6, 7)));

		dst[56] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 0)));
		dst[57] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 1)));
		dst[58] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 2)));
		dst[59] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 3)));
		dst[60] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 4)));
		dst[61] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 5)));
		dst[62] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 6)));
		dst[63] = _mm256_loadu_si256((__m256i*)(ptr + _mm256_extract_epi32(mlutidx7, 7)));
	}

	template<bool quantization, int scale>
	void LocalLUTUpsample::_tensorUpConv64LinearLoadScale(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const int lut_num, const bool isOffset)
	{
		if (!isOffset && quantization) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);

		uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
		uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
		uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();
		const float* sptr = spaceweight.ptr<float>();
		uchar* off = offset_map.ptr<uchar>();

		if (src_highres.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				const int OY4 = y3 * offset_map.cols;
				const int OY5 = y3 * offset_map.cols;
				const int OY6 = y3 * offset_map.cols;
				const int OY7 = y3 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 3);
					const int x1 = max(0, x_ - 2);
					const int x2 = max(0, x_ - 1);
					const int x3 = x_;
					const int x4 = min(x_ + 1, swidth - 1);
					const int x5 = min(x_ + 2, swidth - 1);
					const int x6 = min(x_ + 3, swidth - 1);
					const int x7 = min(x_ + 4, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int X4 = x4 * lut_num;
					const int X5 = x5 * lut_num;
					const int X6 = x6 * lut_num;
					const int X7 = x7 * lut_num;

					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y0 + X4, Y0 + X5, Y0 + X6, Y0 + X7);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3, Y1 + X4, Y1 + X5, Y1 + X6, Y1 + X7);
					const __m256i mlutidx2 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y2 + X4, Y2 + X5, Y2 + X6, Y2 + X7);
					const __m256i mlutidx3 = _mm256_setr_epi32(Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3, Y3 + X4, Y3 + X5, Y3 + X6, Y3 + X7);
					const __m256i mlutidx4 = _mm256_setr_epi32(Y4 + X0, Y4 + X1, Y4 + X2, Y4 + X3, Y4 + X4, Y4 + X5, Y4 + X6, Y4 + X7);
					const __m256i mlutidx5 = _mm256_setr_epi32(Y5 + X0, Y5 + X1, Y5 + X2, Y5 + X3, Y5 + X4, Y5 + X5, Y5 + X6, Y5 + X7);
					const __m256i mlutidx6 = _mm256_setr_epi32(Y6 + X0, Y6 + X1, Y6 + X2, Y6 + X3, Y6 + X4, Y6 + X5, Y6 + X6, Y6 + X7);
					const __m256i mlutidx7 = _mm256_setr_epi32(Y7 + X0, Y7 + X1, Y7 + X2, Y7 + X3, Y7 + X4, Y7 + X5, Y7 + X6, Y7 + X7);

					if constexpr (quantization)
					{
						const int OX0 = x0;
						const int OX1 = x1;
						const int OX2 = x2;
						const int OX3 = x3;
						const int OX4 = x4;
						const int OX5 = x5;
						const int OX6 = x6;
						const int OX7 = x7;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY0 + OX4, OY0 + OX5, OY0 + OX6, OY0 + OX7);
						const __m256i moffidx1 = _mm256_setr_epi32(OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3, OY1 + OX4, OY1 + OX5, OY1 + OX6, OY1 + OX7);
						const __m256i moffidx2 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY2 + OX4, OY2 + OX5, OY2 + OX6, OY2 + OX7);
						const __m256i moffidx3 = _mm256_setr_epi32(OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3, OY3 + OX4, OY3 + OX5, OY3 + OX6, OY3 + OX7);
						const __m256i moffidx4 = _mm256_setr_epi32(OY4 + OX0, OY4 + OX1, OY4 + OX2, OY4 + OX3, OY4 + OX4, OY4 + OX5, OY4 + OX6, OY4 + OX7);
						const __m256i moffidx5 = _mm256_setr_epi32(OY5 + OX0, OY5 + OX1, OY5 + OX2, OY5 + OX3, OY5 + OX4, OY5 + OX5, OY5 + OX6, OY5 + OX7);
						const __m256i moffidx6 = _mm256_setr_epi32(OY6 + OX0, OY6 + OX1, OY6 + OX2, OY6 + OX3, OY6 + OX4, OY6 + OX5, OY6 + OX6, OY6 + OX7);
						const __m256i moffidx7 = _mm256_setr_epi32(OY7 + OX0, OY7 + OX1, OY7 + OX2, OY7 + OX3, OY7 + OX4, OY7 + OX5, OY7 + OX6, OY7 + OX7);

						//offset map for rgb
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));
						__m256 moffb2 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx2));
						__m256 moffb3 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx3));
						__m256 moffb4 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx4));
						__m256 moffb5 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx5));
						__m256 moffb6 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx6));
						__m256 moffb7 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx7));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);
								const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//gray
								int intensity = src[idx];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_pre)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_pre)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_pre)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_pre)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_pre)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_pre)))), moffb7), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_nex)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_nex)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_nex)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_nex)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_nex)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_nex)))), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);

								//gray
								const int intensity = src[idx];
								__m256 mv = _mm256_mul_ps(mw0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mw1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;

				__m256i moffidxY0;
				__m256i moffidxY1;
				__m256i moffidxY2;
				__m256i moffidxY3;
				__m256i moffidxY4;
				__m256i moffidxY5;
				__m256i moffidxY6;
				__m256i moffidxY7;
				if constexpr (quantization)
				{
					moffidxY0 = _mm256_set1_epi32(y0 * offset_map.cols);
					moffidxY1 = _mm256_set1_epi32(y1 * offset_map.cols);
					moffidxY2 = _mm256_set1_epi32(y2 * offset_map.cols);
					moffidxY3 = _mm256_set1_epi32(y3 * offset_map.cols);
					moffidxY4 = _mm256_set1_epi32(y4 * offset_map.cols);
					moffidxY5 = _mm256_set1_epi32(y5 * offset_map.cols);
					moffidxY6 = _mm256_set1_epi32(y6 * offset_map.cols);
					moffidxY7 = _mm256_set1_epi32(y7 * offset_map.cols);
				}
				const __m256i mlutidxY0 = _mm256_set1_epi32(Y0);
				const __m256i mlutidxY1 = _mm256_set1_epi32(Y1);
				const __m256i mlutidxY2 = _mm256_set1_epi32(Y2);
				const __m256i mlutidxY3 = _mm256_set1_epi32(Y3);
				const __m256i mlutidxY4 = _mm256_set1_epi32(Y4);
				const __m256i mlutidxY5 = _mm256_set1_epi32(Y5);
				const __m256i mlutidxY6 = _mm256_set1_epi32(Y6);
				const __m256i mlutidxY7 = _mm256_set1_epi32(Y7);
				__m256i mxstep = _mm256_setr_epi32(-3, -2, -1, 0, 1, 2, 3, 4);
				__m256i mxmax = _mm256_set1_epi32(swidth - 1);
				__m256i mlut_num = _mm256_set1_epi32(lut_num);
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const __m256i mx = _mm256_min_epi32(mxmax, _mm256_max_epi32(_mm256_setzero_si256(), _mm256_add_epi32(_mm256_set1_epi32(x_), mxstep)));
					const __m256i mlutidxX = _mm256_mullo_epi32(mlut_num, mx);
					/*
					const int x0 = max(0, x_ - 3);
					const int x1 = max(0, x_ - 2);
					const int x2 = max(0, x_ - 1);
					const int x3 = x_;
					const int x4 = min(x_ + 1, swidth - 1);
					const int x5 = min(x_ + 2, swidth - 1);
					const int x6 = min(x_ + 3, swidth - 1);
					const int x7 = min(x_ + 4, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int X4 = x4 * lut_num;
					const int X5 = x5 * lut_num;
					const int X6 = x6 * lut_num;
					const int X7 = x7 * lut_num;
					const __m256i mlutidxX = _mm256_setr_epi32(X0, X1, X2, X3, X4, X5, X6, X7);
					*/
					const __m256i mlutidx0 = _mm256_add_epi32(mlutidxY0, mlutidxX);
					const __m256i mlutidx1 = _mm256_add_epi32(mlutidxY1, mlutidxX);
					const __m256i mlutidx2 = _mm256_add_epi32(mlutidxY2, mlutidxX);
					const __m256i mlutidx3 = _mm256_add_epi32(mlutidxY3, mlutidxX);
					const __m256i mlutidx4 = _mm256_add_epi32(mlutidxY4, mlutidxX);
					const __m256i mlutidx5 = _mm256_add_epi32(mlutidxY5, mlutidxX);
					const __m256i mlutidx6 = _mm256_add_epi32(mlutidxY6, mlutidxX);
					const __m256i mlutidx7 = _mm256_add_epi32(mlutidxY7, mlutidxX);

					if constexpr (quantization)
					{
						const __m256i moffidxX = _mm256_add_epi32(mx, _mm256_slli_epi32(mx, 1));//mx*3
						const __m256i moffidx0 = _mm256_add_epi32(moffidxX, moffidxY0);
						const __m256i moffidx1 = _mm256_add_epi32(moffidxX, moffidxY1);
						const __m256i moffidx2 = _mm256_add_epi32(moffidxX, moffidxY2);
						const __m256i moffidx3 = _mm256_add_epi32(moffidxX, moffidxY3);
						const __m256i moffidx4 = _mm256_add_epi32(moffidxX, moffidxY4);
						const __m256i moffidx5 = _mm256_add_epi32(moffidxX, moffidxY5);
						const __m256i moffidx6 = _mm256_add_epi32(moffidxX, moffidxY6);
						const __m256i moffidx7 = _mm256_add_epi32(moffidxX, moffidxY7);

						//offset map for rgb
						const __m256 moffb0 = _mm256_i32gather_epu8cvtps(off, moffidx0);
						const __m256 moffb1 = _mm256_i32gather_epu8cvtps(off, moffidx1);
						const __m256 moffb2 = _mm256_i32gather_epu8cvtps(off, moffidx2);
						const __m256 moffb3 = _mm256_i32gather_epu8cvtps(off, moffidx3);
						const __m256 moffb4 = _mm256_i32gather_epu8cvtps(off, moffidx4);
						const __m256 moffb5 = _mm256_i32gather_epu8cvtps(off, moffidx5);
						const __m256 moffb6 = _mm256_i32gather_epu8cvtps(off, moffidx6);
						const __m256 moffb7 = _mm256_i32gather_epu8cvtps(off, moffidx7);
						const __m256i mone = _mm256_set1_epi32(1);
						const __m256 moffg0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mone));
						const __m256 moffg1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mone));
						const __m256 moffg2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mone));
						const __m256 moffg3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mone));
						const __m256 moffg4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mone));
						const __m256 moffg5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mone));
						const __m256 moffg6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mone));
						const __m256 moffg7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mone));
						const __m256i mtwo = _mm256_set1_epi32(2);
						const __m256 moffr0 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx0, mtwo));
						const __m256 moffr1 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx1, mtwo));
						const __m256 moffr2 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx2, mtwo));
						const __m256 moffr3 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx3, mtwo));
						const __m256 moffr4 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx4, mtwo));
						const __m256 moffr5 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx5, mtwo));
						const __m256 moffr6 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx6, mtwo));
						const __m256 moffr7 = _mm256_i32gather_epu8cvtps(off, _mm256_add_epi32(moffidx7, mtwo));

						const uchar* src = src_highres.ptr<uchar>(y); // reference
						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						uchar* dest = dst.ptr<uchar>(y); // output
						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);

								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;
								__m256i mshiftpre = _mm256_set1_epi32(intensity_shift_pre);
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256i mshiftnex = _mm256_set1_epi32(intensity_shift_nex);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx0, mshiftpre)), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx1, mshiftpre)), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx2, mshiftpre)), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx3, mshiftpre)), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx4, mshiftpre)), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx5, mshiftpre)), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx6, mshiftpre)), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx7, mshiftpre)), moffb7), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx0, mshiftnex)), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx1, mshiftnex)), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx2, mshiftnex)), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx3, mshiftnex)), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx4, mshiftnex)), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx5, mshiftnex)), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx6, mshiftnex)), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutb, _mm256_add_epi32(mlutidx7, mshiftnex)), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));

								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								mshiftpre = _mm256_set1_epi32(intensity_shift_pre);
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mshiftnex = _mm256_set1_epi32(intensity_shift_nex);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx0, mshiftpre)), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx1, mshiftpre)), moffb1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx2, mshiftpre)), moffb2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx3, mshiftpre)), moffb3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx4, mshiftpre)), moffb4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx5, mshiftpre)), moffb5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx6, mshiftpre)), moffb6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx7, mshiftpre)), moffb7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx0, mshiftnex)), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx1, mshiftnex)), moffg1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx2, mshiftnex)), moffg2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx3, mshiftnex)), moffg3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx4, mshiftnex)), moffg4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx5, mshiftnex)), moffg5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx6, mshiftnex)), moffg6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_i32gather_epu8cvtps(lutg, _mm256_add_epi32(mlutidx7, mshiftnex)), moffg7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));

								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								mshiftpre = _mm256_set1_epi32(intensity_shift_pre);
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mshiftnex = _mm256_set1_epi32(intensity_shift_nex);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, mshiftpre))), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, mshiftpre))), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx2, mshiftpre))), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx3, mshiftpre))), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx4, mshiftpre))), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx5, mshiftpre))), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx6, mshiftpre))), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx7, mshiftpre))), moffr7), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, mshiftnex))), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, mshiftnex))), moffr1), mv);
								mv = _mm256_fmadd_ps(mw2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx2, mshiftnex))), moffr2), mv);
								mv = _mm256_fmadd_ps(mw3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx3, mshiftnex))), moffr3), mv);
								mv = _mm256_fmadd_ps(mw4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx4, mshiftnex))), moffr4), mv);
								mv = _mm256_fmadd_ps(mw5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx5, mshiftnex))), moffr5), mv);
								mv = _mm256_fmadd_ps(mw6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx6, mshiftnex))), moffr6), mv);
								mv = _mm256_fmadd_ps(mw7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx7, mshiftnex))), moffr7), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
					else
					{
						const __m256* wptr = (__m256*)spaceweight.ptr<float>();
						const uchar* src = src_highres.ptr<uchar>(y); // reference
						uchar* dest = dst.ptr<uchar>(y); // output	

						__m256i buffB[64];
						__m256i buffG[64];
						__m256i buffR[64];
						const int offset = 0;
						uchar bmin = 255;
						uchar gmin = 255;
						uchar rmin = 255;
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							for (int m = 0; m < scale; m++)
							{
								const int idx = 3 * (x + m);
								bmin = min(bmin, src[idx + 0]);
								gmin = min(gmin, src[idx + 1]);
								rmin = min(rmin, src[idx + 2]);
							}
						}
						loadblock64(buffB, lutb, bmin, mlutidx0, mlutidx1, mlutidx2, mlutidx3, mlutidx4, mlutidx5, mlutidx6, mlutidx7);
						loadblock64(buffG, lutg, gmin, mlutidx0, mlutidx1, mlutidx2, mlutidx3, mlutidx4, mlutidx5, mlutidx6, mlutidx7);
						loadblock64(buffR, lutr, rmin, mlutidx0, mlutidx1, mlutidx2, mlutidx3, mlutidx4, mlutidx5, mlutidx6, mlutidx7);

						for (int n = 0; n < scale; n++)
						{
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = *wptr++;
								const __m256 mw1 = *wptr++;
								const __m256 mw2 = *wptr++;
								const __m256 mw3 = *wptr++;
								const __m256 mw4 = *wptr++;
								const __m256 mw5 = *wptr++;
								const __m256 mw6 = *wptr++;
								const __m256 mw7 = *wptr++;
								const int idx = 3 * (x + m);
								int sidx = src[idx + 0] - bmin;
								__m256 mv = _mm256_mul_ps(mw0, selectcvt(buffB, sidx));
								mv = _mm256_fmadd_ps(mw1, selectcvt(buffB + 8, sidx), mv);
								mv = _mm256_fmadd_ps(mw2, selectcvt(buffB + 16, sidx), mv);
								mv = _mm256_fmadd_ps(mw3, selectcvt(buffB + 24, sidx), mv);
								mv = _mm256_fmadd_ps(mw4, selectcvt(buffB + 32, sidx), mv);
								mv = _mm256_fmadd_ps(mw5, selectcvt(buffB + 40, sidx), mv);
								mv = _mm256_fmadd_ps(mw6, selectcvt(buffB + 48, sidx), mv);
								mv = _mm256_fmadd_ps(mw7, selectcvt(buffB + 56, sidx), mv);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));

								sidx = src[idx + 1] - gmin;
								mv = _mm256_mul_ps(mw0, selectcvt(buffG, sidx));
								mv = _mm256_fmadd_ps(mw1, selectcvt(buffG + 8, sidx), mv);
								mv = _mm256_fmadd_ps(mw2, selectcvt(buffG + 16, sidx), mv);
								mv = _mm256_fmadd_ps(mw3, selectcvt(buffG + 24, sidx), mv);
								mv = _mm256_fmadd_ps(mw4, selectcvt(buffG + 32, sidx), mv);
								mv = _mm256_fmadd_ps(mw5, selectcvt(buffG + 40, sidx), mv);
								mv = _mm256_fmadd_ps(mw6, selectcvt(buffG + 48, sidx), mv);
								mv = _mm256_fmadd_ps(mw7, selectcvt(buffG + 56, sidx), mv);
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));

								sidx = src[idx + 2] - rmin;
								mv = _mm256_mul_ps(mw0, selectcvt(buffR, sidx));
								mv = _mm256_fmadd_ps(mw1, selectcvt(buffR + 8, sidx), mv);
								mv = _mm256_fmadd_ps(mw2, selectcvt(buffR + 16, sidx), mv);
								mv = _mm256_fmadd_ps(mw3, selectcvt(buffR + 24, sidx), mv);
								mv = _mm256_fmadd_ps(mw4, selectcvt(buffR + 32, sidx), mv);
								mv = _mm256_fmadd_ps(mw5, selectcvt(buffR + 40, sidx), mv);
								mv = _mm256_fmadd_ps(mw6, selectcvt(buffR + 48, sidx), mv);
								mv = _mm256_fmadd_ps(mw7, selectcvt(buffR + 56, sidx), mv);
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
							}
							src += src_highres.cols * 3;
							dest += dst.cols * 3;
						}
					}
				}
			}
		}
	}

	template<bool quantization>
	void LocalLUTUpsample::_tensorUpBilateralConv64Linear(const Mat& src_highres, Mat& dst, const Mat& spaceweight, const float sigma_range, const int lut_num, const bool isOffset)
	{
		const int r = (src_low_border.cols - lowres_size.width) / 2;
		float* expTable = (float*)_mm_malloc(sizeof(float) * 256, AVX_ALIGN);
		const float range_coeff = 1.f / (-2.f * sigma_range * sigma_range);
		for (int i = 0; i < 256; i++)
		{
			expTable[i] = exp((i * i) * range_coeff);
		}

		if (!isOffset) offset_map.setTo(0);
		const int swidth = int(src_highres.cols / up_sampling_ratio_resolution);
		const int sheight = int(src_highres.rows / up_sampling_ratio_resolution);

		const int range_quantization_ratio = 256 / lut_num;
		const int sshift = (int)log2(range_quantization_ratio);
		const int rshift = (int)log2(lut_num);

		const int scale = int(up_sampling_ratio_resolution);

		uchar* low = src_low_border.ptr<uchar>();
		uchar* off = offset_map.ptr<uchar>();

		if (src_highres.channels() == 1)
		{
			uchar* lut = LUT_TensorAoS_B.ptr<uchar>();

#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;
				const int YL0 = (y0 + r) * src_low_border.cols;
				const int YL1 = (y1 + r) * src_low_border.cols;
				const int YL2 = (y2 + r) * src_low_border.cols;
				const int YL3 = (y3 + r) * src_low_border.cols;
				const int YL4 = (y4 + r) * src_low_border.cols;
				const int YL5 = (y5 + r) * src_low_border.cols;
				const int YL6 = (y6 + r) * src_low_border.cols;
				const int YL7 = (y7 + r) * src_low_border.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				const int OY4 = y4 * offset_map.cols;
				const int OY5 = y5 * offset_map.cols;
				const int OY6 = y6 * offset_map.cols;
				const int OY7 = y7 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 3);
					const int x1 = max(0, x_ - 2);
					const int x2 = max(0, x_ - 1);
					const int x3 = x_;
					const int x4 = min(x_ + 1, swidth - 1);
					const int x5 = min(x_ + 2, swidth - 1);
					const int x6 = min(x_ + 3, swidth - 1);
					const int x7 = min(x_ + 4, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int X4 = x4 * lut_num;
					const int X5 = x5 * lut_num;
					const int X6 = x6 * lut_num;
					const int X7 = x7 * lut_num;
					const int XL0 = (x0 + r);
					const int XL1 = (x1 + r);
					const int XL2 = (x2 + r);
					const int XL3 = (x3 + r);
					const int XL4 = (x4 + r);
					const int XL5 = (x5 + r);
					const int XL6 = (x6 + r);
					const int XL7 = (x7 + r);

					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y0 + X4, Y0 + X5, Y0 + X6, Y0 + X7);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3, Y1 + X4, Y1 + X5, Y1 + X6, Y1 + X7);
					const __m256i mlutidx2 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y2 + X4, Y2 + X5, Y2 + X6, Y2 + X7);
					const __m256i mlutidx3 = _mm256_setr_epi32(Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3, Y3 + X4, Y3 + X5, Y3 + X6, Y3 + X7);
					const __m256i mlutidx4 = _mm256_setr_epi32(Y4 + X0, Y4 + X1, Y4 + X2, Y4 + X3, Y4 + X4, Y4 + X5, Y4 + X6, Y4 + X7);
					const __m256i mlutidx5 = _mm256_setr_epi32(Y5 + X0, Y5 + X1, Y5 + X2, Y5 + X3, Y5 + X4, Y5 + X5, Y5 + X6, Y5 + X7);
					const __m256i mlutidx6 = _mm256_setr_epi32(Y6 + X0, Y6 + X1, Y6 + X2, Y6 + X3, Y6 + X4, Y6 + X5, Y6 + X6, Y6 + X7);
					const __m256i mlutidx7 = _mm256_setr_epi32(Y7 + X0, Y7 + X1, Y7 + X2, Y7 + X3, Y7 + X4, Y7 + X5, Y7 + X6, Y7 + X7);

					const __m256i mlowidx0 = _mm256_setr_epi32(YL0 + XL0, YL0 + XL1, YL0 + XL2, YL0 + XL3, YL0 + XL4, YL0 + XL5, YL0 + XL6, YL0 + XL7);
					const __m256i mlowidx1 = _mm256_setr_epi32(YL1 + XL0, YL1 + XL1, YL1 + XL2, YL1 + XL3, YL1 + XL4, YL1 + XL5, YL1 + XL6, YL1 + XL7);
					const __m256i mlowidx2 = _mm256_setr_epi32(YL2 + XL0, YL2 + XL1, YL2 + XL2, YL2 + XL3, YL2 + XL4, YL2 + XL5, YL2 + XL6, YL2 + XL7);
					const __m256i mlowidx3 = _mm256_setr_epi32(YL3 + XL0, YL3 + XL1, YL3 + XL2, YL3 + XL3, YL3 + XL4, YL3 + XL5, YL3 + XL6, YL3 + XL7);
					const __m256i mlowidx4 = _mm256_setr_epi32(YL4 + XL0, YL4 + XL1, YL4 + XL2, YL4 + XL3, YL4 + XL4, YL4 + XL5, YL4 + XL6, YL4 + XL7);
					const __m256i mlowidx5 = _mm256_setr_epi32(YL5 + XL0, YL5 + XL1, YL5 + XL2, YL5 + XL3, YL5 + XL4, YL5 + XL5, YL5 + XL6, YL5 + XL7);
					const __m256i mlowidx6 = _mm256_setr_epi32(YL6 + XL0, YL6 + XL1, YL6 + XL2, YL6 + XL3, YL6 + XL4, YL6 + XL5, YL6 + XL6, YL6 + XL7);
					const __m256i mlowidx7 = _mm256_setr_epi32(YL7 + XL0, YL7 + XL1, YL7 + XL2, YL7 + XL3, YL7 + XL4, YL7 + XL5, YL7 + XL6, YL7 + XL7);

					if constexpr (quantization)
					{
						const int OX0 = x0;
						const int OX1 = x1;
						const int OX2 = x2;
						const int OX3 = x3;
						const int OX4 = x4;
						const int OX5 = x5;
						const int OX6 = x6;
						const int OX7 = x7;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY0 + OX4, OY0 + OX5, OY0 + OX6, OY0 + OX7);
						const __m256i moffidx1 = _mm256_setr_epi32(OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3, OY1 + OX4, OY1 + OX5, OY1 + OX6, OY1 + OX7);
						const __m256i moffidx2 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY2 + OX4, OY2 + OX5, OY2 + OX6, OY2 + OX7);
						const __m256i moffidx3 = _mm256_setr_epi32(OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3, OY3 + OX4, OY3 + OX5, OY3 + OX6, OY3 + OX7);
						const __m256i moffidx4 = _mm256_setr_epi32(OY4 + OX0, OY4 + OX1, OY4 + OX2, OY4 + OX3, OY4 + OX4, OY4 + OX5, OY4 + OX6, OY4 + OX7);
						const __m256i moffidx5 = _mm256_setr_epi32(OY5 + OX0, OY5 + OX1, OY5 + OX2, OY5 + OX3, OY5 + OX4, OY5 + OX5, OY5 + OX6, OY5 + OX7);
						const __m256i moffidx6 = _mm256_setr_epi32(OY6 + OX0, OY6 + OX1, OY6 + OX2, OY6 + OX3, OY6 + OX4, OY6 + OX5, OY6 + OX6, OY6 + OX7);
						const __m256i moffidx7 = _mm256_setr_epi32(OY7 + OX0, OY7 + OX1, OY7 + OX2, OY7 + OX3, OY7 + OX4, OY7 + OX5, OY7 + OX6, OY7 + OX7);

						//offset map
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));
						__m256 moffb2 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx2));
						__m256 moffb3 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx3));
						__m256 moffb4 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx4));
						__m256 moffb5 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx5));
						__m256 moffb6 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx6));
						__m256 moffb7 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx7));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);
								const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//gray
								int intensity = src[idx];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256i mlows = _mm256_set1_epi32(intensity);
								__m256i mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx0));
								__m256i mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx1));
								__m256i mlow2 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx2));
								__m256i mlow3 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx3));
								__m256i mlow4 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx4));
								__m256i mlow5 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx5));
								__m256i mlow6 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx6));
								__m256i mlow7 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx7));
								__m256 mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								__m256 mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								__m256 mwr2 = _mm256_mul_ps(mw2, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow2, mlows)), 4));
								__m256 mwr3 = _mm256_mul_ps(mw3, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow3, mlows)), 4));
								__m256 mwr4 = _mm256_mul_ps(mw4, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow4, mlows)), 4));
								__m256 mwr5 = _mm256_mul_ps(mw5, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow5, mlows)), 4));
								__m256 mwr6 = _mm256_mul_ps(mw6, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow6, mlows)), 4));
								__m256 mwr7 = _mm256_mul_ps(mw7, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow7, mlows)), 4));

								__m256 mv = _mm256_mul_ps(mwr0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mwr1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mwr2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_pre)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mwr3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_pre)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mwr4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_pre)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mwr5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_pre)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mwr6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_pre)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mwr7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_pre)))), moffb7), mv);
								const float norm = _mm256_reduceadd_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(mwr0, mwr1), _mm256_add_ps(mwr2, mwr3)), _mm256_add_ps(_mm256_add_ps(mwr4, mwr5), _mm256_add_ps(mwr6, mwr7))));
								float sumpre = _mm256_reduceadd_ps(mv) / norm;

								mv = _mm256_mul_ps(mwr0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mwr1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								mv = _mm256_fmadd_ps(mwr2, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity_shift_nex)))), moffb2), mv);
								mv = _mm256_fmadd_ps(mwr3, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity_shift_nex)))), moffb3), mv);
								mv = _mm256_fmadd_ps(mwr4, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity_shift_nex)))), moffb4), mv);
								mv = _mm256_fmadd_ps(mwr5, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity_shift_nex)))), moffb5), mv);
								mv = _mm256_fmadd_ps(mwr6, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity_shift_nex)))), moffb6), mv);
								mv = _mm256_fmadd_ps(mwr7, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity_shift_nex)))), moffb7), mv);
								float sumnex = _mm256_reduceadd_ps(mv) / norm;

								if (intensity_shift_pre != lut_num - 1)
									dest[idx] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx] = saturate_cast<uchar>(linear_interpolation(float(intensity_shift_pre << sshift), sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else //no quantization
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = (x + m);

								//gray
								int intensity = src[idx];
								__m256i mlows = _mm256_set1_epi32(intensity);
								__m256i mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx0));
								__m256i mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx1));
								__m256i mlow2 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx2));
								__m256i mlow3 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx3));
								__m256i mlow4 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx4));
								__m256i mlow5 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx5));
								__m256i mlow6 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx6));
								__m256i mlow7 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, mlowidx7));
								__m256 mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								__m256 mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								__m256 mwr2 = _mm256_mul_ps(mw2, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow2, mlows)), 4));
								__m256 mwr3 = _mm256_mul_ps(mw3, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow3, mlows)), 4));
								__m256 mwr4 = _mm256_mul_ps(mw4, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow4, mlows)), 4));
								__m256 mwr5 = _mm256_mul_ps(mw5, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow5, mlows)), 4));
								__m256 mwr6 = _mm256_mul_ps(mw6, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow6, mlows)), 4));
								__m256 mwr7 = _mm256_mul_ps(mw7, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow7, mlows)), 4));

								__m256 mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lut, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(mwr0, mwr1), _mm256_add_ps(mwr2, mwr3)), _mm256_add_ps(_mm256_add_ps(mwr4, mwr5), _mm256_add_ps(mwr6, mwr7)))));
							}
						}
					}
				}
			}
		}
		else if (src_highres.channels() == 3)
		{
			uchar* lutb = LUT_TensorAoS_B.ptr<uchar>();
			uchar* lutg = LUT_TensorAoS_G.ptr<uchar>();
			uchar* lutr = LUT_TensorAoS_R.ptr<uchar>();

#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < src_highres.rows; y += scale)
			{
				const int y_ = (int)(y / up_sampling_ratio_resolution);
				const int y0 = max(0, y_ - 3);
				const int y1 = max(0, y_ - 2);
				const int y2 = max(0, y_ - 1);
				const int y3 = y_;
				const int y4 = min(y_ + 1, sheight - 1);
				const int y5 = min(y_ + 2, sheight - 1);
				const int y6 = min(y_ + 3, sheight - 1);
				const int y7 = min(y_ + 4, sheight - 1);

				const float normalize = 1.f / range_quantization_ratio;
				const int Y0 = y0 * lut_num * LUT_TensorAoS_B.cols;
				const int Y1 = y1 * lut_num * LUT_TensorAoS_B.cols;
				const int Y2 = y2 * lut_num * LUT_TensorAoS_B.cols;
				const int Y3 = y3 * lut_num * LUT_TensorAoS_B.cols;
				const int Y4 = y4 * lut_num * LUT_TensorAoS_B.cols;
				const int Y5 = y5 * lut_num * LUT_TensorAoS_B.cols;
				const int Y6 = y6 * lut_num * LUT_TensorAoS_B.cols;
				const int Y7 = y7 * lut_num * LUT_TensorAoS_B.cols;
				const int YL0 = (y0 + r) * 3 * src_low_border.cols;
				const int YL1 = (y1 + r) * 3 * src_low_border.cols;
				const int YL2 = (y2 + r) * 3 * src_low_border.cols;
				const int YL3 = (y3 + r) * 3 * src_low_border.cols;
				const int YL4 = (y4 + r) * 3 * src_low_border.cols;
				const int YL5 = (y5 + r) * 3 * src_low_border.cols;
				const int YL6 = (y6 + r) * 3 * src_low_border.cols;
				const int YL7 = (y7 + r) * 3 * src_low_border.cols;
				const int OY0 = y0 * offset_map.cols;
				const int OY1 = y1 * offset_map.cols;
				const int OY2 = y2 * offset_map.cols;
				const int OY3 = y3 * offset_map.cols;
				for (int x = 0; x < src_highres.cols; x += scale)
				{
					const int x_ = (int)(x / up_sampling_ratio_resolution);
					const int x0 = max(0, x_ - 3);
					const int x1 = max(0, x_ - 2);
					const int x2 = max(0, x_ - 1);
					const int x3 = x_;
					const int x4 = min(x_ + 1, swidth - 1);
					const int x5 = min(x_ + 2, swidth - 1);
					const int x6 = min(x_ + 3, swidth - 1);
					const int x7 = min(x_ + 4, swidth - 1);

					const int X0 = x0 * lut_num;
					const int X1 = x1 * lut_num;
					const int X2 = x2 * lut_num;
					const int X3 = x3 * lut_num;
					const int X4 = x4 * lut_num;
					const int X5 = x5 * lut_num;
					const int X6 = x6 * lut_num;
					const int X7 = x7 * lut_num;
					const int XL0 = (x0 + r) * 3;
					const int XL1 = (x1 + r) * 3;
					const int XL2 = (x2 + r) * 3;
					const int XL3 = (x3 + r) * 3;
					const int XL4 = (x4 + r) * 3;
					const int XL5 = (x5 + r) * 3;
					const int XL6 = (x6 + r) * 3;
					const int XL7 = (x7 + r) * 3;

					const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y0 + X4, Y0 + X5, Y0 + X6, Y0 + X7);
					const __m256i mlutidx1 = _mm256_setr_epi32(Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3, Y1 + X4, Y1 + X5, Y1 + X6, Y1 + X7);
					const __m256i mlutidx2 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y2 + X4, Y2 + X5, Y2 + X6, Y2 + X7);
					const __m256i mlutidx3 = _mm256_setr_epi32(Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3, Y3 + X4, Y3 + X5, Y3 + X6, Y3 + X7);
					const __m256i mlutidx4 = _mm256_setr_epi32(Y4 + X0, Y4 + X1, Y4 + X2, Y4 + X3, Y4 + X4, Y4 + X5, Y4 + X6, Y4 + X7);
					const __m256i mlutidx5 = _mm256_setr_epi32(Y5 + X0, Y5 + X1, Y5 + X2, Y5 + X3, Y5 + X4, Y5 + X5, Y5 + X6, Y5 + X7);
					const __m256i mlutidx6 = _mm256_setr_epi32(Y6 + X0, Y6 + X1, Y6 + X2, Y6 + X3, Y6 + X4, Y6 + X5, Y6 + X6, Y6 + X7);
					const __m256i mlutidx7 = _mm256_setr_epi32(Y7 + X0, Y7 + X1, Y7 + X2, Y7 + X3, Y7 + X4, Y7 + X5, Y7 + X6, Y7 + X7);

					const __m256i mlowidx0 = _mm256_setr_epi32(YL0 + XL0, YL0 + XL1, YL0 + XL2, YL0 + XL3, YL0 + XL4, YL0 + XL5, YL0 + XL6, YL0 + XL7);
					const __m256i mlowidx1 = _mm256_setr_epi32(YL1 + XL0, YL1 + XL1, YL1 + XL2, YL1 + XL3, YL1 + XL4, YL1 + XL5, YL1 + XL6, YL1 + XL7);
					const __m256i mlowidx2 = _mm256_setr_epi32(YL2 + XL0, YL2 + XL1, YL2 + XL2, YL2 + XL3, YL2 + XL4, YL2 + XL5, YL2 + XL6, YL2 + XL7);
					const __m256i mlowidx3 = _mm256_setr_epi32(YL3 + XL0, YL3 + XL1, YL3 + XL2, YL3 + XL3, YL3 + XL4, YL3 + XL5, YL3 + XL6, YL3 + XL7);
					const __m256i mlowidx4 = _mm256_setr_epi32(YL4 + XL0, YL4 + XL1, YL4 + XL2, YL4 + XL3, YL4 + XL4, YL4 + XL5, YL4 + XL6, YL4 + XL7);
					const __m256i mlowidx5 = _mm256_setr_epi32(YL5 + XL0, YL5 + XL1, YL5 + XL2, YL5 + XL3, YL5 + XL4, YL5 + XL5, YL5 + XL6, YL5 + XL7);
					const __m256i mlowidx6 = _mm256_setr_epi32(YL6 + XL0, YL6 + XL1, YL6 + XL2, YL6 + XL3, YL6 + XL4, YL6 + XL5, YL6 + XL6, YL6 + XL7);
					const __m256i mlowidx7 = _mm256_setr_epi32(YL7 + XL0, YL7 + XL1, YL7 + XL2, YL7 + XL3, YL7 + XL4, YL7 + XL5, YL7 + XL6, YL7 + XL7);


					if constexpr (quantization)
					{
						const int OX0 = x0 * 3;
						const int OX1 = x1 * 3;
						const int OX2 = x2 * 3;
						const int OX3 = x3 * 3;
						const __m256i moffidx0 = _mm256_setr_epi32(OY0 + OX0, OY0 + OX1, OY0 + OX2, OY0 + OX3, OY1 + OX0, OY1 + OX1, OY1 + OX2, OY1 + OX3);
						const __m256i moffidx1 = _mm256_setr_epi32(OY2 + OX0, OY2 + OX1, OY2 + OX2, OY2 + OX3, OY3 + OX0, OY3 + OX1, OY3 + OX2, OY3 + OX3);

						//offset map for rgb
						__m256 moffb0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx0));
						__m256 moffb1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, moffidx1));
						__m256 moffg0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx0, _mm256_set1_epi32(1))));
						__m256 moffg1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx1, _mm256_set1_epi32(1))));
						__m256 moffr0 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx0, _mm256_set1_epi32(2))));
						__m256 moffr1 = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(off, _mm256_add_epi32(moffidx1, _mm256_set1_epi32(2))));

						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);
							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr);
								const __m256 mw1 = _mm256_load_ps(wptr + 8);

								const int idx = 3 * (x + m);
								const __m256 mamp = _mm256_set1_ps(float(range_quantization_ratio));

								//b
								int intensity = src[idx + 0];
								int intensity_shift_pre = intensity >> sshift;
								int intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								__m256 mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffb1), mv);
								float sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffb0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffb1), mv);
								float sumnex = _mm256_reduceadd_ps(mv);
								uchar inter_pre = saturate_cast<uchar>(sumpre);
								uchar inter_nex = saturate_cast<uchar>(sumnex);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 0] = saturate_cast<uchar>(linear_interpolation(255.f - range_quantization_ratio, sumpre, float(intensity), 255.f, 255.f));

								//g
								intensity = src[idx + 1];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffg1), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffg0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffg1), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								inter_pre = saturate_cast<uchar>(sumpre);
								inter_nex = saturate_cast<uchar>(sumnex);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 1] = saturate_cast<uchar>(linear_interpolation(255.f - range_quantization_ratio, sumpre, float(intensity), 255.f, 255.f));

								//r
								intensity = src[idx + 2];
								intensity_shift_pre = intensity >> sshift;
								intensity_shift_nex = min(intensity_shift_pre + 1, lut_num - 1);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_pre)))), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_pre)))), moffr1), mv);
								sumpre = _mm256_reduceadd_ps(mv);
								mv = _mm256_mul_ps(mw0, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity_shift_nex)))), moffr0));
								mv = _mm256_fmadd_ps(mw1, _mm256_fmadd_ps(mamp, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity_shift_nex)))), moffr1), mv);
								sumnex = _mm256_reduceadd_ps(mv);
								inter_pre = saturate_cast<uchar>(sumpre);
								inter_nex = saturate_cast<uchar>(sumnex);
								if (intensity_shift_pre != lut_num - 1)
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation_withoutnormalize(float(intensity_shift_pre << sshift), sumpre, float(intensity), float(intensity_shift_nex << sshift), sumnex) * normalize);
								else
									dest[idx + 2] = saturate_cast<uchar>(linear_interpolation(255.f - range_quantization_ratio, sumpre, float(intensity), 255.f, 255.f));
							}
						}
					}
					else //no quantization
					{
						for (int n = 0; n < scale; n++)
						{
							const uchar* src = src_highres.ptr<uchar>(y + n); // reference
							uchar* dest = dst.ptr<uchar>(y + n); // output
							const float* wptr = spaceweight.ptr<float>(scale * n);

							for (int m = 0; m < scale; m++)
							{
								const __m256 mw0 = _mm256_load_ps(wptr + 64 * m + 0);
								const __m256 mw1 = _mm256_load_ps(wptr + 64 * m + 8);
								const __m256 mw2 = _mm256_load_ps(wptr + 64 * m + 16);
								const __m256 mw3 = _mm256_load_ps(wptr + 64 * m + 24);
								const __m256 mw4 = _mm256_load_ps(wptr + 64 * m + 32);
								const __m256 mw5 = _mm256_load_ps(wptr + 64 * m + 40);
								const __m256 mw6 = _mm256_load_ps(wptr + 64 * m + 48);
								const __m256 mw7 = _mm256_load_ps(wptr + 64 * m + 56);
								const int idx = 3 * (x + m);

								//b
								int intensity = src[idx + 0];
								__m256i mlows = _mm256_set1_epi32(intensity);
								__m256i mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx0, _mm256_set1_epi32(0))));
								__m256i mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx1, _mm256_set1_epi32(0))));
								__m256i mlow2 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx2, _mm256_set1_epi32(0))));
								__m256i mlow3 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx3, _mm256_set1_epi32(0))));
								__m256i mlow4 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx4, _mm256_set1_epi32(0))));
								__m256i mlow5 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx5, _mm256_set1_epi32(0))));
								__m256i mlow6 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx6, _mm256_set1_epi32(0))));
								__m256i mlow7 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx7, _mm256_set1_epi32(0))));
								__m256 mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								__m256 mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								__m256 mwr2 = _mm256_mul_ps(mw2, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow2, mlows)), 4));
								__m256 mwr3 = _mm256_mul_ps(mw3, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow3, mlows)), 4));
								__m256 mwr4 = _mm256_mul_ps(mw4, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow4, mlows)), 4));
								__m256 mwr5 = _mm256_mul_ps(mw5, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow5, mlows)), 4));
								__m256 mwr6 = _mm256_mul_ps(mw6, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow6, mlows)), 4));
								__m256 mwr7 = _mm256_mul_ps(mw7, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow7, mlows)), 4));
								__m256 mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutb, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx + 0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(mwr0, mwr1), _mm256_add_ps(mwr2, mwr3)), _mm256_add_ps(_mm256_add_ps(mwr4, mwr5), _mm256_add_ps(mwr6, mwr7)))));

								intensity = src[idx + 1];
								mlows = _mm256_set1_epi32(intensity);
								mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx0, _mm256_set1_epi32(1))));
								mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx1, _mm256_set1_epi32(1))));
								mlow2 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx2, _mm256_set1_epi32(1))));
								mlow3 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx3, _mm256_set1_epi32(1))));
								mlow4 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx4, _mm256_set1_epi32(1))));
								mlow5 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx5, _mm256_set1_epi32(1))));
								mlow6 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx6, _mm256_set1_epi32(1))));
								mlow7 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx7, _mm256_set1_epi32(1))));
								mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								mwr2 = _mm256_mul_ps(mw2, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow2, mlows)), 4));
								mwr3 = _mm256_mul_ps(mw3, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow3, mlows)), 4));
								mwr4 = _mm256_mul_ps(mw4, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow4, mlows)), 4));
								mwr5 = _mm256_mul_ps(mw5, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow5, mlows)), 4));
								mwr6 = _mm256_mul_ps(mw6, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow6, mlows)), 4));
								mwr7 = _mm256_mul_ps(mw7, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow7, mlows)), 4));
								mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutg, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx + 1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(mwr0, mwr1), _mm256_add_ps(mwr2, mwr3)), _mm256_add_ps(_mm256_add_ps(mwr4, mwr5), _mm256_add_ps(mwr6, mwr7)))));

								intensity = src[idx + 2];
								mlows = _mm256_set1_epi32(intensity);
								mlow0 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx0, _mm256_set1_epi32(1))));
								mlow1 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx1, _mm256_set1_epi32(1))));
								mlow2 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx2, _mm256_set1_epi32(1))));
								mlow3 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx3, _mm256_set1_epi32(1))));
								mlow4 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx4, _mm256_set1_epi32(1))));
								mlow5 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx5, _mm256_set1_epi32(1))));
								mlow6 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx6, _mm256_set1_epi32(1))));
								mlow7 = _mm256_cvtepu8_epi32(_mm256_i32gather_epu8(low, _mm256_add_epi32(mlowidx7, _mm256_set1_epi32(1))));
								mwr0 = _mm256_mul_ps(mw0, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow0, mlows)), 4));
								mwr1 = _mm256_mul_ps(mw1, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow1, mlows)), 4));
								mwr2 = _mm256_mul_ps(mw2, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow2, mlows)), 4));
								mwr3 = _mm256_mul_ps(mw3, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow3, mlows)), 4));
								mwr4 = _mm256_mul_ps(mw4, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow4, mlows)), 4));
								mwr5 = _mm256_mul_ps(mw5, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow5, mlows)), 4));
								mwr6 = _mm256_mul_ps(mw6, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow6, mlows)), 4));
								mwr7 = _mm256_mul_ps(mw7, _mm256_i32gather_ps(expTable, _mm256_abs_epi32(_mm256_sub_epi32(mlow7, mlows)), 4));
								mv = _mm256_mul_ps(mwr0, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx0, _mm256_set1_epi32(intensity)))));
								mv = _mm256_fmadd_ps(mwr1, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx1, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr2, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx2, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr3, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx3, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr4, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx4, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr5, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx5, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr6, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx6, _mm256_set1_epi32(intensity)))), mv);
								mv = _mm256_fmadd_ps(mwr7, _mm256_cvtepu8_ps(_mm256_i32gather_epu8(lutr, _mm256_add_epi32(mlutidx7, _mm256_set1_epi32(intensity)))), mv);
								dest[idx + 2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv) / _mm256_reduceadd_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(mwr0, mwr1), _mm256_add_ps(mwr2, mwr3)), _mm256_add_ps(_mm256_add_ps(mwr4, mwr5), _mm256_add_ps(mwr6, mwr7)))));
							}
						}
					}
				}
			}
		}

		_mm_free(expTable);
	}

	void LocalLUTUpsample::tensorUpBilateral64Linear(const Mat& src_highres, Mat& dst, const int lut_num, const float sigma_space, const float sigma_range, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 64, CV_32F);
		cp::setGaussianWeight8x8(weight, sigma_space);

		if (lut_num == 256)
		{
			_tensorUpBilateralConv64Linear<false>(src_highres, dst, weight, sigma_range, lut_num, isOffset);
		}
		else
		{
			_tensorUpBilateralConv64Linear<true>(src_highres, dst, weight, sigma_range, lut_num, isOffset);
		}
	}
#pragma endregion 

	void LocalLUTUpsample::tensorUpBox64Linear(const Mat& src_highres, Mat& dst, const int lut_num, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 64, CV_32F);
		weight.setTo(1.f / 64.f);

		if (lut_num == 256)
			_tensorUpConv64Linear<false>(src_highres, dst, weight, lut_num, isOffset);
		else
			_tensorUpConv64Linear<true>(src_highres, dst, weight, lut_num, isOffset);
	}

	void LocalLUTUpsample::tensorUpGauss64Linear(const Mat& src_highres, Mat& dst, const int lut_num, const float sigma, const bool isOffset)
	{
		const int scale = int(up_sampling_ratio_resolution);
		Mat weight(scale * scale, 64, CV_32F);
		cp::setGaussianWeight8x8(weight, sigma);

		if (lut_num == 256)
		{
			constexpr bool isLoad = false;
			int scale = int(up_sampling_ratio_resolution);
			if (isLoad)
			{
				//implementation for removing gather intrinsic, but it is not slow (and almost the same but has approximation)
				if (scale == 2) _tensorUpConv64LinearLoadScale<false, 2>(src_highres, dst, weight, lut_num, isOffset);
				else if (scale == 4) _tensorUpConv64LinearLoadScale<false, 4>(src_highres, dst, weight, lut_num, isOffset);
				else if (scale == 8) _tensorUpConv64LinearLoadScale<false, 8>(src_highres, dst, weight, lut_num, isOffset);
				else if (scale == 16) _tensorUpConv64LinearLoadScale<false, 16>(src_highres, dst, weight, lut_num, isOffset);
				else _tensorUpConv64Linear<false>(src_highres, dst, weight, lut_num, isOffset);
			}
			else
			{
				if (useSoA)
				{
					_tensorUpConv64LinearSoA<false>(src_highres, dst, weight, lut_num, isOffset);
				}
				else
				{
					/*if (scale == 2) _tensorUpConv64LinearScale<false, 2>(src_highres, dst, weight, lut_num, isOffset);
					else if (scale == 4) _tensorUpConv64LinearScale<false, 4>(src_highres, dst, weight, lut_num, isOffset);
					else if (scale == 8) _tensorUpConv64LinearScale<false, 8>(src_highres, dst, weight, lut_num, isOffset);
					else if (scale == 16) _tensorUpConv64LinearScale<false, 16>(src_highres, dst, weight, lut_num, isOffset);
					else*/
					_tensorUpConv64Linear<false>(src_highres, dst, weight, lut_num, isOffset);
				}
			}
		}
		else
		{
			_tensorUpConv64Linear<true>(src_highres, dst, weight, lut_num, isOffset);
		}
	}

#pragma endregion

#pragma endregion

	void LocalLUTUpsample::upsample(InputArray src_low, InputArray prc_low, InputArray src_high, OutputArray prc_high, const int r, const int lut_num, const int lut_filter_radius, const BUILD_LUT build_lut_method, const UPTENSOR tensorup_method, const BOUNDARY lut_boundary_method, const bool isUseOffsetMap)
	{
		CV_Assert(src_low.depth() == CV_8U);
		CV_Assert(prc_low.depth() == CV_8U);
		CV_Assert(src_low.size() == prc_low.size());
		prc_high.create(src_high.size(), src_low.type());
		Mat src = src_high.getMat();
		Mat dest = prc_high.getMat();

		patch_radius = r;
		const int border = BORDER_REPLICATE;

		up_sampling_ratio_resolution = float(int(src.cols / src_low.size().width));
		const int bitratio = 256 / lut_num;
		const int shift = (int)log2(bitratio);

		lowres_size = src_low.size();
		createLUTTensor(lowres_size.width, lowres_size.height, lut_num);

		copyMakeBorder(src_low, src_low_border, r, r, r, r, border);
		cp::bitshiftRight(src_low_border, src_low_border, shift);//if shift==0, there is no processing
		if (shift == 0)
		{
			offset_map.create(prc_low.size(), prc_low.type());
			copyMakeBorder(prc_low, prc_low_border, r, r, r, r, border);
		}
		else
		{
			cp::bitshiftRight(prc_low, offset_map_buffer, offset_map, shift);
			copyMakeBorder(offset_map_buffer, prc_low_border, r, r, r, r, border);
		}

		//build LUT Tensor
		{
			//cp::Timer t("build");
			switch (build_lut_method)
			{
			case BUILD_LUT::LInf_MIN:
			default:
				buildLocalLUTTensorDistanceMIN(0, lut_num, r, bitratio, lut_filter_radius, lut_boundary_method); break;
			case BUILD_LUT::L1_MIN:
				buildLocalLUTTensorDistanceMIN(1, lut_num, r, bitratio, lut_filter_radius, lut_boundary_method); break;
			case BUILD_LUT::L2_MIN:
				buildLocalLUTTensorDistanceMIN(2, lut_num, r, bitratio, lut_filter_radius, lut_boundary_method); break;
			case BUILD_LUT::FREQUENCY_MAX_WTA:
				buildLocalLUTTensorFrequencyMaxWTA8U(lut_num, r, bitratio, lut_filter_radius, lut_boundary_method); break;
				//getWTA16U(lut_num, r, ratio);		
			case BUILD_LUT::FREQUENCY_MAX_DP:
				buildLocalLUTTensorFrequencyMaxDP(lut_num, r, bitratio); break;
			}
		}

		//interpolate LUT Tensor
		{
			//if (isUseOffsetMap)cout << "use offset map" << endl;
			//cp::Timer t("tensorup");
			switch (tensorup_method)
			{
			case UPTENSOR::NEAREST:
				tensorUpNearestLinear(src, dest, lut_num, isUseOffsetMap); break;
			case UPTENSOR::BOX4:
				tensorUpBox4Linear(src, dest, lut_num, isUseOffsetMap); break;
			case UPTENSOR::BOX16:
				tensorUpBox16Linear(src, dest, lut_num, isUseOffsetMap); break;
			case UPTENSOR::BOX64:
				tensorUpBox64Linear(src, dest, lut_num, isUseOffsetMap); break;
			case UPTENSOR::GAUSS4:
				tensorUpGauss4Linear(src, dest, lut_num, tensor_up_sigma_space, isUseOffsetMap); break;
			case UPTENSOR::GAUSS16:
			default:
				tensorUpGauss16Linear(src, dest, lut_num, tensor_up_sigma_space, isUseOffsetMap); break;
			case UPTENSOR::GAUSS64:
				tensorUpGauss64Linear(src, dest, lut_num, tensor_up_sigma_space, isUseOffsetMap); break;
			case UPTENSOR::LINEAR:
				tensorUpTriLinear(src, dest, lut_num, isUseOffsetMap); break;
			case UPTENSOR::CUBIC:
				tensorUpBiCubicLinear(src, dest, lut_num, tensor_up_cubic_alpha, isUseOffsetMap); break;
			case UPTENSOR::BILATERAL16:
				tensorUpBilateral16Linear(src, dest, lut_num, tensor_up_sigma_space, tensor_up_sigma_range, isUseOffsetMap); break;
			case UPTENSOR::BILATERAL64:
				tensorUpBilateral64Linear(src, dest, lut_num, tensor_up_sigma_space, tensor_up_sigma_range, isUseOffsetMap); break;
			case UPTENSOR::BoxNxN:
				tensorUpBoxNxNLinear(src, dest, tensor_up_kernel_size); break;
			case UPTENSOR::GaussNxN:
				tensorUpGaussNxNLinear(src, dest, tensor_up_kernel_size, tensor_up_sigma_space); break;
			case UPTENSOR::LaplaceNxN:
				tensorUpLaplaceNxNLinear(src, dest, tensor_up_kernel_size, tensor_up_sigma_space); break;
			}
		}
	}
}