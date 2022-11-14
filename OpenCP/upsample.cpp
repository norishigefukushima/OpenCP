#include "downsample.hpp"
#include "upsample.hpp"
#include "shiftimage.hpp"
#include "onelineCVFunctions.hpp"
#include "inlineSIMDFunctions.hpp"
#include "inlineMathFunctions.hpp"
#include "debugcp.hpp"
using namespace std;
using namespace cv;

namespace cp
{

	void vizCubicKernel(const int scale, string wname = "upsample cubic kernel")
	{
		Mat weightmap(scale * scale, 16, CV_32F);
		//weightmap.setTo(1);

		namedWindow(wname);
		int ix = 0; createTrackbar("ix", wname, &ix, scale - 1);
		int iy = 0; createTrackbar("iy", wname, &iy, scale - 1);
		int alpha = 100; createTrackbar("(alpha-200)*0.01", wname, &alpha, 400);
		int key = 0;
		Mat show;

		while (key != 'q')
		{
			cp::setCubicWeight4x4(weightmap, float((alpha - 200) * 0.01));
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

#pragma region nearst
	template <class Type>
	void upsampleNearest_(Mat& src, Mat& dest, const int scale)
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
	void upsampleNearest_<uchar>(Mat& src, Mat& dest, const int scale)
	{
		const int simdsize = src.cols / 16;
		if (scale == 2)
		{
			for (int j = 0; j < src.rows; j++)
			{
				__m128i* s = (__m128i*)(src.ptr<uchar>(j));
				__m128i* dl = (__m128i*)(dest.ptr<uchar>(2 * j));
				__m128i* dh = (__m128i*)(dest.ptr<uchar>(2 * j + 1));

				for (int i = simdsize; i != 0; i--)
				{
					const __m128i sv = _mm_load_si128(s); s++;
					const __m128i dvl = _mm_unpacklo_epi8(sv, sv);
					const __m128i dvh = _mm_unpackhi_epi8(sv, sv);
					_mm_store_si128(dl, dvl);
					_mm_store_si128(dl + 1, dvh);
					_mm_storeu_si128(dh, dvl);
					_mm_storeu_si128(dh + 1, dvh);
					dl += 2;
					dh += 2;
				}
			}
		}
#if 0
		else
		{
			for (int j = 0; j < src.rows; j++)
			{
				int n = j * scale;

				uchar* s = src.ptr<uchar>(j);
				for (int i = 0, m = 0; i < src.cols; i++, m += scale)
				{
					const uchar ltd = s[i];
					for (int l = 0; l < scale; l++)
					{
						uchar* d = dest.ptr<uchar>(n + l);
						for (int k = 0; k < scale; k++)
						{
							d[m + k] = ltd;
						}
					}
				}
			}
		}
#endif
	}

	template <>
	void upsampleNearest_<float>(Mat& src, Mat& dest, const int scale)
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
	void upsampleNearest_<double>(Mat& src, Mat& dest, const int scale)
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

	void upsampleNearest(InputArray src_, OutputArray dest_, const int scale)
	{
		if (scale == 1)
		{
			src_.copyTo(dest_);
			return;
		}

		Mat src = src_.getMat();

		if (dest_.empty() || dest_.size() != Size(src.cols * scale, src.rows * scale))
		{
			dest_.create(Size(src.cols * scale, src.rows * scale), src_.type());
		}
		Mat dest = dest_.getMat();


		if (src.depth() == CV_8U) upsampleNearest_<uchar>(src, dest, scale);
		else if (src.depth() == CV_16S) upsampleNearest_<short>(src, dest, scale);
		else if (src.depth() == CV_16U) upsampleNearest_<ushort>(src, dest, scale);
		else if (src.depth() == CV_32S) upsampleNearest_<int>(src, dest, scale);
		else if (src.depth() == CV_32F) upsampleNearest_<float>(src, dest, scale);
		else if (src.depth() == CV_64F) upsampleNearest_<double>(src, dest, scale);
	}

#if 0
	template <class srcType>
	static void nnUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols);
		const int dh = dest.rows / (src.rows);

		Mat sim; copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		for (int j = 0; j < src.rows; j++)
		{
			int n = j * dh;
			srcType* s = sim.ptr<srcType>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType ltd = s[i];
				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						d[m + k] = ltd;
					}
				}
			}
		}
	}

	void nnUpsample(InputArray src_, OutputArray dest_)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) nnUpsample_<uchar>(src, dest);
		else if (src.depth() == CV_16S) nnUpsample_<short>(src, dest);
		else if (src.depth() == CV_16U) nnUpsample_<ushort>(src, dest);
		else if (src.depth() == CV_32S) nnUpsample_<int>(src, dest);
		else if (src.depth() == CV_32F) nnUpsample_<float>(src, dest);
		else if (src.depth() == CV_64F) nnUpsample_<double>(src, dest);
	}
#endif
#pragma endregion

#pragma region linear
	inline int linearinterpolate_(int lt, int rt, int lb, int rb, double a, double b)
	{
		return (int)((b * a * lt + b * (1.0 - a) * rt + (1.0 - b) * a * lb + (1.0 - b) * (1.0 - a) * rb) + 0.5);
	}

	template <class srcType>
	inline double linearinterpolate_(srcType lt, srcType rt, srcType lb, srcType rb, double a, double b)
	{
		return (b * a * lt + b * (1.0 - a) * rt + (1.0 - b) * a * lb + (1.0 - b) * (1.0 - a) * rb);
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

	void upsampleLinear(InputArray src_, OutputArray dest_, const int scale)
	{
		if (src_.channels() == 1)
		{
			Mat dest = dest_.getMat();
			Mat src = src_.getMat();

			if (src.depth() == CV_8U)       upsampleLinear_<uchar>(src, dest, scale);
			else if (src.depth() == CV_16S) upsampleLinear_<short>(src, dest, scale);
			else if (src.depth() == CV_16U) upsampleLinear_<ushort>(src, dest, scale);
			else if (src.depth() == CV_32S) upsampleLinear_<int>(src, dest, scale);
			else if (src.depth() == CV_32F) upsampleLinear_<float>(src, dest, scale);
			else if (src.depth() == CV_64F) upsampleLinear_<double>(src, dest, scale);
		}
		else
		{
			vector<Mat> s;
			vector<Mat> d(src_.channels());
			split(src_, s);
			for (int c = 0; c < src_.channels(); c++)
			{
				Mat src = s[c];
				Size sz = src.size() * scale;
				d[c].create(sz, src.depth());
				if (src.depth() == CV_8U)       upsampleLinear_<uchar>(src, d[c], scale);
				else if (src.depth() == CV_16S) upsampleLinear_<short>(src, d[c], scale);
				else if (src.depth() == CV_16U) upsampleLinear_<ushort>(src, d[c], scale);
				else if (src.depth() == CV_32S) upsampleLinear_<int>(src, d[c], scale);
				else if (src.depth() == CV_32F) upsampleLinear_<float>(src, d[c], scale);
				else if (src.depth() == CV_64F) upsampleLinear_<double>(src, d[c], scale);
			}
			merge(d, dest_);
		}
	}
#pragma endregion

#pragma region cubic
	void upsampleConv16_8UC1(const Mat& src, Mat& dest, const int scale, const double a)
	{
		const int dw = dest.cols / (src.cols);
		const int dh = dest.rows / (src.rows);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		vector<vector<float>> weight(dh * dw);
		for (int i = 0; i < weight.size(); i++)weight[i].resize(16);

		int idx = 0;
		for (int l = 0; l < dh; l++)
		{
			const double y = (double)l / (double)dh;
			for (int k = 0; k < dw; k++)
			{
				const double x = (double)k / (double)dw;
				weight[idx][0] = float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx][1] = float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx][2] = float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 + y, a));
				weight[idx][3] = float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 + y, a));

				weight[idx][4] = float(cp::cubic(1.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx][5] = float(cp::cubic(0.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx][6] = float(cp::cubic(1.0 - x, a) * cp::cubic(0.0 + y, a));
				weight[idx][7] = float(cp::cubic(2.0 - x, a) * cp::cubic(0.0 + y, a));

				weight[idx][8] =  float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx][9] =  float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx][10] = float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 - y, a));
				weight[idx][11] = float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 - y, a));

				weight[idx][12] = float(cp::cubic(1.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx][13] = float(cp::cubic(0.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx][14] = float(cp::cubic(1.0 - x, a) * cp::cubic(2.0 - y, a));
				weight[idx][15] = float(cp::cubic(2.0 - x, a) * cp::cubic(2.0 - y, a));

				double wsum = 0.0;
				for (int i = 0; i < 16; i++)wsum += weight[idx][i];
				for (int i = 0; i < 16; i++)weight[idx][i] = float(weight[idx][i]/wsum);

				idx++;
			}
		}

		const uchar* srcptr = src.ptr<uchar>(0); // reference
#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < dest.rows; y += scale)
		{
			const int y_ = (int)(y / scale);
			const int y0 = max(0, y_ - 1);
			const int y1 = y0 + 1;
			const int y2 = min(y_ + 1, src.rows - 1);
			const int y3 = min(y_ + 2, src.rows - 1);

			const int Y0 = y0 * src.cols;
			const int Y1 = y1 * src.cols;
			const int Y2 = y2 * src.cols;
			const int Y3 = y3 * src.cols;

			for (int x = 0; x < dest.cols; x += scale)
			{
				const int x_ = (int)(x / scale);
				const int x0 = max(0, x_ - 1);
				const int x1 = x_;
				const int x2 = min(x_ + 1, src.cols - 1);
				const int x3 = min(x_ + 2, src.cols - 1);

				const int X0 = x0;
				const int X1 = x1;
				const int X2 = x2;
				const int X3 = x3;
				const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3);
				const __m256i mlutidx1 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3);

				for (int n = 0; n < scale; n++)
				{
					uchar* destptr = dest.ptr<uchar>(y + n); // output
					for (int m = 0; m < scale; m++)
					{
						const float* weightmap_ptr = &weight[scale * n + m][0];
						const __m256 mw0 = _mm256_load_ps(weightmap_ptr);
						const __m256 mw1 = _mm256_load_ps(weightmap_ptr + 8);

						const int idx = (x + m);

						__m256 mv = _mm256_mul_ps(mw0, _mm256_cvtepi32_ps(_mm256_i32gather_epi8epi32(srcptr, mlutidx0)));
						mv = _mm256_fmadd_ps(mw1, _mm256_cvtepi32_ps(_mm256_i32gather_epi8epi32(srcptr, mlutidx1)), mv);
						destptr[idx] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
					}
				}
			}
		}
	}

	void upsampleConv16_8UC3(const Mat& src, Mat& dest, const int scale, const double a)
	{
		const int dw = dest.cols / (src.cols);
		const int dh = dest.rows / (src.rows);

		float* weight = (float*)_mm_malloc(dh * dw * 16 * sizeof(float), AVX_ALIGN);
		int idx = 0;
		for (int l = 0; l < dh; l++)
		{
			const double y = (double)l / (double)dh;
			for (int k = 0; k < dw; k++)
			{
				const double x = (double)k / (double)dw;
				weight[idx * 16 + 0] = float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx * 16 + 1] = float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx * 16 + 2] = float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 + y, a));
				weight[idx * 16 + 3] = float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 + y, a));

				weight[idx * 16 + 4] = float(cp::cubic(1.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx * 16 + 5] = float(cp::cubic(0.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx * 16 + 6] = float(cp::cubic(1.0 - x, a) * cp::cubic(0.0 + y, a));
				weight[idx * 16 + 7] = float(cp::cubic(2.0 - x, a) * cp::cubic(0.0 + y, a));

				weight[idx * 16 + 8] = float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx * 16 + 9] = float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx * 16 + 10] = float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 - y, a));
				weight[idx * 16 + 11] = float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 - y, a));

				weight[idx * 16 + 12] = float(cp::cubic(1.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx * 16 + 13] = float(cp::cubic(0.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx * 16 + 14] = float(cp::cubic(1.0 - x, a) * cp::cubic(2.0 - y, a));
				weight[idx * 16 + 15] = float(cp::cubic(2.0 - x, a) * cp::cubic(2.0 - y, a));

				double wsum = 0.0;
				for (int i = 0; i < 16; i++)wsum += (double)weight[idx * 16 + i];
				for (int i = 0; i < 16; i++)weight[idx * 16 + i] = float(weight[idx * 16 + i] / wsum);

				idx++;
			}
		}

		const uchar* srcptr = src.ptr<uchar>(); // reference
#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < src.rows; y++)
		{
			const int y0 = max(0, y - 1);
			const int y1 = y0 + 1;
			const int y2 = min(y + 1, src.rows - 1);
			const int y3 = min(y + 2, src.rows - 1);

			const int Y0 = y0 * 3 * src.cols;
			const int Y1 = y1 * 3 * src.cols;
			const int Y2 = y2 * 3 * src.cols;
			const int Y3 = y3 * 3 * src.cols;

			for (int x = 0; x < src.cols; x++)
			{
				const int X0 = max(0, x - 1) * 3;
				const int X1 = x * 3;
				const int X2 = min(x + 1, src.cols - 1) * 3;
				const int X3 = min(x + 2, src.cols - 1) * 3;

				const __m256i mlutidx0 = _mm256_setr_epi32(Y0 + X0, Y0 + X1, Y0 + X2, Y0 + X3, Y1 + X0, Y1 + X1, Y1 + X2, Y1 + X3);
				const __m256i mlutidx1 = _mm256_setr_epi32(Y2 + X0, Y2 + X1, Y2 + X2, Y2 + X3, Y3 + X0, Y3 + X1, Y3 + X2, Y3 + X3);

				__m256 mb0, mg0, mr0;
				_mm256_i32gather_bgr_ps(srcptr, mlutidx0, mb0, mg0, mr0);
				__m256 mb1, mg1, mr1;
				_mm256_i32gather_bgr_ps(srcptr, mlutidx1, mb1, mg1, mr1);

				for (int n = 0; n < scale; n++)
				{
					uchar* destptr = dest.ptr<uchar>(y * scale + n, x * scale); // output
					float* weightmap_ptr = &weight[(scale * n) * 16];
					for (int m = 0; m < scale; m++)
					{
						const __m256 mw0 = _mm256_load_ps(weightmap_ptr);
						const __m256 mw1 = _mm256_load_ps(weightmap_ptr + 8);

						__m256 mv = _mm256_mul_ps(mw0, mb0);
						mv = _mm256_fmadd_ps(mw1, mb1, mv);
						destptr[0] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
						mv = _mm256_mul_ps(mw0, mg0);
						mv = _mm256_fmadd_ps(mw1, mg1, mv);
						destptr[1] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
						mv = _mm256_mul_ps(mw0, mr0);
						mv = _mm256_fmadd_ps(mw1, mr1, mv);
						destptr[2] = saturate_cast<uchar>(_mm256_reduceadd_ps(mv));
						weightmap_ptr += 16;
						destptr += 3;
					}
				}
			}
		}
		_mm_free(weight);
	}

	template <class Type>
	void upsampleCubic_(Mat& src, Mat& dest, const int scale, const double a)
	{
		const int dw = dest.cols / (src.cols - 1);
		const int dh = dest.rows / (src.rows - 1);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		vector<vector<double>> weight(dh * dw);
		for (int i = 0; i < weight.size(); i++)weight[i].resize(16);

		int idx = 0;

		for (int l = 0; l < dh; l++)
		{
			const double y = (double)l / (double)dh;
			for (int k = 0; k < dw; k++)
			{
				const double x = (double)k / (double)dw;

				weight[idx][0] = cp::cubic(1.0 + x, a) * cp::cubic(1.0 + y, a);
				weight[idx][1] = cp::cubic(0.0 + x, a) * cp::cubic(1.0 + y, a);
				weight[idx][2] = cp::cubic(1.0 - x, a) * cp::cubic(1.0 + y, a);
				weight[idx][3] = cp::cubic(2.0 - x, a) * cp::cubic(1.0 + y, a);

				weight[idx][4] = cp::cubic(1.0 + x, a) * cp::cubic(0.0 + y, a);
				weight[idx][5] = cp::cubic(0.0 + x, a) * cp::cubic(0.0 + y, a);
				weight[idx][6] = cp::cubic(1.0 - x, a) * cp::cubic(0.0 + y, a);
				weight[idx][7] = cp::cubic(2.0 - x, a) * cp::cubic(0.0 + y, a);

				weight[idx][8] = cp::cubic(1.0 + x, a) * cp::cubic(1.0 - y, a);
				weight[idx][9] = cp::cubic(0.0 + x, a) * cp::cubic(1.0 - y, a);
				weight[idx][10] = cp::cubic(1.0 - x, a) * cp::cubic(1.0 - y, a);
				weight[idx][11] = cp::cubic(2.0 - x, a) * cp::cubic(1.0 - y, a);

				weight[idx][12] = cp::cubic(1.0 + x, a) * cp::cubic(2.0 - y, a);
				weight[idx][13] = cp::cubic(0.0 + x, a) * cp::cubic(2.0 - y, a);
				weight[idx][14] = cp::cubic(1.0 - x, a) * cp::cubic(2.0 - y, a);
				weight[idx][15] = cp::cubic(2.0 - x, a) * cp::cubic(2.0 - y, a);

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

				weight[idx][0] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][1] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][2] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][3] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][4] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][5] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][6] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][7] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][8] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][9] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][10] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][11] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][12] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(2.0 - y, a)));
				weight[idx][13] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(2.0 - y, a)));
				weight[idx][14] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(2.0 - y, a)));
				weight[idx][15] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(2.0 - y, a)));

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
			v01 = *(__m256*) & s0[sx1];
			v02 = *(__m256*) & s0[sx2];
			v03 = *(__m256*) & s0[sx3];
			v10 = _mm256_set_ps(s1[6], s1[5], s1[4], s1[3], s1[2], s1[1], s1[0], s1[0]);
			v11 = *(__m256*) & s1[sx1];
			v12 = *(__m256*) & s1[sx2];
			v13 = *(__m256*) & s1[sx3];
			v20 = _mm256_set_ps(s2[6], s2[5], s2[4], s2[3], s2[2], s2[1], s2[0], s2[0]);
			v21 = *(__m256*) & s2[sx1];
			v22 = *(__m256*) & s2[sx2];
			v23 = *(__m256*) & s2[sx3];
			v30 = _mm256_set_ps(s3[6], s3[5], s3[4], s3[3], s3[2], s3[1], s3[0], s3[0]);
			v31 = *(__m256*) & s3[sx1];
			v32 = *(__m256*) & s3[sx2];
			v33 = *(__m256*) & s3[sx3];

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
						*(__m256*)& d[dx + 0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
						*(__m256*)& d[dx + 8] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);
					}
				}

				sx += 8;

				int sx0 = sx - 1;
				int sx1 = sx + 0;
				int sx2 = sx + 1;
				int sx3 = sx + 2;

				v00 = *(__m256*) & s0[sx0];
				v01 = *(__m256*) & s0[sx1];
				v02 = *(__m256*) & s0[sx2];
				v03 = *(__m256*) & s0[sx3];
				v10 = *(__m256*) & s1[sx0];
				v11 = *(__m256*) & s1[sx1];
				v12 = *(__m256*) & s1[sx2];
				v13 = *(__m256*) & s1[sx3];
				v20 = *(__m256*) & s2[sx0];
				v21 = *(__m256*) & s2[sx1];
				v22 = *(__m256*) & s2[sx2];
				v23 = *(__m256*) & s2[sx3];
				v30 = *(__m256*) & s3[sx0];
				v31 = *(__m256*) & s3[sx1];
				v32 = *(__m256*) & s3[sx2];
				v33 = *(__m256*) & s3[sx3];
			}
			{
				for (int sx = int(src.cols / 8 - 1) * 8, dx; sx < src.cols; sx += 8)
				{
					dx = amp * sx;

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

							int iend = min(16, dest.cols - amp * sx);
							if (iend == 16)
							{
								*(__m256*)& d[dx + 0] = tmpA[0];
								*(__m256*)& d[dx + 8] = tmpA[1];
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

	static void upsample32fCubicScale4(Mat& src, Mat& dest, double a)
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

				weight[idx][0] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][1] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][2] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][3] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 + y, a)));
				weight[idx][4] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][5] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][6] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][7] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(0.0 + y, a)));
				weight[idx][8] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][9] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][10] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][11] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(1.0 - y, a)));
				weight[idx][12] = _mm256_set1_ps(float(cp::cubic(1.0 + x, a) * cp::cubic(2.0 - y, a)));
				weight[idx][13] = _mm256_set1_ps(float(cp::cubic(0.0 + x, a) * cp::cubic(2.0 - y, a)));
				weight[idx][14] = _mm256_set1_ps(float(cp::cubic(1.0 - x, a) * cp::cubic(2.0 - y, a)));
				weight[idx][15] = _mm256_set1_ps(float(cp::cubic(2.0 - x, a) * cp::cubic(2.0 - y, a)));

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
			v01 = *(__m256*) & s0[sx1];
			v02 = *(__m256*) & s0[sx2];
			v03 = *(__m256*) & s0[sx3];
			v10 = _mm256_set_ps(s1[6], s1[5], s1[4], s1[3], s1[2], s1[1], s1[0], s1[0]);
			v11 = *(__m256*) & s1[sx1];
			v12 = *(__m256*) & s1[sx2];
			v13 = *(__m256*) & s1[sx3];
			v20 = _mm256_set_ps(s2[6], s2[5], s2[4], s2[3], s2[2], s2[1], s2[0], s2[0]);
			v21 = *(__m256*) & s2[sx1];
			v22 = *(__m256*) & s2[sx2];
			v23 = *(__m256*) & s2[sx3];
			v30 = _mm256_set_ps(s3[6], s3[5], s3[4], s3[3], s3[2], s3[1], s3[0], s3[0]);
			v31 = *(__m256*) & s3[sx1];
			v32 = *(__m256*) & s3[sx2];
			v33 = *(__m256*) & s3[sx3];

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

						*(__m256*)& d[dx + 0] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00100000);
						*(__m256*)& d[dx + 8] = _mm256_permute2f128_ps(tmpB[0], tmpB[1], 0b00110001);
						*(__m256*)& d[dx + 16] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00100000);
						*(__m256*)& d[dx + 24] = _mm256_permute2f128_ps(tmpB[2], tmpB[3], 0b00110001);
					}
				}

				sx += 8;

				int sx0 = sx - 1;
				int sx1 = sx + 0;
				int sx2 = sx + 1;
				int sx3 = sx + 2;

				v00 = *(__m256*) & s0[sx0];
				v01 = *(__m256*) & s0[sx1];
				v02 = *(__m256*) & s0[sx2];
				v03 = *(__m256*) & s0[sx3];
				v10 = *(__m256*) & s1[sx0];
				v11 = *(__m256*) & s1[sx1];
				v12 = *(__m256*) & s1[sx2];
				v13 = *(__m256*) & s1[sx3];
				v20 = *(__m256*) & s2[sx0];
				v21 = *(__m256*) & s2[sx1];
				v22 = *(__m256*) & s2[sx2];
				v23 = *(__m256*) & s2[sx3];
				v30 = *(__m256*) & s3[sx0];
				v31 = *(__m256*) & s3[sx1];
				v32 = *(__m256*) & s3[sx2];
				v33 = *(__m256*) & s3[sx3];
			}
			{
				for (int sx = int(src.cols / 8 - 1) * 8, dx; sx < src.cols; sx += 8)
				{
					dx = amp * sx;

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

							int iend = min(32, dest.cols - amp * sx);
							if (iend == 32)
							{
								*(__m256*)& d[dx + 0] = tmpA[0];
								*(__m256*)& d[dx + 8] = tmpA[1];
								*(__m256*)& d[dx + 16] = tmpA[2];
								*(__m256*)& d[dx + 24] = tmpA[3];
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

	static void upsample64fCubicScale2(Mat& src, Mat& dest, double a)
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

				weight[idx][0] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx][1] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx][2] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(1.0 + y, a));
				weight[idx][3] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(1.0 + y, a));
				weight[idx][4] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx][5] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx][6] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(0.0 + y, a));
				weight[idx][7] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(0.0 + y, a));
				weight[idx][8] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx][9] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx][10] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(1.0 - y, a));
				weight[idx][11] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(1.0 - y, a));
				weight[idx][12] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx][13] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx][14] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(2.0 - y, a));
				weight[idx][15] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(2.0 - y, a));

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
			v01 = *(__m256d*) & s0[sx1];
			v02 = *(__m256d*) & s0[sx2];
			v03 = *(__m256d*) & s0[sx3];
			v10 = _mm256_set_pd(s1[2], s1[1], s1[0], s1[0]);
			v11 = *(__m256d*) & s1[sx1];
			v12 = *(__m256d*) & s1[sx2];
			v13 = *(__m256d*) & s1[sx3];
			v20 = _mm256_set_pd(s2[2], s2[1], s2[0], s2[0]);
			v21 = *(__m256d*) & s2[sx1];
			v22 = *(__m256d*) & s2[sx2];
			v23 = *(__m256d*) & s2[sx3];
			v30 = _mm256_set_pd(s3[2], s3[1], s3[0], s3[0]);
			v31 = *(__m256d*) & s3[sx1];
			v32 = *(__m256d*) & s3[sx2];
			v33 = *(__m256d*) & s3[sx3];

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
						*(__m256d*)& d[dx + 0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
						*(__m256d*)& d[dx + 4] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);
					}
				}

				sx += 4;

				int sx0 = sx - 1;
				int sx1 = sx + 0;
				int sx2 = sx + 1;
				int sx3 = sx + 2;

				v00 = *(__m256d*) & s0[sx0];
				v01 = *(__m256d*) & s0[sx1];
				v02 = *(__m256d*) & s0[sx2];
				v03 = *(__m256d*) & s0[sx3];
				v10 = *(__m256d*) & s1[sx0];
				v11 = *(__m256d*) & s1[sx1];
				v12 = *(__m256d*) & s1[sx2];
				v13 = *(__m256d*) & s1[sx3];
				v20 = *(__m256d*) & s2[sx0];
				v21 = *(__m256d*) & s2[sx1];
				v22 = *(__m256d*) & s2[sx2];
				v23 = *(__m256d*) & s2[sx3];
				v30 = *(__m256d*) & s3[sx0];
				v31 = *(__m256d*) & s3[sx1];
				v32 = *(__m256d*) & s3[sx2];
				v33 = *(__m256d*) & s3[sx3];
			}
			{
				for (int sx = int(src.cols / 4 - 1) * 4, dx; sx < src.cols; sx += 4)
				{
					dx = amp * sx;

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

							int iend = min(8, dest.cols - amp * sx);
							if (iend == 8)
							{
								*(__m256d*)& d[dx + 0] = tmpA[0];
								*(__m256d*)& d[dx + 4] = tmpA[1];
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

	static void upsample64fCubicScale4(Mat& src, Mat& dest, double a)
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

				weight[idx][0] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx][1] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(1.0 + y, a));
				weight[idx][2] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(1.0 + y, a));
				weight[idx][3] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(1.0 + y, a));
				weight[idx][4] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx][5] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(0.0 + y, a));
				weight[idx][6] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(0.0 + y, a));
				weight[idx][7] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(0.0 + y, a));
				weight[idx][8] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx][9] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(1.0 - y, a));
				weight[idx][10] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(1.0 - y, a));
				weight[idx][11] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(1.0 - y, a));
				weight[idx][12] = _mm256_set1_pd(cp::cubic(1.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx][13] = _mm256_set1_pd(cp::cubic(0.0 + x, a) * cp::cubic(2.0 - y, a));
				weight[idx][14] = _mm256_set1_pd(cp::cubic(1.0 - x, a) * cp::cubic(2.0 - y, a));
				weight[idx][15] = _mm256_set1_pd(cp::cubic(2.0 - x, a) * cp::cubic(2.0 - y, a));

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
			v01 = *(__m256d*) & s0[sx1];
			v02 = *(__m256d*) & s0[sx2];
			v03 = *(__m256d*) & s0[sx3];
			v10 = _mm256_set_pd(s1[2], s1[1], s1[0], s1[0]);
			v11 = *(__m256d*) & s1[sx1];
			v12 = *(__m256d*) & s1[sx2];
			v13 = *(__m256d*) & s1[sx3];
			v20 = _mm256_set_pd(s2[2], s2[1], s2[0], s2[0]);
			v21 = *(__m256d*) & s2[sx1];
			v22 = *(__m256d*) & s2[sx2];
			v23 = *(__m256d*) & s2[sx3];
			v30 = _mm256_set_pd(s3[2], s3[1], s3[0], s3[0]);
			v31 = *(__m256d*) & s3[sx1];
			v32 = *(__m256d*) & s3[sx2];
			v33 = *(__m256d*) & s3[sx3];

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

						*(__m256d*)& d[dx + 0] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00100000);
						*(__m256d*)& d[dx + 4] = _mm256_permute2f128_pd(tmpB[0], tmpB[1], 0b00110001);
						*(__m256d*)& d[dx + 8] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00100000);
						*(__m256d*)& d[dx + 12] = _mm256_permute2f128_pd(tmpB[2], tmpB[3], 0b00110001);
					}
				}

				sx += 4;

				int sx0 = sx - 1;
				int sx1 = sx + 0;
				int sx2 = sx + 1;
				int sx3 = sx + 2;

				v00 = *(__m256d*) & s0[sx0];
				v01 = *(__m256d*) & s0[sx1];
				v02 = *(__m256d*) & s0[sx2];
				v03 = *(__m256d*) & s0[sx3];
				v10 = *(__m256d*) & s1[sx0];
				v11 = *(__m256d*) & s1[sx1];
				v12 = *(__m256d*) & s1[sx2];
				v13 = *(__m256d*) & s1[sx3];
				v20 = *(__m256d*) & s2[sx0];
				v21 = *(__m256d*) & s2[sx1];
				v22 = *(__m256d*) & s2[sx2];
				v23 = *(__m256d*) & s2[sx3];
				v30 = *(__m256d*) & s3[sx0];
				v31 = *(__m256d*) & s3[sx1];
				v32 = *(__m256d*) & s3[sx2];
				v33 = *(__m256d*) & s3[sx3];
			}
			{
				for (int sx = int(src.cols / 4 - 1) * 4, dx; sx < src.cols; sx += 4)
				{
					dx = amp * sx;

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

							int iend = min(16, dest.cols - amp * sx);
							if (iend == 16)
							{
								*(__m256d*)& d[dx + 0] = tmpA[0];
								*(__m256d*)& d[dx + 4] = tmpA[1];
								*(__m256d*)& d[dx + 8] = tmpA[2];
								*(__m256d*)& d[dx + 12] = tmpA[3];
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

	static void upsampleCubicGray(InputArray src_, OutputArray dest_, const int scale, const double a)
	{
		if (scale == 1)
		{
			src_.copyTo(dest_);
			return;
		}

		Mat src = src_.getMat();

		if (dest_.empty() || dest_.size() != Size(src.cols * scale, src.rows * scale))
		{
			dest_.create(Size(src.cols * scale, src.rows * scale), src_.type());
		}

		Mat dest = dest_.getMat();

		if (src.depth() == CV_8U)
		{
			upsampleConv16_8UC1(src, dest, scale, a);
			//upsampleCubic_<uchar>(src, dest, scale, a);
		}
		else if (src.depth() == CV_16S) upsampleCubic_<short>(src, dest, scale, a);
		else if (src.depth() == CV_16U) upsampleCubic_<ushort>(src, dest, scale, a);
		else if (src.depth() == CV_32S) upsampleCubic_<int>(src, dest, scale, a);
		else if (src.depth() == CV_32F)
		{
			if (scale == 2)
				upsample32fCubicScale2(src, dest, a);
			else if (scale == 4)
				upsample32fCubicScale4(src, dest, a);
			else
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
		if (src_.type() == CV_8UC1)
		{
			Mat d = dest_.getMat();
			upsampleConv16_8UC1(src_.getMat(), d, scale, a);
		}
		else if (src_.type() == CV_8UC3)
		{
			Mat d = dest_.getMat();
			upsampleConv16_8UC3(src_.getMat(), d, scale, a);
		}
		else
		{
			if (src_.channels() == 1)
			{
				upsampleCubicGray(src_, dest_, scale, a);
			}
			else
			{
				vector<Mat> v;
				split(src_, v);
				vector<Mat> d(v.size());
				for (int i = 0; i < v.size(); i++)
				{
					upsampleCubicGray(v[i], d[i], scale, a);
				}
				merge(d, dest_);
			}
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
						uchar* dest_ptr = (uchar*)dest->ptr<uchar>(y + n); // output
						for (int m = 0; m < scale; m++)
						{
							int idx = n * scale + m;
							const float* weightmap_ptr = weightmap->ptr<float>(idx);
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
							float* dest_ptr = dest->ptr<float>(y + n); // output
							for (int m = 0; m < scale; m++)
							{
								int idx = n * scale + m;
								const float* weightmap_ptr = weightmap->ptr<float>(idx);
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
							float* dest_ptr = dest->ptr<float>(y + n); // output
							for (int m = 0; m < scale; m++)
							{
								int idx = n * scale + m;
								const float* weightmap_ptr = weightmap->ptr<float>(idx);
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
		dest.create(src.size() * scale, src.type());

		Mat weightmap(scale * scale, 16, CV_32F);
		cp::setCubicWeight4x4(weightmap, float(a));
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
#pragma endregion

#pragma region weighted cubic
	template<typename T>
	void upsampleWeightedCubic(Mat& src, Mat& guide, Mat& dest, const int scale, const double a)
	{
		Mat weightmap(scale * scale, 16, CV_32F);

		Mat sguide;
		//resize(guide, sguide, src.size(), 0, 0, INTER_NEAREST);
		cp::downsample(guide, sguide, scale, cp::Downsample(INTER_NEAREST), 4);

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


		float n = float(nrm * 0.1);
		for (int i = 0; i < 256; i++)
		{
			rweight[i] = float(exp(pow(i, n) / (-1.0 / n * pow(sr, n))));
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
		cp::setGaussianWeight4x4(weightmap, (float)ss);

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
					uchar* guide_ptr = guide.ptr<uchar>(y + n); // reference
					T* dest_ptr = dest.ptr<T>(y + n); // output

					for (int m = 0; m < scale; m++)
					{
						int idx = n * scale + m;
						float* weightmap_ptr = weightmap.ptr<float>(idx);
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
		if (dest_.empty() || dest_.size() != Size(src.cols * scale, src.rows * scale))
		{
			dest_.create(Size(src.cols * scale, src.rows * scale), src_.type());
		}

		Mat dest = dest_.getMat();

		upsampleWeightedCubic<uchar>(src, guide, dest, scale, a);
	}
#pragma endregion

	void setUpsampleMask(InputArray src, OutputArray dst)
	{
		Mat dest = dst.getMat();
		if (dest.empty())
		{
			cout << "please alloc dest Mat" << endl;
			return;
		}
		dest.setTo(0);
		const int dw = dest.cols / (src.size().width);
		const int dh = dest.rows / (src.size().height);

		for (int j = 0; j < src.size().height; j++)
		{
			int n = j * dh;
			uchar* d = dest.ptr<uchar>(n);
			for (int i = 0, m = 0; i < src.size().width; i++, m += dw)
			{
				d[m] = 255;
			}
		}
	}

	void resizeShift(InputArray src_, OutputArray dest_, const int scale, const int resize_method, const double shiftx, const double shifty)
	{
		Mat src = src_.getMat();

		Mat a = cp::convert(src, CV_32F);
		Mat b;
		resize(a, b, Size(), scale, scale, resize_method);

		cp::warpShiftSubpix(b, b, shiftx, shifty, resize_method);

		b.convertTo(dest_, src_.depth());
	}
}