
#include "jointBilateralUpsample.hpp"
#include "downsample.hpp"
#include "inlineSimdFunctions.hpp"
#include "fmath/fmath.hpp"

using namespace std;
using namespace cv;

#define USE_SIMD_JBU
namespace cp
{
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
			int n = j*dh;
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

	inline int linearinterpolate_(int lt, int rt, int lb, int rb, double a, double b)
	{
		return (int)((b*a*lt + b*(1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb) + 0.5);
	}

	template <class srcType>
	inline double linearinterpolate_(srcType lt, srcType rt, srcType lb, srcType rb, double a, double b)
	{
		return (b*a*lt + b*(1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb);
	}

	template <class srcType>
	static void linearUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols - 1);
		const int dh = dest.rows / (src.rows - 1);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		Mat sim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, BORDER_REPLICATE);
		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;
			srcType* s = sim.ptr<srcType>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType ltd = s[i];
				const srcType rtd = s[i + 1];
				const srcType lbd = s[i + sim.cols];
				const srcType rbd = s[i + 1 + sim.cols];
				for (int l = 0; l < dh; l++)
				{
					double beta = 1.0 - (double)l / dh;
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;
						d[m + k] = saturate_cast<srcType> (linearinterpolate_<srcType>(ltd, rtd, lbd, rbd, alpha, beta));
					}
				}
			}
		}
	}

	void linearUpsample(InputArray src_, OutputArray dest_)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) linearUpsample_<uchar>(src, dest);
		else if (src.depth() == CV_16S) linearUpsample_<short>(src, dest);
		else if (src.depth() == CV_16U) linearUpsample_<ushort>(src, dest);
		else if (src.depth() == CV_32S) linearUpsample_<int>(src, dest);
		else if (src.depth() == CV_32F) linearUpsample_<float>(src, dest);
		else if (src.depth() == CV_64F) linearUpsample_<double>(src, dest);
	}

	inline double cubicfunc(double x, double a = -1.0)
	{
		double X = abs(x);
		if (X <= 1)
			return ((a + 2.0)*x*x*x - (a + 3.0)*x*x + 1.0);
		else if (X <= 2)
			return (a*x*x*x - 5.0*a*x*x + 8.0*a*x - 4.0*a);
		else
			return 0.0;
	}

	template <class srcType>
	static void cubicUpsample_(Mat& src, Mat& dest, double a)
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

				weight[idx][0] = cubicfunc(1.0 + x, a)*cubicfunc(1.0 + y, a);
				weight[idx][1] = cubicfunc(0.0 + x, a)*cubicfunc(1.0 + y, a);
				weight[idx][2] = cubicfunc(1.0 - x, a)*cubicfunc(1.0 + y, a);
				weight[idx][3] = cubicfunc(2.0 - x, a)*cubicfunc(1.0 + y, a);

				weight[idx][4] = cubicfunc(1.0 + x, a)*cubicfunc(0.0 + y, a);
				weight[idx][5] = cubicfunc(0.0 + x, a)*cubicfunc(0.0 + y, a);
				weight[idx][6] = cubicfunc(1.0 - x, a)*cubicfunc(0.0 + y, a);
				weight[idx][7] = cubicfunc(2.0 - x, a)*cubicfunc(0.0 + y, a);

				weight[idx][8] = cubicfunc(1.0 + x, a)*cubicfunc(1.0 - y, a);
				weight[idx][9] = cubicfunc(0.0 + x, a)*cubicfunc(1.0 - y, a);
				weight[idx][10] = cubicfunc(1.0 - x, a)*cubicfunc(1.0 - y, a);
				weight[idx][11] = cubicfunc(2.0 - x, a)*cubicfunc(1.0 - y, a);

				weight[idx][12] = cubicfunc(1.0 + x, a)*cubicfunc(2.0 - y, a);
				weight[idx][13] = cubicfunc(0.0 + x, a)*cubicfunc(2.0 - y, a);
				weight[idx][14] = cubicfunc(1.0 - x, a)*cubicfunc(2.0 - y, a);
				weight[idx][15] = cubicfunc(2.0 - x, a)*cubicfunc(2.0 - y, a);

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
			int n = j*dh;
			srcType* s = sim.ptr<srcType>(j + 1) + 1;

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType v00 = s[i - 1 - sim.cols];
				const srcType v01 = s[i - 0 - sim.cols];
				const srcType v02 = s[i + 1 - sim.cols];
				const srcType v03 = s[i + 2 - sim.cols];
				const srcType v10 = s[i - 1];
				const srcType v11 = s[i - 0];
				const srcType v12 = s[i + 1];
				const srcType v13 = s[i + 2];
				const srcType v20 = s[i - 1 + sim.cols];
				const srcType v21 = s[i - 0 + sim.cols];
				const srcType v22 = s[i + 1 + sim.cols];
				const srcType v23 = s[i + 2 + sim.cols];
				const srcType v30 = s[i - 1 + 2 * sim.cols];
				const srcType v31 = s[i - 0 + 2 * sim.cols];
				const srcType v32 = s[i + 1 + 2 * sim.cols];
				const srcType v33 = s[i + 2 + 2 * sim.cols];

				int idx = 0;
				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						
						d[m + k] = saturate_cast<srcType>(
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

	void cubicUpsample(InputArray src_, OutputArray dest_, double a)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) cubicUpsample_<uchar>(src, dest, a);
		else if (src.depth() == CV_16S) cubicUpsample_<short>(src, dest, a);
		else if (src.depth() == CV_16U) cubicUpsample_<ushort>(src, dest, a);
		else if (src.depth() == CV_32S) cubicUpsample_<int>(src, dest, a);
		else if (src.depth() == CV_32F) cubicUpsample_<float>(src, dest, a);
		else if (src.depth() == CV_64F) cubicUpsample_<double>(src, dest, a);
	}

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
			int n = j*dh;
			uchar* d = dest.ptr<uchar>(n);
			for (int i = 0, m = 0; i < src.size().width; i++, m += dw)
			{
				d[m] = 255;
			}
		}
	}

#pragma region function JointBilateralUpsample
	static void jointBilateralUpsamplingAllocBorder(Mat& src, Mat& guide, Mat& dest, const int r, const double sigma_r, const double sigma_s)
	{
		const int scale = guide.cols / src.cols;
		int border = BORDER_REPLICATE;
		Mat simb;  copyMakeBorder(src, simb, r, r, r, r, border);

		const int d = 2 * r + 1;
		const int ksize = d * d;
		Mat weightmap(scale * scale, d * d, CV_32F);

		Mat sguide;
		//resize(guide, sguide, src.size(), 0, 0, INTER_NEAREST);
		cp::downsample(guide, sguide, scale, cp::Downsample(INTER_NEAREST), 4);
		Mat gimb; copyMakeBorder(sguide, gimb, r, r, r, r, border);

		float* range_weight = (float*)_mm_malloc(256 * sizeof(float), AVX_ALIGN);

#ifdef LP_NORM
		//lp norm
		float n = 2.f;
		for (int i = 0; i < 256; i++)
		{
			rweight[i] = exp(pow(i, n) / (-n * pow(sigma_r, n)));
		}
#else
		const float rcoeff = float(1.0 / (-2.0 * sigma_r * sigma_r));
		for (int i = 0; i < 256; i++)
		{
			range_weight[i] = fmath::exp(i * i * rcoeff);
		}
#endif

		//compute spatial weight
		const float spaceCoeff = float(1.0 / (-2.0 * sigma_s * sigma_s * scale * scale));
		for (int j = 0; j < scale; j++)
		{
			for (int i = 0; i < scale; i++)
			{
				float* wmap = weightmap.ptr<float>(j * scale + i);
				int count = 0;
				float wsum = 0.f;
				for (int n = 0; n < d; n++)
				{
					for (int m = 0; m < d; m++)
					{
						float xf = abs(m - r) + (float)i / scale;
						float yf = abs(n - r) + (float)j / scale;
						float dist = hypot(xf, yf);
						float w = fmath::exp(dist * dist * spaceCoeff);
						wsum += w;
						wmap[count++] = w;
					}
				}
				for (int n = 0; n < d * d; n++)
				{
					wmap[n] /= wsum;
				}
			}
		}

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < dest.rows; y += scale)
		{
			const int y0 = (int)(y / scale);

			float* neighbor__b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);

			const int CV_DECL_ALIGNED(AVX_ALIGN) v32f_absmask_[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			const __m256 v32f_absmask = _mm256_load_ps((float*)v32f_absmask_);

			for (int x = 0; x < dest.cols; x += scale)
			{
				const int x0 = (int)(x / scale);
				int count = 0;
				for (int n = 0; n < d; n++)
				{
					uchar* sb = simb.ptr<uchar>(y0 + n);
					uchar* gb = gimb.ptr<uchar>(y0 + n);
					for (int m = 0; m < d; m++)
					{
						const int index = 3 * (x0 + m);
						neighbor__b[count] = (float)sb[index + 0];
						gneighbor_b[count] = (float)gb[index + 0];
						neighbor__g[count] = (float)sb[index + 1];
						gneighbor_g[count] = (float)gb[index + 1];
						neighbor__r[count] = (float)sb[index + 2];
						gneighbor_r[count] = (float)gb[index + 2];
						count++;
					}
				}

				for (int n = 0; n < scale; n++)
				{
					uchar* guide_ptr = guide.ptr<uchar>(y + n); // reference
					uchar* dest_ptr = dest.ptr<uchar>(y + n); // output
					for (int m = 0; m < scale; m++)
					{
						int idx = n * scale + m;
						float* weightmap_ptr = weightmap.ptr<float>(idx);

						const int index = 3 * (x + m);
						const uchar gb = guide_ptr[index + 0];
						const uchar gg = guide_ptr[index + 1];
						const uchar gr = guide_ptr[index + 2];

						float vb = 0.f;
						float wb = 0.f;
						float vg = 0.f;
						float wg = 0.f;
						float vr = 0.f;
						float wr = 0.f;

#ifdef USE_SIMD_JBU
						__m256 mgb = _mm256_set1_ps(gb);
						__m256 mgg = _mm256_set1_ps(gg);
						__m256 mgr = _mm256_set1_ps(gr);

						__m256 mvb = _mm256_setzero_ps();
						__m256 mwb = _mm256_setzero_ps();
						__m256 mvg = _mm256_setzero_ps();
						__m256 mwg = _mm256_setzero_ps();
						__m256 mvr = _mm256_setzero_ps();
						__m256 mwr = _mm256_setzero_ps();

						const int simdend = ksize / 8 * 8;// nxn kernel(n: odd value) must be 8*m+1
						for (int k = 0; k < simdend; k += 8)
						{
							__m256 mw = _mm256_load_ps(weightmap_ptr + k);

							__m256 mg = _mm256_load_ps(gneighbor_b + k);
							__m256 ms = _mm256_load_ps(neighbor__b + k);
							__m256i midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgb, mg), v32f_absmask));
							__m256 mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvb = _mm256_fmadd_ps(mrw, ms, mvb);
							mwb = _mm256_add_ps(mrw, mwb);

							mg = _mm256_load_ps(gneighbor_g + k);
							ms = _mm256_load_ps(neighbor__g + k);
							midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgg, mg), v32f_absmask));
							mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvg = _mm256_fmadd_ps(mrw, ms, mvg);
							mwg = _mm256_add_ps(mrw, mwg);

							mg = _mm256_load_ps(gneighbor_r + k);
							ms = _mm256_load_ps(neighbor__r + k);
							midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgr, mg), v32f_absmask));
							mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvr = _mm256_fmadd_ps(mrw, ms, mvr);
							mwr = _mm256_add_ps(mrw, mwr);
						}
						vb = _mm256_reduceadd_ps(mvb);
						vg = _mm256_reduceadd_ps(mvg);
						vr = _mm256_reduceadd_ps(mvr);
						wb = _mm256_reduceadd_ps(mwb);
						wg = _mm256_reduceadd_ps(mwg);
						wr = _mm256_reduceadd_ps(mwr);

						int ed = ksize - 1;
						float cwb = range_weight[abs(gb - (int)gneighbor_b[ed])] * weightmap_ptr[ed];
						vb += cwb * neighbor__b[ed];
						wb += cwb;

						float cwg = range_weight[abs(gg - (int)gneighbor_g[ed])] * weightmap_ptr[ed];
						vg += cwg * neighbor__g[ed];
						wg += cwg;

						float cwr = range_weight[abs(gr - (int)gneighbor_r[ed])] * weightmap_ptr[ed];
						vr += cwr * neighbor__r[ed];
						wr += cwr;
#else
						for (int k = 0; k < d * d; k++)
						{
							float cwb = rweight[abs(gb - (int)gneighbor_b[k])] * weightmap_ptr[k];
							vb += cwb * neighbor__b[k];
							wb += cwb;

							float cwg = rweight[abs(gg - (int)gneighbor_g[k])] * weightmap_ptr[k];
							vg += cwg * neighbor__g[k];
							wg += cwg;

							float cwr = rweight[abs(gr - (int)gneighbor_r[k])] * weightmap_ptr[k];
							vr += cwr * neighbor__r[k];
							wr += cwr;
						}
#endif
						dest_ptr[index + 0] = saturate_cast<uchar>(vb / wb);
						dest_ptr[index + 1] = saturate_cast<uchar>(vg / wg);
						dest_ptr[index + 2] = saturate_cast<uchar>(vr / wr);
					}
				}
			}

			_mm_free(neighbor__b);
			_mm_free(neighbor__g);
			_mm_free(neighbor__r);
			_mm_free(gneighbor_b);
			_mm_free(gneighbor_g);
			_mm_free(gneighbor_r);
		}

		_mm_free(range_weight);
	}

	static void jointBilateralUpsamplingComputeBorder(Mat& src, Mat& guide, Mat& dest, const int r, const double sigma_r, const double sigma_s)
	{
		const int scale = guide.cols / src.cols;
		int border = BORDER_REPLICATE;


		const int d = 2 * r + 1;
		const int ksize = d * d;
		Mat weightmap(scale * scale, ksize, CV_32F);

		Mat sguide;
		//resize(guide, gim, src.size(), 0, 0, INTER_NEAREST);
		cp::downsample(guide, sguide, scale, cp::Downsample(INTER_NEAREST), 4);

		float* range_weight = (float*)_mm_malloc(256 * sizeof(float), AVX_ALIGN);

#ifdef LP_NORM
		//lp norm
		float n = 2.f;
		for (int i = 0; i < 256; i++)
		{
			rweight[i] = exp(pow(i, n) / (-n * pow(sigma_r, n)));
		}
#else
		const float rcoeff = float(1.0 / (-2.0 * sigma_r * sigma_r));
		for (int i = 0; i < 256; i++)
		{
			range_weight[i] = fmath::exp(i * i * rcoeff);
		}
#endif

		const float spaceCoeff = float(1.0 / (-2.0 * sigma_s * sigma_s * scale * scale));
		for (int j = 0; j < scale; j++)
		{
			for (int i = 0; i < scale; i++)
			{
				float* wmap = weightmap.ptr<float>(j * scale + i);
				int count = 0;
				float wsum = 0.f;
				for (int n = 0; n < d; n++)
				{
					for (int m = 0; m < d; m++)
					{
						float xf = abs(m - r) + (float)i / scale;
						float yf = abs(n - r) + (float)j / scale;
						float dist = hypot(xf, yf);
						float w = fmath::exp(dist * dist * spaceCoeff);
						wsum += w;
						wmap[count++] = w;
					}
				}
				for (int n = 0; n < d * d; n++)
				{
					wmap[n] /= wsum;
				}
			}
		}

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < dest.rows; y += scale)
		{
			const int y0 = (int)(y / scale);

			float* neighbor__b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);

			const int CV_DECL_ALIGNED(32) v32f_absmask_[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			const __m256 v32f_absmask = _mm256_load_ps((float*)v32f_absmask_);

			for (int x = 0; x < dest.cols; x += scale)
			{
				const int x0 = (int)(x / scale);
				int count = 0;
				for (int n = -r; n <= r; n++)
				{
					const int Y = max(0, min(src.rows - 1, y0 + n));
					uchar* sb = src.ptr<uchar>(Y);
					uchar* gb = sguide.ptr<uchar>(Y);
					for (int m = -r; m <= r; m++)
					{
						const int index = 3 * max(0, min(src.cols - 1, x0 + m));

						neighbor__b[count] = (float)sb[index + 0];
						gneighbor_b[count] = (float)gb[index + 0];
						neighbor__g[count] = (float)sb[index + 1];
						gneighbor_g[count] = (float)gb[index + 1];
						neighbor__r[count] = (float)sb[index + 2];
						gneighbor_r[count] = (float)gb[index + 2];
						count++;
					}
				}

				for (int n = 0; n < scale; n++)
				{
					uchar* guide_ptr = guide.ptr<uchar>(y + n); // reference
					uchar* dest_ptr = dest.ptr<uchar>(y + n); // output
					for (int m = 0; m < scale; m++)
					{
						int idx = n * scale + m;
						float* weightmap_ptr = weightmap.ptr<float>(idx);

						const int index = 3 * (x + m);
						const uchar gb = guide_ptr[index + 0];
						const uchar gg = guide_ptr[index + 1];
						const uchar gr = guide_ptr[index + 2];

						float vb = 0.f;
						float wb = 0.f;
						float vg = 0.f;
						float wg = 0.f;
						float vr = 0.f;
						float wr = 0.f;

#ifdef USE_SIMD_JBU
						__m256 mgb = _mm256_set1_ps(gb);
						__m256 mgg = _mm256_set1_ps(gg);
						__m256 mgr = _mm256_set1_ps(gr);

						__m256 mvb = _mm256_setzero_ps();
						__m256 mwb = _mm256_setzero_ps();
						__m256 mvg = _mm256_setzero_ps();
						__m256 mwg = _mm256_setzero_ps();
						__m256 mvr = _mm256_setzero_ps();
						__m256 mwr = _mm256_setzero_ps();

						const int simdend = ksize / 8 * 8;// nxn kernel(n: odd value) must be 8*m+1
						for (int k = 0; k < simdend; k += 8)
						{
							__m256 mw = _mm256_load_ps(weightmap_ptr + k);

							__m256 mg = _mm256_load_ps(gneighbor_b + k);
							__m256 ms = _mm256_load_ps(neighbor__b + k);
							__m256i midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgb, mg), v32f_absmask));
							__m256 mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvb = _mm256_fmadd_ps(mrw, ms, mvb);
							mwb = _mm256_add_ps(mrw, mwb);

							mg = _mm256_load_ps(gneighbor_g + k);
							ms = _mm256_load_ps(neighbor__g + k);
							midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgg, mg), v32f_absmask));
							mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvg = _mm256_fmadd_ps(mrw, ms, mvg);
							mwg = _mm256_add_ps(mrw, mwg);

							mg = _mm256_load_ps(gneighbor_r + k);
							ms = _mm256_load_ps(neighbor__r + k);
							midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgr, mg), v32f_absmask));
							mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvr = _mm256_fmadd_ps(mrw, ms, mvr);
							mwr = _mm256_add_ps(mrw, mwr);
						}
						vb = _mm256_reduceadd_ps(mvb);
						vg = _mm256_reduceadd_ps(mvg);
						vr = _mm256_reduceadd_ps(mvr);
						wb = _mm256_reduceadd_ps(mwb);
						wg = _mm256_reduceadd_ps(mwg);
						wr = _mm256_reduceadd_ps(mwr);

						int ed = ksize - 1;
						float cwb = range_weight[abs(gb - (int)gneighbor_b[ed])] * weightmap_ptr[ed];
						vb += cwb * neighbor__b[ed];
						wb += cwb;

						float cwg = range_weight[abs(gg - (int)gneighbor_g[ed])] * weightmap_ptr[ed];
						vg += cwg * neighbor__g[ed];
						wg += cwg;

						float cwr = range_weight[abs(gr - (int)gneighbor_r[ed])] * weightmap_ptr[ed];
						vr += cwr * neighbor__r[ed];
						wr += cwr;
#else
						for (int k = 0; k < d * d; k++)
						{
							float cwb = rweight[abs(gb - (int)gneighbor_b[k])] * weightmap_ptr[k];
							vb += cwb * neighbor__b[k];
							wb += cwb;

							float cwg = rweight[abs(gg - (int)gneighbor_g[k])] * weightmap_ptr[k];
							vg += cwg * neighbor__g[k];
							wg += cwg;

							float cwr = rweight[abs(gr - (int)gneighbor_r[k])] * weightmap_ptr[k];
							vr += cwr * neighbor__r[k];
							wr += cwr;
						}
#endif
						dest_ptr[index + 0] = saturate_cast<uchar>(vb / wb);
						dest_ptr[index + 1] = saturate_cast<uchar>(vg / wg);
						dest_ptr[index + 2] = saturate_cast<uchar>(vr / wr);
					}
				}
			}

			_mm_free(neighbor__b);
			_mm_free(neighbor__g);
			_mm_free(neighbor__r);
			_mm_free(gneighbor_b);
			_mm_free(gneighbor_g);
			_mm_free(gneighbor_r);
		}

		_mm_free(range_weight);
	}

	static void jointBilateralUpsamplingComputeBorderNoGuideDownsample(Mat& src, Mat& guide, Mat& dest, const int r, const double sigma_r, const double sigma_s)
	{
		const int scale = guide.cols / src.cols;
		int border = BORDER_REPLICATE;


		const int d = 2 * r + 1;
		const int ksize = d * d;
		Mat weightmap(scale * scale, d * d, CV_32F);

		float* range_weight = (float*)_mm_malloc(256 * sizeof(float), AVX_ALIGN);

#ifdef LP_NORM
		//lp norm
		float n = 2.f;
		for (int i = 0; i < 256; i++)
		{
			rweight[i] = exp(pow(i, n) / (-n * pow(sigma_r, n)));
		}
#else
		const float rcoeff = float(1.0 / (-2.0 * sigma_r * sigma_r));
		for (int i = 0; i < 256; i++)
		{
			range_weight[i] = fmath::exp(i * i * rcoeff);
		}
#endif

		const float spaceCoeff = float(1.0 / (-2.0 * sigma_s * sigma_s * scale * scale));
		for (int j = 0; j < scale; j++)
		{
			for (int i = 0; i < scale; i++)
			{
				float* wmap = weightmap.ptr<float>(j * scale + i);
				int count = 0;
				float wsum = 0.f;
				for (int n = 0; n < d; n++)
				{
					for (int m = 0; m < d; m++)
					{
						float xf = abs(m - r) + (float)i / scale;
						float yf = abs(n - r) + (float)j / scale;
						float dist = hypot(xf, yf);
						float w = fmath::exp(dist * dist * spaceCoeff);
						wsum += w;
						wmap[count++] = w;
					}
				}
				for (int n = 0; n < d * d; n++)
				{
					wmap[n] /= wsum;
				}
			}
		}

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < dest.rows; y += scale)
		{
			const int y0 = (int)(y / scale);

			float* neighbor__b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);

			const int CV_DECL_ALIGNED(32) v32f_absmask_[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			const __m256 v32f_absmask = _mm256_load_ps((float*)v32f_absmask_);

			for (int x = 0; x < dest.cols; x += scale)
			{
				const int x0 = (int)(x / scale);
				int count = 0;
				for (int n = -r; n <= r; n++)
				{
					const int Y = max(0, min(src.rows - 1, y0 + n));
					const int GY = max(0, min(guide.rows - 1, y + n * scale));
					uchar* sb = src.ptr<uchar>(Y);
					uchar* gb = guide.ptr<uchar>(GY);
					for (int m = -r; m <= r; m++)
					{
						const int sindex = 3 * max(0, min(src.cols - 1, x0 + m));
						const int gindex = 3 * max(0, min(guide.cols - 1, x + m * scale));

						neighbor__b[count] = (float)sb[sindex + 0];
						gneighbor_b[count] = (float)gb[gindex + 0];
						neighbor__g[count] = (float)sb[sindex + 1];
						gneighbor_g[count] = (float)gb[gindex + 1];
						neighbor__r[count] = (float)sb[sindex + 2];
						gneighbor_r[count] = (float)gb[gindex + 2];
						count++;
					}
				}

				for (int n = 0; n < scale; n++)
				{
					uchar* guide_ptr = guide.ptr<uchar>(y + n); // reference
					uchar* dest_ptr = dest.ptr<uchar>(y + n); // output
					for (int m = 0; m < scale; m++)
					{
						int idx = n * scale + m;
						float* weightmap_ptr = weightmap.ptr<float>(idx);

						const int index = 3 * (x + m);
						const uchar gb = guide_ptr[index + 0];
						const uchar gg = guide_ptr[index + 1];
						const uchar gr = guide_ptr[index + 2];

						float vb = 0.f;
						float wb = 0.f;
						float vg = 0.f;
						float wg = 0.f;
						float vr = 0.f;
						float wr = 0.f;

#ifdef USE_SIMD_JBU
						__m256 mgb = _mm256_set1_ps(gb);
						__m256 mgg = _mm256_set1_ps(gg);
						__m256 mgr = _mm256_set1_ps(gr);

						__m256 mvb = _mm256_setzero_ps();
						__m256 mwb = _mm256_setzero_ps();
						__m256 mvg = _mm256_setzero_ps();
						__m256 mwg = _mm256_setzero_ps();
						__m256 mvr = _mm256_setzero_ps();
						__m256 mwr = _mm256_setzero_ps();

						const int simdend = ksize / 8 * 8;// nxn kernel(n: odd value) must be 8*m+1
						for (int k = 0; k < simdend; k += 8)
						{
							__m256 mw = _mm256_load_ps(weightmap_ptr + k);

							__m256 mg = _mm256_load_ps(gneighbor_b + k);
							__m256 ms = _mm256_load_ps(neighbor__b + k);
							__m256i midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgb, mg), v32f_absmask));
							__m256 mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvb = _mm256_fmadd_ps(mrw, ms, mvb);
							mwb = _mm256_add_ps(mrw, mwb);

							mg = _mm256_load_ps(gneighbor_g + k);
							ms = _mm256_load_ps(neighbor__g + k);
							midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgg, mg), v32f_absmask));
							mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvg = _mm256_fmadd_ps(mrw, ms, mvg);
							mwg = _mm256_add_ps(mrw, mwg);

							mg = _mm256_load_ps(gneighbor_r + k);
							ms = _mm256_load_ps(neighbor__r + k);
							midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgr, mg), v32f_absmask));
							mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
							mvr = _mm256_fmadd_ps(mrw, ms, mvr);
							mwr = _mm256_add_ps(mrw, mwr);
						}
						vb = _mm256_reduceadd_ps(mvb);
						vg = _mm256_reduceadd_ps(mvg);
						vr = _mm256_reduceadd_ps(mvr);
						wb = _mm256_reduceadd_ps(mwb);
						wg = _mm256_reduceadd_ps(mwg);
						wr = _mm256_reduceadd_ps(mwr);

						int ed = ksize - 1;
						float cwb = range_weight[abs(gb - (int)gneighbor_b[ed])] * weightmap_ptr[ed];
						vb += cwb * neighbor__b[ed];
						wb += cwb;

						float cwg = range_weight[abs(gg - (int)gneighbor_g[ed])] * weightmap_ptr[ed];
						vg += cwg * neighbor__g[ed];
						wg += cwg;

						float cwr = range_weight[abs(gr - (int)gneighbor_r[ed])] * weightmap_ptr[ed];
						vr += cwr * neighbor__r[ed];
						wr += cwr;
#else
						for (int k = 0; k < d * d; k++)
						{
							float cwb = rweight[abs(gb - (int)gneighbor_b[k])] * weightmap_ptr[k];
							vb += cwb * neighbor__b[k];
							wb += cwb;

							float cwg = rweight[abs(gg - (int)gneighbor_g[k])] * weightmap_ptr[k];
							vg += cwg * neighbor__g[k];
							wg += cwg;

							float cwr = rweight[abs(gr - (int)gneighbor_r[k])] * weightmap_ptr[k];
							vr += cwr * neighbor__r[k];
							wr += cwr;
						}
#endif
						dest_ptr[index + 0] = saturate_cast<uchar>(vb / wb);
						dest_ptr[index + 1] = saturate_cast<uchar>(vg / wg);
						dest_ptr[index + 2] = saturate_cast<uchar>(vr / wr);
					}
				}
			}

			_mm_free(neighbor__b);
			_mm_free(neighbor__g);
			_mm_free(neighbor__r);
			_mm_free(gneighbor_b);
			_mm_free(gneighbor_g);
			_mm_free(gneighbor_r);
		}

		_mm_free(range_weight);
	}

	void jointBilateralUpsampe(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int r, const double sigma_r, const double sigma_s, const JBUSchedule schedule)
	{
		switch (schedule)
		{

		case JBUSchedule::CLASS:
		default:
		{
			JointBilateralUpsample jbu;
			jbu.upsample(src, guide, dest, r, sigma_r, sigma_s);
		}
		break;

		case JBUSchedule::ALLOC_BORDER_OMP:jointBilateralUpsamplingAllocBorder(src.getMat(), guide.getMat(), dest.getMat(), r, sigma_r, sigma_s); break;
		case JBUSchedule::COMPUTE_BORDER_OMP:jointBilateralUpsamplingComputeBorder(src.getMat(), guide.getMat(), dest.getMat(), r, sigma_r, sigma_s); break;
		case JBUSchedule::COMPUTE_BORDER_NODOWNSAMPLE_OMP:jointBilateralUpsamplingComputeBorderNoGuideDownsample(src.getMat(), guide.getMat(), dest.getMat(), r, sigma_r, sigma_s); break;

		}
		return;
	}

#pragma endregion

#pragma region class JointBilateralUpsample
	template<typename T>
	class JointBilateralUpsampe32F_ParallelBody : public cv::ParallelLoopBody
	{
	private:
		const cv::Mat* slow_b;
		const cv::Mat* glow_b;
		const cv::Mat* guide;
		const cv::Mat* weightmap;
		cv::Mat* dest;

		const int scale;
		const int r;
		const float* range_weight;
	public:
		JointBilateralUpsampe32F_ParallelBody(const cv::Mat& src, const cv::Mat& guide_low, const cv::Mat& guide_high, const cv::Mat& weightmap, cv::Mat& dst, const int scale, const int r, const float* rweight)
			: slow_b(&src), glow_b(&guide_low), guide(&guide_high), weightmap(&weightmap), dest(&dst), scale(scale), r(r), range_weight(rweight)
		{
		}

		void operator() (const cv::Range& range) const
		{
			const int d = 2 * r + 1;
			const int ksize = d * d;

			float* neighbor__b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* neighbor__r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_b = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_g = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);
			float* gneighbor_r = (float*)_mm_malloc(sizeof(float) * ksize, AVX_ALIGN);

			for (int y = range.start; y < range.end; y += scale)
			{
				const int y0 = (int)(y / scale);
				const int CV_DECL_ALIGNED(AVX_ALIGN) v32f_absmask_[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
				const __m256 v32f_absmask = _mm256_load_ps((float*)v32f_absmask_);

				for (int x = 0; x < dest->cols; x += scale)
				{
					const int x0 = (int)(x / scale);

					int count = 0;
					for (int n = 0; n < d; n++)
					{
						const T* sb = slow_b->ptr<T>(y0 + n);
						const T* gb = glow_b->ptr<T>(y0 + n);
						for (int m = 0; m < d; m++)
						{
							const int index = 3 * (x0 + m);
							neighbor__b[count] = (float)sb[index + 0];
							gneighbor_b[count] = (float)gb[index + 0];
							neighbor__g[count] = (float)sb[index + 1];
							gneighbor_g[count] = (float)gb[index + 1];
							neighbor__r[count] = (float)sb[index + 2];
							gneighbor_r[count] = (float)gb[index + 2];
							count++;
						}
					}

					for (int n = 0; n < scale; n++)
					{
						const T* guide_ptr = guide->ptr<T>(y + n); // reference
						T* dest_ptr = dest->ptr<T>(y + n); // output
						for (int m = 0; m < scale; m++)
						{
							int idx = n * scale + m;
							const float* weightmap_ptr = weightmap->ptr<float>(idx);

							const int index = 3 * (x + m);
							const T gb = guide_ptr[index + 0];
							const T gg = guide_ptr[index + 1];
							const T gr = guide_ptr[index + 2];

							float vb = 0.f;
							float wb = 0.f;
							float vg = 0.f;
							float wg = 0.f;
							float vr = 0.f;
							float wr = 0.f;
#ifdef USE_SIMD_JBU
							__m256 mgb = _mm256_set1_ps(gb);
							__m256 mgg = _mm256_set1_ps(gg);
							__m256 mgr = _mm256_set1_ps(gr);

							__m256 mvb = _mm256_setzero_ps();
							__m256 mwb = _mm256_setzero_ps();
							__m256 mvg = _mm256_setzero_ps();
							__m256 mwg = _mm256_setzero_ps();
							__m256 mvr = _mm256_setzero_ps();
							__m256 mwr = _mm256_setzero_ps();

							const int simdend = ksize / 8 * 8;// nxn kernel(n: odd value) must be 8*m+1
							for (int k = 0; k < simdend; k += 8)
							{
								__m256 mw = _mm256_load_ps(weightmap_ptr + k);

								__m256 mg = _mm256_load_ps(gneighbor_b + k);
								__m256 ms = _mm256_load_ps(neighbor__b + k);
								__m256i midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgb, mg), v32f_absmask));
								__m256 mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
								mvb = _mm256_fmadd_ps(mrw, ms, mvb);
								mwb = _mm256_add_ps(mrw, mwb);

								mg = _mm256_load_ps(gneighbor_g + k);
								ms = _mm256_load_ps(neighbor__g + k);
								midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgg, mg), v32f_absmask));
								mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
								mvg = _mm256_fmadd_ps(mrw, ms, mvg);
								mwg = _mm256_add_ps(mrw, mwg);

								mg = _mm256_load_ps(gneighbor_r + k);
								ms = _mm256_load_ps(neighbor__r + k);
								midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(mgr, mg), v32f_absmask));
								mrw = _mm256_mul_ps(mw, _mm256_i32gather_ps(range_weight, midx, sizeof(float)));
								mvr = _mm256_fmadd_ps(mrw, ms, mvr);
								mwr = _mm256_add_ps(mrw, mwr);
							}
							vb = _mm256_reduceadd_ps(mvb);
							vg = _mm256_reduceadd_ps(mvg);
							vr = _mm256_reduceadd_ps(mvr);
							wb = _mm256_reduceadd_ps(mwb);
							wg = _mm256_reduceadd_ps(mwg);
							wr = _mm256_reduceadd_ps(mwr);

							int ed = ksize - 1;
							float cwb = range_weight[(int)abs(gb - gneighbor_b[ed])] * weightmap_ptr[ed];
							vb += cwb * neighbor__b[ed];
							wb += cwb;

							float cwg = range_weight[(int)abs(gg - gneighbor_g[ed])] * weightmap_ptr[ed];
							vg += cwg * neighbor__g[ed];
							wg += cwg;

							float cwr = range_weight[(int)abs(gr - gneighbor_r[ed])] * weightmap_ptr[ed];
							vr += cwr * neighbor__r[ed];
							wr += cwr;
#else
							for (int k = 0; k < d * d; k++)
							{
								float cwb = rweight[abs(gb - (int)gneighbor_b[k])] * weightmap_ptr[k];
								vb += cwb * neighbor__b[k];
								wb += cwb;

								float cwg = rweight[abs(gg - (int)gneighbor_g[k])] * weightmap_ptr[k];
								vg += cwg * neighbor__g[k];
								wg += cwg;

								float cwr = rweight[abs(gr - (int)gneighbor_r[k])] * weightmap_ptr[k];
								vr += cwr * neighbor__r[k];
								wr += cwr;
							}
#endif
							dest_ptr[index + 0] = saturate_cast<T>(vb / wb);
							dest_ptr[index + 1] = saturate_cast<T>(vg / wg);
							dest_ptr[index + 2] = saturate_cast<T>(vr / wr);
						}
					}
				}
			}

			_mm_free(neighbor__b);
			_mm_free(neighbor__g);
			_mm_free(neighbor__r);
			_mm_free(gneighbor_b);
			_mm_free(gneighbor_g);
			_mm_free(gneighbor_r);
		}
	};

	template<typename T>
	class JointBilateralUpsampe64F_ParallelBody : public cv::ParallelLoopBody
	{
	private:
		const cv::Mat* slow_b;
		const cv::Mat* glow_b;
		const cv::Mat* guide;
		const cv::Mat* weightmap;
		cv::Mat* dest;

		const int scale;
		const int r;
		const double* range_weight;
	public:
		JointBilateralUpsampe64F_ParallelBody(const cv::Mat& src, const cv::Mat& guide_low, const cv::Mat& guide_high, const cv::Mat& weightmap, cv::Mat& dst, const int scale, const int r, const double* rweight)
			: slow_b(&src), glow_b(&guide_low), guide(&guide_high), weightmap(&weightmap), dest(&dst), scale(scale), r(r), range_weight(rweight)
		{
		}

		void operator() (const cv::Range& range) const
		{
			const int d = 2 * r + 1;
			const int ksize = d * d;

			double* neighbor__b = (double*)_mm_malloc(sizeof(double) * ksize, AVX_ALIGN);
			double* neighbor__g = (double*)_mm_malloc(sizeof(double) * ksize, AVX_ALIGN);
			double* neighbor__r = (double*)_mm_malloc(sizeof(double) * ksize, AVX_ALIGN);
			double* gneighbor_b = (double*)_mm_malloc(sizeof(double) * ksize, AVX_ALIGN);
			double* gneighbor_g = (double*)_mm_malloc(sizeof(double) * ksize, AVX_ALIGN);
			double* gneighbor_r = (double*)_mm_malloc(sizeof(double) * ksize, AVX_ALIGN);

			static const long long CV_DECL_ALIGNED(AVX_ALIGN) v64f_absmask_[] = { 0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff };

			for (int y = range.start; y < range.end; y += scale)
			{
				const int y0 = (int)(y / scale);
				const __m256d v64f_absmask = _mm256_load_pd((double*)v64f_absmask_);

				for (int x = 0; x < dest->cols; x += scale)
				{
					const int x0 = (int)(x / scale);

					int count = 0;
					for (int n = 0; n < d; n++)
					{
						for (int m = 0; m < d; m++)
						{
							neighbor__b[count] = (double)slow_b->at<T>(y0 + n, 3 * (x0 + m) + 0);
							gneighbor_b[count] = (double)glow_b->at<T>(y0 + n, 3 * (x0 + m) + 0);
							neighbor__g[count] = (double)slow_b->at<T>(y0 + n, 3 * (x0 + m) + 1);
							gneighbor_g[count] = (double)glow_b->at<T>(y0 + n, 3 * (x0 + m) + 1);
							neighbor__r[count] = (double)slow_b->at<T>(y0 + n, 3 * (x0 + m) + 2);
							gneighbor_r[count] = (double)glow_b->at<T>(y0 + n, 3 * (x0 + m) + 2);
							count++;
						}
					}

					for (int n = 0; n < scale; n++)
					{
						const T* guide_ptr = guide->ptr<T>(y + n); // reference
						T* dest_ptr = dest->ptr<T>(y + n); // output
						for (int m = 0; m < scale; m++)
						{
							int idx = n * scale + m;
							const double* weightmap_ptr = weightmap->ptr<double>(idx);

							const T gb = guide_ptr[3 * (x + m) + 0];
							const T gg = guide_ptr[3 * (x + m) + 1];
							const T gr = guide_ptr[3 * (x + m) + 2];

							double vb = 0.0;
							double wb = 0.0;
							double vg = 0.0;
							double wg = 0.0;
							double vr = 0.0;
							double wr = 0.0;
#ifdef USE_SIMD_JBU
							__m256d mgb = _mm256_set1_pd(gb);
							__m256d mgg = _mm256_set1_pd(gg);
							__m256d mgr = _mm256_set1_pd(gr);

							__m256d mvb = _mm256_setzero_pd();
							__m256d mwb = _mm256_setzero_pd();
							__m256d mvg = _mm256_setzero_pd();
							__m256d mwg = _mm256_setzero_pd();
							__m256d mvr = _mm256_setzero_pd();
							__m256d mwr = _mm256_setzero_pd();

							const int simdend = ksize / 4 * 4;// nxn kernel(n: odd value) must be 8*m+1
							for (int k = 0; k < simdend; k += 4)
							{
								__m256d mw = _mm256_load_pd(weightmap_ptr + k);

								__m256d mg = _mm256_load_pd(gneighbor_b + k);
								__m256d ms = _mm256_load_pd(neighbor__b + k);
								__m128i midx = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(mgb, mg), v64f_absmask));
								__m256d mrw = _mm256_mul_pd(mw, _mm256_i32gather_pd(range_weight, midx, sizeof(double)));
								mvb = _mm256_fmadd_pd(mrw, ms, mvb);
								mwb = _mm256_add_pd(mrw, mwb);

								mg = _mm256_load_pd(gneighbor_g + k);
								ms = _mm256_load_pd(neighbor__g + k);
								midx = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(mgg, mg), v64f_absmask));
								mrw = _mm256_mul_pd(mw, _mm256_i32gather_pd(range_weight, midx, sizeof(double)));
								mvg = _mm256_fmadd_pd(mrw, ms, mvg);
								mwg = _mm256_add_pd(mrw, mwg);

								mg = _mm256_load_pd(gneighbor_r + k);
								ms = _mm256_load_pd(neighbor__r + k);
								midx = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(mgr, mg), v64f_absmask));
								mrw = _mm256_mul_pd(mw, _mm256_i32gather_pd(range_weight, midx, sizeof(double)));
								mvr = _mm256_fmadd_pd(mrw, ms, mvr);
								mwr = _mm256_add_pd(mrw, mwr);
							}
							vb = _mm256_reduceadd_pd(mvb);
							vg = _mm256_reduceadd_pd(mvg);
							vr = _mm256_reduceadd_pd(mvr);
							wb = _mm256_reduceadd_pd(mwb);
							wg = _mm256_reduceadd_pd(mwg);
							wr = _mm256_reduceadd_pd(mwr);

							int ed = ksize - 1;
							double cwb = range_weight[(int)abs(gb - gneighbor_b[ed])] * weightmap_ptr[ed];
							vb += cwb * neighbor__b[ed];
							wb += cwb;

							double cwg = range_weight[(int)abs(gg - gneighbor_g[ed])] * weightmap_ptr[ed];
							vg += cwg * neighbor__g[ed];
							wg += cwg;

							double cwr = range_weight[(int)abs(gr - gneighbor_r[ed])] * weightmap_ptr[ed];
							vr += cwr * neighbor__r[ed];
							wr += cwr;
#else
							for (int k = 0; k < d * d; k++)
							{
								float cwb = rweight[abs(gb - (int)gneighbor_b[k])] * weightmap_ptr[k];
								vb += cwb * neighbor__b[k];
								wb += cwb;

								float cwg = rweight[abs(gg - (int)gneighbor_g[k])] * weightmap_ptr[k];
								vg += cwg * neighbor__g[k];
								wg += cwg;

								float cwr = rweight[abs(gr - (int)gneighbor_r[k])] * weightmap_ptr[k];
								vr += cwr * neighbor__r[k];
								wr += cwr;
							}
#endif
							dest_ptr[3 * (x + m) + 0] = saturate_cast<T>(vb / wb);
							dest_ptr[3 * (x + m) + 1] = saturate_cast<T>(vg / wg);
							dest_ptr[3 * (x + m) + 2] = saturate_cast<T>(vr / wr);
						}
					}
				}
			}

			_mm_free(neighbor__b);
			_mm_free(neighbor__g);
			_mm_free(neighbor__r);
			_mm_free(gneighbor_b);
			_mm_free(gneighbor_g);
			_mm_free(gneighbor_r);
		}
	};

	void JointBilateralUpsample::upsample(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int r, const double sigma_r, const double sigma_s)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
		CV_Assert(src.channels() == 3);
		dest.create(guide.size(), src.type());

		const int scale = guide.size().width / src.size().width;
		int border = BORDER_REPLICATE;
		copyMakeBorder(src, src_b, r, r, r, r, border);

		cp::downsample(guide.getMat(), guide_low, scale, cp::Downsample(INTER_NEAREST), 0);
		copyMakeBorder(guide_low, guide_low_b, r, r, r, r, border);

		const int d = 2 * r + 1;
		weightmap.create(scale * scale, d * d, CV_32F);
		const float spaceCoeff = float(1.0 / (-2.0 * sigma_s * sigma_s * scale * scale));
		float space_w_min = FLT_MAX;
		for (int j = 0; j < scale; j++)
		{
			for (int i = 0; i < scale; i++)
			{
				float* wmap = weightmap.ptr<float>(j * scale + i);
				int count = 0;
				float wsum = 0.f;
				for (int n = 0; n < d; n++)
				{
					for (int m = 0; m < d; m++)
					{
						float xf = abs(m - r) + (float)i / scale;
						float yf = abs(n - r) + (float)j / scale;
						float dist = hypot(xf, yf);
						float w = fmath::exp(dist * dist * spaceCoeff);
						wsum += w;
						wmap[count++] = w;
					}
				}
				wsum = 1.f / wsum;
				for (int n = 0; n < d * d; n++)
				{
					float v = wmap[n] * wsum;
					v = (v < FLT_MIN) ? 0.f : v;
					wmap[n] = v;
					space_w_min = min(v, space_w_min);
				}
			}
		}

		//float CV_DECL_ALIGNED(AVX_ALIGN) range_weight[256];
		float* range_weight = (float*)_mm_malloc(256 * sizeof(float), AVX_ALIGN);
#ifdef LP_NORM
		//lp norm
		float n = 2.f;
		for (int i = 0; i < 256; i++)
		{
			rweight[i] = exp(pow(i, n) / (-n * pow(sigma_r, n)));
		}
#else
		for (int i = 0; i < 256; i++)
		{
			float v = fmath::exp(i * i / (-2.f * float(sigma_r * sigma_r)));
			range_weight[i] = float(v * space_w_min < FLT_MIN) ? 0.f : v;
			//range_weight[i] =  v;
		}

#endif

		Mat dst = dest.getMat();
		Mat gid = guide.getMat();

		const double nstripes = (dest.size().height / scale);
		if (src.depth() == CV_8U)
		{
			cv::parallel_for_
			(
				cv::Range(0, dest.size().height),
				JointBilateralUpsampe32F_ParallelBody<uchar>(src_b, guide_low_b, gid, weightmap, dst, scale, r, range_weight),
				nstripes
			);
		}
		else if (src.depth() == CV_32F)
		{
			cv::parallel_for_
			(
				cv::Range(0, dest.size().height),
				JointBilateralUpsampe32F_ParallelBody<float>(src_b, guide_low_b, gid, weightmap, dst, scale, r, range_weight),
				nstripes
			);
		}

		_mm_free(range_weight);
	}

	void JointBilateralUpsample::upsample64F(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int r, const double sigma_r, const double sigma_s)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
		CV_Assert(src.channels() == 3);
		dest.create(guide.size(), src.type());

		const int scale = guide.size().width / src.size().width;
		int border = BORDER_REPLICATE;
		copyMakeBorder(src, src_b, r, r, r, r, border);

		cp::downsample(guide.getMat(), guide_low, scale, cp::Downsample(INTER_NEAREST), 0);
		copyMakeBorder(guide_low, guide_low_b, r, r, r, r, border);

		const int d = 2 * r + 1;
		weightmap.create(scale * scale, d * d, CV_64F);
		const double spaceCoeff = 1.0 / (-2.0 * sigma_s * sigma_s * scale * scale);
		double space_w_min = DBL_MAX;
		for (int j = 0; j < scale; j++)
		{
			for (int i = 0; i < scale; i++)
			{
				double* wmap = weightmap.ptr<double>(j * scale + i);
				int count = 0;
				double wsum = 0.0;
				for (int n = 0; n < d; n++)
				{
					for (int m = 0; m < d; m++)
					{
						double xf = abs(m - r) + (double)i / scale;
						double yf = abs(n - r) + (double)j / scale;
						double dist = hypot(xf, yf);
						double w = (double)std::exp(dist * dist * spaceCoeff);
						wsum += w;
						wmap[count++] = w;
					}
				}
				wsum = 1.0 / wsum;
				for (int n = 0; n < d * d; n++)
				{
					double v = wmap[n] * wsum;
					v = (v < DBL_MIN) ? 0.f : v;
					wmap[n] = v;
					space_w_min = min(v, space_w_min);
				}
			}
		}

		double CV_DECL_ALIGNED(AVX_ALIGN) range_weight[256];
#ifdef LP_NORM
		//lp norm
		float n = 2.f;
		for (int i = 0; i < 256; i++)
		{
			rweight[i] = exp(pow(i, n) / (-n * pow(sigma_r, n)));
		}
#else

		for (int i = 0; i < 256; i++)
		{
			double v = double(std::exp(i * i / (-2.0 * sigma_r * sigma_r)));
			range_weight[i] = (v * space_w_min < DBL_MIN) ? 0.0 : v;
			//range_weight[i] =  v;
		}

#endif

		Mat dst = dest.getMat();
		Mat gid = guide.getMat();

		if (src.depth() == CV_8U)
		{
			cv::parallel_for_
			(
				cv::Range(0, dest.size().height),
				JointBilateralUpsampe64F_ParallelBody<uchar>(src_b, guide_low_b, gid, weightmap, dst, scale, r, range_weight),
				8
			);
		}
		else if (src.depth() == CV_32F)
		{
			cv::parallel_for_
			(
				cv::Range(0, dest.size().height),
				JointBilateralUpsampe64F_ParallelBody<float>(src_b, guide_low_b, gid, weightmap, dst, scale, r, range_weight),
				8
			);
		}
	}
#pragma endregion
}