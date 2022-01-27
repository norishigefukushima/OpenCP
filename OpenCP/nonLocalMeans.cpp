#include "nonLocalMeans.hpp"
#include "color.hpp"
#include "blend.hpp"
#include "inlineSIMDFunctions.hpp"
#include "inlineMathFunctions.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	inline __m128i _mm_movelh_si128(__m128i a, __m128i b)
	{
		return _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)));
	}

#pragma region NLM no optimization
	template <class srcType>
	class NonlocalMeansFilterNoOptimizationInvorker_ : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int templeteWindowSizeX;
		int searchWindowSizeX;
		int templeteWindowSizeY;
		int searchWindowSizeY;

		float* w;
	public:

		NonlocalMeansFilterNoOptimizationInvorker_(Mat& src_, Mat& dest_, int templeteWindowSizeX_, int templeteWindowSizeY_, int searchWindowSizeX_, int searchWindowSizeY_, float* weight)
			: im(&src_), dest(&dest_), templeteWindowSizeX(templeteWindowSizeX_), searchWindowSizeX(searchWindowSizeX_), templeteWindowSizeY(templeteWindowSizeY_), searchWindowSizeY(searchWindowSizeY_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range& r) const
		{
			const int tr_x = templeteWindowSizeX >> 1;
			const int sr_x = searchWindowSizeX >> 1;
			const int tr_y = templeteWindowSizeY >> 1;
			const int sr_y = searchWindowSizeY >> 1;
			const int cstep = (im->cols - templeteWindowSizeX) * im->channels();
			const int imstep = im->cols * im->channels();

			const int tD = templeteWindowSizeX * templeteWindowSizeY;
			const double tdiv = 1.0 / (double)(tD);//templete square div


			if (im->channels() == 3)
			{
				for (int j = r.start; j < r.end; j++)
				{
					srcType* d = dest->ptr<srcType>(j);
					for (int i = 0; i < dest->cols; i++)
					{
						double r = 0.0;
						double g = 0.0;
						double b = 0.0;
						double tweight = 0.0;
						//search loop
						const srcType* tprt = im->ptr<srcType>(sr_y + j) + 3 * (sr_x + i);
						const srcType* sptr2 = im->ptr<srcType>(j) + 3 * (i);

						for (int l = searchWindowSizeY; l--;)
						{
							const srcType* sptr = sptr2 + imstep * (l);
							for (int k = searchWindowSizeX; k--;)
							{
								//templete loop
								double e = 0.0;
								const srcType* t = tprt;
								srcType* s = (srcType*)(sptr + 3 * k);
								for (int n = templeteWindowSizeY; n--;)
								{
									for (int m = templeteWindowSizeX; m--;)
									{
										// computing color L2 norm
										e += abs(s[0] - t[0]) + abs(s[1] - t[1]) + abs(s[2] - t[2]);//L2 norm
										s += 3, t += 3;
									}
									t += cstep;
									s += cstep;
								}
								const int ediv = cvRound(e * tdiv);
								float www = w[ediv];

								const srcType* ss = sptr2 + imstep * (tr_y + l) + 3 * (tr_x + k);
								r += ss[0] * www;
								g += ss[1] * www;
								b += ss[2] * www;
								//get weighted Euclidean distance
								tweight += www;
							}
						}
						d[0] = saturate_cast<srcType>(r / tweight);
						d[1] = saturate_cast<srcType>(g / tweight);
						d[2] = saturate_cast<srcType>(b / tweight);
						d += 3;
					}//i
				}//j
			}
			else if (im->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					srcType* d = dest->ptr<srcType>(j);
					for (int i = 0; i < dest->cols; i++)
					{
						double value = 0.0;
						double tweight = 0.0;
						//search loop
						srcType* tprt = im->ptr<srcType>(sr_y + j) + (sr_x + i);
						srcType* sptr2 = im->ptr<srcType>(j) + (i);

						for (int l = searchWindowSizeY; l--;)
						{
							srcType* sptr = sptr2 + imstep * (l);
							for (int k = searchWindowSizeX; k--;)
							{
								//templete loop
								double e = 0.0;
								srcType* t = tprt;
								srcType* s = sptr + k;
								for (int n = templeteWindowSizeY; n--;)
								{
									for (int m = templeteWindowSizeX; m--;)
									{
										// computing color L2 norm
										e += abs(*s - *t);
										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								const int ediv = cvRound(e * tdiv);
								float www = w[ediv];
								value += sptr2[imstep * (tr_y + l) + tr_x + k] * www;
								//get weighted Euclidean distance
								tweight += www;
							}
						}
						//weight normalization
						*(d++) = saturate_cast<srcType>(value / tweight);
					}//i
				}//j
			}
		}
	};

	void nonLocalMeansFilterNoOptimization(const Mat& src, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int borderType)
	{
		dest.create(src.size(), src.type());

		const int bbx = (templeteWindowSize.width >> 1) + (searchWindowSize.width >> 1);
		const int bby = (templeteWindowSize.height >> 1) + (searchWindowSize.height >> 1);

		//create large size image for bounding box;
		Mat im;
		copyMakeBorder(src, im, bby, bby, bbx, bbx, borderType);

		vector<float> weight(256 * 256 * src.channels());
		float* w = &weight[0];
		const double gauss_sd = (sigma == 0.0) ? h : sigma;
		double gauss_color_coeff = -(1.0 / (double)(src.channels())) * (1.0 / (h * h));
		for (int i = 0; i < 256 * 256 * src.channels(); i++)
		{
			double v = std::exp(max(i - 2.0 * gauss_sd * gauss_sd, 0.0) * gauss_color_coeff);
			w[i] = (float)v;
		}

		if (src.depth() == CV_8U)
		{
			NonlocalMeansFilterNoOptimizationInvorker_<uchar> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		if (src.depth() == CV_16U)
		{
			NonlocalMeansFilterNoOptimizationInvorker_<ushort> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		if (src.depth() == CV_16S)
		{
			NonlocalMeansFilterNoOptimizationInvorker_<short> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		else if (src.depth() == CV_32F)
		{
			NonlocalMeansFilterNoOptimizationInvorker_<float> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		else if (src.depth() == CV_64F)
		{
			NonlocalMeansFilterNoOptimizationInvorker_<double> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
	}
#pragma endregion

#pragma region NLM
	template<int norm>
	class NonlocalMeansFilterInvorker8u_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
	public:

		NonlocalMeansFilterInvorker8u_AVX(Mat& src_, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight)
			: im(&src_), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr <uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}
						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));
						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr <uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}
						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));
						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			//unroll1(r);
			//unroll2(r);
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	template<int norm>
	class NonlocalMeansFilterInvorker32f_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
	public:

		NonlocalMeansFilterInvorker32f_AVX(Mat& src_, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight)
			: im(&src_), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = im->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = im->ptr <float>(sr_y + j) + sr_x;
					const float* sptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = im->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = im->ptr <float>(sr_y + j) + sr_x;
					const float* sptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	void nonLocalMeansFilter(InputArray src, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp, const int patchnorm, const int borderType)
	{
		nonLocalMeansFilter(src, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma, powexp, patchnorm, borderType);
	}

	void nonLocalMeansFilter(InputArray src_, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma, const double powexp, const int patchnorm, const int borderType)
	{
#pragma region alloc
		Mat src = src_.getMat();

		const int bbx = (patchWindowSize.width >> 1) + (kernelWindowSize.width >> 1);
		const int bby = (patchWindowSize.height >> 1) + (kernelWindowSize.height >> 1);

		//create large size image for bounding box;
		const int dpad = get_simd_ceil(src.cols, 32) - src.cols;
		const int spad = get_simd_ceil(src.cols + 2 * bbx, 32) - (src.cols + 2 * bbx);
		Mat dst(Size(src.cols + dpad, src.rows), src.type());

		Mat im;
		if (src.channels() == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (src.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, im);
		}

#pragma endregion
#pragma region weight computation;
		const int range_size = (patchnorm == 2) ? (int)ceil(sqrt(255 * 255 * src.channels() * patchWindowSize.area())) : (int)256 * src.channels();
		float* range_weight = (float*)_mm_malloc(sizeof(float) * range_size, AVX_ALIGN);
		double gauss_color_coeff = (1.0 / (sigma));
		if (powexp == 0)
		{
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (i <= sigma) ? 1.f : 0.f;
			}
		}
		else
		{
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (float)std::exp(-pow(abs(i * gauss_color_coeff), powexp) / powexp);
			}
		}
#pragma endregion

		if (src.depth() == CV_8U)
		{
			if (patchnorm == 1)
			{
				NonlocalMeansFilterInvorker8u_AVX<1> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				NonlocalMeansFilterInvorker8u_AVX<2> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else if (src.depth() == CV_32F)
		{
			if (patchnorm == 1)
			{
				NonlocalMeansFilterInvorker32f_AVX<1> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				NonlocalMeansFilterInvorker32f_AVX<2> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			if (patchnorm == 1)
			{
				NonlocalMeansFilterInvorker32f_AVX<1> body(imf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				NonlocalMeansFilterInvorker32f_AVX<2> body(imf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			dstf.convertTo(dst, src.depth());
		}

		Mat(dst(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
		_mm_free(range_weight);
	}


#pragma endregion

#pragma region JNLM
	template<int norm>
	class JointNonlocalMeansFilterInvorker8u_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* gim;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
	public:

		JointNonlocalMeansFilterInvorker8u_AVX(Mat& src_, Mat& guide_, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight)
			: im(&src_), gim(&guide_), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = gim->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = gim->ptr<uchar>(j);
					const uchar* rptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						const uchar* rptr2 = rptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = gim->ptr <uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = gim->ptr<uchar>(j);
					const uchar* rptr2_ = im->ptr <uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						const uchar* rptr2 = rptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}
						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));
						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = im->ptr <uchar>(sr_y + j) + (sr_x + i);
						const uchar* sptr2 = im->ptr<uchar>(j) + (i);
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}
						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));
						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			//unroll1(r);
			//unroll2(r);
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	template<int norm>
	class JointNonlocalMeansFilterInvorker32f_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* gim;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
	public:

		JointNonlocalMeansFilterInvorker32f_AVX(Mat& src_, Mat& guide_, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight)
			: im(&src_), gim(&guide_), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr <float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr <float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;

									const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;

									__m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4);
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4);
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4);
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4);
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	void jointNonLocalMeansFilter(InputArray src, InputArray guide, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp, const int patchnorm, const int borderType)
	{
		jointNonLocalMeansFilter(src, guide, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma, powexp, patchnorm, borderType);
	}

	void jointNonLocalMeansFilter(InputArray src_, InputArray guide_, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma, const double powexp, const int patchnorm, const int borderType)
	{
		CV_Assert(src_.channels() == guide_.channels());
#pragma region alloc
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();

		const int bbx = (patchWindowSize.width >> 1) + (kernelWindowSize.width >> 1);
		const int bby = (patchWindowSize.height >> 1) + (kernelWindowSize.height >> 1);

		//create large size image for bounding box;
		const int dpad = get_simd_ceil(src.cols, 32) - src.cols;
		const int spad = get_simd_ceil(src.cols + 2 * bbx, 32) - (src.cols + 2 * bbx);
		Mat dst(Size(src.cols + dpad, src.rows), src.type());

		Mat im, gim;
		if (src.channels() == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (src.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, im);
		}
		if (guide.channels() == 1)
		{
			copyMakeBorder(guide, gim, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (src.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(guide, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, gim);
		}
#pragma endregion
#pragma region weight computation;
		const int range_size = (patchnorm == 2) ? (int)ceil(sqrt(255 * 255 * src.channels() * patchWindowSize.area())) : (int)256 * src.channels();
		float* range_weight = (float*)_mm_malloc(sizeof(float) * range_size, AVX_ALIGN);
		double gauss_color_coeff = (1.0 / (sigma));
		if (powexp == 0)
		{
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (i <= sigma) ? 1.f : 0.f;
			}
		}
		else
		{
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (float)std::exp(-pow(abs(i * gauss_color_coeff), powexp) / powexp);
			}
		}
#pragma endregion

		if (src.depth() == CV_8U)
		{
			if (patchnorm == 1)
			{
				JointNonlocalMeansFilterInvorker8u_AVX<1> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				JointNonlocalMeansFilterInvorker8u_AVX<2> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else if (src.depth() == CV_32F)
		{
			if (patchnorm == 1)
			{
				JointNonlocalMeansFilterInvorker32f_AVX<1> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				JointNonlocalMeansFilterInvorker32f_AVX<2> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else
		{
			Mat imf, gimf, dstf;
			im.convertTo(imf, CV_32F);
			gim.convertTo(gimf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			if (patchnorm == 1)
			{
				JointNonlocalMeansFilterInvorker32f_AVX<1> body(imf, gimf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				JointNonlocalMeansFilterInvorker32f_AVX<2> body(imf, gimf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			dstf.convertTo(dst, src.depth());
		}

		Mat(dst(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
		_mm_free(range_weight);
	}

#pragma endregion

#pragma region PBF
	template<int norm>
	class PatchBilateralFilterInvorker8u_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
		float* space_weight;
	public:

		PatchBilateralFilterInvorker8u_AVX(Mat& src_, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight, float* space_weight)
			: im(&src_), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight), space_weight(space_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = im->ptr <uchar>(sr_y + j) + (sr_x + i);
						const uchar* sptr2 = im->ptr<uchar>(j) + (i);
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));

						//_mm_storeu_si128((__m128i*)(d + 0), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1))));
						//_mm_storeu_si128((__m128i*)(d + 16), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3))));

						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = im->ptr <uchar>(sr_y + j) + (sr_x + i);
						const uchar* sptr2 = im->ptr<uchar>(j) + (i);
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));

						//_mm_storeu_si128((__m128i*)(d + 0), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1))));
						//_mm_storeu_si128((__m128i*)(d + 16), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3))));

						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			//unroll1(r);
			//unroll2(r);
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	template<int norm>
	class PatchBilateralFilterInvorker32f_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
		float* space_weight;
	public:

		PatchBilateralFilterInvorker32f_AVX(Mat& src_, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight, float* space_weight)
			: im(&src_), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight), space_weight(space_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = im->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = im->ptr <float>(sr_y + j) + (sr_x + i);
						const float* sptr2 = im->ptr<float>(j) + (i);
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = im->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();

						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();

						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();

						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = im->ptr <float>(sr_y + j) + (sr_x + i);
						const float* sptr2 = im->ptr<float>(j) + (i);
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = sptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	void patchBilateralFilter(InputArray src, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, const int borderType)
	{
		patchBilateralFilter(src, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
	}

	void patchBilateralFilter(InputArray src_, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, const int borderType)
	{
#pragma region alloc
		Mat src = src_.getMat();

		const int bbx = (patchWindowSize.width >> 1) + (kernelWindowSize.width >> 1);
		const int bby = (patchWindowSize.height >> 1) + (kernelWindowSize.height >> 1);
		//	const int D = searchWindowSize*searchWindowSize;

		//create large size image for bounding box;
		const int dpad = get_simd_ceil(src.cols, 32) - src.cols;
		const int spad = get_simd_ceil(src.cols + 2 * bbx, 32) - (src.cols + 2 * bbx);
		Mat dst(Size(src.cols + dpad, src.rows), src.type());

		Mat im;
		if (src.channels() == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (src.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, im);
		}
#pragma endregion
#pragma region weight computation
		//space weight
		const int space_size = kernelWindowSize.area();
		float* space_weight = (float*)_mm_malloc(sizeof(float) * space_size, AVX_ALIGN);
		for (int i = 0; i < space_size; i++)space_weight[i] = 1.f;
		const int radiusW = kernelWindowSize.width / 2;
		const int radiusH = kernelWindowSize.height / 2;
		int sindex = 0;
		if (powexp_space == 0)
		{
			for (int j = -radiusH; j <= radiusH; j++)
			{
				for (int i = -radiusW; i <= radiusW; i++)
				{
					const float dist = sqrtf(float(i * i + j * j));
					space_weight[sindex++] = (dist <= sigma_space) ? 1.f : 0.f;
				}
			}
		}
		else
		{
			double gauss_space_coeff = (1.0 / (sigma_space));
			for (int j = -radiusH; j <= radiusH; j++)
			{
				for (int i = -radiusW; i <= radiusW; i++)
				{
					const float dist = sqrtf(float(i * i + j * j));
					space_weight[sindex++] = (float)std::exp(-pow(abs(dist * gauss_space_coeff), powexp_space) / powexp_space);
				}
			}
		}

		//range weight
		const int range_size = (patchnorm == 2) ? (int)ceil(sqrt(255 * 255 * src.channels() * patchWindowSize.area())) : (int)256 * src.channels();
		float* range_weight = (float*)_mm_malloc(sizeof(float) * range_size, AVX_ALIGN);
		if (powexp_range == 0)
		{
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (i <= sigma_range) ? 1.f : 0.f;
			}
		}
		else
		{
			double gauss_color_coeff = (1.0 / (sigma_range));
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (float)std::exp(-pow(abs(i * gauss_color_coeff), powexp_range) / powexp_range);
			}
		}
#pragma endregion

		if (src.depth() == CV_8U)
		{
			if (patchnorm == 1)
			{
				PatchBilateralFilterInvorker8u_AVX<1> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				PatchBilateralFilterInvorker8u_AVX<2> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else if (src.depth() == CV_32F)
		{
			if (patchnorm == 1)
			{
				PatchBilateralFilterInvorker32f_AVX<1> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				PatchBilateralFilterInvorker32f_AVX<2> body(im, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			if (patchnorm == 1)
			{
				PatchBilateralFilterInvorker32f_AVX<1> body(imf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				PatchBilateralFilterInvorker32f_AVX<2> body(imf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			dstf.convertTo(dst, src.depth());
		}

		Mat(dst(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
		_mm_free(space_weight);
		_mm_free(range_weight);
	}
#pragma endregion

#pragma region JPBF
	template<int norm>
	class JointPatchBilateralFilterInvorker8u_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* gim;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
		float* space_weight;
	public:

		JointPatchBilateralFilterInvorker8u_AVX(Mat& src, Mat& guide, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight, float* space_weight)
			: im(&src), gim(&guide), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight), space_weight(space_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = gim->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = gim->ptr<uchar>(j);
					const uchar* rptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						const uchar* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = gim->ptr <uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = gim->ptr<uchar>(j);
					const uchar* rptr2_ = im->ptr <uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						const uchar* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));

						//_mm_storeu_si128((__m128i*)(d + 0), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1))));
						//_mm_storeu_si128((__m128i*)(d + 16), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3))));

						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);
			//const __m256i mtdivi = _mm256_set1_epi32(patchSize);//not used in shift mode
			const int divshift = (int)floor(log2((double)patchSize));

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = gim->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = gim->ptr<uchar>(j);
					const uchar* rptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						const uchar* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											const __m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											const __m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256i v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_adds_epu16(mdist0, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_adds_epu16(mdist0, v);

											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffb));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffg));
											mdist1 = _mm256_adds_epu16(mdist1, v);
											v = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffr));
											mdist1 = _mm256_adds_epu16(mdist1, v);

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffb = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + 0)), _mm256_loadu_si256((__m256i*)(t + 0)));
											__m256i diffg = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep)), _mm256_loadu_si256((__m256i*)(t + colorstep)));
											__m256i diffr = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)(s + colorstep2)), _mm256_loadu_si256((__m256i*)(t + colorstep2)));

											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffb = _mm256_permute2x128_si256(diffb, diffb, 0x01);
											diffg = _mm256_permute2x128_si256(diffg, diffg, 0x01);
											diffr = _mm256_permute2x128_si256(diffr, diffr, 0x01);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffb));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffg));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffr));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);

											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffb, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffg, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffr, _MM_SHUFFLE(3, 2, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep)), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + colorstep2)), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep)), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8 + colorstep2)), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep)), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16 + colorstep2)), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep)), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24 + colorstep2)), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps2epu8_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps2epu8_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps2epu8_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						if (i == dest->cols - 32)_mm256_storescalar_ps2epu8_color(d + 72, mb3, mg3, mr3);
						else _mm256_storeu_ps2epu8_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = gim->ptr <uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = gim->ptr<uchar>(j);
					const uchar* rptr2_ = im->ptr <uchar>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						const uchar* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const uchar* t = tprt;
								const uchar* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256i mdist0 = _mm256_setzero_si256();
									__m256i mdist1 = _mm256_setzero_si256();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											const __m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											mdist0 = _mm256_adds_epu16(mdist0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diffv)));
											mdist1 = _mm256_adds_epu16(mdist1, _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(diffv)));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									//const __m256 mrw0 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), mtdivi), 4);
									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist0)), divshift), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									//const __m256 mrw1 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), mtdivi), 4);
									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist0)), divshift), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									//const __m256 mrw2 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), mtdivi), 4);
									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdist1)), divshift), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									//const __m256 mrw3 = _mm256_i32gather_ps(range_weight, _mm256_div_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), mtdivi), 4);
									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_srai_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256hi_si128(mdist1)), divshift), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else //compute patch distance L2
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											__m256i diffv = _mm256_absdiff_epu8(_mm256_loadu_si256((__m256i*)s), _mm256_loadu_si256((__m256i*)t));
											__m256 v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist0 = _mm256_fmadd_ps(v, v, mdist0);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist1 = _mm256_fmadd_ps(v, v, mdist1);

											diffv = _mm256_permute2x128_si256(diffv, diffv, 0x01);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(diffv));
											mdist2 = _mm256_fmadd_ps(v, v, mdist2);
											v = _mm256_cvtepu8_ps(_mm256_castsi256_si128(_mm256_shuffle_epi32(diffv, _MM_SHUFFLE(2, 1, 3, 2))));
											mdist3 = _mm256_fmadd_ps(v, v, mdist3);
											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const uchar* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_load_epu8cvtps((__m128i*)(rptr + 0)), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_load_epu8cvtps((__m128i*)(rptr + 8)), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_load_epu8cvtps((__m128i*)(rptr + 16)), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_load_epu8cvtps((__m128i*)(rptr + 24)), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_si256((__m256i*)d, _mm256_cvtpsx4_epu8(_mm256_div_ps(mv0, mw0), _mm256_div_ps(mv1, mw1), _mm256_div_ps(mv2, mw2), _mm256_div_ps(mv3, mw3)));

						//_mm_storeu_si128((__m128i*)(d + 0), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1))));
						//_mm_storeu_si128((__m128i*)(d + 16), _mm_movelh_si128(_mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3))));

						//_mm_storel_epi64((__m128i*)(d + 0), _mm256_cvtps_epu8(_mm256_div_ps(mv0, mw0)));
						//_mm_storel_epi64((__m128i*)(d + 8), _mm256_cvtps_epu8(_mm256_div_ps(mv1, mw1)));
						//_mm_storel_epi64((__m128i*)(d + 16), _mm256_cvtps_epu8(_mm256_div_ps(mv2, mw2)));
						//_mm_storel_epi64((__m128i*)(d + 24), _mm256_cvtps_epu8(_mm256_div_ps(mv3, mw3)));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			//unroll1(r);
			//unroll2(r);
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	template<int norm>
	class JointPatchBilateralFilterInvorker32f_AVX : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* gim;
		Mat* dest;
		int patchWindowSizeX;
		int patchWindowSizeY;
		int kernelWindowSizeX;
		int kernelWindowSizeY;

		float* range_weight;
		float* space_weight;
	public:

		JointPatchBilateralFilterInvorker32f_AVX(Mat& src, Mat& guide, Mat& dest_, int patchWindowSizeX_, int patchWindowSizeY_, int kernelWindowSizeX_, int kernelWindowSizeY_, float* range_weight, float* space_weight)
			: im(&src), gim(&guide), dest(&dest_), patchWindowSizeX(patchWindowSizeX_), patchWindowSizeY(patchWindowSizeY_), kernelWindowSizeX(kernelWindowSizeX_), kernelWindowSizeY(kernelWindowSizeY_), range_weight(range_weight), space_weight(space_weight)
		{
			;
		}

		template<int patchWindowSizeX, int patchWindowSizeY>
		void unroll4_(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr <float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		void unroll4(const cv::Range& r) const
		{
			const int tr_x = patchWindowSizeX >> 1;
			const int sr_x = kernelWindowSizeX >> 1;
			const int tr_y = patchWindowSizeY >> 1;
			const int sr_y = kernelWindowSizeY >> 1;
			const int cstep = im->cols - patchWindowSizeX;
			const int imstep = im->cols;

			const int patchSize = patchWindowSizeX * patchWindowSizeY;
			const float tdiv = 1.f / (float)(patchSize);//templete square div
			const __m256 mtdiv = _mm256_set1_ps(tdiv);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mr0 = _mm256_setzero_ps();
						__m256 mg0 = _mm256_setzero_ps();
						__m256 mb0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mr1 = _mm256_setzero_ps();
						__m256 mg1 = _mm256_setzero_ps();
						__m256 mb1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mr2 = _mm256_setzero_ps();
						__m256 mg2 = _mm256_setzero_ps();
						__m256 mb2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mr3 = _mm256_setzero_ps();
						__m256 mg3 = _mm256_setzero_ps();
						__m256 mb3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = (sptr + k);
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(t + colorstep))));
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + colorstep2), _mm256_lddqu_ps(t + colorstep2))));

											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8 + colorstep2), _mm256_lddqu_ps(t + 8 + colorstep2))));

											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16 + colorstep2), _mm256_lddqu_ps(t + 16 + colorstep2))));

											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24 + colorstep2), _mm256_lddqu_ps(t + 24 + colorstep2))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0,
												_mm256_lddqu_ps(s), _mm256_lddqu_ps(s + colorstep), _mm256_lddqu_ps(s + colorstep2),
												_mm256_lddqu_ps(t), _mm256_lddqu_ps(t + colorstep), _mm256_lddqu_ps(t + colorstep2));
											mdist1 = _mm256_ssdadd_ps(mdist1,
												_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(s + 8 + colorstep), _mm256_lddqu_ps(s + 8 + colorstep2),
												_mm256_lddqu_ps(t + 8), _mm256_lddqu_ps(t + 8 + colorstep), _mm256_lddqu_ps(t + 8 + colorstep2));
											mdist2 = _mm256_ssdadd_ps(mdist2,
												_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(s + 16 + colorstep), _mm256_lddqu_ps(s + 16 + colorstep2),
												_mm256_lddqu_ps(t + 16), _mm256_lddqu_ps(t + 16 + colorstep), _mm256_lddqu_ps(t + 16 + colorstep2));
											mdist3 = _mm256_ssdadd_ps(mdist3,
												_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(s + 24 + colorstep), _mm256_lddqu_ps(s + 24 + colorstep2),
												_mm256_lddqu_ps(t + 24), _mm256_lddqu_ps(t + 24 + colorstep), _mm256_lddqu_ps(t + 24 + colorstep2));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + (tr_x + k);
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mb0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr), mb0);
									mg0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep), mg0);
									mr0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + colorstep2), mr0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mb1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mb1);
									mg1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep), mg1);
									mr1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8 + colorstep2), mr1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mb2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mb2);
									mg2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep), mg2);
									mr2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16 + colorstep2), mr2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mb3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mb3);
									mg3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep), mg3);
									mr3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24 + colorstep2), mr3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						mb0 = _mm256_div_ps(mb0, mw0);
						mg0 = _mm256_div_ps(mg0, mw0);
						mr0 = _mm256_div_ps(mr0, mw0);
						_mm256_storeu_ps_color(d, mb0, mg0, mr0);

						mb1 = _mm256_div_ps(mb1, mw1);
						mg1 = _mm256_div_ps(mg1, mw1);
						mr1 = _mm256_div_ps(mr1, mw1);
						_mm256_storeu_ps_color(d + 24, mb1, mg1, mr1);

						mb2 = _mm256_div_ps(mb2, mw2);
						mg2 = _mm256_div_ps(mg2, mw2);
						mr2 = _mm256_div_ps(mr2, mw2);
						_mm256_storeu_ps_color(d + 48, mb2, mg2, mr2);

						mb3 = _mm256_div_ps(mb3, mw3);
						mg3 = _mm256_div_ps(mg3, mw3);
						mr3 = _mm256_div_ps(mr3, mw3);
						_mm256_storeu_ps_color(d + 72, mb3, mg3, mr3);

						d += 96;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = gim->ptr <float>(sr_y + j) + sr_x;
					const float* sptr2_ = gim->ptr<float>(j);
					const float* rptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 32)
					{
						__m256 mv0 = _mm256_setzero_ps();
						__m256 mw0 = _mm256_setzero_ps();
						__m256 mv1 = _mm256_setzero_ps();
						__m256 mw1 = _mm256_setzero_ps();
						__m256 mv2 = _mm256_setzero_ps();
						__m256 mw2 = _mm256_setzero_ps();
						__m256 mv3 = _mm256_setzero_ps();
						__m256 mw3 = _mm256_setzero_ps();

						//kernel loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* rptr2 = rptr2_ + i;
						int sindex = 0;
						for (int l = kernelWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = kernelWindowSizeX; k--;)
							{
								//patch loop
								const float* t = tprt;
								const float* s = sptr + k;
								if constexpr (norm == 1) // computing color L1 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_add_ps(mdist0, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s), _mm256_lddqu_ps(t))));
											mdist1 = _mm256_add_ps(mdist1, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8))));
											mdist2 = _mm256_add_ps(mdist2, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16))));
											mdist3 = _mm256_add_ps(mdist3, _mm256_abs_ps(_mm256_sub_ps(_mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24))));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									const __m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist0, mtdiv)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									const __m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist1, mtdiv)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									const __m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist2, mtdiv)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									const __m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_mul_ps(mdist3, mtdiv)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
								else // computing color L2 norm
								{
									__m256 mdist0 = _mm256_setzero_ps();
									__m256 mdist1 = _mm256_setzero_ps();
									__m256 mdist2 = _mm256_setzero_ps();
									__m256 mdist3 = _mm256_setzero_ps();
									for (int n = patchWindowSizeY; n--;)
									{
										for (int m = patchWindowSizeX; m--;)
										{
											mdist0 = _mm256_ssdadd_ps(mdist0, _mm256_lddqu_ps(s), _mm256_lddqu_ps(t));
											mdist1 = _mm256_ssdadd_ps(mdist1, _mm256_lddqu_ps(s + 8), _mm256_lddqu_ps(t + 8));
											mdist2 = _mm256_ssdadd_ps(mdist2, _mm256_lddqu_ps(s + 16), _mm256_lddqu_ps(t + 16));
											mdist3 = _mm256_ssdadd_ps(mdist3, _mm256_lddqu_ps(s + 24), _mm256_lddqu_ps(t + 24));

											s++, t++;
										}
										t += cstep;
										s += cstep;
									}

									const float* rptr = rptr2 + imstep * (tr_y + l) + tr_x + k;
									const __m256 sw = _mm256_set1_ps(space_weight[sindex++]);

									__m256 mrw0 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist0)), 4));
									mv0 = _mm256_fmadd_ps(mrw0, _mm256_lddqu_ps(rptr + 0), mv0);
									mw0 = _mm256_add_ps(mrw0, mw0);

									__m256 mrw1 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist1)), 4));
									mv1 = _mm256_fmadd_ps(mrw1, _mm256_lddqu_ps(rptr + 8), mv1);
									mw1 = _mm256_add_ps(mrw1, mw1);

									__m256 mrw2 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist2)), 4));
									mv2 = _mm256_fmadd_ps(mrw2, _mm256_lddqu_ps(rptr + 16), mv2);
									mw2 = _mm256_add_ps(mrw2, mw2);

									__m256 mrw3 = _mm256_mul_ps(sw, _mm256_i32gather_ps(range_weight, _mm256_cvtps_epi32(_mm256_fastsqrt_ps(mdist3)), 4));
									mv3 = _mm256_fmadd_ps(mrw3, _mm256_lddqu_ps(rptr + 24), mv3);
									mw3 = _mm256_add_ps(mrw3, mw3);
								}
							}
						}

						//weight normalization
						_mm256_storeu_ps(d + 0, _mm256_div_ps(mv0, mw0));
						_mm256_storeu_ps(d + 8, _mm256_div_ps(mv1, mw1));
						_mm256_storeu_ps(d + 16, _mm256_div_ps(mv2, mw2));
						_mm256_storeu_ps(d + 24, _mm256_div_ps(mv3, mw3));
						d += 32;
					}//i
				}//j
			}
		}

		virtual void operator()(const cv::Range& r) const
		{
			if (patchWindowSizeX == 3 && patchWindowSizeY == 3) unroll4_<3, 3>(r);
			else if (patchWindowSizeX == 5 && patchWindowSizeY == 5) unroll4_<5, 5>(r);
			else if (patchWindowSizeX == 7 && patchWindowSizeY == 7) unroll4_<7, 7>(r);
			else unroll4(r);
		}
	};

	void jointPatchBilateralFilter(InputArray src, InputArray guide, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const  double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, const int borderType)
	{
		jointPatchBilateralFilter(src, guide, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
	}

	void jointPatchBilateralFilter(InputArray src_, InputArray guide_, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, const int borderType)
	{
#pragma region alloc
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();

		const int bbx = (patchWindowSize.width >> 1) + (kernelWindowSize.width >> 1);
		const int bby = (patchWindowSize.height >> 1) + (kernelWindowSize.height >> 1);
		//	const int D = searchWindowSize*searchWindowSize;

		//create large size image for bounding box;
		const int dpad = get_simd_ceil(src.cols, 32) - src.cols;
		const int spad = get_simd_ceil(src.cols + 2 * bbx, 32) - (src.cols + 2 * bbx);
		Mat dst(Size(src.cols + dpad, src.rows), src.type());

		Mat im, gim;
		if (src.channels() == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (src.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, im);
		}
		if (guide.channels() == 1)
		{
			copyMakeBorder(guide, gim, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (guide.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(guide, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, gim);
		}
#pragma endregion
#pragma region weight computation
		//space weight
		const int space_size = kernelWindowSize.area();
		float* space_weight = (float*)_mm_malloc(sizeof(float) * space_size, AVX_ALIGN);
		for (int i = 0; i < space_size; i++)space_weight[i] = 1.f;
		const int radiusW = kernelWindowSize.width / 2;
		const int radiusH = kernelWindowSize.height / 2;
		int sindex = 0;
		if (powexp_space == 0)
		{
			for (int j = -radiusH; j <= radiusH; j++)
			{
				for (int i = -radiusW; i <= radiusW; i++)
				{
					const float dist = sqrtf(float(i * i + j * j));
					space_weight[sindex++] = (dist <= sigma_space) ? 1.f : 0.f;
				}
			}
		}
		else
		{
			double gauss_space_coeff = (1.0 / (sigma_space));
			for (int j = -radiusH; j <= radiusH; j++)
			{
				for (int i = -radiusW; i <= radiusW; i++)
				{
					const float dist = sqrtf(float(i * i + j * j));
					space_weight[sindex++] = (float)std::exp(-pow(abs(dist * gauss_space_coeff), powexp_space) / powexp_space);
				}
			}
		}

		//range weight
		const int range_size = (patchnorm == 2) ? (int)ceil(sqrt(255 * 255 * src.channels() * patchWindowSize.area())) : (int)256 * src.channels();
		float* range_weight = (float*)_mm_malloc(sizeof(float) * range_size, AVX_ALIGN);
		if (powexp_range == 0)
		{
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (i <= sigma_range) ? 1.f : 0.f;
			}
		}
		else
		{
			double gauss_color_coeff = (1.0 / (sigma_range));
			for (int i = 0; i < range_size; i++)
			{
				range_weight[i] = (float)std::exp(-pow(abs(i * gauss_color_coeff), powexp_range) / powexp_range);
			}
		}
#pragma endregion

		if (src.depth() == CV_8U)
		{
			if (patchnorm == 1)
			{
				JointPatchBilateralFilterInvorker8u_AVX<1> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				JointPatchBilateralFilterInvorker8u_AVX<2> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else if (src.depth() == CV_32F)
		{
			if (patchnorm == 1)
			{
				JointPatchBilateralFilterInvorker32f_AVX<1> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				JointPatchBilateralFilterInvorker32f_AVX<2> body(im, gim, dst, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
		}
		else
		{
			Mat imf, gimf, dstf;
			im.convertTo(imf, CV_32F);
			gim.convertTo(gimf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			if (patchnorm == 1)
			{
				JointPatchBilateralFilterInvorker32f_AVX<1> body(imf, gimf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			else
			{
				JointPatchBilateralFilterInvorker32f_AVX<2> body(imf, gimf, dstf, patchWindowSize.width, patchWindowSize.height, kernelWindowSize.width, kernelWindowSize.height, range_weight, space_weight);
				cv::parallel_for_(Range(0, dst.rows), body);
			}
			dstf.convertTo(dst, src.depth());
		}

		Mat(dst(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
		_mm_free(space_weight);
		_mm_free(range_weight);
	}
#pragma endregion

#pragma region Separable

	void nonLocalMeansFilterSeparable(InputArray src, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma, const double powexp, const int patchnorm, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		switch (method)
		{
		case cp::SEPARABLE_METHOD::SWITCH_VH:
		{
			nonLocalMeansFilter(src, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma, powexp, patchnorm, borderType);
			jointNonLocalMeansFilter(dest, src, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma * alpha, powexp, patchnorm, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::DIRECT_VH:
		{
			nonLocalMeansFilter(src, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma, powexp, patchnorm, borderType);
			nonLocalMeansFilter(dest, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma, powexp, patchnorm, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::SWITCH_HV:
		{
			nonLocalMeansFilter(src, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma, powexp, patchnorm, borderType);
			jointNonLocalMeansFilter(dest, src, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma * alpha, powexp, patchnorm, borderType);
		}
		break;

		case cp::SEPARABLE_METHOD::DIRECT_HV:
		{
			nonLocalMeansFilter(src, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma, powexp, patchnorm, borderType);
			nonLocalMeansFilter(dest, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma, powexp, patchnorm, borderType);
		}
		break;
		default:
			break;
		}
	}

	void nonLocalMeansFilterSeparable(InputArray src, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp, const int patchnorm, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		nonLocalMeansFilterSeparable(src, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma, powexp, patchnorm, method, alpha, borderType);
	}

	void jointNonLocalMeansFilterSeparable(InputArray src, InputArray guide, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma, const double powexp, const int patchnorm, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		switch (method)
		{
		case cp::SEPARABLE_METHOD::SWITCH_VH:
		{
			jointNonLocalMeansFilter(src, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma, powexp, patchnorm, borderType);
			jointNonLocalMeansFilter(dest, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma * alpha, powexp, patchnorm, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::DIRECT_VH:
		{
			jointNonLocalMeansFilter(src, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma, powexp, patchnorm, borderType);
			jointNonLocalMeansFilter(dest, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma, powexp, patchnorm, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::SWITCH_HV:
		{
			jointNonLocalMeansFilter(src, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma, powexp, patchnorm, borderType);
			jointNonLocalMeansFilter(dest, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma * alpha, powexp, patchnorm, borderType);
		}
		break;

		case cp::SEPARABLE_METHOD::DIRECT_HV:
		{
			jointNonLocalMeansFilter(src, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma, powexp, patchnorm, borderType);
			jointNonLocalMeansFilter(dest, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma, powexp, patchnorm, borderType);
		}
		break;
		default:
			break;
		}
	}

	void jointNonLocalMeansFilterSeparable(InputArray src, InputArray guide, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp, const int patchnorm, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		jointNonLocalMeansFilterSeparable(src, guide, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma, powexp, patchnorm, method, alpha, borderType);
	}

	void patchBilateralFilterSeparable(InputArray src, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		switch (method)
		{
		case cp::SEPARABLE_METHOD::SWITCH_VH:
		{
			patchBilateralFilter(src, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			jointPatchBilateralFilter(dest, src, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range * alpha, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::DIRECT_VH:
		{
			patchBilateralFilter(src, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			patchBilateralFilter(dest, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::SWITCH_HV:
		{
			patchBilateralFilter(src, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			jointPatchBilateralFilter(dest, src, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range * alpha, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;

		case cp::SEPARABLE_METHOD::DIRECT_HV:
		{
			patchBilateralFilter(src, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			patchBilateralFilter(dest, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;
		default:
			break;
		}
	}

	void patchBilateralFilterSeparable(InputArray src, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		patchBilateralFilterSeparable(src, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, method, alpha, borderType);
	}

	void jointPatchBilateralFilterSeparable(InputArray src, InputArray guide, OutputArray dest, const Size patchWindowSize, const Size kernelWindowSize, const double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		switch (method)
		{
		case cp::SEPARABLE_METHOD::SWITCH_VH:
		{
			jointPatchBilateralFilter(src, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			jointPatchBilateralFilter(dest, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range * alpha, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::DIRECT_VH:
		{
			jointPatchBilateralFilter(src, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			jointPatchBilateralFilter(dest, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;
		case cp::SEPARABLE_METHOD::SWITCH_HV:
		{
			jointPatchBilateralFilter(src, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			jointPatchBilateralFilter(dest, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range * alpha, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;

		case cp::SEPARABLE_METHOD::DIRECT_HV:
		{
			jointPatchBilateralFilter(src, guide, dest, patchWindowSize, Size(kernelWindowSize.width, 1), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
			jointPatchBilateralFilter(dest, guide, dest, patchWindowSize, Size(1, kernelWindowSize.height), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, borderType);
		}
		break;
		default:
			break;
		}
	}

	void jointPatchBilateralFilterSeparable(InputArray src, InputArray guide, OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma_range, const double powexp_range, const int patchnorm, const double sigma_space, const double powexp_space, SEPARABLE_METHOD method, const double alpha, const int borderType)
	{
		jointPatchBilateralFilterSeparable(src, guide, dest, Size(patchWindowSize, patchWindowSize), Size(kernelWindowSize, kernelWindowSize), sigma_range, powexp_range, patchnorm, sigma_space, powexp_space, method, alpha, borderType);
	}

#pragma endregion



#pragma region wnlm
	class WeightedJointNonlocalMeansFilterInvorker32f_SSE4 : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* weight;
		Mat* guide;
		Mat* dest;
		int templeteWindowSize;
		int searchWindowSize;

		float* w;
	public:

		WeightedJointNonlocalMeansFilterInvorker32f_SSE4(Mat& src_, Mat& weight_, Mat& guide_, Mat& dest_, int templeteWindowSize_, int searchWindowSize_, float* weight)
			: im(&src_), weight(&weight_), guide(&guide_), dest(&dest_), templeteWindowSize(templeteWindowSize_), searchWindowSize(searchWindowSize_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range& r) const
		{
			const int tr = templeteWindowSize >> 1;
			const int sr = searchWindowSize >> 1;
			const int cstep = guide->cols - templeteWindowSize;
			const int imstep = im->cols;
			int cng = (guide->rows - 2 * (tr + sr)) / dest->rows;

			const int tD = templeteWindowSize * templeteWindowSize;
			const float tdiv = 1.f / (float)(tD);//templete square div
			__m128 mtdiv = _mm_set1_ps(tdiv);
			const int CV_DECL_ALIGNED(16) v32f_absmask_[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			const __m128 v32f_absmask = _mm_load_ps((float*)v32f_absmask_);

			if (dest->channels() == 3 && cng == 3)
			{
				const int colorstep = guide->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				int CV_DECL_ALIGNED(16) buf[4];
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = guide->ptr<float>(sr + j) + sr;
					const float* sptr2_ = guide->ptr<float>(j);
					const float* vptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 4)
					{
						__m128 mr = _mm_setzero_ps();
						__m128 mg = _mm_setzero_ps();
						__m128 mb = _mm_setzero_ps();
						__m128 mtweight = _mm_setzero_ps();

						//search loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* vptr2 = vptr2_ + i;
						for (int l = searchWindowSize; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								float* t = (float*)tprt;
								float* s = (float*)(sptr + k);
								//colorstep
								__m128 me = _mm_setzero_ps();
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
									{
										// computing color L2 norm
										__m128 s0 = _mm_sub_ps(_mm_loadu_ps(s), _mm_loadu_ps(t));
										me = _mm_add_ps(me, _mm_and_ps(s0, v32f_absmask));

										s0 = _mm_sub_ps(_mm_loadu_ps(s + colorstep), _mm_loadu_ps(t + colorstep));
										me = _mm_add_ps(me, _mm_and_ps(s0, v32f_absmask));

										s0 = _mm_sub_ps(_mm_loadu_ps(s + colorstep2), _mm_loadu_ps(t + colorstep2));
										me = _mm_add_ps(me, _mm_and_ps(s0, v32f_absmask));

										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_mul_ps(mtdiv, me)));
								__m128 www = _mm_set_ps(w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);

								const float* ss = vptr2 + imstep * (tr + l) + (tr + k);
								float* wptr = weight->ptr<float>(j + tr + l) + (tr + k + i);
								www = _mm_mul_ps(www, _mm_loadu_ps(wptr));
								mg = _mm_add_ps(mg, _mm_mul_ps(www, _mm_loadu_ps(ss)));
								mb = _mm_add_ps(mb, _mm_mul_ps(www, _mm_loadu_ps(ss + colorstep)));
								mr = _mm_add_ps(mr, _mm_mul_ps(www, _mm_loadu_ps(ss + colorstep2)));
								mtweight = _mm_add_ps(mtweight, www);
							}
						}
						mb = _mm_div_ps(mb, mtweight);
						mg = _mm_div_ps(mg, mtweight);
						mr = _mm_div_ps(mr, mtweight);

						__m128 a = _mm_shuffle_ps(mr, mr, _MM_SHUFFLE(3, 0, 1, 2));
						__m128 b = _mm_shuffle_ps(mg, mg, _MM_SHUFFLE(1, 2, 3, 0));
						__m128 c = _mm_shuffle_ps(mb, mb, _MM_SHUFFLE(2, 3, 0, 1));

						_mm_stream_ps((d), _mm_blend_ps(_mm_blend_ps(b, a, 4), c, 2));
						_mm_stream_ps((d + 4), _mm_blend_ps(_mm_blend_ps(c, b, 4), a, 2));
						_mm_stream_ps((d + 8), _mm_blend_ps(_mm_blend_ps(a, c, 4), b, 2));

						d += 12;
					}//i
				}//j
			}
			else if (dest->channels() == 1 && cng == 3)
			{
				const int colorstep = guide->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				int CV_DECL_ALIGNED(16) buf[4];
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = guide->ptr<float>(sr + j) + sr;
					const float* sptr2_ = guide->ptr<float>(j);
					const float* vptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 4)
					{
						__m128 mvalue = _mm_setzero_ps();
						__m128 mtweight = _mm_setzero_ps();

						//search loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* vptr2 = vptr2_ + i;
						for (int l = searchWindowSize; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								float* t = (float*)tprt;
								float* s = (float*)(sptr + k);
								//colorstep
								__m128 me = _mm_setzero_ps();
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
									{
										// computing color L2 norm
										__m128 s0 = _mm_sub_ps(_mm_loadu_ps(s), _mm_loadu_ps(t));
										me = _mm_add_ps(me, _mm_and_ps(s0, v32f_absmask));

										s0 = _mm_sub_ps(_mm_loadu_ps(s + colorstep), _mm_loadu_ps(t + colorstep));
										me = _mm_add_ps(me, _mm_and_ps(s0, v32f_absmask));

										s0 = _mm_sub_ps(_mm_loadu_ps(s + colorstep2), _mm_loadu_ps(t + colorstep2));
										me = _mm_add_ps(me, _mm_and_ps(s0, v32f_absmask));

										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_mul_ps(mtdiv, me)));
								__m128 www = _mm_set_ps(w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);
								float* wptr = weight->ptr<float>(j + tr + l) + (tr + k + i);
								www = _mm_mul_ps(www, _mm_loadu_ps(wptr));
								mvalue = _mm_add_ps(mvalue, _mm_mul_ps(www, _mm_loadu_ps(vptr2 + imstep * (tr + l) + tr + k)));
								mtweight = _mm_add_ps(mtweight, www);
							}
						}
						_mm_stream_ps(d, _mm_div_ps(mvalue, mtweight));
						d += 4;
					}//i
				}//j
			}
			else if (dest->channels() == 3 && cng == 1)
			{

				const int colorstep = guide->size().area();
				const int colorstep2 = colorstep * 2;
				int CV_DECL_ALIGNED(16) buf[4];
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = guide->ptr<float>(sr + j) + sr;
					const float* sptr2_ = guide->ptr<float>(j);
					const float* vptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 4)
					{
						__m128 mr = _mm_setzero_ps();
						__m128 mg = _mm_setzero_ps();
						__m128 mb = _mm_setzero_ps();
						__m128 mtweight = _mm_setzero_ps();

						//search loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						const float* vptr2 = vptr2_ + i;
						for (int l = searchWindowSize; l--;)
						{
							const float* sptr = sptr2 + imstep * (l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								float* t = (float*)tprt;
								float* s = (float*)(sptr + k);
								//colorstep
								__m128 me = _mm_setzero_ps();
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
									{
										// computing color L2 norm
										__m128 ms = _mm_loadu_ps(s);
										__m128 mt = _mm_loadu_ps(t);
										mt = _mm_sub_ps(ms, mt);
										me = _mm_add_ps(me, _mm_and_ps(mt, v32f_absmask));

										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_mul_ps(mtdiv, me)));
								__m128 www = _mm_set_ps(w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);

								const float* ss = vptr2 + imstep * (tr + l) + (tr + k);
								float* wptr = weight->ptr<float>(j + tr + l) + (tr + k + i);
								www = _mm_mul_ps(www, _mm_loadu_ps(wptr));
								mg = _mm_add_ps(mg, _mm_mul_ps(www, _mm_loadu_ps(ss)));
								mb = _mm_add_ps(mb, _mm_mul_ps(www, _mm_loadu_ps(ss + colorstep)));
								mr = _mm_add_ps(mr, _mm_mul_ps(www, _mm_loadu_ps(ss + colorstep2)));
								mtweight = _mm_add_ps(mtweight, www);
							}
						}
						mb = _mm_div_ps(mb, mtweight);
						mg = _mm_div_ps(mg, mtweight);
						mr = _mm_div_ps(mr, mtweight);

						__m128 a = _mm_shuffle_ps(mr, mr, _MM_SHUFFLE(3, 0, 1, 2));
						__m128 b = _mm_shuffle_ps(mg, mg, _MM_SHUFFLE(1, 2, 3, 0));
						__m128 c = _mm_shuffle_ps(mb, mb, _MM_SHUFFLE(2, 3, 0, 1));

						_mm_stream_ps((d), _mm_blend_ps(_mm_blend_ps(b, a, 4), c, 2));
						_mm_stream_ps((d + 4), _mm_blend_ps(_mm_blend_ps(c, b, 4), a, 2));
						_mm_stream_ps((d + 8), _mm_blend_ps(_mm_blend_ps(a, c, 4), b, 2));

						d += 12;
					}//i
				}//j
			}
			else if (dest->channels() == 1 && cng == 1)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 4)
					{
						__m128 mvalue = _mm_setzero_ps();
						__m128 mtweight = _mm_setzero_ps();
						//search loop
						float* tprt = guide->ptr<float>(sr + j) + (sr + i);
						float* sptr2 = guide->ptr<float>(j) + (i);
						float* vptr2 = im->ptr<float>(j) + (i);

						for (int l = searchWindowSize; l--;)
						{
							float* sptr = sptr2 + imstep * (l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								__m128 me = _mm_setzero_ps();
								float* t = tprt;
								float* s = sptr + k;
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
									{
										// computing color L2 norm
										__m128 ms = _mm_loadu_ps(s);
										__m128 mt = _mm_loadu_ps(t);
										mt = _mm_sub_ps(ms, mt);
										me = _mm_add_ps(me, _mm_and_ps(mt, v32f_absmask));

										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_mul_ps(mtdiv, me)));
								__m128 www = _mm_set_ps(w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);
								float* wptr = weight->ptr<float>(j + tr + l) + (tr + k + i);
								//www=_mm_mul_ps(www,_mm_loadu_ps(wptr));
								mvalue = _mm_add_ps(mvalue, _mm_mul_ps(www, _mm_loadu_ps(vptr2 + imstep * (tr + l) + tr + k)));
								mtweight = _mm_add_ps(mtweight, www);
							}
						}
						//weight normalization
						_mm_stream_ps(d, _mm_div_ps(mvalue, mtweight));
						d += 4;
					}//i
				}//j
			}
		}
	};

	void weightedJointNonLocalMeansFilter(Mat& src, Mat& weightMap, Mat& guide, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int tr = templeteWindowSize >> 1;
		const int sr = searchWindowSize >> 1;
		const int bb = sr + tr;
		//	const int D = searchWindowSize*searchWindowSize;

		int dpad;
		int spad;
		//create large size image for bounding box;
		//if(src.depth()==CV_8U)
		//{
		//	dpad = (16- src.cols%16)%16;
		//	spad =  (16-(src.cols+2*bb)%16)%16;
		//}
		//else
		{
			dpad = (4 - src.cols % 4) % 4;
			spad = (4 - (src.cols + 2 * bb) % 4) % 4;
		}
		Mat dst = Mat::zeros(Size(src.cols + dpad, src.rows), dest.type());

		Mat im, gim, wim;
		copyMakeBorder(weightMap, wim, bb, bb, bb, bb + spad, cv::BORDER_DEFAULT);
		if (src.channels() == 1)
		{
			copyMakeBorder(src, im, bb, bb, bb, bb + spad, cv::BORDER_DEFAULT);
		}
		else if (src.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bb, bb, bb, bb + spad, cv::BORDER_DEFAULT);
			cvtColorBGR2PLANE(temp, im);
		}
		if (guide.channels() == 1)
		{
			copyMakeBorder(guide, gim, bb, bb, bb, bb + spad, cv::BORDER_DEFAULT);
		}
		else if (guide.channels() == 3)
		{
			Mat temp;
			copyMakeBorder(guide, temp, bb, bb, bb, bb + spad, cv::BORDER_DEFAULT);
			cvtColorBGR2PLANE(temp, gim);
		}

		//weight computation;
		vector<float> weight(256 * guide.channels());
		float* w = &weight[0];
		const double gauss_sd = (sigma == 0.0) ? h : sigma;
		double gauss_color_coeff = -(1.0 / (double)(guide.channels())) * (1.0 / (h * h));
		for (int i = 0; i < 256 * guide.channels(); i++)
		{
			double v = std::exp(max(i * i - 2.0 * gauss_sd * gauss_sd, 0.0) * gauss_color_coeff);
			w[i] = (float)v;
		}

		if (src.depth() == CV_8U)
		{
			/*JointNonlocalMeansFilterInvorker8u_SSE4 body(im,gim,dst,templeteWindowSize,searchWindowSize,w);
			cv::parallel_for_(Range(0, dst.rows), body);*/
			Mat imf, gimf, dstf;
			im.convertTo(imf, CV_32F);
			gim.convertTo(gimf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			WeightedJointNonlocalMeansFilterInvorker32f_SSE4 body(imf, wim, gimf, dstf, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_8U);
		}
		if (src.depth() == CV_16U)
		{
			Mat imf, gimf, dstf;
			im.convertTo(imf, CV_32F);
			gim.convertTo(gimf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			WeightedJointNonlocalMeansFilterInvorker32f_SSE4 body(imf, wim, gimf, dstf, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_16U);
		}
		if (src.depth() == CV_16S)
		{
			Mat imf, gimf, dstf;
			im.convertTo(imf, CV_32F);
			gim.convertTo(gimf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			WeightedJointNonlocalMeansFilterInvorker32f_SSE4 body(imf, wim, gimf, dstf, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_16S);
		}
		else if (src.depth() == CV_32F)
		{
			WeightedJointNonlocalMeansFilterInvorker32f_SSE4 body(im, wim, gim, dst, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dst.rows), body);
		}
		else if (src.depth() == CV_64F)
		{
			Mat imf, gimf, dstf;
			im.convertTo(imf, CV_32F);
			gim.convertTo(gimf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			WeightedJointNonlocalMeansFilterInvorker32f_SSE4 body(imf, wim, gimf, dstf, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_64F);
		}
		Mat(dst(Rect(0, 0, dest.cols, dest.rows))).copyTo(dest);
	}
#pragma endregion
}