#include "nonLocalMeans.hpp"
#include "jointNonLocalMeans.hpp"
#include "color.hpp"
#include "blend.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	template <class srcType>
	class NonlocalMeansFilterNonzeroBaseInvorker_ : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int templeteWindowSizeX;
		int templeteWindowSizeY;
		int searchWindowSizeX;
		int searchWindowSizeY;

		float* w;
	public:

		NonlocalMeansFilterNonzeroBaseInvorker_(Mat& src_, Mat& dest_, int templeteWindowSizeX_, int templeteWindowSizeY_, int searchWindowSizeX_, int searchWindowSizeY_, float* weight)
			: im(&src_), dest(&dest_), templeteWindowSizeX(templeteWindowSizeX_), searchWindowSizeX(searchWindowSizeX_), templeteWindowSizeY(templeteWindowSizeY_), searchWindowSizeY(searchWindowSizeY_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range &r) const
		{
			const int tr_x = templeteWindowSizeX >> 1;
			const int sr_x = searchWindowSizeX >> 1;
			const int tr_y = templeteWindowSizeY >> 1;
			const int sr_y = searchWindowSizeY >> 1;
			const int cstep = (im->cols - templeteWindowSizeX)*im->channels();
			const int imstep = im->cols*im->channels();

			const int tD = templeteWindowSizeX*templeteWindowSizeY;
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
						float wmax = 0.f;
						//search loop
						const srcType* tprt = im->ptr<srcType>(sr_y + j) + 3 * (sr_x + i);
						const srcType* sptr2 = im->ptr<srcType>(j) +3 * (i);

						for (int l = searchWindowSizeY; l--;)
						{
							const srcType* sptr = sptr2 + imstep*(l);
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
										//e += abs(s[0]-t[0])+abs(s[1]-t[1])+abs(s[2]-t[2]);//L2 norm

										e += (s[0] - t[0])*(s[0] - t[0]) + (s[1] - t[1])*(s[1] - t[1]) + (s[2] - t[2])*(s[2] - t[2]);//L2 norm
										s += 3, t += 3;
									}
									t += cstep;
									s += cstep;
								}

								const int ediv = cvRound(e*tdiv);
								float www = w[ediv];

								if (k == sr_x&&l == sr_y){ ; }
								else wmax = max(wmax, www);
								const srcType* ss = sptr2 + imstep*(tr_y + l) + 3 * (tr_x + k);
								r += ss[0] * www;
								g += ss[1] * www;
								b += ss[2] * www;
								//get weighted Euclidean distance
								tweight += www;
							}
						}

						const srcType* ss = sptr2 + imstep*(tr_y + sr_y) + 3 * (tr_x + sr_x);
						r -= ss[0];
						g -= ss[1];
						b -= ss[2];
						tweight -= 1.0;

						r += wmax*ss[0];
						g += wmax*ss[1];
						b += wmax*ss[2];
						tweight += wmax;

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
						float wmax = 0.f;
						//search loop
						srcType* tprt = im->ptr<srcType>(sr_y + j) + (sr_x + i);
						srcType* sptr2 = im->ptr<srcType>(j) +(i);

						for (int l = searchWindowSizeY; l--;)
						{
							srcType* sptr = sptr2 + imstep*(l);
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
										//e += abs(*s-*t);
										e += (*s - *t)*(*s - *t);
										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								const int ediv = cvRound(e*tdiv);
								float www = w[ediv];
								if (k == sr_x&&l == sr_y){ ; }
								else wmax = max(wmax, www);
								value += sptr2[imstep*(tr_y + l) + tr_x + k] * www;
								//get weighted Euclidean distance
								tweight += www;
							}
						}
						const srcType* ss = sptr2 + imstep*(tr_y + sr_y) + (tr_x + sr_x);
						value -= ss[0];
						tweight -= 1.0;

						value += wmax*ss[0];
						tweight += wmax;

						//weight normalization
						*(d++) = saturate_cast<srcType>(value / tweight);
					}//i
				}//j
			}
		}
	};

	template <class srcType>
	class NonlocalMeansFilterBaseInvorker_ : public cv::ParallelLoopBody
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

		NonlocalMeansFilterBaseInvorker_(Mat& src_, Mat& dest_, int templeteWindowSizeX_, int templeteWindowSizeY_, int searchWindowSizeX_, int searchWindowSizeY_, float* weight)
			: im(&src_), dest(&dest_), templeteWindowSizeX(templeteWindowSizeX_), searchWindowSizeX(searchWindowSizeX_), templeteWindowSizeY(templeteWindowSizeY_), searchWindowSizeY(searchWindowSizeY_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range &r) const
		{
			const int tr_x = templeteWindowSizeX >> 1;
			const int sr_x = searchWindowSizeX >> 1;
			const int tr_y = templeteWindowSizeY >> 1;
			const int sr_y = searchWindowSizeY >> 1;
			const int cstep = (im->cols - templeteWindowSizeX)*im->channels();
			const int imstep = im->cols*im->channels();

			const int tD = templeteWindowSizeX*templeteWindowSizeY;
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
						const srcType* sptr2 = im->ptr<srcType>(j) +3 * (i);

						for (int l = searchWindowSizeY; l--;)
						{
							const srcType* sptr = sptr2 + imstep*(l);
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
								const int ediv = cvRound(e*tdiv);
								float www = w[ediv];

								const srcType* ss = sptr2 + imstep*(tr_y + l) + 3 * (tr_x + k);
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
						srcType* sptr2 = im->ptr<srcType>(j) +(i);

						for (int l = searchWindowSizeY; l--;)
						{
							srcType* sptr = sptr2 + imstep*(l);
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
								const int ediv = cvRound(e*tdiv);
								float www = w[ediv];
								value += sptr2[imstep*(tr_y + l) + tr_x + k] * www;
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

	class NonlocalMeansFilterInvorker32f_SSE4 : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int templeteWindowSizeX;
		int templeteWindowSizeY;
		int searchWindowSizeX;
		int searchWindowSizeY;

		float* w;
	public:

		NonlocalMeansFilterInvorker32f_SSE4(Mat& src_, Mat& dest_, int templeteWindowSizeX_, int templeteWindowSizeY_, int searchWindowSizeX_, int searchWindowSizeY_, float* weight)
			: im(&src_), dest(&dest_), templeteWindowSizeX(templeteWindowSizeX_), templeteWindowSizeY(templeteWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range &r) const
		{
			const int tr_x = templeteWindowSizeX >> 1;
			const int sr_x = searchWindowSizeX >> 1;
			const int tr_y = templeteWindowSizeY >> 1;
			const int sr_y = searchWindowSizeY >> 1;
			const int cstep = im->cols - templeteWindowSizeX;
			const int imstep = im->cols;

			const int tD = templeteWindowSizeX*templeteWindowSizeY;
			const float tdiv = 1.f / (float)(tD);//templete square div
			__m128 mtdiv = _mm_set1_ps(tdiv);
			const int CV_DECL_ALIGNED(16) v32f_absmask_[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			const __m128 v32f_absmask = _mm_load_ps((float*)v32f_absmask_);

			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				int CV_DECL_ALIGNED(16) buf[4];
				for (int j = r.start; j < r.end; j++)
				{
					float* d = dest->ptr<float>(j);
					const float* tprt_ = im->ptr<float>(sr_y + j) + sr_x;
					const float* sptr2_ = im->ptr<float>(j);
					for (int i = 0; i < dest->cols; i += 4)
					{
						__m128 mr = _mm_setzero_ps();
						__m128 mg = _mm_setzero_ps();
						__m128 mb = _mm_setzero_ps();
						__m128 mtweight = _mm_setzero_ps();

						//search loop
						const float* tprt = tprt_ + i;
						const float* sptr2 = sptr2_ + i;
						for (int l = searchWindowSizeY; l--;)
						{
							const float* sptr = sptr2 + imstep*(l);
							for (int k = searchWindowSizeX; k--;)
							{
								//templete loop
								float* t = (float*)tprt;
								float* s = (float*)(sptr + k);
								//colorstep
								__m128 me = _mm_setzero_ps();
								for (int n = templeteWindowSizeY; n--;)
								{
									for (int m = templeteWindowSizeX; m--;)
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

								const float* ss = sptr2 + imstep*(tr_y + l) + (tr_x + k);
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
			else if (dest->channels() == 1)
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
						float* tprt = im->ptr<float>(sr_y + j) + (sr_x + i);
						float* sptr2 = im->ptr<float>(j) +(i);

						for (int l = searchWindowSizeY; l--;)
						{
							float* sptr = sptr2 + imstep*(l);
							for (int k = searchWindowSizeX; k--;)
							{
								//templete loop
								__m128 me = _mm_setzero_ps();
								float* t = tprt;
								float* s = sptr + k;
								for (int n = templeteWindowSizeY; n--;)
								{
									for (int m = templeteWindowSizeX; m--;)
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
								mvalue = _mm_add_ps(mvalue, _mm_mul_ps(www, _mm_loadu_ps(sptr2 + imstep*(tr_y + l) + tr_x + k)));
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

	class NonlocalMeansFilterInvorker8u_SSE4 : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* dest;
		int templeteWindowSizeX;
		int templeteWindowSizeY;
		int searchWindowSizeX;
		int searchWindowSizeY;

		float* w;
	public:

		NonlocalMeansFilterInvorker8u_SSE4(Mat& src_, Mat& dest_, int templeteWindowSizeX_, int templeteWindowSizeY_, int searchWindowSizeX_, int searchWindowSizeY_, float* weight)
			: im(&src_), dest(&dest_), templeteWindowSizeX(templeteWindowSizeX_), templeteWindowSizeY(templeteWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range &r) const
		{
			const int tr_x = templeteWindowSizeX >> 1;
			const int sr_x = searchWindowSizeX >> 1;
			const int tr_y = templeteWindowSizeY >> 1;
			const int sr_y = searchWindowSizeY >> 1;
			const int cstep = im->cols - templeteWindowSizeX;
			const int imstep = im->cols;

			const int tD = templeteWindowSizeX*templeteWindowSizeY;
			const float tdiv = 1.f / (float)(tD);//templete square div
			__m128 mtdiv = _mm_set1_ps(tdiv);
			if (dest->channels() == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				int CV_DECL_ALIGNED(16) buf[16];
				const __m128i zero = _mm_setzero_si128();
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = im->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = im->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 16)
					{
						__m128 mr0 = _mm_setzero_ps();
						__m128 mr1 = _mm_setzero_ps();
						__m128 mr2 = _mm_setzero_ps();
						__m128 mr3 = _mm_setzero_ps();

						__m128 mg0 = _mm_setzero_ps();
						__m128 mg1 = _mm_setzero_ps();
						__m128 mg2 = _mm_setzero_ps();
						__m128 mg3 = _mm_setzero_ps();

						__m128 mb0 = _mm_setzero_ps();
						__m128 mb1 = _mm_setzero_ps();
						__m128 mb2 = _mm_setzero_ps();
						__m128 mb3 = _mm_setzero_ps();

						__m128 mtweight0 = _mm_setzero_ps();
						__m128 mtweight1 = _mm_setzero_ps();
						__m128 mtweight2 = _mm_setzero_ps();
						__m128 mtweight3 = _mm_setzero_ps();

						//search loop
						const uchar* tprt = tprt_ + i;
						const uchar* sptr2 = sptr2_ + i;
						for (int l = searchWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep*(l);
							for (int k = searchWindowSizeX; k--;)
							{
								uchar* t = (uchar*)tprt;
								uchar* s = (uchar*)(sptr + k);
								//colorstep
								__m128i me0 = _mm_setzero_si128();
								__m128i me1 = _mm_setzero_si128();

								for (int n = templeteWindowSizeY; n--;)
								{
									for (int m = templeteWindowSizeX; m--;)
									{
										// computing color L2 norm
										__m128i ms = _mm_loadu_si128((__m128i*)s);
										__m128i mt = _mm_loadu_si128((__m128i*)t);
										ms = _mm_add_epi8(_mm_subs_epu8(ms, mt), _mm_subs_epu8(mt, ms));

										__m128i s0 = _mm_unpacklo_epi8(ms, zero);
										__m128i s1 = _mm_unpackhi_epi8(ms, zero);

										me0 = _mm_adds_epu16(me0, s0);
										me1 = _mm_adds_epu16(me1, s1);

										ms = _mm_loadu_si128((__m128i*)(s + colorstep));
										mt = _mm_loadu_si128((__m128i*)(t + colorstep));
										ms = _mm_add_epi8(_mm_subs_epu8(ms, mt), _mm_subs_epu8(mt, ms));

										s0 = _mm_unpacklo_epi8(ms, zero);
										s1 = _mm_unpackhi_epi8(ms, zero);

										me0 = _mm_adds_epu16(me0, s0);
										me1 = _mm_adds_epu16(me1, s1);

										ms = _mm_loadu_si128((__m128i*)(s + colorstep2));
										mt = _mm_loadu_si128((__m128i*)(t + colorstep2));
										ms = _mm_add_epi8(_mm_subs_epu8(ms, mt), _mm_subs_epu8(mt, ms));

										s0 = _mm_unpacklo_epi8(ms, zero);
										s1 = _mm_unpackhi_epi8(ms, zero);

										me0 = _mm_adds_epu16(me0, s0);
										me1 = _mm_adds_epu16(me1, s1);

										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								__m128i mme0 = _mm_unpacklo_epi16(me0, zero);
								__m128i mme1 = _mm_unpackhi_epi16(me0, zero);
								__m128i mme2 = _mm_unpacklo_epi16(me1, zero);
								__m128i mme3 = _mm_unpackhi_epi16(me1, zero);

								_mm_store_si128((__m128i*)(buf), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme0))));
								_mm_store_si128((__m128i*)(buf + 4), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme1))));
								_mm_store_si128((__m128i*)(buf + 8), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme2))));
								_mm_store_si128((__m128i*)(buf + 12), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme3))));

								const uchar* ss = sptr2 + imstep*(tr_y + l) + (tr_x + k);
								const __m128i sc0 = _mm_loadu_si128((__m128i*)(ss));
								const __m128i sc1 = _mm_loadu_si128((__m128i*)(ss + colorstep));
								const __m128i sc2 = _mm_loadu_si128((__m128i*)(ss + colorstep2));

								__m128i sr0 = _mm_unpacklo_epi8(sc0, zero);
								__m128i sr1 = _mm_unpackhi_epi16(sr0, zero);
								sr0 = _mm_unpacklo_epi16(sr0, zero);
								__m128i sr2 = _mm_unpackhi_epi8(sc0, zero);
								__m128i sr3 = _mm_unpackhi_epi16(sr2, zero);
								sr2 = _mm_unpacklo_epi16(sr2, zero);

								__m128i sg0 = _mm_unpacklo_epi8(sc1, zero);
								__m128i sg1 = _mm_unpackhi_epi16(sg0, zero);
								sg0 = _mm_unpacklo_epi16(sg0, zero);
								__m128i sg2 = _mm_unpackhi_epi8(sc1, zero);
								__m128i sg3 = _mm_unpackhi_epi16(sg2, zero);
								sg2 = _mm_unpacklo_epi16(sg2, zero);

								__m128i sb0 = _mm_unpacklo_epi8(sc2, zero);
								__m128i sb1 = _mm_unpackhi_epi16(sb0, zero);
								sb0 = _mm_unpacklo_epi16(sb0, zero);
								__m128i sb2 = _mm_unpackhi_epi8(sc2, zero);
								__m128i sb3 = _mm_unpackhi_epi16(sb2, zero);
								sb2 = _mm_unpacklo_epi16(sb2, zero);

								__m128 www = _mm_set_ps(w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);
								mr0 = _mm_add_ps(mr0, _mm_mul_ps(www, _mm_cvtepi32_ps(sr0)));
								mg0 = _mm_add_ps(mg0, _mm_mul_ps(www, _mm_cvtepi32_ps(sg0)));
								mb0 = _mm_add_ps(mb0, _mm_mul_ps(www, _mm_cvtepi32_ps(sb0)));
								mtweight0 = _mm_add_ps(mtweight0, www);

								www = _mm_set_ps(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]]);
								mr1 = _mm_add_ps(mr1, _mm_mul_ps(www, _mm_cvtepi32_ps(sr1)));
								mg1 = _mm_add_ps(mg1, _mm_mul_ps(www, _mm_cvtepi32_ps(sg1)));
								mb1 = _mm_add_ps(mb1, _mm_mul_ps(www, _mm_cvtepi32_ps(sb1)));
								mtweight1 = _mm_add_ps(mtweight1, www);

								www = _mm_set_ps(w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]]);
								mr2 = _mm_add_ps(mr2, _mm_mul_ps(www, _mm_cvtepi32_ps(sr2)));
								mg2 = _mm_add_ps(mg2, _mm_mul_ps(www, _mm_cvtepi32_ps(sg2)));
								mb2 = _mm_add_ps(mb2, _mm_mul_ps(www, _mm_cvtepi32_ps(sb2)));
								mtweight2 = _mm_add_ps(mtweight2, www);

								www = _mm_set_ps(w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]]);
								mr3 = _mm_add_ps(mr3, _mm_mul_ps(www, _mm_cvtepi32_ps(sr3)));
								mg3 = _mm_add_ps(mg3, _mm_mul_ps(www, _mm_cvtepi32_ps(sg3)));
								mb3 = _mm_add_ps(mb3, _mm_mul_ps(www, _mm_cvtepi32_ps(sb3)));
								mtweight3 = _mm_add_ps(mtweight3, www);
							}
						}

						mr0 = _mm_div_ps(mr0, mtweight0);
						mr1 = _mm_div_ps(mr1, mtweight1);
						mr2 = _mm_div_ps(mr2, mtweight2);
						mr3 = _mm_div_ps(mr3, mtweight3);
						__m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(mr0), _mm_cvtps_epi32(mr1)), _mm_packs_epi32(_mm_cvtps_epi32(mr2), _mm_cvtps_epi32(mr3)));
						mg0 = _mm_div_ps(mg0, mtweight0);
						mg1 = _mm_div_ps(mg1, mtweight1);
						mg2 = _mm_div_ps(mg2, mtweight2);
						mg3 = _mm_div_ps(mg3, mtweight3);
						__m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(mg0), _mm_cvtps_epi32(mg1)), _mm_packs_epi32(_mm_cvtps_epi32(mg2), _mm_cvtps_epi32(mg3)));
						mb0 = _mm_div_ps(mb0, mtweight0);
						mb1 = _mm_div_ps(mb1, mtweight1);
						mb2 = _mm_div_ps(mb2, mtweight2);
						mb3 = _mm_div_ps(mb3, mtweight3);
						__m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(mb0), _mm_cvtps_epi32(mb1)), _mm_packs_epi32(_mm_cvtps_epi32(mb2), _mm_cvtps_epi32(mb3)));

						//sse4///
						const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
						const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
						const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

						const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
						const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

						a = _mm_shuffle_epi8(a, mask1);
						b = _mm_shuffle_epi8(b, mask2);
						c = _mm_shuffle_epi8(c, mask3);
						_mm_stream_si128((__m128i*)(d), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
						_mm_stream_si128((__m128i*)(d + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
						_mm_stream_si128((__m128i*)(d + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));
						d += 48;
					}//i
				}//j
			}
			else if (dest->channels() == 1)
			{
				int CV_DECL_ALIGNED(16) buf[16];
				const __m128i zero = _mm_setzero_si128();
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					for (int i = 0; i < dest->cols; i += 16)
					{
						__m128 mvalue0 = _mm_setzero_ps();
						__m128 mvalue1 = _mm_setzero_ps();
						__m128 mvalue2 = _mm_setzero_ps();
						__m128 mvalue3 = _mm_setzero_ps();
						__m128 mtweight0 = _mm_setzero_ps();
						__m128 mtweight1 = _mm_setzero_ps();
						__m128 mtweight2 = _mm_setzero_ps();
						__m128 mtweight3 = _mm_setzero_ps();

						//search loop
						uchar* tprt = im->ptr<uchar>(sr_y + j) + (sr_x + i);
						uchar* sptr2 = im->ptr<uchar>(j) +(i);
						for (int l = searchWindowSizeY; l--;)
						{
							uchar* sptr = sptr2 + imstep*(l);
							for (int k = searchWindowSizeX; k--;)
							{
								//templete loop
								__m128i me0 = _mm_setzero_si128();
								__m128i me1 = _mm_setzero_si128();

								uchar* t = tprt;
								uchar* s = sptr + k;
								for (int n = templeteWindowSizeY; n--;)
								{
									for (int m = templeteWindowSizeX; m--;)
									{
										// computing color L2 norm
										__m128i ms = _mm_loadu_si128((__m128i*)s);
										__m128i mt = _mm_loadu_si128((__m128i*)t);
										ms = _mm_add_epi8(_mm_subs_epu8(ms, mt), _mm_subs_epu8(mt, ms));

										__m128i s0 = _mm_unpacklo_epi8(ms, zero);
										__m128i s1 = _mm_unpackhi_epi8(ms, zero);

										me0 = _mm_adds_epu16(me0, s0);
										me1 = _mm_adds_epu16(me1, s1);

										s++, t++;
									}
									t += cstep;
									s += cstep;
								}
								__m128i mme0 = _mm_unpacklo_epi16(me0, zero);
								__m128i mme1 = _mm_unpackhi_epi16(me0, zero);
								__m128i mme2 = _mm_unpacklo_epi16(me1, zero);
								__m128i mme3 = _mm_unpackhi_epi16(me1, zero);

								_mm_store_si128((__m128i*)(buf), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme0))));
								_mm_store_si128((__m128i*)(buf + 4), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme1))));
								_mm_store_si128((__m128i*)(buf + 8), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme2))));
								_mm_store_si128((__m128i*)(buf + 12), _mm_cvtps_epi32(_mm_mul_ps(mtdiv, _mm_cvtepi32_ps(mme3))));

								const __m128i sc = _mm_loadu_si128((__m128i*)(sptr2 + imstep*(tr_y + l) + tr_x + k));
								__m128i s0 = _mm_unpacklo_epi8(sc, zero);
								__m128i s1 = _mm_unpackhi_epi16(s0, zero);
								s0 = _mm_unpacklo_epi16(s0, zero);
								__m128i s2 = _mm_unpackhi_epi8(sc, zero);
								__m128i s3 = _mm_unpackhi_epi16(s2, zero);
								s2 = _mm_unpacklo_epi16(s2, zero);

								__m128 www = _mm_set_ps(w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);
								mvalue0 = _mm_add_ps(mvalue0, _mm_mul_ps(www, _mm_cvtepi32_ps(s0)));
								mtweight0 = _mm_add_ps(mtweight0, www);

								www = _mm_set_ps(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]]);
								mvalue1 = _mm_add_ps(mvalue1, _mm_mul_ps(www, _mm_cvtepi32_ps(s1)));
								mtweight1 = _mm_add_ps(mtweight1, www);

								www = _mm_set_ps(w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]]);
								mvalue2 = _mm_add_ps(mvalue2, _mm_mul_ps(www, _mm_cvtepi32_ps(s2)));
								mtweight2 = _mm_add_ps(mtweight2, www);

								www = _mm_set_ps(w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]]);
								mvalue3 = _mm_add_ps(mvalue3, _mm_mul_ps(www, _mm_cvtepi32_ps(s3)));
								mtweight3 = _mm_add_ps(mtweight3, www);
							}
						}
						//weight normalization
						_mm_stream_si128((__m128i*)(d), _mm_packus_epi16(
							_mm_packs_epi32(_mm_cvtps_epi32(_mm_div_ps(mvalue0, mtweight0)), _mm_cvtps_epi32(_mm_div_ps(mvalue1, mtweight1))),
							_mm_packs_epi32(_mm_cvtps_epi32(_mm_div_ps(mvalue2, mtweight2)), _mm_cvtps_epi32(_mm_div_ps(mvalue3, mtweight3)))));
						d += 16;
					}//i
				}//j
			}
		}
	};


	void nonLocalMeansFilterBase(Mat& src, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int borderType)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int bbx = (templeteWindowSize.width >> 1) + (searchWindowSize.width >> 1);
		const int bby = (templeteWindowSize.height >> 1) + (searchWindowSize.height >> 1);

		//create large size image for bounding box;
		Mat im;
		copyMakeBorder(src, im, bby, bby, bbx, bbx, borderType);

		vector<float> weight(256 * 256 * src.channels());
		float* w = &weight[0];
		const double gauss_sd = (sigma == 0.0) ? h : sigma;
		double gauss_color_coeff = -(1.0 / (double)(src.channels()))*(1.0 / (h*h));
		for (int i = 0; i < 256 * 256 * src.channels(); i++)
		{
			double v = std::exp(max(i - 2.0*gauss_sd*gauss_sd, 0.0)*gauss_color_coeff);
			w[i] = (float)v;
		}

		if (src.depth() == CV_8U)
		{
			//NonlocalMeansFilterBaseInvorker_<uchar> body(im,dest,templeteWindowSize,searchWindowSize,w);
			NonlocalMeansFilterNonzeroBaseInvorker_<uchar> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		if (src.depth() == CV_16U)
		{
			NonlocalMeansFilterBaseInvorker_<ushort> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		if (src.depth() == CV_16S)
		{
			NonlocalMeansFilterBaseInvorker_<short> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		else if (src.depth() == CV_32F)
		{
			NonlocalMeansFilterBaseInvorker_<float> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		else if (src.depth() == CV_64F)
		{
			NonlocalMeansFilterBaseInvorker_<double> body(im, dest, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
	}

	void nonLocalMeansFilter_SSE(Mat& src, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int borderType)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int bbx = (templeteWindowSize.width >> 1) + (searchWindowSize.width >> 1);
		const int bby = (templeteWindowSize.height >> 1) + (searchWindowSize.height >> 1);
		//	const int D = searchWindowSize*searchWindowSize;

		int dpad;
		int spad;
		//create large size image for bounding box;
		if (src.depth() == CV_8U)
		{
			dpad = (16 - src.cols % 16) % 16;
			spad = (16 - (src.cols + 2 * bbx) % 16) % 16;
		}
		else
		{
			dpad = (4 - src.cols % 4) % 4;
			spad = (4 - (src.cols + 2 * bbx) % 4) % 4;
		}
		Mat dst = Mat::zeros(Size(src.cols + dpad, src.rows), dest.type());

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

		//weight computation;
		vector<float> weight(256 * src.channels());
		weight.resize(256);
		float* w = &weight[0];
		const double gauss_sd = (sigma == 0.0) ? h : sigma;
		double gauss_color_coeff = -(1.0 / (double)(src.channels()))*(1.0 / (h*h));
		for (int i = 0; i < 256 * src.channels(); i++)
		{
			double v = std::exp(max(i*i - 2.0*gauss_sd*gauss_sd, 0.0)*gauss_color_coeff);
			w[i] = (float)v;
		}

		if (src.depth() == CV_8U)
		{
			NonlocalMeansFilterInvorker8u_SSE4 body(im, dst, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
		}
		if (src.depth() == CV_16U)
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			NonlocalMeansFilterInvorker32f_SSE4 body(imf, dstf, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_16U);
		}
		if (src.depth() == CV_16S)
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			NonlocalMeansFilterInvorker32f_SSE4 body(imf, dstf, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_16S);
		}
		else if (src.depth() == CV_32F)
		{
			NonlocalMeansFilterInvorker32f_SSE4 body(im, dst, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
		}
		else if (src.depth() == CV_64F)
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			NonlocalMeansFilterInvorker32f_SSE4 body(imf, dstf, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_64F);
		}

		Mat(dst(Rect(0, 0, dest.cols, dest.rows))).copyTo(dest);
	}

	void nonLocalMeansFilter_SP(Mat& src, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int borderType)
	{
		nonLocalMeansFilter_SSE(src, dest, templeteWindowSize, Size(searchWindowSize.width, 1), h, sigma, borderType);
		nonLocalMeansFilter_SSE(dest, dest, templeteWindowSize, Size(1, searchWindowSize.height), h, sigma, borderType);
	}

	void nonLocalMeansFilter(Mat& src, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int method, int borderType)
	{
		CV_Assert(FILTER_CIRCLE);
		if (sigma < 1.0) sigma = h;

		if (method == FILTER_SLOWEST)
		{
			nonLocalMeansFilterBase(src, dest, templeteWindowSize, searchWindowSize, h, sigma, borderType);
		}
		else if (method == FILTER_RECTANGLE || method == FILTER_DEFAULT)
		{
			nonLocalMeansFilter_SSE(src, dest, templeteWindowSize, searchWindowSize, h, sigma, borderType);
		}
		else if (method == FILTER_SEPARABLE)
		{
			nonLocalMeansFilter_SP(src, dest, templeteWindowSize, searchWindowSize, h, sigma, borderType);
		}
	}

	void nonLocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma, int method, int borderType)
	{
		nonLocalMeansFilter(src, dest, Size(templeteWindowSize, templeteWindowSize), Size(searchWindowSize, searchWindowSize), h, sigma, method, borderType);
	}

	void separableNonLocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma, double alpha, int method, int borderType)
	{
		separableNonLocalMeansFilter(src, dest, Size(templeteWindowSize, templeteWindowSize), Size(searchWindowSize, searchWindowSize), h, sigma, alpha, method, borderType);
	}


	void separableNonLocalMeansFilter(Mat& src, Mat& dst, Size templeteSize, Size kernelSize, double h, double sigma_offset, double alpha, int method, int borderType)
	{
		if (method == DUAL_KERNEL_HV)
		{
			nonLocalMeansFilter(src, dst, templeteSize, Size(kernelSize.width, 1), h, sigma_offset, method, borderType);
			jointNonLocalMeansFilter(dst, src, dst, templeteSize, Size(1, kernelSize.height), h*alpha, sigma_offset, method, borderType);
		}
		else if (method == DUAL_KERNEL_VH)
		{
			nonLocalMeansFilter(src, dst, templeteSize, Size(1, kernelSize.height), h, sigma_offset, method, borderType);
			jointNonLocalMeansFilter(dst, src, dst, templeteSize, Size(kernelSize.width, 1), h*alpha, sigma_offset, method, borderType);
		}
		else if (method == DUAL_KERNEL_HVVH)
		{
			nonLocalMeansFilter(src, dst, templeteSize, Size(kernelSize.width, 1), h, sigma_offset, method, borderType);
			jointNonLocalMeansFilter(dst, src, dst, templeteSize, Size(1, kernelSize.height), h*alpha, sigma_offset, method, borderType);
			Mat dst2;
			nonLocalMeansFilter(src, dst2, templeteSize, Size(1, kernelSize.height), h, sigma_offset, method, borderType);
			jointNonLocalMeansFilter(dst2, src, dst2, templeteSize, Size(kernelSize.width, 1), h*alpha, sigma_offset, method, borderType);
			alphaBlend(dst, dst2, 0.5, dst);
		}
		/*else if (method==DUAL_KERNEL_CROSS)
		{
		bilateralFilter_direction_8u(src, dst, kernelSize, sigma_color, sigma_space, borderType, 1, true);
		jointBilateralFilter_direction_8u(dst, src, dst, kernelSize, sigma_color*alpha, sigma_space, borderType, -1, true);
		}
		else if (method==DUAL_KERNEL_CROSSCROSS)
		{
		bilateralFilter_direction_8u(src, dst, kernelSize, sigma_color, sigma_space, borderType, 1, true);
		jointBilateralFilter_direction_8u(dst, src, dst, kernelSize, sigma_color*alpha, sigma_space, borderType, -1,true);

		Mat dst2(src.size(),src.type());
		bilateralFilter_direction_8u(src, dst2, kernelSize, sigma_color, sigma_space, borderType, -1, true);
		jointBilateralFilter_direction_8u(dst2, src, dst2, kernelSize, sigma_color*alpha, sigma_space, borderType, 1, true);

		alphaBlend(dst,dst2,0.5,dst);
		}*/
	}


	
	void epsillonFilter(Mat& src, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, int borderType)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int bbx = (templeteWindowSize.width >> 1) + (searchWindowSize.width >> 1);
		const int bby = (templeteWindowSize.height >> 1) + (searchWindowSize.height >> 1);
		//	const int D = searchWindowSize*searchWindowSize;

		int dpad;
		int spad;
		//create large size image for bounding box;
		if (src.depth() == CV_8U)
		{
			dpad = (16 - src.cols % 16) % 16;
			spad = (16 - (src.cols + 2 * bbx) % 16) % 16;
		}
		else
		{
			dpad = (4 - src.cols % 4) % 4;
			spad = (4 - (src.cols + 2 * bbx) % 4) % 4;
		}
		Mat dst = Mat::zeros(Size(src.cols + dpad, src.rows), dest.type());

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

		//weight computation;
		vector<float> weight(256 * src.channels());
		weight.resize(256);
		float* w = &weight[0];
		
		double gauss_color_coeff = -(1.0 / (double)(src.channels()))*(1.0 / (h*h));
		for (int i = 0; i < 256 * src.channels(); i++)
		{
			double v = (i<=h) ? 1.0:0.0;
			w[i] = (float)v;
		}

		if (src.depth() == CV_8U)
		{
			NonlocalMeansFilterInvorker8u_SSE4 body(im, dst, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
		}
		if (src.depth() == CV_16U)
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			NonlocalMeansFilterInvorker32f_SSE4 body(imf, dstf, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_16U);
		}
		if (src.depth() == CV_16S)
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			NonlocalMeansFilterInvorker32f_SSE4 body(imf, dstf, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_16S);
		}
		else if (src.depth() == CV_32F)
		{
			NonlocalMeansFilterInvorker32f_SSE4 body(im, dst, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
		}
		else if (src.depth() == CV_64F)
		{
			Mat imf, dstf;
			im.convertTo(imf, CV_32F);
			dst.convertTo(dstf, CV_32F);
			NonlocalMeansFilterInvorker32f_SSE4 body(imf, dstf, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst, CV_64F);
		}

		Mat(dst(Rect(0, 0, dest.cols, dest.rows))).copyTo(dest);
	}
}