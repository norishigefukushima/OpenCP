#include "jointNonLocalMeans.hpp"
#include "color.hpp"

using namespace std;
using namespace cv;

namespace cp
{

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
		virtual void operator()(const cv::Range &r) const
		{
			const int tr = templeteWindowSize >> 1;
			const int sr = searchWindowSize >> 1;
			const int cstep = guide->cols - templeteWindowSize;
			const int imstep = im->cols;
			int cng = (guide->rows - 2 * (tr + sr)) / dest->rows;

			const int tD = templeteWindowSize*templeteWindowSize;
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
							const float* sptr = sptr2 + imstep*(l);
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

								const float* ss = vptr2 + imstep*(tr + l) + (tr + k);
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
							const float* sptr = sptr2 + imstep*(l);
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
								mvalue = _mm_add_ps(mvalue, _mm_mul_ps(www, _mm_loadu_ps(vptr2 + imstep*(tr + l) + tr + k)));
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
							const float* sptr = sptr2 + imstep*(l);
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

								const float* ss = vptr2 + imstep*(tr + l) + (tr + k);
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
						float* sptr2 = guide->ptr<float>(j) +(i);
						float* vptr2 = im->ptr<float>(j) +(i);

						for (int l = searchWindowSize; l--;)
						{
							float* sptr = sptr2 + imstep*(l);
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
								mvalue = _mm_add_ps(mvalue, _mm_mul_ps(www, _mm_loadu_ps(vptr2 + imstep*(tr + l) + tr + k)));
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

	class JointNonlocalMeansFilterInvorker8u_SSE4 : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* guide;
		Mat* dest;
		int templeteWindowSizeX;
		int searchWindowSizeX;
		int templeteWindowSizeY;
		int searchWindowSizeY;

		float* w;
	public:

		JointNonlocalMeansFilterInvorker8u_SSE4(Mat& src_, Mat& guide_, Mat& dest_, int templeteWindowSizeX_, int templeteWindowSizeY_, int searchWindowSizeX_, int searchWindowSizeY_, float* weight)
			: im(&src_), guide(&guide_), dest(&dest_), templeteWindowSizeX(templeteWindowSizeX_), templeteWindowSizeY(templeteWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), w(weight)
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
			int cng = (guide->rows - 2 * (tr_y + sr_y)) / dest->rows;

			if (dest->channels() == 3 && cng == 3)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				int CV_DECL_ALIGNED(16) buf[16];
				const __m128i zero = _mm_setzero_si128();
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = guide->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = guide->ptr<uchar>(j);
					const uchar* vptr2_ = im->ptr<uchar>(j);
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
						const uchar* vptr2 = vptr2_ + i;
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

								const uchar* ss = vptr2 + imstep*(tr_y + l) + (tr_x + k);
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
			else if (dest->channels() == 3 && cng == 1)
			{
				const int colorstep = im->size().area() / 3;
				const int colorstep2 = colorstep * 2;
				int CV_DECL_ALIGNED(16) buf[16];
				const __m128i zero = _mm_setzero_si128();
				for (int j = r.start; j < r.end; j++)
				{
					uchar* d = dest->ptr<uchar>(j);
					const uchar* tprt_ = guide->ptr<uchar>(sr_y + j) + sr_x;
					const uchar* sptr2_ = guide->ptr<uchar>(j);
					const uchar* vptr2_ = im->ptr<uchar>(j);
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
						const uchar* vptr2 = vptr2_ + i;
						for (int l = searchWindowSizeY; l--;)
						{
							const uchar* sptr = sptr2 + imstep*(l);
							for (int k = searchWindowSizeX; k--;)
							{
								uchar* t = (uchar*)tprt;
								uchar* s = (uchar*)(sptr + k);
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

								const uchar* ss = vptr2 + imstep*(tr_y + l) + (tr_x + k);
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
			else if (dest->channels() == 1 && cng == 3)
			{
				const int colorstep = guide->size().area() / 3;
				const int colorstep2 = colorstep * 2;
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
						uchar* tprt = guide->ptr<uchar>(sr_y + j) + (sr_x + i);
						uchar* sptr2 = guide->ptr<uchar>(j) +(i);
						uchar* vptr2 = im->ptr<uchar>(j) +(i);
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

								const __m128i sc = _mm_loadu_si128((__m128i*)(vptr2 + imstep*(tr_y + l) + tr_x + k));
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
			else if (dest->channels() == 1 && cng == 1)
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
						uchar* tprt = guide->ptr<uchar>(sr_y + j) + (sr_x + i);
						uchar* sptr2 = guide->ptr<uchar>(j) +(i);
						uchar* vptr2 = im->ptr<uchar>(j) +(i);
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

								const __m128i sc = _mm_loadu_si128((__m128i*)(vptr2 + imstep*(tr_y + l) + tr_x + k));
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

	class JointNonlocalMeansFilterInvorker32f_SSE4 : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* guide;
		Mat* dest;
		int templeteWindowSize;
		int searchWindowSize;

		float* w;
	public:

		JointNonlocalMeansFilterInvorker32f_SSE4(Mat& src_, Mat& guide_, Mat& dest_, int templeteWindowSize_, int searchWindowSize_, float* weight)
			: im(&src_), guide(&guide_), dest(&dest_), templeteWindowSize(templeteWindowSize_), searchWindowSize(searchWindowSize_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range &r) const
		{
			const int tr = templeteWindowSize >> 1;
			const int sr = searchWindowSize >> 1;
			const int cstep = guide->cols - templeteWindowSize;
			const int imstep = im->cols;
			int cng = (guide->rows - 2 * (tr + sr)) / dest->rows;

			const int tD = templeteWindowSize*templeteWindowSize;
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
							const float* sptr = sptr2 + imstep*(l);
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

								const float* ss = vptr2 + imstep*(tr + l) + (tr + k);
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
							const float* sptr = sptr2 + imstep*(l);
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
								mvalue = _mm_add_ps(mvalue, _mm_mul_ps(www, _mm_loadu_ps(vptr2 + imstep*(tr + l) + tr + k)));
								mtweight = _mm_add_ps(mtweight, www);
							}
						}
						_mm_store_ps(d, _mm_div_ps(mvalue, mtweight));
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
							const float* sptr = sptr2 + imstep*(l);
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

								const float* ss = vptr2 + imstep*(tr + l) + (tr + k);
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
						float* sptr2 = guide->ptr<float>(j) +(i);
						float* vptr2 = im->ptr<float>(j) +(i);

						for (int l = searchWindowSize; l--;)
						{
							float* sptr = sptr2 + imstep*(l);
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
								mvalue = _mm_add_ps(mvalue, _mm_mul_ps(www, _mm_loadu_ps(vptr2 + imstep*(tr + l) + tr + k)));
								mtweight = _mm_add_ps(mtweight, www);
							}
						}
						//weight normalization
						_mm_store_ps(d, _mm_div_ps(mvalue, mtweight));
						d += 4;
					}//i
				}//j
			}
		}
	};

	/*
	template <class T>
	class JointNonlocalMeansFilterInvorker_ : public cv::ParallelLoopBody
	{
	private:
	Mat* im;
	Mat* gim;
	Mat* dest;
	int templeteWindowSize;
	int searchWindowSize;

	float* w;
	public:

	JointNonlocalMeansFilterInvorker_(Mat& src_, Mat& guide_, Mat& dest_, int templeteWindowSize_, int searchWindowSize_, float* weight)
	: im(&src_), gim(&guide_), dest(&dest_), templeteWindowSize(templeteWindowSize_),searchWindowSize(searchWindowSize_), w(weight)
	{
	;
	}
	virtual void operator()( const cv::Range &r ) const
	{
	const int tr = templeteWindowSize>>1;
	const int sr = searchWindowSize>>1;
	const int D = searchWindowSize*searchWindowSize;
	const int cstep  = (im->cols-templeteWindowSize)*im->channels();
	const int csstep = (im->cols-searchWindowSize  )*im->channels();
	const int imstep = im->cols*im->channels();

	const int H=D/2+1;

	const int tD = templeteWindowSize*templeteWindowSize;
	const double tdiv = 1.0/(double)(tD);//templete square div


	int* ww=new int[D];
	float* nw=new float[D];

	if(im->channels()==3)
	{
	for(int j=r.start;j<r.end;j++)
	{
	T* d = dest->ptr<T>(j);
	for(int i=0;i<dest->cols;i++)
	{
	float tweight=0.f;
	//search loop
	const T* tprt = gim->ptr<T>(sr+j) + 3*(sr+i);
	const T* sptr2 = gim->ptr<T>(j) + 3*(i);

	for(int l=searchWindowSize,count=D-1;l--;)
	{
	const T* sptr = sptr2 +imstep*(l);
	for (int k=searchWindowSize;k--;)
	{
	//templete loop
	double e=0.0;
	const T* t = tprt;
	T* s = (T*)(sptr+3*k);
	for(int n=templeteWindowSize;n--;)
	{
	for(int m=templeteWindowSize;m--;)
	{
	// computing color L2 norm
	e += (s[0]-t[0])*(s[0]-t[0])+(s[1]-t[1])*(s[1]-t[1])+(s[2]-t[2])*(s[2]-t[2]);//L2 norm
	s+=3,t+=3;
	}
	t+=cstep;
	s+=cstep;
	}
	const int ediv = (int)(e*tdiv);
	ww[count--]=ediv;

	//get weighted Euclidean distance
	tweight+=w[ediv];
	}
	}
	//weight normalization
	if(tweight==0.0)
	{
	//memset(nw,0,sizeof(double)*D);
	for(int z=0;z<D;z++) nw[z]=0.0;
	nw[H-1]=1.0;
	}
	else
	{
	double itweight=1.0/(double)tweight;
	for(int z=0;z<D;z++) nw[z]=w[ww[z]]*itweight;
	}

	double r=0.0,g=0.0,b=0.0;
	T* s = im->ptr<T>(j+tr); s+=3*(tr+i);
	for(int l=searchWindowSize,count=0;l--;)
	{
	for(int k=searchWindowSize;k--;)
	{
	r += s[0]*nw[count];
	g += s[1]*nw[count];
	b += s[2]*nw[count++];
	s+=3;
	}
	s+=csstep;
	}
	d[0] = saturate_cast<T>(r);
	d[1] = saturate_cast<T>(g);
	d[2] = saturate_cast<T>(b);
	d+=3;
	}//i
	}//j
	}
	else if(im->channels()==1)
	{
	for(int j=r.start;j<r.end;j++)
	{
	T* d = dest->ptr<T>(j);
	for(int i=0;i<dest->cols;i++)
	{
	float tweight=0.f;
	//search loop
	T* tprt = im->ptr<T>(sr+j) + (sr+i);
	T* sptr2 = im->ptr<T>(j) + (i);
	for(int l=searchWindowSize,count=D-1;l--;)
	{
	T* sptr = sptr2 +imstep*(l);
	for (int k=searchWindowSize;k--;)
	{
	//templete loop
	double e=0.0;
	T* t = tprt;
	T* s = sptr+k;
	for(int n=templeteWindowSize;n--;)
	{
	for(int m=templeteWindowSize;m--;)
	{
	// computing color L2 norm
	e += (*s-*t)*(*s-*t);
	s++,t++;
	}
	t+=cstep;
	s+=cstep;
	}
	const int ediv = (int)(e*tdiv);

	ww[count--]=ediv;
	//get weighted Euclidean distance
	tweight+=w[ediv];
	}
	}
	//weight normalization
	if(tweight==0.0)
	{
	for(int z=0;z<D;z++) nw[z]=0;
	nw[H-1]=1;
	}
	else
	{
	double itweight=1.0/(double)tweight;
	for(int z=0;z<D;z++) nw[z]=w[ww[z]]*itweight;
	}

	double v=0.0;
	T* s = im->ptr<T>(j+tr); s+=(tr+i);
	for(int l=searchWindowSize,count=0;l--;)
	{
	for(int k=searchWindowSize;k--;)
	{
	v += *(s++)*nw[count++];
	}
	s+=csstep;
	}
	*(d++) = saturate_cast<T>(v);
	}//i
	}//j
	}
	delete[] ww;
	delete[] nw;
	}
	};
	*/



	template <class T>
	class JointNonlocalMeansFilterBaseInvorker_ : public cv::ParallelLoopBody
	{
	private:
		Mat* im;
		Mat* guide;
		Mat* dest;
		int templeteWindowSize;
		int searchWindowSize;

		float* w;
	public:

		JointNonlocalMeansFilterBaseInvorker_(Mat& src_, Mat& guide_, Mat& dest_, int templeteWindowSize_, int searchWindowSize_, float* weight)
			: im(&src_), guide(&guide_), dest(&dest_), templeteWindowSize(templeteWindowSize_), searchWindowSize(searchWindowSize_), w(weight)
		{
			;
		}
		virtual void operator()(const cv::Range &r) const
		{
			const int tr = templeteWindowSize >> 1;
			const int sr = searchWindowSize >> 1;
			const int cstep = (guide->cols - templeteWindowSize)*guide->channels();
			const int imstep = im->cols*im->channels();
			const int imstepg = guide->cols*guide->channels();
			const int tD = templeteWindowSize*templeteWindowSize;
			const double tdiv = 1.0 / (double)(tD);//templete square div

			if (im->channels() == 3 && guide->channels() == 3)
			{
				for (int j = r.start; j < r.end; j++)
				{
					T* d = dest->ptr<T>(j);
					for (int i = 0; i < dest->cols; i++)
					{
						double r = 0.0;
						double g = 0.0;
						double b = 0.0;
						double tweight = 0.0;
						//search loop
						const T* tprt = guide->ptr<T>(sr + j) + 3 * (sr + i);
						const T* sptr2 = guide->ptr<T>(j) +3 * (i);
						const T* vptr2 = im->ptr<T>(j) +3 * (i);

						for (int l = searchWindowSize; l--;)
						{
							const T* sptr = sptr2 + imstep*(l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								double e = 0.0;
								const T* t = tprt;
								T* s = (T*)(sptr + 3 * k);
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
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

								const T* ss = vptr2 + imstep*(tr + l) + 3 * (tr + k);
								r += ss[0] * www;
								g += ss[1] * www;
								b += ss[2] * www;
								//get weighted Euclidean distance
								tweight += www;
							}
						}
						d[0] = saturate_cast<T>(r / tweight);
						d[1] = saturate_cast<T>(g / tweight);
						d[2] = saturate_cast<T>(b / tweight);
						d += 3;
					}//i
				}//j
			}
			else if (im->channels() == 3 && guide->channels() == 1)
			{
				for (int j = r.start; j < r.end; j++)
				{
					T* d = dest->ptr<T>(j);
					for (int i = 0; i < dest->cols; i++)
					{
						double r = 0.0;
						double g = 0.0;
						double b = 0.0;
						double tweight = 0.0;
						//search loop
						T* tprt = guide->ptr<T>(sr + j) + (sr + i);
						T* sptr2 = guide->ptr<T>(j) +(i);
						T* vptr2 = im->ptr<T>(j) +(3 * i);

						for (int l = searchWindowSize; l--;)
						{
							T* sptr = sptr2 + imstepg*(l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								double e = 0.0;
								T* t = tprt;
								T* s = sptr + k;
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
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

								const T* ss = vptr2 + imstep*(tr + l) + 3 * (tr + k);
								r += ss[0] * www;
								g += ss[1] * www;
								b += ss[2] * www;
								//get weighted Euclidean distance
								tweight += www;
							}
						}
						//weight normalization
						d[0] = saturate_cast<T>(r / tweight);
						d[1] = saturate_cast<T>(g / tweight);
						d[2] = saturate_cast<T>(b / tweight);
						d += 3;
					}//i
				}//j
			}
			else if (im->channels() == 1 && guide->channels() == 3)
			{
				//cout<<"C1G3"<<endl;
				for (int j = r.start; j < r.end; j++)
				{
					T* d = dest->ptr<T>(j);
					for (int i = 0; i < dest->cols; i++)
					{
						double value = 0.0;
						double tweight = 0.0;
						//search loop
						const T* tprt = guide->ptr<T>(sr + j) + 3 * (sr + i);
						const T* sptr2 = guide->ptr<T>(j) +(3 * i);
						const T* vptr2 = im->ptr<T>(j) +(i);

						for (int l = searchWindowSize; l--;)
						{
							const T* sptr = sptr2 + imstepg*(l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								double e = 0.0;
								T* t = (T*)tprt;
								T* s = (T*)(sptr + 3 * k);
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
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
								value += vptr2[imstep*(tr + l) + tr + k] * www;

								//get weighted Euclidean distance
								tweight += www;
							}
						}
						*(d++) = saturate_cast<T>(value / tweight);
					}//i
				}//j
			}
			else if (im->channels() == 1 && guide->channels() == 1)
			{
				//cout<<"C1G1"<<endl;
				for (int j = r.start; j < r.end; j++)
				{
					T* d = dest->ptr<T>(j);
					for (int i = 0; i < dest->cols; i++)
					{
						double value = 0.0;
						double tweight = 0.0;
						//search loop
						T* tprt = guide->ptr<T>(sr + j) + (sr + i);
						T* sptr2 = guide->ptr<T>(j) +(i);
						T* vptr2 = im->ptr<T>(j) +(i);

						for (int l = searchWindowSize; l--;)
						{
							T* sptr = sptr2 + imstepg*(l);
							for (int k = searchWindowSize; k--;)
							{
								//templete loop
								double e = 0.0;
								T* t = tprt;
								T* s = sptr + k;
								for (int n = templeteWindowSize; n--;)
								{
									for (int m = templeteWindowSize; m--;)
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
								value += vptr2[imstep*(tr + l) + tr + k] * www;
								//get weighted Euclidean distance
								tweight += www;
							}
						}
						//weight normalization
						*(d++) = saturate_cast<T>(value / tweight);
					}//i
				}//j
			}
		}
	};


	void jointNonLocalMeansFilterBase(Mat& src, Mat& guide, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma, int borderType)
	{
		if (templeteWindowSize > searchWindowSize)
		{
			//cout<<format("searchWindowSize should be larger than templeteWindowSize: now T: %d, S %d",templeteWindowSize,searchWindowSize)<<endl;
			return;
		}
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int tr = templeteWindowSize >> 1;
		const int sr = searchWindowSize >> 1;
		const int bb = sr + tr;
		//	const int D = searchWindowSize*searchWindowSize;

		//create large size image for bounding box;
		Mat im, gim;
		copyMakeBorder(src, im, bb, bb, bb, bb, borderType);
		copyMakeBorder(guide, gim, bb, bb, bb, bb, borderType);

		//weight computation;
		vector<float> weight(256 * guide.channels());
		float* w = &weight[0];
		const double gauss_sd = (sigma == 0.0) ? h : sigma;
		double gauss_color_coeff = -(1.0 / (double)(guide.channels()))*(1.0 / (h*h));
		for (int i = 0; i < 256 * guide.channels(); i++)
		{
			double v = std::exp(max(i*i - 2.0*gauss_sd*gauss_sd, 0.0)*gauss_color_coeff);
			w[i] = (float)v;
		}

		if (src.depth() == CV_8U)
		{
			JointNonlocalMeansFilterBaseInvorker_<uchar> body(im, gim, dest, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		if (src.depth() == CV_16U)
		{
			JointNonlocalMeansFilterBaseInvorker_<ushort> body(im, gim, dest, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		if (src.depth() == CV_16S)
		{
			JointNonlocalMeansFilterBaseInvorker_<short> body(im, gim, dest, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		else if (src.depth() == CV_32F)
		{
			JointNonlocalMeansFilterBaseInvorker_<float> body(im, gim, dest, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
		else if (src.depth() == CV_64F)
		{
			JointNonlocalMeansFilterBaseInvorker_<double> body(im, gim, dest, templeteWindowSize, searchWindowSize, w);
			cv::parallel_for_(Range(0, dest.rows), body);
		}
	}

	void jointNonLocalMeansFilter_SSE(Mat& src, Mat& guide, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int borderType)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int bbx = (templeteWindowSize.width >> 1) + (searchWindowSize.width >> 1);
		const int bby = (templeteWindowSize.height >> 1) + (searchWindowSize.height >> 1);

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

		//weight computation;
		vector<float> weight(256 * guide.channels());
		float* w = &weight[0];
		const double gauss_sd = (sigma == 0.0) ? h : sigma;
		double gauss_color_coeff = -(1.0 / (double)(guide.channels()))*(1.0 / (h*h));
		for (int i = 0; i < 256 * guide.channels(); i++)
		{
			double v = std::exp(max(i*i - 2.0*gauss_sd*gauss_sd, 0.0)*gauss_color_coeff);
			w[i] = (float)v;
		}

		if (src.depth() == CV_8U)
		{
			JointNonlocalMeansFilterInvorker8u_SSE4 body(im, gim, dst, templeteWindowSize.width, templeteWindowSize.height, searchWindowSize.width, searchWindowSize.height, w);
			cv::parallel_for_(Range(0, dst.rows), body);
			/*Mat imf,gimf,dstf;
			im.convertTo(imf,CV_32F);
			gim.convertTo(gimf,CV_32F);
			dst.convertTo(dstf,CV_32F);
			JointNonlocalMeansFilterInvorker32f_SSE4 body(imf,gimf,dstf,templeteWindowSize,searchWindowSize,w);
			cv::parallel_for_(Range(0, dst.rows), body);
			dstf.convertTo(dst,CV_8U);*/
		}/*
		if(src.depth()==CV_16U)
		{
		Mat imf,gimf,dstf;
		im.convertTo(imf,CV_32F);
		gim.convertTo(gimf,CV_32F);
		dst.convertTo(dstf,CV_32F);
		JointNonlocalMeansFilterInvorker32f_SSE4 body(imf,gimf,dstf,templeteWindowSize,searchWindowSize,w);
		cv::parallel_for_(Range(0, dst.rows), body);
		dstf.convertTo(dst,CV_16U);
		}
		if(src.depth()==CV_16S)
		{
		Mat imf,gimf,dstf;
		im.convertTo(imf,CV_32F);
		gim.convertTo(gimf,CV_32F);
		dst.convertTo(dstf,CV_32F);
		JointNonlocalMeansFilterInvorker32f_SSE4 body(imf,gimf,dstf,templeteWindowSize,searchWindowSize,w);
		cv::parallel_for_(Range(0, dst.rows), body);
		dstf.convertTo(dst,CV_16S);
		}
		else if(src.depth()==CV_32F)
		{
		JointNonlocalMeansFilterInvorker32f_SSE4 body(im,gim,dst,templeteWindowSize,searchWindowSize,w);
		cv::parallel_for_(Range(0, dst.rows), body);
		}
		else if(src.depth()==CV_64F)
		{
		Mat imf,gimf,dstf;
		im.convertTo(imf,CV_32F);
		gim.convertTo(gimf,CV_32F);
		dst.convertTo(dstf,CV_32F);
		JointNonlocalMeansFilterInvorker32f_SSE4 body(imf,gimf,dstf,templeteWindowSize,searchWindowSize,w);
		cv::parallel_for_(Range(0, dst.rows), body);
		dstf.convertTo(dst,CV_64F);
		}*/

		Mat(dst(Rect(0, 0, dest.cols, dest.rows))).copyTo(dest);
	}

	void jointNonLocalMeansFilter_SP(Mat& src, Mat& guide, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int borderType)
	{
		jointNonLocalMeansFilter_SSE(src, guide, dest, templeteWindowSize, Size(searchWindowSize.width, 1), h, sigma, borderType);
		jointNonLocalMeansFilter_SSE(dest, guide, dest, templeteWindowSize, Size(1, searchWindowSize.height), h, sigma, borderType);
	}

	void jointNonLocalMeansFilter(Mat& src, Mat& guide, Mat& dest, Size templeteWindowSize, Size searchWindowSize, double h, double sigma, int method, int borderType)
	{
		CV_Assert(FILTER_CIRCLE);
		if (sigma < 1.0) sigma = h;

		if (method == FILTER_SLOWEST)
		{
			jointNonLocalMeansFilterBase(src, guide, dest, templeteWindowSize.width, searchWindowSize.height, h, sigma, borderType);
		}
		else if (method == FILTER_RECTANGLE || method == FILTER_DEFAULT)
		{
			jointNonLocalMeansFilter_SSE(src, guide, dest, templeteWindowSize, searchWindowSize, h, sigma, borderType);
		}
		else if (method == FILTER_SEPARABLE)
		{
			jointNonLocalMeansFilter_SP(src, guide, dest, templeteWindowSize, searchWindowSize, h, sigma, borderType);
		}
	}

	void jointNonLocalMeansFilter(Mat& src, Mat& guide, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma, int method, int borderType)
	{
		jointNonLocalMeansFilter(src, guide, dest, Size(templeteWindowSize, templeteWindowSize), Size(searchWindowSize, searchWindowSize), h, sigma, method, borderType);
	}

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
		double gauss_color_coeff = -(1.0 / (double)(guide.channels()))*(1.0 / (h*h));
		for (int i = 0; i < 256 * guide.channels(); i++)
		{
			double v = std::exp(max(i*i - 2.0*gauss_sd*gauss_sd, 0.0)*gauss_color_coeff);
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
}