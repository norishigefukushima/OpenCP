#include "opencp.hpp"

using namespace std;
using namespace cv;

void Hadamard2D4x4(float* src);
void Hadamard2D4x4andThresh(float* src, float thresh);
void Hadamard2D4x4andThreshandIDHT(float* src, float thresh);
void Hadamard2D8x8andThresh(float* src, float thresh);
void Hadamard2D8x8(float* src);
void Hadamard2D8x8andThreshandIDHT(float* src, float thresh);

void Hadamard2D16x16andThreshandIDHT(float* src, float thresh);
void Hadamard2D16x16(float* src);

void dct4x4_1d_llm_fwd_sse(float* s, float* d);//8add, 4 mul
void dct4x4_1d_llm_inv_sse(float* s, float* d);

void transpose4x4(float* src, float* dest);
void transpose4x4(float* src);
void dct4x4_llm_sse(float* a, float* b, float* temp, int flag);
void fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(float* s, float threshold);
int  fDCT4x4_32f_and_threshold_and_iDCT4x4_nonzero_32f(float* s, float threshold);

void fDCT8x8_32f(const float* s, float* d, float* temp);
void iDCT8x8_32f(const float* s, float* d, float* temp);

void fDCT8x8_32f_and_threshold(const float* s, float* d, float threshold, float* temp);

void fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(float* s, float threshold);
int  fDCT8x8_32f_and_threshold_and_iDCT8x8_nonzero_32f(float* s, float threshold);

void iDCT8x8_32f(float* s);
void fDCT8x8_32f_and_threshold(float* s, float threshold);

void fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack(float* src, float* dest, float thresh);

namespace cp
{
	
#define _KEEP_00_COEF_

	class DenoiseWeightedDCTShrinkageInvorker8x8 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float* weight;
		float thresh;
		int width;
		int height;

	public:

		DenoiseWeightedDCTShrinkageInvorker8x8(float *sim, float* dim, float* wmap, float Th, int w, int h) : src(sim), dest(dim), weight(wmap), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 8 + 1;
			const int wstep = width - 8 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			const int w4 = 4 * width;
			const int w5 = 5 * width;
			const int w6 = 6 * width;
			const int w7 = 7 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(8, 8), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				float* w0 = &weight[width*j];
				const int sz = sizeof(float) * 8;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 8, s0 + w1, sz);
					memcpy(ptch + 16, s0 + w2, sz);
					memcpy(ptch + 24, s0 + w3, sz);
					memcpy(ptch + 32, s0 + w4, sz);
					memcpy(ptch + 40, s0 + w5, sz);
					memcpy(ptch + 48, s0 + w6, sz);
					memcpy(ptch + 56, s0 + w7, sz);

					int v = fDCT8x8_32f_and_threshold_and_iDCT8x8_nonzero_32f(ptch, thresh);

					/*fDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));
					float f0=*(float*)patch.data;
					compare(abs(patch),thresh,mask,CMP_LT);
					patch.setTo(0.f,mask);
					*(float*)patch.data = f0;
					iDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));*/
					//dct(patch,patch,DCT_INVERSE);


					const __m128 mw = _mm_set1_ps((float)(v));

					//add data
					for (int jp = 0; jp < 8; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						float* w = &w0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);
						__m128 mw1 = _mm_loadu_ps(w);

						_mm_storeu_ps(w, _mm_add_ps(mw1, mw));
						_mm_storeu_ps(d, _mm_add_ps(sp1, _mm_mul_ps(mp1, mw)));

						s += 4;
						d += 4;
						w += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						mw1 = _mm_loadu_ps(w);

						_mm_storeu_ps(w, _mm_add_ps(mw1, mw));
						_mm_storeu_ps(d, _mm_add_ps(sp1, _mm_mul_ps(mp1, mw)));
					}
					s0++;
					d0++;
					w0++;
				}
			}
		}
	};

	class DenoiseDCTShrinkageInvorker8x8 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDCTShrinkageInvorker8x8(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 8 + 1;
			const int wstep = width - 8 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			const int w4 = 4 * width;
			const int w5 = 5 * width;
			const int w6 = 6 * width;
			const int w7 = 7 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(8, 8), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 8;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 8, s0 + w1, sz);
					memcpy(ptch + 16, s0 + w2, sz);
					memcpy(ptch + 24, s0 + w3, sz);
					memcpy(ptch + 32, s0 + w4, sz);
					memcpy(ptch + 40, s0 + w5, sz);
					memcpy(ptch + 48, s0 + w6, sz);
					memcpy(ptch + 56, s0 + w7, sz);

					fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(ptch, thresh);

					/*fDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));
					float f0=*(float*)patch.data;
					compare(abs(patch),thresh,mask,CMP_LT);
					patch.setTo(0.f,mask);
					*(float*)patch.data = f0;
					iDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));*/
					//dct(patch,patch,DCT_INVERSE);


					//add data
					for (int jp = 0; jp < 8; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));

						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}
	};


	class DenoiseWeightedDCTShrinkageInvorker4x4 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float* weight;
		float thresh;
		int width;
		int height;

	public:
		DenoiseWeightedDCTShrinkageInvorker4x4(float *sim, float* dim, float* wmap, float Th, int w, int h) : src(sim), dest(dim), weight(wmap), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;

			int j;
			Mat buff(Size(4, 4), CV_32F); Mat mask;
			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				float* w0 = &weight[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);

					//Mat p2 = patch.clone();
					int v = fDCT4x4_32f_and_threshold_and_iDCT4x4_nonzero_32f(patch.ptr<float>(0), thresh);

					//float vv = norm(patch,p2,NORM_L1);
					//add data
					//const __m128 mw = _mm_set1_ps((float)(256.f-v*v));
					const __m128 mw = _mm_set1_ps((float)(v));
					//const __m128 mw = _mm_set1_ps((float)(vv));
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						float* w = &w0[(jp)*width];

						__m128 mp1 = _mm_load_ps(s);
						__m128 mw1 = _mm_loadu_ps(w);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(w, _mm_add_ps(mw1, mw));
						_mm_storeu_ps(d, _mm_add_ps(sp1, _mm_mul_ps(mw, mp1)));
					}
					s0++;
					d0++;
					w0++;
				}
			}
		}
	};
	class DenoiseDCTShrinkageInvorker4x4 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:
		DenoiseDCTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;

			int j;
			Mat buff(Size(4, 4), CV_32F); Mat mask;
			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);

					fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(patch.ptr<float>(0), thresh);

					//#define _HARD_THRESHOLDING_
					//#ifdef _HARD_THRESHOLDING_
					//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),0);
					//				//dct(patch,patch);
					//#ifdef _KEEP_00_COEF_
					//				float f0=*(float*)patch.data;
					//#endif
					//				compare(abs(patch),thresh,mask,CMP_LT);
					//				patch.setTo(0.f,mask);
					//
					//#ifdef _KEEP_00_COEF_
					//				*(float*)patch.data = f0;
					//#endif
					//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),DCT_INVERSE);
					//				//dct(patch,patch,DCT_INVERSE);
					//
					//#else
					//				dct(patch,patch);
					//				Mat dst;
					//				max(abs(patch)-25*thresh,0.f,dst);
					//				compare(patch,0,mask,CMP_LT);
					//				Mat(-1*dst).copyTo(dst,mask);
					//				dst.at<float>(0,0)=patch.at<float>(0,0);
					//				dct(dst,patch,DCT_INVERSE);
					//#endif

					//add data
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}
	};


	class ShearableDenoiseDCTShrinkageInvorker4x4 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;
		int direct;


	public:
		ShearableDenoiseDCTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h, int dr) : src(sim), dest(dim), width(w), height(h), thresh(Th), direct(dr)
		{
			;
		}

		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;

			int j;
			Mat buff(Size(4, 4), CV_32F); Mat mask;
			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					if (direct == 0)
					{
						memcpy(ptch, s0, sz);
						memcpy(ptch + 4, s0 + w1, sz);
						memcpy(ptch + 8, s0 + w2, sz);
						memcpy(ptch + 12, s0 + w3, sz);
					}
					else if (direct == 1)
					{
						memcpy(ptch, s0, sz);
						memcpy(ptch + 4, s0 + w1, sz);
						memcpy(ptch + 8, s0 + w2 + 1, sz);
						memcpy(ptch + 12, s0 + w3 + 1, sz);
					}
					else if (direct == 2)
					{
						memcpy(ptch, s0, sz);
						memcpy(ptch + 4, s0 + w1 + 1, sz);
						memcpy(ptch + 8, s0 + w2 + 2, sz);
						memcpy(ptch + 12, s0 + w3 + 3, sz);
					}
					else if (direct == 3)
					{
						memcpy(ptch, s0, sz);
						memcpy(ptch + 4, s0 + w1 + 2, sz);
						memcpy(ptch + 8, s0 + w2 + 4, sz);
						memcpy(ptch + 12, s0 + w3 + 6, sz);

						/**(ptch  ) = *(s0     );
						*(ptch+1) = *(s0   +1);
						*(ptch+2) = *(s0-w1+2);
						*(ptch+3) = *(s0-w1+3);

						*(ptch+4) = *(s0+w1  );
						*(ptch+5) = *(s0+w1+1);
						*(ptch+6) = *(s0   +2);
						*(ptch+7) = *(s0   +3);

						*(ptch+ 8) = *(s0+w2  );
						*(ptch+ 9) = *(s0+w2+1);
						*(ptch+10) = *(s0+w1+2);
						*(ptch+11) = *(s0+w1+3);

						*(ptch+12) = *(s0+w3  );
						*(ptch+13) = *(s0+w3+1);
						*(ptch+14) = *(s0+w2+2);
						*(ptch+15) = *(s0+w2+3);*/

					}
					else if (direct == -3)
					{
						memcpy(ptch, s0, sz);
						memcpy(ptch + 4, s0 + w1 - 2, sz);
						memcpy(ptch + 8, s0 + w2 - 4, sz);
						memcpy(ptch + 12, s0 + w3 - 6, sz);
					}
					else if (direct == -1)
					{
						memcpy(ptch, s0, sz);
						memcpy(ptch + 4, s0 + w1, sz);
						memcpy(ptch + 8, s0 + w2 - 1, sz);
						memcpy(ptch + 12, s0 + w3 - 1, sz);
					}
					else if (direct == -2)
					{
						memcpy(ptch, s0, sz);
						memcpy(ptch + 4, s0 + w1 - 1, sz);
						memcpy(ptch + 8, s0 + w2 - 2, sz);
						memcpy(ptch + 12, s0 + w3 - 3, sz);
					}


					fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(patch.ptr<float>(0), thresh);

					//#define _HARD_THRESHOLDING_
					//#ifdef _HARD_THRESHOLDING_
					//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),0);
					//				//dct(patch,patch);
					//#ifdef _KEEP_00_COEF_
					//				float f0=*(float*)patch.data;
					//#endif
					//				compare(abs(patch),thresh,mask,CMP_LT);
					//				patch.setTo(0.f,mask);
					//
					//#ifdef _KEEP_00_COEF_
					//				*(float*)patch.data = f0;
					//#endif
					//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),DCT_INVERSE);
					//				//dct(patch,patch,DCT_INVERSE);
					//
					//#else
					//				dct(patch,patch);
					//				Mat dst;
					//				max(abs(patch)-25*thresh,0.f,dst);
					//				compare(patch,0,mask,CMP_LT);
					//				Mat(-1*dst).copyTo(dst,mask);
					//				dst.at<float>(0,0)=patch.at<float>(0,0);
					//				dct(dst,patch,DCT_INVERSE);
					//#endif

					//add data

					if (direct == 0)
					{
						for (int jp = 0; jp < 4; jp++)
						{
							float* s = patch.ptr<float>(jp);
							float* d = &d0[(jp)*width];
							__m128 mp1 = _mm_load_ps(s);
							__m128 sp1 = _mm_loadu_ps(d);

							_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						}
					}
					else if (direct == 1)
					{
						for (int jp = 0; jp < 4; jp++)
						{
							float* s = patch.ptr<float>(jp);
							float* d = &d0[(jp)*width + jp / 2];
							__m128 mp1 = _mm_load_ps(s);
							__m128 sp1 = _mm_loadu_ps(d);

							_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						}
					}
					else if (direct == 2)
					{
						for (int jp = 0; jp < 4; jp++)
						{
							float* s = patch.ptr<float>(jp);
							float* d = &d0[(jp)*width + jp];
							__m128 mp1 = _mm_load_ps(s);
							__m128 sp1 = _mm_loadu_ps(d);

							_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						}
					}
					else if (direct == 3)
					{
						for (int jp = 0; jp < 4; jp++)
						{
							float* s = patch.ptr<float>(jp);
							float* d = &d0[(jp)*width + 2 * jp];
							__m128 mp1 = _mm_load_ps(s);
							__m128 sp1 = _mm_loadu_ps(d);

							_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						}
					}
					else if (direct == -3)
					{
						for (int jp = 0; jp < 4; jp++)
						{
							float* s = patch.ptr<float>(jp);
							float* d = &d0[(jp)*width - 2 * jp];
							__m128 mp1 = _mm_load_ps(s);
							__m128 sp1 = _mm_loadu_ps(d);

							_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						}
					}
					/*else if(direct==3)
					{
					for (int jp = 0; jp < 4; jp ++)
					{
					float* s =patch.ptr<float>(jp);
					d0[(jp)*width+0] += s[0];
					d0[(jp)*width+1] += s[1];
					d0[(jp)*width+2] += s[2];
					d0[(jp)*width+3] += s[3];
					}
					}*/
					else if (direct == -1)
					{
						for (int jp = 0; jp < 4; jp++)
						{
							float* s = patch.ptr<float>(jp);
							float* d = &d0[(jp)*width - jp / 2];
							__m128 mp1 = _mm_load_ps(s);
							__m128 sp1 = _mm_loadu_ps(d);

							_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						}
					}
					else if (direct == -2)
					{
						for (int jp = 0; jp < 4; jp++)
						{
							float* s = patch.ptr<float>(jp);
							float* d = &d0[(jp)*width - jp];
							__m128 mp1 = _mm_load_ps(s);
							__m128 sp1 = _mm_loadu_ps(d);

							_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						}
					}
					s0++;
					d0++;
				}
			}
		}
	};

	class DenoiseDCTShrinkageInvorker4x4to8x8
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDCTShrinkageInvorker4x4to8x8(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;

			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				Mat patchU(Size(8, 8), CV_32F, Scalar::all(0));
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[2 * width*j * 2];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);
					Mat mask;
#define _HARD_THRESHOLDING_
#ifdef _HARD_THRESHOLDING_
					resize(patch, patchU, Size(8, 8), 0, 0, cv::INTER_LANCZOS4);
					dct(patchU, patchU);
#ifdef _KEEP_00_COEF_
					float f0 = *(float*)patchU.data;
#endif
					compare(abs(patchU), thresh, mask, CMP_LT);
					patchU.setTo(0.f, mask);

#ifdef _KEEP_00_COEF_
					*(float*)patchU.data = f0;
#endif
					//dct(patch,patch,DCT_INVERSE);resize(patch,patchU,Size(8,8),0,0,1);

					//patch.copyTo(patchU(Rect(0,0,4,4)));dct(patchU,patchU,DCT_INVERSE);
					//resize(patch,patchU,Size(8,8));

					dct(patchU, patchU, DCT_INVERSE);

					//Mat p;patchU.convertTo(p,CV_8U);imshow("patch",p);waitKey();

#else
					dct(patch, patch);
					Mat dst;
					max(abs(patch) - 25 * thresh, 0.f, dst);
					compare(patch, 0, mask, CMP_LT);
					Mat(-1 * dst).copyTo(dst, mask);
					dst.at<float>(0, 0) = patch.at<float>(0, 0);

					dct(dst, patch, DCT_INVERSE);
#endif

					//fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(ptch, thresh);

					//add data
					for (int jp = 0; jp < 8; jp++)
					{
						float* s = patchU.ptr<float>(jp);
						float* d = &d0[(jp)*width * 2];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));

						mp1 = _mm_load_ps(s + 4);
						sp1 = _mm_loadu_ps(d + 4);
						_mm_storeu_ps(d + 4, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0 += 2;
				}
			}
		}
	};

	class DenoiseDCTShrinkageInvorker2x2 : public cv::ParallelLoopBody
	{
	private:


		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDCTShrinkageInvorker2x2(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{

		}
		virtual void operator() (const Range& range) const
		{
			//2x2 patch
			const int size1 = width * height;
			const int hstep = height - 2 + 1;
			const int wstep = width - 2 + 1;

			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 2), CV_32F);
				float* ptch = patch.ptr<float>(0);
				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i += 4)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + width, sz);

					fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack((float*)patch.data, (float*)patch.data, thresh);

					//add data
					__m128 mp1 = _mm_loadu_ps(ptch);
					__m128 sp1 = _mm_loadu_ps(d0);
					_mm_storeu_ps(d0, _mm_add_ps(sp1, mp1));
					mp1 = _mm_loadu_ps(ptch + 4);
					sp1 = _mm_loadu_ps(d0 + width);
					_mm_storeu_ps(d0 + width, _mm_add_ps(sp1, mp1));

					memcpy(ptch, s0 + 1, sz);
					memcpy(ptch + 4, s0 + width + 1, sz);

					fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack((float*)patch.data, (float*)patch.data, thresh);

					//add data
					mp1 = _mm_loadu_ps(ptch);
					sp1 = _mm_loadu_ps(d0 + 1);
					_mm_storeu_ps(d0 + 1, _mm_add_ps(sp1, mp1));
					mp1 = _mm_loadu_ps(ptch + 4);
					sp1 = _mm_loadu_ps(d0 + width + 1);
					_mm_storeu_ps(d0 + width + 1, _mm_add_ps(sp1, mp1));

					s0 += 4;
					d0 += 4;
				}
			}
		}

	};

	class DenoiseDCTShrinkageInvorkerTest : public cv::ParallelLoopBody
	{
	private:


		float* src;
		float* dest;
		float thresh;
		int width;
		int height;
		Size patch_size;

	public:

		DenoiseDCTShrinkageInvorkerTest(float *sim, float* dim, float Th, int w, int h, Size psize) : src(sim), dest(dim), width(w), height(h), patch_size(psize), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			int pwidth = patch_size.width;
			int pheight = patch_size.height;
			const int size1 = width * height;
			const int hstep = height - pheight + 1;
			const int wstep = width - pwidth + 1;

			int j;
			Mat d = Mat(Size(width, height), CV_32F, dest);
			for (j = range.start; j != range.end; j++)
			{
				Mat patch(patch_size, CV_32F);
				Mat mask;//(patch_size, CV_8U);

				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float)*patch_size.width;
				for (int i = 0; i < wstep; i++)
				{
					for (int k = 0; k < patch_size.height; k++)
					{
						memcpy(ptch + k*patch_size.width, s0 + k*width, sz);
					}


					/*dct(patch,patch);
					compare(abs(patch),thresh,mask,CMP_LT);
					patch.setTo(0.f,mask);
					dct(patch,patch,DCT_INVERSE);*/

					dct(patch, patch);
					Mat dst;
					max(abs(patch) - sqrt(80000000000000.3*thresh), 0.f, dst);
					compare(patch, 0, mask, CMP_LT);
					Mat(-1 * dst).copyTo(dst, mask);
					dst.at<float>(0, 0) = patch.at<float>(0, 0);
					dct(dst, patch, DCT_INVERSE);

					/*dct(patch,patch);
					GaussianBlur(patch,patch,Size(5,5),thresh*0.01);
					dct(patch,patch,DCT_INVERSE);*/

					Mat r = d(Rect(i, j, patch_size.width, patch_size.height));
					r += patch;
					s0++;
					d0++;
				}
			}
		}
	};
	class DenoiseDCTShrinkageInvorker : public cv::ParallelLoopBody
	{
	private:


		float* src;
		float* dest;
		float thresh;
		int width;
		int height;
		Size patch_size;

	public:

		DenoiseDCTShrinkageInvorker(float *sim, float* dim, float Th, int w, int h, Size psize) : src(sim), dest(dim), width(w), height(h), patch_size(psize), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			int pwidth = patch_size.width;
			int pheight = patch_size.height;
			const int size1 = width * height;
			const int hstep = height - pheight + 1;
			const int wstep = width - pwidth + 1;

			int j;
			Mat d = Mat(Size(width, height), CV_32F, dest);
			for (j = range.start; j != range.end; j++)
			{
				Mat patch(patch_size, CV_32F);
				Mat mask;//(patch_size, CV_8U);

				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];

				const int sz = sizeof(float)*patch_size.width;
				for (int i = 0; i < wstep; i++)
				{
					for (int k = 0; k < patch_size.height; k++)
					{
						memcpy(ptch + k*patch_size.width, s0 + k*width, sz);
					}

					//Mat show1; patch.convertTo(show1,CV_8U);imshow("patchb",show1); 

					dct(patch, patch);
#ifdef _KEEP_00_COEF_
					float f0 = *(float*)patch.data;
#endif
					compare(abs(patch), thresh, mask, CMP_LT);
					patch.setTo(0.f, mask);
#ifdef _KEEP_00_COEF_
					*(float*)patch.data = f0;
#endif

					dct(patch, patch, DCT_INVERSE);

					//Mat show; patch.convertTo(show,CV_8U);imshow("patch",show); waitKey(0);
					Mat r = d(Rect(i, j, patch_size.width, patch_size.height));
					r += patch;
					s0++;
				}
			}
		}
	};

	class DenoiseDHTShrinkageInvorker16x16 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDHTShrinkageInvorker16x16(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 16 + 1;
			const int wstep = width - 16 + 1;

			int j;
			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(16, 16), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 16;
				for (int i = 0; i < wstep; i++)
				{
					for (int n = 0; n < 16; n++)
						memcpy(ptch + 16 * n, s0 + n*width, sz);

					Hadamard2D16x16andThreshandIDHT(ptch, thresh);

					//add data
					for (int jp = 0; jp < 16; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						s += 4;
						d += 4;
					}
					s0++;
					d0++;
				}
			}
		}

	};

	class DenoiseDHTShrinkageInvorker8x8 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDHTShrinkageInvorker8x8(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 8 + 1;
			const int wstep = width - 8 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			const int w4 = 4 * width;
			const int w5 = 5 * width;
			const int w6 = 6 * width;
			const int w7 = 7 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(8, 8), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 8;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 8, s0 + w1, sz);
					memcpy(ptch + 16, s0 + w2, sz);
					memcpy(ptch + 24, s0 + w3, sz);
					memcpy(ptch + 32, s0 + w4, sz);
					memcpy(ptch + 40, s0 + w5, sz);
					memcpy(ptch + 48, s0 + w6, sz);
					memcpy(ptch + 56, s0 + w7, sz);


					Hadamard2D8x8andThreshandIDHT(ptch, thresh);


					//add data
					for (int jp = 0; jp < 8; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));

						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}

	};

	class DenoiseDHTShrinkageInvorker4x4 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDHTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);


					Hadamard2D4x4andThreshandIDHT(ptch, thresh);

					//add data
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}
	};


	class DenoiseDHTShrinkageInvorker4x4S : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDHTShrinkageInvorker4x4S(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1 + 1, sz);
					memcpy(ptch + 8, s0 + w2 + 2, sz);
					memcpy(ptch + 12, s0 + w3 + 3, sz);

					Hadamard2D4x4andThreshandIDHT(ptch, thresh);

					//add data
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d + jp);

						_mm_storeu_ps(d + jp, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}
	};


	class DenoiseDHTShrinkageInvorker4x4S2 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDHTShrinkageInvorker4x4S2(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1 - 1, sz);
					memcpy(ptch + 8, s0 + w2 - 2, sz);
					memcpy(ptch + 12, s0 + w3 - 3, sz);


					Hadamard2D4x4andThreshandIDHT(ptch, thresh);

					//add data
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d - jp);

						_mm_storeu_ps(d - jp, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}
	};





	void ivDWT8x8(float* src)
	{
		float* s = src;
		for (int i = 0; i < 8; i++)
		{
			float s0 = s[0] + s[32];
			float s1 = s[0] - s[32];
			float s2 = s[8] + s[40];
			float s3 = s[8] - s[40];
			float s4 = s[16] + s[48];
			float s5 = s[16] - s[48];
			float s6 = s[24] + s[56];
			float s7 = s[24] - s[56];

			s[0] = s0; s[16] = s2;
			s[8] = s1; s[24] = s3;
			s[32] = s4; s[48] = s6;
			s[40] = s5; s[56] = s7;
			s++;
		}
	}

	void fvDWT8x8(float* src)
	{
		float* s = src;
		for (int i = 0; i < 8; i++)
		{
			float v0 = (s[0] + s[8])*0.5f;
			float v4 = (s[0] - s[8])*0.5f;
			float v1 = (s[16] + s[24])*0.5f;
			float v5 = (s[16] - s[24])*0.5f;

			float v2 = (s[32] + s[40])*0.5f;
			float v6 = (s[32] - s[40])*0.5f;
			float v3 = (s[48] + s[56])*0.5f;
			float v7 = (s[48] - s[56])*0.5f;

			s[0] = v0; s[16] = v2;
			s[8] = v1; s[24] = v3;
			s[32] = v4; s[48] = v6;
			s[40] = v5; s[56] = v7;
			s++;
		}
	}

	void ihDWT8x8(float* src)
	{
		float* s = src;
		for (int i = 0; i < 8; i++)
		{
			float s0 = s[0] + s[4];
			float s1 = s[0] - s[4];
			float s2 = s[1] + s[5];
			float s3 = s[1] - s[5];
			float s4 = s[2] + s[6];
			float s5 = s[2] - s[6];
			float s6 = s[3] + s[7];
			float s7 = s[3] - s[7];

			s[0] = s0; s[2] = s2;
			s[1] = s1; s[3] = s3;
			s[4] = s4; s[6] = s6;
			s[5] = s5; s[7] = s7;
			s += 8;
		}
	}

	void fhDWT8x8(float* src)
	{
		float* s = src;
		for (int i = 0; i < 8; i++)
		{
			float v0 = (s[0] + s[1])*0.5f;
			float v4 = (s[0] - s[1])*0.5f;
			float v1 = (s[2] + s[3])*0.5f;
			float v5 = (s[2] - s[3])*0.5f;

			float v2 = (s[4] + s[5])*0.5f;
			float v6 = (s[4] - s[5])*0.5f;
			float v3 = (s[6] + s[7])*0.5f;
			float v7 = (s[6] - s[7])*0.5f;

			s[0] = v0; s[2] = v2;
			s[1] = v1; s[3] = v3;
			s[4] = v4; s[6] = v6;
			s[5] = v5; s[7] = v7;
			s += 8;
		}
	}

	void ihDWT4x4(float* src)
	{
		float* s = src;
		for (int i = 0; i < 4; i++)
		{
			float s0 = s[0] + s[2];
			float s1 = s[0] - s[2];
			float s2 = s[1] + s[3];
			float s3 = s[1] - s[3];
			s[0] = s0; s[2] = s2;
			s[1] = s1; s[3] = s3;
			s += 4;
		}
	}
	void ivDWT4x4(float* src)
	{
		float* s = src;
		for (int i = 0; i < 4; i++)
		{
			float s0 = s[0] + s[8];
			float s1 = s[0] - s[8];
			float s2 = s[4] + s[12];
			float s3 = s[4] - s[12];
			s[0] = s0; s[8] = s2;
			s[4] = s1; s[12] = s3;

			s++;
		}
	}

	void fhDWT4x4(float* src)
	{
		float* s = src;
		for (int i = 0; i < 4; i++)
		{
			float v0 = (s[0] + s[1])*0.5f;
			float v2 = (s[0] - s[1])*0.5f;
			float v1 = (s[2] + s[3])*0.5f;
			float v3 = (s[2] - s[3])*0.5f;
			s[0] = v0; s[2] = v2;
			s[1] = v1; s[3] = v3;
			s += 4;
		}
	}
	void fvDWT4x4(float* src)
	{
		float* s = src;
		for (int i = 0; i < 4; i++)
		{
			float v0 = (s[0] + s[4])*0.5f;
			float v2 = (s[0] - s[4])*0.5f;
			float v1 = (s[8] + s[12])*0.5f;
			float v3 = (s[8] - s[12])*0.5f;
			s[0] = v0; s[8] = v2;
			s[4] = v1; s[12] = v3;
			s++;
		}
	}


	void iDWT4x4(float* src)
	{
		ivDWT4x4(src);
		ihDWT4x4(src);
		ivDWT4x4(src);
		ihDWT4x4(src);
	}
	void fDWT4x4(float* src)
	{
		fhDWT4x4(src);
		fvDWT4x4(src);
		fhDWT4x4(src);
		fvDWT4x4(src);
	}


	void iDWT8x8(float* src)
	{
		ivDWT8x8(src);
		ihDWT8x8(src);
		ivDWT8x8(src);
		ihDWT8x8(src);
		ivDWT8x8(src);
		ihDWT8x8(src);
	}
	void fDWT8x8(float* src)
	{
		fhDWT8x8(src);
		fvDWT8x8(src);

		fhDWT8x8(src);
		fvDWT8x8(src);
		fhDWT8x8(src);
		fvDWT8x8(src);
	}

	void DWT2D4x4andThreshandIDWT(float* src, float thresh)
	{

	}

	void printMat_float(Mat& src)
	{
		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				cout << format("%5.2f ", src.at<float>(j, i));
			}
			cout << endl;
		}
		cout << endl;
	}
	class DenoiseDWTShrinkageInvorker8x8 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDWTShrinkageInvorker8x8(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 8 + 1;
			const int wstep = width - 8 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			const int w4 = 4 * width;
			const int w5 = 5 * width;
			const int w6 = 6 * width;
			const int w7 = 7 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(8, 8), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 8;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 8, s0 + w1, sz);
					memcpy(ptch + 16, s0 + w2, sz);
					memcpy(ptch + 24, s0 + w3, sz);
					memcpy(ptch + 32, s0 + w4, sz);
					memcpy(ptch + 40, s0 + w5, sz);
					memcpy(ptch + 48, s0 + w6, sz);
					memcpy(ptch + 56, s0 + w7, sz);

					//if(j==200)
					{

						//printMat_float(patch);
						fDWT8x8(ptch);
						//printMat_float(patch);

						//float th = thresh*0.133;
						float th = 3.2f;

						for (int i = 1; i < 64; i++)
						{
							//ptch[i] = (abs(ptch[i])<th) ? 0.f: ptch[i];
							if (ptch[i] >= 0.f) ptch[i] = max(ptch[i] - th, 0.f);
							else ptch[i] = -max(-ptch[i] - th, 0.f);
						}
						//printMat_float(patch);

						iDWT8x8(ptch);
						//printMat_float(patch);
						//getchar();
					}

					//add data
					for (int jp = 0; jp < 8; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));

						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}
	};


	class DenoiseDWTShrinkageInvorker4x4 : public cv::ParallelLoopBody
	{
	private:
		float* src;
		float* dest;
		float thresh;
		int width;
		int height;

	public:

		DenoiseDWTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
		{
			;
		}
		virtual void operator() (const Range& range) const
		{
			const int size1 = width * height;
			const int hstep = height - 4 + 1;
			const int wstep = width - 4 + 1;
			const int w1 = 1 * width;
			const int w2 = 2 * width;
			const int w3 = 3 * width;
			int j;

			for (j = range.start; j != range.end; j++)
			{
				Mat patch(Size(4, 4), CV_32F);
				float* ptch = patch.ptr<float>(0);

				float* s0 = &src[width*j];
				float* d0 = &dest[width*j];
				const int sz = sizeof(float) * 4;
				for (int i = 0; i < wstep; i++)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);

					fDWT4x4(ptch);
					float th = thresh*0.2f;
					for (int i = 1; i < 16; i++)
					{
						//ptch[i] = (abs(ptch[i])<th) ? 0.f: ptch[i];
						if (ptch[i] >= 0.f) ptch[i] = max(ptch[i] - th, 0.f);
						else ptch[i] = -max(-ptch[i] - th, 0.f);
					}
					iDWT4x4(ptch);


					//add data
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
					s0++;
					d0++;
				}
			}
		}
	};








	void DenoiseDXTShrinkage::cvtColorOrder32F_BGR2BBBBGGGGRRRR(const Mat& src, Mat& dest)
	{
		vector<Mat> v(3);
		dest.create(Size(src.cols, src.rows * 3), CV_32F);
		Mat a(src.size(), CV_32F, dest.ptr<float>(0));
		Mat b(src.size(), CV_32F, dest.ptr<float>(src.rows));
		Mat c(src.size(), CV_32F, dest.ptr<float>(2 * src.rows));
		v[0] = a;
		v[1] = b;
		v[2] = c;
		split(src, v);
	}
	void DenoiseDXTShrinkage::cvtColorOrder32F_BBBBGGGGRRRR2BGR(const Mat& src, Mat& dest)
	{
		vector<Mat> v(3);

		Size size_ = Size(src.cols, src.rows / 3);
		Mat a(size_, CV_32F, (float*)src.ptr<float>(0));
		Mat b(size_, CV_32F, (float*)src.ptr<float>(size_.height));
		Mat c(size_, CV_32F, (float*)src.ptr<float>(2 * size_.height));

		v[0] = a;
		v[1] = b;
		v[2] = c;

		merge(v, dest);
	}

	void DenoiseDXTShrinkage::init(Size size_, int color_, Size patch_size_)
	{
		int w = size_.width + 2 * patch_size_.width;
		w += ((4 - w % 4) % 4);

		channel = color_;

		size = Size(w, size_.height + 2 * patch_size_.height);
		patch_size = patch_size_;
	}

	DenoiseDXTShrinkage::DenoiseDXTShrinkage()
	{
		isSSE = true;
	}

	DenoiseDXTShrinkage::DenoiseDXTShrinkage(Size size_, int color, Size patch_size_)
	{
		isSSE = true;
		init(size_, color, patch_size_);
	}

	void DenoiseDXTShrinkage::div(float* inplace0, float* inplace1, float* inplace2, float* wmap0, float* wmap1, float* wmap2, const int size1)
	{
		float* s0 = inplace0;
		float* s1 = inplace1;
		float* s2 = inplace2;

		float* w0 = wmap0;
		float* w1 = wmap1;
		float* w2 = wmap2;

		for (int i = 0; i < size1; i += 4)
		{
			__m128 md0 = _mm_load_ps(s0);
			__m128 md1 = _mm_load_ps(s1);
			__m128 md2 = _mm_load_ps(s2);

			__m128 mdiv0 = _mm_load_ps(w0);
			__m128 mdiv1 = _mm_load_ps(w1);
			__m128 mdiv2 = _mm_load_ps(w2);
			_mm_store_ps(s0, _mm_div_ps(md0, mdiv0));
			_mm_store_ps(s1, _mm_div_ps(md1, mdiv1));
			_mm_store_ps(s2, _mm_div_ps(md2, mdiv2));

			s0 += 4;
			s1 += 4;
			s2 += 4;

			w0 += 4;
			w1 += 4;
			w2 += 4;
		}
	}

	void DenoiseDXTShrinkage::div(float* inplace0, float* inplace1, float* inplace2, const int patch_area, const int size1)
	{
		float* s0 = inplace0;
		float* s1 = inplace1;
		float* s2 = inplace2;
		const __m128 mdiv = _mm_set1_ps(1.f / (float)patch_area);
		for (int i = 0; i < size1; i += 4)
		{
			__m128 md0 = _mm_load_ps(s0);
			__m128 md1 = _mm_load_ps(s1);
			__m128 md2 = _mm_load_ps(s2);
			_mm_store_ps(s0, _mm_mul_ps(md0, mdiv));
			_mm_store_ps(s1, _mm_mul_ps(md1, mdiv));
			_mm_store_ps(s2, _mm_mul_ps(md2, mdiv));

			s0 += 4;
			s1 += 4;
			s2 += 4;
		}
	}


	void DenoiseDXTShrinkage::div(float* inplace0, float* wmap, const int size1)
	{
		float* s0 = inplace0;
		float* w0 = wmap;
		for (int i = 0; i < size1; i += 4)
		{
			__m128 md0 = _mm_load_ps(s0);
			__m128 mdiv = _mm_load_ps(w0);
			_mm_store_ps(s0, _mm_div_ps(md0, mdiv));
			s0 += 4;
			w0 += 4;
		}
	}

	void DenoiseDXTShrinkage::div(float* inplace0, const int patch_area, const int size1)
	{
		float* s0 = inplace0;
		const __m128 mdiv = _mm_set1_ps(1.f / (float)patch_area);
		for (int i = 0; i < size1; i += 4)
		{
			__m128 md0 = _mm_load_ps(s0);
			_mm_store_ps(s0, _mm_mul_ps(md0, mdiv));
			s0 += 4;
		}
	}
	void DenoiseDXTShrinkage::body(float *src0, float* dest0, float *src1, float* dest1, float *src2, float* dest2, float Th)
	{
		/*
		const int hstep= size.height - patch_size.height + 1;
		const int wstep = size.width - patch_size.width + 1;
		const int width = size.width;

		const int w1=width;
		const int w2=2*width;
		const int w3=3*width;
		const int w4=4*width;
		const int w5=5*width;
		const int w6=6*width;
		const int w7=7*width;
		#pragma omp parallel for
		for (int j = 0; j < hstep; j ++)
		{
		Mat temp(patch_size, CV_32F);
		Mat patch(patch_size, CV_32F);

		float* t=temp.ptr<float>(0);
		float* ptch=patch.ptr<float>(0);

		float* s0= &src0[width*j];
		float* s1= &src1[width*j];
		float* s2= &src2[width*j];
		float* d0= &dest0[width*j];
		float* d1= &dest1[width*j];
		float* d2= &dest2[width*j];

		const int sz = sizeof(float)*8;

		for (int i = 0; i < wstep; i ++)
		{
		memcpy(ptch   ,s0,sz);
		memcpy(ptch+ 8,s0+w1,sz);
		memcpy(ptch+16,s0+w2,sz);
		memcpy(ptch+24,s0+w3,sz);
		memcpy(ptch+32,s0+w4,sz);
		memcpy(ptch+40,s0+w5,sz);
		memcpy(ptch+48,s0+w6,sz);
		memcpy(ptch+56,s0+w7,sz);
		fDCT2DandThresholdandIDCT(ptch, Th, t);

		//add data
		for (int jp = 0; jp < 8; jp ++)
		{
		float* s =patch.ptr<float>(jp);
		float* d = &d0[(jp)*width];
		__m128 mp1 = _mm_load_ps(s);
		__m128 sp1 = _mm_loadu_ps(d);

		_mm_storeu_ps(d,_mm_add_ps(sp1,mp1));

		s+=4;
		d+=4;

		mp1 = _mm_load_ps(s);
		sp1 = _mm_loadu_ps(d);
		_mm_storeu_ps(d,_mm_add_ps(sp1,mp1));
		}
		s0++,d0++;
		}
		for (int i = 0; i < wstep; i ++)
		{
		memcpy(ptch   ,s1,sz);
		memcpy(ptch+ 8,s1+w1,sz);
		memcpy(ptch+16,s1+w2,sz);
		memcpy(ptch+24,s1+w3,sz);
		memcpy(ptch+32,s1+w4,sz);
		memcpy(ptch+40,s1+w5,sz);
		memcpy(ptch+48,s1+w6,sz);
		memcpy(ptch+56,s1+w7,sz);

		fDCT2DandThresholdandIDCT(ptch, Th, t);
		//add data
		for (int jp = 0; jp < 8; jp ++)
		{
		float* s =patch.ptr<float>(jp);
		float* d = &d1[(jp)*width];
		__m128 mp1 = _mm_load_ps(s);
		__m128 sp1 = _mm_loadu_ps(d);

		_mm_storeu_ps(d,_mm_add_ps(sp1,mp1));

		s+=4;
		d+=4;

		mp1 = _mm_load_ps(s);
		sp1 = _mm_loadu_ps(d);
		_mm_storeu_ps(d,_mm_add_ps(sp1,mp1));
		}
		s1++,d1++;
		}
		for (int i = 0; i < wstep; i ++)
		{
		memcpy(ptch   ,s2,sz);
		memcpy(ptch+ 8,s2+w1,sz);
		memcpy(ptch+16,s2+w2,sz);
		memcpy(ptch+24,s2+w3,sz);
		memcpy(ptch+32,s2+w4,sz);
		memcpy(ptch+40,s2+w5,sz);
		memcpy(ptch+48,s2+w6,sz);
		memcpy(ptch+56,s2+w7,sz);

		fDCT2DandThresholdandIDCT(ptch, Th, t);
		//add data
		for (int jp = 0; jp < 8; jp ++)
		{
		float* s =patch.ptr<float>(jp);
		float* d = &d2[(jp)*width];
		__m128 mp1 = _mm_load_ps(s);
		__m128 sp1 = _mm_loadu_ps(d);

		_mm_storeu_ps(d,_mm_add_ps(sp1,mp1));

		s+=4;
		d+=4;

		mp1 = _mm_load_ps(s);
		sp1 = _mm_loadu_ps(d);
		_mm_storeu_ps(d,_mm_add_ps(sp1,mp1));
		}
		s2++,d2++;
		}
		}
		*/
	}

	void DenoiseDXTShrinkage::bodyTest(float *src, float* dest, float Th)
	{
		DenoiseDCTShrinkageInvorkerTest invork(src, dest, Th, size.width, size.height, patch_size);
		parallel_for_(Range(0, size.height - patch_size.height), invork);
	}


	void DenoiseDXTShrinkage::body(float *src, float* dest, float* wmap, float Th)
	{
		if (basis == DenoiseDCT)
		{
			if (isSSE)
			{
				if (patch_size.width == 2)
				{
					DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 4)
				{
					DenoiseWeightedDCTShrinkageInvorker4x4 invork(src, dest, wmap, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
				}
				else if (patch_size.width == 8)
				{
					DenoiseWeightedDCTShrinkageInvorker8x8 invork(src, dest, wmap, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
				}
				else
				{
					DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else if (basis == DenoiseDHT)
		{
			if (isSSE)
			{
				if (patch_size.width == 2)
				{
					//2x2 is same as DCT
					DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 4)
				{
					ShearableDenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height, 0);
					//DenoiseDHTShrinkageInvorker4x4 invork(src,dest,Th, size.width,size.height);			
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12);
				}
				else if (patch_size.width == 8)
				{
					DenoiseDHTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 16)
				{
					DenoiseDHTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else
				{
					DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else if (basis == DenoiseDWT)
		{
			if (patch_size.width == 4)
			{
				DenoiseDWTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
				//DenoiseDHTShrinkageInvorker4x4S invork(src,dest,Th, size.width,size.height);			
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDWTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
	}

	void DenoiseDXTShrinkage::body(float *src, float* dest, float Th)
	{
		if (basis == DenoiseDCT)
		{
			if (isSSE)
			{
				if (patch_size.width == 2)
				{
					DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 4)
				{
					DenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
				}
				else if (patch_size.width == 8)
				{
					DenoiseDCTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
				}
				else
				{
					DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else if (basis == DenoiseDHT)
		{
			if (isSSE)
			{
				if (patch_size.width == 2)
				{
					//2x2 is same as DCT
					DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 4)
				{
					ShearableDenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height, 0);
					//DenoiseDHTShrinkageInvorker4x4 invork(src,dest,Th, size.width,size.height);			
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12);
				}
				else if (patch_size.width == 8)
				{
					DenoiseDHTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 16)
				{
					DenoiseDHTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else
				{
					DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else if (basis == DenoiseDWT)
		{
			if (patch_size.width == 4)
			{
				DenoiseDWTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
				//DenoiseDHTShrinkageInvorker4x4S invork(src,dest,Th, size.width,size.height);			
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDWTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
	}


	void DenoiseDXTShrinkage::body(float *src, float* dest, float Th, int dr)
	{
		if (basis == DenoiseDCT)
		{
			if (isSSE)
			{
				if (patch_size.width == 2)
				{
					DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 4)
				{
					ShearableDenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height, dr);
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
				}
				else if (patch_size.width == 8)
				{
					DenoiseDCTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else
				{
					DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else if (basis == DenoiseDHT)
		{
			if (isSSE)
			{
				if (patch_size.width == 2)
				{
					//2x2 is same as DCT
					DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 4)
				{
					ShearableDenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height, dr);
					//DenoiseDHTShrinkageInvorker4x4 invork(src,dest,Th, size.width,size.height);			
					parallel_for_(Range(0, size.height - patch_size.height), invork, 12);
				}
				else if (patch_size.width == 8)
				{
					DenoiseDHTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else if (patch_size.width == 16)
				{
					DenoiseDHTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
				else
				{
					DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
					parallel_for_(Range(0, size.height - patch_size.height), invork);
				}
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else if (basis == DenoiseDWT)
		{
			if (patch_size.width == 4)
			{
				DenoiseDWTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
				//DenoiseDHTShrinkageInvorker4x4S invork(src,dest,Th, size.width,size.height);			
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDWTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
	}


	void DenoiseDXTShrinkage::shearable(Mat& src_, Mat& dest, float sigma, Size psize, int transform_basis, int direct)
	{
		Mat src;
		if (src_.depth() != CV_32F)src_.convertTo(src, CV_MAKETYPE(CV_32F, src_.channels()));
		else src = src_;

		basis = transform_basis;
		if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);

		int w = src.cols + 4 * psize.width;
		w = ((4 - w % 4) % 4);
		Mat im; copyMakeBorder(src, im, psize.height, psize.height, 2 * psize.width, 2 * psize.width + w, cv::BORDER_REPLICATE);

		const int width = im.cols;
		const int height = im.rows;
		const int size1 = width*height;
		float* ipixels;
		float* opixels;

		// Threshold
		float Th = 3 * sigma;

		// DCT window size
		{
#ifdef _CALCTIME_
			CalcTime t("color");
#endif
			if (channel == 3)
			{
				cvtColorOrder32F_BGR2BBBBGGGGRRRR(im, buff);
			}
			else
			{
				buff = im.clone();
			}

			sum = Mat::zeros(buff.size(), CV_32F);
			ipixels = buff.ptr<float>(0);
			opixels = sum.ptr<float>(0);

			if (channel == 3)
			{
				decorrelateColorForward(ipixels, ipixels, width, height);
			}
		}

	{
#ifdef _CALCTIME_
		CalcTime t("body");
#endif
		if (channel == 3)
		{
			Size s = size;
			size = Size(size.width + 2 * patch_size.width, size.height);
			body(ipixels, opixels, Th, direct);
			body(ipixels + size1, opixels + size1, Th, direct);
			body(ipixels + 2 * size1, opixels + 2 * size1, Th, direct);
			size = s;
		}
		else
		{
			body(ipixels, opixels, Th, direct);
		}
		//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

	}

	{
#ifdef _CALCTIME_
		CalcTime t("div");
#endif

		if (channel == 3)
		{
			float* d0 = &opixels[0];
			float* d1 = &opixels[size1];
			float* d2 = &opixels[2 * size1];
			div(d0, d1, d2, patch_size.area(), size1);

		}
		else
		{
			float* d0 = &opixels[0];
			div(d0, patch_size.area(), size1);
		}
	}

	{
#ifdef _CALCTIME_
		CalcTime t("inv color");
#endif
		// inverse 3-point DCT transform in the color dimension
		if (channel == 3)
		{
			decorrelateColorInvert(opixels, opixels, width, height);
			cvtColorOrder32F_BBBBGGGGRRRR2BGR(sum, im);
		}
		else
		{
			im = sum;
		}

		Mat im2;
		if (src_.depth() != CV_32F) im.convertTo(im2, src_.type());
		else im2 = im;

		Mat(im2(Rect(patch_size.width * 2, patch_size.height, src.cols, src.rows))).copyTo(dest);
	}
	}


	void DenoiseDXTShrinkage::weighted(Mat& src_, Mat& dest, float sigma, Size psize, int transform_basis)
	{
		Mat src;
		if (src_.depth() != CV_32F)src_.convertTo(src, CV_MAKETYPE(CV_32F, src_.channels()));
		else src = src_;

		basis = transform_basis;
		if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);

		int w = src.cols + 2 * psize.width;
		w = ((4 - w % 4) % 4);

		copyMakeBorder(src, im, psize.height, psize.height, psize.width, psize.width + w, cv::BORDER_REPLICATE);

		const int width = im.cols;
		const int height = im.rows;
		const int size1 = width*height;
		float* ipixels;
		float* opixels;

		// Threshold
		float Th = 3 * sigma;

		Mat weight0 = Mat::zeros(Size(width, height), CV_32F);
		Mat weight1 = Mat::zeros(Size(width, height), CV_32F);
		Mat weight2 = Mat::zeros(Size(width, height), CV_32F);
		// DCT window size
		{
#ifdef _CALCTIME_
			CalcTime t("color");
#endif
			if (channel == 3)
			{
				cvtColorOrder32F_BGR2BBBBGGGGRRRR(im, buff);
			}
			else
			{
				buff = im.clone();
			}

			sum = Mat::zeros(buff.size(), CV_32F);
			ipixels = buff.ptr<float>(0);
			opixels = sum.ptr<float>(0);

			if (channel == 3)
			{
				decorrelateColorForward(ipixels, ipixels, width, height);
			}
		}

	{
#ifdef _CALCTIME_
		CalcTime t("body");
#endif
		if (channel == 3)
		{
			float* w0 = weight0.ptr<float>(0);
			float* w1 = weight1.ptr<float>(0);
			float* w2 = weight2.ptr<float>(0);

			body(ipixels, opixels, w0, Th);
			body(ipixels + size1, opixels + size1, w1, Th);
			body(ipixels + 2 * size1, opixels + 2 * size1, w2, Th);
		}
		else
		{
			body(ipixels, opixels, (float*)weight0.data, Th);
		}
		//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

	}

	{
#ifdef _CALCTIME_
		CalcTime t("div");
#endif

		if (channel == 3)
		{
			float* d0 = &opixels[0];
			float* d1 = &opixels[size1];
			float* d2 = &opixels[2 * size1];

			float* w0 = weight0.ptr<float>(0);
			float* w1 = weight1.ptr<float>(0);
			float* w2 = weight2.ptr<float>(0);

			div(d0, d1, d2, w0, w1, w2, size1);

			//guiAlphaBlend(weight0,weight1);
			//guiAlphaBlend(weight0,weight2);
		}
		else
		{
			float* d0 = &opixels[0];
			float* w0 = (float*)weight0.data;
			div(d0, w0, size1);
		}
	}

	{
#ifdef _CALCTIME_
		CalcTime t("inv color");
#endif
		// inverse 3-point DCT transform in the color dimension
		if (channel == 3)
		{
			decorrelateColorInvert(opixels, opixels, width, height);
			cvtColorOrder32F_BBBBGGGGRRRR2BGR(sum, im);
		}
		else
		{
			im = sum;
		}

		Mat im2;
		if (src_.depth() != CV_32F) im.convertTo(im2, src_.type());
		else im2 = im;

		Mat(im2(Rect(patch_size.width, patch_size.height, src.cols, src.rows))).copyTo(dest);
	}
	}

	void DenoiseDXTShrinkage::operator()(Mat& src_, Mat& dest, float sigma, Size psize, int transform_basis)
	{
		Mat src;
		if (src_.depth() != CV_32F)src_.convertTo(src, CV_MAKETYPE(CV_32F, src_.channels()));
		else src = src_;

		basis = transform_basis;
		if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);

		int w = src.cols + 2 * psize.width;
		w = ((4 - w % 4) % 4);

		copyMakeBorder(src, im, psize.height, psize.height, psize.width, psize.width + w, cv::BORDER_REPLICATE);

		const int width = im.cols;
		const int height = im.rows;
		const int size1 = width*height;
		float* ipixels;
		float* opixels;

		// Threshold
		float Th = 3 * sigma;

		// DCT window size
		{
#ifdef _CALCTIME_
			CalcTime t("color");
#endif
			if (channel == 3)
			{
				cvtColorOrder32F_BGR2BBBBGGGGRRRR(im, buff);
			}
			else
			{
				buff = im.clone();
			}

			sum = Mat::zeros(buff.size(), CV_32F);
			ipixels = buff.ptr<float>(0);
			opixels = sum.ptr<float>(0);

			if (channel == 3)
			{
				decorrelateColorForward(ipixels, ipixels, width, height);
			}
		}

	{
#ifdef _CALCTIME_
		CalcTime t("body");
#endif
		if (channel == 3)
		{
			body(ipixels, opixels, Th);
			body(ipixels + size1, opixels + size1, Th);
			body(ipixels + 2 * size1, opixels + 2 * size1, Th);
		}
		else
		{
			body(ipixels, opixels, Th);
		}
		//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

	}

	{
#ifdef _CALCTIME_
		CalcTime t("div");
#endif

		if (channel == 3)
		{
			float* d0 = &opixels[0];
			float* d1 = &opixels[size1];
			float* d2 = &opixels[2 * size1];
			div(d0, d1, d2, patch_size.area(), size1);
		}
		else
		{
			float* d0 = &opixels[0];
			div(d0, patch_size.area(), size1);
		}
	}

	{
#ifdef _CALCTIME_
		CalcTime t("inv color");
#endif
		// inverse 3-point DCT transform in the color dimension
		if (channel == 3)
		{
			decorrelateColorInvert(opixels, opixels, width, height);
			cvtColorOrder32F_BBBBGGGGRRRR2BGR(sum, im);
		}
		else
		{
			im = sum;
		}

		Mat im2;
		if (src_.depth() != CV_32F) im.convertTo(im2, src_.type());
		else im2 = im;

		Mat(im2(Rect(patch_size.width, patch_size.height, src.cols, src.rows))).copyTo(dest);
	}
	}

	void DenoiseDXTShrinkage::test(Mat& src, Mat& dest, float sigma, Size psize)
	{
		if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);
		sum.setTo(0);
		copyMakeBorder(src, im, psize.height, psize.height, psize.width, psize.width, cv::BORDER_REPLICATE);

		const int width = im.cols;
		const int height = im.rows;
		const int size1 = width*height;

		float* ipixels;
		float* opixels;

		// Threshold
		float Th = 3 * sigma;

		// DCT window size
		int width_p, height_p;
		width_p = psize.width;
		height_p = psize.height;

		{
#ifdef _CALCTIME_
			CalcTime t("color");
#endif
			cvtColorOrder32F_BGR2BBBBGGGGRRRR(im, buff);
			sum = Mat::zeros(buff.size(), CV_32F);
			ipixels = buff.ptr<float>(0);
			opixels = sum.ptr<float>(0);

			decorrelateColorForward(ipixels, ipixels, width, height);
		}

	{
#ifdef _CALCTIME_
		CalcTime t("body");
#endif
		bodyTest(ipixels, opixels, Th);
		bodyTest(ipixels + size1, opixels + size1, Th);
		bodyTest(ipixels + 2 * size1, opixels + 2 * size1, Th);
		//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

	}
	{
#ifdef _CALCTIME_
		CalcTime t("div");
#endif
		float* d0 = &opixels[0];
		float* d1 = &opixels[size1];
		float* d2 = &opixels[2 * size1];
		div(d0, d1, d2, patch_size.area(), size1);
	}

	{
#ifdef _CALCTIME_
		CalcTime t("inv color");
#endif
		// inverse 3-point DCT transform in the color dimension

		decorrelateColorInvert(opixels, opixels, width, height);
		cvtColorOrder32F_BBBBGGGGRRRR2BGR(sum, im);
		Mat(im(Rect(patch_size.width, patch_size.height, src.cols, src.rows))).copyTo(dest);
	}
	}

	void DenoiseDXTShrinkage::decorrelateColorInvert(float* src, float* dest, int width, int height)
	{
		const float c00 = 0.57735f;
		const float c01 = 0.707107f;
		const float c02 = 0.408248f;
		const float c12 = -0.816497f;

		const int size1 = width*height;
		const int size2 = 2 * size1;


		//#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			float* s0 = src + width*j;
			float* s1 = s0 + size1;
			float* s2 = s0 + size2;
			float* d0 = dest + width*j;
			float* d1 = d0 + size1;
			float* d2 = d0 + size2;

			const __m128 mc00 = _mm_set1_ps(c00);
			const __m128 mc01 = _mm_set1_ps(c01);
			const __m128 mc02 = _mm_set1_ps(c02);
			const __m128 mc12 = _mm_set1_ps(c12);
			int i = 0;
			//#ifdef _SSE
			for (i = 0; i < width - 4; i += 4)
			{
				__m128 ms0 = _mm_load_ps(s0);
				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(s2);

				__m128 cs000 = _mm_mul_ps(mc00, ms0);
				__m128 cs002 = _mm_add_ps(cs000, _mm_mul_ps(mc02, ms2));
				_mm_store_ps(d0, _mm_add_ps(cs002, _mm_mul_ps(mc01, ms1)));
				_mm_store_ps(d1, _mm_add_ps(cs000, _mm_mul_ps(mc12, ms2)));
				_mm_store_ps(d2, _mm_sub_ps(cs002, _mm_mul_ps(mc01, ms1)));

				d0 += 4, d1 += 4, d2 += 4, s0 += 4, s1 += 4, s2 += 4;
			}
			//#endif
			for (; i < width; i++)
			{
				float v0 = c00* *s0 + c01* *s1 + c02* *s2;
				float v1 = c00* *s0 + c12* *s2;
				float v2 = c00* *s0 - c01* *s1 + c02* *s2;

				*d0++ = v0;
				*d1++ = v1;
				*d2++ = v2;
				s0++, s1++, s2++;
			}
		}
	}
	
	void DenoiseDXTShrinkage::decorrelateColorForward(float* src, float* dest, int width,
		int height)
	{
		const float c0 = 0.57735f;
		const float c1 = 0.707107f;
		const float c20 = 0.408248f;
		const float c21 = -0.816497f;

		const int size1 = width*height;
		const int size2 = 2 * size1;

		//#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			float* s0 = src + width*j;
			float* s1 = s0 + size1;
			float* s2 = s0 + size2;
			float* d0 = dest + width*j;
			float* d1 = d0 + size1;
			float* d2 = d0 + size2;
			const __m128 mc0 = _mm_set1_ps(c0);
			const __m128 mc1 = _mm_set1_ps(c1);
			const __m128 mc20 = _mm_set1_ps(c20);
			const __m128 mc21 = _mm_set1_ps(c21);
			int i = 0;
			//#ifdef _SSE
			for (i = 0; i < width - 4; i += 4)
			{
				__m128 ms0 = _mm_load_ps(s0);
				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(s2);

				__m128 ms02a = _mm_add_ps(ms0, ms2);

				_mm_store_ps(d0, _mm_mul_ps(mc0, _mm_add_ps(ms1, ms02a)));
				_mm_store_ps(d1, _mm_mul_ps(mc1, _mm_sub_ps(ms0, ms2)));
				_mm_store_ps(d2, _mm_add_ps(_mm_mul_ps(mc20, ms02a), _mm_mul_ps(mc21, ms1)));

				d0 += 4, d1 += 4, d2 += 4, s0 += 4, s1 += 4, s2 += 4;
			}
			//#endif
			for (; i < width; i++)
			{
				float v0 = c0*(*s0 + *s1 + *s2);
				float v1 = c1*(*s0 - *s2);
				float v2 = (*s0 + *s2)*c20 + *s1 *c21;

				*d0++ = v0;
				*d1++ = v1;
				*d2++ = v2;
				s0++, s1++, s2++;
			}
		}
	}
}