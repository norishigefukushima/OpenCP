#include "costVolumeFilter.hpp"
#include "guidedFilter.hpp"
#include "jointBilateralFilter.hpp"
#include "color.hpp"

using namespace std;
using namespace cv;

namespace cp
{

class jointBilateralRefinement_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	jointBilateralRefinement_8u_InvokerSSE4(vector<Mat>& _destVolume, const vector<Mat>& _tempVolume, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight) :
		tempVolume(_tempVolume), destVolume(_destVolume), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
		;
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, cn = 1, k;
		int cng = (guide->rows) / tempVolume[0].rows;
		//cout<<(guide->rows) <<","<< tempVolume[0].rows<<","<<cng<<endl;
		Size size = destVolume[0].size();
		//imshow("wwww",weightMap);waitKey();

#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1 && cng == 1)
		{
			uchar CV_DECL_ALIGNED(16) buf[16];
			Mat coeff = Mat::zeros(Size(16 * (maxk), 1), CV_32F);
			vector<float*> sptr(tempVolume.size());
			vector<float*> sptrj(tempVolume.size());
			vector<float*> dptr(tempVolume.size());
			vector<float*> dptrj(tempVolume.size());
			for (int n = 0; n < (int)tempVolume.size(); n++)
			{
				sptr[n] = (float*)tempVolume[n].ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				dptr[n] = (float*)destVolume[n].ptr<float>(range.start);
			}
			uchar* gptr = (uchar*)guide->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);

			const int sstep = tempVolume[0].cols;
			const int gstep = guide->cols;
			const int dstep = destVolume[0].cols;

			//cout<<sstep<<","<<gstep<<","<<dstep<<endl;
			//for(int n=0;n<(int)tempVolume.size();n++){imshow("dest",destVolume[n]);imshow("src",tempVolume[n]);waitKey();}

			//int skipcount=0;
			for (i = range.start; i != range.end; i++, gptr += gstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 16)//16画素づつ処理
						//for(; j < 0; j+=16)//16画素づつ処理
					{
						int* ofs = &space_ofs[0];
						int* gofs = &space_guide_ofs[0];
						float* spw = space_weight;

						const uchar* gptrj = gptr + j;
						const __m128i sval = _mm_load_si128((__m128i*)(gptrj));
						float* cef = coeff.ptr<float>(0);
						float  CV_DECL_ALIGNED(16) normal[16];
						memset(normal, 0, sizeof(float) * 16);
						for (k = 0; k < maxk; k++, gofs++)
						{
							const float spk = space_weight[k];
							const __m128i sref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
							_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, sref), _mm_subs_epu8(sref, sval)));

							for (int n = 0; n < 16; n++)
							{
								*cef = spk*color_weight[buf[n]];
								normal[n] += *cef;
								cef++;
							}
						}

						for (int n = 0; n < (int)tempVolume.size(); n++)
						{

							for (int m = 0; m < 16; m += 4)
							{
								sptrj[n] = sptr[n] + j + m;
								dptrj[n] = dptr[n] + j + m;
								bool flag = true;
								for (int idx = 0; idx < 4; idx++)
								{
									if (dptrj[n][idx] != sptrj[n][idx])
									{
										flag = false;
									}
								}
								if (flag)
								{
									//skipcount+=4;
									_mm_stream_ps(dptrj[n], _mm_mul_ps(_mm_load_ps(dptrj[n]), _mm_load_ps(normal + m)));
								}
								else
								{
									ofs = &space_ofs[0];
									cef = coeff.ptr<float>(0); cef += m;
									__m128 tval1 = _mm_setzero_ps();
									ofs = &space_ofs[0];
									for (k = 0; k < maxk; k++, ofs++, cef += 16)
									{
										__m128 _w = _mm_load_ps(cef);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
										__m128 _valF = _mm_loadu_ps((sptrj[n] + *ofs));
										_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
										tval1 = _mm_add_ps(tval1, _valF);
									}
									_mm_stream_ps((dptrj[n]), tval1);
								}
							}
						}
					}
				}
#endif
				/*	for(; j < size.width; j++)
				{
				const uchar val0 = gptr[j];
				float sum=0.0f;
				float wsum=0.0f;
				for(k=0 ; k < maxk; k++ )
				{
				/*int gval = gptr[j + space_guide_ofs[k]];
				int val = sptr[j + space_ofs[k]];
				float w = wptr[j + space_w_ofs[k]]*space_weight[k]*color_weight[std::abs(gval - val0)];
				sum += val*w;
				wsum += w;
				}
				//overflow is not possible here => there is no need to use CV_CAST_8U
				dptr[0][j] = (uchar)cvRound(sum/wsum);
				}*/
				for (int n = 0; n < (int)tempVolume.size(); n++)
				{
					sptr[n] += sstep;
					dptr[n] += dstep;
				}
			}
			//cout<<"skiprate: "<<100.0*(double)skipcount/(tempVolume.size()*size.area())<<endl;
		}
		else if (cn == 1 && cng == 3)
		{
			short CV_DECL_ALIGNED(16) buf[16];
			Mat coeff = Mat::zeros(Size(16 * (maxk + 1), 1), CV_32F);
			vector<float*> sptr(tempVolume.size());
			vector<float*> sptrj(tempVolume.size());
			vector<float*> dptr(tempVolume.size());
			for (int n = 0; n < (int)tempVolume.size(); n++)
			{
				sptr[n] = (float*)tempVolume[n].ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				dptr[n] = (float*)destVolume[n].ptr<float>(range.start);
			}
			uchar* gptrr = (uchar*)guide->ptr(3 * (range.start + radiusV) + 0) + 16 * (radiusH / 16 + 1);
			uchar* gptrg = (uchar*)guide->ptr(3 * (range.start + radiusV) + 1) + 16 * (radiusH / 16 + 1);
			uchar* gptrb = (uchar*)guide->ptr(3 * (range.start + radiusV) + 2) + 16 * (radiusH / 16 + 1);

			const int sstep = tempVolume[0].cols;
			const int gstep = guide->cols * 3;
			const int dstep = destVolume[0].cols;

			//cout<<sstep<<","<<gstep<<","<<dstep<<endl;
			for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					//cout<<i<<" ";
					for (; j < size.width; j += 16)//16画素づつ処理
						//for(; j < 0; j+=16)//16画素づつ処理
					{
						int* ofs = &space_ofs[0];
						int* gofs = &space_guide_ofs[0];
						float* spw = space_weight;

						const uchar* gptrrj = gptrr + j;
						const uchar* gptrgj = gptrg + j;
						const uchar* gptrbj = gptrb + j;

						const __m128i bval = _mm_load_si128((__m128i*)(gptrbj));
						const __m128i gval = _mm_load_si128((__m128i*)(gptrgj));
						const __m128i rval = _mm_load_si128((__m128i*)(gptrrj));

						float* cef = coeff.ptr<float>(0);

						__m128i m1, m2, n1, n2;
						const __m128i zero = _mm_setzero_si128();
						for (k = 0; k <= maxk; k++, gofs++)
						{
							const float spk = space_weight[k];
							const __m128i bref = _mm_loadu_si128((__m128i*)(gptrbj + *gofs));
							const __m128i gref = _mm_loadu_si128((__m128i*)(gptrgj + *gofs));
							const __m128i rref = _mm_loadu_si128((__m128i*)(gptrrj + *gofs));

							m1 = _mm_add_epi8(_mm_subs_epu8(rval, rref), _mm_subs_epu8(rref, rval));
							m2 = _mm_unpackhi_epi8(m1, zero);
							m1 = _mm_unpacklo_epi8(m1, zero);

							n1 = _mm_add_epi8(_mm_subs_epu8(gval, gref), _mm_subs_epu8(gref, gval));
							n2 = _mm_unpackhi_epi8(n1, zero);
							n1 = _mm_unpacklo_epi8(n1, zero);

							m1 = _mm_add_epi16(m1, n1);
							m2 = _mm_add_epi16(m2, n2);

							n1 = _mm_add_epi8(_mm_subs_epu8(bval, bref), _mm_subs_epu8(bref, bval));
							n2 = _mm_unpackhi_epi8(n1, zero);
							n1 = _mm_unpacklo_epi8(n1, zero);

							m1 = _mm_add_epi16(m1, n1);
							m2 = _mm_add_epi16(m2, n2);

							_mm_store_si128((__m128i*)(buf + 8), m2);
							_mm_store_si128((__m128i*)buf, m1);

							for (int n = 0; n < 16; n++)
							{
								*cef = spk*color_weight[buf[n]];
								cef++;
							}
						}

						for (int n = 0; n < (int)tempVolume.size(); n++)
						{
							ofs = &space_ofs[0];
							cef = coeff.ptr<float>(0);
							__m128 _valF, _w;
							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();
							sptrj[n] = sptr[n] + j;
							for (k = 0; k <= maxk; k++, ofs++, cef += 16)
							{
								_w = _mm_load_ps(cef);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）

								_valF = _mm_loadu_ps((sptrj[n] + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval1 = _mm_add_ps(tval1, _valF);

								_w = _mm_load_ps(cef + 4);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 4 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval2 = _mm_add_ps(tval2, _valF);

								_w = _mm_load_ps(cef + 8);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 8 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval3 = _mm_add_ps(tval3, _valF);

								_w = _mm_load_ps(cef + 12);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 12 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval4 = _mm_add_ps(tval4, _valF);
							}
							_mm_stream_ps((dptr[n] + j), tval1);
							_mm_stream_ps((dptr[n] + j + 4), tval2);
							_mm_stream_ps((dptr[n] + j + 8), tval3);
							_mm_stream_ps((dptr[n] + j + 12), tval4);
						}
					}
				}
#endif
				/*	for(; j < size.width; j++)
				{
				const uchar val0 = gptr[j];
				float sum=0.0f;
				float wsum=0.0f;
				for(k=0 ; k < maxk; k++ )
				{
				/*int gval = gptr[j + space_guide_ofs[k]];
				int val = sptr[j + space_ofs[k]];
				float w = wptr[j + space_w_ofs[k]]*space_weight[k]*color_weight[std::abs(gval - val0)];
				sum += val*w;
				wsum += w;
				}
				//overflow is not possible here => there is no need to use CV_CAST_8U
				dptr[0][j] = (uchar)cvRound(sum/wsum);
				}*/
				for (int n = 0; n < (int)tempVolume.size(); n++)
				{
					sptr[n] += sstep;
					dptr[n] += dstep;
				}
			}
		}
	}
private:
	vector<Mat> tempVolume;
	vector<Mat> destVolume;
	const Mat* guide;
	int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
	float *space_weight, *color_weight;
};

void jointBilateralRefinement_8u(const vector<Mat>& srcVolume, const Mat& guide, vector<Mat>& destVolume, Size kernelSize, double sigma_color, double sigma_space, int borderType)
{
	//if(kernelSize.width<=1 && kernelSize.height<=1){ src.copyTo(dst);return;}
	int cn = 1;
	int cng = guide.channels();
	int i, j, maxk;
	Size size = srcVolume[0].size();

	CV_Assert(
		(guide.type() == CV_8UC1 || guide.type() == CV_8UC3));

	if (sigma_color <= 0)
		sigma_color = 1;
	if (sigma_space <= 0)
		sigma_space = 1;

	double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
	double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

	int radiusH = kernelSize.width >> 1;
	int radiusV = kernelSize.height >> 1;

	Mat tempg;

	int dpad = (16 - size.width % 16) % 16;
	int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
	if (spad < 16) spad += 16;
	int lpad = 16 * (radiusH / 16 + 1) - radiusH;
	int rpad = spad - lpad;

	vector<Mat> tempVolume(srcVolume.size());
	if (cn == 1 && cng == 1)
	{
		for (int n = 0; n < (int)srcVolume.size(); n++)
		{
			copyMakeBorder(srcVolume[n], tempVolume[n], radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
	}
	else if (cn == 1 && cng == 3)
	{
		for (int n = 0; n < (int)srcVolume.size(); n++)
		{
			copyMakeBorder(srcVolume[n], tempVolume[n], radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		Mat temp2;
		copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		splitBGRLineInterleave(temp2, tempg);

	}


	vector<float> _color_weight(cng * 256);
	vector<float> _space_weight(kernelSize.area() + 1);
	vector<int> _space_ofs(kernelSize.area() + 1);
	vector<int> _space_guide_ofs(kernelSize.area() + 1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];
	int* space_guide_ofs = &_space_guide_ofs[0];



	// initialize color-related bilateral filter coefficients

	for (i = 0; i < 256 * cng; i++)
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for (i = -radiusV, maxk = 0; i <= radiusV; i++)
	{
		j = -radiusH;

		for (; j <= radiusH; j++)
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if (r > max(radiusV, radiusH))
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			//space_ofs[maxk++] = (int)(i*temp.step + j*cn);
			space_ofs[maxk] = (int)(i*tempVolume[0].cols*cn + j);
			space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
		}
	}

	vector<Mat> dstVolume(srcVolume.size());
	for (int n = 0; n < (int)srcVolume.size(); n++)
	{
		//dstVolume[n] = Mat::zeros(Size(size.width+dpad, size.height),CV_32F);
		copyMakeBorder(destVolume[n], dstVolume[n], 0, 0, 0, dpad, borderType);
	}

	jointBilateralRefinement_8u_InvokerSSE4 body(dstVolume, tempVolume, tempg, radiusH, radiusV, maxk, space_ofs, space_guide_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);

	for (int n = 0; n < (int)srcVolume.size(); n++)
	{
		Mat(dstVolume[n](Rect(0, 0, size.width, size.height))).copyTo(destVolume[n]);
		//imshow("dv",dstVolume[n]);imshow("idv",tempVolume[n]);waitKey();	
	}

}

class jointBilateralRefinement2_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	jointBilateralRefinement2_8u_InvokerSSE4(vector<Mat>& _destVolume, const vector<Mat>& _tempVolume, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight) :
		tempVolume(_tempVolume), destVolume(_destVolume), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, cn = 1, k;
		int cng = (guide->rows) / tempVolume[0].rows;
		//cout<<(guide->rows) <<","<< tempVolume[0].rows<<","<<cng<<endl;
		Size size = destVolume[0].size();
		//imshow("wwww",weightMap);waitKey();

#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1 && cng == 1)
		{
			uchar CV_DECL_ALIGNED(16) buf[16];
			Mat coeff = Mat::zeros(Size(16 * (maxk + 1), 1), CV_32F);
			vector<float*> sptr(tempVolume.size());
			vector<float*> sptrj(tempVolume.size());
			vector<float*> dptr(tempVolume.size());
			for (int n = 0; n < (int)tempVolume.size(); n++)
			{
				sptr[n] = (float*)tempVolume[n].ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				dptr[n] = (float*)destVolume[n].ptr<float>(range.start);
			}
			uchar* gptr = (uchar*)guide->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);

			const int sstep = tempVolume[0].cols;
			const int gstep = guide->cols;
			const int dstep = destVolume[0].cols;

			//cout<<sstep<<","<<gstep<<","<<dstep<<endl;
			for (i = range.start; i != range.end; i++, gptr += gstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 16)//16画素づつ処理
						//for(; j < 0; j+=16)//16画素づつ処理
					{
						int* ofs = &space_ofs[0];
						int* gofs = &space_guide_ofs[0];
						float* spw = space_weight;

						const uchar* gptrj = gptr + j;
						const __m128i sval = _mm_load_si128((__m128i*)(gptrj));
						float* cef = coeff.ptr<float>(0);
						for (k = 0; k <= maxk; k++, gofs++)
						{
							const float spk = space_weight[k];
							const __m128i sref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
							_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, sref), _mm_subs_epu8(sref, sval)));

							for (int n = 0; n < 16; n++)
							{
								*cef = spk*color_weight[buf[n]];
								cef++;
							}
						}


						for (int n = 0; n < (int)tempVolume.size(); n++)
						{
							sptrj[n] = sptr[n] + j;
							ofs = &space_ofs[0];
							cef = coeff.ptr<float>(0);
							__m128 _valF, _w;
							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();
							for (k = 0; k <= maxk; k++, ofs++, cef += 16)
							{
								_w = _mm_load_ps(cef);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）

								_valF = _mm_loadu_ps((sptrj[n] + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval1 = _mm_add_ps(tval1, _valF);

								_w = _mm_load_ps(cef + 4);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 4 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval2 = _mm_add_ps(tval2, _valF);

								_w = _mm_load_ps(cef + 8);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 8 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval3 = _mm_add_ps(tval3, _valF);

								_w = _mm_load_ps(cef + 12);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 12 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval4 = _mm_add_ps(tval4, _valF);
							}
							_mm_stream_ps((dptr[n] + j), tval1);
							_mm_stream_ps((dptr[n] + j + 4), tval2);
							_mm_stream_ps((dptr[n] + j + 8), tval3);
							_mm_stream_ps((dptr[n] + j + 12), tval4);
						}
					}
				}
#endif
				/*	for(; j < size.width; j++)
				{
				const uchar val0 = gptr[j];
				float sum=0.0f;
				float wsum=0.0f;
				for(k=0 ; k < maxk; k++ )
				{
				/*int gval = gptr[j + space_guide_ofs[k]];
				int val = sptr[j + space_ofs[k]];
				float w = wptr[j + space_w_ofs[k]]*space_weight[k]*color_weight[std::abs(gval - val0)];
				sum += val*w;
				wsum += w;
				}
				//overflow is not possible here => there is no need to use CV_CAST_8U
				dptr[0][j] = (uchar)cvRound(sum/wsum);
				}*/
				for (int n = 0; n < (int)tempVolume.size(); n++)
				{
					sptr[n] += sstep;
					dptr[n] += dstep;
				}
			}
		}
		else if (cn == 1 && cng == 3)
		{
			short CV_DECL_ALIGNED(16) buf[16];
			Mat coeff = Mat::zeros(Size(16 * (maxk + 1), 1), CV_32F);
			vector<float*> sptr(tempVolume.size());
			vector<float*> sptrj(tempVolume.size());
			vector<float*> dptr(tempVolume.size());
			for (int n = 0; n < (int)tempVolume.size(); n++)
			{
				sptr[n] = (float*)tempVolume[n].ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				dptr[n] = (float*)destVolume[n].ptr<float>(range.start);
			}
			uchar* gptrr = (uchar*)guide->ptr(3 * (range.start + radiusV) + 0) + 16 * (radiusH / 16 + 1);
			uchar* gptrg = (uchar*)guide->ptr(3 * (range.start + radiusV) + 1) + 16 * (radiusH / 16 + 1);
			uchar* gptrb = (uchar*)guide->ptr(3 * (range.start + radiusV) + 2) + 16 * (radiusH / 16 + 1);

			const int sstep = tempVolume[0].cols;
			const int gstep = guide->cols * 3;
			const int dstep = destVolume[0].cols;

			//cout<<sstep<<","<<gstep<<","<<dstep<<endl;
			for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					//cout<<i<<" ";
					for (; j < size.width; j += 16)//16画素づつ処理
						//for(; j < 0; j+=16)//16画素づつ処理
					{
						int* ofs = &space_ofs[0];
						int* gofs = &space_guide_ofs[0];
						float* spw = space_weight;

						const uchar* gptrrj = gptrr + j;
						const uchar* gptrgj = gptrg + j;
						const uchar* gptrbj = gptrb + j;

						const __m128i bval = _mm_load_si128((__m128i*)(gptrbj));
						const __m128i gval = _mm_load_si128((__m128i*)(gptrgj));
						const __m128i rval = _mm_load_si128((__m128i*)(gptrrj));

						float* cef = coeff.ptr<float>(0);

						__m128i m1, m2, n1, n2;
						const __m128i zero = _mm_setzero_si128();
						for (k = 0; k <= maxk; k++, gofs++)
						{
							const float spk = space_weight[k];
							const __m128i bref = _mm_loadu_si128((__m128i*)(gptrbj + *gofs));
							const __m128i gref = _mm_loadu_si128((__m128i*)(gptrgj + *gofs));
							const __m128i rref = _mm_loadu_si128((__m128i*)(gptrrj + *gofs));

							m1 = _mm_add_epi8(_mm_subs_epu8(rval, rref), _mm_subs_epu8(rref, rval));
							m2 = _mm_unpackhi_epi8(m1, zero);
							m1 = _mm_unpacklo_epi8(m1, zero);

							n1 = _mm_add_epi8(_mm_subs_epu8(gval, gref), _mm_subs_epu8(gref, gval));
							n2 = _mm_unpackhi_epi8(n1, zero);
							n1 = _mm_unpacklo_epi8(n1, zero);

							m1 = _mm_add_epi16(m1, n1);
							m2 = _mm_add_epi16(m2, n2);

							n1 = _mm_add_epi8(_mm_subs_epu8(bval, bref), _mm_subs_epu8(bref, bval));
							n2 = _mm_unpackhi_epi8(n1, zero);
							n1 = _mm_unpacklo_epi8(n1, zero);

							m1 = _mm_add_epi16(m1, n1);
							m2 = _mm_add_epi16(m2, n2);

							_mm_store_si128((__m128i*)(buf + 8), m2);
							_mm_store_si128((__m128i*)buf, m1);

							for (int n = 0; n < 16; n++)
							{
								*cef = spk*color_weight[buf[n]];
								cef++;
							}
						}

						for (int n = 0; n < (int)tempVolume.size(); n++)
						{
							ofs = &space_ofs[0];
							cef = coeff.ptr<float>(0);
							__m128 _valF, _w;
							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();
							sptrj[n] = sptr[n] + j;
							for (k = 0; k <= maxk; k++, ofs++, cef += 16)
							{
								_w = _mm_load_ps(cef);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）

								_valF = _mm_loadu_ps((sptrj[n] + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval1 = _mm_add_ps(tval1, _valF);

								_w = _mm_load_ps(cef + 4);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 4 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval2 = _mm_add_ps(tval2, _valF);

								_w = _mm_load_ps(cef + 8);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 8 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval3 = _mm_add_ps(tval3, _valF);

								_w = _mm_load_ps(cef + 12);//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_valF = _mm_loadu_ps((sptrj[n] + 12 + *ofs));
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval4 = _mm_add_ps(tval4, _valF);
							}
							_mm_stream_ps((dptr[n] + j), tval1);
							_mm_stream_ps((dptr[n] + j + 4), tval2);
							_mm_stream_ps((dptr[n] + j + 8), tval3);
							_mm_stream_ps((dptr[n] + j + 12), tval4);
						}
					}
				}
#endif
				/*	for(; j < size.width; j++)
				{
				const uchar val0 = gptr[j];
				float sum=0.0f;
				float wsum=0.0f;
				for(k=0 ; k < maxk; k++ )
				{
				/*int gval = gptr[j + space_guide_ofs[k]];
				int val = sptr[j + space_ofs[k]];
				float w = wptr[j + space_w_ofs[k]]*space_weight[k]*color_weight[std::abs(gval - val0)];
				sum += val*w;
				wsum += w;
				}
				//overflow is not possible here => there is no need to use CV_CAST_8U
				dptr[0][j] = (uchar)cvRound(sum/wsum);
				}*/
				for (int n = 0; n < (int)tempVolume.size(); n++)
				{
					sptr[n] += sstep;
					dptr[n] += dstep;
				}
			}
		}
	}
private:
	vector<Mat> tempVolume;
	vector<Mat> destVolume;
	const Mat* guide;
	int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
	float *space_weight, *color_weight;
};

void jointBilateralRefinement2_8u(const vector<Mat>& srcVolume, const Mat& guide, vector<Mat>& destVolume, Size kernelSize, double sigma_color, double sigma_space, int borderType)
{
	//if(kernelSize.width<=1 && kernelSize.height<=1){ src.copyTo(dst);return;}
	int cn = 1;
	int cng = guide.channels();
	int i, j, maxk;
	Size size = srcVolume[0].size();

	CV_Assert(
		(guide.type() == CV_8UC1 || guide.type() == CV_8UC3));

	if (sigma_color <= 0)
		sigma_color = 1;
	if (sigma_space <= 0)
		sigma_space = 1;

	double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
	double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

	int radiusH = kernelSize.width >> 1;
	int radiusV = kernelSize.height >> 1;

	Mat tempg;

	int dpad = (16 - size.width % 16) % 16;
	int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
	if (spad < 16) spad += 16;
	int lpad = 16 * (radiusH / 16 + 1) - radiusH;
	int rpad = spad - lpad;

	vector<Mat> tempVolume(srcVolume.size());
	if (cn == 1 && cng == 1)
	{
		for (int n = 0; n < (int)srcVolume.size(); n++)
		{
			copyMakeBorder(srcVolume[n], tempVolume[n], radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
	}
	else if (cn == 1 && cng == 3)
	{
		for (int n = 0; n < (int)srcVolume.size(); n++)
		{
			copyMakeBorder(srcVolume[n], tempVolume[n], radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		Mat temp2;
		copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		splitBGRLineInterleave(temp2, tempg);

	}


	vector<float> _color_weight(cng * 256);
	vector<float> _space_weight(kernelSize.area() + 1);
	vector<int> _space_ofs(kernelSize.area() + 1);
	vector<int> _space_guide_ofs(kernelSize.area() + 1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];
	int* space_guide_ofs = &_space_guide_ofs[0];



	// initialize color-related bilateral filter coefficients

	for (i = 0; i < 256 * cng; i++)
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for (i = -radiusV, maxk = 0; i <= radiusV; i++)
	{
		j = -radiusH;

		for (; j <= radiusH; j++)
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if (r > max(radiusV, radiusH))
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			//space_ofs[maxk++] = (int)(i*temp.step + j*cn);
			space_ofs[maxk] = (int)(i*tempVolume[0].cols*cn + j);
			space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
		}
	}

	vector<Mat> dstVolume(srcVolume.size());
	for (int n = 0; n < (int)srcVolume.size(); n++)
	{
		dstVolume[n] = Mat::zeros(Size(size.width + dpad, size.height), CV_32F);
	}

	jointBilateralRefinement2_8u_InvokerSSE4 body(dstVolume, tempVolume, tempg, radiusH, radiusV, maxk, space_ofs, space_guide_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);

	for (int n = 0; n < (int)srcVolume.size(); n++)
	{
		Mat(dstVolume[n](Rect(0, 0, size.width, size.height))).copyTo(destVolume[n]);
		//imshow("dv",dstVolume[n]);imshow("idv",tempVolume[n]);waitKey();	
	}

}












/////


#define JBF2

void CostVolumeRefinement::buildCostVolume(Mat& disp, Mat& mask, int data_trunc, int metric)
{
	Size size = disp.size();

	Mat a(size, disp.type());
	Mat v;
	for (int i = 0; i < numDisparity; i++)
	{
		a.setTo(minDisparity + i);
		if (metric == L1_NORM)
		{
			absdiff(a, disp, v);
			min(v, data_trunc, v);
		}
		else
		{

			//v=a-disp;
			//cv::subtract(a,disp,v,Mat(),VOLUME_TYPE);
			absdiff(a, disp, v);
			cv::multiply(v, v, v);
			min(v, data_trunc*data_trunc, v);
		}
		v.setTo(data_trunc, mask);
		v.convertTo(dsv[i], VOLUME_TYPE);
	}
}

void CostVolumeRefinement::buildWeightedCostVolume(Mat& disp, Mat& weightMap, int dtrunc, int metric)
{
	Size size = disp.size();


	if (metric == L1_NORM)
	{
		/*Mat a(size,disp.type());
		Mat v;
		for(int i=0;i<=numDisparity;i++)
		{
		a.setTo(minDisparity+i);
		if(metric==L1_NORM)
		{
		absdiff(a,disp,v);
		min(v,dtrunc,v);
		v.convertTo(dsv[i],VOLUME_TYPE);
		}
		}*/
	}
	else if (metric == L2_NORM)
	{
		const int imsize = size.area();
		const int rem = (4 - imsize % 4) % 4;
		for (int i = 0; i <= numDisparity; i++)
		{
			if (dsv[i].empty()) dsv[i] = Mat::zeros(size, CV_32F);
		}
		float d2 = (float)(dtrunc*dtrunc);
		const __m128 clipMax = _mm_set1_ps(d2);
		const __m128i z = _mm_setzero_si128();
		for (int i = 0; i <= numDisparity; i++)
		{
			short* dsp = disp.ptr<short>(0);
			float* vptr = dsv[i].ptr<float>(0);
			float* wptr = weightMap.ptr<float>(0);
			const __m128 ii = _mm_set1_ps((float)(i + minDisparity));

			int n = 0;
			for (; n < imsize - rem; n += 4)
			{
				__m128i d = _mm_loadl_epi64((const __m128i*)(dsp + n));
				__m128 w = _mm_load_ps((wptr + n));
				__m128 df = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d, z));

				df = _mm_sub_ps(df, ii);
				df = _mm_mul_ps(w, _mm_min_ps(_mm_mul_ps(df, df), clipMax));
				_mm_stream_ps(vptr + n, df);
			}
			for (n = imsize - rem; n < imsize; n++)
			{
				float vv = (float)(dsp[n] - (i + minDisparity));
				vv = std::min<float>(vv*vv, d2);
				vptr[n] = (float)vv*wptr[n];
			}
		}
	}

}
void CostVolumeRefinement::buildCostVolume(Mat& disp, int dtrunc, int metric)
{
	//if(metric==L2_NORM) cout<<"L2\n";
	Size size = disp.size();
	//bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);

	if (metric == L1_NORM)
	{
		Mat a(size, disp.type());
		Mat v;
		for (int i = 0; i <= numDisparity; i++)
		{
			a.setTo(minDisparity + i);
			if (metric == L1_NORM)
			{
				absdiff(a, disp, v);
				min(v, dtrunc, v);
				v.convertTo(dsv[i], VOLUME_TYPE);
			}
		}
	}
	else if (metric == EXP)
	{
		double coeff = -0.5 / (dtrunc*dtrunc);
		vector<float> _weight(256);
		float* weight = &_weight[0];
		for (int i = 0; i < 256; i++)
		{
			weight[i] = 1.0f - (float)std::exp(i*i*coeff);
			if (weight[i] < 1.0 / 255.0)weight[i] = 0.0;
		}

		//Mat a(size,disp.type());
		//Mat v;
		for (int i = 0; i <= numDisparity; i++)
		{
			if (dsv[i].empty()) dsv[i] = Mat::zeros(size, CV_32F);
			//a.setTo(minDisparity+i);
			//absdiff(a,disp,v);

			const int imsize = size.area();
			short* dsp = disp.ptr<short>(0);
			float* vptr = dsv[i].ptr<float>(0);
			for (int n = 0; n < imsize; n++)
			{
				*vptr = weight[abs(*dsp - (minDisparity + i))];
				vptr++; dsp++;
			}


			//cv::LUT(v,w,dsv[i]);
		}
	}
	else if (metric == L2_NORM)
	{
		Mat disps;
		if (disp.depth() == CV_16S)disps = disp;
		else disp.convertTo(disps, CV_16S);
		bool isSSE = true;
		if (isSSE)
		{
			const int imsize = size.area();
			const int rem = (4 - imsize % 4) % 4;
			for (int i = 0; i <= numDisparity; i++)
			{
				if (dsv[i].empty()) dsv[i] = Mat::zeros(size, CV_32F);
			}
			float d2 = (float)(dtrunc*dtrunc);
			const __m128 clipMax = _mm_set1_ps(d2);
			const __m128i z = _mm_setzero_si128();
			for (int i = 0; i <= numDisparity; i++)
			{
				short* dsp = disps.ptr<short>(0);
				float* vptr = dsv[i].ptr<float>(0);
				const __m128 ii = _mm_set1_ps((float)(i + minDisparity));

				int n = 0;
				for (; n < imsize - rem; n += 4)
				{
					__m128i d = _mm_loadl_epi64((const __m128i*)(dsp + n));
					__m128 df = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d, z));

					df = _mm_sub_ps(df, ii);
					df = _mm_min_ps(_mm_mul_ps(df, df), clipMax);
					_mm_stream_ps(vptr + n, df);
				}
				for (n = imsize - rem; n < imsize; n++)
				{
					float vv = (float)(dsp[n] - (i + minDisparity));
					vv = std::min<float>(vv*vv, d2);
					vptr[n] = (float)vv;
				}
			}
		}
		else
		{
			Mat a(size, disp.type());
			Mat v;
			for (int i = 0; i < numDisparity; i++)
			{
				a.setTo(minDisparity + i);
				//v=a-disp;
				//cv::subtract(a,disp,v,Mat(),VOLUME_TYPE);
				absdiff(a, disp, v);
				cv::multiply(v, v, v);
				min(v, dtrunc*dtrunc, v);
				v.convertTo(dsv[i], VOLUME_TYPE);
			}
		}
	}
}
void CostVolumeRefinement::wta(Mat& dest)
{
	Size size = dest.size();
	Mat cost = Mat::ones(size, VOLUME_TYPE)*FLT_MAX;
	Mat mask;
	const int imsize = size.area();
	/*for(int i=0;i<=numDisparity;i++)
	{
	Mat pcost;
	cost.copyTo(pcost);
	min(pcost,dsv[i],cost);
	compare(pcost,cost,mask,cv::CMP_NE);
	dest.setTo(i+minDisparity,mask);
	}*/

	int i = 0;
	short* dst = dest.ptr<short>(0);
	vector<float*> vp(numDisparity + 1);
	for (int n = 0; n <= numDisparity; n++)
	{
		vp[n] = dsv[n].ptr<float>(0);
	}
	for (i = 0; i < imsize - 4; i += 4)
	{
		__m128 mine = _mm_set1_ps(FLT_MAX);
		__m128 minv = _mm_setzero_ps();
		__m128i z = _mm_setzero_si128();
		for (int n = 0; n <= numDisparity; n++)
		{
			__m128 eval = _mm_load_ps(vp[n] + i);
			__m128 pmine = mine;
			mine = _mm_min_ps(mine, eval);
			__m128 msk = _mm_cmpeq_ps(pmine, mine);
			__m128 v = _mm_set1_ps((float)(n + minDisparity));
			minv = _mm_blendv_ps(v, minv, msk);
		}
		__m128i minvv = _mm_packs_epi32(_mm_cvtps_epi32(minv), z);
		_mm_storel_epi64((__m128i*)(dst + i), minvv);
	}
	for (; i < imsize; i++)
	{
		float mine = FLT_MAX;
		short minv = 0;
		for (int n = 0; n <= numDisparity; n++)
		{
			float eval = dsv[n].at<float>(i);
			if (eval < mine)
			{
				mine = eval;
				minv = n;
			}
		}
		dst[i] = minv + minDisparity;
	}

	/*namedWindow("mincost");
	static int amp = 2;
	createTrackbar("a","mincost",&amp,50);

	cost*=amp;

	imshow("mincost",cost);*/
}
void CostVolumeRefinement::subpixelInterpolation(Mat& dest, int method)
{
	if (method == SUBPIXEL_NONE)
	{
		dest *= 16;
		return;
	}
	short* disp = dest.ptr<short>(0);
	const int imsize = dest.size().area();
	if (method == SUBPIXEL_QUAD)
	{
		for (int j = 0; j < imsize; j++)
		{
			short d = disp[j];
			int l = d - minDisparity;
			if (l<1 || l>numDisparity - 2)
			{
				disp[j] = 16 * d;
			}
			else
			{
				float f = dsv[l].at<float>(j);
				float p = dsv[l + 1].at<float>(j);
				float m = dsv[l - 1].at<float>(j);

				float md = ((p + m - (f*2.f))*2.f);
				if (md != 0)
				{
					float dd = (float)d - (float)(p - m) / (float)md;
					disp[j] = (short)(16.f*dd + 0.5f);
				}
			}
		}
	}
	else if (method == SUBPIXEL_LINEAR)
	{
		for (int j = 0; j < imsize; j++)
		{
			short d = disp[j];
			int l = d - minDisparity;
			if (l<1 || l>numDisparity - 2)
			{
				disp[j] = 16 * d;
			}
			else
			{
				const double m1 = (double)dsv[l].at<float>(j);
				const double m3 = (double)dsv[l + 1].at<float>(j);
				const double m2 = (double)dsv[l - 1].at<float>(j);
				const double m31 = m3 - m1;
				const double m21 = m2 - m1;
				double md;

				if (m2 > m3)
				{
					md = 0.5 - 0.25*((m31*m31) / (m21*m21) + m31 / m21);
				}
				else
				{
					md = -(0.5 - 0.25*((m21*m21) / (m31*m31) + m21 / m31));

				}

				disp[j] = (short)(16.0*((double)d + md) + 0.5);
			}
		}
	}
}



CostVolumeRefinement::CostVolumeRefinement(int disparitymin, int disparity_range)
{
	//sub_method = SUBPIXEL_NONE;
	sub_method = SUBPIXEL_QUAD;
	minDisparity = disparitymin;
	numDisparity = disparity_range;
	dsv.resize(disparity_range + 1);
	dsv2.resize(disparity_range + 1);
}
/*
void CostVolumeRefinement::crossBasedAdaptiveboxRefinement(Mat& disp, Mat& guide,Mat& dest, int data_trunc, int metric, int r, int thresh,int iter)
{
if(iter==0)disp.convertTo(dest,CV_16S,16);
if(dest.empty())dest.create(disp.size(),CV_16S);

CrossBasedLocalFilter cbabf(guide,r,thresh);
Mat in = disp.clone();
for(int i=0;i<iter;i++)
{
{
CalcTime t("build");
buildCostVolume(in,data_trunc,metric);
}
{
CalcTime t("filter");
for(int n=0;n<numDisparity;n++)
{
cbabf(dsv[n],dsv[n]);
}
}
{
CalcTime t("wta");
wta(dest);
}
//dest.copyTo(in);

{
CalcTime t("dubpix");
subpixelInterpolation(dest,sub_method);
}
medianBlur(dest,dest,5);
blurRemoveMinMax(dest,dest,1,1);
dest.convertTo(in,CV_8U,1.0/16);

//

}
}
*/
void CostVolumeRefinement::medianRefinement(Mat& disp, Mat& dest, int data_trunc, int metric, int r, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);
	Mat in = disp.clone();
	for (int i = 0; i < iter; i++)
	{
		{
			//			CalcTime t("build");
			buildCostVolume(in, data_trunc, metric);
		}
		{
			//	CalcTime t("filter");
			for (int n = 0; n <= numDisparity; n++)
			{
				medianBlur(dsv[n], dsv[n], 2 * r + 1);
			}
		}
		{
			//	CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);


		{
			//		CalcTime t("dubpix");
			subpixelInterpolation(dest, sub_method);
		}

	}
}
void CostVolumeRefinement::gaussianRefinement(Mat& disp, Mat& dest, int data_trunc, int metric, int r, double sigma, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//	CalcTime t("build");
			buildCostVolume(in, data_trunc, metric);
		}
		{
			//			CalcTime t("filter");
#pragma omp parallel for
			for (int n = 0; n <= numDisparity; n++)
			{
				GaussianBlur(dsv[n], dsv[n], Size(2 * r + 1, 2 * r + 1), sigma);
			}
		}
		{
			//		CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("dubpix");
			subpixelInterpolation(dest, sub_method);
		}
		//dest.convertTo(in,CV_8U,1.0/16);

	}
}
void CostVolumeRefinement::weightedGaussianRefinement(Mat& disp, Mat& weight, Mat& dest, int data_trunc, int metric, int r, double sigma, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//	CalcTime t("build");
			buildWeightedCostVolume(in, weight, data_trunc, metric);
		}
		{
			//			CalcTime t("filter");
#pragma omp parallel for
			for (int n = 0; n <= numDisparity; n++)
			{
				GaussianBlur(dsv[n], dsv[n], Size(2 * r + 1, 2 * r + 1), sigma); \
			}
		}
		{
			//		CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("dubpix");
			subpixelInterpolation(dest, sub_method);
		}
		//dest.convertTo(in,CV_8U,1.0/16);

	}
}
void CostVolumeRefinement::boxRefinement(Mat& disp, Mat& dest, int data_trunc, int metric, int r, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);
	Mat in = disp.clone();
	for (int i = 0; i < iter; i++)
	{
		{
			//cout<<"build\n";
			//	CalcTime t("build");
			buildCostVolume(in, data_trunc, metric);
		}
		{
			//cout<<"filter\n";
			//			CalcTime t("filter");
#pragma omp parallel for
			for (int n = 0; n <= numDisparity; n++)
			{
				//Mat temp;
				boxFilter(dsv[n], dsv[n], CV_32F, Size(2 * r + 1, 2 * r + 1), Point(-1, -1), false);
			}
		}
		{
			//cout<<"wta\n";
			//		CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//cout<<"subpix\n";
			//	CalcTime t("dubpix");
			//subpixelInterpolation(dest,sub_method);
			subpixelInterpolation(dest, SUBPIXEL_NONE);
		}
		//dest.convertTo(in,CV_8U,1.0/16);

	}
}

void CostVolumeRefinement::weightedBoxRefinement(Mat& disp, Mat& weight, Mat& dest, int data_trunc, int metric, int r, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);
	Mat in = disp.clone();
	for (int i = 0; i < iter; i++)
	{
		{
			buildWeightedCostVolume(in, weight, data_trunc, metric);
		}
		{
#pragma omp parallel for
			for (int n = 0; n <= numDisparity; n++)
			{
				//Mat temp;
				boxFilter(dsv[n], dsv[n], CV_32F, Size(2 * r + 1, 2 * r + 1), Point(-1, -1), false);
			}
		}
		{
			wta(dest);
		}
		dest.copyTo(in);
		{
			//subpixelInterpolation(dest,sub_method);
			subpixelInterpolation(dest, SUBPIXEL_NONE);
		}
		//dest.convertTo(in,CV_8U,1.0/16);
	}
}


/*
class box_Invoker
{
public:
box_Invoker(vector<Mat>& _src, vector<Mat>& _dest, const int _radius) :
src(_src), dest(_dest), radius(_radius)
{
}

virtual void operator() (const BlockedRange& range) const
{

for(int i = range.start; i != range.end(); i++)
{
boxFilter(src[i],dest[i],CV_32F,Size(2*radius+1,2*radius+1),Point(-1,-1),true);
}
}
private:
vector<Mat> src;
vector<Mat> dest;

const int radius;
};*/

void CostVolumeRefinement::jointBilateralRefinement(Mat& disp, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	Mat guidef; guide.convertTo(guidef, VOLUME_TYPE);
	for (int i = 0; i < iter; i++)
	{
		{
			buildCostVolume(in, data_trunc, metric);
		}
		{
			//CalcTime t("filter");	
			{
				//CalcTime t("filter box");
				for (int n = 0; n <= numDisparity; n++)
				{
					//if(dsv2[i].empty())dsv2[i].create(dsv[i].size(),dsv[i].type());
					//boxFilter(dsv[n],dsv2[n],CV_32F,Size(2*r+1,2*r+1),Point(-1,-1),true);
					jointBilateralFilter(dsv[n], guidef, dsv2[n], Size(2 * r + 1, 2 * r + 1), sigma_c, sigma_s);
				}

				//box_Invoker body(dsv,dsv2,r);
				//parallel_for(BlockedRange(0, numDisparity+1,1), body);
			}
			{
				//CalcTime t("filter joint");
				//jointBilateralRefinement_8u(dsv,guide,dsv2,Size(2*r+1,2*r+1),sigma_c,sigma_s,cv::BORDER_REPLICATE);
			}
			{
				//CalcTime t("copy");
				for (int n = 0; n <= numDisparity; n++)
				{
					dsv2[n].copyTo(dsv[n]);
				}
			}
		}
		{
			//CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//CalcTime t("sub pix");
			//subpixelInterpolation(dest,SUBPIXEL_NONE);
			subpixelInterpolation(dest, sub_method);
		}
	}
}

#ifdef JBF2
void CostVolumeRefinement::jointBilateralRefinement2(Mat& disp, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//	CalcTime t("build");
			buildCostVolume(in, data_trunc, metric);
		}

		{
			//	CalcTime t("filter");	
			jointBilateralRefinement2_8u(dsv, guide, dsv, Size(2 * r + 1, 2 * r + 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
		}
		{
			//	CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("sub pix");
			//subpixelInterpolation(dest,SUBPIXEL_NONE);
			subpixelInterpolation(dest, sub_method);
		}
	}
}


void CostVolumeRefinement::jointBilateralRefinementSP2(Mat& disp, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//			CalcTime t("build");
			buildCostVolume(in, data_trunc, metric);
		}


		{
			//	CalcTime t("filter");	
			jointBilateralRefinement2_8u(dsv, guide, dsv, Size(2 * r + 1, 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
			jointBilateralRefinement2_8u(dsv, guide, dsv, Size(1, 2 * r + 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
		}
		{
			//	CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("sub pix");
			//subpixelInterpolation(dest,SUBPIXEL_NONE);
			subpixelInterpolation(dest, sub_method);
		}
	}
}


void CostVolumeRefinement::jointBilateralRefinementSP(Mat& disp, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//			CalcTime t("build");
			buildCostVolume(in, data_trunc, metric);
		}


		/*
		jointBilateralRefinement_8u(dsv,guide,dsv2,Size(2*r+1,2*r+1),sigma_c,sigma_s,cv::BORDER_REPLICATE);
		for(int n=0;n<=numDisparity;n++)
		{
		dsv2[n].copyTo(dsv[n]);
		}
		*/
		for (int n = 0; n <= numDisparity; n++)
		{
			boxFilter(dsv[n], dsv2[n], CV_32F, Size(2 * r + 1, 1), Point(-1, -1), true);
		}
		{
			//	CalcTime t("filter joint");	
			jointBilateralRefinement_8u(dsv, guide, dsv2, Size(2 * r + 1, 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
		}
		for (int n = 0; n <= numDisparity; n++)
		{
			boxFilter(dsv2[n], dsv[n], CV_32F, Size(1, 2 * r + 1), Point(-1, -1), true);
		}
		{
			//	CalcTime t("filter");	
			jointBilateralRefinement_8u(dsv2, guide, dsv, Size(1, 2 * r + 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
		}
		{
			//	CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("sub pix");
			//subpixelInterpolation(dest,SUBPIXEL_NONE);
			subpixelInterpolation(dest, sub_method);
		}
	}
}
void CostVolumeRefinement::weightedJointBilateralRefinement(Mat& disp, Mat& weight, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//			CalcTime t("build");
			buildWeightedCostVolume(in, weight, data_trunc, metric);
		}

		{
			//	CalcTime t("filter");	
			for (int n = 0; n <= numDisparity; n++)
			{
				boxFilter(dsv[n], dsv2[n], CV_32F, Size(2 * r + 1, 2 * r + 1), Point(-1, -1), true);
			}
			jointBilateralRefinement_8u(dsv, guide, dsv2, Size(2 * r + 1, 2 * r + 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
			for (int n = 0; n <= numDisparity; n++)
			{
				dsv2[n].copyTo(dsv[n]);
			}
		}
		{
			//	CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("sub pix");
			subpixelInterpolation(dest, sub_method);
		}
	}
}

void CostVolumeRefinement::weightedJointBilateralRefinementSP(Mat& disp, Mat& weight, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double sigma_c, double sigma_s, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//	CalcTime t("build");
			buildWeightedCostVolume(in, weight, data_trunc, metric);
		}

		{
			//	CalcTime t("filter");	
			jointBilateralRefinement2_8u(dsv, guide, dsv, Size(2 * r + 1, 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
		}

{
	//	CalcTime t("filter");	
	jointBilateralRefinement2_8u(dsv, guide, dsv, Size(1, 2 * r + 1), sigma_c, sigma_s, cv::BORDER_REPLICATE);
}
		{
			//	CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("sub pix");
			subpixelInterpolation(dest, sub_method);
		}
	}
}
#endif
void CostVolumeRefinement::weightedGuidedRefinement(Mat& disp, Mat& weight, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double eps, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);
	for (int i = 0; i < iter; i++)
	{
		{
			//			CalcTime t("build");
			buildWeightedCostVolume(in, weight, data_trunc, metric);
		}
		{
			//	CalcTime t("filter");
#pragma omp parallel for
			for (int n = 0; n <= numDisparity; n++)
			{
				guidedImageFilter(dsv[n], guide, dsv[n], r, (float)eps);
			}
		}
		{
			//	CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//	CalcTime t("dubpix");
			subpixelInterpolation(dest, sub_method);
		}
	}
}


class guided_Invoker : public cv::ParallelLoopBody
{
public:
	guided_Invoker(vector<Mat> _dest, vector<Mat> _src, Mat& _guide, const int _radius, const float _eps) :
		src(_src), dest(_dest), guide(&_guide), radius(_radius), eps(_eps)
	{
	}

	virtual void operator() (const Range& range) const
	{
		//imshow("a",*guide);waitKey();
		for (int i = range.start; i != range.end; i++)
		{
			//Mat g;
			//cp::guidedFilter((const Mat)src[i], (const Mat)(*guide), (Mat)dest[i], radius, eps);
			Mat dst;
			cp::guidedImageFilter(src[i], *guide, dst, radius, eps);
			dst.copyTo(dest[i]);			
		}
	}
private:
	vector<Mat> src;
	vector<Mat> dest;
	Mat *guide;

	const int radius;
	const float eps;
};

void CostVolumeRefinement::guidedRefinement(Mat& disp, Mat& guide, Mat& dest, int data_trunc, int metric, int r, double eps, int iter)
{
	if (iter == 0)disp.convertTo(dest, CV_16S, 16);
	if (dest.empty())dest.create(disp.size(), CV_16S);

	Mat in; disp.convertTo(in, CV_16S);


	for (int i = 0; i < iter; i++)
	{
		{
			//CalcTime t("build");
			buildCostVolume(in, data_trunc, metric);
		}
		{
			//CalcTime t("filter");

			guided_Invoker body(dsv, dsv, guide, r, (float)eps);
			parallel_for_(Range(0, numDisparity + 1), body);

			//#pragma omp parallel for
			/*	for(int n=0;n<=numDisparity;n++)
			{
			guidedFilter(dsv[n],guide,dsv[n],r,(float)eps);
			}*/

		}
		{
			//CalcTime t("wta");
			wta(dest);
		}
		dest.copyTo(in);
		{
			//CalcTime t("dubpix");
			subpixelInterpolation(dest, sub_method);
		}
	}
}
}