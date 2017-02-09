#include "binalyWeightedRangeFilter.hpp"
#include "color.hpp"
using namespace std;
using namespace cv;
namespace cp
{
	//for test, non SSE code for tbb
	class BinalyWeightedRangeFilter_8u_Invoker : public cv::ParallelLoopBody
	{
	public:
		BinalyWeightedRangeFilter_8u_Invoker(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, uchar _threshold) :
			temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			Size size = dest->size();

			if (cn == 1)
			{
				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;
				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
				{
					j = 0;
					for (; j < size.width; j++)
					{
						const uchar val0 = sptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							uchar val = sptr[j + space_ofs[k]];
							float w = (abs(val - val0) <= threshold) ? 1.f : 0.f;
							sum += val*w;
							wsum += w;
						}
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs;
		uchar threshold;
	};

	void binalyWeightedRangeFilterNONSSE_8u(const Mat& src, Mat& dst, Size kernelSize, uchar threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		if (dst.empty())dst = Mat::zeros(src.size(), src.type());
		int cn = src.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];

		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		BinalyWeightedRangeFilter_8u_Invoker body(dest, temp, radiusH, radiusV, maxk, space_ofs, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	//for test, non SSE code for tbb
	class BinalyWeightedRangeFilter_32f_Invoker : public cv::ParallelLoopBody
	{
	public:
		BinalyWeightedRangeFilter_32f_Invoker(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, float _threshold) :
			temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			Size size = dest->size();

			if (cn == 1)
			{
				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;
				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
				{
					j = 0;
					for (; j < size.width; j++)
					{
						const float val0 = sptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float val = sptr[j + space_ofs[k]];
							float w = (abs(val - val0) <= threshold) ? 1.f : 0.f;
							sum += val*w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs;
		float threshold;
	};

	void binalyWeightedRangeFilterNONSSE_32f(const Mat& src, Mat& dst, Size kernelSize, float threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		if (dst.empty())dst = Mat::zeros(src.size(), src.type());
		int cn = src.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];

		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		BinalyWeightedRangeFilter_32f_Invoker body(dest, temp, radiusH, radiusV, maxk, space_ofs, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	class BinalyWeightedRangeFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		BinalyWeightedRangeFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, uchar _threshold) :
			temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			Size size = dest->size();

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1)
			{
				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;
				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];

							const uchar* sptrj = sptr + j;
							const __m128i sval0 = _mm_load_si128((__m128i*)(sptrj));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 tval1 = _mm_set1_ps(0.0f);
							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 tval2 = _mm_set1_ps(0.0f);
							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 tval3 = _mm_set1_ps(0.0f);
							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 tval4 = _mm_set1_ps(0.0f);

							//const __m128i _x80 = _mm_set1_epi8('\x80');
							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8((char)1);
							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));

								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_add_epi8(_mm_subs_epu8(sval0, sref), _mm_subs_epu8(sref, sval0)), mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i m1 = _mm_unpacklo_epi8(sref, zero);
								__m128i m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128 _valF = _mm_cvtepi32_ps(m1);
								__m128 _w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);

								m1 = _mm_unpackhi_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								_valF = _mm_cvtepi32_ps(m1);
								_w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_valF = _mm_cvtepi32_ps(m2);
								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_mul_ps(_w, _valF);
								wval4 = _mm_add_ps(wval4, _w);
								tval4 = _mm_add_ps(tval4, _valF);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							tval2 = _mm_div_ps(tval2, wval2);
							tval3 = _mm_div_ps(tval3, wval3);
							tval4 = _mm_div_ps(tval4, wval4);
							_mm_stream_si128((__m128i*)(dptr + j), _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)), _mm_packs_epi32(_mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar val0 = sptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							uchar val = sptr[j + space_ofs[k]];
							float w = (abs(val - val0) <= threshold) ? 1.f : 0.f;
							sum += val*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else
			{
				// under construction
				const int sstep = 3 * temp->cols;
				const int dstep = dest->cols * 3;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;

							const __m128i bval0 = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i gval0 = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i rval0 = _mm_load_si128((__m128i*)(sptrrj));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 rval2 = _mm_set1_ps(0.0f);
							__m128 gval2 = _mm_set1_ps(0.0f);
							__m128 bval2 = _mm_set1_ps(0.0f);

							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 rval3 = _mm_set1_ps(0.0f);
							__m128 gval3 = _mm_set1_ps(0.0f);
							__m128 bval3 = _mm_set1_ps(0.0f);

							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 rval4 = _mm_set1_ps(0.0f);
							__m128 gval4 = _mm_set1_ps(0.0f);
							__m128 bval4 = _mm_set1_ps(0.0f);

							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8('\x01');
							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								__m128i v = _mm_add_epi8(_mm_subs_epu8(bval0, bref), _mm_subs_epu8(bref, bval0));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(gval0, gref), _mm_subs_epu8(gref, gval0)));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(rval0, rref), _mm_subs_epu8(rref, rval0)));

								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(v, mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128i r1 = _mm_unpacklo_epi8(rref, zero);
								__m128i r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								__m128i g1 = _mm_unpacklo_epi8(gref, zero);
								__m128i g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								__m128i b1 = _mm_unpacklo_epi8(bref, zero);
								__m128i b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);


								__m128 _w = _mm_cvtepi32_ps(w1);
								__m128 _valr = _mm_cvtepi32_ps(r1);
								__m128 _valg = _mm_cvtepi32_ps(g1);
								__m128 _valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval1 = _mm_add_ps(rval1, _valr);
								gval1 = _mm_add_ps(gval1, _valg);
								bval1 = _mm_add_ps(bval1, _valb);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval2 = _mm_add_ps(rval2, _valr);
								gval2 = _mm_add_ps(gval2, _valg);
								bval2 = _mm_add_ps(bval2, _valb);
								wval2 = _mm_add_ps(wval2, _w);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);


								_w = _mm_cvtepi32_ps(w1);
								_valr = _mm_cvtepi32_ps(r1);
								_valg = _mm_cvtepi32_ps(g1);
								_valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval3 = _mm_add_ps(wval3, _w);
								rval3 = _mm_add_ps(rval3, _valr);
								gval3 = _mm_add_ps(gval3, _valg);
								bval3 = _mm_add_ps(bval3, _valb);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval4 = _mm_add_ps(wval4, _w);
								rval4 = _mm_add_ps(rval4, _valr);
								gval4 = _mm_add_ps(gval4, _valg);
								bval4 = _mm_add_ps(bval4, _valb);
							}

							rval1 = _mm_div_ps(rval1, wval1);
							rval2 = _mm_div_ps(rval2, wval2);
							rval3 = _mm_div_ps(rval3, wval3);
							rval4 = _mm_div_ps(rval4, wval4);
							__m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)), _mm_packs_epi32(_mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
							gval1 = _mm_div_ps(gval1, wval1);
							gval2 = _mm_div_ps(gval2, wval2);
							gval3 = _mm_div_ps(gval3, wval3);
							gval4 = _mm_div_ps(gval4, wval4);
							__m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)), _mm_packs_epi32(_mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
							bval1 = _mm_div_ps(bval1, wval1);
							bval2 = _mm_div_ps(bval2, wval2);
							bval3 = _mm_div_ps(bval3, wval3);
							bval4 = _mm_div_ps(bval4, wval4);
							__m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)), _mm_packs_epi32(_mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

							//sse4///


							const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
							const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
							const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

							const __m128i bmask1 = _mm_setr_epi8
								(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);

							const __m128i bmask2 = _mm_setr_epi8
								(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

							a = _mm_shuffle_epi8(a, mask1);
							b = _mm_shuffle_epi8(b, mask2);
							c = _mm_shuffle_epi8(c, mask3);
							uchar* dptrc = dptr + 3 * j;
							_mm_stream_si128((__m128i*)(dptrc), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
							_mm_stream_si128((__m128i*)(dptrc + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
							_mm_stream_si128((__m128i*)(dptrc + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;

						int r0 = sptrrj[0];
						int g0 = sptrgj[0];
						int b0 = sptrbj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = (std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0) < threshold) ? 1.f : 0.f;
							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U

						wsum = 1.f / wsum;
						b0 = cvRound(sum_b*wsum);
						g0 = cvRound(sum_g*wsum);
						r0 = cvRound(sum_r*wsum);
						dptr[3 * j] = (uchar)r0; dptr[3 * j + 1] = (uchar)g0; dptr[3 * j + 2] = (uchar)b0;
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs;
		uchar threshold;//*space_weight,
	};


	class CenterReplacedBinalyWeightedRangeFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		CenterReplacedBinalyWeightedRangeFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _center, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, uchar _threshold) :
			temp(&_temp), center(&_center), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			Size size = dest->size();

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1)
			{
				uchar* cptr = (uchar*)center->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;
				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, cptr += sstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];

							//float* spw = space_weight;
							const uchar* sptrj = sptr + j;
							const __m128i sval0 = _mm_load_si128((__m128i*)(cptr + j));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 tval1 = _mm_set1_ps(0.0f);
							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 tval2 = _mm_set1_ps(0.0f);
							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 tval3 = _mm_set1_ps(0.0f);
							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 tval4 = _mm_set1_ps(0.0f);

							//const __m128i _x80 = _mm_set1_epi8('\x80');
							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8('\x01');
							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));

								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_add_epi8(_mm_subs_epu8(sval0, sref), _mm_subs_epu8(sref, sval0)), mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i m1 = _mm_unpacklo_epi8(sref, zero);
								__m128i m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128 _valF = _mm_cvtepi32_ps(m1);
								__m128 _w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);

								m1 = _mm_unpackhi_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								_valF = _mm_cvtepi32_ps(m1);
								_w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_valF = _mm_cvtepi32_ps(m2);
								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_mul_ps(_w, _valF);
								wval4 = _mm_add_ps(wval4, _w);
								tval4 = _mm_add_ps(tval4, _valF);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							tval2 = _mm_div_ps(tval2, wval2);
							tval3 = _mm_div_ps(tval3, wval3);
							tval4 = _mm_div_ps(tval4, wval4);
							_mm_stream_si128((__m128i*)(dptr + j), _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)), _mm_packs_epi32(_mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar val0 = cptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							uchar val = sptr[j + space_ofs[k]];
							float w = (abs(val - val0) < threshold) ? 1.f : 0.f;
							sum += val*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else
			{
				// under construction
				const int sstep = 3 * temp->cols;
				const int dstep = dest->cols * 3;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* cptrr = (uchar*)center->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* cptrg = (uchar*)center->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* cptrb = (uchar*)center->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, cptrr += sstep, cptrg += sstep, cptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;

							const __m128i bval0 = _mm_load_si128((__m128i*)(cptrb + j));
							const __m128i gval0 = _mm_load_si128((__m128i*)(cptrg + j));
							const __m128i rval0 = _mm_load_si128((__m128i*)(cptrr + j));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 rval2 = _mm_set1_ps(0.0f);
							__m128 gval2 = _mm_set1_ps(0.0f);
							__m128 bval2 = _mm_set1_ps(0.0f);

							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 rval3 = _mm_set1_ps(0.0f);
							__m128 gval3 = _mm_set1_ps(0.0f);
							__m128 bval3 = _mm_set1_ps(0.0f);

							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 rval4 = _mm_set1_ps(0.0f);
							__m128 gval4 = _mm_set1_ps(0.0f);
							__m128 bval4 = _mm_set1_ps(0.0f);

							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8('\x01');
							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								__m128i v = _mm_add_epi8(_mm_subs_epu8(bval0, bref), _mm_subs_epu8(bref, bval0));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(gval0, gref), _mm_subs_epu8(gref, gval0)));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(rval0, rref), _mm_subs_epu8(rref, rval0)));

								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(v, mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128i r1 = _mm_unpacklo_epi8(rref, zero);
								__m128i r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								__m128i g1 = _mm_unpacklo_epi8(gref, zero);
								__m128i g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								__m128i b1 = _mm_unpacklo_epi8(bref, zero);
								__m128i b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);


								__m128 _w = _mm_cvtepi32_ps(w1);
								__m128 _valr = _mm_cvtepi32_ps(r1);
								__m128 _valg = _mm_cvtepi32_ps(g1);
								__m128 _valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval1 = _mm_add_ps(rval1, _valr);
								gval1 = _mm_add_ps(gval1, _valg);
								bval1 = _mm_add_ps(bval1, _valb);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval2 = _mm_add_ps(rval2, _valr);
								gval2 = _mm_add_ps(gval2, _valg);
								bval2 = _mm_add_ps(bval2, _valb);
								wval2 = _mm_add_ps(wval2, _w);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);


								_w = _mm_cvtepi32_ps(w1);
								_valr = _mm_cvtepi32_ps(r1);
								_valg = _mm_cvtepi32_ps(g1);
								_valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval3 = _mm_add_ps(wval3, _w);
								rval3 = _mm_add_ps(rval3, _valr);
								gval3 = _mm_add_ps(gval3, _valg);
								bval3 = _mm_add_ps(bval3, _valb);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval4 = _mm_add_ps(wval4, _w);
								rval4 = _mm_add_ps(rval4, _valr);
								gval4 = _mm_add_ps(gval4, _valg);
								bval4 = _mm_add_ps(bval4, _valb);
							}

							rval1 = _mm_div_ps(rval1, wval1);
							rval2 = _mm_div_ps(rval2, wval2);
							rval3 = _mm_div_ps(rval3, wval3);
							rval4 = _mm_div_ps(rval4, wval4);
							__m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)), _mm_packs_epi32(_mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
							gval1 = _mm_div_ps(gval1, wval1);
							gval2 = _mm_div_ps(gval2, wval2);
							gval3 = _mm_div_ps(gval3, wval3);
							gval4 = _mm_div_ps(gval4, wval4);
							__m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)), _mm_packs_epi32(_mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
							bval1 = _mm_div_ps(bval1, wval1);
							bval2 = _mm_div_ps(bval2, wval2);
							bval3 = _mm_div_ps(bval3, wval3);
							bval4 = _mm_div_ps(bval4, wval4);
							__m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)), _mm_packs_epi32(_mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

							//sse4///


							const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
							const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
							const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

							const __m128i bmask1 = _mm_setr_epi8
								(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);

							const __m128i bmask2 = _mm_setr_epi8
								(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

							a = _mm_shuffle_epi8(a, mask1);
							b = _mm_shuffle_epi8(b, mask2);
							c = _mm_shuffle_epi8(c, mask3);
							uchar* dptrc = dptr + 3 * j;
							_mm_stream_si128((__m128i*)(dptrc), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
							_mm_stream_si128((__m128i*)(dptrc + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
							_mm_stream_si128((__m128i*)(dptrc + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;

						int r0 = cptrr[j];
						int g0 = cptrg[j];
						int b0 = cptrb[j];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = (std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0) < threshold) ? 1.f : 0.f;
							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U

						wsum = 1.f / wsum;
						b0 = cvRound(sum_b*wsum);
						g0 = cvRound(sum_g*wsum);
						r0 = cvRound(sum_r*wsum);
						dptr[3 * j] = (uchar)r0; dptr[3 * j + 1] = (uchar)g0; dptr[3 * j + 2] = (uchar)b0;
					}
				}
			}
		}
	private:
		const Mat *temp;
		const Mat *center;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs;
		uchar threshold;//*space_weight,
	};

	class JointBinalyWeightedRangeFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		JointBinalyWeightedRangeFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_ofsj, uchar _threshold) :
			temp(&_temp), joint(&guide), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_ofsj(_space_ofsj), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			int cng = (joint->rows - 2 * radiusV) / dest->rows;
			Size size = dest->size();

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				uchar* jptr = (uchar*)joint->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;
				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, jptr += sstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];

							//float* spw = space_weight;
							const uchar* sptrj = sptr + j;
							const uchar* jptrj = jptr + j;
							const __m128i vval0 = _mm_load_si128((__m128i*)(jptrj));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 tval1 = _mm_set1_ps(0.0f);
							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 tval2 = _mm_set1_ps(0.0f);
							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 tval3 = _mm_set1_ps(0.0f);
							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 tval4 = _mm_set1_ps(0.0f);

							//const __m128i _x80 = _mm_set1_epi8('\x80');
							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8('\x01');
							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								__m128i vref = _mm_loadu_si128((__m128i*)(jptrj + *ofs));

								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_add_epi8(_mm_subs_epu8(vval0, vref), _mm_subs_epu8(vref, vval0)), mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128i m1 = _mm_unpacklo_epi8(sref, zero);
								__m128i m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								__m128 _valF = _mm_cvtepi32_ps(m1);
								__m128 _w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);

								m1 = _mm_unpackhi_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								_valF = _mm_cvtepi32_ps(m1);
								_w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_valF = _mm_cvtepi32_ps(m2);
								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_mul_ps(_w, _valF);
								wval4 = _mm_add_ps(wval4, _w);
								tval4 = _mm_add_ps(tval4, _valF);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							tval2 = _mm_div_ps(tval2, wval2);
							tval3 = _mm_div_ps(tval3, wval3);
							tval4 = _mm_div_ps(tval4, wval4);
							_mm_stream_si128((__m128i*)(dptr + j), _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)), _mm_packs_epi32(_mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar val0 = sptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							uchar val = jptr[j + space_ofs[k]];
							float w = (abs(val - val0) < threshold) ? 1.f : 0.f;
							uchar v = sptr[j + space_ofs[k]];
							sum += v*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 1 && cng == 3)
			{
				const int sstep = temp->cols;
				const int sstep3 = joint->cols * 3;
				const int dstep = dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				uchar* jptrr = (uchar*)joint->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* jptrg = (uchar*)joint->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* jptrb = (uchar*)joint->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, sptr += sstep, jptrr += sstep3, jptrg += sstep3, jptrb += sstep3, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* ofsj = &space_ofsj[0];
							const uchar* jptrrj = jptrr + j;
							const uchar* jptrgj = jptrg + j;
							const uchar* jptrbj = jptrb + j;
							const uchar* sptrj = sptr + j;

							const __m128i bval0 = _mm_load_si128((__m128i*)(jptrbj));
							const __m128i gval0 = _mm_load_si128((__m128i*)(jptrgj));
							const __m128i rval0 = _mm_load_si128((__m128i*)(jptrrj));


							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 tval1 = _mm_set1_ps(0.0f);
							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 tval2 = _mm_set1_ps(0.0f);
							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 tval3 = _mm_set1_ps(0.0f);
							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 tval4 = _mm_set1_ps(0.0f);

							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8('\x01');
							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, ofsj++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));

								__m128i bval = _mm_loadu_si128((__m128i*)(jptrbj + *ofsj));
								__m128i gval = _mm_loadu_si128((__m128i*)(jptrgj + *ofsj));
								__m128i rval = _mm_loadu_si128((__m128i*)(jptrrj + *ofsj));

								__m128i v = _mm_add_epi8(_mm_subs_epu8(bval0, bval), _mm_subs_epu8(bval, bval0));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(gval0, gval), _mm_subs_epu8(gval, gval0)));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(rval0, rval), _mm_subs_epu8(rval, rval0)));

								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(v, mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128i m1 = _mm_unpacklo_epi8(sref, zero);
								__m128i m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								__m128 _valF = _mm_cvtepi32_ps(m1);
								__m128 _w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);

								m1 = _mm_unpackhi_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								_valF = _mm_cvtepi32_ps(m1);
								_w = _mm_cvtepi32_ps(w1);
								_valF = _mm_mul_ps(_w, _valF);
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_valF = _mm_cvtepi32_ps(m2);
								_w = _mm_cvtepi32_ps(w2);
								_valF = _mm_mul_ps(_w, _valF);
								wval4 = _mm_add_ps(wval4, _w);
								tval4 = _mm_add_ps(tval4, _valF);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							tval2 = _mm_div_ps(tval2, wval2);
							tval3 = _mm_div_ps(tval3, wval3);
							tval4 = _mm_div_ps(tval4, wval4);
							_mm_stream_si128((__m128i*)(dptr + j), _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)), _mm_packs_epi32(_mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* jptrrj = jptrr + j;
						const uchar* jptrgj = jptrg + j;
						const uchar* jptrbj = jptrb + j;

						const uchar* sptrj = sptr + j;

						int r0 = jptrrj[0];
						int g0 = jptrgj[0];
						int b0 = jptrbj[0];

						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = jptrrj[space_ofsj[k]], g = jptrgj[space_ofsj[k]], b = jptrbj[space_ofsj[k]];
							float w = (std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0) < threshold) ? 1.f : 0.f;

							sum += sptrj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[j] = saturate_cast<uchar>(sum / wsum);
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				const int sstep = 3 * temp->cols;
				const int dstep = dest->cols * 3;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* jptrr = (uchar*)joint->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* jptrg = (uchar*)joint->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* jptrb = (uchar*)joint->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, jptrr += sstep, jptrg += sstep, jptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];

							const uchar* jptrrj = jptrr + j;
							const uchar* jptrgj = jptrg + j;
							const uchar* jptrbj = jptrb + j;
							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;

							const __m128i bval0 = _mm_load_si128((__m128i*)(jptrbj));
							const __m128i gval0 = _mm_load_si128((__m128i*)(jptrgj));
							const __m128i rval0 = _mm_load_si128((__m128i*)(jptrrj));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 rval2 = _mm_set1_ps(0.0f);
							__m128 gval2 = _mm_set1_ps(0.0f);
							__m128 bval2 = _mm_set1_ps(0.0f);

							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 rval3 = _mm_set1_ps(0.0f);
							__m128 gval3 = _mm_set1_ps(0.0f);
							__m128 bval3 = _mm_set1_ps(0.0f);

							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 rval4 = _mm_set1_ps(0.0f);
							__m128 gval4 = _mm_set1_ps(0.0f);
							__m128 bval4 = _mm_set1_ps(0.0f);

							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8('\x01');
							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								__m128i bval = _mm_loadu_si128((__m128i*)(jptrbj + *ofs));
								__m128i gval = _mm_loadu_si128((__m128i*)(jptrgj + *ofs));
								__m128i rval = _mm_loadu_si128((__m128i*)(jptrrj + *ofs));

								__m128i v = _mm_add_epi8(_mm_subs_epu8(bval0, bval), _mm_subs_epu8(bval, bval0));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(gval0, gval), _mm_subs_epu8(gval, gval0)));
								v = _mm_adds_epu8(v, _mm_add_epi8(_mm_subs_epu8(rval0, rval), _mm_subs_epu8(rval, rval0)));

								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(v, mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128i r1 = _mm_unpacklo_epi8(rref, zero);
								__m128i r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								__m128i g1 = _mm_unpacklo_epi8(gref, zero);
								__m128i g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								__m128i b1 = _mm_unpacklo_epi8(bref, zero);
								__m128i b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);


								__m128 _w = _mm_cvtepi32_ps(w1);
								__m128 _valr = _mm_cvtepi32_ps(r1);
								__m128 _valg = _mm_cvtepi32_ps(g1);
								__m128 _valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval1 = _mm_add_ps(rval1, _valr);
								gval1 = _mm_add_ps(gval1, _valg);
								bval1 = _mm_add_ps(bval1, _valb);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval2 = _mm_add_ps(rval2, _valr);
								gval2 = _mm_add_ps(gval2, _valg);
								bval2 = _mm_add_ps(bval2, _valb);
								wval2 = _mm_add_ps(wval2, _w);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);


								_w = _mm_cvtepi32_ps(w1);
								_valr = _mm_cvtepi32_ps(r1);
								_valg = _mm_cvtepi32_ps(g1);
								_valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval3 = _mm_add_ps(wval3, _w);
								rval3 = _mm_add_ps(rval3, _valr);
								gval3 = _mm_add_ps(gval3, _valg);
								bval3 = _mm_add_ps(bval3, _valb);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval4 = _mm_add_ps(wval4, _w);
								rval4 = _mm_add_ps(rval4, _valr);
								gval4 = _mm_add_ps(gval4, _valg);
								bval4 = _mm_add_ps(bval4, _valb);
							}

							rval1 = _mm_div_ps(rval1, wval1);
							rval2 = _mm_div_ps(rval2, wval2);
							rval3 = _mm_div_ps(rval3, wval3);
							rval4 = _mm_div_ps(rval4, wval4);
							__m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)), _mm_packs_epi32(_mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
							gval1 = _mm_div_ps(gval1, wval1);
							gval2 = _mm_div_ps(gval2, wval2);
							gval3 = _mm_div_ps(gval3, wval3);
							gval4 = _mm_div_ps(gval4, wval4);
							__m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)), _mm_packs_epi32(_mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
							bval1 = _mm_div_ps(bval1, wval1);
							bval2 = _mm_div_ps(bval2, wval2);
							bval3 = _mm_div_ps(bval3, wval3);
							bval4 = _mm_div_ps(bval4, wval4);
							__m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)), _mm_packs_epi32(_mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

							//sse4///


							const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
							const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
							const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

							const __m128i bmask1 = _mm_setr_epi8
								(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);

							const __m128i bmask2 = _mm_setr_epi8
								(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

							a = _mm_shuffle_epi8(a, mask1);
							b = _mm_shuffle_epi8(b, mask2);
							c = _mm_shuffle_epi8(c, mask3);
							uchar* dptrc = dptr + 3 * j;
							_mm_stream_si128((__m128i*)(dptrc), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
							_mm_stream_si128((__m128i*)(dptrc + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
							_mm_stream_si128((__m128i*)(dptrc + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* jptrrj = jptrr + j;
						const uchar* jptrgj = jptrg + j;
						const uchar* jptrbj = jptrb + j;

						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;

						int r0 = jptrrj[0];
						int g0 = jptrgj[0];
						int b0 = jptrbj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = jptrrj[space_ofs[k]], g = jptrgj[space_ofs[k]], b = jptrbj[space_ofs[k]];
							float w = (std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0) < threshold) ? 1.f : 0.f;

							int r1 = sptrrj[space_ofs[k]], g1 = sptrgj[space_ofs[k]], b1 = sptrbj[space_ofs[k]];
							sum_b += b1*w;
							sum_g += g1*w;
							sum_r += r1*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U

						wsum = 1.f / wsum;
						b0 = cvRound(sum_b*wsum);
						g0 = cvRound(sum_g*wsum);
						r0 = cvRound(sum_r*wsum);
						dptr[3 * j] = (uchar)r0; dptr[3 * j + 1] = (uchar)g0; dptr[3 * j + 2] = (uchar)b0;
					}
				}
			}
			else if (cn == 3 && cng == 1)
			{
				const int sstep3 = 3 * temp->cols;
				const int sstep = joint->cols;
				const int dstep = dest->cols * 3;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* jptr = (uchar*)joint->ptr(radiusV + range.start) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep3, sptrg += sstep3, sptrb += sstep3, jptr += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* ofsj = &space_ofsj[0];

							const uchar* jptrj = jptr + j;
							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;

							const __m128i vval0 = _mm_load_si128((__m128i*)(jptrj));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							__m128 wval2 = _mm_set1_ps(0.0f);
							__m128 rval2 = _mm_set1_ps(0.0f);
							__m128 gval2 = _mm_set1_ps(0.0f);
							__m128 bval2 = _mm_set1_ps(0.0f);

							__m128 wval3 = _mm_set1_ps(0.0f);
							__m128 rval3 = _mm_set1_ps(0.0f);
							__m128 gval3 = _mm_set1_ps(0.0f);
							__m128 bval3 = _mm_set1_ps(0.0f);

							__m128 wval4 = _mm_set1_ps(0.0f);
							__m128 rval4 = _mm_set1_ps(0.0f);
							__m128 gval4 = _mm_set1_ps(0.0f);
							__m128 bval4 = _mm_set1_ps(0.0f);

							const __m128i mth = _mm_set1_epi8(threshold);
							const __m128i one = _mm_set1_epi8('\x01');
							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, ofsj++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								__m128i vref = _mm_loadu_si128((__m128i*)(jptrj + *ofsj));
								__m128i _wi = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_add_epi8(_mm_subs_epu8(vval0, vref), _mm_subs_epu8(vref, vval0)), mth), zero);
								_wi = _mm_blendv_epi8(zero, one, _wi);

								__m128i w1 = _mm_unpacklo_epi8(_wi, zero);
								__m128i w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								__m128i r1 = _mm_unpacklo_epi8(rref, zero);
								__m128i r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								__m128i g1 = _mm_unpacklo_epi8(gref, zero);
								__m128i g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								__m128i b1 = _mm_unpacklo_epi8(bref, zero);
								__m128i b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								__m128 _w = _mm_cvtepi32_ps(w1);
								__m128 _valr = _mm_cvtepi32_ps(r1);
								__m128 _valg = _mm_cvtepi32_ps(g1);
								__m128 _valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval1 = _mm_add_ps(rval1, _valr);
								gval1 = _mm_add_ps(gval1, _valg);
								bval1 = _mm_add_ps(bval1, _valb);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval2 = _mm_add_ps(rval2, _valr);
								gval2 = _mm_add_ps(gval2, _valg);
								bval2 = _mm_add_ps(bval2, _valb);
								wval2 = _mm_add_ps(wval2, _w);

								w1 = _mm_unpackhi_epi8(_wi, zero);
								w2 = _mm_unpackhi_epi16(w1, zero);
								w1 = _mm_unpacklo_epi16(w1, zero);

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);


								_w = _mm_cvtepi32_ps(w1);
								_valr = _mm_cvtepi32_ps(r1);
								_valg = _mm_cvtepi32_ps(g1);
								_valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval3 = _mm_add_ps(wval3, _w);
								rval3 = _mm_add_ps(rval3, _valr);
								gval3 = _mm_add_ps(gval3, _valg);
								bval3 = _mm_add_ps(bval3, _valb);

								_w = _mm_cvtepi32_ps(w2);
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								wval4 = _mm_add_ps(wval4, _w);
								rval4 = _mm_add_ps(rval4, _valr);
								gval4 = _mm_add_ps(gval4, _valg);
								bval4 = _mm_add_ps(bval4, _valb);
							}

							rval1 = _mm_div_ps(rval1, wval1);
							rval2 = _mm_div_ps(rval2, wval2);
							rval3 = _mm_div_ps(rval3, wval3);
							rval4 = _mm_div_ps(rval4, wval4);
							__m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)), _mm_packs_epi32(_mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
							gval1 = _mm_div_ps(gval1, wval1);
							gval2 = _mm_div_ps(gval2, wval2);
							gval3 = _mm_div_ps(gval3, wval3);
							gval4 = _mm_div_ps(gval4, wval4);
							__m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)), _mm_packs_epi32(_mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
							bval1 = _mm_div_ps(bval1, wval1);
							bval2 = _mm_div_ps(bval2, wval2);
							bval3 = _mm_div_ps(bval3, wval3);
							bval4 = _mm_div_ps(bval4, wval4);
							__m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)), _mm_packs_epi32(_mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

							//sse4///


							const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
							const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
							const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

							const __m128i bmask1 = _mm_setr_epi8
								(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);

							const __m128i bmask2 = _mm_setr_epi8
								(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

							a = _mm_shuffle_epi8(a, mask1);
							b = _mm_shuffle_epi8(b, mask2);
							c = _mm_shuffle_epi8(c, mask3);
							uchar* dptrc = dptr + 3 * j;
							_mm_stream_si128((__m128i*)(dptrc), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
							_mm_stream_si128((__m128i*)(dptrc + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
							_mm_stream_si128((__m128i*)(dptrc + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* jptrj = jptr + j;

						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;

						int v0 = jptrj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int v = jptrj[space_ofsj[k]];
							float w = (std::abs(v - v0) < threshold) ? 1.f : 0.f;

							int r1 = sptrrj[space_ofs[k]], g1 = sptrgj[space_ofs[k]], b1 = sptrbj[space_ofs[k]];
							sum_b += b1*w;
							sum_g += g1*w;
							sum_r += r1*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U

						wsum = 1.f / wsum;
						uchar b0 = cvRound(sum_b*wsum);
						uchar g0 = cvRound(sum_g*wsum);
						uchar r0 = cvRound(sum_r*wsum);
						dptr[3 * j] = (uchar)r0; dptr[3 * j + 1] = (uchar)g0; dptr[3 * j + 2] = (uchar)b0;
					}
				}
			}
		}
	private:
		const Mat *temp;
		const Mat *joint;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs, *space_ofsj;
		uchar threshold;//*space_weight,
	};

	class CenterReplacedBinalyWeightedRangeFilter_32f_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		CenterReplacedBinalyWeightedRangeFilter_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _center, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, float _threshold) :
			temp(&_temp), center(&_center), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			Size size = dest->size();

#if CV_SSE4_1
			const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1)
			{
				float* cptr = (float*)center->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, cptr += sstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);

						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];

							const float* sptrj = sptr + j;
							const __m128 sval0 = _mm_load_ps(cptr + j);

							__m128 tval = _mm_set1_ps(0.f);
							__m128 wval = _mm_set1_ps(0.f);
							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128 sref = _mm_loadu_ps((sptrj + *ofs));

								__m128 _w = _mm_cmple_ps(_mm_and_ps(_mm_sub_ps(sval0, sref), *(const __m128*)v32f_absmask), mth);

								_w = _mm_blendv_ps(zeros, ones, _w);

								sref = _mm_mul_ps(_w, sref);
								tval = _mm_add_ps(tval, sref);
								wval = _mm_add_ps(wval, _w);
							}
							tval = _mm_div_ps(tval, wval);
							_mm_stream_ps((dptr + j), tval);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float val0 = cptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float val = sptr[j + space_ofs[k]];
							float w = (abs(val - val0) <= threshold) ? 1.f : 0.f;

							sum += val*w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			else
			{
				//underconstruction
				const int sstep = 3 * temp->cols;
				const int dstep = dest->cols * 3;
				float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* cptrb = (float*)center->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* cptrg = (float*)center->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* cptrr = (float*)center->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, cptrr += sstep, cptrg += sstep, cptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);

						for (; j < size.width; j += 4)//16‰æ‘f‚Ã‚Âˆ—
						{
							int* ofs = &space_ofs[0];

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;

							const __m128 bval = _mm_load_ps((cptrb + j));
							const __m128 gval = _mm_load_ps((cptrg + j));
							const __m128 rval = _mm_load_ps((cptrr + j));

							//d‚Ý‚Æ•½ŠŠ‰»Œã‚Ì‰æ‘f‚SƒuƒƒbƒN‚Ã‚Â
							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								__m128 v = _mm_add_ps(_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval, bref), *(const __m128*)v32f_absmask));

								__m128 _w = _mm_cmple_ps(v, mth);
								_w = _mm_blendv_ps(zeros, ones, _w);

								rref = _mm_mul_ps(_w, rref);
								gref = _mm_mul_ps(_w, gref);
								bref = _mm_mul_ps(_w, bref);

								rval1 = _mm_add_ps(rval1, rref);
								gval1 = _mm_add_ps(gval1, gref);
								bval1 = _mm_add_ps(bval1, bref);
								wval1 = _mm_add_ps(wval1, _w);
							}

							rval1 = _mm_div_ps(rval1, wval1);
							gval1 = _mm_div_ps(gval1, wval1);
							bval1 = _mm_div_ps(bval1, wval1);

							float* dptrc = dptr + 3 * j;
							__m128 a = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));

							_mm_stream_ps((dptrc), _mm_blend_ps(_mm_blend_ps(b, a, 4), c, 2));
							_mm_stream_ps((dptrc + 4), _mm_blend_ps(_mm_blend_ps(c, b, 4), a, 2));
							_mm_stream_ps((dptrc + 8), _mm_blend_ps(_mm_blend_ps(a, c, 4), b, 2));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						float r0 = cptrr[j];
						float g0 = cptrg[j];
						float b0 = cptrb[j];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = ((std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)) <= threshold) ? 1.f : 0.f;

							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[3 * j] = sum_b*wsum;
						dptr[3 * j + 1] = sum_g*wsum;
						dptr[3 * j + 2] = sum_r*wsum;
					}
				}
			}
		}
	private:
		const Mat *temp;
		const Mat *center;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs;
		float threshold;
	};

	class BinalyWeightedRangeFilter_32f_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		BinalyWeightedRangeFilter_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, float _threshold) :
			temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			Size size = dest->size();

#if CV_SSE4_1
			const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1)
			{
				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							//float* spw = space_weight;

							const float* sptrj = sptr + j;
							const __m128 sval0 = _mm_load_ps(sptrj);

							__m128 tval = _mm_set1_ps(0.f);
							__m128 wval = _mm_set1_ps(0.f);
							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128 sref = _mm_loadu_ps((sptrj + *ofs));

								__m128 _w = _mm_cmple_ps(_mm_and_ps(_mm_sub_ps(sval0, sref), *(const __m128*)v32f_absmask), mth);

								_w = _mm_blendv_ps(zeros, ones, _w);

								sref = _mm_mul_ps(_w, sref);
								tval = _mm_add_ps(tval, sref);
								wval = _mm_add_ps(wval, _w);
							}
							tval = _mm_div_ps(tval, wval);
							_mm_stream_ps((dptr + j), tval);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float val0 = sptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float val = sptr[j + space_ofs[k]];
							float w = (abs(val - val0) <= threshold) ? 1.f : 0.f;

							sum += val*w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			else
			{
				//underconstruction
				const int sstep = 3 * temp->cols;
				const int dstep = dest->cols * 3;
				float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);

						for (; j < size.width; j += 4)//16‰æ‘f‚Ã‚Âˆ—
						{
							int* ofs = &space_ofs[0];

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;

							const __m128 bval = _mm_load_ps((sptrbj));
							const __m128 gval = _mm_load_ps((sptrgj));
							const __m128 rval = _mm_load_ps((sptrrj));

							//d‚Ý‚Æ•½ŠŠ‰»Œã‚Ì‰æ‘f‚SƒuƒƒbƒN‚Ã‚Â
							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								__m128 v = _mm_add_ps(_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval, bref), *(const __m128*)v32f_absmask));

								__m128 _w = _mm_cmple_ps(v, mth);
								_w = _mm_blendv_ps(zeros, ones, _w);

								rref = _mm_mul_ps(_w, rref);
								gref = _mm_mul_ps(_w, gref);
								bref = _mm_mul_ps(_w, bref);

								rval1 = _mm_add_ps(rval1, rref);
								gval1 = _mm_add_ps(gval1, gref);
								bval1 = _mm_add_ps(bval1, bref);
								wval1 = _mm_add_ps(wval1, _w);
							}

							rval1 = _mm_div_ps(rval1, wval1);
							gval1 = _mm_div_ps(gval1, wval1);
							bval1 = _mm_div_ps(bval1, wval1);

							float* dptrc = dptr + 3 * j;
							__m128 a = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));

							_mm_stream_ps((dptrc), _mm_blend_ps(_mm_blend_ps(b, a, 4), c, 2));
							_mm_stream_ps((dptrc + 4), _mm_blend_ps(_mm_blend_ps(c, b, 4), a, 2));
							_mm_stream_ps((dptrc + 8), _mm_blend_ps(_mm_blend_ps(a, c, 4), b, 2));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						float r0 = sptrrj[0];
						float g0 = sptrgj[0];
						float b0 = sptrbj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = ((std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)) <= threshold) ? 1.f : 0.f;

							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[3 * j] = sum_b*wsum;
						dptr[3 * j + 1] = sum_g*wsum;
						dptr[3 * j + 2] = sum_r*wsum;
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs;
		float threshold;
	};


	class JointBinalyWeightedRangeFilter_32f_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		JointBinalyWeightedRangeFilter_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_ofsj, float _threshold) :
			temp(&_temp), guide(&_guide), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_ofsj(_space_ofsj), threshold(_threshold)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = dest->channels();
			int cng = (guide->rows - 2 * radiusV) / dest->rows;
			Size size = dest->size();

#if CV_SSE4_1
			const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* jptr = (float*)guide->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, jptr += sstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];

							const float* sptrj = sptr + j;
							const float* jptrj = jptr + j;
							const __m128 jval0 = _mm_load_ps(jptrj);

							__m128 tval = _mm_set1_ps(0.f);
							__m128 wval = _mm_set1_ps(0.f);
							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128 jref = _mm_loadu_ps((jptrj + *ofs));
								__m128 _w = _mm_cmple_ps(_mm_and_ps(_mm_sub_ps(jval0, jref), *(const __m128*)v32f_absmask), mth);

								_w = _mm_blendv_ps(zeros, ones, _w);
								__m128 sref = _mm_loadu_ps((sptrj + *ofs));
								sref = _mm_mul_ps(_w, sref);
								tval = _mm_add_ps(tval, sref);
								wval = _mm_add_ps(wval, _w);
							}
							tval = _mm_div_ps(tval, wval);
							_mm_stream_ps((dptr + j), tval);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float val0 = jptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float val = jptr[j + space_ofs[k]];
							float w = (abs(val - val0) <= threshold) ? 1.f : 0.f;

							sum += sptr[j + space_ofs[k]] * w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			if (cn == 1 && cng == 3)
			{
				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* jptrb = (float*)guide->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* jptrg = (float*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* jptrr = (float*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int sstep3 = guide->cols * 3;
				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, jptrb += sstep3, jptrg += sstep3, jptrr += sstep3)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* ofsj = &space_ofsj[0];

							const float* sptrj = sptr + j;

							const float* jptrrj = jptrr + j;
							const float* jptrgj = jptrg + j;
							const float* jptrbj = jptrb + j;

							const __m128 bvalj = _mm_load_ps((jptrbj));
							const __m128 gvalj = _mm_load_ps((jptrgj));
							const __m128 rvalj = _mm_load_ps((jptrrj));

							__m128 tval = _mm_set1_ps(0.f);
							__m128 wval = _mm_set1_ps(0.f);
							for (k = 0; k < maxk; k++, ofs++, ofsj++)
							{
								__m128 brefj = _mm_loadu_ps((jptrbj + *ofsj));
								__m128 grefj = _mm_loadu_ps((jptrgj + *ofsj));
								__m128 rrefj = _mm_loadu_ps((jptrrj + *ofsj));

								__m128 v = _mm_add_ps(_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rvalj, rrefj), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gvalj, grefj), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bvalj, brefj), *(const __m128*)v32f_absmask));

								__m128 _w = _mm_cmple_ps(v, mth);

								_w = _mm_blendv_ps(zeros, ones, _w);

								__m128 sref = _mm_loadu_ps((sptrj + *ofs));
								sref = _mm_mul_ps(_w, sref);
								tval = _mm_add_ps(tval, sref);
								wval = _mm_add_ps(wval, _w);
							}
							tval = _mm_div_ps(tval, wval);
							_mm_stream_ps((dptr + j), tval);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* jptrrj = jptrr + j;
						const float* jptrgj = jptrg + j;
						const float* jptrbj = jptrb + j;

						float r0 = jptrrj[0];
						float g0 = jptrgj[0];
						float b0 = jptrbj[0];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float rj = jptrrj[space_ofsj[k]], gj = jptrgj[space_ofsj[k]], bj = jptrbj[space_ofsj[k]];
							float w = ((std::abs(bj - b0) + std::abs(gj - g0) + std::abs(rj - r0)) <= threshold) ? 1.f : 0.f;


							sum += sptr[j + space_ofs[k]] * w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			if (cn == 3 && cng == 1)
			{
				const int sstep3 = 3 * temp->cols;
				const int sstep = guide->cols;
				const int dstep = dest->cols * 3;
				float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* jptr = (float*)guide->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, dptr += dstep, sptrr += sstep3, sptrg += sstep3, sptrb += sstep3, jptr += sstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* ofsj = &space_ofsj[0];

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;

							const float* jptrj = jptr + j;
							const __m128 jval0 = _mm_load_ps(jptrj);

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);
							for (k = 0; k < maxk; k++, ofs++, ofsj++)
							{
								__m128 jref = _mm_loadu_ps((jptrj + *ofsj));
								__m128 _w = _mm_cmple_ps(_mm_and_ps(_mm_sub_ps(jval0, jref), *(const __m128*)v32f_absmask), mth);

								_w = _mm_blendv_ps(zeros, ones, _w);

								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								rref = _mm_mul_ps(_w, rref);
								gref = _mm_mul_ps(_w, gref);
								bref = _mm_mul_ps(_w, bref);

								rval1 = _mm_add_ps(rval1, rref);
								gval1 = _mm_add_ps(gval1, gref);
								bval1 = _mm_add_ps(bval1, bref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							rval1 = _mm_div_ps(rval1, wval1);
							gval1 = _mm_div_ps(gval1, wval1);
							bval1 = _mm_div_ps(bval1, wval1);

							float* dptrc = dptr + 3 * j;
							__m128 a = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));

							_mm_stream_ps((dptrc), _mm_blend_ps(_mm_blend_ps(b, a, 4), c, 2));
							_mm_stream_ps((dptrc + 4), _mm_blend_ps(_mm_blend_ps(c, b, 4), a, 2));
							_mm_stream_ps((dptrc + 8), _mm_blend_ps(_mm_blend_ps(a, c, 4), b, 2));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						const float* jptrj = jptr + j;
						float r0 = jptrj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float rj = jptrj[space_ofsj[k]];
							float w = ((std::abs(rj - r0)) <= threshold) ? 1.f : 0.f;

							float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[3 * j] = sum_b*wsum;
						dptr[3 * j + 1] = sum_g*wsum;
						dptr[3 * j + 2] = sum_r*wsum;
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				//underconstruction
				const int sstep = 3 * temp->cols;
				const int dstep = dest->cols * 3;
				float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* jptrb = (float*)guide->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* jptrg = (float*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* jptrr = (float*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, jptrr += sstep, jptrg += sstep, jptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						const __m128 mth = _mm_set1_ps(threshold);
						const __m128 ones = _mm_set1_ps(1.f);
						const __m128 zeros = _mm_set1_ps(0.f);

						for (; j < size.width; j += 4)//16‰æ‘f‚Ã‚Âˆ—
						{
							int* ofs = &space_ofs[0];

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;

							const float* jptrrj = jptrr + j;
							const float* jptrgj = jptrg + j;
							const float* jptrbj = jptrb + j;

							const __m128 bvalj = _mm_load_ps((jptrbj));
							const __m128 gvalj = _mm_load_ps((jptrgj));
							const __m128 rvalj = _mm_load_ps((jptrrj));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							for (k = 0; k < maxk; k++, ofs++)
							{
								__m128 brefj = _mm_loadu_ps((jptrbj + *ofs));
								__m128 grefj = _mm_loadu_ps((jptrgj + *ofs));
								__m128 rrefj = _mm_loadu_ps((jptrrj + *ofs));

								__m128 v = _mm_add_ps(_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rvalj, rrefj), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gvalj, grefj), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bvalj, brefj), *(const __m128*)v32f_absmask));

								__m128 _w = _mm_cmple_ps(v, mth);
								_w = _mm_blendv_ps(zeros, ones, _w);

								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								rref = _mm_mul_ps(_w, rref);
								gref = _mm_mul_ps(_w, gref);
								bref = _mm_mul_ps(_w, bref);

								rval1 = _mm_add_ps(rval1, rref);
								gval1 = _mm_add_ps(gval1, gref);
								bval1 = _mm_add_ps(bval1, bref);
								wval1 = _mm_add_ps(wval1, _w);
							}

							rval1 = _mm_div_ps(rval1, wval1);
							gval1 = _mm_div_ps(gval1, wval1);
							bval1 = _mm_div_ps(bval1, wval1);

							float* dptrc = dptr + 3 * j;
							__m128 a = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));

							_mm_stream_ps((dptrc), _mm_blend_ps(_mm_blend_ps(b, a, 4), c, 2));
							_mm_stream_ps((dptrc + 4), _mm_blend_ps(_mm_blend_ps(c, b, 4), a, 2));
							_mm_stream_ps((dptrc + 8), _mm_blend_ps(_mm_blend_ps(a, c, 4), b, 2));
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						const float* jptrrj = jptrr + j;
						const float* jptrgj = jptrg + j;
						const float* jptrbj = jptrb + j;

						float r0 = jptrrj[0];
						float g0 = jptrgj[0];
						float b0 = jptrbj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float rj = jptrrj[space_ofs[k]], gj = jptrgj[space_ofs[k]], bj = jptrbj[space_ofs[k]];
							float w = ((std::abs(bj - b0) + std::abs(gj - g0) + std::abs(rj - r0)) <= threshold) ? 1.f : 0.f;

							float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[3 * j] = sum_b*wsum;
						dptr[3 * j + 1] = sum_g*wsum;
						dptr[3 * j + 2] = sum_r*wsum;
					}
				}
			}
		}
	private:
		const Mat *temp, *guide;
		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs, *space_ofsj;
		float threshold;
	};

	void binalyWeightedRangeFilter_32f(const Mat& src, Mat& dst, Size kernelSize, float threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp;

		int dpad = (4 - src.cols % 4) % 4;
		int spad = dpad + (4 - (2 * radiusH) % 16) % 4;
		if (spad < 4) spad += 4;
		int lpad = 4 * (radiusH / 4 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];

		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		BinalyWeightedRangeFilter_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}


	void centerReplacedBinalyWeightedRangeFilter_32f(const Mat& src, const Mat& center, Mat& dst, Size kernelSize, float threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, centert;

		int dpad = (4 - src.cols % 4) % 4;
		int spad = dpad + (4 - (2 * radiusH) % 16) % 4;
		if (spad < 4) spad += 4;
		int lpad = 4 * (radiusH / 4 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(center, centert, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
			copyMakeBorder(center, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, centert);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];

		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		CenterReplacedBinalyWeightedRangeFilter_32f_InvokerSSE4 body(dest, temp, centert, radiusH, radiusV, maxk, space_ofs, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void centerReplacedBinalyWeightedRangeFilter_8u(const Mat& src, const Mat& center, Mat& dst, Size kernelSize, uchar threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp;
		Mat centert;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(center, centert, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(center, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, centert);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];

		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		CenterReplacedBinalyWeightedRangeFilter_8u_InvokerSSE4 body(dest, temp, centert, radiusH, radiusV, maxk, space_ofs, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void binalyWeightedRangeFilter_8u(const Mat& src, Mat& dst, Size kernelSize, uchar threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];

		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		BinalyWeightedRangeFilter_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void jointBinalyWeightedRangeFilter_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, uchar threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp;
		Mat gim;
		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, gim, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, gim);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, gim, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, gim);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];

		vector<int> _space_ofsj(kernelSize.area() + 1);
		int* space_ofsj = &_space_ofsj[0];

		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk] = (int)(i*temp.cols*cn + j);
					space_ofsj[maxk++] = (int)(i*gim.cols*cng + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		JointBinalyWeightedRangeFilter_8u_InvokerSSE4 body(dest, temp, gim, radiusH, radiusV, maxk, space_ofs, space_ofsj, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void jointBinalyWeightedRangeFilter_32f(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, float threshold, int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, gim;

		int dpad = (4 - src.cols % 4) % 4;
		int spad = dpad + (4 - (2 * radiusH) % 16) % 4;
		if (spad < 4) spad += 4;
		int lpad = 4 * (radiusH / 4 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, gim, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, gim);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, gim, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, gim);
		}

		vector<int> _space_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		vector<int> _space_ofsj(kernelSize.area() + 1);
		int* space_ofsj = &_space_ofsj[0];
		// initialize space-related bilateral filter coefficients
		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				for (j = -radiusH; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH))
						continue;
					space_ofs[maxk] = (int)(i*temp.cols*cn + j);
					space_ofsj[maxk++] = (int)(i*gim.cols*cng + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		JointBinalyWeightedRangeFilter_32f_InvokerSSE4 body(dest, temp, gim, radiusH, radiusV, maxk, space_ofs, space_ofsj, threshold);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void binalyWeightedRangeFilterSP_8u(const Mat& src, Mat& dst, Size kernelSize, uchar threshold, int borderType)
	{
		if (kernelSize.width <= 1) src.copyTo(dst);
		else binalyWeightedRangeFilter_8u(src, dst, Size(kernelSize.width, 1), threshold, borderType, false);

		if (kernelSize.width > 1)
			binalyWeightedRangeFilter_8u(dst, dst, Size(1, kernelSize.height), threshold, borderType, false);
	}

	void binalyWeightedRangeFilterSP_32f(const Mat& src, Mat& dst, Size kernelSize, float threshold, int borderType)
	{
		if (kernelSize.width <= 1) src.copyTo(dst);
		else binalyWeightedRangeFilter_32f(src, dst, Size(kernelSize.width, 1), threshold, borderType, false);

		if (kernelSize.width > 1)
			binalyWeightedRangeFilter_32f(dst, dst, Size(1, kernelSize.height), threshold, borderType, false);
	}

	void binalyWeightedRangeFilter(cv::InputArray src_, cv::OutputArray dst_, int D, float threshold, int method, int borderType)
	{
		binalyWeightedRangeFilter(src_, dst_, Size(D, D), threshold, method, borderType);
	}

	void binalyWeightedRangeFilter(cv::InputArray src_, cv::OutputArray dst_, Size kernelSize, float threshold, int method, int borderType)
	{
		if (dst_.empty())dst_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dst = dst_.getMat();

		if (method == FILTER_SLOWEST)
		{
			if (src.depth() == CV_8U)
			{
				binalyWeightedRangeFilterNONSSE_8u(src, dst, kernelSize, (uchar)threshold, borderType, false);
			}
			else if (src.depth() == CV_32F)
			{
				binalyWeightedRangeFilterNONSSE_32f(src, dst, kernelSize, threshold, borderType, false);
			}
			else
			{
				CV_Assert(method != FILTER_SLOWEST);
			}
		}
		else if (method == FILTER_CIRCLE || method == FILTER_DEFAULT)
		{
			if (src.depth() == CV_8U)
			{
				binalyWeightedRangeFilter_8u(src, dst, kernelSize, (uchar)threshold, borderType, false);
			}
			else if (src.depth() == CV_16S || src.depth() == CV_16U)
			{
				Mat srcf; src.convertTo(srcf, CV_32F);
				Mat destf(src.size(), CV_32F);

				binalyWeightedRangeFilter_32f(srcf, destf, kernelSize, threshold, borderType, false);
				destf.convertTo(dst, src.depth(), 1.0, 0.0);
			}
			else if (src.depth() == CV_32F)
			{
				binalyWeightedRangeFilter_32f(src, dst, kernelSize, threshold, borderType, false);
			}
		}
		else if (method == FILTER_RECTANGLE)
		{
			if (src.depth() == CV_8U)
			{
				binalyWeightedRangeFilter_8u(src, dst, kernelSize, (uchar)threshold, borderType, true);
			}
			else if (src.depth() == CV_16S || src.depth() == CV_16U)
			{
				Mat srcf; src.convertTo(srcf, CV_32F);
				Mat destf(src.size(), CV_32F);

				binalyWeightedRangeFilter_32f(srcf, destf, kernelSize, threshold, borderType, true);
				destf.convertTo(dst, src.depth(), 1.0, 0.0);
			}
			else if (src.depth() == CV_32F)
			{
				binalyWeightedRangeFilter_32f(src, dst, kernelSize, threshold, borderType, true);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			if (src.type() == CV_MAKE_TYPE(CV_8U, src.channels()))
			{
				binalyWeightedRangeFilterSP_8u(src, dst, kernelSize, (uchar)threshold, borderType);
			}
			else if (src.type() == CV_MAKE_TYPE(CV_32F, src.channels()))
			{
				binalyWeightedRangeFilterSP_32f(src, dst, kernelSize, threshold, borderType);
			}
		}
	}

	void jointBinalyWeightedRangeFilter(cv::InputArray src_, cv::InputArray guide_, cv::OutputArray dst_, int D, float threshold, int method, int borderType)
	{
		jointBinalyWeightedRangeFilter(src_, guide_, dst_, Size(D, D), threshold, method, borderType);
	}

	void jointBinalyWeightedRangeFilterSP_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, uchar threshold, int borderType)
	{
		if (kernelSize.width <= 1) src.copyTo(dst);
		else jointBinalyWeightedRangeFilter_8u(src, guide, dst, Size(kernelSize.width, 1), threshold, borderType, false);

		if (kernelSize.width > 1)
			jointBinalyWeightedRangeFilter_8u(dst, guide, dst, Size(1, kernelSize.height), threshold, borderType, false);
	}

	void jointBinalyWeightedRangeFilterSP_32f(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, float threshold, int borderType)
	{
		if (kernelSize.width <= 1) src.copyTo(dst);
		else jointBinalyWeightedRangeFilter_32f(src, guide, dst, Size(kernelSize.width, 1), threshold, borderType, false);

		if (kernelSize.width > 1)
			jointBinalyWeightedRangeFilter_32f(dst, guide, dst, Size(1, kernelSize.height), threshold, borderType, false);
	}

	void jointBinalyWeightedRangeFilter(cv::InputArray src_, cv::InputArray guide_, cv::OutputArray dst_, Size kernelSize, float threshold, int method, int borderType)
	{
		if (dst_.empty())dst_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();
		Mat dst = dst_.getMat();

		CV_Assert(method != FILTER_SLOWEST);
		CV_Assert(src.depth() == guide.depth());

		if (method == FILTER_CIRCLE || method == FILTER_DEFAULT)
		{
			if (src.depth() == CV_8U)
			{
				jointBinalyWeightedRangeFilter_8u(src, guide, dst, kernelSize, (uchar)threshold, borderType, false);
			}
			else if (src.depth() == CV_16S || src.depth() == CV_16U)
			{
				Mat srcf, guidef;
				Mat destf(src.size(), CV_32F);
				src.convertTo(srcf, CV_32F);
				guide.convertTo(guidef, CV_32F);

				jointBinalyWeightedRangeFilter_32f(srcf, guidef, destf, kernelSize, threshold, borderType, false);
				destf.convertTo(dst, src.depth(), 1.0, 0.0);
			}
			else if (src.depth() == CV_32F)
			{
				jointBinalyWeightedRangeFilter_32f(src, guide, dst, kernelSize, threshold, borderType, false);
			}
		}
		else if (method == FILTER_RECTANGLE)
		{
			if (src.depth() == CV_8U)
			{
				jointBinalyWeightedRangeFilter_8u(src, guide, dst, kernelSize, (uchar)threshold, borderType, true);
			}
			else if (src.depth() == CV_16S || src.depth() == CV_16U)
			{
				Mat srcf, guidef;
				Mat destf(src.size(), CV_32F);
				src.convertTo(srcf, CV_32F);
				guide.convertTo(guidef, CV_32F);

				jointBinalyWeightedRangeFilter_32f(srcf, guidef, destf, kernelSize, threshold, borderType, true);
				destf.convertTo(dst, src.depth(), 1.0, 0.0);
			}
			else if (src.depth() == CV_32F)
			{
				jointBinalyWeightedRangeFilter_32f(src, guide, dst, kernelSize, threshold, borderType, true);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			if (src.depth() == CV_8U)
			{
				jointBinalyWeightedRangeFilterSP_8u(src, guide, dst, kernelSize, (uchar)threshold, borderType);
			}
			else if (src.depth() == CV_32F)
			{
				jointBinalyWeightedRangeFilterSP_32f(src, guide, dst, kernelSize, threshold, borderType);
			}
		}
	}


	void centerReplacedBinalyWeightedRangeFilterSP_8u(const Mat& src, const Mat& center, Mat& dst, Size kernelSize, uchar threshold, int borderType)
	{
		if (kernelSize.width <= 1) src.copyTo(dst);
		else centerReplacedBinalyWeightedRangeFilter_8u(src, center, dst, Size(kernelSize.width, 1), threshold, borderType, false);

		if (kernelSize.width > 1)
			centerReplacedBinalyWeightedRangeFilter_8u(dst, center, dst, Size(1, kernelSize.height), threshold, borderType, false);
	}

	void centerReplacedBinalyWeightedRangeFilterSP_32f(const Mat& src, const Mat& center, Mat& dst, Size kernelSize, float threshold, int borderType)
	{
		if (kernelSize.width <= 1) src.copyTo(dst);
		else centerReplacedBinalyWeightedRangeFilter_32f(src, center, dst, Size(kernelSize.width, 1), threshold, borderType, false);

		if (kernelSize.width > 1)
			centerReplacedBinalyWeightedRangeFilter_32f(dst, center, dst, Size(1, kernelSize.height), threshold, borderType, false);
	}

	void centerReplacedBinalyWeightedRangeFilter(const Mat& src, const Mat& center, Mat& dst, Size kernelSize, float threshold, int method, int borderType)
	{
		if (dst.empty())dst.create(src.size(), src.type());
		CV_Assert(method != FILTER_SLOWEST);

		if (method == FILTER_CIRCLE || method == FILTER_DEFAULT)
		{
			if (src.depth() == CV_8U)
			{
				centerReplacedBinalyWeightedRangeFilter_8u(src, center, dst, kernelSize, (uchar)threshold, borderType, false);
			}
			else if (src.depth() == CV_16S || src.depth() == CV_16U)
			{
				Mat srcf; src.convertTo(srcf, CV_32F);
				Mat centerf; center.convertTo(centerf, CV_32F);
				Mat destf(src.size(), CV_32F);

				centerReplacedBinalyWeightedRangeFilter_32f(srcf, centerf, destf, kernelSize, threshold, borderType, false);
				destf.convertTo(dst, src.depth(), 1.0);
			}
			else if (src.depth() == CV_32F)
			{
				centerReplacedBinalyWeightedRangeFilter_32f(src, center, dst, kernelSize, threshold, borderType, false);
			}
		}
		else if (method == FILTER_RECTANGLE)
		{
			if (src.depth() == CV_8U)
			{
				centerReplacedBinalyWeightedRangeFilter_8u(src, center, dst, kernelSize, (uchar)threshold, borderType, true);
			}
			else if (src.depth() == CV_16S || src.depth() == CV_16U)
			{
				Mat srcf; src.convertTo(srcf, CV_32F);
				Mat centerf; center.convertTo(centerf, CV_32F);
				Mat destf(src.size(), CV_32F);

				centerReplacedBinalyWeightedRangeFilter_32f(srcf, centerf, destf, kernelSize, threshold, borderType, true);
				destf.convertTo(dst, src.depth(), 1.0, 0.0);
			}
			else if (src.depth() == CV_32F)
			{
				centerReplacedBinalyWeightedRangeFilter_32f(src, center, dst, kernelSize, threshold, borderType, true);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			if (src.type() == CV_MAKE_TYPE(CV_8U, src.channels()))
			{
				centerReplacedBinalyWeightedRangeFilterSP_8u(src, center, dst, kernelSize, (uchar)threshold, borderType);
			}
			else if (src.type() == CV_MAKE_TYPE(CV_32F, src.channels()))
			{
				centerReplacedBinalyWeightedRangeFilterSP_32f(src, center, dst, kernelSize, threshold, borderType);
			}
		}
	}
}