#include "weightedHistogramFilter.hpp"
#include "copyMakeBorder.hpp"
#include "debugcp.hpp"
#include <omp.h>
#include <inlineSIMDFunctions.hpp>
using namespace std;
using namespace cv;

namespace cp
{

#pragma region WeightedHistogram

	static float* createLUTHistogram(const float clip, float sigmaHistogram)
	{
		const int r = (int)ceil(clip * sigmaHistogram);
		const int size = get_simd_ceil(2 * r + 1, 8);
		float* lut = (float*)_mm_malloc(size * sizeof(float), AVX_ALIGN);
		const float coeff = 1.f / (-2.f * sigmaHistogram * sigmaHistogram);


		for (int i = 0; i < size; i += 8)_mm256_store_ps(lut + i, _mm256_setzero_ps());

		for (int i = 0; i < r; i++)
		{
			const float v = exp(i * i * coeff);
			lut[r + i] = v;
			lut[r - i] = v;
		}


		return lut;
	}

	static float* createLUTRange(const int range_max, float sigmaColor)
	{
		float* lut = (float*)_mm_malloc(range_max * sizeof(float), AVX_ALIGN);
		const float coeff = 1.f / (-2.f * sigmaColor * sigmaColor);
		for (int i = 0; i < range_max; i++)
		{
			lut[i] = exp(i * i * coeff);
		}
		return lut;
	}

	static float* createLUTSpace(const int r, const float sigmaSpace)
	{
		float* lut = (float*)_mm_malloc((2 * r + 1) * (2 * r + 1) * sizeof(float), AVX_ALIGN);
		const float coeff = 1.f / (-2.f * sigmaSpace * sigmaSpace);
		for (int j = 0, idx = 0; j < 2 * r + 1; j++)
		{
			float jd = float((j - r) * (j - r));
			for (int i = 0; i < 2 * r + 1; i++, idx++)
			{
				lut[idx] = exp(((i - r) * (i - r) + jd) * coeff);
			}
		}
		return lut;
	}

	class WeightedHistogram
	{
	private:
		float* histbuff;
		int histbuffsize;
	public:
		int r;
		int histMin;
		int histMax;
		int histSize;

		float sigmaHistogram;
		float clip;
		float* hist;
		int simd_range;//get_simd_ceil(2 * r + 1, 8);
		void print_parameter()
		{
			print_debug(sigmaHistogram);
			print_debug(clip);
			print_debug(r);
			print_debug(histSize);

			print_debug(simd_range);
			print_debug(histbuffsize);
		}
		WeightedHistogram(const float sigmaHistogram, const int mode = WeightedHistogram::MAX, const int max_val = 256, const float clip = 1.f)
		{
			histSize = max_val + 1;
			this->clip = clip;
			this->sigmaHistogram = sigmaHistogram;
			r = (int)ceil(sigmaHistogram * clip);

			simd_range = get_simd_ceil(2 * r + 1, 8);
			histbuffsize = max_val + 1 + simd_range;
			histbuff = (float*)_mm_malloc(sizeof(float) * histbuffsize, AVX_ALIGN);
			hist = histbuff + r;//border

			this->mode = mode;

			clear();
		}

		virtual ~WeightedHistogram()
		{
			_mm_free(histbuff);
		}
		void clear()
		{
			histMin = 0;
			histMax = histSize;
			const int simd_size = histbuffsize / 8;
			float* h = histbuff;
			for (int i = 0; i < simd_size; i++)
			{
				_mm256_store_ps(h, _mm256_setzero_ps());
				h += 8;
			}
		}
		virtual void add(const float addval, const int bin) = 0;
		virtual void addWithRange(const float addval, const int bin) = 0;

		enum
		{
			MAX = 0,
			MEDIAN
		};
		int mode;

		int returnVal()
		{
			if (mode == 0)
				return returnMax();
			else
				return returnMedian();
		}
		int returnMax()
		{
			float maxv = 0.f;
			int maxbin;
#if 0
			for (int i = histMin; i < histMax; i++)
			{
				if (hist[i] > maxv)
				{
					maxv = hist[i];
					maxbin = i;
				}
			}
#else 
			__m256 mmaxv = _mm256_setzero_ps();
			__m256 mindex = _mm256_setzero_ps();
			const __m256 step = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
			for (int i = histMin; i < histMax; i += 8)
			{
				__m256 mhist = _mm256_loadu_ps(hist + i);
				__m256 mmask = _mm256_cmp_ps(mhist, mmaxv, _CMP_GT_OQ);
				mmaxv = _mm256_blendv_ps(mmaxv, mhist, mmask);
				//integer operation is better
				mindex = _mm256_blendv_ps(mindex, _mm256_add_ps(step, _mm256_set1_ps(i)), mmask);
			}

			maxv = ((float*)&mmaxv)[0];
			maxbin = (int)((float*)&mindex)[0];
			for (int i = 1; i < 8; i++)
			{
				if (((float*)&mmaxv)[i] > maxv)
				{
					maxv = ((float*)&mmaxv)[i];
					maxbin = (int)(((float*)&mindex)[i]);
				}
			}
#endif
			return maxbin;
		}
		int returnMedian()
		{
			float maxval = 0.f;
			for (int i = histMin; i < histMax; i++)
			{
				maxval += hist[i];
			}
			const float half_max = maxval * 0.5f;

			int maxbin;
			maxval = 0.f;
			for (int i = histMin; i < histMax; i++)
			{
				maxval += hist[i];
				if (maxval > half_max)
				{
					maxbin = i;
					break;
				}
			}
			return maxbin;
		}
		int returnMaxwithRange()
		{
			histMin = max(histMin, 0);
			histMax = min(histMax, 255);
			float maxv = 0.f;
			int maxbin;

			for (int i = histMin; i < histMax; i++)
			{
				if (hist[i] > maxv)
				{
					maxv = hist[i];
					maxbin = i;
				}
			}
			return maxbin;
		}
		int returnMedianwithRange()
		{
			histMin = max(histMin, 0);
			histMax = min(histMax, 255);
			float maxval = 0.f;
			for (int i = histMin; i < histMax; i++)
			{
				maxval += hist[i];
			}
			const float half_max = maxval * 0.5f;

			int maxbin;
			maxval = 0.f;
			for (int i = histMin; i < histMax; i++)
			{
				maxval += hist[i];
				if (maxval > half_max)
				{
					maxbin = i;
					break;
				}
			}
			return maxbin;
		}
	};

	class WeightedHistogramIMPULSE : public WeightedHistogram
	{
	public:
		WeightedHistogramIMPULSE(float sigmaHistogram, int mode = WeightedHistogram::MAX, const int max_val = 256) :WeightedHistogram(sigmaHistogram, mode, max_val)
		{
			;
		}
		void add(const float addval, const int bin)
		{
			hist[bin] += addval;
		}
		void addWithRange(float addval, int bin)
		{
			;
		}
	};

	class WeightedHistogramLINEAR : public WeightedHistogram
	{
		const float div;
	public:
		WeightedHistogramLINEAR(float sigmaHistogram, int mode, const int max_val) :
			WeightedHistogram(sigmaHistogram, mode, max_val), div(1.f / float(sigmaHistogram))
		{
			;
		}

		void add(const float addval, const int bin)
		{
			__m256 mv = _mm256_set1_ps(addval);
			const __m256 step = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
			float* hptr = hist + bin - r;
			//for (int i = 0; i < simd_size; i++)
			for (int i = 0; i < simd_range; i += 8)
			{
				__m256 mvv = _mm256_abs_ps(_mm256_mul_ps(_mm256_set1_ps(div), _mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps((float)i), step), _mm256_set1_ps(sigmaHistogram))));
				_mm256_store_ps(hptr + i, _mm256_max_ps(_mm256_setzero_ps(), _mm256_fmadd_ps(mv, _mm256_sub_ps(_mm256_set1_ps(1.f), mvv), _mm256_load_ps(hptr + i))));

				//const float v = abs(i - truncate) * div;
				//const float val = max(0.f, addval * (1.f - v));
				//hptr[i] += val;
			}
		}
		void addWithRange(float addval, int bin)
		{
			;
		}
	};

	class WeightedHistogramQUADRIC : public WeightedHistogram
	{
		const float div;
	public:
		WeightedHistogramQUADRIC(float sigmaHistogram, int mode, const int max_val) :
			WeightedHistogram(sigmaHistogram, mode, max_val), div(1.f / float(sigmaHistogram))
		{
			;
		}

		void add(const float addval, const int bin)
		{
			const __m256 step = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
			__m256 mv = _mm256_set1_ps(addval);
			float* hptr = hist + bin - r;
			//for (int i = 0; i < simd_size; i++)
			for (int i = 0; i < simd_range; i += 8)
			{
				__m256 mvv = _mm256_mul_ps(_mm256_set1_ps(div), _mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps((float)i), step), _mm256_set1_ps(sigmaHistogram)));
				_mm256_store_ps(hptr + i, _mm256_max_ps(_mm256_setzero_ps(), _mm256_fmadd_ps(mv, _mm256_fnmadd_ps(mvv, mvv, _mm256_set1_ps(1.f)), _mm256_load_ps(hptr + i))));

				//const float v = (i - truncate) * div;
				//const float val = max(0.f, addval * (1.f - v * v)));
				//hptr[i] += val;
			}
		}
		void addWithRange(float addval, int bin)
		{
			;
		}
	};

	class WeightedHistogramGAUSSIAN : public WeightedHistogram
	{
		float* gauss;
	public:
		WeightedHistogramGAUSSIAN(float sigmaHistogram, int mode, const int max_val, float* lut, const float clip_val) :
			WeightedHistogram(sigmaHistogram, mode, max_val, clip_val)
		{
			gauss = lut;
		}

		void add(const float addval, const int bin)
		{
			__m256 mv = _mm256_set1_ps(addval);
			float* hptr = hist + bin - r;
			for (int i = 0; i < simd_range; i += 8)
			{
				_mm256_store_ps(hptr + i, _mm256_fmadd_ps(mv, _mm256_load_ps(gauss + i), _mm256_load_ps(hptr + i)));
			}
			/*
			hist[bin] += addval;
			for (int i = 1; i < truncate; i++)
			{
				float val = addval * gauss[i];
				hist[bin + i] += val;
				hist[bin - i] += val;
			}
			*/
		}
		void addWithRange(float addval, int bin)
		{
			;
		}
	};

	Ptr<WeightedHistogram> createWeightedHistogram(WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, float sigmaHistogram, int mode = WeightedHistogram::MAX, const int max_val = 256, float* lut = nullptr, const float clip_val = 3.f)
	{
		Ptr<WeightedHistogram> ret;
		switch (weightFunctionType)
		{
		case	WHF_HISTOGRAM_WEIGHT_FUNCTION::IMPULSE: ret = new WeightedHistogramIMPULSE(sigmaHistogram, mode, max_val); break;
		case 	WHF_HISTOGRAM_WEIGHT_FUNCTION::LINEAR:	ret = new WeightedHistogramLINEAR(sigmaHistogram, mode, max_val); break;
		case	WHF_HISTOGRAM_WEIGHT_FUNCTION::QUADRIC: ret = new WeightedHistogramQUADRIC(sigmaHistogram, mode, max_val); break;
		case	WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN:ret = new WeightedHistogramGAUSSIAN(sigmaHistogram, mode, max_val, lut, clip_val); break;
		default:
			break;
		}
		return ret;
	}


#pragma endregion

	enum class WHF_WEIGHT
	{
		BOX,
		GAUSSIAN,
		BILATERAL
	};

	std::string getWHFHistogramWeightName(const WHF_HISTOGRAM_WEIGHT_FUNCTION method)
	{
		string ret = "no support WHF_HISTOGRAM_WEIGHT";
		switch (method)
		{
		case WHF_HISTOGRAM_WEIGHT_FUNCTION::IMPULSE:	ret = "IMPULSE"; break;
		case WHF_HISTOGRAM_WEIGHT_FUNCTION::LINEAR:		ret = "LINEAR"; break;
		case WHF_HISTOGRAM_WEIGHT_FUNCTION::QUADRIC:	ret = "QUADRIC"; break;
		case WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN:	ret = "GAUSSIAN"; break;
		default: break;
		}
		return ret;
	}

	std::string getWHFOperationName(const WHF_OPERATION method)
	{
		string ret = "no support WHF_OPERATION";
		switch (method)
		{
		case BOX_MODE:			ret = "BOX_MODE"; break;
		case GAUSSIAN_MODE:		ret = "GAUSSIAN_MODE"; break;
		case BILATERAL_MODE:	ret = "BILATERAL_MODE"; break;
		case BOX_MEDIAN:		ret = "BOX_MEDIAN"; break;
		case GAUSSIAN_MEDIAN:	ret = "GAUSSIAN_MEDIAN"; break;
		case BILATERAL_MEDIAN:	ret = "BILATERAL_MEDIAN"; break;
		default:
			break;
		}
		return ret;
	}

	template<typename srcType, typename guideType>
	void weightedHistogramFilter_(Mat& src, Mat& guide, Mat& dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType, Mat& mask)
	{
		const bool isSkipMask = (mask.empty()) ? false : true;

		double src_maxf;
		minMaxLoc(src, NULL, &src_maxf);
		int src_max = int(src_maxf);
		src_max = get_simd_ceil(src_max, 8);
		int width = src.cols;
		int height = src.rows;

		Mat srcBorder; copyMakeBorder(src, srcBorder, r, r, r, r, borderType);

		const int mode = (method >= WHF_OPERATION::BOX_MEDIAN) ? WeightedHistogram::MEDIAN : WeightedHistogram::MAX;
		WHF_WEIGHT weight_method;
		switch (method)
		{
		case BOX_MODE:
		case BOX_MEDIAN:
			weight_method = WHF_WEIGHT::BOX; break;
		case GAUSSIAN_MODE:
		case GAUSSIAN_MEDIAN:
			weight_method = WHF_WEIGHT::GAUSSIAN; break;
		case BILATERAL_MODE:
		case BILATERAL_MEDIAN:
			weight_method = WHF_WEIGHT::BILATERAL; break;
		default:
			break;
		}

		float* luth = (weightFunctionType == WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN) ? createLUTHistogram(3.f, (float)sigmaHistogram) : nullptr;
		float* luts = createLUTSpace(r, (float)sigmaSpace);
		if (guide.channels() == 3)
		{
			vector<Mat> guideBGR;
			cp::splitCopyMakeBorder(guide, guideBGR, r, r, r, r, borderType);

			float* lutc = createLUTRange(443, (float)sigmaColor);

			if (weight_method == WHF_WEIGHT::BILATERAL)
			{
				int thread_max = omp_get_max_threads();
				int thread_range = height / thread_max;
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;

						h->clear();
						const guideType bb = guideBGR[0].at<guideType>(y + r, x + r);
						const guideType gg = guideBGR[1].at<guideType>(y + r, x + r);
						const guideType rr = guideBGR[2].at<guideType>(y + r, x + r);

						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							guideType* bp = guideBGR[0].ptr<guideType>(y + j, x);
							guideType* gp = guideBGR[1].ptr<guideType>(y + j, x);
							guideType* rp = guideBGR[2].ptr<guideType>(y + j, x);

							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								float addval = luts[idx];

								int diff = cvRound(sqrt((bb - *bp) * (bb - *bp) + (gg - *gp) * (gg - *gp) + (rr - *rp) * (rr - *rp)));
								addval *= lutc[diff];
								h->add(addval, (int)*sp);

								sp++;
								bp++;
								gp++;
								rp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			if (weight_method == WHF_WEIGHT::GAUSSIAN)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								h->add(luts[idx], (int)*sp);
								sp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			else if (weight_method == WHF_WEIGHT::BOX)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								h->add(1.f, (int)srcBorder.at<srcType>(y + j, x + i));
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			_mm_free(lutc);
		}
		else if (guide.channels() == 1)
		{
			Mat G; copyMakeBorder(guide, G, r, r, r, r, borderType);

			float* lutc = createLUTRange(256, (float)sigmaColor);
			float* luts = createLUTSpace(r, (float)sigmaSpace);

			if (weight_method == WHF_WEIGHT::BILATERAL)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						const guideType gg = guide.at<guideType>(y, x);
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							guideType* gp = G.ptr<guideType>(y + j); gp += x;
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								float addval = luts[idx];
								int diff = abs(gg - *gp);
								addval *= lutc[diff];
								h->add(addval, (int)*sp);
								sp++;
								gp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			if (weight_method == WHF_WEIGHT::GAUSSIAN)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								float addval = luts[idx];
								h->add(addval, (int)*sp);
								sp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			else if (weight_method == WHF_WEIGHT::BOX)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								h->add(1.f, (int)srcBorder.at<srcType>(y + j, x + i));
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}

			_mm_free(lutc);
		}
		_mm_free(luts);
		_mm_free(luth);
	}


	template<typename srcType, typename guideType>
	void weightedWeightedHistogramFilter_(Mat& src, Mat& weight, Mat& guide, Mat& dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType, Mat& mask)
	{
		CV_Assert(src.channels() == 1);
		const bool isSkipMask = (mask.empty()) ? false : true;

		double src_maxf;
		minMaxLoc(src, NULL, &src_maxf);
		int src_max = int(src_maxf);
		src_max = get_simd_ceil(src_max, 8);
		int width = src.cols;
		int height = src.rows;

		Mat srcBorder; copyMakeBorder(src, srcBorder, r, r, r, r, borderType);
		Mat weightBorder; copyMakeBorder(weight, weightBorder, r, r, r, r, borderType);

		const int mode = (method >= WHF_OPERATION::BOX_MEDIAN) ? WeightedHistogram::MEDIAN : WeightedHistogram::MAX;
		WHF_WEIGHT weight_method;
		switch (method)
		{
		case BOX_MODE:
		case BOX_MEDIAN:
			weight_method = WHF_WEIGHT::BOX; break;
		case GAUSSIAN_MODE:
		case GAUSSIAN_MEDIAN:
			weight_method = WHF_WEIGHT::GAUSSIAN; break;
		case BILATERAL_MODE:
		case BILATERAL_MEDIAN:
			weight_method = WHF_WEIGHT::BILATERAL; break;
		default:
			break;
		}

		float* luth = (weightFunctionType == WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN) ? createLUTHistogram(3.f, (float)sigmaHistogram) : nullptr;
		float* luts = createLUTSpace(r, (float)sigmaSpace);
		if (guide.channels() == 3)
		{
			vector<Mat> guideBGR;
			cp::splitCopyMakeBorder(guide, guideBGR, r, r, r, r, borderType);

			float* lutc = createLUTRange(443, (float)sigmaColor);

			if (weight_method == WHF_WEIGHT::BILATERAL)
			{
				int thread_max = omp_get_max_threads();
				int thread_range = height / thread_max;
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;

						h->clear();
						const guideType bb = guideBGR[0].at<guideType>(y + r, x + r);
						const guideType gg = guideBGR[1].at<guideType>(y + r, x + r);
						const guideType rr = guideBGR[2].at<guideType>(y + r, x + r);

						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							float* wp = weightBorder.ptr<float>(y + j, x);
							guideType* bp = guideBGR[0].ptr<guideType>(y + j, x);
							guideType* gp = guideBGR[1].ptr<guideType>(y + j, x);
							guideType* rp = guideBGR[2].ptr<guideType>(y + j, x);

							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								float addval = luts[idx] * *wp;

								int diff = cvRound(sqrt((bb - *bp) * (bb - *bp) + (gg - *gp) * (gg - *gp) + (rr - *rp) * (rr - *rp)));
								addval *= lutc[diff];
								h->add(addval, (int)*sp);

								sp++;
								wp++;
								bp++;
								gp++;
								rp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			if (weight_method == WHF_WEIGHT::GAUSSIAN)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							float* wp = weightBorder.ptr<float>(y + j, x);
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								h->add(luts[idx] * *wp, (int)*sp);
								sp++;
								wp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			else if (weight_method == WHF_WEIGHT::BOX)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								h->add(weightBorder.at<float>(y + j, x + i), (int)srcBorder.at<srcType>(y + j, x + i));
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			_mm_free(lutc);
		}
		else if (guide.channels() == 1)
		{
			Mat G; copyMakeBorder(guide, G, r, r, r, r, borderType);

			float* lutc = createLUTRange(256, (float)sigmaColor);
			float* luts = createLUTSpace(r, (float)sigmaSpace);

			if (weight_method == WHF_WEIGHT::BILATERAL)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						const guideType gg = guide.at<guideType>(y, x);
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							float* wp = weightBorder.ptr<float>(y + j, x);
							guideType* gp = G.ptr<guideType>(y + j); gp += x;
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								float addval = luts[idx] * *wp;
								int diff = abs(gg - *gp);
								addval *= lutc[diff];
								h->add(addval, (int)*sp);
								sp++;
								wp++;
								gp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			if (weight_method == WHF_WEIGHT::GAUSSIAN)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							srcType* sp = srcBorder.ptr<srcType>(y + j, x);
							float* wp = weightBorder.ptr<float>(y + j, x);
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								float addval = luts[idx] * *wp;
								h->add(addval, (int)*sp);
								sp++;
								wp++;
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}
			else if (weight_method == WHF_WEIGHT::BOX)
			{
#pragma omp parallel for
				for (int y = 0; y < height; y++)
				{
					Ptr<WeightedHistogram> h = createWeightedHistogram(weightFunctionType, (float)sigmaHistogram, mode, src_max, luth);
					for (int x = 0; x < width; x++)
					{
						if (isSkipMask)
							if (mask.at<uchar>(y, x) == 0)continue;
						h->clear();
						for (int j = 0, idx = 0; j < 2 * r + 1; j++)
						{
							for (int i = 0; i < 2 * r + 1; i++, idx++)
							{
								h->add(weightBorder.at<float>(y + j, x + i), (int)srcBorder.at<srcType>(y + j, x + i));
							}
						}
						dst.at<srcType>(y, x) = saturate_cast<srcType>(h->returnVal());
					}
				}
			}

			_mm_free(lutc);
		}
		_mm_free(luts);
		_mm_free(luth);
	}


	void weightedHistogramFilter(InputArray src_, InputArray guide_, OutputArray dst_, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType, InputArray mask_)
	{
		dst_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();
		Mat dst = dst_.getMat();
		Mat mask = mask_.getMat();
		CV_Assert(guide.depth() == CV_8U);
		if (src.channels() == 1)
		{
			if (src.depth() == CV_8U && guide.depth() == CV_8U)weightedHistogramFilter_<uchar, uchar>(src, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
			if (src.depth() == CV_16S && guide.depth() == CV_8U)weightedHistogramFilter_<short, uchar>(src, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
			if (src.depth() == CV_16U && guide.depth() == CV_8U)weightedHistogramFilter_<ushort, uchar>(src, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
			if (src.depth() == CV_32F && guide.depth() == CV_8U)weightedHistogramFilter_<float, uchar>(src, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
		}
		else
		{
			vector<Mat> v;
			split(src, v);
			for (int i = 0; i < src.channels(); i++)
			{
				if (src.depth() == CV_8U && guide.depth() == CV_8U)weightedHistogramFilter_<uchar, uchar>(v[i], guide, v[i], r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
				if (src.depth() == CV_16S && guide.depth() == CV_8U)weightedHistogramFilter_<short, uchar>(v[i], guide, v[i], r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
				if (src.depth() == CV_16U && guide.depth() == CV_8U)weightedHistogramFilter_<ushort, uchar>(v[i], guide, v[i], r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
				if (src.depth() == CV_32F && guide.depth() == CV_8U)weightedHistogramFilter_<float, uchar>(v[i], guide, v[i], r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
			}
			merge(v, dst);
		}
	}

	void weightedWeightedHistogramFilter(InputArray src_, InputArray weight_, InputArray guide_, OutputArray dst_, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType, InputArray mask_)
	{
		dst_.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat weight = weight_.getMat();
		Mat guide = guide_.getMat();
		Mat dst = dst_.getMat();
		Mat mask = mask_.getMat();

		CV_Assert(guide.depth() == CV_8U);
		CV_Assert(weight.depth() == CV_32F);

		if (src.depth() == CV_8U && guide.depth() == CV_8U)weightedWeightedHistogramFilter_<uchar, uchar>(src, weight, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
		if (src.depth() == CV_16S && guide.depth() == CV_8U)weightedWeightedHistogramFilter_<short, uchar>(src, weight, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
		if (src.depth() == CV_16U && guide.depth() == CV_8U)weightedWeightedHistogramFilter_<ushort, uchar>(src, weight, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
		if (src.depth() == CV_32F && guide.depth() == CV_8U)weightedWeightedHistogramFilter_<float, uchar>(src, weight, guide, dst, r, sigmaColor, sigmaSpace, sigmaHistogram, weightFunctionType, method, borderType, mask);
	}

	void weightedModeFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType, cv::InputArray mask)
	{
		Mat s = src.getMat();
		Mat g = guide.getMat();
		if (dst.empty()) dst.create(src.size(), src.type());
		Mat d = dst.getMat();
		weightedHistogramFilter(s, g, d, r, sigmaColor, sigmaSpace, sigmaHistogram, WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN, WHF_OPERATION::BILATERAL_MODE, borderType, mask);
	}

	void weightedWeightedModeFilter(cv::InputArray src, cv::InputArray weight, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType, cv::InputArray mask)
	{
		Mat s = src.getMat();
		Mat w = weight.getMat();
		Mat g = guide.getMat();
		if (dst.empty()) dst.create(src.size(), src.type());
		Mat d = dst.getMat();
		weightedWeightedHistogramFilter(s, w, g, d, r, sigmaColor, sigmaSpace, sigmaHistogram, WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN, WHF_OPERATION::BILATERAL_MODE, borderType, mask);
	}

	void weightedMedianFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType, cv::InputArray mask)
	{
		Mat s = src.getMat();
		Mat g = guide.getMat();
		if (dst.empty()) dst.create(src.size(), src.type());
		Mat d = dst.getMat();
		weightedHistogramFilter(s, g, d, r, sigmaColor, sigmaSpace, sigmaHistogram, WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN, WHF_OPERATION::BILATERAL_MEDIAN, borderType, mask);
	}

	void weightedWeightedMedianFilter(cv::InputArray src, cv::InputArray weight, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const int borderType, cv::InputArray mask)
	{
		Mat s = src.getMat();
		Mat w = weight.getMat();
		Mat g = guide.getMat();
		if (dst.empty()) dst.create(src.size(), src.type());
		Mat d = dst.getMat();

		weightedWeightedHistogramFilter(s, w, g, d, r, sigmaColor, sigmaSpace, sigmaHistogram, WHF_HISTOGRAM_WEIGHT_FUNCTION::GAUSSIAN, WHF_OPERATION::BILATERAL_MEDIAN, borderType, mask);
	}
}