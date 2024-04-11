#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

using namespace std;
using namespace cv;
namespace cp
{
#pragma region xorshift
	struct xorshift_32_4 {
		static unsigned int x[4];
		static int cnt;
		unsigned int operator()() {
			if (cnt != 4) return x[cnt++];
			__m128i a;
			a = _mm_loadu_si128((__m128i*)x);
			a = _mm_xor_si128(a, _mm_slli_epi32(a, 13));
			a = _mm_xor_si128(a, _mm_srli_epi32(a, 17));
			a = _mm_xor_si128(a, _mm_slli_epi32(a, 5));
			_mm_storeu_si128((__m128i*)x, a);
			cnt = 1;
			return x[0];
		}
	};
	int xorshift_32_4::cnt = 0;
	unsigned int xorshift_32_4::x[] = { 0xf247756d, 0x1654caaa, 0xb2f5e564, 0x7d986dd7 };

	//https://github.com/Tlapesium/cpp-lib/blob/master/utility/xorshift.cpp
		//need AVX2
	struct _MM_XORSHIFT32_AVX2
	{
		__m256 normal;
		__m256i x;
		int cnt = 0;
		//public:
		_MM_XORSHIFT32_AVX2()
		{
			x = _mm256_set_epi32(0xd5eae750, 0xc784b986, 0x16bcf701, 0x65032360, 0xb628094f, 0xd8281e7b, 0xecfa5dc8, 0x3b828203);
			normal = _mm256_set1_ps(1.f / (INT_MAX));
		}
		__m256 next32f()
		{
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
			x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
			return _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
		}
		__m256i next()
		{
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
			x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
			return x;
		}

		unsigned int operator()()
		{
			if (cnt != 8) return x.m256i_i32[cnt++];
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
			x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
			x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
			cnt = 1;
			return x.m256i_i32[0];
		}
	};

	struct xorshift_64_4 {
		static unsigned long long int x[4];
		static int cnt;
		unsigned long long int operator()() {
			if (cnt != 4) return x[cnt++];
			__m256i a;
			a = _mm256_loadu_si256((__m256i*)x);
			a = _mm256_xor_si256(a, _mm256_slli_epi64(a, 7));
			a = _mm256_xor_si256(a, _mm256_srli_epi64(a, 9));
			_mm256_storeu_si256((__m256i*)x, a);
			cnt = 1;
			return x[0];
		}
	};

	int xorshift_64_4::cnt = 0;
	unsigned long long int xorshift_64_4::x[] = { 0xf77bcfb23d5143cfULL, 0xbda154512ac6f703ULL, 0xb2ef653838c2edf3ULL, 0xa7dbfba7cef3c195ULL };
#pragma endregion

	void TexturenessSampling::mask2samples(const int sample_num, const Mat& sampleMask, const vector<cv::Mat>& guide, cv::Mat& dest, Rect roi)
	{
		const int channels = (int)guide.size();
		if (dest.size() != Size(sample_num, channels)) dest.create(Size(sample_num, channels), CV_32F);

		AutoBuffer<const float*> s(channels);
		AutoBuffer<float*> d(channels);
		for (int c = 0; c < channels; c++)
		{
			d[c] = dest.ptr<float>(c);
		}

		for (int y = 0, count = 0; y < sampleMask.rows; y++)
		{
			for (int c = 0; c < channels; c++)
			{
				s[c] = guide[c].ptr<float>(y + roi.y, roi.x);
			}

			const uchar* mask_ptr = sampleMask.ptr<uchar>(y);
			for (int x = 0; x < sampleMask.cols; x++)
			{
				if (mask_ptr[x] == 255)
				{
					for (int c = 0; c < channels; c++)
					{
						d[c][count] = s[c][x];
					}
					count++;
					if (count == sample_num) return;
				}
			}
		}
	}

	void TexturenessSampling::gradientMax(const Mat& src, Mat& dest, const bool isAccumurate)
	{
		if (dest.empty()) dest.create(src.size(), src.type());

		if (isAccumurate)
		{
			for (int j = 0; j < src.rows; j++)
			{
				const float* sc = src.ptr<float>(j);
				const float* sm = src.ptr<float>(max(j - 1, 0));
				const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
				//const float* smm = src.ptr<float>(max(j - 2, 0));
				//const float* spp = src.ptr<float>(min(j + 2, src.rows - 1));
				float* d = dest.ptr<float>(j);
				if constexpr (true)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
						const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
						const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
						const __m256 msp = _mm256_loadu_ps(sp + i + 0);
						const __m256 msm = _mm256_loadu_ps(sm + i + 0);

						_mm256_storeu_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_max_ps(
							_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
							_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm)))
						));
					}
				}
				else
				{
					for (int i = 1; i < src.cols - 1; i++)
					{
						//const float dx = 0.25f * (sc[i - 1] + sc[i + 1] + sc[i - 2] + sc[i + 2]);
						//const float dy = 0.25f * (sm[i] + sp[i] + smm[i] + spp[i]);
						//const float dx = 0.5f * (sc[i - 1] + sc[i + 1]);
						//const float dy = 0.5f * (sm[i] + sp[i]);

						//d[i] += sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
						//d[i] += sqrt(max((sc[i] - dx) * (sc[i] - dx), (sc[i] - dy) * (sc[i] - dy)));
						d[i] += max(max(abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])), max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
						//d[i] += 0.5f * ((abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])) + max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
					}
				}
				d[0] = 0.f;
				d[src.cols - 1] = 0.f;
			}
		}
		else
		{
			for (int j = 0; j < src.rows; j++)
			{
				const float* sc = src.ptr<float>(j);
				const float* sm = src.ptr<float>(max(j - 1, 0));
				const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
				//const float* smm = src.ptr<float>(max(j - 2, 0));
				//const float* spp = src.ptr<float>(min(j + 2, src.rows - 1));
				float* d = dest.ptr<float>(j);
				if constexpr (true)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
						const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
						const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
						const __m256 msp = _mm256_loadu_ps(sp + i + 0);
						const __m256 msm = _mm256_loadu_ps(sm + i + 0);

						_mm256_storeu_ps(d + i, _mm256_max_ps(
							_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
							_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
						));
					}
				}
				else
				{
					for (int i = 1; i < src.cols - 1; i++)
					{
						//const float dx = 0.25f * (sc[i - 1] + sc[i + 1] + sc[i - 2] + sc[i + 2]);
						//const float dy = 0.25f * (sm[i] + sp[i] + smm[i] + spp[i]);
						//const float dx = 0.5f * (sc[i - 1] + sc[i + 1]);
						//const float dy = 0.5f * (sm[i] + sp[i]);

						//d[i] = sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
						//d[i] = sqrt(max((sc[i] - dx) * (sc[i] - dx), (sc[i] - dy) * (sc[i] - dy)));
						d[i] = max(max(abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])), max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
						//d[i] = 0.5f*((abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1]))+ max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
					}
				}
				d[0] = 0.f;
				d[src.cols - 1] = 0.f;
			}
		}
	}

	void TexturenessSampling::gradientMaxDiffuse3x3(const Mat& src, Mat& dest, const bool isAccumurate, const float sigma)
	{
		if (dest.empty()) dest.create(src.size(), CV_32F);
		vector<Mat> linebuffer(3);
		for (int l = 0; l < 3; l++)
		{
			linebuffer[l].create(Size(src.cols + 2, 1), CV_32F);
			linebuffer[l].at<float>(0, 0) = 0.f;
			linebuffer[l].at<float>(0, 1) = 0.f;
			linebuffer[l].at<float>(0, linebuffer[l].cols - 2) = 0.f;
			linebuffer[l].at<float>(0, linebuffer[l].cols - 1) = 0.f;
		}
		float w0 = exp(-1.f / (2.f * sigma * sigma));
		float normal = 2.f * w0 + 1.f;
		const __m256 g0 = _mm256_set1_ps(w0 / normal);
		const __m256 g1 = _mm256_set1_ps(1.f / normal);
		if (isAccumurate)
		{
			for (int j = 0; j < src.rows; j++)
			{
				const float* sc = src.ptr<float>(j);
				const float* sm = src.ptr<float>(max(j - 1, 0));
				const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
				float* d = dest.ptr<float>(j);
				float* b = linebuffer[j % 3].ptr<float>(0, 1);
				for (int i = 0; i < src.cols; i += 8)
				{
					const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
					const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
					const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
					const __m256 msp = _mm256_loadu_ps(sp + i + 0);
					const __m256 msm = _mm256_loadu_ps(sm + i + 0);

					_mm256_storeu_ps(b + i, _mm256_max_ps(
						_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
						_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
					));
				}
				b[0] = 0.f;
				b[src.cols - 1] = 0.f;
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 blur = _mm256_mul_ps(g0, _mm256_loadu_ps(b + i - 1));
					blur = _mm256_fmadd_ps(g1, _mm256_loadu_ps(b + i + 0), blur);
					blur = _mm256_fmadd_ps(g0, _mm256_loadu_ps(b + i + 1), blur);
					_mm256_storeu_ps(b + i, blur);
				}
				for (int i = 0; i < src.cols; i += 8)
				{
					_mm256_store_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(b + i)));
				}
			}
		}
		else
		{
			for (int j = 0; j < src.rows; j++)
			{
				const float* sc = src.ptr<float>(j);
				const float* sm = src.ptr<float>(max(j - 1, 0));
				const float* sp = src.ptr<float>(min(j + 1, src.rows - 1));
				float* d = dest.ptr<float>(j);
				float* b = linebuffer[j % 3].ptr<float>(0, 1);
				for (int i = 0; i < src.cols; i += 8)
				{
					const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
					const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
					const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
					const __m256 msp = _mm256_loadu_ps(sp + i + 0);
					const __m256 msm = _mm256_loadu_ps(sm + i + 0);

					_mm256_storeu_ps(b + i, _mm256_max_ps(
						_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
						_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
					));
				}
				b[0] = 0.f;
				b[src.cols - 1] = 0.f;
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 blur = _mm256_mul_ps(g0, _mm256_loadu_ps(b + i - 1));
					blur = _mm256_fmadd_ps(g1, _mm256_loadu_ps(b + i + 0), blur);
					blur = _mm256_fmadd_ps(g0, _mm256_loadu_ps(b + i + 1), blur);
					_mm256_storeu_ps(b + i, blur);
				}
				for (int i = 0; i < src.cols; i += 8)
				{
					_mm256_store_ps(d + i, _mm256_loadu_ps(b + i));
				}
			}
		}
	}




	float TexturenessSampling::getmax(const Mat& src)
	{
		const int size = src.size().area();
		const int simdSize = get_simd_floor(size, 8);
		__m256 mmax = _mm256_setzero_ps();
		const float* s = src.ptr<float>();
		for (int i = 0; i < simdSize; i += 8)
		{
			mmax = _mm256_max_ps(mmax, _mm256_load_ps(s + i));
		}
		return _mm256_reducemax_ps(mmax);
	}

	void TexturenessSampling::normalize_max1(Mat& srcdest)
	{
		const int size = srcdest.size().area();
		const int simdSize = get_simd_floor(size, 8);
		__m256 mmax = _mm256_setzero_ps();
		float* s = srcdest.ptr<float>();
		for (int i = 0; i < simdSize; i += 8)
		{
			mmax = _mm256_max_ps(mmax, _mm256_load_ps(s + i));
		}
		float maxval = 0.f;
		for (int i = 0; i < 8; i++)
		{
			maxval = max(maxval, mmax.m256_f32[i]);
		}

		__m256 minv = _mm256_set1_ps(1.f / maxval);
		for (int i = 0; i < simdSize; i += 8)
		{
			_mm256_store_ps(s + i, _mm256_mul_ps(minv, _mm256_load_ps(s + i)));
		}
	}

	float TexturenessSampling::computeTexturenessDoG(const vector<Mat>& src, Mat& dest, const float sigma)
	{
		for (int c = 0; c < src.size(); c++)
		{
			const int r = (int)ceil(sigma * 1.5);
			const int d = 2 * r + 1;
			GaussianBlur(src[c], src_32f, Size(d, d), sigma);
			//Laplacian(src[c], src_32f, CV_32F, d);
			//gf->filter(src[c], src_32f, ss1, 1);
			if (c == 0)
			{
				absdiff(src_32f, src[c], dest);
			}
			else
			{
				absdiff(src_32f, src[c], src_32f);
				add(src_32f, dest, dest);
			}
		}

		if (edgeDiffusionSigma != 0.f)
		{
			Size ksize = Size(3, 3);
			GaussianBlur(dest, dest, ksize, edgeDiffusionSigma);
		}
		return getmax(dest);
		//normalize_max1(dest);
		//normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);
	}
	float TexturenessSampling::computeTexturenessDoG(const Mat& src, Mat& dest, const float sigma)
	{
		vector<Mat> a(1);
		a[0] = src;
		return computeTexturenessDoG(a, dest, sigma);
	}


	void TexturenessSampling::computeTexturenessBilateral(const vector<Mat>& src, Mat& dest, const float sigmaSpace, const float sr)
	{
		for (int c = 0; c < src.size(); c++)
		{
			bilateralFilter(src[c], src_32f, (int)ceil(sigmaSpace * 2) * 2 + 1, sr, sigmaSpace);
			//bilateralFilterLocalStatisticsPrior(src[c], src_32f, float(sr), (float)ss1, sr * 0.8f);
			if (c == 0)
			{
				absdiff(src_32f, src[c], dest);
			}
			else
			{
				absdiff(src_32f, src[c], src_32f);
				add(src_32f, dest, dest);
			}
		}

		if (edgeDiffusionSigma != 0)
		{
			Size ksize = Size(3, 3);
			GaussianBlur(dest, dest, ksize, edgeDiffusionSigma);
		}

		normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);
	}

	void TexturenessSampling::remap(Mat& srcdest, const float scale, const float rangemax)
	{
		const int n = srcdest.size().area();
		const __m256 ms = _mm256_set1_ps(scale / rangemax);
		const __m256 ones = _mm256_set1_ps(1.f);
		//#pragma omp parallel for schedule (dynamic)
		const int sizeSimd = n / 8;
		float* srcPtr = srcdest.ptr<float>();
		//result[i] = min(v[i] * s, 1.f);
		for (int i = 0; i < sizeSimd; i++)
		{
			_mm256_store_ps(srcPtr, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(srcPtr), ms), ones));
			srcPtr += 8;
		}
	}
	float TexturenessSampling::computeScaleHistogram(Mat& src, const float sampling_ratio, const int numBins, const float rangemin, const float rangemax)
	{
		//cp::Timer t("tt2");
		//int binNum = (int)(bin_ratio*dest.size().area()*sampling_ratio);
		if constexpr (false) //opencv
		{
			int histSize[] = { numBins };
			float value_ranges[] = { rangemin, rangemax };
			const float* ranges[] = { value_ranges };
			const int channels[] = { 0 };
			const int dims = 1;
			calcHist(&src, 1, channels, Mat(), histogram, dims, histSize, ranges, true, false);
		}
		else
		{
			histogram.create(Size(numBins, 1), CV_32S);
			histogram.setTo(0);
			int* h = histogram.ptr<int>();
			const float* s = src.ptr<float>();
			const float amp = numBins / rangemax;
			for (int i = 0; i < src.size().area(); i++)
			{
				const int idx = int(s[i] * amp);
				h[idx]++;
			}
		}

		int H_k = 0;//cumulative sum of histogram
		float X_k = 0.f;//sum of hi*xi

		float scale = 0.f;//scaling factor
		float x = 0.f;//bin center
		const float inv_m = 1.f / numBins;//1/m
		const float offset = inv_m * 0.5f;
		const int n = src.size().area();
		const int nt = int(n * (1.f - sampling_ratio));
		const float sx_max = 1.f + FLT_EPSILON;
		const float sx_min = 1.f - FLT_EPSILON;
		//cout << n<<","<<nt<<"," <<sampling_ratio<< endl;
		int argi = 0;
		for (int i = 0; i < numBins; i++)
		{
			//const int h_i = saturate_cast<int>(hist.at<float>(i));
			const int h_i = histogram.at<int>(i);
			H_k += h_i;

			x = i * inv_m + offset;
			X_k += x * h_i;

			scale = (H_k - nt) / X_k;//eq (5)
			const float sx = scale * x;
			if (sx_min < sx /*&& sx < sx_max*/)
			{
				argi = i;
				break;
			}
		}
		/*print_debug(argi);
		if (argi == 0)
		{
			for (int i = 0; i < numBins; i++)
			{
				cout<<i<<", "<<histogram.at<int>(i)<<endl;
			}
			imshowScale("test", src, 255); waitKey();
		}*/
		return scale;
	}

	//source [0:1]
	//_MM_XORSHIFT32_AVX2 mrand;
	int TexturenessSampling::randDither(const Mat& src, Mat& dest)
	{
		const __m256 normal = _mm256_set1_ps(1.f / (INT_MAX * 2.f));
		__m256i x = _mm256_set_epi32(0xd5eae750, 0xc784b986, 0x16bcf701, 0x65032360, 0xb628094f, 0xd8281e7b, 0xecfa5dc8, 0x3b828203);

		const float* s = src.ptr<float>();
		uchar* d = dest.ptr<uchar>();

		const __m256 maxval = _mm256_set1_ps(255.f);
		const __m256 mone = _mm256_set1_ps(1.0f);
		__m256i mcount = _mm256_setzero_si256();
		const int size = src.size().area();
		const __m256i fmask = _mm256_set1_epi32(0x3f800000);
		//unsigned res = (v >> 9) | 0x3f800000;
		//return (*(float*)&res) - 1.0f;

		if constexpr (true)
		{
			for (int i = 0; i < size; i += 8)
			{
				x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
				x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
				x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
				//__m256 mrng = _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
				//__m256 mrng = _mm256_fmadd_ps(normal, _mm256_cvtepi32_ps(x), mone);
				__m256 mrng = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_srli_epi32(x, 8), fmask)), mone);
				//__m256 mrng = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_srli_epi32(x, 9), fmask)), mone);
				__m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(s + i), mrng, _CMP_LE_OQ);
				__m256 mdst = _mm256_andnot_ps(mask, maxval);
				mcount = _mm256_sub_epi32(mcount, _mm256_castps_si256(mask));
				_mm_storel_epi64((__m128i*)(d + i), _mm256_cvtps_epu8(mdst));
			}
		}
		else
		{
			for (int i = 0; i < size; i += 16)
			{
				x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
				x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
				x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
				__m256 mrng = _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
				__m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(s + i), mrng, _CMP_LE_OQ);
				__m256 mdst = _mm256_andnot_ps(mask, maxval);
				mcount = _mm256_sub_epi32(mcount, _mm256_castps_si256(mask));
				_mm_storel_epi64((__m128i*)(d + i), _mm256_cvtps_epu8(mdst));

				x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
				x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
				x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
				mrng = _mm256_mul_ps(normal, _mm256_abs_ps(_mm256_cvtepi32_ps(x)));
				mask = _mm256_cmp_ps(_mm256_loadu_ps(s + i + 8), mrng, _CMP_LE_OQ);
				mdst = _mm256_andnot_ps(mask, maxval);
				mcount = _mm256_sub_epi32(mcount, _mm256_castps_si256(mask));
				_mm_storel_epi64((__m128i*)(d + i + 8), _mm256_cvtps_epu8(mdst));
			}
		}

		const int count = size - _mm256_reduceadd_epi32(mcount);
		//cout << count<<"/"<< src.size().area() << endl;
		/*
		//RNG rng;
		_MM_XORSHIFT32_AVX2 mrand;

		const float* s = src.ptr<float>();
		uchar* d = dest.ptr<uchar>();
		int count = 0;
		for (int i = 0; i < src.size().area(); i++)
		{
			__m256 vv = mrand.next32f();
			//const float v = rng.uniform(0.f, 1.f);
			const float v = vv.m256_f32[0];
			if ((v > s[i]))
			{
				d[i] = 0;
			}
			else
			{
				d[i] = 255;
				count++;
			}
		}
		*/
		/*randu(dest, 0, 256);
		const float* s = src.ptr<float>();
		uchar* d = dest.ptr<uchar>();
		int count = 0;
		for (int i = 0; i < src.size().area(); i++)
		{
			if ((d[i] > s[i] * 255.f))
			{
				d[i] = 0;
			}
			else
			{
				d[i] = 255;
				count++;
			}
		}*/

		return count;
	}

	template<int channels>
	int TexturenessSampling::remapRandomDitherSamples(const int sample_num, const float scale, const float rangemax, const Mat& prob, const vector<cv::Mat>& guide, cv::Mat& samples, Rect roi)
	{
		const float smax = scale / rangemax;
		//const int channels = (int)guide.size();
		if (samples.size() != Size(sample_num, channels)) samples.create(Size(sample_num, channels), CV_32F);

		AutoBuffer<const float*> s(channels);
		AutoBuffer<float*> d(channels);

		for (int c = 0; c < channels; c++)
		{
			d[c] = samples.ptr<float>(c);
		}

		int count = 0;
		for (int j = 0; j < prob.rows; j++)
		{
			const float* pptr = prob.ptr<float>(j);
			for (int c = 0; c < channels; c++)
			{
				s[c] = guide[c].ptr<float>(j + roi.y, roi.x);
			}
			for (int i = 0; i < prob.cols; i++)
			{
				const float v = rng.uniform(0.f, 1.f);
				if ((v <= min(1.f, smax * pptr[i])))
				{
					for (int c = 0; c < channels; c++)
					{
						d[c][count] = s[c][i];
					}
					count++;
					if (count == sample_num) return sample_num;
				}
			}
		}

		for (int i = count; i < sample_num; i++)
		{
			const int idx = rng.uniform(0, count);
			for (int c = 0; c < channels; c++)
			{
				d[c][i] = d[c][idx];
			}
		}

		return count;
	}

	//  x 7
	//3 5 1
	template<int channels>
	int TexturenessSampling::remapFloydSteinbergDitherSamples(const int sample_num, const float scale, const float rangemax, Mat& prob, const vector<cv::Mat>& guide, cv::Mat& samples, Rect roi)
	{
		const float smax = scale / rangemax;
		//const int channels = (int)guide.size();
		if (samples.size() != Size(sample_num, channels)) samples.create(Size(sample_num, channels), CV_32F);

		AutoBuffer<const float*> s(channels);
		AutoBuffer<float*> d(channels);

		for (int c = 0; c < channels; c++)
		{
			d[c] = samples.ptr<float>(c);
		}

		const int coeff1 = 1;
		const int coeff3 = 3;
		const int coeff5 = 5;
		const int coeff7 = 7;

		float total = 1.f / (coeff1 + coeff3 + coeff5 + coeff7);
		const float coeff7_16 = coeff7 * total;
		const float coeff5_16 = coeff5 * total;
		const float coeff3_16 = coeff3 * total;
		const float coeff1_16 = coeff1 * total;

		int count = 0;
		for (int j = 0; j < prob.rows - 1; j++)
		{
			float e;//error
			float* pc = prob.ptr<float>(j);
			float* pn = prob.ptr<float>(j + 1);
			for (int c = 0; c < channels; c++)
			{
				s[c] = guide[c].ptr<float>(j + roi.y, roi.x);
			}

			if (j % 2 == 1)
			{
				int i = 0;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i + 1] += e * coeff7_16;
					pn[i + 0] += e * coeff5_16;
					pn[i + 1] += e * coeff1_16;
				}
				for (i = 1; i < prob.cols - 1; i++)
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i + 1] += e * coeff7_16;
					pn[i - 1] += e * coeff3_16;
					pn[i + 0] += e * coeff5_16;
					pn[i + 1] += e * coeff1_16;
				}
				i = prob.cols - 1;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pn[i - 1] += e * 0.5f;
					pn[i + 0] += e * 0.5f;
				}
			}
			else
			{
				int i = prob.cols - 1;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i - 1] += e * coeff7_16;
					pn[i + 0] += e * coeff5_16;
					pn[i - 1] += e * coeff1_16;
				}
				for (i = prob.cols - 2; i > 0; i--)
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i - 1] += e * coeff7_16;
					pn[i + 1] += e * coeff3_16;
					pn[i + 0] += e * coeff5_16;
					pn[i - 1] += e * coeff1_16;
				}
				i = 0;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pn[i + 1] += e * 0.5f;
					pn[i + 0] += e * 0.5f;
				}
			}
		}
		// bottom y loop
		{
			float* pc = prob.ptr<float>(prob.rows - 1);
			float e;//error
			for (int i = 0; i < prob.cols - 1; i++)
			{
				if (pc[i] >= 0.5f)
				{
					e = pc[i] - 1.f;
					for (int c = 0; c < channels; c++)
					{
						d[c][count] = s[c][i];
					}
					count++;
					if (count == sample_num) return sample_num;
				}
				else
				{
					e = pc[i];
				}
				pc[i + 1] += e;
			}

			int i = prob.cols - 1;
			if (pc[i] >= 0.5f)
			{
				e = pc[i] - 1.f;
				for (int c = 0; c < channels; c++)
				{
					d[c][count] = s[c][i];
				}
				count++;
				if (count == sample_num) return sample_num;
			}
			else
			{
				e = pc[i];
			}
		}

		//last random padding
		for (int i = count; i < sample_num; i++)
		{
			const int idx = rng.uniform(0, count);
			for (int c = 0; c < channels; c++)
			{
				d[c][i] = d[c][idx];
			}
		}

		return count;
	}

	template<int channels>
	int TexturenessSampling::remapFloydSteinbergDitherSamplesWeight(const int sample_num, const float scale, const float rangemax, Mat& prob, const vector<cv::Mat>& guide, cv::Mat& samples, cv::Mat& weightsamples, Rect roi)
	{
		const float smax = scale / rangemax;
		//const int channels = (int)guide.size();
		if (samples.size() != Size(sample_num, channels)) samples.create(Size(sample_num, channels), CV_32F);
		if (weightsamples.size() != Size(sample_num,1)) weightsamples.create(Size(sample_num, 1), CV_32F);

		AutoBuffer<const float*> s(channels);
		AutoBuffer<float*> d(channels);

		for (int c = 0; c < channels; c++)
		{
			d[c] = samples.ptr<float>(c);
		}
		float* dw = weightsamples.ptr<float>();

		const int coeff1 = 1;
		const int coeff3 = 3;
		const int coeff5 = 5;
		const int coeff7 = 7;

		float total = 1.f / (coeff1 + coeff3 + coeff5 + coeff7);
		const float coeff7_16 = coeff7 * total;
		const float coeff5_16 = coeff5 * total;
		const float coeff3_16 = coeff3 * total;
		const float coeff1_16 = coeff1 * total;

		int count = 0;
		for (int j = 0; j < prob.rows - 1; j++)
		{
			float e;//error
			const float* wi = weight.ptr<float>(j);
			float* pc = prob.ptr<float>(j);
			float* pn = prob.ptr<float>(j + 1);
			for (int c = 0; c < channels; c++)
			{
				s[c] = guide[c].ptr<float>(j + roi.y, roi.x);
			}

			if (j % 2 == 1)
			{
				int i = 0;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						dw[count] = wi[i];
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i + 1] += e * coeff7_16;
					pn[i + 0] += e * coeff5_16;
					pn[i + 1] += e * coeff1_16;
				}
				for (i = 1; i < prob.cols - 1; i++)
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						dw[count] = wi[i];
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i + 1] += e * coeff7_16;
					pn[i - 1] += e * coeff3_16;
					pn[i + 0] += e * coeff5_16;
					pn[i + 1] += e * coeff1_16;
				}
				i = prob.cols - 1;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						dw[count] = wi[i];
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pn[i - 1] += e * 0.5f;
					pn[i + 0] += e * 0.5f;
				}
			}
			else
			{
				int i = prob.cols - 1;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						dw[count] = wi[i];
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i - 1] += e * coeff7_16;
					pn[i + 0] += e * coeff5_16;
					pn[i - 1] += e * coeff1_16;
				}
				for (i = prob.cols - 2; i > 0; i--)
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						dw[count] = wi[i];
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pc[i - 1] += e * coeff7_16;
					pn[i + 1] += e * coeff3_16;
					pn[i + 0] += e * coeff5_16;
					pn[i - 1] += e * coeff1_16;
				}
				i = 0;
				{
					if (pc[i] >= 0.5f)
					{
						e = pc[i] - 1.f;
						for (int c = 0; c < channels; c++)
						{
							d[c][count] = s[c][i];
						}
						dw[count] = wi[i];
						count++;
						if (count == sample_num) return sample_num;
					}
					else
					{
						e = pc[i];
					}
					pn[i + 1] += e * 0.5f;
					pn[i + 0] += e * 0.5f;
				}
			}
		}
		// bottom y loop
		{
			const float* wi = weight.ptr<float>(prob.rows - 1);
			float* pc = prob.ptr<float>(prob.rows - 1);
			float e;//error
			for (int i = 0; i < prob.cols - 1; i++)
			{
				if (pc[i] >= 0.5f)
				{
					e = pc[i] - 1.f;
					for (int c = 0; c < channels; c++)
					{
						d[c][count] = s[c][i];
					}
					dw[count] = wi[i];
					count++;
					if (count == sample_num) return sample_num;
				}
				else
				{
					e = pc[i];
				}
				pc[i + 1] += e;
			}

			int i = prob.cols - 1;
			if (pc[i] >= 0.5f)
			{
				e = pc[i] - 1.f;
				for (int c = 0; c < channels; c++)
				{
					d[c][count] = s[c][i];
				}
				dw[count] = wi[i];
				count++;
				if (count == sample_num) return sample_num;
			}
			else
			{
				e = pc[i];
			}
		}

		//last random padding
		for (int i = count; i < sample_num; i++)
		{
			const int idx = rng.uniform(0, count);
			for (int c = 0; c < channels; c++)
			{
				d[c][i] = d[c][idx];
			}
			dw[i] = dw[idx];
		}

		return count;
	}

	int TexturenessSampling::dither(Mat& importance, Mat& dest, const int ditheringMethod)
	{
		int sample_num = 0;
		if (ditheringMethod >= 0)
		{
			sample_num = ditherDestruction(importance, dest, ditheringMethod, MEANDERING);
		}
		else
		{
			sample_num = randDither(importance, dest);
		}
		return sample_num;
	}

	TexturenessSampling::TexturenessSampling()
	{
		;
	}
	TexturenessSampling::TexturenessSampling(const float edgeDiffusionSigma) :
		edgeDiffusionSigma(edgeDiffusionSigma)
	{
		;
	}

	void TexturenessSampling::setEdgeDiffusionSigma(const float sigma)
	{
		this->edgeDiffusionSigma = sigma;
	}

	void TexturenessSampling::setUseWeight(const bool flag)
	{
		this->isUseWeight = flag;
	}
	
	void TexturenessSampling::generateDoG(const vector<Mat>& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const float sigma)
	{
		//generate(src[0], dest, sample_num, sampling_ratio, ditheringMethod); 
		//cp::imshowScale("src[0]", src[0]); imshow("importance", importance); imshow("mask", dest); waitKey();
		//return;

		if (isUseAverage)
		{
			cp::cvtColorAverageGray(src, gray, true);
			generateDoG(gray, dest, sample_num, sampling_ratio, ditheringMethod, sigma);
		}
		else
		{
			if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src[0].size()) dest.create(src[0].size(), CV_8UC1);

			importance.create(src[0].rows, src[0].cols, CV_32F);//importance map (n pixels)
			float maxval = computeTexturenessDoG(src, importance, sigma);
			computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
			sample_num = dither(importance, dest, ditheringMethod);
			//cout << sample_num << endl;
			//cp::imshowScale("src[0]", src[0]); cp::imshowScale("importance", importance, 255); imshow("mask", dest); waitKey();
		}
	}

	void TexturenessSampling::generateDoG(const Mat& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const float sigma)
	{
		//CV_Assert(src.depth() == CV_32F);	
		if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src.size())dest.create(src.size(), CV_8UC1);

		importance.create(src.rows, src.cols, CV_32F);//importance map (n pixels)
		float maxval = computeTexturenessDoG(src, importance, sigma);
		float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
		remap(importance, scale, maxval);
		sample_num = dither(importance, dest, ditheringMethod);
	}

	void TexturenessSampling::generateGradientMax(const vector<Mat>& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage)
	{
		//generate(src[0], dest, sample_num, sampling_ratio, ditheringMethod); 
		//cp::imshowScale("src[0]", src[0]); imshow("importance", importance); imshow("mask", dest); waitKey();
		//return;

		if (isUseAverage)
		{
			cout << "generateGradientMax use average" << endl;
			cp::cvtColorAverageGray(src, gray, true);
			generateGradientMax(gray, dest, sample_num, sampling_ratio, ditheringMethod);
		}
		else
		{
			if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src[0].size()) dest.create(src[0].size(), CV_8UC1);
			importance.create(src[0].rows, src[0].cols, CV_32F);//importance map (n pixels)
			Rect roi = Rect(0, 0, src[0].cols, src[0].rows);
			const float maxval = computeTexturenessGradientMax(src, importance, roi);
			computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
			sample_num = dither(importance, dest, ditheringMethod);
		}
	}
	void TexturenessSampling::generateGradientMax(const Mat& src, Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod)
	{
		//CV_Assert(src.depth() == CV_32F);	
		if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src.size())dest.create(src.size(), CV_8UC1);

		importance.create(src.rows, src.cols, CV_32F);//importance map (n pixels)
		Rect roi = Rect(0, 0, src.cols, src.rows);
		const float maxval = computeTexturenessGradientMax(src, importance, roi);
		computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
		sample_num = dither(importance, dest, ditheringMethod);
	}

	inline int scaling(const float val, int scale)
	{
		return saturate_cast<int>(val / scale);
	}
	float colorNormalize(const vector<Mat>& src, Mat& dest, const Rect roi)
	{
		int scale = 4;
		int channels = src.size();
		AutoBuffer<int> step(channels);
		step[0] = 1;
		for (int c = 1; c < channels; c++)
		{
			step[c] = step[c - 1] * (256 / scale);
		}
		std::unordered_map<int, int> mp;
		//int countmax = 0;
		for (int j = roi.y; j < roi.y + roi.height; j++)
		{
			AutoBuffer<const float*> sc(channels);
			for (int c = 0; c < channels; c++)
			{
				sc[c] = src[c].ptr<float>(j);
			}

			for (int i = roi.x; i < roi.x + roi.width; i++)
			{
				int idx = 0;
				for (int c = 0; c < channels; c++)
				{
					idx += step[c] * scaling(sc[c][i], scale);
				}
				mp[idx]++;
				//countmax = max(mp[idx], countmax);
			}
		}

		float maxval = 0.f;
		for (int j = roi.y; j < roi.y + roi.height; j++)
		{
			AutoBuffer<const float*> sc(channels);
			for (int c = 0; c < channels; c++)
			{
				sc[c] = src[c].ptr<float>(j);
			}
			float* d = dest.ptr<float>(j - roi.y);
			for (int i = roi.x; i < roi.x + roi.width; i++)
			{
				int idx = 0;
				for (int c = 0; c < channels; c++)
				{
					idx += step[c] * scaling(sc[c][i], scale);
				}
				const float v = 1.f / mp[idx];
				d[i - roi.x] = v;
				maxval = max(maxval, v);
			}
		}

		return maxval;
	}

	//return maxval;
	template<int channels>
	float TexturenessSampling::gradientMaxDiffuse3x3(const vector<Mat>& src, Mat& dest, const bool isAccumurate, const float sigma, const Rect roi)
	{
		const Size dsize = Size(roi.width, roi.height);
		if (dest.empty()) dest.create(dsize, CV_32F);

		vector<Mat> linebuffer(3);
		for (int l = 0; l < 3; l++)
		{
			linebuffer[l].create(Size(dsize.width + 2, 1), CV_32F);
			linebuffer[l].at<float>(0, 0) = 0.f;
			linebuffer[l].at<float>(0, 1) = 0.f;
			linebuffer[l].at<float>(0, linebuffer[l].cols - 2) = 0.f;
			linebuffer[l].at<float>(0, linebuffer[l].cols - 1) = 0.f;
		}

		const float w0 = exp(-1.f / (2.f * sigma * sigma));
		const float normal = 2.f * w0 + 1.f;
		const __m256 g0 = _mm256_set1_ps(w0 / normal);
		const __m256 g1 = _mm256_set1_ps(1.f / normal);

		const int sh = src[0].rows;
		__m256 mmax = _mm256_setzero_ps();
		for (int j = roi.y; j < roi.y + roi.height; j++)
		{
			AutoBuffer<const float*> sc(channels);
			AutoBuffer<const float*> sm(channels);
			AutoBuffer<const float*> sp(channels);
			for (int c = 0; c < channels; c++)
			{
				sc[c] = src[c].ptr<float>(j);
				sm[c] = src[c].ptr<float>(max(j - 1, 0));
				sp[c] = src[c].ptr<float>(min(j + 1, sh - 1));
			}

			float* b = linebuffer[j % 3].ptr<float>(0, 1);
			for (int i = roi.x; i < roi.x + roi.width; i += 8)
			{
				__m256 macc = _mm256_setzero_ps();
				for (int c = 0; c < channels; c++)
				{
					const __m256 mscp = _mm256_loadu_ps(sc[c] + i - 1);
					const __m256 mscc = _mm256_loadu_ps(sc[c] + i - 0);
					const __m256 mscm = _mm256_loadu_ps(sc[c] + i + 1);
					const __m256 msp = _mm256_loadu_ps(sp[c] + i + 0);
					const __m256 msm = _mm256_loadu_ps(sm[c] + i + 0);

					macc = _mm256_add_ps(macc, _mm256_max_ps(
						_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
						_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
					));
				}
				_mm256_storeu_ps(b + i - roi.x, macc);
			}
			//replicate
			b[-1] = b[0];
			b[dsize.width] = b[dsize.width - 1];

			float* d = dest.ptr<float>(j - roi.y);
			for (int i = 0; i < roi.width; i += 8)
			{
				__m256 blur = _mm256_mul_ps(g0, _mm256_loadu_ps(b + i - 1));
				blur = _mm256_fmadd_ps(g1, _mm256_loadu_ps(b + i + 0), blur);
				blur = _mm256_fmadd_ps(g0, _mm256_loadu_ps(b + i + 1), blur);
				_mm256_storeu_ps(d + i, blur);
				mmax = _mm256_max_ps(mmax, blur);
			}
		}

		return _mm256_reducemax_ps(mmax);
	}

	float TexturenessSampling::computeTexturenessGradientMax(const vector<Mat>& src, Mat& dest, const Rect roi)
	{
		float maxval = 0.f;
		if (src.size() == 1) maxval = gradientMaxDiffuse3x3<1>(src, dest, false, edgeDiffusionSigma, roi);
		if (src.size() == 2) maxval = gradientMaxDiffuse3x3<2>(src, dest, false, edgeDiffusionSigma, roi);
		if (src.size() == 3) maxval = gradientMaxDiffuse3x3<3>(src, dest, false, edgeDiffusionSigma, roi);
		//gradientMaxDiffuse3x3(src[0], dest, false, edgeDiffusionSigma);
		/*for (int c = 0; c < src.size(); c++)
		{
			gradientMaxDiffuse3x3(src[c], dest, c == 0 ? false : true, edgeDiffusionSigma);
		}*/
		/*
		for (int c = 0; c < src.size(); c++)
		{
			gradientMax(src[c], dest, c == 0 ? false : true);
		}

		if (edgeDiffusionSigma != 0)
		{
			Size ksize = Size(3, 3);
			GaussianBlur(dest, dest, ksize, edgeDiffusionSigma);
		}
		*/
		//mul
		return maxval;
		//multiply(dest, 1.f / maxval, dest);
		//normalize_max1(dest);
		//normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);
	}
	float TexturenessSampling::computeTexturenessGradientMax(const Mat& src, Mat& dest, const Rect roi)
	{
		vector<Mat> a(1);
		a[0] = src;
		return computeTexturenessGradientMax(a, dest, roi);
	}


	int TexturenessSampling::generateUniformSamples(const vector<Mat>& image, Mat& samples, const float sampling_ratio, const int ditheringMethod, const Rect roi)
	{
		int sample_num = 0;
		const Size size = Size(roi.width, roi.height);
		const int idealsamples = int(roi.width * roi.height * (sampling_ratio));

		const float p = float(idealsamples) / size.area();
		if (importance.size() != size) importance.create(size, CV_32F);//importance map (n pixels)
		importance.setTo(p);

		if (ditheringMethod < 0)
		{
			if (image.size() == 1) sample_num = remapRandomDitherSamples<1>(idealsamples, 1.f, p, importance, image, samples, roi);
			if (image.size() == 2) sample_num = remapRandomDitherSamples<2>(idealsamples, 1.f, p, importance, image, samples, roi);
			if (image.size() == 3) sample_num = remapRandomDitherSamples<3>(idealsamples, 1.f, p, importance, image, samples, roi);
		}
		else
		{
			if (image.size() == 1) sample_num = remapFloydSteinbergDitherSamples<1>(idealsamples, 1.f, p, importance, image, samples, roi);
			if (image.size() == 2) sample_num = remapFloydSteinbergDitherSamples<2>(idealsamples, 1.f, p, importance, image, samples, roi);
			if (image.size() == 3) sample_num = remapFloydSteinbergDitherSamples<3>(idealsamples, 1.f, p, importance, image, samples, roi);
		}

		return sample_num;
	}

	int TexturenessSampling::generateColorNormalizedSamples(const vector<Mat>& image, Mat& samples, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const Rect roi)
	{
		int sample_num = 0;
		if (isUseAverage)
		{
			cout << "generateGradientMax use average" << endl;
			cp::cvtColorAverageGray(image, gray, true);
			generateGradientMax(gray, samples, sample_num, sampling_ratio, ditheringMethod);
		}
		else
		{
			const Size size = Size(roi.width, roi.height);
			const int idealsamples = int(roi.width * roi.height * (sampling_ratio));
			if (importance.size() != size) importance.create(size, CV_32F);//importance map (n pixels)
			//const float maxval = computeTexturenessGradientMax(image, importance, roi);
			const float maxval = colorNormalize(image, importance, roi);
			//imshowScale("imp", importance, 255); waitKey();
			const float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
			//print_debug(scale);
			if (ditheringMethod < 0)
			{
				if (image.size() == 1) sample_num = remapRandomDitherSamples<1>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 2) sample_num = remapRandomDitherSamples<2>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 3) sample_num = remapRandomDitherSamples<3>(idealsamples, scale, maxval, importance, image, samples, roi);
			}
			else
			{
				remap(importance, scale, maxval);
				if (image.size() == 1) sample_num = remapFloydSteinbergDitherSamples<1>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 2) sample_num = remapFloydSteinbergDitherSamples<2>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 3) sample_num = remapFloydSteinbergDitherSamples<3>(idealsamples, scale, maxval, importance, image, samples, roi);
			}
		}
		return sample_num;
	}

	int TexturenessSampling::generateGradientMaxSamples(const vector<Mat>& image, Mat& samples, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const Rect roi)
	{
		int sample_num = 0;
		if (isUseAverage)
		{
			cout << "generateGradientMax use average" << endl;
			cp::cvtColorAverageGray(image, gray, true);
			generateGradientMax(gray, samples, sample_num, sampling_ratio, ditheringMethod);
		}
		else
		{
			const Size size = Size(roi.width, roi.height);
			const int idealsamples = int(roi.width * roi.height * (sampling_ratio));
			if (importance.size() != size) importance.create(size, CV_32F);//importance map (n pixels)
			const float maxval = computeTexturenessGradientMax(image, importance, roi);
			const float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
			if (ditheringMethod < 0)
			{
				if (image.size() == 1) sample_num = remapRandomDitherSamples<1>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 2) sample_num = remapRandomDitherSamples<2>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 3) sample_num = remapRandomDitherSamples<3>(idealsamples, scale, maxval, importance, image, samples, roi);
			}
			else
			{
				remap(importance, scale, maxval);
				if (image.size() == 1) sample_num = remapFloydSteinbergDitherSamples<1>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 2) sample_num = remapFloydSteinbergDitherSamples<2>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 3) sample_num = remapFloydSteinbergDitherSamples<3>(idealsamples, scale, maxval, importance, image, samples, roi);
			}
		}
		return sample_num;
	}

	int TexturenessSampling::generateGradientMaxSamplesWeight(const vector<Mat>& image, Mat& samples, Mat& samplesweight, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const Rect roi)
	{
		int sample_num = 0;
		if (isUseAverage)
		{
			cout << "generateGradientMax use average" << endl;
			cp::cvtColorAverageGray(image, gray, true);
			generateGradientMax(gray, samples, sample_num, sampling_ratio, ditheringMethod);
		}
		else
		{
			const Size size = Size(roi.width, roi.height);
			const int idealsamples = int(roi.width * roi.height * (sampling_ratio));
			if (importance.size() != size) importance.create(size, CV_32F);//importance map (n pixels)
			const float maxval = computeTexturenessGradientMax(image, importance, roi);
			const float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
			if (ditheringMethod < 0)
			{
				if (image.size() == 1) sample_num = remapRandomDitherSamples<1>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 2) sample_num = remapRandomDitherSamples<2>(idealsamples, scale, maxval, importance, image, samples, roi);
				if (image.size() == 3) sample_num = remapRandomDitherSamples<3>(idealsamples, scale, maxval, importance, image, samples, roi);
			}
			else
			{			
				remap(importance, scale, maxval);
				importance.copyTo(weight);
				if (image.size() == 1) sample_num = remapFloydSteinbergDitherSamplesWeight<1>(idealsamples, scale, maxval, importance, image, samples, samplesweight, roi);
				if (image.size() == 2) sample_num = remapFloydSteinbergDitherSamplesWeight<2>(idealsamples, scale, maxval, importance, image, samples, samplesweight, roi);
				if (image.size() == 3) sample_num = remapFloydSteinbergDitherSamplesWeight<3>(idealsamples, scale, maxval, importance, image, samples, samplesweight, roi);
			}
		}
		return sample_num;
	}

	int TexturenessSampling::generateDoGSamples(const vector<Mat>& image, Mat& samples, const float sigma, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const Rect roi)
	{
		const Size size = Size(roi.width, roi.height);
		if (importance.size() != size) importance.create(size, CV_32F);//importance map (n pixels)

		for (int c = 0; c < image.size(); c++)
		{
			const int r = (int)ceil(sigma * 1.5);
			const int d = 2 * r + 1;

			Mat im = image[c](roi).clone();
			GaussianBlur(im, src_32f, Size(d, d), sigma);
			//Laplacian(src[c], src_32f, CV_32F, d);
			//gf->filter(src[c], src_32f, ss1, 1);
			if (c == 0)
			{
				absdiff(src_32f, im, importance);
			}
			else
			{
				absdiff(src_32f, im, src_32f);
				add(src_32f, importance, importance);
			}
		}

		if (edgeDiffusionSigma != 0.f)
		{
			Size ksize = Size(3, 3);
			GaussianBlur(importance, importance, ksize, edgeDiffusionSigma);
		}

		const float maxval = getmax(importance);
		const float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);


		const int idealsamples = int(roi.width * roi.height * (sampling_ratio));
		int sample_num = 0;
		if (ditheringMethod < 0)
		{
			if (image.size() == 1) sample_num = remapRandomDitherSamples<1>(idealsamples, scale, maxval, importance, image, samples, roi);
			if (image.size() == 2) sample_num = remapRandomDitherSamples<2>(idealsamples, scale, maxval, importance, image, samples, roi);
			if (image.size() == 3) sample_num = remapRandomDitherSamples<3>(idealsamples, scale, maxval, importance, image, samples, roi);
		}
		else
		{
			remap(importance, scale, maxval);
			if (isUseWeight) importance.copyTo(weight);
			if (image.size() == 1) sample_num = remapFloydSteinbergDitherSamples<1>(idealsamples, scale, maxval, importance, image, samples, roi);
			if (image.size() == 2) sample_num = remapFloydSteinbergDitherSamples<2>(idealsamples, scale, maxval, importance, image, samples, roi);
			if (image.size() == 3) sample_num = remapFloydSteinbergDitherSamples<3>(idealsamples, scale, maxval, importance, image, samples, roi);
		}
		/*
		const float maxval = computeTexturenessGradientMax(image, importance, roi);
		float scale = computeScaleHistogram(importance, sampling_ratio, 100, 0.f, maxval);
		remap(importance, scale, maxval);
		sampleMask.create(importance.size(), CV_8U);
		int sample_num = dither(importance, sampleMask, ditheringMethod);
		sample_num = get_simd_floor(sample_num, 8);
		mask2samples(sample_num, sampleMask, image, samples, roi);
		*/
		return sample_num;
	}
}