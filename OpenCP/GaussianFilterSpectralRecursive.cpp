#include "GaussianFilterSpectralRecursive.hpp"
using namespace std;
using namespace cv;

#define ___MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3, out4, out5, out6, out7, __in0, __in1, __in2, __in3, __in4, __in5, __in6, __in7, __out0, __out1, __out2, __out3, __out4, __out5, __out6, __out7, __tmp0, __tmp1, __tmp2, __tmp3, __tmp4, __tmp5, __tmp6, __tmp7, __tmpp0, __tmpp1, __tmpp2, __tmpp3, __tmpp4, __tmpp5, __tmpp6, __tmpp7) \
  do { \
    __m256 __in0 = (in0), __in1 = (in1), __in2 = (in2), __in3 = (in3), __in4 = (in4), __in5 = (in5), __in6 = (in6), __in7 = (in7); \
    __m256 __tmp0, __tmp1, __tmp2, __tmp3, __tmp4, __tmp5, __tmp6, __tmp7; \
    __m256 __tmpp0, __tmpp1, __tmpp2, __tmpp3, __tmpp4, __tmpp5, __tmpp6, __tmpp7; \
    __m256 __out0, __out1, __out2, __out3, __out4, __out5, __out6, __out7; \
    __tmp0  = _mm256_unpacklo_ps(__in0, __in1); \
    __tmp1  = _mm256_unpackhi_ps(__in0, __in1); \
    __tmp2  = _mm256_unpacklo_ps(__in2, __in3); \
    __tmp3  = _mm256_unpackhi_ps(__in2, __in3); \
    __tmp4  = _mm256_unpacklo_ps(__in4, __in5); \
    __tmp5  = _mm256_unpackhi_ps(__in4, __in5); \
    __tmp6  = _mm256_unpacklo_ps(__in6, __in7); \
    __tmp7  = _mm256_unpackhi_ps(__in6, __in7); \
    __tmpp0 = _mm256_shuffle_ps(__tmp0, __tmp2, 0x44); \
    __tmpp1 = _mm256_shuffle_ps(__tmp0, __tmp2, 0xEE); \
    __tmpp2 = _mm256_shuffle_ps(__tmp1, __tmp3, 0x44); \
    __tmpp3 = _mm256_shuffle_ps(__tmp1, __tmp3, 0xEE); \
    __tmpp4 = _mm256_shuffle_ps(__tmp4, __tmp6, 0x44); \
    __tmpp5 = _mm256_shuffle_ps(__tmp4, __tmp6, 0xEE); \
    __tmpp6 = _mm256_shuffle_ps(__tmp5, __tmp7, 0x44); \
    __tmpp7 = _mm256_shuffle_ps(__tmp5, __tmp7, 0xEE); \
    __out0  = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x20); \
    __out1  = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x20); \
    __out2  = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x20); \
    __out3  = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x20); \
    __out4  = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x31); \
    __out5  = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x31); \
    __out6  = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x31); \
    __out7  = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x31); \
    (out0)  = __out0, (out1) = __out1, (out2) = __out2, (out3) = __out3, (out4) = __out4, (out5) = __out5, (out6) = __out6, (out7) = __out7; \
          } while (0)
#define _MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7) \
      ___MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4, in5, in6, in7, \
          __in0##__LINE__, __in1##__LINE__, __in2##__LINE__, __in3##__LINE__, __in4##__LINE__, __in5##__LINE__, __in6##__LINE__, __in7##__LINE__, \
          __out0##__LINE__, __out1##__LINE__, __out2##__LINE__, __out3##__LINE__, __out4##__LINE__, __out5##__LINE__, __out6##__LINE__, __out7##__LINE__, \
          __tmp0##__LINE__, __tmp1##__LINE__, __tmp2##__LINE__, __tmp3##__LINE__, __tmp4##__LINE__, __tmp5##__LINE__, __tmp6##__LINE__, __tmp7##__LINE__, \
          __tmpp0##__LINE__, __tmpp1##__LINE__, __tmpp2##__LINE__, __tmpp3##__LINE__, __tmpp4##__LINE__, __tmpp5##__LINE__, __tmpp6##__LINE__, __tmpp7##__LINE__)

#define ___MM256_TRANSPOSE4_PD(in0, in1, in2, in3, out0, out1, out2, out3, __in0, __in1, __in2, __in3, __out0, __out1, __out2, __out3, __tmp0, __tmp1, __tmp2, __tmp3) \
  do { \
    __m256d __in0 = (in0), __in1 = (in1), __in2 = (in2), __in3 = (in3); \
    __m256d __tmp0, __tmp1, __tmp2, __tmp3; \
    __m256d __out0, __out1, __out2, __out3; \
    __tmp0 = _mm256_shuffle_pd(__in0, __in1, 0x0); \
    __tmp1 = _mm256_shuffle_pd(__in0, __in1, 0xf); \
    __tmp2 = _mm256_shuffle_pd(__in2, __in3, 0x0); \
    __tmp3 = _mm256_shuffle_pd(__in2, __in3, 0xf); \
    __out0 = _mm256_permute2f128_pd(__tmp0, __tmp2, 0x20); \
    __out1 = _mm256_permute2f128_pd(__tmp1, __tmp3, 0x20); \
    __out2 = _mm256_permute2f128_pd(__tmp0, __tmp2, 0x31); \
    __out3 = _mm256_permute2f128_pd(__tmp1, __tmp3, 0x31); \
    (out0) = __out0, (out1) = __out1, (out2) = __out2, (out3) = __out3; \
          } while (0)
#define _MM256_TRANSPOSE4_PD(in0, in1, in2, in3) \
      ___MM256_TRANSPOSE4_PD(in0, in1, in2, in3, in0, in1, in2, in3, \
          __in0##__LINE__, __in1##__LINE__, __in2##__LINE__, __in3##__LINE__, \
          __out0##__LINE__, __out1##__LINE__, __out2##__LINE__, __out3##__LINE__, \
          __tmp0##__LINE__, __tmp1##__LINE__, __tmp2##__LINE__, __tmp3##__LINE__)

namespace cp
{
	//spectral recursive Gaussian Filter
	//K. Sugimoto and S. Kamata: "Fast Gaussian filter with second-order shift property of DCT-5", Proc. IEEE Int. Conf. on Image Process. (ICIP2013), pp.514-518 (Sep. 2013).
	namespace spectral_recursive_filter
	{
#define USE_BORDER_REPLICATE 1 //BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
		//#define USE_BORDER_REFLECT 1 //BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
		//#define USE_BORDER_REFLECT_101 1 //BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`

		//extrapolation functions
		//(atE() and atS() require variables w and h in their scope respectively)
#ifdef USE_BORDER_REPLICATE 
#define atW(x) (std::max(x,0))
#define atN(y) (std::max(y,0))
#define atE(x) (std::min(x,w-1))
#define atS(y) (std::min(y,h-1))
#elif USE_BORDER_REFLECT 
#define atW(x) ( x < 0 ? std::abs(x+1) : std::abs(x))
#define atN(y) ( y < 0 ? std::abs(y+1) : std::abs(y))
#define atE(x) ( x < w ? w-1-std::abs(w-1-(x)) : w-std::abs(w-1-(x)))
#define atS(y) ( y < h ? h-1-std::abs(h-1-(y)) : h-std::abs(h-1-(y)))
//#define atE(x) ( x < w ? w-1-std::abs(w-1-(x)) : w-2-std::abs(w-1-(x)))
//#define atS(y) ( y < h ? h-1-std::abs(h-1-(y)) : h-2-std::abs(h-1-(y)))
#elif USE_BORDER_REFLECT_101
#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))
#endif
		class gauss
		{
		private:
			static const int K = 2; //order of approximation (do not change!)

		private:
			double sx, sy; //scale of Gaussian
			int rx, ry; //kernel radius
			std::vector<double> spectX, spectY;
			std::vector<double> tableX, tableY; //look-up tables

		public:
			gauss(double sx, double sy) :sx(sx), sy(sy)
			{
				if (sx < 0.0 || sy < 0.0)
					throw std::invalid_argument("\'sx\' and \'sy\' should be nonnegative!");

				rx = estimate_radius(sx);
				ry = estimate_radius(sy);
				spectX = gen_spectrum(sx, rx);
				spectY = gen_spectrum(sy, ry);
				tableX = build_lookup_table(rx, spectX);
				tableY = build_lookup_table(ry, spectY);
			}
			~gauss() {}

		private:
			static inline double phase(int r)
			{
				return 2.0*CV_PI / (r + 1 + r); //DCT/DST-5
			}
			static inline int estimate_radius(double s)
			{
				//return (s<4.0) ? int(3.3333*s-0.3333+0.5) : int(3.4113*s-0.6452+0.5); //K==3
				return (s < 4.0) ? int(3.0000*s - 0.2000 + 0.5) : int(3.0000*s + 0.5); //K==2
			}
			static inline std::vector<double> gen_spectrum(double s, int r)
			{
				const double phi = phase(r);
				std::vector<double> spect(K);
				for (int k = 1; k <= K; k++)
					spect[k - 1] = 2.0*exp(-0.5*s*s*phi*phi*k*k);
				return spect;
			}
			static inline std::vector<double> build_lookup_table(int r, std::vector<double>& spect)
			{
				assert(spect.size() == K);
				const double phi = phase(r);
				std::vector<double> table(K*(1 + r));
				for (int u = 0; u <= r; ++u)
					for (int k = 1; k <= K; ++k)
						table[K*u + k - 1] = cos(k*phi*u)*spect[k - 1];
				return table;
			}

			template <typename T>
			inline void filter_h(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}
			template <typename T>
			inline void filter_v(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}
			template <typename T>
			inline void filter_sse_h(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}
			template <typename T>
			inline void filter_sse_v(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}
			template <typename T>
			inline void filter_avx_h(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}
			template <typename T>
			inline void filter_avx_v(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}
			template <typename T>
			inline void filter_avx_h_OMP(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}
			template <typename T>
			inline void filter_avx_v_OMP(int w, int h, T* src, T* dst)
			{
				throw std::invalid_argument("Unsupported element type!");
			}

		public:

			template <typename T>
			void filter(int w, int h, T* src, T* dst)
			{
				if (w <= 4.0*sx || h <= 4.0*sy)
					throw std::invalid_argument("\'sx\' and \'sy\' should be less than about w/4 or h/4!");

				//filtering is skipped if s==0.0
				if (sx == 0.0 && sy == 0.0)
					return;
				else if (sx == 0.0)
					filter_v<T>(w, h, src, dst);
				else if (sy == 0.0)
					filter_h<T>(w, h, src, dst);
				else
				{
					filter_v<T>(w, h, src, dst);
					filter_h<T>(w, h, dst, dst); //only filter_h() allows src==dst.
				}
			}

			template <typename T>
			void filter_sse(int w, int h, T* src, T* dst)
			{
				if (w <= 4.0*sx || h <= 4.0*sy)
					throw std::invalid_argument("\'sx\' and \'sy\' should be less than about w/4 or h/4!");

				//filtering is skipped if s==0.0
				if (sx == 0.0 && sy == 0.0)
					return;
				else if (sx == 0.0)
					filter_sse_v<T>(w, h, src, dst);
				else if (sy == 0.0)
					filter_sse_h<T>(w, h, src, dst);
				else
				{
					filter_sse_v<T>(w, h, src, dst);
					filter_sse_h<T>(w, h, dst, dst); //only filter_h() allows src==dst.
				}
			}
			
			template <typename T>
			void filter_avx(int w, int h, T* src, T* dst)
			{
				if (w <= 4.0*sx || h <= 4.0*sy)
					throw std::invalid_argument("\'sx\' and \'sy\' should be less than about w/4 or h/4!");

				//filtering is skipped if s==0.0
				if (sx == 0.0 && sy == 0.0)
					return;
				else if (sx == 0.0)
					filter_avx_v<T>(w, h, src, dst);
				else if (sy == 0.0)
					filter_avx_h<T>(w, h, src, dst);
				else
				{
					filter_avx_v<T>(w, h, src, dst);
					filter_avx_h<T>(w, h, dst, dst); //only filter_h() allows src==dst.
				}
			}

			template <typename T>
			void filter_avxOMP(int w, int h, T* src, T* dst)
			{
				if (w <= 4.0*sx || h <= 4.0*sy)
					throw std::invalid_argument("\'sx\' and \'sy\' should be less than about w/4 or h/4!");

				//filtering is skipped if s==0.0
				if (sx == 0.0 && sy == 0.0)
					return;
				else if (sx == 0.0)
					filter_avx_v_OMP<T>(w, h, src, dst);
				else if (sy == 0.0)
					filter_avx_h_OMP<T>(w, h, src, dst);
				else
				{
					filter_avx_v_OMP<T>(w, h, src, dst);
					filter_avx_h_OMP<T>(w, h, dst, dst); //only filter_h() allows src==dst.
				}
			}

			void filter(const cv::Mat& src, cv::Mat& dst)
			{
				//checking the format of input/output images
				if (src.size() != dst.size())
					throw std::invalid_argument("\'src\' and \'dst\' should have the same size!");
				if (src.type() != dst.type())
					throw std::invalid_argument("\'src\' and \'dst\' should have the same element type!");
				if (src.channels() != 1 || dst.channels() != 1)
					throw std::invalid_argument("Multi-channel images are unsupported!");
				if (src.isSubmatrix() || dst.isSubmatrix())
					throw std::invalid_argument("Subimages are unsupported!");

				switch (src.type())
				{
				case CV_32FC1:
					filter_avx<float>(src.cols, src.rows, reinterpret_cast<float*>(src.data), reinterpret_cast<float*>(dst.data));
					break;
				case CV_64FC1:
					filter_avx<double>(src.cols, src.rows, reinterpret_cast<double*>(src.data), reinterpret_cast<double*>(dst.data));
					break;
				default:
					throw std::invalid_argument("Unsupported element type!");
					break;
				}
			}
		};

		//*************************************************************************************************

		template<>
		inline void gauss::filter_h<float>(int w, int h, float* src, float* dst)
		{
			const int r = rx;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableX.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableX[t]);

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectX[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectX[1]), cfR2 = table[K*r + 1];

			float sum, a1, a2, b1, b2;
			float dA, dB, delta;
			std::vector<float> buf(w); //to allow for src==dst
			for (int y = 0; y < h; ++y)
			{
				std::copy(&src[w*y], &src[w*y + w], buf.begin());

				sum = buf[0];
				a1 = buf[0] * table[0]; b1 = buf[1] * table[0];
				a2 = buf[0] * table[1]; b2 = buf[1] * table[1];
				for (int u = 1; u <= r; ++u)
				{
					const float sumA = buf[atW(0 - u)] + buf[0 + u];
					const float sumB = buf[atW(1 - u)] + buf[1 + u];
					sum += sumA;
					a1 += sumA*table[K*u + 0]; b1 += sumB*table[K*u + 0];
					a2 += sumA*table[K*u + 1]; b2 += sumB*table[K*u + 1];
				}

				//the first pixel (x=0)
				float* q = &dst[w*y];
				q[0] = norm*(sum + a1 + a2);
				dA = buf[atE(0 + r + 1)] - buf[atW(0 - r)];
				sum += dA;

				//the other pixels (0<x<w)
				int x = 1;
				while (true) //four-length ring buffers
				{
					q[x] = norm*(sum + b1 + b2);
					dB = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dA - dB;
					sum += dB;
					a1 += -cf11*b1 + cfR1*delta;
					a2 += -cf12*b2 + cfR2*delta;
					x++; if (w <= x) break;

					q[x] = norm*(sum - a1 - a2);
					dA = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dB - dA;
					sum += dA;
					b1 += +cf11*a1 + cfR1*delta;
					b2 += +cf12*a2 + cfR2*delta;
					x++; if (w <= x) break;

					q[x] = norm*(sum - b1 - b2);
					dB = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dA - dB;
					sum += dB;
					a1 += -cf11*b1 - cfR1*delta;
					a2 += -cf12*b2 - cfR2*delta;
					x++; if (w <= x) break;

					q[x] = norm*(sum + a1 + a2);
					dA = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dB - dA;
					sum += dA;
					b1 += +cf11*a1 - cfR1*delta;
					b2 += +cf12*a2 - cfR2*delta;
					x++; if (w <= x) break;
				}
			}
		}

		template<>
		inline void gauss::filter_v<float>(int w, int h, float* src, float* dst)
		{
			const int r = ry;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableY.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableY[t]);

			//work space to keep raster scanning
			std::vector<float> workspace((2 * K + 1)*w);

			//calculating the first and second terms
			for (int x = 0; x < w; ++x)
			{
				float* ws = &workspace[(2 * K + 1)*x];
				ws[0] = src[x];
				ws[1] = src[x + w] * table[0]; ws[2] = src[x] * table[0];
				ws[3] = src[x + w] * table[1]; ws[4] = src[x] * table[1];
			}
			for (int v = 1; v <= r; ++v)
			{
				for (int x = 0; x < w; ++x)
				{
					const float sum0 = src[x + w*atN(0 - v)] + src[x + w*(0 + v)];
					const float sum1 = src[x + w*atN(1 - v)] + src[x + w*(1 + v)];
					float* ws = &workspace[(2 * K + 1)*x];
					ws[0] += sum0;
					ws[1] += sum1*table[K*v + 0]; ws[2] += sum0*table[K*v + 0];
					ws[3] += sum1*table[K*v + 1]; ws[4] += sum0*table[K*v + 1];
				}
			}

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectY[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectY[1]), cfR2 = table[K*r + 1];

			float *q, *p0N, *p0S, *p1N, *p1S;
			for (int y = 0; y < 1; ++y) //the first line (y=0)
			{
				q = &dst[w*y];
				p1N = &src[w*atN(0 - r)]; p1S = &src[w*atS(0 + r + 1)];
				for (int x = 0; x < w; ++x)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					q[x] = norm*(ws[0] + ws[2] + ws[4]);
					ws[0] += p1S[x] - p1N[x];
				}
			}
			for (int y = 1; y < h; ++y) //remaining lines (with two-length ring buffers)
			{
				q = &dst[w*y];
				p0N = &src[w*atN(y - r - 1)]; p0S = &src[w*atS(y + r)];
				p1N = &src[w*atN(y - r)]; p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w; ++x)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					q[x] = norm*(ws[0] + ws[1] + ws[3]);

					const float d0 = p0S[x] - p0N[x];
					const float d1 = p1S[x] - p1N[x];
					const float delta = d1 - d0;

					ws[0] += d1;
					ws[2] = cfR1*delta + cf11*ws[1] - ws[2];
					ws[4] = cfR2*delta + cf12*ws[3] - ws[4];
				}
				y++; if (h <= y) break; //to the next line

				q = &dst[w*y];
				p0N = &src[w*atN(y - r - 1)]; p0S = &src[w*atS(y + r)];
				p1N = &src[w*atN(y - r)]; p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w; ++x)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					q[x] = norm*(ws[0] + ws[2] + ws[4]);

					const float d0 = p0S[x] - p0N[x];
					const float d1 = p1S[x] - p1N[x];
					const float delta = d1 - d0;

					ws[0] += d1;
					ws[1] = cfR1*delta + cf11*ws[2] - ws[1];
					ws[3] = cfR2*delta + cf12*ws[4] - ws[3];
				}
			}
		}

		template<>
		inline void gauss::filter_sse_h<float>(int w, int h, float* src, float* dst)
		{
			const int B = sizeof(__m128) / sizeof(float);
			const int r = rx;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableX.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableX[t]);

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectX[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectX[1]), cfR2 = table[K*r + 1];

			//to allow for src==dst
			std::vector<float> buf(B*w);

			assert(h%B == 0);
			for (int y = 0; y < h / B*B; y += B)
			{
				std::copy(&src[w*y], &src[w*(y + B)], buf.begin());

				__m128 pv0 = _mm_set_ps(buf[w * 3 + 0], buf[w * 2 + 0], buf[w * 1 + 0], buf[w * 0 + 0]);
				__m128 pv1 = _mm_set_ps(buf[w * 3 + 1], buf[w * 2 + 1], buf[w * 1 + 1], buf[w * 0 + 1]);

				__m128 sum = pv0;
				__m128 a1 = _mm_mul_ps(_mm_set1_ps(table[0]), pv0);
				__m128 b1 = _mm_mul_ps(_mm_set1_ps(table[0]), pv1);
				__m128 a2 = _mm_mul_ps(_mm_set1_ps(table[1]), pv0);
				__m128 b2 = _mm_mul_ps(_mm_set1_ps(table[1]), pv1);

				for (int u = 1; u <= r; ++u)
				{
					const float* p0M = &buf[atW(0 - u)];
					const float* p1M = &buf[atW(1 - u)];
					const float* p0P = &buf[(0 + u)];
					const float* p1P = &buf[(1 + u)];
					__m128 pv0M = _mm_set_ps(p0M[w * 3], p0M[w * 2], p0M[w * 1], p0M[w * 0]);
					__m128 pv1M = _mm_set_ps(p1M[w * 3], p1M[w * 2], p1M[w * 1], p1M[w * 0]);
					__m128 pv0P = _mm_set_ps(p0P[w * 3], p0P[w * 2], p0P[w * 1], p0P[w * 0]);
					__m128 pv1P = _mm_set_ps(p1P[w * 3], p1P[w * 2], p1P[w * 1], p1P[w * 0]);
					__m128 sumA = _mm_add_ps(pv0M, pv0P);
					__m128 sumB = _mm_add_ps(pv1M, pv1P);

					sum = _mm_add_ps(sum, sumA);
					a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_set1_ps(table[K*u + 0]), sumA));
					b1 = _mm_add_ps(b1, _mm_mul_ps(_mm_set1_ps(table[K*u + 0]), sumB));
					a2 = _mm_add_ps(a2, _mm_mul_ps(_mm_set1_ps(table[K*u + 1]), sumA));
					b2 = _mm_add_ps(b2, _mm_mul_ps(_mm_set1_ps(table[K*u + 1]), sumB));
				}

				//sliding convolution
				float *pA, *pB;
				__m128 pvA, pvB;
				__m128 dvA, dvB, delta;
				__m128 qv[B];

				//the first four pixels (0<=x<B)
				for (int x = 0; x < B; x += B)
				{
					//the first pixel (x=0)
					qv[0] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(a1, a2)));

					pA = &buf[atW(0 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(0 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm_sub_ps(pvB, pvA);

					sum = _mm_add_ps(sum, dvA);
					//b1=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1),delta),_mm_mul_ps(_mm_set1_ps(cf11),a1)),b1);
					//b2=_mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2),delta),_mm_mul_ps(_mm_set1_ps(cf12),a2)),b2);

					//the second pixel (x=1)
					qv[1] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(b1, b2)));

					pA = &buf[atW(1 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(1 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm_sub_ps(pvB, pvA);
					delta = _mm_sub_ps(dvB, dvA);

					sum = _mm_add_ps(sum, dvB);
					a1 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), b1)), a1);
					a2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(a1, a2)));

					pA = &buf[atW(2 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(2 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm_sub_ps(pvB, pvA);
					delta = _mm_sub_ps(dvA, dvB);

					sum = _mm_add_ps(sum, dvA);
					b1 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), a1)), b1);
					b2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(b1, b2)));

					pA = &buf[atW(3 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(3 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm_sub_ps(pvB, pvA);
					delta = _mm_sub_ps(dvB, dvA);

					sum = _mm_add_ps(sum, dvB);
					a1 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), b1)), a1);
					a2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), b2)), a2);

					//output with transposition
					_MM_TRANSPOSE4_PS(qv[0], qv[1], qv[2], qv[3]);
					_mm_storeu_ps(&dst[w*(y + 0)], qv[0]);
					_mm_storeu_ps(&dst[w*(y + 1)], qv[1]);
					_mm_storeu_ps(&dst[w*(y + 2)], qv[2]);
					_mm_storeu_ps(&dst[w*(y + 3)], qv[3]);
				}

				//the other pixels (B<=x<w)
				for (int x = B; x < w / B*B; x += B) //four-length ring buffers
				{
					//the first pixel (x=0)
					qv[0] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(a1, a2)));

					pA = &buf[atW(x + 0 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 0 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm_sub_ps(pvB, pvA);
					delta = _mm_sub_ps(dvA, dvB);

					sum = _mm_add_ps(sum, dvA);
					b1 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), a1)), b1);
					b2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), a2)), b2);

					//the second pixel (x=1)
					qv[1] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(b1, b2)));

					pA = &buf[atW(x + 1 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 1 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm_sub_ps(pvB, pvA);
					delta = _mm_sub_ps(dvB, dvA);

					sum = _mm_add_ps(sum, dvB);
					a1 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), b1)), a1);
					a2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(a1, a2)));

					pA = &buf[atW(x + 2 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 2 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm_sub_ps(pvB, pvA);
					delta = _mm_sub_ps(dvA, dvB);

					sum = _mm_add_ps(sum, dvA);
					b1 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), a1)), b1);
					b2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(sum, _mm_add_ps(b1, b2)));

					pA = &buf[atW(x + 3 - r)]; pvA = _mm_set_ps(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 3 + r + 1)]; pvB = _mm_set_ps(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm_sub_ps(pvB, pvA);
					delta = _mm_sub_ps(dvB, dvA);

					sum = _mm_add_ps(sum, dvB);
					a1 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), b1)), a1);
					a2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), b2)), a2);

					//output with transposition
					_MM_TRANSPOSE4_PS(qv[0], qv[1], qv[2], qv[3]);
					_mm_storeu_ps(&dst[w*(y + 0) + x], qv[0]);
					_mm_storeu_ps(&dst[w*(y + 1) + x], qv[1]);
					_mm_storeu_ps(&dst[w*(y + 2) + x], qv[2]);
					_mm_storeu_ps(&dst[w*(y + 3) + x], qv[3]);
				}
			}
		}

		template<>
		inline void gauss::filter_sse_v<float>(int w, int h, float* src, float* dst)
		{
			assert(w % sizeof(__m128) == 0);

			const int B = sizeof(__m128) / sizeof(float);
			const int r = ry;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableY.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableY[t]);

			//work space to keep raster scanning
			float* workspace = reinterpret_cast<float*>(_mm_malloc(sizeof(float)*(2 * K + 1)*w, sizeof(__m128)));

			//calculating the first and second terms
			for (int x = 0; x < w / B*B; x += B)
			{
				float* ws = &workspace[(2 * K + 1)*x];
				__m128 p0 = _mm_load_ps(&src[x]);
				__m128 p1 = _mm_load_ps(&src[x + w]);
				_mm_store_ps(&ws[B * 0], p0);
				_mm_store_ps(&ws[B * 1], _mm_mul_ps(p1, _mm_set1_ps(table[0])));
				_mm_store_ps(&ws[B * 2], _mm_mul_ps(p0, _mm_set1_ps(table[0])));
				_mm_store_ps(&ws[B * 3], _mm_mul_ps(p1, _mm_set1_ps(table[1])));
				_mm_store_ps(&ws[B * 4], _mm_mul_ps(p0, _mm_set1_ps(table[1])));
			}
			for (int v = 1; v <= r; ++v)
			{
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					__m128 sum0 = _mm_add_ps(_mm_load_ps(&src[x + w*atN(0 - v)]), _mm_load_ps(&src[x + w*(0 + v)]));
					__m128 sum1 = _mm_add_ps(_mm_load_ps(&src[x + w*atN(1 - v)]), _mm_load_ps(&src[x + w*(1 + v)]));
					_mm_store_ps(&ws[B * 0], _mm_add_ps(_mm_load_ps(&ws[B * 0]), sum0));
					_mm_store_ps(&ws[B * 1], _mm_add_ps(_mm_load_ps(&ws[B * 1]), _mm_mul_ps(sum1, _mm_set1_ps(table[K*v + 0]))));
					_mm_store_ps(&ws[B * 2], _mm_add_ps(_mm_load_ps(&ws[B * 2]), _mm_mul_ps(sum0, _mm_set1_ps(table[K*v + 0]))));
					_mm_store_ps(&ws[B * 3], _mm_add_ps(_mm_load_ps(&ws[B * 3]), _mm_mul_ps(sum1, _mm_set1_ps(table[K*v + 1]))));
					_mm_store_ps(&ws[B * 4], _mm_add_ps(_mm_load_ps(&ws[B * 4]), _mm_mul_ps(sum0, _mm_set1_ps(table[K*v + 1]))));
				}
			}

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectY[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectY[1]), cfR2 = table[K*r + 1];

			//sliding convolution
			for (int y = 0; y < 1; ++y) //the first line (y=0)
			{
				float* q = &dst[w*y];
				const float* p1N = &src[w*atN(y - r)];
				const float* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					const __m128 a0 = _mm_load_ps(&ws[B * 0]);
					const __m128 a2 = _mm_load_ps(&ws[B * 2]);
					const __m128 a4 = _mm_load_ps(&ws[B * 4]);
					_mm_store_ps(&q[x], _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(a0, _mm_add_ps(a2, a4))));

					const __m128 d = _mm_sub_ps(_mm_load_ps(&p1S[x]), _mm_load_ps(&p1N[x]));
					_mm_store_ps(&ws[B * 0], _mm_add_ps(a0, d));
				}
			}
			for (int y = 1; y < h; ++y) //the other lines
			{
				float* q = &dst[w*y];
				const float* p0N = &src[w*atN(y - r - 1)];
				const float* p1N = &src[w*atN(y - r)];
				const float* p0S = &src[w*atS(y + r)];
				const float* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					const __m128 a0 = _mm_load_ps(&ws[B * 0]);
					const __m128 a1 = _mm_load_ps(&ws[B * 1]);
					const __m128 a3 = _mm_load_ps(&ws[B * 3]);
					_mm_store_ps(&q[x], _mm_mul_ps(_mm_set1_ps(norm), _mm_add_ps(a0, _mm_add_ps(a1, a3))));

					const __m128 d0 = _mm_sub_ps(_mm_load_ps(&p0S[x]), _mm_load_ps(&p0N[x]));
					const __m128 d1 = _mm_sub_ps(_mm_load_ps(&p1S[x]), _mm_load_ps(&p1N[x]));
					const __m128 delta = _mm_sub_ps(d1, d0);

					_mm_store_ps(&ws[B * 0], _mm_add_ps(a0, d1));
					_mm_store_ps(&ws[B * 1], _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR1), delta), _mm_mul_ps(_mm_set1_ps(cf11), a1)), _mm_load_ps(&ws[B * 2])));
					_mm_store_ps(&ws[B * 2], a1);
					_mm_store_ps(&ws[B * 3], _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(cfR2), delta), _mm_mul_ps(_mm_set1_ps(cf12), a3)), _mm_load_ps(&ws[B * 4])));
					_mm_store_ps(&ws[B * 4], a3);
				}
			}
			_mm_free(workspace);
		}

		template<>
		inline void gauss::filter_avx_h_OMP<float>(int w, int h, float* src, float* dst)
		{
			const int B = sizeof(__m256) / sizeof(float);
			const int r = rx;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableX.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableX[t]);

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectX[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectX[1]), cfR2 = table[K*r + 1];

			assert(h%B == 0);
#pragma omp parallel for
			for (int y = 0; y < h / B*B; y += B)
			{
				//to allow for src==dst
				std::vector<float> buf(B*w);
				std::copy(&src[w*y], &src[w*(y + B)], buf.begin());

				__m256 pv0 = _mm256_set_ps(buf[w * 7 + 0], buf[w * 6 + 0], buf[w * 5 + 0], buf[w * 4 + 0], buf[w * 3 + 0], buf[w * 2 + 0], buf[w * 1 + 0], buf[w * 0 + 0]);
				__m256 pv1 = _mm256_set_ps(buf[w * 7 + 1], buf[w * 6 + 1], buf[w * 5 + 1], buf[w * 4 + 1], buf[w * 3 + 1], buf[w * 2 + 1], buf[w * 1 + 1], buf[w * 0 + 1]);

				__m256 sum = pv0;
				__m256 a1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), pv0);
				__m256 b1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), pv1);
				__m256 a2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), pv0);
				__m256 b2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), pv1);

				for (int u = 1; u <= r; ++u)
				{
					const float* p0M = &buf[atW(0 - u)];
					const float* p1M = &buf[atW(1 - u)];
					const float* p0P = &buf[(0 + u)];
					const float* p1P = &buf[(1 + u)];
					__m256 pv0M = _mm256_set_ps(p0M[w * 7], p0M[w * 6], p0M[w * 5], p0M[w * 4], p0M[w * 3], p0M[w * 2], p0M[w * 1], p0M[w * 0]);
					__m256 pv1M = _mm256_set_ps(p1M[w * 7], p1M[w * 6], p1M[w * 5], p1M[w * 4], p1M[w * 3], p1M[w * 2], p1M[w * 1], p1M[w * 0]);
					__m256 pv0P = _mm256_set_ps(p0P[w * 7], p0P[w * 6], p0P[w * 5], p0P[w * 4], p0P[w * 3], p0P[w * 2], p0P[w * 1], p0P[w * 0]);
					__m256 pv1P = _mm256_set_ps(p1P[w * 7], p1P[w * 6], p1P[w * 5], p1P[w * 4], p1P[w * 3], p1P[w * 2], p1P[w * 1], p1P[w * 0]);
					__m256 sumA = _mm256_add_ps(pv0M, pv0P);
					__m256 sumB = _mm256_add_ps(pv1M, pv1P);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 0]), sumA));
					b1 = _mm256_add_ps(b1, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 0]), sumB));
					a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 1]), sumA));
					b2 = _mm256_add_ps(b2, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 1]), sumB));
				}

				//sliding convolution
				float *pA, *pB;
				__m256 pvA, pvB;
				__m256 dvA, dvB, delta;
				__m256 qv[B];

				//the first eight pixels (0<=x<B)
				for (int x = 0; x < B; x += B)
				{
					//the first pixel (x=0)
					qv[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(0 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(0 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);

					sum = _mm256_add_ps(sum, dvA);
					//b1=_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1),delta),_mm256_mul_ps(_mm256_set1_ps(cf11),a1)),b1);
					//b2=_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2),delta),_mm256_mul_ps(_mm256_set1_ps(cf12),a2)),b2);

					//the second pixel (x=1)
					qv[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(1 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(1 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(2 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(2 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(3 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(3 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the fifth pixel (x=4)
					qv[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(4 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(4 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the sixth pixel (x=5)
					qv[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(5 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(5 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the seventh pixel (x=6)
					qv[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(6 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(6 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the eighth pixel (x=7)
					qv[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(7 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(7 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//output with transposition
					_MM256_TRANSPOSE8_PS(qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], qv[6], qv[7]);
					_mm256_storeu_ps(&dst[w*(y + 0)], qv[0]);
					_mm256_storeu_ps(&dst[w*(y + 1)], qv[1]);
					_mm256_storeu_ps(&dst[w*(y + 2)], qv[2]);
					_mm256_storeu_ps(&dst[w*(y + 3)], qv[3]);
					_mm256_storeu_ps(&dst[w*(y + 4)], qv[4]);
					_mm256_storeu_ps(&dst[w*(y + 5)], qv[5]);
					_mm256_storeu_ps(&dst[w*(y + 6)], qv[6]);
					_mm256_storeu_ps(&dst[w*(y + 7)], qv[7]);
				}

				//the other pixels (B<=x<w)
				for (int x = B; x < w / B*B; x += B) //eight-length ring buffers
				{
					//the first pixel (x=0)
					qv[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 0 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 0 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the second pixel (x=1)
					qv[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 1 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 1 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 2 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 2 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 3 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 3 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the fifth pixel (x=4)
					qv[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 4 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 4 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the sixth pixel (x=5)
					qv[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 5 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 5 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the seventh pixel (x=6)
					qv[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 6 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 6 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the eighth pixel (x=7)
					qv[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 7 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 7 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//output with transposition
					_MM256_TRANSPOSE8_PS(qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], qv[6], qv[7]);
					_mm256_storeu_ps(&dst[w*(y + 0) + x], qv[0]);
					_mm256_storeu_ps(&dst[w*(y + 1) + x], qv[1]);
					_mm256_storeu_ps(&dst[w*(y + 2) + x], qv[2]);
					_mm256_storeu_ps(&dst[w*(y + 3) + x], qv[3]);
					_mm256_storeu_ps(&dst[w*(y + 4) + x], qv[4]);
					_mm256_storeu_ps(&dst[w*(y + 5) + x], qv[5]);
					_mm256_storeu_ps(&dst[w*(y + 6) + x], qv[6]);
					_mm256_storeu_ps(&dst[w*(y + 7) + x], qv[7]);
				}
			}
		}

		template<>
		inline void gauss::filter_avx_h<float>(int w, int h, float* src, float* dst)
		{
			const int B = sizeof(__m256) / sizeof(float);
			const int r = rx;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableX.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableX[t]);

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectX[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectX[1]), cfR2 = table[K*r + 1];

			//to allow for src==dst
			std::vector<float> buf(B*w);

			assert(h%B == 0);
			for (int y = 0; y < h / B*B; y += B)
			{
				std::copy(&src[w*y], &src[w*(y + B)], buf.begin());

				__m256 pv0 = _mm256_set_ps(buf[w * 7 + 0], buf[w * 6 + 0], buf[w * 5 + 0], buf[w * 4 + 0], buf[w * 3 + 0], buf[w * 2 + 0], buf[w * 1 + 0], buf[w * 0 + 0]);
				__m256 pv1 = _mm256_set_ps(buf[w * 7 + 1], buf[w * 6 + 1], buf[w * 5 + 1], buf[w * 4 + 1], buf[w * 3 + 1], buf[w * 2 + 1], buf[w * 1 + 1], buf[w * 0 + 1]);

				__m256 sum = pv0;
				__m256 a1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), pv0);
				__m256 b1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), pv1);
				__m256 a2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), pv0);
				__m256 b2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), pv1);

				for (int u = 1; u <= r; ++u)
				{
					const float* p0M = &buf[atW(0 - u)];
					const float* p1M = &buf[atW(1 - u)];
					const float* p0P = &buf[(0 + u)];
					const float* p1P = &buf[(1 + u)];
					__m256 pv0M = _mm256_set_ps(p0M[w * 7], p0M[w * 6], p0M[w * 5], p0M[w * 4], p0M[w * 3], p0M[w * 2], p0M[w * 1], p0M[w * 0]);
					__m256 pv1M = _mm256_set_ps(p1M[w * 7], p1M[w * 6], p1M[w * 5], p1M[w * 4], p1M[w * 3], p1M[w * 2], p1M[w * 1], p1M[w * 0]);
					__m256 pv0P = _mm256_set_ps(p0P[w * 7], p0P[w * 6], p0P[w * 5], p0P[w * 4], p0P[w * 3], p0P[w * 2], p0P[w * 1], p0P[w * 0]);
					__m256 pv1P = _mm256_set_ps(p1P[w * 7], p1P[w * 6], p1P[w * 5], p1P[w * 4], p1P[w * 3], p1P[w * 2], p1P[w * 1], p1P[w * 0]);
					__m256 sumA = _mm256_add_ps(pv0M, pv0P);
					__m256 sumB = _mm256_add_ps(pv1M, pv1P);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 0]), sumA));
					b1 = _mm256_add_ps(b1, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 0]), sumB));
					a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 1]), sumA));
					b2 = _mm256_add_ps(b2, _mm256_mul_ps(_mm256_set1_ps(table[K*u + 1]), sumB));
				}

				//sliding convolution
				float *pA, *pB;
				__m256 pvA, pvB;
				__m256 dvA, dvB, delta;
				__m256 qv[B];

				//the first eight pixels (0<=x<B)
				for (int x = 0; x < B; x += B)
				{
					//the first pixel (x=0)
					qv[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(0 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(0 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);

					sum = _mm256_add_ps(sum, dvA);
					//b1=_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1),delta),_mm256_mul_ps(_mm256_set1_ps(cf11),a1)),b1);
					//b2=_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2),delta),_mm256_mul_ps(_mm256_set1_ps(cf12),a2)),b2);

					//the second pixel (x=1)
					qv[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(1 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(1 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(2 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(2 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(3 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(3 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the fifth pixel (x=4)
					qv[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(4 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(4 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the sixth pixel (x=5)
					qv[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(5 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(5 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the seventh pixel (x=6)
					qv[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(6 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(6 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the eighth pixel (x=7)
					qv[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(7 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(7 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//output with transposition
					_MM256_TRANSPOSE8_PS(qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], qv[6], qv[7]);
					_mm256_storeu_ps(&dst[w*(y + 0)], qv[0]);
					_mm256_storeu_ps(&dst[w*(y + 1)], qv[1]);
					_mm256_storeu_ps(&dst[w*(y + 2)], qv[2]);
					_mm256_storeu_ps(&dst[w*(y + 3)], qv[3]);
					_mm256_storeu_ps(&dst[w*(y + 4)], qv[4]);
					_mm256_storeu_ps(&dst[w*(y + 5)], qv[5]);
					_mm256_storeu_ps(&dst[w*(y + 6)], qv[6]);
					_mm256_storeu_ps(&dst[w*(y + 7)], qv[7]);
				}

				//the other pixels (B<=x<w)
				for (int x = B; x < w / B*B; x += B) //eight-length ring buffers
				{
					//the first pixel (x=0)
					qv[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 0 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 0 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the second pixel (x=1)
					qv[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 1 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 1 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 2 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 2 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 3 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 3 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the fifth pixel (x=4)
					qv[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 4 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 4 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the sixth pixel (x=5)
					qv[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 5 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 5 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//the seventh pixel (x=6)
					qv[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));

					pA = &buf[atW(x + 6 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 6 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvA, dvB);

					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), b1);
					b2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), a2)), b2);

					//the eighth pixel (x=7)
					qv[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));

					pA = &buf[atW(x + 7 - r)]; pvA = _mm256_set_ps(pA[w * 7], pA[w * 6], pA[w * 5], pA[w * 4], pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 7 + r + 1)]; pvB = _mm256_set_ps(pB[w * 7], pB[w * 6], pB[w * 5], pB[w * 4], pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_ps(pvB, pvA);
					delta = _mm256_sub_ps(dvB, dvA);

					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta), _mm256_mul_ps(_mm256_set1_ps(cf11), b1)), a1);
					a2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta), _mm256_mul_ps(_mm256_set1_ps(cf12), b2)), a2);

					//output with transposition
					_MM256_TRANSPOSE8_PS(qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], qv[6], qv[7]);
					_mm256_storeu_ps(&dst[w*(y + 0) + x], qv[0]);
					_mm256_storeu_ps(&dst[w*(y + 1) + x], qv[1]);
					_mm256_storeu_ps(&dst[w*(y + 2) + x], qv[2]);
					_mm256_storeu_ps(&dst[w*(y + 3) + x], qv[3]);
					_mm256_storeu_ps(&dst[w*(y + 4) + x], qv[4]);
					_mm256_storeu_ps(&dst[w*(y + 5) + x], qv[5]);
					_mm256_storeu_ps(&dst[w*(y + 6) + x], qv[6]);
					_mm256_storeu_ps(&dst[w*(y + 7) + x], qv[7]);
				}
			}
		}

		template<>
		inline void gauss::filter_avx_v<float>(int w, int h, float* src, float* dst)
		{
			assert(w % sizeof(__m256) == 0);

			const int B = sizeof(__m256) / sizeof(float);
			const int r = ry;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableY.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableY[t]);

			//work space to keep raster scanning
			float* workspace = reinterpret_cast<float*>(_mm_malloc(sizeof(float)*(2 * K + 1)*w, sizeof(__m256)));

			//calculating the first and second terms
			for (int x = 0; x < w / B*B; x += B)
			{
				float* ws = &workspace[(2 * K + 1)*x];
				__m256 p0 = _mm256_load_ps(&src[x]);
				__m256 p1 = _mm256_load_ps(&src[x + w]);
				_mm256_store_ps(&ws[B * 0], p0);
				_mm256_store_ps(&ws[B * 1], _mm256_mul_ps(p1, _mm256_set1_ps(table[0])));
				_mm256_store_ps(&ws[B * 2], _mm256_mul_ps(p0, _mm256_set1_ps(table[0])));
				_mm256_store_ps(&ws[B * 3], _mm256_mul_ps(p1, _mm256_set1_ps(table[1])));
				_mm256_store_ps(&ws[B * 4], _mm256_mul_ps(p0, _mm256_set1_ps(table[1])));
			}
			for (int v = 1; v <= r; ++v)
			{
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					__m256 sum0 = _mm256_add_ps(_mm256_load_ps(&src[x + w*atN(0 - v)]), _mm256_load_ps(&src[x + w*(0 + v)]));
					__m256 sum1 = _mm256_add_ps(_mm256_load_ps(&src[x + w*atN(1 - v)]), _mm256_load_ps(&src[x + w*(1 + v)]));
					_mm256_store_ps(&ws[B * 0], _mm256_add_ps(_mm256_load_ps(&ws[B * 0]), sum0));
					_mm256_store_ps(&ws[B * 1], _mm256_add_ps(_mm256_load_ps(&ws[B * 1]), _mm256_mul_ps(sum1, _mm256_set1_ps(table[K*v + 0]))));
					_mm256_store_ps(&ws[B * 2], _mm256_add_ps(_mm256_load_ps(&ws[B * 2]), _mm256_mul_ps(sum0, _mm256_set1_ps(table[K*v + 0]))));
					_mm256_store_ps(&ws[B * 3], _mm256_add_ps(_mm256_load_ps(&ws[B * 3]), _mm256_mul_ps(sum1, _mm256_set1_ps(table[K*v + 1]))));
					_mm256_store_ps(&ws[B * 4], _mm256_add_ps(_mm256_load_ps(&ws[B * 4]), _mm256_mul_ps(sum0, _mm256_set1_ps(table[K*v + 1]))));
				}
			}

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectY[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectY[1]), cfR2 = table[K*r + 1];

			//sliding convolution
			for (int y = 0; y < 1; ++y) //the first line (y=0)
			{
				float* q = &dst[w*y];
				const float* p1N = &src[w*atN(y - r)];
				const float* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					const __m256 a0 = _mm256_load_ps(&ws[B * 0]);
					const __m256 a2 = _mm256_load_ps(&ws[B * 2]);
					const __m256 a4 = _mm256_load_ps(&ws[B * 4]);
					_mm256_store_ps(&q[x], _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(a0, _mm256_add_ps(a2, a4))));

					const __m256 d0 = _mm256_sub_ps(_mm256_load_ps(&p1S[x]), _mm256_load_ps(&p1N[x]));
					_mm256_store_ps(&ws[B * 0], _mm256_add_ps(a0, d0));
				}
			}
			for (int y = 1; y < h; ++y) //the other lines
			{
				float* q = &dst[w*y];
				const float* p0N = &src[w*atN(y - r - 1)];
				const float* p1N = &src[w*atN(y - r)];
				const float* p0S = &src[w*atS(y + r)];
				const float* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					const __m256 a0 = _mm256_load_ps(&ws[B * 0]);
					const __m256 a1 = _mm256_load_ps(&ws[B * 1]);
					const __m256 a3 = _mm256_load_ps(&ws[B * 3]);
					_mm256_store_ps(&q[x], _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(a0, _mm256_add_ps(a1, a3))));
					const __m256 d0 = _mm256_sub_ps(_mm256_load_ps(&p0S[x]), _mm256_load_ps(&p0N[x]));
					const __m256 d1 = _mm256_sub_ps(_mm256_load_ps(&p1S[x]), _mm256_load_ps(&p1N[x]));
					const __m256 delta1 = _mm256_sub_ps(d1, d0);

					_mm256_store_ps(&ws[B * 0], _mm256_add_ps(a0, d1));
					_mm256_store_ps(&ws[B * 1], _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta1), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), _mm256_load_ps(&ws[B * 2])));
					_mm256_store_ps(&ws[B * 2], a1);
					_mm256_store_ps(&ws[B * 3], _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta1), _mm256_mul_ps(_mm256_set1_ps(cf12), a3)), _mm256_load_ps(&ws[B * 4])));
					_mm256_store_ps(&ws[B * 4], a3);
				}
			}
			_mm_free(workspace);
		}

		template<>
		inline void gauss::filter_avx_v_OMP<float>(int w, int h, float* src, float* dst)
		{
			assert(w % sizeof(__m256) == 0);

			const int B = sizeof(__m256) / sizeof(float);
			const int r = ry;
			const float norm = float(1.0 / (r + 1 + r));
			std::vector<float> table(tableY.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = float(tableY[t]);

			//work space to keep raster scanning
			float* workspace = reinterpret_cast<float*>(_mm_malloc(sizeof(float)*(2 * K + 1)*w, sizeof(__m256)));

			//calculating the first and second terms
			for (int x = 0; x < w / B*B; x += B)
			{
				float* ws = &workspace[(2 * K + 1)*x];
				__m256 p0 = _mm256_load_ps(&src[x]);
				__m256 p1 = _mm256_load_ps(&src[x + w]);
				_mm256_store_ps(&ws[B * 0], p0);
				_mm256_store_ps(&ws[B * 1], _mm256_mul_ps(p1, _mm256_set1_ps(table[0])));
				_mm256_store_ps(&ws[B * 2], _mm256_mul_ps(p0, _mm256_set1_ps(table[0])));
				_mm256_store_ps(&ws[B * 3], _mm256_mul_ps(p1, _mm256_set1_ps(table[1])));
				_mm256_store_ps(&ws[B * 4], _mm256_mul_ps(p0, _mm256_set1_ps(table[1])));
			}
			//#pragma omp parallel for
			for (int v = 1; v <= r; ++v)
			{
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					__m256 sum0 = _mm256_add_ps(_mm256_load_ps(&src[x + w*atN(0 - v)]), _mm256_load_ps(&src[x + w*(0 + v)]));
					__m256 sum1 = _mm256_add_ps(_mm256_load_ps(&src[x + w*atN(1 - v)]), _mm256_load_ps(&src[x + w*(1 + v)]));
					_mm256_store_ps(&ws[B * 0], _mm256_add_ps(_mm256_load_ps(&ws[B * 0]), sum0));
					_mm256_store_ps(&ws[B * 1], _mm256_add_ps(_mm256_load_ps(&ws[B * 1]), _mm256_mul_ps(sum1, _mm256_set1_ps(table[K*v + 0]))));
					_mm256_store_ps(&ws[B * 2], _mm256_add_ps(_mm256_load_ps(&ws[B * 2]), _mm256_mul_ps(sum0, _mm256_set1_ps(table[K*v + 0]))));
					_mm256_store_ps(&ws[B * 3], _mm256_add_ps(_mm256_load_ps(&ws[B * 3]), _mm256_mul_ps(sum1, _mm256_set1_ps(table[K*v + 1]))));
					_mm256_store_ps(&ws[B * 4], _mm256_add_ps(_mm256_load_ps(&ws[B * 4]), _mm256_mul_ps(sum0, _mm256_set1_ps(table[K*v + 1]))));
				}
			}

			const float cf11 = float(table[K * 1 + 0] * 2.0 / spectY[0]), cfR1 = table[K*r + 0];
			const float cf12 = float(table[K * 1 + 1] * 2.0 / spectY[1]), cfR2 = table[K*r + 1];
			//sliding convolution
			//#pragma omp parallel for
			for (int y = 0; y < 1; ++y) //the first line (y=0)
			{
				float* q = &dst[w*y];
				const float* p1N = &src[w*atN(y - r)];
				const float* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					const __m256 a0 = _mm256_load_ps(&ws[B * 0]);
					const __m256 a2 = _mm256_load_ps(&ws[B * 2]);
					const __m256 a4 = _mm256_load_ps(&ws[B * 4]);
					_mm256_store_ps(&q[x], _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(a0, _mm256_add_ps(a2, a4))));

					const __m256 d0 = _mm256_sub_ps(_mm256_load_ps(&p1S[x]), _mm256_load_ps(&p1N[x]));
					_mm256_store_ps(&ws[B * 0], _mm256_add_ps(a0, d0));

				}
			}

			for (int y = 1; y < h; ++y) //the other lines	
			{
				float* q = &dst[w*y];
				const float* p0N = &src[w*atN(y - r - 1)];
				const float* p1N = &src[w*atN(y - r)];
				const float* p0S = &src[w*atS(y + r)];
				const float* p1S = &src[w*atS(y + r + 1)];
				//#pragma omp parallel for			
				for (int x = 0; x < w / B*B; x += B)
				{
					float* ws = &workspace[(2 * K + 1)*x];
					const __m256 a0 = _mm256_load_ps(&ws[B * 0]);
					const __m256 a1 = _mm256_load_ps(&ws[B * 1]);
					const __m256 a3 = _mm256_load_ps(&ws[B * 3]);
					_mm256_store_ps(&q[x], _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(a0, _mm256_add_ps(a1, a3))));
					const __m256 d0 = _mm256_sub_ps(_mm256_load_ps(&p0S[x]), _mm256_load_ps(&p0N[x]));
					const __m256 d1 = _mm256_sub_ps(_mm256_load_ps(&p1S[x]), _mm256_load_ps(&p1N[x]));
					const __m256 delta1 = _mm256_sub_ps(d1, d0);

					_mm256_store_ps(&ws[B * 0], _mm256_add_ps(a0, d1));
					_mm256_store_ps(&ws[B * 1], _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR1), delta1), _mm256_mul_ps(_mm256_set1_ps(cf11), a1)), _mm256_load_ps(&ws[B * 2])));
					_mm256_store_ps(&ws[B * 2], a1);
					_mm256_store_ps(&ws[B * 3], _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cfR2), delta1), _mm256_mul_ps(_mm256_set1_ps(cf12), a3)), _mm256_load_ps(&ws[B * 4])));
					_mm256_store_ps(&ws[B * 4], a3);
				}
			}
			_mm_free(workspace);
		}

		template<>
		inline void gauss::filter_h<double>(int w, int h, double* src, double* dst)
		{
			const int r = rx;
			const double norm = double(1.0 / (r + 1 + r));
			std::vector<double> table(tableX.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = double(tableX[t]);

			const double cf11 = double(table[K * 1 + 0] * 2.0 / spectX[0]), cfR1 = table[K*r + 0];
			const double cf12 = double(table[K * 1 + 1] * 2.0 / spectX[1]), cfR2 = table[K*r + 1];

			double sum, a1, a2, b1, b2;
			double dA, dB, delta;
			std::vector<double> buf(w); //to allow for src==dst
			for (int y = 0; y < h; ++y)
			{
				std::copy(&src[w*y], &src[w*y + w], buf.begin());

				sum = buf[0];
				a1 = buf[0] * table[0]; b1 = buf[1] * table[0];
				a2 = buf[0] * table[1]; b2 = buf[1] * table[1];
				for (int u = 1; u <= r; ++u)
				{
					const double sumA = buf[atW(0 - u)] + buf[0 + u];
					const double sumB = buf[atW(1 - u)] + buf[1 + u];
					sum += sumA;
					a1 += sumA*table[K*u + 0]; b1 += sumB*table[K*u + 0];
					a2 += sumA*table[K*u + 1]; b2 += sumB*table[K*u + 1];
				}

				//the first pixel (x=0)
				double* q = &dst[w*y];
				q[0] = norm*(sum + a1 + a2);
				dA = buf[atE(0 + r + 1)] - buf[atW(0 - r)];
				sum += dA;

				//the other pixels (0<x<w)
				int x = 1;
				while (true) //four-length ring buffers
				{
					q[x] = norm*(sum + b1 + b2);
					dB = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dA - dB;
					sum += dB;
					a1 += -cf11*b1 + cfR1*delta;
					a2 += -cf12*b2 + cfR2*delta;
					x++; if (w <= x) break;

					q[x] = norm*(sum - a1 - a2);
					dA = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dB - dA;
					sum += dA;
					b1 += +cf11*a1 + cfR1*delta;
					b2 += +cf12*a2 + cfR2*delta;
					x++; if (w <= x) break;

					q[x] = norm*(sum - b1 - b2);
					dB = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dA - dB;
					sum += dB;
					a1 += -cf11*b1 - cfR1*delta;
					a2 += -cf12*b2 - cfR2*delta;
					x++; if (w <= x) break;

					q[x] = norm*(sum + a1 + a2);
					dA = buf[atE(x + r + 1)] - buf[atW(x - r)]; delta = dB - dA;
					sum += dA;
					b1 += +cf11*a1 - cfR1*delta;
					b2 += +cf12*a2 - cfR2*delta;
					x++; if (w <= x) break;
				}
			}
		}

		template<>
		inline void gauss::filter_v<double>(int w, int h, double* src, double* dst)
		{
			const int r = ry;
			const double norm = double(1.0 / (r + 1 + r));
			std::vector<double> table(tableY.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = double(tableY[t]);

			//work space to keep raster scanning
			std::vector<double> workspace((2 * K + 1)*w);

			//calculating the first and second terms
			for (int x = 0; x < w; ++x)
			{
				double* ws = &workspace[(2 * K + 1)*x];
				ws[0] = src[x];
				ws[1] = src[x + w] * table[0]; ws[2] = src[x] * table[0];
				ws[3] = src[x + w] * table[1]; ws[4] = src[x] * table[1];
			}
			for (int v = 1; v <= r; ++v)
			{
				for (int x = 0; x < w; ++x)
				{
					const double sum0 = src[x + w*atN(0 - v)] + src[x + w*(0 + v)];
					const double sum1 = src[x + w*atN(1 - v)] + src[x + w*(1 + v)];
					double* ws = &workspace[(2 * K + 1)*x];
					ws[0] += sum0;
					ws[1] += sum1*table[K*v + 0]; ws[2] += sum0*table[K*v + 0];
					ws[3] += sum1*table[K*v + 1]; ws[4] += sum0*table[K*v + 1];
				}
			}

			const double cf11 = double(table[K * 1 + 0] * 2.0 / spectY[0]), cfR1 = table[K*r + 0];
			const double cf12 = double(table[K * 1 + 1] * 2.0 / spectY[1]), cfR2 = table[K*r + 1];

			double *q, *p0N, *p0S, *p1N, *p1S;
			for (int y = 0; y < 1; ++y) //the first line (y=0)
			{
				q = &dst[w*y];
				p1N = &src[w*atN(0 - r)]; p1S = &src[w*atS(0 + r + 1)];
				for (int x = 0; x < w; ++x)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					q[x] = norm*(ws[0] + ws[2] + ws[4]);
					ws[0] += p1S[x] - p1N[x];
				}
			}
			for (int y = 1; y < h; ++y) //remaining lines (with two-length ring buffers)
			{
				q = &dst[w*y];
				p0N = &src[w*atN(y - r - 1)]; p0S = &src[w*atS(y + r)];
				p1N = &src[w*atN(y - r)]; p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w; ++x)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					q[x] = norm*(ws[0] + ws[1] + ws[3]);

					const double d0 = p0S[x] - p0N[x];
					const double d1 = p1S[x] - p1N[x];
					const double delta = d1 - d0;

					ws[0] += d1;
					ws[2] = cfR1*delta + cf11*ws[1] - ws[2];
					ws[4] = cfR2*delta + cf12*ws[3] - ws[4];
				}
				y++; if (h <= y) break; //to the next line

				q = &dst[w*y];
				p0N = &src[w*atN(y - r - 1)]; p0S = &src[w*atS(y + r)];
				p1N = &src[w*atN(y - r)]; p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w; ++x)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					q[x] = norm*(ws[0] + ws[2] + ws[4]);

					const double d0 = p0S[x] - p0N[x];
					const double d1 = p1S[x] - p1N[x];
					const double delta = d1 - d0;

					ws[0] += d1;
					ws[1] = cfR1*delta + cf11*ws[2] - ws[1];
					ws[3] = cfR2*delta + cf12*ws[4] - ws[3];
				}
			}
		}


#define _MM_TRANSPOSE2_PD(row0, row1) {          \
            __m128d tmp1, tmp0;                      \
                                                     \
            tmp0   = _mm_shuffle_pd(row0, row1, 0);	 \
            tmp1   = _mm_shuffle_pd(row0, row1, 3);	 \
            row0   = tmp0;			 			     \
            row1   = tmp1;							 \
				        }

		template<>
		inline void gauss::filter_sse_h<double>(int w, int h, double* src, double* dst)
		{
			const int B = sizeof(__m128d) / sizeof(double);
			const int r = rx;
			const double norm = double(1.0 / (r + 1 + r));
			std::vector<double> table(tableX.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = double(tableX[t]);

			const double cf11 = double(table[K * 1 + 0] * 2.0 / spectX[0]), cfR1 = table[K*r + 0];
			const double cf12 = double(table[K * 1 + 1] * 2.0 / spectX[1]), cfR2 = table[K*r + 1];

			//to allow for src==dst
			std::vector<double> buf(B*w);

			assert(h%B == 0);
			for (int y = 0; y < h / B*B; y += B)
			{
				std::copy(&src[w*y], &src[w*(y + B)], buf.begin());

				__m128d pv0 = _mm_set_pd(buf[w * 1 + 0], buf[w * 0 + 0]);
				__m128d pv1 = _mm_set_pd(buf[w * 1 + 1], buf[w * 0 + 1]);

				__m128d sum = pv0;
				__m128d a1 = _mm_mul_pd(_mm_set1_pd(table[0]), pv0);
				__m128d b1 = _mm_mul_pd(_mm_set1_pd(table[0]), pv1);
				__m128d a2 = _mm_mul_pd(_mm_set1_pd(table[1]), pv0);
				__m128d b2 = _mm_mul_pd(_mm_set1_pd(table[1]), pv1);

				for (int u = 1; u <= r; ++u)
				{
					const double* p0M = &buf[atW(0 - u)];
					const double* p1M = &buf[atW(1 - u)];
					const double* p0P = &buf[(0 + u)];
					const double* p1P = &buf[(1 + u)];
					__m128d pv0M = _mm_set_pd(p0M[w * 1], p0M[w * 0]);
					__m128d pv1M = _mm_set_pd(p1M[w * 1], p1M[w * 0]);
					__m128d pv0P = _mm_set_pd(p0P[w * 1], p0P[w * 0]);
					__m128d pv1P = _mm_set_pd(p1P[w * 1], p1P[w * 0]);
					__m128d sumA = _mm_add_pd(pv0M, pv0P);
					__m128d sumB = _mm_add_pd(pv1M, pv1P);

					sum = _mm_add_pd(sum, sumA);
					a1 = _mm_add_pd(a1, _mm_mul_pd(_mm_set1_pd(table[K*u + 0]), sumA));
					b1 = _mm_add_pd(b1, _mm_mul_pd(_mm_set1_pd(table[K*u + 0]), sumB));
					a2 = _mm_add_pd(a2, _mm_mul_pd(_mm_set1_pd(table[K*u + 1]), sumA));
					b2 = _mm_add_pd(b2, _mm_mul_pd(_mm_set1_pd(table[K*u + 1]), sumB));
				}

				//sliding convolution
				double *pA, *pB;
				__m128d pvA, pvB;
				__m128d dvA, dvB, delta;
				__m128d qv[B];

				//the first four pixels (0<=x<B)
				for (int x = 0; x < B; x += B)
				{
					//the first pixel (x=0)
					qv[0] = _mm_mul_pd(_mm_set1_pd(norm), _mm_add_pd(sum, _mm_add_pd(a1, a2)));

					pA = &buf[atW(0 - r)]; pvA = _mm_set_pd(pA[w * 1], pA[w * 0]);
					pB = &buf[(0 + r + 1)]; pvB = _mm_set_pd(pB[w * 1], pB[w * 0]);
					dvA = _mm_sub_pd(pvB, pvA);

					sum = _mm_add_pd(sum, dvA);
					//b1=_mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR1),delta),_mm_mul_pd(_mm_set1_pd(cf11),a1)),b1);
					//b2=_mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR2),delta),_mm_mul_pd(_mm_set1_pd(cf12),a2)),b2);

					//the second pixel (x=1)
					qv[1] = _mm_mul_pd(_mm_set1_pd(norm), _mm_add_pd(sum, _mm_add_pd(b1, b2)));

					pA = &buf[atW(1 - r)]; pvA = _mm_set_pd(pA[w * 1], pA[w * 0]);
					pB = &buf[(1 + r + 1)]; pvB = _mm_set_pd(pB[w * 1], pB[w * 0]);
					dvB = _mm_sub_pd(pvB, pvA);
					delta = _mm_sub_pd(dvB, dvA);

					sum = _mm_add_pd(sum, dvB);
					a1 = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR1), delta), _mm_mul_pd(_mm_set1_pd(cf11), b1)), a1);
					a2 = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR2), delta), _mm_mul_pd(_mm_set1_pd(cf12), b2)), a2);

					//output with transposition
					_MM_TRANSPOSE2_PD(qv[0], qv[1]);
					_mm_storeu_pd(&dst[w*(y + 0)], qv[0]);
					_mm_storeu_pd(&dst[w*(y + 1)], qv[1]);
				}

				//the other pixels (B<=x<w)
				for (int x = B; x < w / B*B; x += B) //four-length ring buffers
				{
					//the first pixel (x=0)
					qv[0] = _mm_mul_pd(_mm_set1_pd(norm), _mm_add_pd(sum, _mm_add_pd(a1, a2)));

					pA = &buf[atW(x + 0 - r)]; pvA = _mm_set_pd(pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 0 + r + 1)]; pvB = _mm_set_pd(pB[w * 1], pB[w * 0]);
					dvA = _mm_sub_pd(pvB, pvA);
					delta = _mm_sub_pd(dvA, dvB);

					sum = _mm_add_pd(sum, dvA);
					b1 = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR1), delta), _mm_mul_pd(_mm_set1_pd(cf11), a1)), b1);
					b2 = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR2), delta), _mm_mul_pd(_mm_set1_pd(cf12), a2)), b2);

					//the second pixel (x=1)
					qv[1] = _mm_mul_pd(_mm_set1_pd(norm), _mm_add_pd(sum, _mm_add_pd(b1, b2)));

					pA = &buf[atW(x + 1 - r)]; pvA = _mm_set_pd(pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 1 + r + 1)]; pvB = _mm_set_pd(pB[w * 1], pB[w * 0]);
					dvB = _mm_sub_pd(pvB, pvA);
					delta = _mm_sub_pd(dvB, dvA);

					sum = _mm_add_pd(sum, dvB);
					a1 = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR1), delta), _mm_mul_pd(_mm_set1_pd(cf11), b1)), a1);
					a2 = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR2), delta), _mm_mul_pd(_mm_set1_pd(cf12), b2)), a2);

					//output with transposition
					_MM_TRANSPOSE2_PD(qv[0], qv[1]);
					_mm_storeu_pd(&dst[w*(y + 0) + x], qv[0]);
					_mm_storeu_pd(&dst[w*(y + 1) + x], qv[1]);
				}
			}
		}

		template<>
		inline void gauss::filter_sse_v<double>(int w, int h, double* src, double* dst)
		{
			assert(w % sizeof(__m128d) == 0);

			const int B = sizeof(__m128d) / sizeof(double);
			const int r = ry;
			const double norm = double(1.0 / (r + 1 + r));
			std::vector<double> table(tableY.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = double(tableY[t]);

			//work space to keep raster scanning
			double* workspace = reinterpret_cast<double*>(_mm_malloc(sizeof(double)*(2 * K + 1)*w, sizeof(__m128d)));

			//calculating the first and second terms
			for (int x = 0; x < w / B*B; x += B)
			{
				double* ws = &workspace[(2 * K + 1)*x];
				__m128d p0 = _mm_load_pd(&src[x]);
				__m128d p1 = _mm_load_pd(&src[x + w]);
				_mm_store_pd(&ws[B * 0], p0);
				_mm_store_pd(&ws[B * 1], _mm_mul_pd(p1, _mm_set1_pd(table[0])));
				_mm_store_pd(&ws[B * 2], _mm_mul_pd(p0, _mm_set1_pd(table[0])));
				_mm_store_pd(&ws[B * 3], _mm_mul_pd(p1, _mm_set1_pd(table[1])));
				_mm_store_pd(&ws[B * 4], _mm_mul_pd(p0, _mm_set1_pd(table[1])));
			}
			for (int v = 1; v <= r; ++v)
			{
				for (int x = 0; x < w / B*B; x += B)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					__m128d sum0 = _mm_add_pd(_mm_load_pd(&src[x + w*atN(0 - v)]), _mm_load_pd(&src[x + w*(0 + v)]));
					__m128d sum1 = _mm_add_pd(_mm_load_pd(&src[x + w*atN(1 - v)]), _mm_load_pd(&src[x + w*(1 + v)]));
					_mm_store_pd(&ws[B * 0], _mm_add_pd(_mm_load_pd(&ws[B * 0]), sum0));
					_mm_store_pd(&ws[B * 1], _mm_add_pd(_mm_load_pd(&ws[B * 1]), _mm_mul_pd(sum1, _mm_set1_pd(table[K*v + 0]))));
					_mm_store_pd(&ws[B * 2], _mm_add_pd(_mm_load_pd(&ws[B * 2]), _mm_mul_pd(sum0, _mm_set1_pd(table[K*v + 0]))));
					_mm_store_pd(&ws[B * 3], _mm_add_pd(_mm_load_pd(&ws[B * 3]), _mm_mul_pd(sum1, _mm_set1_pd(table[K*v + 1]))));
					_mm_store_pd(&ws[B * 4], _mm_add_pd(_mm_load_pd(&ws[B * 4]), _mm_mul_pd(sum0, _mm_set1_pd(table[K*v + 1]))));
				}
			}

			const double cf11 = double(table[K * 1 + 0] * 2.0 / spectY[0]), cfR1 = table[K*r + 0];
			const double cf12 = double(table[K * 1 + 1] * 2.0 / spectY[1]), cfR2 = table[K*r + 1];

			//sliding convolution
			for (int y = 0; y < 1; ++y) //the first line (y=0)
			{
				double* q = &dst[w*y];
				const double* p1N = &src[w*atN(y - r)];
				const double* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					const __m128d a0 = _mm_load_pd(&ws[B * 0]);
					const __m128d a2 = _mm_load_pd(&ws[B * 2]);
					const __m128d a4 = _mm_load_pd(&ws[B * 4]);
					_mm_store_pd(&q[x], _mm_mul_pd(_mm_set1_pd(norm), _mm_add_pd(a0, _mm_add_pd(a2, a4))));

					const __m128d d = _mm_sub_pd(_mm_load_pd(&p1S[x]), _mm_load_pd(&p1N[x]));
					_mm_store_pd(&ws[B * 0], _mm_add_pd(a0, d));
				}
			}
			for (int y = 1; y < h; ++y) //the other lines
			{
				double* q = &dst[w*y];
				const double* p0N = &src[w*atN(y - r - 1)];
				const double* p1N = &src[w*atN(y - r)];
				const double* p0S = &src[w*atS(y + r)];
				const double* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					const __m128d a0 = _mm_load_pd(&ws[B * 0]);
					const __m128d a1 = _mm_load_pd(&ws[B * 1]);
					const __m128d a3 = _mm_load_pd(&ws[B * 3]);
					_mm_store_pd(&q[x], _mm_mul_pd(_mm_set1_pd(norm), _mm_add_pd(a0, _mm_add_pd(a1, a3))));

					const __m128d d0 = _mm_sub_pd(_mm_load_pd(&p0S[x]), _mm_load_pd(&p0N[x]));
					const __m128d d1 = _mm_sub_pd(_mm_load_pd(&p1S[x]), _mm_load_pd(&p1N[x]));
					const __m128d delta = _mm_sub_pd(d1, d0);

					_mm_store_pd(&ws[B * 0], _mm_add_pd(a0, d1));
					_mm_store_pd(&ws[B * 1], _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR1), delta), _mm_mul_pd(_mm_set1_pd(cf11), a1)), _mm_load_pd(&ws[B * 2])));
					_mm_store_pd(&ws[B * 2], a1);
					_mm_store_pd(&ws[B * 3], _mm_sub_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(cfR2), delta), _mm_mul_pd(_mm_set1_pd(cf12), a3)), _mm_load_pd(&ws[B * 4])));
					_mm_store_pd(&ws[B * 4], a3);
				}
			}
			_mm_free(workspace);
		}

		template<>
		inline void gauss::filter_avx_h<double>(int w, int h, double* src, double* dst)
		{
			const int B = sizeof(__m256d) / sizeof(double);
			const int r = rx;
			const double norm = double(1.0 / (r + 1 + r));
			std::vector<double> table(tableX.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = double(tableX[t]);

			const double cf11 = double(table[K * 1 + 0] * 2.0 / spectX[0]), cfR1 = table[K*r + 0];
			const double cf12 = double(table[K * 1 + 1] * 2.0 / spectX[1]), cfR2 = table[K*r + 1];

			//to allow for src==dst
			std::vector<double> buf(B*w);

			assert(h%B == 0);
			for (int y = 0; y < h / B*B; y += B)
			{
				std::copy(&src[w*y], &src[w*(y + B)], buf.begin());

				__m256d pv0 = _mm256_set_pd(buf[w * 3 + 0], buf[w * 2 + 0], buf[w * 1 + 0], buf[w * 0 + 0]);
				__m256d pv1 = _mm256_set_pd(buf[w * 3 + 1], buf[w * 2 + 1], buf[w * 1 + 1], buf[w * 0 + 1]);

				__m256d sum = pv0;
				__m256d a1 = _mm256_mul_pd(_mm256_set1_pd(table[0]), pv0);
				__m256d b1 = _mm256_mul_pd(_mm256_set1_pd(table[0]), pv1);
				__m256d a2 = _mm256_mul_pd(_mm256_set1_pd(table[1]), pv0);
				__m256d b2 = _mm256_mul_pd(_mm256_set1_pd(table[1]), pv1);

				for (int u = 1; u <= r; ++u)
				{
					const double* p0M = &buf[atW(0 - u)];
					const double* p1M = &buf[atW(1 - u)];
					const double* p0P = &buf[(0 + u)];
					const double* p1P = &buf[(1 + u)];
					__m256d pv0M = _mm256_set_pd(p0M[w * 3], p0M[w * 2], p0M[w * 1], p0M[w * 0]);
					__m256d pv1M = _mm256_set_pd(p1M[w * 3], p1M[w * 2], p1M[w * 1], p1M[w * 0]);
					__m256d pv0P = _mm256_set_pd(p0P[w * 3], p0P[w * 2], p0P[w * 1], p0P[w * 0]);
					__m256d pv1P = _mm256_set_pd(p1P[w * 3], p1P[w * 2], p1P[w * 1], p1P[w * 0]);
					__m256d sumA = _mm256_add_pd(pv0M, pv0P);
					__m256d sumB = _mm256_add_pd(pv1M, pv1P);

					sum = _mm256_add_pd(sum, sumA);
					a1 = _mm256_add_pd(a1, _mm256_mul_pd(_mm256_set1_pd(table[K*u + 0]), sumA));
					b1 = _mm256_add_pd(b1, _mm256_mul_pd(_mm256_set1_pd(table[K*u + 0]), sumB));
					a2 = _mm256_add_pd(a2, _mm256_mul_pd(_mm256_set1_pd(table[K*u + 1]), sumA));
					b2 = _mm256_add_pd(b2, _mm256_mul_pd(_mm256_set1_pd(table[K*u + 1]), sumB));
				}

				//sliding convolution
				double *pA, *pB;
				__m256d pvA, pvB;
				__m256d dvA, dvB, delta;
				__m256d qv[B];

				//the first four pixels (0<=x<B)
				for (int x = 0; x < B; x += B)
				{
					//the first pixel (x=0)
					qv[0] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(a1, a2)));

					pA = &buf[atW(0 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(0 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_pd(pvB, pvA);

					sum = _mm256_add_pd(sum, dvA);
					//b1=_mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1),delta),_mm256_mul_pd(_mm256_set1_pd(cf11),a1)),b1);
					//b2=_mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2),delta),_mm256_mul_pd(_mm256_set1_pd(cf12),a2)),b2);

					//the second pixel (x=1)
					qv[1] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(b1, b2)));

					pA = &buf[atW(1 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(1 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_pd(pvB, pvA);
					delta = _mm256_sub_pd(dvB, dvA);

					sum = _mm256_add_pd(sum, dvB);
					a1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), b1)), a1);
					a2 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(a1, a2)));

					pA = &buf[atW(2 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(2 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_pd(pvB, pvA);
					delta = _mm256_sub_pd(dvA, dvB);

					sum = _mm256_add_pd(sum, dvA);
					b1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), a1)), b1);
					b2 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(b1, b2)));

					pA = &buf[atW(3 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[(3 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_pd(pvB, pvA);
					delta = _mm256_sub_pd(dvB, dvA);

					sum = _mm256_add_pd(sum, dvB);
					a1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), b1)), a1);
					a2 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), b2)), a2);

					//output with transposition
					_MM256_TRANSPOSE4_PD(qv[0], qv[1], qv[2], qv[3]);
					_mm256_storeu_pd(&dst[w*(y + 0)], qv[0]);
					_mm256_storeu_pd(&dst[w*(y + 1)], qv[1]);
					_mm256_storeu_pd(&dst[w*(y + 2)], qv[2]);
					_mm256_storeu_pd(&dst[w*(y + 3)], qv[3]);
				}

				//the other pixels (B<=x<w)
				for (int x = B; x < w / B*B; x += B) //four-length ring buffers
				{
					//the first pixel (x=0)
					qv[0] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(a1, a2)));

					pA = &buf[atW(x + 0 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 0 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_pd(pvB, pvA);
					delta = _mm256_sub_pd(dvA, dvB);

					sum = _mm256_add_pd(sum, dvA);
					b1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), a1)), b1);
					b2 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), a2)), b2);

					//the second pixel (x=1)
					qv[1] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(b1, b2)));

					pA = &buf[atW(x + 1 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 1 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_pd(pvB, pvA);
					delta = _mm256_sub_pd(dvB, dvA);

					sum = _mm256_add_pd(sum, dvB);
					a1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), b1)), a1);
					a2 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), b2)), a2);

					//the third pixel (x=2)
					qv[2] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(a1, a2)));

					pA = &buf[atW(x + 2 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 2 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvA = _mm256_sub_pd(pvB, pvA);
					delta = _mm256_sub_pd(dvA, dvB);

					sum = _mm256_add_pd(sum, dvA);
					b1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), a1)), b1);
					b2 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), a2)), b2);

					//the forth pixel (x=3)
					qv[3] = _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(sum, _mm256_add_pd(b1, b2)));

					pA = &buf[atW(x + 3 - r)]; pvA = _mm256_set_pd(pA[w * 3], pA[w * 2], pA[w * 1], pA[w * 0]);
					pB = &buf[atE(x + 3 + r + 1)]; pvB = _mm256_set_pd(pB[w * 3], pB[w * 2], pB[w * 1], pB[w * 0]);
					dvB = _mm256_sub_pd(pvB, pvA);
					delta = _mm256_sub_pd(dvB, dvA);

					sum = _mm256_add_pd(sum, dvB);
					a1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), b1)), a1);
					a2 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), b2)), a2);

					//output with transposition
					_MM256_TRANSPOSE4_PD(qv[0], qv[1], qv[2], qv[3]);
					_mm256_storeu_pd(&dst[w*(y + 0) + x], qv[0]);
					_mm256_storeu_pd(&dst[w*(y + 1) + x], qv[1]);
					_mm256_storeu_pd(&dst[w*(y + 2) + x], qv[2]);
					_mm256_storeu_pd(&dst[w*(y + 3) + x], qv[3]);
				}
			}
		}

		template<>
		inline void gauss::filter_avx_v<double>(int w, int h, double* src, double* dst)
		{
			assert(w % sizeof(__m256d) == 0);

			const int B = sizeof(__m256d) / sizeof(double);
			const int r = ry;
			const double norm = double(1.0 / (r + 1 + r));
			std::vector<double> table(tableY.size());
			for (int t = 0; t<int(table.size()); ++t)
				table[t] = double(tableY[t]);

			//work space to keep raster scanning
			double* workspace = reinterpret_cast<double*>(_mm_malloc(sizeof(double)*(2 * K + 1)*w, sizeof(__m256d)));

			//calculating the first and second terms
			for (int x = 0; x < w / B*B; x += B)
			{
				double* ws = &workspace[(2 * K + 1)*x];
				__m256d p0 = _mm256_load_pd(&src[x]);
				__m256d p1 = _mm256_load_pd(&src[x + w]);
				_mm256_store_pd(&ws[B * 0], p0);
				_mm256_store_pd(&ws[B * 1], _mm256_mul_pd(p1, _mm256_set1_pd(table[0])));
				_mm256_store_pd(&ws[B * 2], _mm256_mul_pd(p0, _mm256_set1_pd(table[0])));
				_mm256_store_pd(&ws[B * 3], _mm256_mul_pd(p1, _mm256_set1_pd(table[1])));
				_mm256_store_pd(&ws[B * 4], _mm256_mul_pd(p0, _mm256_set1_pd(table[1])));
			}
			for (int v = 1; v <= r; ++v)
			{
				for (int x = 0; x < w / B*B; x += B)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					__m256d sum0 = _mm256_add_pd(_mm256_load_pd(&src[x + w*atN(0 - v)]), _mm256_load_pd(&src[x + w*(0 + v)]));
					__m256d sum1 = _mm256_add_pd(_mm256_load_pd(&src[x + w*atN(1 - v)]), _mm256_load_pd(&src[x + w*(1 + v)]));
					_mm256_store_pd(&ws[B * 0], _mm256_add_pd(_mm256_load_pd(&ws[B * 0]), sum0));
					_mm256_store_pd(&ws[B * 1], _mm256_add_pd(_mm256_load_pd(&ws[B * 1]), _mm256_mul_pd(sum1, _mm256_set1_pd(table[K*v + 0]))));
					_mm256_store_pd(&ws[B * 2], _mm256_add_pd(_mm256_load_pd(&ws[B * 2]), _mm256_mul_pd(sum0, _mm256_set1_pd(table[K*v + 0]))));
					_mm256_store_pd(&ws[B * 3], _mm256_add_pd(_mm256_load_pd(&ws[B * 3]), _mm256_mul_pd(sum1, _mm256_set1_pd(table[K*v + 1]))));
					_mm256_store_pd(&ws[B * 4], _mm256_add_pd(_mm256_load_pd(&ws[B * 4]), _mm256_mul_pd(sum0, _mm256_set1_pd(table[K*v + 1]))));
				}
			}

			const double cf11 = double(table[K * 1 + 0] * 2.0 / spectY[0]), cfR1 = table[K*r + 0];
			const double cf12 = double(table[K * 1 + 1] * 2.0 / spectY[1]), cfR2 = table[K*r + 1];

			//sliding convolution
			for (int y = 0; y < 1; ++y) //the first line (y=0)
			{
				double* q = &dst[w*y];
				const double* p1N = &src[w*atN(y - r)];
				const double* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					const __m256d a0 = _mm256_load_pd(&ws[B * 0]);
					const __m256d a2 = _mm256_load_pd(&ws[B * 2]);
					const __m256d a4 = _mm256_load_pd(&ws[B * 4]);
					_mm256_store_pd(&q[x], _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(a0, _mm256_add_pd(a2, a4))));

					const __m256d d = _mm256_sub_pd(_mm256_load_pd(&p1S[x]), _mm256_load_pd(&p1N[x]));
					_mm256_store_pd(&ws[B * 0], _mm256_add_pd(a0, d));
				}
			}
			for (int y = 1; y < h; ++y) //the other lines
			{
				double* q = &dst[w*y];
				const double* p0N = &src[w*atN(y - r - 1)];
				const double* p1N = &src[w*atN(y - r)];
				const double* p0S = &src[w*atS(y + r)];
				const double* p1S = &src[w*atS(y + r + 1)];
				for (int x = 0; x < w / B*B; x += B)
				{
					double* ws = &workspace[(2 * K + 1)*x];
					const __m256d a0 = _mm256_load_pd(&ws[B * 0]);
					const __m256d a1 = _mm256_load_pd(&ws[B * 1]);
					const __m256d a3 = _mm256_load_pd(&ws[B * 3]);
					_mm256_store_pd(&q[x], _mm256_mul_pd(_mm256_set1_pd(norm), _mm256_add_pd(a0, _mm256_add_pd(a1, a3))));

					const __m256d d0 = _mm256_sub_pd(_mm256_load_pd(&p0S[x]), _mm256_load_pd(&p0N[x]));
					const __m256d d1 = _mm256_sub_pd(_mm256_load_pd(&p1S[x]), _mm256_load_pd(&p1N[x]));
					const __m256d delta = _mm256_sub_pd(d1, d0);

					_mm256_store_pd(&ws[B * 0], _mm256_add_pd(a0, d1));
					_mm256_store_pd(&ws[B * 1], _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR1), delta), _mm256_mul_pd(_mm256_set1_pd(cf11), a1)), _mm256_load_pd(&ws[B * 2])));
					_mm256_store_pd(&ws[B * 2], a1);
					_mm256_store_pd(&ws[B * 3], _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(cfR2), delta), _mm256_mul_pd(_mm256_set1_pd(cf12), a3)), _mm256_load_pd(&ws[B * 4])));
					_mm256_store_pd(&ws[B * 4], a3);
				}
			}
			_mm_free(workspace);
		}
	}

	void GaussianBlurSR_cliped_32F(InputArray src_, OutputArray dest, float sigma)
	{
		Mat src = src_.getMat();
		if (dest.empty())dest.create(src_.size(), src.type());
		Mat dst = dest.getMat();

		if (src.channels() == 1)
		{
			spectral_recursive_filter::gauss srf_gauss(sigma, sigma);
			if (src.data != dst.data)
			{
				srf_gauss.filter(src, dst);
			}
			else
			{
				Mat a(src.size(), src.type());
				srf_gauss.filter(src, a);
				a.copyTo(dst);
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> plane;
			split(src, plane);

			Mat temp(src.size(), CV_32F);
			spectral_recursive_filter::gauss srf_gauss(sigma, sigma);
			srf_gauss.filter(plane[0], temp); temp.copyTo(plane[0]);
			srf_gauss.filter(plane[1], temp); temp.copyTo(plane[1]);
			srf_gauss.filter(plane[2], temp); temp.copyTo(plane[2]);

			merge(plane, dest);
		}
	}

	void GaussianBlurSR_cliped_64F(InputArray src_, OutputArray dest, double sigma)
	{
		Mat src = src_.getMat();
		Mat srcf2(src.size(), CV_64FC(src.channels()));

		if (src.channels() == 1)
		{
			spectral_recursive_filter::gauss srf_gauss(sigma, sigma);
			srf_gauss.filter(src, srcf2);
		}
		else if (src.channels() == 3)
		{
			vector<Mat> plane;
			split(src, plane);

			Mat temp(src.size(), CV_64F);
			spectral_recursive_filter::gauss srf_gauss(sigma, sigma);
			srf_gauss.filter(plane[0], temp); temp.copyTo(plane[0]);
			srf_gauss.filter(plane[1], temp); temp.copyTo(plane[1]);
			srf_gauss.filter(plane[2], temp); temp.copyTo(plane[2]);

			merge(plane, srcf2);
		}

		if (dest.empty()) srcf2.copyTo(dest);
		else if (dest.depth() != CV_64F) srcf2.convertTo(dest, CV_64F);
		else srcf2.copyTo(dest);
	}

	void GaussianBlurSR(InputArray src, OutputArray dest, const double sigma)
	{
		int SIMDSTEP = 4;
		if (src.depth() == CV_64F) SIMDSTEP = 2;

		int xpad = src.size().width%SIMDSTEP;
		int ypad = src.size().height%SIMDSTEP;

		xpad = (SIMDSTEP - xpad) % SIMDSTEP;
		ypad = (SIMDSTEP - ypad) % SIMDSTEP;

		if (xpad == 0 && ypad == 0)
		{
			if (src.depth() == CV_64F)
			{
				GaussianBlurSR_cliped_64F(src, dest, sigma);
			}
			else if (src.depth() == CV_32F)
			{
				GaussianBlurSR_cliped_32F(src, dest, (float)sigma);
			}
			else
			{
				Mat srcf, destf;
				src.getMat().convertTo(srcf, CV_32F);
				GaussianBlurSR_cliped_32F(srcf, destf, (float)sigma);
				destf.convertTo(dest, src.depth(), 1.0, 0.5);
			}
		}
		else
		{
			Mat s, d;
			copyMakeBorder(src, s, 0, ypad, 0, xpad, BORDER_REFLECT);

			if (src.depth() == CV_64F)
			{
				GaussianBlurSR_cliped_64F(s, d, sigma);
			}
			else if (src.depth() == CV_32F)
			{
				GaussianBlurSR_cliped_32F(s, d, (float)sigma);
			}
			else
			{
				Mat srcf, destf;
				s.convertTo(srcf, CV_32F);
				GaussianBlurSR_cliped_32F(srcf, destf, (float)sigma);
				destf.convertTo(d, src.depth(), 1.0, 0.5);
			}
			Mat(d(Rect(Point(0, 0), src.size()))).copyTo(dest);
		}
	}
}