#include "multiscalefilter/MultiScaleFilter.hpp"
#include "multiscalefilter/FourierSeriesExpansion.h"
#include <search1D.hpp>

using namespace cv;
using namespace std;
namespace cp
{
#pragma region Fourier
	class ComputeCompressiveT_Fast32F : public cp::Search1D32F
	{
		int cbf_order;
		float sigma_range;
		float Salpha, Sbeta, Ssigma;
		int windowType;
		int Imin = 0;
		int Imax = 0;
		int Irange = 0;
		std::vector<float> ideal;
		int window_type;
		int integration_interval;

		template <typename type, int windowType>
		float getArbitraryRangeKernelErrorEven(const type T, const int order, const type sigma_range, const int Imin, const int Imax, std::vector<type>& ideal, const int integration_interval, int window_type,
			float Salpha, float Sbeta, float Ssigma)
		{
			std::vector<type> alpha(order + 1);
			std::vector<type> beta(order + 1);
			//const type gaussRangeCoeff = (type)(-0.5 / (sigma_range * sigma_range));
			const type omega = (float)(CV_2PI / T);

			type alphaSum = 1.f;
			{
				FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, Ssigma, 0, windowType);
				const double normal = 2.0 / Fourier.init(0, T / 2, integration_interval);
				for (int n = 1; n < order + 1; n++)
				{
					FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, Ssigma, n, windowType);
					if (windowType == GAUSS)
					{
						alpha[n] = (type)(normal * Fourier.ct(0, T / 2, integration_interval));
						alphaSum += alpha[n];
					}
					else if (windowType == S_TONE)
					{
						beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					}
					else if (windowType == HAT)
					{
						beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					}
					else if (windowType == SMOOTH_HAT)
					{
						beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					}
				}
			}

			for (int n = 1; n < order + 1; ++n)
			{
				alpha[n] /= alphaSum;
			}
			double mse = 0.0;

			std::vector<type> cosTable(order + 1);
			std::vector<type> sinTable(order + 1);

			const type t = (type)0.0;
			//const type t = (type)(Irange * 0.5 + Imin);
			//const type t = (type)(Imax); 
			//const type t = (type)((Imax - Imin) * 0.5 + Imin);
			//const type t = (type)(Imin);

			for (int n = 1; n < order + 1; n++)
			{
				cosTable[n] = (type)cos(omega * n * t);
				sinTable[n] = (type)sin(omega * n * t);
			}
#if 1
			for (int i = 0; i <= Irange; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)(1.0 / alphaSum);
				for (int n = 1; n < order + 1; n++)
				{
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * cos(omegai * n) + sinTable[n] * sin(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					type we = ideal[(int)abs(i - t)];
					double sub = double(wr - we);
					mse += sub * sub;
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
#else
			cp::Plot pt;
			pt.setPlotTitle(0, "ideal");
			pt.setPlotTitle(1, "approx");
			pt.setPlotTitle(2, "diff");

			print_debug(alphaSum);

			for (int i = 0; i <= 255; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)(1.0 / alphaSum);
				for (int n = 1; n < cbf_order + 1; n++)
				{
					//if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * sin(omegai * n) - sinTable[n] * cos(omegai * n));
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * cos(omegai * n) + sinTable[n] * sin(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					//type we = (i - t) * ideal[abs(i - t)];
					type we = ideal[abs(i - t)];
					pt.push_back(i, we, 0);
					pt.push_back(i, wr, 1);
					pt.push_back(i, (we - wr) * 20, 2);

					if (i <= Irange)
						mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
			print_debug4(Imin, Imax, T, sqrt(mse));
			pt.plot();
#endif

			//mse = sqrt(mse);
			return float(mse);
		}

		template <typename type, int windowType>
		float getArbitraryRangeKernelErrorOdd(const type T, const int cbf_order, const type sigma_range, const int Imin, const int Imax, std::vector<type>& ideal, const int integration_interval, int window_type,
			float Salpha, float Sbeta, float Ssigma)
		{
			std::vector<type> alpha(cbf_order + 1);
			std::vector<type> beta(cbf_order + 1);
			const type omega = (float)(CV_2PI / T);

			type alphaSum = 0.f;
			for (int n = 1; n < cbf_order + 1; n++)
			{
				FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, Ssigma, n, windowType);
				if (windowType == GAUSS)
				{
					alpha[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
					alphaSum += alpha[n];
				}
				else if (windowType == S_TONE)
				{
					beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
				}
				else if (windowType == HAT)
				{
					beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
				}
				else if (windowType == SMOOTH_HAT)
				{
					beta[n] = (type)(4.0 * Fourier.st(0, T / 2, integration_interval) / T);
				}
			}

			std::vector<type> cosTable(cbf_order + 1);
			std::vector<type> sinTable(cbf_order + 1);
			const int t = 0;
			//const type t = (type)(Irange * 0.5 + Imin);
			//const type t = (type)(Imax); 
			//const type t = (type)(Imin);

			for (int n = 1; n < cbf_order + 1; n++)
			{
				cosTable[n] = (type)cos(omega * n * t);
				sinTable[n] = (type)sin(omega * n * t);
			}

			type mse = (type)0.0;

#if 1
			for (int i = 0; i <= Irange; i++)
				//for (int i = 0; i <= 1; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)0.0;
				for (int n = 1; n < cbf_order + 1; n++)
				{
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * sin(omegai * n) - sinTable[n] * cos(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					type we = (i - t) * ideal[abs(i - t)];
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
#else
			cp::Plot pt;
			pt.setPlotTitle(0, "ideal");
			pt.setPlotTitle(1, "approx");
			for (int i = 0; i <= 255; i++)
			{
				const type omegai = (type)(omega * i);
				type s = (type)(0.0);
				for (int n = 1; n < cbf_order + 1; n++)
				{
					if constexpr (windowType == GAUSS) s += alpha[n] * (cosTable[n] * sin(omegai * n) - sinTable[n] * cos(omegai * n));
					else if constexpr (windowType == S_TONE) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
					else if constexpr (windowType == SMOOTH_HAT) s += beta[n] * (sin(omegai * n) * cosTable[n] - cos(omegai * n) * sinTable[n]);
				}

				if constexpr (windowType == GAUSS)
				{
					type wr = s;
					type we = (i - t) * ideal[abs(i - t)];
					pt.push_back(i, we, 0);
					pt.push_back(i, wr, 1);

					if (i <= Irange)
						mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == S_TONE)
				{
					type wr = s + Imin;
					type we = getSToneCurve<float>((float)i, float(Imin), Ssigma, Sbeta, Salpha);
					mse += (type)((we - wr) * (we - wr));
				}
				else if constexpr (windowType == HAT)
				{
					type wr = s + Imin;
					type we = (type)((i - Imin) * std::max(0.0, 1.0 - abs((i - Imin) / sigma_range)));
					mse += (we - wr) * (we - wr);
				}
				else if constexpr (windowType == SMOOTH_HAT)
				{
					type wr = s + Imin;
					//we = (i - Imin) * getSmoothingHat(i, Imin, sigma_range, 5);
					type we = getSmoothingHat(float(i), float(Imin), sigma_range, 5);
					mse += (we - wr) * (we - wr);
				}
			}
			print_debug4(Imin, Imax, T, sqrt(mse));
			pt.plot();
#endif

			//mse = sqrt(mse);
			return mse;
		}

		float getError(float a)
		{
			float ret = 0.f;
			const bool isEven = true;
			if (isEven)
			{
				if (windowType == GAUSS) ret = getArbitraryRangeKernelErrorEven<float, GAUSS>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == S_TONE) ret = getArbitraryRangeKernelErrorEven<float, S_TONE>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == HAT) ret = getArbitraryRangeKernelErrorEven<float, HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == SMOOTH_HAT) ret = getArbitraryRangeKernelErrorEven<float, SMOOTH_HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
			}
			else
			{
				if (windowType == GAUSS) ret = getArbitraryRangeKernelErrorOdd<float, GAUSS>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == S_TONE) ret = getArbitraryRangeKernelErrorOdd<float, S_TONE>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == HAT) ret = getArbitraryRangeKernelErrorOdd<float, HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
				if (windowType == SMOOTH_HAT) ret = getArbitraryRangeKernelErrorOdd<float, SMOOTH_HAT>(float(Irange) * a, cbf_order, sigma_range, Imin, Imax, ideal, integration_interval, window_type, Salpha, Sbeta, Ssigma);
			}
			return ret;
		}

	public:
		ComputeCompressiveT_Fast32F(int Irange, int Imin, int Imax, int cbf_order, float sigma_range, int integration_interval, int window_type, float Salpha, float Sbeta, float Ssigma, int windowType)
		{
			this->Irange = Irange;
			this->cbf_order = cbf_order;
			this->sigma_range = sigma_range;
			this->Imin = Imin;
			this->Imax = Imax;
			this->integration_interval = integration_interval;
			this->window_type = window_type;
			this->Salpha = Salpha;
			this->Sbeta = Sbeta;
			this->Ssigma = Ssigma;
			this->windowType = windowType;

#if 1
			ideal.resize(256);
#else
			ideal.resize(Irange + 1);
#endif
			for (int i = 0; i < ideal.size(); i++)
				//for (int i = 0; i < 256; i++)
			{
				ideal[i] = (float)getRangeKernelFunction(double(i), sigma_range, window_type);
			}
		}
	};

	double getOptimalT_32F(const int cbf_order, const double sigma_range, const int Imin, const int Imax, float Salpha, float Sbeta, float Ssigma, int windowType,
		int integral_interval = 100, double serch_T_min = 0.0, double serch_T_max = 5.0, double search_T_diff_min = 0.01, double search_iteration_max = 20)
	{
		const int Irange = Imax - Imin;

		ComputeCompressiveT_Fast32F ct(Irange, Imin, Imax, cbf_order, (float)sigma_range, integral_interval, windowType, Salpha, Sbeta, Ssigma, windowType);
		double ret = (double)ct.goldenSectionSearch((float)serch_T_min, (float)serch_T_max, (float)search_T_diff_min, (int)search_iteration_max);
		//double ret = (double)ct.linearSearch(serch_T_min, serch_T_max, 0.001);
		return ret;
	}

	inline double dfCompressive(double x, const int K, const double Irange, const double sigma_range)
	{
		const double s = sigma_range / Irange;
		const double kappa = (2 * K + 1) * CV_PI;
		const double psi = kappa * s / x;
		const double phi = (x - 1.0) / s;
		return (-kappa * exp(-phi * phi) + psi * psi * exp(-psi * psi));
	}

	double computeCompressiveT_ClosedForm(int order, double sigma_range, const double intensityRange)
	{
		double x, diff;

		double x1 = 1.0, x2 = 15.0;
		int loop = 20;
		for (int i = 0; i < loop; ++i)
		{
			x = (x1 + x2) / 2.0;
			diff = dfCompressive(x, order, intensityRange, sigma_range);
			((0.0 <= diff) ? x2 : x1) = x;
		}
		return x;
	}

	inline __m256 getAdaptiveAlpha(__m256 coeff, __m256 base, __m256 sigma, __m256 boost)
	{
		__m256 a = _mm256_mul_ps(coeff, sigma);
		a = _mm256_exp_ps((_mm256_mul_ps(_mm256_set1_ps(-0.5), _mm256_mul_ps(a, a))));
		return _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(sigma, sigma), sigma), boost), base), a);
	}

	inline float getAdaptiveAlpha(float coeff, float base, float sigma, float boost)
	{
		float a = coeff * sigma;
		a = exp(-0.5f * a * a);
		return sigma * sigma * sigma * boost * base * a;
	}
#pragma endregion

#pragma region LocalMultiScaleFilterFourierReference

	void LocalMultiScaleFilterFourierReference::initRangeFourier(const int order, const float sigma_range, const float boost)
	{
		if (alpha.size() != order)
		{
			alpha.resize(order);
			beta.resize(order);
		}

		if (omega.size() != order) omega.resize(order);

		T = float(intensityRange * computeCompressiveT_ClosedForm(order, sigma_range, intensityRange));
		//T = intensityRange * getOptimalT_32F(1.0, order, sigma_range, 0, (int)intensityRange, Salpha, Sbeta, sigma_range, windowType, 100, 0, 5.0, 0.001, 20, 0);

		for (int k = 0; k < order; k++)
		{
			omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
			const double coeff_kT = omega[k] * sigma_range;
			alpha[k] = float(2.0 * exp(-0.5 * coeff_kT * coeff_kT) * sqrt(CV_2PI) * sigma_range / T);
		}
	}

	void LocalMultiScaleFilterFourierReference::remapCos(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);

		if (isSIMD)
		{
			const __m256 momega = _mm256_set1_ps(omega);
			const __m256* msrc = (__m256*)src.ptr<float>();
			__m256* mdest = (__m256*)dest.ptr<float>();
			const int SIZE = src.size().area() / 8;

			for (int i = 0; i < SIZE; i++)
			{
				*(mdest++) = _mm256_cos_ps(_mm256_mul_ps(momega, *msrc++));
			}
		}
		else
		{
			const float* msrc = src.ptr<float>();
			float* mdest = dest.ptr<float>();
			const int SIZE = src.size().area();

			for (int i = 0; i < SIZE; i++)
			{
				mdest[i] = cos(omega * msrc[i]);
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::remapSin(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);

		if (isSIMD)
		{
			const __m256 momega = _mm256_set1_ps(omega);
			const __m256* msrc = (__m256*)src.ptr<float>();
			__m256* mdest = (__m256*)dest.ptr<float>();
			const int SIZE = src.size().area() / 8;

			for (int i = 0; i < SIZE; i++)
			{
				*(mdest++) = _mm256_sin_ps(_mm256_mul_ps(momega, *msrc++));
			}
		}
		else
		{
			const float* msrc = src.ptr<float>();
			float* mdest = dest.ptr<float>();
			const int SIZE = src.size().area();

			for (int i = 0; i < SIZE; i++)
			{
				mdest[i] = sin(omega * msrc[i]);
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::productSumPyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const float sigma, const float boost)
	{
		dest.create(srccos.size(), CV_32F);

		if (isSIMD)
		{
			const int SIZE = srccos.size().area() / 8;

			const __m256* scptr = (__m256*)srccos.ptr<float>();
			const __m256* ssptr = (__m256*)srcsin.ptr<float>();
			const __m256* gptr = (__m256*)gauss.ptr<float>();
			__m256* dptr = (__m256*)dest.ptr<float>();

			__m256 malpha = _mm256_set1_ps(-sigma * sigma * omega * alpha * boost);
			const __m256 momega_k = _mm256_set1_ps(omega);
			for (int i = 0; i < SIZE; i++)
			{
				const __m256 ms = _mm256_mul_ps(momega_k, *gptr++);
				const __m256 msin = _mm256_sin_ps(ms);
				const __m256 mcos = _mm256_cos_ps(ms);
				*(dptr) = _mm256_fmadd_ps(malpha, _mm256_fmsub_ps(msin, *(scptr++), _mm256_mul_ps(mcos, *(ssptr++))), *(dptr));
				dptr++;
			}
		}
		else
		{
			const int SIZE = srccos.size().area();

			const float* cosptr = srccos.ptr<float>();
			const float* sinptr = srcsin.ptr<float>();
			const float* gptr = gauss.ptr<float>();
			float* dptr = dest.ptr<float>();

			const float lalpha = -sigma * sigma * omega * alpha * boost;
			for (int i = 0; i < SIZE; i++)
			{
				const float ms = omega * gptr[i];
				dptr[i] += lalpha * (sin(ms) * cosptr[i] - cos(ms) * (sinptr[i]));
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::productSumAdaptivePyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const Mat& sigma, const Mat& boost)
	{
		dest.create(srccos.size(), CV_32F);

		if (isSIMD)
		{
			const int SIZE = srccos.size().area() / 8;

			const __m256* scptr = (__m256*)srccos.ptr<float>();
			const __m256* ssptr = (__m256*)srcsin.ptr<float>();
			const __m256* gptr = (__m256*)gauss.ptr<float>();
			__m256* dptr = (__m256*)dest.ptr<float>();

			const float base = -float(2.0 * sqrt(CV_2PI) * omega / T);
			const __m256 mbase = _mm256_set1_ps(base);//for adaptive
			__m256* adaptiveSigma = (__m256*)sigma.ptr<float>();
			__m256* adaptiveBoost = (__m256*)boost.ptr<float>();
			const __m256 momega_k = _mm256_set1_ps(omega);

			for (int i = 0; i < SIZE; i++)
			{
				__m256 malpha = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma++, *adaptiveBoost++);
				const __m256 ms = _mm256_mul_ps(momega_k, *gptr++);
				const __m256 msin = _mm256_sin_ps(ms);
				const __m256 mcos = _mm256_cos_ps(ms);
				*(dptr) = _mm256_fmadd_ps(malpha, _mm256_fmsub_ps(msin, *(scptr++), _mm256_mul_ps(mcos, *(ssptr++))), *(dptr));
				dptr++;
			}
		}
		else
		{
			const int SIZE = srccos.size().area();

			const float* scptr = srccos.ptr<float>();
			const float* ssptr = srcsin.ptr<float>();
			const float* gptr = gauss.ptr<float>();
			float* dptr = dest.ptr<float>();

			const float base = -float(2.0 * sqrt(CV_2PI) * omega / T);
			const float* adaptiveSigma = sigma.ptr<float>();
			const float* adaptiveBoost = boost.ptr<float>();

			for (int i = 0; i < SIZE; i++)
			{
				float lalpha = getAdaptiveAlpha(omega, base, adaptiveSigma[i], adaptiveBoost[i]);
				const float ms = omega * gptr[i];
				dptr[i] += lalpha * (sin(ms) * scptr[i] - cos(ms) * (ssptr[i]));
			}
		}
	}

	void LocalMultiScaleFilterFourierReference::pyramid(const Mat& src, Mat& dest)
	{
		pyramidComputeMethod = PyramidComputeMethod::Full;

		//alloc buffer
		GaussianPyramid.resize(level + 1);
		LaplacianPyramid.resize(level + 1);
		FourierPyramidCos.resize(level + 1);
		FourierPyramidSin.resize(level + 1);
		if (src.depth() == CV_8U) src.convertTo(GaussianPyramid[0], CV_32F);
		else src.copyTo(GaussianPyramid[0]);
		layerSize.resize(level + 1);
		for (int l = 0; l < level + 1; l++) layerSize[l] = GaussianPyramid[l].size();

		//compute alpha, omega, T
		initRangeFourier(order, sigma_range, boost);

		//Build Gaussian Pyramid
		buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);

		//Build Laplacian Pyramid for DC
		buildLaplacianPyramid(GaussianPyramid, LaplacianPyramid, level, sigma_space);
		for (int k = 0; k < order; k++)
		{
			remapCos(src, FourierPyramidCos[0], omega[k]);
			remapSin(src, FourierPyramidSin[0], omega[k]);

			buildLaplacianPyramid(FourierPyramidCos[0], FourierPyramidCos, level, sigma_space);
			buildLaplacianPyramid(FourierPyramidSin[0], FourierPyramidSin, level, sigma_space);

			if (adaptiveMethod == AdaptiveMethod::FIX)
			{
				for (int l = 0; l < level; l++)
				{
					productSumPyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], sigma_range, boost);
				}
			}
			else
			{
				for (int l = 0; l < level; l++)
				{
					productSumAdaptivePyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], adaptiveSigmaMap[l], adaptiveBoostMap[l]);
				}
			}
		}
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//collapse Laplacian Pyramid
		collapseLaplacianPyramid(LaplacianPyramid, dest, src.depth());
	}

	void LocalMultiScaleFilterFourierReference::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		allocSpaceWeight(sigma_space);

		this->order = order;

		this->sigma_space = sigma_space;
		this->level = max(level, 1);
		this->sigma_range = sigma_range;
		this->boost = boost;

		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);

		freeSpaceWeight();
	}

#pragma endregion

#pragma region LocalMultiScaleFilterFourier
	LocalMultiScaleFilterFourier::~LocalMultiScaleFilterFourier()
	{
		_mm_free(sinTable);
		_mm_free(cosTable);
	}

	template<typename Type>
	void LocalMultiScaleFilterFourier::kernelPlot(const int window_type, const int order, const int R, const double boost, const double sigma_range, float Salpha, float Sbeta, float Ssigma, const int Imin, const int Imax, const int Irange, const Type T,
		Type* sinTable, Type* cosTable, std::vector<Type>& alpha, std::vector<Type>& beta, int windowType, const std::string wname, const cv::Size windowSize)
	{
		cp::Plot pt(windowSize);
		pt.setPlotTitle(0, "Ideal");
		pt.setPlotTitle(1, "y=xf(x)");
		pt.setPlotTitle(2, "y=x");
		pt.setPlotTitle(3, "y=boost*x");
		pt.setPlotSymbolALL(0);
		pt.setPlotLineWidthALL(2);
		pt.setKey(cp::Plot::KEY::LEFT_TOP);

		namedWindow(wname);
		createTrackbar("t", wname, &kernel_plotting_t, 255);
		createTrackbar("amp_pow", wname, &kernel_plotting_amp, 100);
		const int t = kernel_plotting_t;
		double error = 0.0;

		if (kernel_plotting_amp != 0) pt.setPlotTitle(4, "diff");

		for (int s = 0; s <= R; s++)
		{
			Type wr = (Type)0.0;

			for (int k = 0; k < order; ++k)
			{
				float* ct = &cosTable[256 * k];
				float* st = &sinTable[256 * k];
				double omega = CV_2PI / T * (double)(k + 1);
				const double lalpha = sigma_range * sigma_range * omega * alpha[k] * boost;

				switch (windowType)
				{
				case GAUSS:
					wr += Type(lalpha * (st[s] * ct[t] - ct[s] * st[t])); break;
				case S_TONE:
					wr += Type(beta[k] * (sinTable[256 * k + s] * cosTable[256 * k + t] - cosTable[256 * k + s] * sinTable[256 * k + t])); break;
				case HAT:
					wr += Type(alpha[k] * (sinTable[256 * k + s] * cosTable[256 * k + t] - cosTable[256 * k + s] * sinTable[256 * k + t])); break;
				case SMOOTH_HAT:
					wr += Type(alpha[k] * (sinTable[256 * k + s] * cosTable[256 * k + t] - cosTable[256 * k + s] * sinTable[256 * k + t])); break;
				}
			}

			double ideal_value = 0.0;
			double apprx_value = 0.0;
			switch (windowType)
			{
			case GAUSS:
				ideal_value = s + double(s - t) * boost * getRangeKernelFunction(double(s - t), sigma_range, window_type);
				apprx_value = s + double(wr);
				break;

			case S_TONE:
				ideal_value = getSToneCurve(float(s), float(t), Ssigma, Sbeta, Salpha);
				apprx_value = t + wr;
				pt.push_back(s, t + wr, 1);
				break;
			case HAT:
				ideal_value = s + (s - t) * std::max(0.0, 1.0 - abs((float)(s - t) / sigma_range));
				apprx_value = s + wr;
				break;
			case SMOOTH_HAT:
				//v = s + (s - t) * getSmoothingHat(s, t, sigma_range, 5);
				ideal_value = s + getSmoothingHat(float(s), float(t), float(sigma_range), 10);
				apprx_value = s + wr;
				break;
			}

			pt.push_back(s, ideal_value, 0);//ideal
			pt.push_back(s, apprx_value, 1);//approx
			pt.push_back(s, s, 2);
			pt.push_back(s, boost * (s - t) + s, 3);
			error += (ideal_value - apprx_value) * (ideal_value - apprx_value);
			if (kernel_plotting_amp != 0) pt.push_back(s, (ideal_value - apprx_value) * pow(10.0, kernel_plotting_amp), 4);
		}
		error = 20.0 * std::log10(R / sqrt(error / (R + 1)));

		//pt.setYRange(0, 1);
		pt.setXRange(0, R + 1);
		pt.setYRange(-128, 256 + 128);
		pt.setIsDrawMousePosition(true);
		pt.setGrid(2);
		//pt.plot(wname, false, "", format("err. total: %2.2f, err. current: %2.2f, (min,max)=(%d  %d)", error, errort, Imin, Imax));
		pt.plot(wname, false, "", cv::format("Kernel Err: %6.2lf dB (min,max)=(%d, %d)", error, Imin, Imax));
	}

	void LocalMultiScaleFilterFourier::initRangeFourier(const int order, const float sigma_range, const float boost)
	{
		bool isRecompute = true;
		if (preorder == order &&
			presigma_range == sigma_range &&
			predetail_param == boost &&
			preIntensityMin == intensityMin &&
			preIntensityRange == intensityRange &&
			preperiodMethod == periodMethod)
			isRecompute = false;
		if (!isRecompute)return;

		//recomputing flag setting
		preorder = order;
		presigma_range = sigma_range;
		predetail_param = boost;
		preIntensityMin = intensityMin;
		preIntensityRange = intensityRange;
		preperiodMethod = periodMethod;

		//alloc
		if (alpha.size() != order)
		{
			alpha.resize(order);
			beta.resize(order);
		}
		if (omega.size() != order) omega.resize(order);

		//compute T
		//static int rangeMax = 255;//if use opt
		//int rangeMax = 255;
		switch (periodMethod)
		{
			//Using the derivative of a Gaussian function
		case GAUSS_DIFF:
			//cout << "Gauss Diff" << endl;
			T = float(intensityRange * computeCompressiveT_ClosedForm(order, sigma_range, intensityRange));
			break;
			//minimizing the squared error
		case OPTIMIZE:
			//cout << "Optimize" << endl;
			T = float(intensityRange * getOptimalT_32F(order, sigma_range, (int)intensityMin, (int)intensityMax, Salpha, Sbeta, sigma_range, windowType, 100, 1.0, 20.0, 0.001, 20));
			break;
		case PRE_SET:
			//static int T_ = 4639;//4639, 1547, 3093, 6186
			int T_ = 4639;//4639, 1547, 3093, 6186
			//cv::createTrackbar("T", "", &T_, 20000);
			T = T_ * 0.1f;
			break;
		}
		//cout << "T : " << T << endl;

		//compute omega and alpha
		if (periodMethod == GAUSS_DIFF)
		{
			for (int k = 0; k < order; k++)
			{
				omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
				const double coeff_kT = omega[k] * sigma_range;
				switch (windowType)
				{
				case GAUSS:
					alpha[k] = float(2.0 * exp(-0.5 * coeff_kT * coeff_kT) * sqrt(CV_2PI) * sigma_range / T);
					//alpha[k] = exp(-0.5 * coeff_kT * coeff_kT);
					break;

				default:
				{
					FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, sigma_range, k + 1, windowType);
					alpha[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					beta[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					break;
				}
				}
			}
		}
		else
		{
			FourierDecomposition Fourier(T, sigma_range, 0, 0, 0, 0, windowType);
			double a0 = 2.0 / Fourier.init(0, T / 2, 100);
			double alphaSum = a0;
			for (int k = 0; k < order; k++)
			{
				omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
				switch (windowType)
				{
				case GAUSS:
				{
					FourierDecomposition Fourier(T, sigma_range, 0, 0, 0, k + 1, windowType);
					alpha[k] = float(a0 * Fourier.st(0, T / 2, 100));
					alphaSum += double(alpha[k]);
					break;
				}
				default:
				{
					FourierDecomposition Fourier(T, sigma_range, Sbeta, Salpha, sigma_range, k + 1, windowType);
					alpha[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					beta[k] = float(4.0 * Fourier.st(0, T / 2, 100) / T);
					break;
				}
				}
			}

			alphaSum *= sqrt(CV_2PI);
			for (int k = 0; k < order; k++)
			{
				alpha[k] = float(alpha[k] / alphaSum);
			}
		}

		//compute cos/sin table
		if (isUseFourierTable0)
		{
			_mm_free(sinTable);
			_mm_free(cosTable);
			sinTable = (float*)_mm_malloc(sizeof(float) * FourierTableSize * order, AVX_ALIGN);
			cosTable = (float*)_mm_malloc(sizeof(float) * FourierTableSize * order, AVX_ALIGN);

			const int TABLESIZE = get_simd_floor(FourierTableSize, 8);
			const __m256 m8 = _mm256_set1_ps(8.f);
			for (int k = 0; k < order; k++)
			{
				float* sinptr = sinTable + FourierTableSize * k;
				float* cosptr = cosTable + FourierTableSize * k;

				const __m256 momega_k = _mm256_set1_ps(omega[k]);
				__m256 unit = _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
				for (int i = 0; i < TABLESIZE; i += 8)
				{
					__m256 base = _mm256_mul_ps(momega_k, unit);
					_mm256_store_ps(sinptr + i, _mm256_sin_ps(base));
					_mm256_store_ps(cosptr + i, _mm256_cos_ps(base));
					unit = _mm256_add_ps(unit, m8);
				}
				for (int i = TABLESIZE; i < FourierTableSize; i++)
				{
					sinptr[i] = sin(omega[k] * i);
					cosptr[i] = cos(omega[k] * i);
				}
			}
		}

		//cout << alphaSum << endl;
		if (false)//normalize test
		{
			float alphaSum = 1.0;
			for (int k = 0; k < order; k++)
			{
				alphaSum += alpha[k];
				//print_debug2(k, alpha[k]);
			}

			cout << "before " << alphaSum * sqrt(CV_2PI) * sigma_range / T << "," << T << endl;
			for (int k = 0; k < order; k++)
			{
				alpha[k] = alpha[k] / alphaSum;
			}
			alphaSum = 1.f / alphaSum;
			cout << "vv " << alphaSum << endl;
			for (int k = 0; k < order; k++)
			{
				alphaSum += alpha[k];
			}
			cout << "normal " << alphaSum << endl;
		}
	}

	void LocalMultiScaleFilterFourier::allocImageBuffer(const int order, const int level)
	{
		if (ImageStack.size() != level + 1)
		{
			ImageStack.resize(0);
			ImageStack.resize(level + 1);
			DetailStack.resize(level + 1);
		}

		if (isParallel)
		{
			const int OoT = max(order, threadMax);
			if (FourierStackCos.size() != OoT)
			{
				FourierStackCos.resize(OoT);
				FourierStackSin.resize(OoT);
				destEachOrder.resize(OoT);
				for (int i = 0; i < OoT; i++)
				{
					FourierStackCos[i].resize(level + 1);
					FourierStackSin[i].resize(level + 1);
					destEachOrder[i].resize(level);
				}
			}
		}
		else
		{
			if (FourierStackCos.size() != 1)
			{
				FourierStackCos.resize(1);
				FourierStackSin.resize(1);
				destEachOrder.resize(1);
				for (int i = 0; i < 1; i++)
				{
					destEachOrder[i].resize(level);
				}
			}
		}
	}

#pragma region pyramid
	template<bool isInit, bool adaptive_method, bool is_use_fourier_table0, bool is_use_fourier_table_level, int D, int D2>
	void LocalMultiScaleFilterFourier::buildLaplacianFourierPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		//const int D = 2 * radius + 1;
		//const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1) FourierPyramidCos.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1) FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* linebuffer = (float*)_mm_malloc(sizeof(float) * linesize * 4, AVX_ALIGN);;
		float* spcosline_e = linebuffer + 0 * linesize;
		float* spsinline_e = linebuffer + 1 * linesize;
		float* spcosline_o = linebuffer + 2 * linesize;
		float* spsinline_o = linebuffer + 3 * linesize;

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 meven_ratio = _mm256_set1_ps(evenratio);
		const __m256 modd__ratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			//l=0
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZEREMAP = width * (D - 1) / 8;
				const int SIZEREMAP32 = SIZEREMAP / 4;
				const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;
				if (isUseFourierTable0)
				{
#ifdef USE_GATHER8U
					const __m64* sptr = (__m64*)(src8u.ptr<uchar>());
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#else
					const __m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>();
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						__m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#endif
				}
				else
				{
					__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
					for (int i = 0; i < SIZEREMAP32; i++)
					{
						__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
					for (int i = 0; i < SIZEREMAP8; i++)
					{
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH32 = get_simd_floor(width, 32);
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND32 = get_simd_floor(hend, 32);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL32 = get_simd_floor(hendl, 32);
			const int HENDL = get_simd_floor(hendl, 8);

			const int SIZEREMAP = 2 * width / 8;
			const int SIZEREMAP32 = SIZEREMAP / 4;
			const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					if (isUseFourierTable0)
					{
#ifdef USE_GATHER8U
						const __m64* sptr = (__m64*)(src8u.ptr<uchar>(j + D - 1));
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; ++i)
						{
							__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#else
						__m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>(j + D - 1);
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; i++)
						{
							const __m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#endif
					}
					else
					{
						__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}

				//v filter
				__m256* spc = (__m256*)spcosline_e;
				__m256* sps = (__m256*)spsinline_e;
				for (int i = 0; i < WIDTH32; i += 32)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumc1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumc2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumc3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sums0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sums1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sums2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sums3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					ss += width;
					sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc0);
						sumc1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 8), sumc1);
						sumc2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 16), sumc2);
						sumc3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 24), sumc3);
						sums0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums0);
						sums1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 8), sums1);
						sums2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 16), sums2);
						sums3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 24), sums3);
						sc += width;
						ss += width;
					}
					*spc++ = sumc0;
					*spc++ = sumc1;
					*spc++ = sumc2;
					*spc++ = sumc3;
					*sps++ = sums0;
					*sps++ = sums1;
					*sps++ = sums2;
					*sps++ = sums3;
				}
				for (int i = WIDTH32; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					*spc++ = sumc;
					*sps++ = sums;
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND32; i += 32)
				{
					float* cosi0 = spcosline_e + i;
					float* sini0 = spsinline_e + i;
					float* cosi1 = spcosline_e + i + 8;
					float* sini1 = spsinline_e + i + 8;
					float* cosi2 = spcosline_e + i + 16;
					float* sini2 = spsinline_e + i + 16;
					float* cosi3 = spcosline_e + i + 24;
					float* sini3 = spsinline_e + i + 24;
					__m256 sum0 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum1 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum2 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum3 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum0 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)), sum0);
						sum1 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)), sum1);
						sum2 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)), sum2);
						sum3 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)), sum3);
					}
					sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum0));

					sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));
					_mm_storeu_ps(dfpyn_sin + ((i + 8) >> 1), _mm256_castps256hi_ps128(sum1));

					sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));
					_mm_storeu_ps(dfpyn_sin + ((i + 16) >> 1), _mm256_castps256hi_ps128(sum2));

					sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
					_mm_storeu_ps(dfpyn_sin + ((i + 24) >> 1), _mm256_castps256hi_ps128(sum3));
				}
				for (int i = HEND32; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else 
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (is_use_fourier_table0)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				__m256* spce = (__m256*)(spcosline_e + rs);
				__m256* spco = (__m256*)(spcosline_o + rs);
				__m256* spse = (__m256*)(spsinline_e + rs);
				__m256* spso = (__m256*)(spsinline_o + rs);
				for (int i = 0; i < HENDL32; i += 32)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumse0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sumco0 = _mm256_setzero_ps();
					__m256 sumso0 = _mm256_setzero_ps();
					__m256 sumce1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumse1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sumco1 = _mm256_setzero_ps();
					__m256 sumso1 = _mm256_setzero_ps();
					__m256 sumce2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumse2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sumco2 = _mm256_setzero_ps();
					__m256 sumso2 = _mm256_setzero_ps();
					__m256 sumce3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sumse3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					__m256 sumco3 = _mm256_setzero_ps();
					__m256 sumso3 = _mm256_setzero_ps();
					ss += widths;
					sc += widths;
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						__m256 msc = _mm256_loadu_ps(sc);
						sumce0 = _mm256_fmadd_ps(W[m], msc, sumce0);
						sumco0 = _mm256_fmadd_ps(W[m - 1], msc, sumco0);
						msc = _mm256_loadu_ps(sc + 8);
						sumce1 = _mm256_fmadd_ps(W[m], msc, sumce1);
						sumco1 = _mm256_fmadd_ps(W[m - 1], msc, sumco1);
						msc = _mm256_loadu_ps(sc + 16);
						sumce2 = _mm256_fmadd_ps(W[m], msc, sumce2);
						sumco2 = _mm256_fmadd_ps(W[m - 1], msc, sumco2);
						msc = _mm256_loadu_ps(sc + 24);
						sumce3 = _mm256_fmadd_ps(W[m], msc, sumce3);
						sumco3 = _mm256_fmadd_ps(W[m - 1], msc, sumco3);
						sc += widths;
						//sin
						__m256 mss = _mm256_loadu_ps(ss);
						sumse0 = _mm256_fmadd_ps(W[m], mss, sumse0);
						sumso0 = _mm256_fmadd_ps(W[m - 1], mss, sumso0);
						mss = _mm256_loadu_ps(ss + 8);
						sumse1 = _mm256_fmadd_ps(W[m], mss, sumse1);
						sumso1 = _mm256_fmadd_ps(W[m - 1], mss, sumso1);
						mss = _mm256_loadu_ps(ss + 16);
						sumse2 = _mm256_fmadd_ps(W[m], mss, sumse2);
						sumso2 = _mm256_fmadd_ps(W[m - 1], mss, sumso2);
						mss = _mm256_loadu_ps(ss + 24);
						sumse3 = _mm256_fmadd_ps(W[m], mss, sumse3);
						sumso3 = _mm256_fmadd_ps(W[m - 1], mss, sumso3);
						ss += widths;
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce0);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce1);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce2);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce3);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco0);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco1);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco2);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco3);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse0);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse1);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse2);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse3);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso0);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso1);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso2);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso3);
				}
				for (int i = HENDL32; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso);
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spcosline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spsinline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					else
					{
						dste[I] += alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					else
					{
						dste[I + 1] += alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					else
					{
						dsto[I] += alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);
			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (is_use_fourier_table_level)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(modd__ratio, sumso));
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spsinline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spcosline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}
					const int I = i << 1;
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}

				}
#ifdef MASKSTORE
				if (hendl != HENDL)//last
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				ppye_cos += 2 * width;
				ppye_sin += 2 * width;
				ppyo_cos += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(linebuffer);
		_mm_free(W);
	}

	template<bool isInit, bool adaptive_method, bool is_use_fourier_table0, bool is_use_fourier_table_level>
	void LocalMultiScaleFilterFourier::buildLaplacianFourierPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		const int D = 2 * radius + 1;
		const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1)FourierPyramidCos.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1)FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* linebuffer = (float*)_mm_malloc(sizeof(float) * linesize * 4, AVX_ALIGN);;
		float* spcosline_e = linebuffer + 0 * linesize;
		float* spsinline_e = linebuffer + 1 * linesize;
		float* spcosline_o = linebuffer + 2 * linesize;
		float* spsinline_o = linebuffer + 3 * linesize;

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 meven_ratio = _mm256_set1_ps(evenratio);
		const __m256 modd__ratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			//l=0
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZEREMAP = width * (D - 1) / 8;
				const int SIZEREMAP32 = SIZEREMAP / 4;
				const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;
				if (isUseFourierTable0)
				{
#ifdef USE_GATHER8U
					const __m64* sptr = (__m64*)(src8u.ptr<uchar>());
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#else
					const __m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>();
					const float* stable = &sinTable[FourierTableSize * k];
					const float* ctable = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZEREMAP32; ++i)
					{
						__m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

						idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					for (int i = 0; i < SIZEREMAP8; ++i)
					{
						const __m256i idx = _mm256_cvtps_epi32(*gptr++);
						*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
						*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
#endif
				}
				else
				{
					__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
					for (int i = 0; i < SIZEREMAP32; i++)
					{
						__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);

						ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
					for (int i = 0; i < SIZEREMAP8; i++)
					{
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH32 = get_simd_floor(width, 32);
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND32 = get_simd_floor(hend, 32);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL32 = get_simd_floor(hendl, 32);
			const int HENDL = get_simd_floor(hendl, 8);

			const int SIZEREMAP = 2 * width / 8;
			const int SIZEREMAP32 = SIZEREMAP / 4;
			const int SIZEREMAP8 = (SIZEREMAP * 8 - SIZEREMAP32 * 32) / 8;

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					if (isUseFourierTable0)
					{
#ifdef USE_GATHER8U
						const __m64* sptr = (__m64*)(src8u.ptr<uchar>(j + D - 1));
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; ++i)
						{
							__m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)(sptr++));
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#else
						__m256* gptr = (__m256*)GaussianPyramid[0].ptr<float>(j + D - 1);
						const float* stable = &sinTable[FourierTableSize * k];
						const float* ctable = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));

							idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
						for (int i = 0; i < SIZEREMAP8; i++)
						{
							const __m256i idx = _mm256_cvtps_epi32(*gptr++);
							*(splatCos++) = _mm256_i32gather_ps(ctable, idx, sizeof(float));
							*(splatSin++) = _mm256_i32gather_ps(stable, idx, sizeof(float));
						}
#endif
					}
					else
					{
						__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
						for (int i = 0; i < SIZEREMAP32; i++)
						{
							__m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);

							ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
						for (int i = 0; i < SIZEREMAP8; ++i)
						{
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}

				//v filter
				__m256* spc = (__m256*)spcosline_e;
				__m256* sps = (__m256*)spsinline_e;
				for (int i = 0; i < WIDTH32; i += 32)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumc1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumc2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumc3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sums0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sums1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sums2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sums3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					ss += width;
					sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc0);
						sumc1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 8), sumc1);
						sumc2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 16), sumc2);
						sumc3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc + 24), sumc3);
						sums0 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums0);
						sums1 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 8), sums1);
						sums2 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 16), sums2);
						sums3 = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss + 24), sums3);
						sc += width;
						ss += width;
					}
					*spc++ = sumc0;
					*spc++ = sumc1;
					*spc++ = sumc2;
					*spc++ = sumc3;
					*sps++ = sums0;
					*sps++ = sums1;
					*sps++ = sums2;
					*sps++ = sums3;
				}
				for (int i = WIDTH32; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					*spc++ = sumc;
					*sps++ = sums;
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND32; i += 32)
				{
					float* cosi0 = spcosline_e + i;
					float* sini0 = spsinline_e + i;
					float* cosi1 = spcosline_e + i + 8;
					float* sini1 = spsinline_e + i + 8;
					float* cosi2 = spcosline_e + i + 16;
					float* sini2 = spsinline_e + i + 16;
					float* cosi3 = spcosline_e + i + 24;
					float* sini3 = spsinline_e + i + 24;
					__m256 sum0 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum1 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum2 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)));
					__m256 sum3 = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum0 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi0++), _mm256_loadu_ps(sini0++), _MM_SHUFFLE(2, 0, 2, 0)), sum0);
						sum1 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi1++), _mm256_loadu_ps(sini1++), _MM_SHUFFLE(2, 0, 2, 0)), sum1);
						sum2 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi2++), _mm256_loadu_ps(sini2++), _MM_SHUFFLE(2, 0, 2, 0)), sum2);
						sum3 = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi3++), _mm256_loadu_ps(sini3++), _MM_SHUFFLE(2, 0, 2, 0)), sum3);
					}
					sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum0));

					sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));
					_mm_storeu_ps(dfpyn_sin + ((i + 8) >> 1), _mm256_castps256hi_ps128(sum1));

					sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));
					_mm_storeu_ps(dfpyn_sin + ((i + 16) >> 1), _mm256_castps256hi_ps128(sum2));

					sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
					_mm_storeu_ps(dfpyn_sin + ((i + 24) >> 1), _mm256_castps256hi_ps128(sum3));
				}
				for (int i = HEND32; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else 
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (is_use_fourier_table0)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				__m256* spce = (__m256*)(spcosline_e + rs);
				__m256* spco = (__m256*)(spcosline_o + rs);
				__m256* spse = (__m256*)(spsinline_e + rs);
				__m256* spso = (__m256*)(spsinline_o + rs);
				for (int i = 0; i < HENDL32; i += 32)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc));
					__m256 sumse0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss));
					__m256 sumco0 = _mm256_setzero_ps();
					__m256 sumso0 = _mm256_setzero_ps();
					__m256 sumce1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 8));
					__m256 sumse1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 8));
					__m256 sumco1 = _mm256_setzero_ps();
					__m256 sumso1 = _mm256_setzero_ps();
					__m256 sumce2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 16));
					__m256 sumse2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 16));
					__m256 sumco2 = _mm256_setzero_ps();
					__m256 sumso2 = _mm256_setzero_ps();
					__m256 sumce3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc + 24));
					__m256 sumse3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss + 24));
					__m256 sumco3 = _mm256_setzero_ps();
					__m256 sumso3 = _mm256_setzero_ps();
					ss += widths;
					sc += widths;
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						__m256 msc = _mm256_loadu_ps(sc);
						sumce0 = _mm256_fmadd_ps(W[m], msc, sumce0);
						sumco0 = _mm256_fmadd_ps(W[m - 1], msc, sumco0);
						msc = _mm256_loadu_ps(sc + 8);
						sumce1 = _mm256_fmadd_ps(W[m], msc, sumce1);
						sumco1 = _mm256_fmadd_ps(W[m - 1], msc, sumco1);
						msc = _mm256_loadu_ps(sc + 16);
						sumce2 = _mm256_fmadd_ps(W[m], msc, sumce2);
						sumco2 = _mm256_fmadd_ps(W[m - 1], msc, sumco2);
						msc = _mm256_loadu_ps(sc + 24);
						sumce3 = _mm256_fmadd_ps(W[m], msc, sumce3);
						sumco3 = _mm256_fmadd_ps(W[m - 1], msc, sumco3);
						sc += widths;
						//sin
						__m256 mss = _mm256_loadu_ps(ss);
						sumse0 = _mm256_fmadd_ps(W[m], mss, sumse0);
						sumso0 = _mm256_fmadd_ps(W[m - 1], mss, sumso0);
						mss = _mm256_loadu_ps(ss + 8);
						sumse1 = _mm256_fmadd_ps(W[m], mss, sumse1);
						sumso1 = _mm256_fmadd_ps(W[m - 1], mss, sumso1);
						mss = _mm256_loadu_ps(ss + 16);
						sumse2 = _mm256_fmadd_ps(W[m], mss, sumse2);
						sumso2 = _mm256_fmadd_ps(W[m - 1], mss, sumso2);
						mss = _mm256_loadu_ps(ss + 24);
						sumse3 = _mm256_fmadd_ps(W[m], mss, sumse3);
						sumso3 = _mm256_fmadd_ps(W[m - 1], mss, sumso3);
						ss += widths;
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce0);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce1);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce2);
					*spce++ = _mm256_mul_ps(meven_ratio, sumce3);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco0);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco1);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco2);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco3);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse0);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse1);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse2);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse3);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso0);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso1);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso2);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso3);
				}
				for (int i = HENDL32; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					*spce++ = _mm256_mul_ps(meven_ratio, sumce);
					*spco++ = _mm256_mul_ps(modd__ratio, sumco);
					*spse++ = _mm256_mul_ps(meven_ratio, sumse);
					*spso++ = _mm256_mul_ps(modd__ratio, sumso);
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						//sin
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spcosline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_e + HENDL + rs, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spsinline_o + HENDL + rs, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#ifdef MASKSTORE
				//last 
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);
					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					else
					{
						dste[I] += alphak * evenratio * (sin(os) * sumcee - cos(os) * sumsee);
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					else
					{
						dste[I + 1] += alphak * oddratio * (sin(os) * sumcoe - cos(os) * sumsoe);
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					else
					{
						dsto[I] += alphak * evenratio * (sin(os) * sumceo - cos(os) * sumseo);
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * (sin(os) * sumcoo - cos(os) * sumsoo);
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			const __m128i maskhend = get_storemask1(hend, 8);
			const __m256i maskhendl = get_simd_residualmask_epi32(hendl);
			__m256i maskhendll, maskhendlr;
			get_storemask2(hendl, maskhendll, maskhendlr, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					const float* ss = sfpy_sin + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spcosline_e[i] = sumc;
					spsinline_e[i] = sums;
				}
				sfpy_cos += 2 * width;
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;
					float* sini = spsinline_e + i;
#if 1
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sum));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sum));
#else
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sumc = _mm256_shuffle_ps(sumc, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256hi_ps128(sumc));
#endif
				}
#ifdef MASKSTORE
				{
					float* cosi = spcosline_e + HEND;
					float* sini = spsinline_e + HEND;
					__m256 sum = _mm256_mul_ps(W[0], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)));
					for (int m = 1; m < D; m++)
					{
						sum = _mm256_fmadd_ps(W[m], _mm256_shuffle_ps(_mm256_loadu_ps(cosi++), _mm256_loadu_ps(sini++), _MM_SHUFFLE(2, 0, 2, 0)), sum);
					}
					sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_maskstore_ps(dfpyn_cos + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum));
					_mm_maskstore_ps(dfpyn_sin + (HEND >> 1), maskhend, _mm256_castps256hi_ps128(sum));
				}
#else
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
					dfpyn_sin[i >> 1] = sums;
				}
#endif
				dfpyn_cos += widths;
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			float* stable = nullptr;
			float* ctable = nullptr;
			if constexpr (is_use_fourier_table_level)
			{
				stable = &sinTable[FourierTableSize * k];
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(modd__ratio, sumso));
				}
#ifdef MASKSTORE
				{
					float* sc = sfpy_cos + HENDL;
					float* ss = sfpy_sin + HENDL;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumco = _mm256_setzero_ps();
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_maskstore_ps(spcosline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumce));
					_mm256_maskstore_ps(spsinline_e + rs + HENDL, maskhendl, _mm256_mul_ps(meven_ratio, sumse));
					_mm256_maskstore_ps(spcosline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumco));
					_mm256_maskstore_ps(spsinline_o + rs + HENDL, maskhendl, _mm256_mul_ps(modd__ratio, sumso));
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float* ss = sfpy_sin + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumco = 0.f;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumse += GaussWeight[m] * *ss;
						sumco += GaussWeight[m - 1] * *sc;
						sumso += GaussWeight[m - 1] * *ss;
						sc += widths;
						ss += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spsinline_e[i + rs] = sumse * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}
#endif

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}
					const int I = i << 1;
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}

				}
#ifdef MASKSTORE
				//last
				{
					float* cosie = spcosline_e + HENDL;
					float* cosio = spcosline_o + HENDL;
					float* sinie = spsinline_e + HENDL;
					float* sinio = spsinline_o + HENDL;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						//cos
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
						//sin
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = (HENDL << 1);
					//even line
					__m256 cos0, cos1, sin0, sin1;
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin, mcos;
					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dste + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I, maskhendll, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos0, _mm256_mul_ps(mcos, sin0)), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
						mcos = _mm256_cos_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_mul_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1))));
					}
					else
					{
						_mm256_maskstore_ps(dsto + I + 8, maskhendlr, _mm256_fmadd_ps(malpha_k, _mm256_fmsub_ps(msin, cos1, _mm256_mul_ps(mcos, sin1)), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
#else
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]) - cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]) - cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]) - cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]) - cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
#endif
				sfpy_cos += widths;
				sfpy_sin += widths;
				ppye_cos += 2 * width;
				ppye_sin += 2 * width;
				ppyo_cos += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(linebuffer);
		_mm_free(W);
	}




	template<bool isInit, bool adaptive_method, bool is_use_fourier_table0, bool is_use_fourier_table_level, int D, int D2>
	void LocalMultiScaleFilterFourier::buildLaplacianSinPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		//const int D = 2 * radius + 1;
		//const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1)FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spsinline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spsinline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = -sigma_range * sigma_range * omega[k] * alpha[k] * boost;//-alphak
		__m256 malpha_k = _mm256_set1_ps(alphak);//-alphak
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);//-alphak
		const float base = -float(2.0 * sqrt(CV_2PI) * omega[k] / T);//-base
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* sptr = (__m64*)src8u.ptr<uchar>();
					const float* sinptr = &sinTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)sptr++);
						*(splatSin++) = _mm256_i32gather_ps(sinptr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* sxiPtr = &sinTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatSin++) = _mm256_i32gather_ps(sxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* ctable = nullptr;
			if constexpr (is_use_fourier_table0)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 mcos;
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * cos(os) * sumsee;
					}
					else
					{
						dste[I] += alphak * evenratio * cos(os) * sumsee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * cos(os) * sumsoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * cos(os) * sumsoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * cos(os) * sumseo;
					}
					else
					{
						dsto[I] += alphak * evenratio * cos(os) * sumseo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * cos(os) * sumsoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * cos(os) * sumsoo;
					}
				}
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel

			float* ctable = nullptr;
			if constexpr (is_use_fourier_table_level)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;

					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 mcos;
					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
				sfpy_sin += widths;
				ppye_sin += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spsinline_o);
		_mm_free(spsinline_e);
		_mm_free(W);
	}

	template<bool isInit, bool adaptive_method, bool is_use_fourier_table0, bool is_use_fourier_table_level>
	void LocalMultiScaleFilterFourier::buildLaplacianSinPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidSin)
	{
		const int rs = radius >> 1;
		const int D = 2 * radius + 1;
		const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidSin.size() != level + 1)FourierPyramidSin.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidSin[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidSin[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidSin[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spsinline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spsinline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = -sigma_range * sigma_range * omega[k] * alpha[k] * boost;//-alphak
		__m256 malpha_k = _mm256_set1_ps(alphak);//-alphak
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);//-alphak
		const float base = -float(2.0 * sqrt(CV_2PI) * omega[k] / T);//-base
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatSin = (__m256*)FourierPyramidSin[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* guidePtr = (__m64*)src8u.ptr<uchar>();
					const float* sxiPtr = &sinTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
						*(splatSin++) = _mm256_i32gather_ps(sxiPtr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatSin++) = _mm256_sin_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_sin = FourierPyramidSin[0].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatSin = (__m256*)(sfpy_sin + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* sxiPtr = &sinTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatSin++) = _mm256_i32gather_ps(sxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatSin++) = _mm256_sin_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* ctable = nullptr;
			if constexpr (is_use_fourier_table0)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;
					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 s2 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 mcos;
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumseo, sumsoo);
					s2 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					sin1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * cos(os) * sumsee;
					}
					else
					{
						dste[I] += alphak * evenratio * cos(os) * sumsee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * cos(os) * sumsoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * cos(os) * sumsoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * cos(os) * sumseo;
					}
					else
					{
						dsto[I] += alphak * evenratio * cos(os) * sumseo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * cos(os) * sumsoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * cos(os) * sumsoo;
					}
				}
				sfpy_sin += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_sin = FourierPyramidSin[l].ptr<float>();
			float* dfpyn_sin = FourierPyramidSin[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* ss = sfpy_sin + i;
					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += width;
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(ss), sums); ss += width;
					}
					_mm256_storeu_ps(spsinline_e + i, sums);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* ss = sfpy_sin + i;
					float sums = GaussWeight[0] * *ss; ss += width;
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * *ss; ss += width;
					}
					spsinline_e[i] = sums;
				}
				sfpy_sin += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* sini = spsinline_e + i;

					__m256 sums = _mm256_mul_ps(W[0], _mm256_loadu_ps(sini++));
					for (int m = 1; m < D; m++)
					{
						sums = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sini++), sums);
					}
					sums = _mm256_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 0, 2, 0));
					sums = _mm256_permute4x64_ps(sums, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_sin + (i >> 1), _mm256_castps256_ps128(sums));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sums = GaussWeight[0] * spsinline_e[i];
					for (int m = 1; m < D; m++)
					{
						sums += GaussWeight[m] * spsinline_e[i + m];
					}
					dfpyn_sin[i >> 1] = sums;
				}
				dfpyn_sin += widths;
			}
#pragma endregion

#pragma region LaplacianLevel

			float* ctable = nullptr;
			if constexpr (is_use_fourier_table_level)
			{
				ctable = &cosTable[FourierTableSize * k];
			}
			sfpy_sin = FourierPyramidSin[l + 1].ptr<float>(0, rs);
			float* ppye_sin = FourierPyramidSin[l].ptr<float>(radius, radius);
			float* ppyo_sin = FourierPyramidSin[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* ss = sfpy_sin + i;

					__m256 sumse = _mm256_mul_ps(W[0], _mm256_loadu_ps(ss)); ss += widths;
					__m256 sumso = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mss = _mm256_loadu_ps(ss); ss += widths;
						sumse = _mm256_fmadd_ps(W[m], mss, sumse);
						sumso = _mm256_fmadd_ps(W[m - 1], mss, sumso);
					}
					_mm256_storeu_ps(spsinline_e + rs + i, _mm256_mul_ps(mevenratio, sumse));
					_mm256_storeu_ps(spsinline_o + rs + i, _mm256_mul_ps(moddratio, sumso));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* ss = sfpy_sin + i;
					float sumse = GaussWeight[0] * *ss; ss += widths;
					float sumso = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumse += GaussWeight[m] * *ss;
						sumso += GaussWeight[m - 1] * *ss;
						ss += widths;
					}
					spsinline_e[i + rs] = sumse * evenratio;
					spsinline_o[i + rs] = sumso * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;
					__m256 sumsee = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinie++));
					__m256 sumsoe = _mm256_setzero_ps();
					__m256 sumseo = _mm256_mul_ps(W[0], _mm256_loadu_ps(sinio++));
					__m256 sumsoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mse = _mm256_loadu_ps(sinie++);
						sumsee = _mm256_fmadd_ps(W[m], mse, sumsee);
						sumsoe = _mm256_fmadd_ps(W[m - 1], mse, sumsoe);
						const __m256 mso = _mm256_loadu_ps(sinio++);
						sumseo = _mm256_fmadd_ps(W[m], mso, sumseo);
						sumsoo = _mm256_fmadd_ps(W[m - 1], mso, sumsoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumsee, sumsoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumsee, sumsoe);
					__m256 sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 mcos;
					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppye_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppye_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumseo, sumsoo);
					temp1 = _mm256_unpackhi_ps(sumseo, sumsoo);
					sin0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					sin1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						mcos = _mm256_cos_ps(ms);
					}
					sin0 = _mm256_fmsub_ps(mevenoddratio, sin0, _mm256_loadu_ps(ppyo_sin + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin0), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_i32gather_ps(ctable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						mcos = _mm256_cos_ps(ms);
					}
					sin1 = _mm256_fmsub_ps(mevenoddratio, sin1, _mm256_loadu_ps(ppyo_sin + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(mcos, sin1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(mcos, sin1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sinie = spsinline_e + i;
					float* sinio = spsinline_o + i;

					float sumsee = GaussWeight[0] * *(sinie++);
					float sumsoe = 0.f;
					float sumseo = GaussWeight[0] * *(sinio++);
					float sumsoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//sin
						sumsee += GaussWeight[m] * *sinie;
						sumsoe += GaussWeight[m - 1] * *sinie++;
						sumseo += GaussWeight[m] * *sinio;
						sumsoo += GaussWeight[m - 1] * *sinio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					else
					{
						dste[I] += alphak * (cos(os) * (evenratio * sumsee - ppye_sin[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (cos(os) * (oddratio * sumsoe - ppye_sin[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					else
					{
						dsto[I] += alphak * (cos(os) * (evenratio * sumseo - ppyo_sin[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (cos(os) * (oddratio * sumsoo - ppyo_sin[I + 1]));
					}
				}
				sfpy_sin += widths;
				ppye_sin += 2 * width;
				ppyo_sin += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spsinline_o);
		_mm_free(spsinline_e);
		_mm_free(W);
	}


	template<bool isInit, bool adaptive_method, bool is_use_fourier_table0, bool is_use_fourier_table_level>
	void LocalMultiScaleFilterFourier::buildLaplacianCosPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos)
	{
		const int rs = radius >> 1;
		const int D = 2 * radius + 1;
		const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1)FourierPyramidCos.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spcosline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spcosline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* guidePtr = (__m64*)src8u.ptr<uchar>();
					const float* cxiPtr = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
						*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* cxiPtr = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region Laplacian0
			float* stable = nullptr;
			if constexpr (is_use_fourier_table0)
			{
				stable = &sinTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 msin;
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}
					if constexpr (is_use_fourier_table0)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * sin(os) * sumcee;
					}
					else
					{
						dste[I] += alphak * evenratio * sin(os) * sumcee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * sin(os) * sumcoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * sin(os) * sumcoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * sin(os) * sumceo;
					}
					else
					{
						dsto[I] += alphak * evenratio * sin(os) * sumceo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * sin(os) * sumcoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * sin(os) * sumcoo;
					}
				}
				sfpy_cos += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			float* stable = nullptr;
			if constexpr (is_use_fourier_table_level)
			{
				stable = &sinTable[FourierTableSize * k];
			}
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 msin;
					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
						msin = _mm256_sin_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
						msin = _mm256_sin_ps(ms);
					}
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}

					if constexpr (is_use_fourier_table_level)
					{
						const __m256i idx = _mm256_cvtps_epi32(_mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_i32gather_ps(stable, idx, sizeof(float));
					}
					else
					{
						const __m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
						msin = _mm256_sin_ps(ms);
					}
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
				}
				sfpy_cos += widths;
				ppye_cos += 2 * width;
				ppyo_cos += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spcosline_o);
		_mm_free(spcosline_e);
		_mm_free(W);
	}

	template<bool isInit, bool adaptive_method, bool is_use_fourier_table0, bool is_use_fourier_table_level, int D, int D2>
	void LocalMultiScaleFilterFourier::buildLaplacianCosPyramidIgnoreBoundary(const vector<Mat>& GaussianPyramid, const Mat& src8u, vector<Mat>& destPyramid, const int k, const int level, vector<Mat>& FourierPyramidCos)
	{
		const int rs = radius >> 1;
		//const int D = 2 * radius + 1;
		//const int D2 = 2 * (2 * rs + 1);
		if (destPyramid.size() != level + 1)destPyramid.resize(level + 1);
		if (FourierPyramidCos.size() != level + 1)FourierPyramidCos.resize(level + 1);

		const Size imSize = GaussianPyramid[0].size();
		//if(destPyramid[0].size()!=imSize)
		destPyramid[0].create(imSize, CV_32F);
		FourierPyramidCos[0].create(imSize, CV_32F);
		for (int l = 1; l < level; l++)
		{
			const Size pySize = GaussianPyramid[l - 1].size() / 2;
			destPyramid[l].create(pySize, CV_32F);
			FourierPyramidCos[l].create(pySize, CV_32F);
		}
		{
			//last  level
			const Size pySize = GaussianPyramid[level - 1].size() / 2;
			destPyramid[level].create(pySize, CV_32F);
			FourierPyramidCos[level].create(pySize, CV_32F);
		}

		const int linesize = destPyramid[0].cols;
		float* spcosline_e = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		float* spcosline_o = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);

		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);

		const __m256 momega_k = _mm256_set1_ps(omega[k]);
		float alphak = sigma_range * sigma_range * omega[k] * alpha[k] * boost;
		__m256 malpha_k = _mm256_set1_ps(alphak);
		__m256 mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, malpha_k);
		const float base = float(2.0 * sqrt(CV_2PI) * omega[k] / T);
		const __m256 mbase = _mm256_set1_ps(base);//for adaptive

#pragma region remap top
		{
			const int width = GaussianPyramid[0].cols;
			const int height = GaussianPyramid[0].rows;
			const int widths = GaussianPyramid[1].cols;
			//splat
			{
				__m256* sptr = (__m256*)GaussianPyramid[0].ptr<float>();
				__m256* splatCos = (__m256*)FourierPyramidCos[0].ptr<float>();
				const int SIZE = width * (D - 1) / 8;

				if (isUseFourierTable0)
				{
					const __m64* guidePtr = (__m64*)src8u.ptr<uchar>();
					const float* cxiPtr = &cosTable[FourierTableSize * k];
					for (int i = 0; i < SIZE; ++i)
					{
						const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
						*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
					}
				}
				else
				{
					for (int i = 0; i < SIZE; ++i)
					{
						//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
						const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
						*(splatCos++) = _mm256_cos_ps(ms);
					}
				}
			}
#pragma endregion

#pragma region Gaussian0
			float* sfpy_cos = FourierPyramidCos[0].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//remap line
				{
					__m256* sptr = (__m256*)(GaussianPyramid[0].ptr<float>(j + D - 1));
					__m256* splatCos = (__m256*)(sfpy_cos + (D - 1) * width);
					const int SIZE = 2 * width / 8;

					if (isUseFourierTable0)
					{
						const __m64* guidePtr = (__m64*)src8u.ptr<uchar>(j + D - 1);
						const float* cxiPtr = &cosTable[FourierTableSize * k];
						for (int i = 0; i < SIZE; ++i)
						{
							const __m256i idx = _mm256_cvtepu8_epi32(*(__m128i*)guidePtr++);
							*(splatCos++) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
						}
					}
					else
					{
						for (int i = 0; i < SIZE; ++i)
						{
							//const __m256 ms = _mm256_mul_ps(momega, _mm256_cvtepu8_ps(*(__m128i*)guidePtr++));
							const __m256 ms = _mm256_mul_ps(momega_k, *sptr++);
							*(splatCos++) = _mm256_cos_ps(ms);
						}
					}
				}
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));
				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region Laplacian0
			sfpy_cos = FourierPyramidCos[1].ptr<float>(0, rs);
			const float* gpye_0 = GaussianPyramid[0].ptr<float>(radius, radius);//GaussianPyramid[0]
			const float* gpyo_0 = GaussianPyramid[0].ptr<float>(radius + 1, radius);//GaussianPyramid[0]
			float* dste = destPyramid[0].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[0].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;
					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[0].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[0].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					__m256 s1 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 s2 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					__m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I));
					__m256 msin = _mm256_sin_ps(ms);
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_0 + I + 8));
					msin = _mm256_sin_ps(ms);
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					s1 = _mm256_unpacklo_ps(sumceo, sumcoo);
					s2 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(s1, s2, 0x20);
					cos1 = _mm256_permute2f128_ps(s1, s2, 0x31);

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I));
					msin = _mm256_sin_ps(ms);
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}
					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_0 + I + 8));
					msin = _mm256_sin_ps(ms);
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						mevenodd_alpha_k = _mm256_mul_ps(mevenoddratio, getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++));
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(mevenodd_alpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_0[I];

					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I), adaptiveBoostBorder[0].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * evenratio * sin(os) * sumcee;
					}
					else
					{
						dste[I] += alphak * evenratio * sin(os) * sumcee;
					}
					os = omega[k] * gpye_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * oddratio * sin(os) * sumcoe;
					}
					else
					{
						dste[I + 1] += alphak * oddratio * sin(os) * sumcoe;
					}
					os = omega[k] * gpyo_0[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * evenratio * sin(os) * sumceo;
					}
					else
					{
						dsto[I] += alphak * evenratio * sin(os) * sumceo;
					}
					os = omega[k] * gpyo_0[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[0].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[0].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * oddratio * sin(os) * sumcoo;
					}
					else
					{
						dsto[I + 1] += alphak * oddratio * sin(os) * sumcoo;
					}
				}
				sfpy_cos += widths;
				gpye_0 += 2 * width;
				gpyo_0 += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		for (int l = 1; l < level; l++)
		{
			const int width = GaussianPyramid[l].cols;
			const int height = GaussianPyramid[l].rows;
			const int widths = GaussianPyramid[l + 1].cols;

#pragma region GaussianLevel
			float* sfpy_cos = FourierPyramidCos[l].ptr<float>();
			float* dfpyn_cos = FourierPyramidCos[l + 1].ptr<float>(rs, rs);

			const int hend = width - 2 * radius;
			const int hendl = widths - 2 * (rs);
			const int vend = height - 2 * radius;
			const int WIDTH = get_simd_floor(width, 8);
			const int HEND = get_simd_floor(hend, 8);
			const int HENDL = get_simd_floor(hendl, 8);

			for (int j = 0; j < vend; j += 2)
			{
				//v filter
				for (int i = 0; i < WIDTH; i += 8)
				{
					const float* sc = sfpy_cos + i;
					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(sc), sumc); sc += width;
					}
					_mm256_storeu_ps(spcosline_e + i, sumc);
				}
				for (int i = WIDTH; i < width; i++)
				{
					const float* sc = sfpy_cos + i;
					float sumc = GaussWeight[0] * *sc; sc += width;
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * *sc; sc += width;
					}
					spcosline_e[i] = sumc;
				}
				sfpy_cos += 2 * width;

				//h filter
				for (int i = 0; i < HEND; i += 8)
				{
					float* cosi = spcosline_e + i;

					__m256 sumc = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosi++));
					for (int m = 1; m < D; m++)
					{
						sumc = _mm256_fmadd_ps(W[m], _mm256_loadu_ps(cosi++), sumc);
					}
					sumc = _mm256_shuffle_ps(sumc, sumc, _MM_SHUFFLE(2, 0, 2, 0));
					sumc = _mm256_permute4x64_ps(sumc, _MM_SHUFFLE(3, 1, 2, 0));
					_mm_storeu_ps(dfpyn_cos + (i >> 1), _mm256_castps256_ps128(sumc));

				}
				for (int i = HEND; i < hend; i += 2)
				{
					float sumc = GaussWeight[0] * spcosline_e[i];
					for (int m = 1; m < D; m++)
					{
						sumc += GaussWeight[m] * spcosline_e[i + m];
					}
					dfpyn_cos[i >> 1] = sumc;
				}
				dfpyn_cos += widths;
			}
#pragma endregion

#pragma region LaplacianLevel
			sfpy_cos = FourierPyramidCos[l + 1].ptr<float>(0, rs);
			float* ppye_cos = FourierPyramidCos[l].ptr<float>(radius, radius);
			float* ppyo_cos = FourierPyramidCos[l].ptr<float>(radius + 1, radius);
			const float* gpye_l = GaussianPyramid[l].ptr<float>(radius, radius);//GaussianPyramid[l]
			const float* gpyo_l = GaussianPyramid[l].ptr<float>(radius + 1, radius);//GaussianPyramid[l]
			float* dste = destPyramid[l].ptr<float>(radius, radius);//destPyramid
			float* dsto = destPyramid[l].ptr<float>(radius + 1, radius);//destPyramid
			__m256* adaptiveSigma_e = nullptr;
			__m256* adaptiveBoost_e = nullptr;
			__m256* adaptiveSigma_o = nullptr;
			__m256* adaptiveBoost_o = nullptr;

			for (int j = 0; j < vend; j += 2)
			{
				// v filter							
				for (int i = 0; i < HENDL; i += 8)
				{
					float* sc = sfpy_cos + i;

					__m256 sumce = _mm256_mul_ps(W[0], _mm256_loadu_ps(sc)); sc += widths;
					__m256 sumco = _mm256_setzero_ps();
					for (int m = 2; m < D2; m += 2)
					{
						const __m256 msc = _mm256_loadu_ps(sc); sc += widths;
						sumce = _mm256_fmadd_ps(W[m], msc, sumce);
						sumco = _mm256_fmadd_ps(W[m - 1], msc, sumco);
					}
					_mm256_storeu_ps(spcosline_e + rs + i, _mm256_mul_ps(mevenratio, sumce));
					_mm256_storeu_ps(spcosline_o + rs + i, _mm256_mul_ps(moddratio, sumco));
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* sc = sfpy_cos + i;
					float sumce = GaussWeight[0] * *sc; sc += widths;
					float sumco = 0.f;
					for (int m = 2; m < D2; m += 2)
					{
						sumce += GaussWeight[m] * *sc;
						sumco += GaussWeight[m - 1] * *sc;
						sc += widths;
					}
					spcosline_e[i + rs] = sumce * evenratio;
					spcosline_o[i + rs] = sumco * oddratio;
				}

				//h filter
				if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
				{
					adaptiveSigma_e = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius, radius);
					adaptiveBoost_e = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius, radius);
					adaptiveSigma_o = (__m256*)adaptiveSigmaBorder[l].ptr<float>(j + radius + 1, radius);
					adaptiveBoost_o = (__m256*)adaptiveBoostBorder[l].ptr<float>(j + radius + 1, radius);
				}
				for (int i = 0; i < HENDL; i += 8)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					__m256 sumcee = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosie++));
					__m256 sumcoe = _mm256_setzero_ps();
					__m256 sumceo = _mm256_mul_ps(W[0], _mm256_loadu_ps(cosio++));
					__m256 sumcoo = _mm256_setzero_ps();

					for (int m = 2; m < D2; m += 2)
					{
						const __m256 mce = _mm256_loadu_ps(cosie++);
						sumcee = _mm256_fmadd_ps(W[m], mce, sumcee);
						sumcoe = _mm256_fmadd_ps(W[m - 1], mce, sumcoe);
						const __m256 mco = _mm256_loadu_ps(cosio++);
						sumceo = _mm256_fmadd_ps(W[m], mco, sumceo);
						sumcoo = _mm256_fmadd_ps(W[m - 1], mco, sumcoo);
					}

					const int I = i << 1;
					//even line
					__m256 temp0 = _mm256_unpacklo_ps(sumcee, sumcoe);
					__m256 temp1 = _mm256_unpackhi_ps(sumcee, sumcoe);
					__m256 cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					__m256 cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					__m256 ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I));
					__m256 msin = _mm256_sin_ps(ms);
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppye_cos + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dste + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dste + I)));
					}

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpye_l + I + 8));
					msin = _mm256_sin_ps(ms);
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppye_cos + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_e++, *adaptiveBoost_e++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dste + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dste + I + 8)));
					}

					//odd line
					temp0 = _mm256_unpacklo_ps(sumceo, sumcoo);
					temp1 = _mm256_unpackhi_ps(sumceo, sumcoo);
					cos0 = _mm256_permute2f128_ps(temp0, temp1, 0x20);
					cos1 = _mm256_permute2f128_ps(temp0, temp1, 0x31);

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I));
					msin = _mm256_sin_ps(ms);
					cos0 = _mm256_fmsub_ps(mevenoddratio, cos0, _mm256_loadu_ps(ppyo_cos + I));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos0)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos0), _mm256_loadu_ps(dsto + I)));
					}

					ms = _mm256_mul_ps(momega_k, _mm256_loadu_ps(gpyo_l + I + 8));
					msin = _mm256_sin_ps(ms);
					cos1 = _mm256_fmsub_ps(mevenoddratio, cos1, _mm256_loadu_ps(ppyo_cos + I + 8));
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						malpha_k = getAdaptiveAlpha(momega_k, mbase, *adaptiveSigma_o++, *adaptiveBoost_o++);
					}
					if constexpr (isInit)
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_mul_ps(malpha_k, _mm256_mul_ps(msin, cos1)));
					}
					else
					{
						_mm256_storeu_ps(dsto + I + 8, _mm256_fmadd_ps(malpha_k, _mm256_mul_ps(msin, cos1), _mm256_loadu_ps(dsto + I + 8)));
					}
				}
				for (int i = HENDL; i < hendl; i++)
				{
					float* cosie = spcosline_e + i;
					float* cosio = spcosline_o + i;
					float sumcee = GaussWeight[0] * *(cosie++);
					float sumcoe = 0.f;
					float sumceo = GaussWeight[0] * *(cosio++);
					float sumcoo = 0.f;

					for (int m = 2; m < D2; m += 2)
					{
						//cos				
						sumcee += GaussWeight[m] * *cosie;
						sumcoe += GaussWeight[m - 1] * *cosie++;
						sumceo += GaussWeight[m] * *cosio;
						sumcoo += GaussWeight[m - 1] * *cosio++;
					}
					const int I = i << 1;
					float os = omega[k] * gpye_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I), adaptiveBoostBorder[l].at<float>(j + radius, radius + I));
					}
					if constexpr (isInit)
					{
						dste[I] = alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					else
					{
						dste[I] += alphak * (sin(os) * (evenratio * sumcee - ppye_cos[I]));
					}
					os = omega[k] * gpye_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dste[I + 1] = alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					else
					{
						dste[I + 1] += alphak * (sin(os) * (oddratio * sumcoe - ppye_cos[I + 1]));
					}
					os = omega[k] * gpyo_l[I];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I));
					}
					if constexpr (isInit)
					{
						dsto[I] = alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					else
					{
						dsto[I] += alphak * (sin(os) * (evenratio * sumceo - ppyo_cos[I]));
					}
					os = omega[k] * gpyo_l[I + 1];
					if constexpr (adaptive_method == AdaptiveMethod::ADAPTIVE)
					{
						alphak = getAdaptiveAlpha(omega[k], base, adaptiveSigmaBorder[l].at<float>(j + radius + 1, radius + I + 1), adaptiveBoostBorder[l].at<float>(j + radius + 1, radius + I + 1));
					}
					if constexpr (isInit)
					{
						dsto[I + 1] = alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
					else
					{
						dsto[I + 1] += alphak * (sin(os) * (oddratio * sumcoo - ppyo_cos[I + 1]));
					}
				}
				sfpy_cos += widths;
				ppye_cos += 2 * width;
				ppyo_cos += 2 * width;
				gpye_l += 2 * width;
				gpyo_l += 2 * width;
				dste += 2 * width;
				dsto += 2 * width;
			}
#pragma endregion
		}

		_mm_free(spcosline_o);
		_mm_free(spcosline_e);
		_mm_free(W);
	}

	//summation of srcPyramid for each order -> destPyramid
	void LocalMultiScaleFilterFourier::sumPyramid(const vector<vector<Mat>>& srcPyramids, vector<Mat>& destPyramid, const int numberPyramids, const int level, vector<bool>& used)
	{
		vector<vector<int>> h(level);
		for (int l = 0; l < level; l++)
		{
			h[l].resize(threadMax + 1);
			h[l][0] = 0;
			const int basestep = destPyramid[l].rows / threadMax;
			int rem = destPyramid[l].rows % threadMax;

			for (int t = 0; t < threadMax; t++)
			{
				h[l][t + 1] = h[l][t] + basestep + ((rem > 0) ? 1 : 0);
				rem--;
				//print_debug4(l, t, h[l][t + 1], destPyramid[l].rows);
			}
		}

#pragma omp parallel for schedule (dynamic)
		for (int t = 0; t < threadMax; t++)
		{
			for (int l = 0; l < level; l++)
			{
				const int hs = h[l][t];
				const int he = h[l][t + 1];
				const int w = destPyramid[l].cols;
				const int size = w * (he - hs);
				const int SIZE32 = get_simd_floor(size, 32);
				const int SIZE8 = get_simd_floor(size, 8);
				float* d = destPyramid[l].ptr<float>(hs);
				for (int k = numberPyramids - 1; k >= 0; --k)
				{
					if (used[k])
					{
						const float* s = srcPyramids[k][l].ptr<float>(hs);
						for (int i = 0; i < SIZE32; i += 32)
						{
							_mm256_storeu_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(s + i)));
							_mm256_storeu_ps(d + i + 8, _mm256_add_ps(_mm256_loadu_ps(d + i + 8), _mm256_loadu_ps(s + i + 8)));
							_mm256_storeu_ps(d + i + 16, _mm256_add_ps(_mm256_loadu_ps(d + i + 16), _mm256_loadu_ps(s + i + 16)));
							_mm256_storeu_ps(d + i + 24, _mm256_add_ps(_mm256_loadu_ps(d + i + 24), _mm256_loadu_ps(s + i + 24)));
						}
						for (int i = SIZE32; i < SIZE8; i += 8)
						{
							_mm256_storeu_ps(d + i, _mm256_add_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(s + i)));
						}
						for (int i = SIZE8; i < size; i++)
						{
							d[i] += s[i];
						}
					}
				}
			}
		}
	}


	void LocalMultiScaleFilterFourier::pyramidParallel(const Mat& src, Mat& dest)
	{
		//cout << "pyramidParallel "<< endl;
		//print_debug(pyramidComputeMethod);
		//print_debug(computeScheduleFourier);
		//print_debug(isUseFourierTable0);
		layerSize.resize(level + 1);
		allocImageBuffer(order, level);

		const int gfRadius = getGaussianRadius(sigma_space);
		//print_debug(gfRadius);
		const int lowr = 2 * gfRadius + gfRadius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);
		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (src.depth() == CV_8U)
			{
				copyMakeBorder(src, src8u, r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				src8u.convertTo(ImageStack[0], CV_32F);
			}
			else
			{
				cv::copyMakeBorder(src, ImageStack[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				if (isUseFourierTable0) ImageStack[0].convertTo(src8u, CV_8U);
			}

			if (adaptiveMethod)
			{
				adaptiveBoostBorder.resize(level);
				adaptiveSigmaBorder.resize(level);

				bool isEachBorder = false;
				if (isEachBorder)
				{
					for (int l = 0; l < level; l++)
					{
						int rr = (r_pad0 >> l);
						cv::copyMakeBorder(adaptiveBoostMap[l], adaptiveBoostBorder[l], rr, rr, rr, rr, borderType);
						cv::copyMakeBorder(adaptiveSigmaMap[l], adaptiveSigmaBorder[l], rr, rr, rr, rr, borderType);
					}
				}
				else
				{
					cv::copyMakeBorder(adaptiveBoostMap[0], adaptiveBoostBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
					cv::copyMakeBorder(adaptiveSigmaMap[0], adaptiveSigmaBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
					//print_debug2(srcf.size(), adaptiveBoostMap[0].size());
					//print_debug2(border.size(), adaptiveBoostBorder[0].size());
					for (int l = 0; l < level - 1; l++)
					{
						resize(adaptiveBoostBorder[l], adaptiveBoostBorder[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
						resize(adaptiveSigmaBorder[l], adaptiveSigmaBorder[l + 1], Size(), 0.5, 0.5, INTER_NEAREST);
					}
				}
			}
		}
		else
		{
			if (src.depth() == CV_8U)
			{
				src.convertTo(ImageStack[0], CV_32F);
				src8u = src;
			}
			else
			{
				src.copyTo(ImageStack[0]);
				if (isUseFourierTable0) src.convertTo(src8u, CV_8U);
			}
		}

		for (int l = 0; l < level + 1; l++)
		{
			layerSize[l] = ImageStack[l].size();
		}

		//compute alpha, T, and table: 0.065ms
		{
			//cp::Timer t("initRangeFourier");
			initRangeFourier(order, sigma_range, boost);
		}

		//Build Gaussian Pyramid for Input Image
		buildGaussianPyramid(ImageStack[0], ImageStack, level, sigma_space);

		//Build Outoput Laplacian Pyramid
		if (computeScheduleFourier == MergeFourier) //merge cos and sin
		{
			vector<bool> init(threadMax);
			for (int t = 0; t < threadMax; t++)init[t] = true;

#pragma omp parallel for schedule (dynamic)
			for (int k = 0; k < order + 1; k++)
			{
				const int tidx = omp_get_thread_num();
				if (k == order)
				{
					//DC pyramid
					buildLaplacianPyramid(ImageStack, DetailStack, level, sigma_space);
				}
				else
				{
					if (init[tidx])
					{
#pragma omp critical
						init[tidx] = false;
						if (isUseFourierTable0)
						{
							if (isUseFourierTableLevel)
							{
								if (radius == 2)
								{
									if (adaptiveMethod) buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<true, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<true, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<true, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<true, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
							}
						}
					}
					else
					{
						if (isUseFourierTable0)
						{
							if (isUseFourierTableLevel)
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
									else buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								}
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
								else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackCos[tidx], FourierStackSin[tidx]);
							}
						}
					}
				}
			}

			for (int i = 0; i < threadMax; i++)init[i] = init[i] ? false : true;
			sumPyramid(destEachOrder, DetailStack, threadMax, level, init);
		}
		else if (computeScheduleFourier == SplitFourier) //split cos and sin
		{
			vector<bool> init(threadMax);
			for (int t = 0; t < threadMax; t++)init[t] = true;

			const int NC = 2 * order;
#pragma omp parallel for schedule (dynamic)
			for (int nc = 0; nc < NC + 1; nc++)
			{
				const int tidx = omp_get_thread_num();
				if (nc == NC)
				{
					//DC pyramid
					buildLaplacianPyramid(ImageStack, DetailStack, level, sigma_space);
				}
				else
				{
					const int k = nc / 2;

					if (init[tidx])
					{
#pragma omp critical
						init[tidx] = false;
						if (nc % 2 == 0)
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<true, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<true, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<true, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
							}
						}
						else
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<true, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<true, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<true, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<true, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<true, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
							}
						}
					}
					else
					{
						if (nc % 2 == 0)
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
							}
						}
						else
						{
							if (isUseFourierTable0)
							{
								if (isUseFourierTableLevel)
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
								else
								{
									if (radius == 2)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else if (radius == 4)
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
									else
									{
										if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
										else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									}
								}
							}
							else
							{
								if (radius == 2)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else if (radius == 4)
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
								else
								{
									if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
									else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, destEachOrder[tidx], k, level, FourierStackSin[tidx]);
								}
							}
						}
					}
				}
			}

			for (int i = 0; i < threadMax; i++)init[i] = init[i] ? false : true;
			//for (int i = 0; i < threadMax; i++) { if (!init[i])cout << i << "not used" << endl; }
			sumPyramid(destEachOrder, DetailStack, threadMax, level, init);
			//sumPyramid(destEachOrder, LaplacianPyramid, 1, level);//for non parallel
		}
		DetailStack[level] = ImageStack[level];

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			collapseLaplacianPyramid(DetailStack, DetailStack[0], CV_32F);
			DetailStack[0](Rect(r_pad0, r_pad0, src.cols, src.rows)).copyTo(dest);
		}
		else
		{
			collapseLaplacianPyramid(DetailStack, dest, src.depth());
		}

		if (isPlot)
		{
			kernelPlot(GAUSS, order, 255, boost, sigma_range, Salpha, Sbeta, sigma_range, 0, 255, 255, T, sinTable, cosTable, alpha, beta, windowType, "GFP f(x)");
			isPlotted = true;
		}
		else
		{
			if (isPlotted)
			{
				cv::destroyWindow("GFP f(x)");
				isPlotted = false;
			}
		}
	}

	void LocalMultiScaleFilterFourier::pyramidSerial(const Mat& src, Mat& dest)
	{
		layerSize.resize(level + 1);
		allocImageBuffer(order, level);

		const int lowr = 3 * radius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);

		if (src.depth() == CV_8U)
		{
			src.convertTo(ImageStack[0], CV_32F);
			src8u = src;
		}
		else
		{
			ImageStack[0] = src;
#ifdef USE_GATHER8U
			if (isUseFourierTable0) src.convertTo(src8u, CV_8U);
#endif
		}

		if (adaptiveMethod)
		{
			adaptiveBoostBorder.resize(level);
			adaptiveSigmaBorder.resize(level);
			for (int l = 0; l < level; l++)
			{
				adaptiveBoostBorder[l] = adaptiveBoostMap[l];
				adaptiveSigmaBorder[l] = adaptiveSigmaMap[l];
			}
		}

		for (int l = 0; l < level + 1; l++)
		{
			layerSize[l] = ImageStack[l].size();
		}

		{
			//cp::Timer t("initRangeFourier");
			//compute alpha, T, and table: 0.065ms
			initRangeFourier(order, sigma_range, boost);
		}

		//Build Gaussian Pyramid for Input Image
		buildGaussianLaplacianPyramid(ImageStack[0], ImageStack, DetailStack, level, sigma_space);

		//Build Outoput Laplacian Pyramid
		if (isUseFourierTable0)
		{
			if (isUseFourierTableLevel)
			{
				if (computeScheduleFourier == MergeFourier) //merge cos and sin use table
				{
					for (int k = 0; k < order; k++)
					{
						if (radius == 2)
						{
							if (adaptiveMethod) buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						}
					}
				}
				else if (computeScheduleFourier == SplitFourier) //split cos and sin use table
				{
					const int NC = 2 * order;
					for (int nc = 0; nc < NC; nc++)
					{
						const int k = nc / 2;

						if (nc % 2 == 0)
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, true>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, true>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
						}
					}
				}
			}
			else
			{
				if (computeScheduleFourier == MergeFourier) //merge cos and sin use table
				{
					for (int k = 0; k < order; k++)
					{
						if (radius == 2)
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
							else              buildLaplacianFourierPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						}
					}
				}
				else if (computeScheduleFourier == SplitFourier) //split cos and sin use table
				{
					const int NC = 2 * order;
					for (int nc = 0; nc < NC; nc++)
					{
						const int k = nc / 2;

						if (nc % 2 == 0)
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianSinPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
						}
						else
						{
							if (radius == 2)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else if (radius == 4)
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
							else
							{
								if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, true, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
								else          buildLaplacianCosPyramidIgnoreBoundary<false, false, true, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							}
						}
					}
				}
			}
		}
		else
		{
			if (computeScheduleFourier == MergeFourier) //merge cos and sin
			{
				for (int k = 0; k < order; k++)
				{
					if (radius == 2)
					{
						if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
					}
					else if (radius == 4)
					{
						if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
					}
					else
					{
						if (adaptiveMethod)buildLaplacianFourierPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
						else buildLaplacianFourierPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, DetailStack, k, level, FourierStackCos[0], FourierStackSin[0]);
					}
				}
			}
			else if (computeScheduleFourier == SplitFourier) //split cos and sin
			{
				const int NC = 2 * order;
				for (int nc = 0; nc < NC; nc++)
				{
					const int k = nc / 2;

					if (nc % 2 == 0)
					{
						if (radius == 2)
						{
							if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianSinPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							else          buildLaplacianSinPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
						}
					}
					else
					{
						if (radius == 2)
						{
							if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 5, 6>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
						}
						else if (radius == 4)
						{
							if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false, 9, 10>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
						}
						else
						{
							if (adaptiveMethod)buildLaplacianCosPyramidIgnoreBoundary<false, true, false, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
							else          buildLaplacianCosPyramidIgnoreBoundary<false, false, false, false>(ImageStack, src8u, DetailStack, k, level, FourierStackSin[0]);
						}
					}
				}
			}
		}

		DetailStack[level] = ImageStack[level];
		collapseLaplacianPyramid(DetailStack, dest, src.depth());
	}

	void LocalMultiScaleFilterFourier::pyramid(const Mat& src, Mat& dest)
	{
		rangeDescope(src);

		if (isParallel)	
			pyramidParallel(src, dest);
		else 
			pyramidSerial(src, dest);
	}
#pragma endregion

#pragma region DoG
	void LocalMultiScaleFilterFourier::remapCosSin(const cv::Mat& src, int k, Mat& destCos, Mat& destSin, bool isCompute)
	{
		if (destCos.empty()) destCos.create(src.size(), CV_32F);
		if (destSin.empty()) destSin.create(src.size(), CV_32F);

		const int simdsize = sizeof(__m256) / sizeof(float);
		const __m256* guidePtr = (const __m256*)src.ptr<float>();
		__m256* mcos = (__m256*)destCos.ptr<float>();
		__m256* msin = (__m256*)destSin.ptr<float>();

		if (isCompute)
		{
			const __m256 momega = _mm256_set1_ps(omega[k]);
			for (int i = 0; i < src.size().area() / simdsize; i++)
			{
				const __m256 idx = _mm256_mul_ps(momega, *guidePtr);
				*(mcos) = _mm256_cos_ps(idx);
				*(msin) = _mm256_sin_ps(idx);
				guidePtr++;
				mcos++;
				msin++;
			}
		}
		else
		{
			const float* sxiPtr = &sinTable[FourierTableSize * k];
			const float* cxiPtr = &cosTable[FourierTableSize * k];
			for (int i = 0; i < src.size().area() / simdsize; i++)
			{
				const __m256i idx = _mm256_cvtps_epi32(*guidePtr);
				*(mcos) = _mm256_i32gather_ps(cxiPtr, idx, sizeof(float));
				*(msin) = _mm256_i32gather_ps(sxiPtr, idx, sizeof(float));
				guidePtr++;
				mcos++;
				msin++;
			}
		}
	}

	void LocalMultiScaleFilterFourier::productSummingTrig(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, float sigma_range, bool isCompute)
	{
		const int simdsize = sizeof(__m256) / sizeof(float);

		AutoBuffer <__m256> malpha(order);
		AutoBuffer <__m256> momega(order);//for compute

		for (int k = 0; k < order; k++)
		{
			const float alphak = -sigma_range * sigma_range * omega[k] * alpha[k] * boost;
			malpha[k] = _mm256_set1_ps(alphak);
			momega[k] = _mm256_set1_ps(omega[k]);
		}

		const int size = src[0].cols / simdsize;
		if (isCompute)
		{
#pragma omp parallel for schedule (dynamic)
			for (int j = 0; j < src[0].rows; j++)
			{
				for (int l = 0; l < level; l++)
				{
					__m256* destPtr = (__m256*)dest[l].ptr<float>(j);
					const __m256* guidePtr = (const __m256*)src[l].ptr<float>(j);//sigma_n
					AutoBuffer<const __m256*> mcosSplat(order);
					AutoBuffer<const __m256*> msinSplat(order);
					for (int k = 0; k < order; k++)
					{
						mcosSplat[k] = (const __m256*)FourierStackCos[k][l].ptr<float>(j);
						msinSplat[k] = (const __m256*)FourierStackSin[k][l].ptr<float>(j);
					}

					for (int i = 0; i < size; i++)
					{
						__m256 ret = _mm256_setzero_ps();
						const __m256 mg = *guidePtr;
						for (int k = order - 1; k >= 0; k--)
						{
							const __m256 mog = _mm256_mul_ps(momega[k], mg);
							const __m256 msin = _mm256_sin_ps(mog);
							const __m256 mcos = _mm256_cos_ps(mog);
							const __m256 a = _mm256_fnmadd_ps(mcos, *(msinSplat[k]), _mm256_mul_ps(msin, *(mcosSplat[k])));
							ret = _mm256_fmadd_ps(malpha[k], a, ret);
							mcosSplat[k]++;
							msinSplat[k]++;
						}
						*(destPtr) = _mm256_add_ps(*(destPtr), ret);
						guidePtr++;
						destPtr++;
					}
				}
			}
		}
		else
		{
#pragma omp parallel for schedule (dynamic)
			for (int j = 0; j < src[0].rows; j++)
			{
				AutoBuffer<__m256*> destPtr(level);
				AutoBuffer<const __m256*> guidePtr(level);
				AutoBuffer <AutoBuffer<const __m256*>> mcosSplat(level);
				AutoBuffer <AutoBuffer<const __m256*>> msinSplat(level);
				for (int l = 0; l < level; l++)
				{
					destPtr[l] = (__m256*)dest[l].ptr<float>(j);
					guidePtr[l] = (const __m256*)src[l].ptr<float>(j);//sigma_n
					mcosSplat[l].resize(order);
					msinSplat[l].resize(order);
					for (int k = 0; k < order; k++)
					{
						mcosSplat[l][k] = (const __m256*)FourierStackCos[k][l].ptr<float>(j);
						msinSplat[l][k] = (const __m256*)FourierStackSin[k][l].ptr<float>(j);
					}
				}

				for (int i = 0; i < size; i++)
				{
					for (int l = 0; l < level; l++)
					{
						const __m256i mg = _mm256_cvtps_epi32(*guidePtr[l]);
						for (int k = 0; k < order; k++)
						{
							const int tidx = FourierTableSize * k;
							const __m256 msin = _mm256_i32gather_ps(sinTable + tidx, mg, 4);
							const __m256 mcos = _mm256_i32gather_ps(cosTable + tidx, mg, 4);
							const __m256 a = _mm256_fmsub_ps(msin, *(mcosSplat[l][k]), _mm256_mul_ps(mcos, *(msinSplat[l][k])));
							*(destPtr[l]) = _mm256_fmadd_ps(malpha[k], a, *(destPtr[l]));
							mcosSplat[l][k]++;
							msinSplat[l][k]++;
						}
						guidePtr[l]++;
						destPtr[l]++;
					}
				}
			}
		}
	}

	void LocalMultiScaleFilterFourier::dog(const Mat& src, Mat& dest)
	{
		//initRangeTable(sigma_range, boost);
		cv::Mat srcf;
		if (src.depth() == CV_8U)
		{
			src.convertTo(srcf, CV_32F);
		}
		else
		{
			srcf = src.clone();
		}

		initRangeFourier(order, sigma_range, boost);
		if (FourierStackCos.size() != order)FourierStackCos.resize(order);
		if (FourierStackSin.size() != order)FourierStackSin.resize(order);
#pragma omp parallel for schedule (dynamic)
		for (int k = -1; k < order; k++)
		{
			if (k < 0) //for DC
			{
				buildGaussianStack(srcf, ImageStack, sigma_space, level);
				if (DetailStack.size() != level + 1)
				{
					DetailStack.clear();
					DetailStack.resize(level + 1);
				}
				for (int l = 0; l < level; l++)
				{
					cv::subtract(ImageStack[l], ImageStack[l + 1], DetailStack[l]);
				}
				DetailStack[level] = ImageStack[level];
			}
			else
			{
				if (FourierStackCos[k].size() != level + 1) FourierStackCos[k].resize(level + 1);
				if (FourierStackSin[k].size() != level + 1) FourierStackSin[k].resize(level + 1);
				remapCosSin(srcf, k, FourierStackCos[k][0], FourierStackSin[k][0], isCompute);
				buildDoGStack(FourierStackCos[k][0], FourierStackCos[k], sigma_space, level);
				buildDoGStack(FourierStackSin[k][0], FourierStackSin[k], sigma_space, level);
			}
		}
		productSummingTrig(ImageStack, DetailStack, sigma_range, isCompute);
		collapseDoGStack(DetailStack, dest, src.depth());
	}

	void LocalMultiScaleFilterFourier::cog(const Mat& src, Mat& dest)
	{
		//initRangeTable(sigma_range, boost);
		cv::Mat srcf;
		if (src.depth() == CV_8U)
		{
			src.convertTo(srcf, CV_32F);
		}
		else
		{
			srcf = src.clone();
		}

		initRangeFourier(order, sigma_range, boost);
		if (FourierStackCos.size() != order) FourierStackCos.resize(order);
		if (FourierStackSin.size() != order) FourierStackSin.resize(order);
#pragma omp parallel for schedule (dynamic)
		for (int k = -1; k < order; k++)
		{
			if (k < 0) //for DC
			{
				buildGaussianStack(srcf, ImageStack, sigma_space, level);
				if (DetailStack.size() != level + 1)
				{
					DetailStack.clear();
					DetailStack.resize(level + 1);
				}
				buildCoGStack(src, DetailStack, sigma_space, level);
				DetailStack[level] = ImageStack[level];
			}
			else
			{
				if (FourierStackCos[k].size() != level + 1) FourierStackCos[k].resize(level + 1);
				if (FourierStackSin[k].size() != level + 1) FourierStackSin[k].resize(level + 1);
				remapCosSin(srcf, k, FourierStackCos[k][0], FourierStackSin[k][0], isCompute);
				buildCoGStack(FourierStackCos[k][0], FourierStackCos[k], sigma_space, level);
				buildCoGStack(FourierStackSin[k][0], FourierStackSin[k], sigma_space, level);
			}
		}
		productSummingTrig(ImageStack, DetailStack, sigma_range, isCompute);
		collapseCoGStack(DetailStack, dest, src.depth());
	}
#pragma endregion

	void LocalMultiScaleFilterFourier::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod)
	{
		if(scaleSpaceMethod== ScaleSpace::Pyramid) allocSpaceWeight(sigma_space);

		this->order = order;

		this->sigma_space = sigma_space;
		this->level = max(level, 1);
		this->boost = boost;
		this->sigma_range = sigma_range;
		this->scalespaceMethod = scaleSpaceMethod;

		body(src, dest);
		if (scaleSpaceMethod == ScaleSpace::Pyramid) freeSpaceWeight();
	}

#pragma region setter_getter
	void LocalMultiScaleFilterFourier::setPeriodMethod(Period scaleSpaceMethod)
	{
		periodMethod = scaleSpaceMethod;
	}

	void LocalMultiScaleFilterFourier::setComputeScheduleMethod(int schedule, bool useTable0, bool useTableLevel)
	{
		computeScheduleFourier = schedule;
		isUseFourierTable0 = useTable0;
		isUseFourierTableLevel = useTableLevel;
	}

	string LocalMultiScaleFilterFourier::getComputeScheduleName()
	{
		string ret = "";
		if (computeScheduleFourier == MergeFourier) ret += "MergeFourier_";
		if (computeScheduleFourier == SplitFourier) ret += "SplitFourier_";

		if (isUseFourierTable0) ret += "Gather0_";
		else  ret += "Compute0_";
		if (isUseFourierTableLevel) ret += "GatherL";
		else  ret += "ComputeL";

		return ret;
	}

	void LocalMultiScaleFilterFourier::setIsParallel(const bool flag)
	{
		this->isParallel = flag;
	}

	void  LocalMultiScaleFilterFourier::setIsPlot(const bool flag)
	{
		this->isPlot = flag;
	}

	std::string LocalMultiScaleFilterFourier::getPeriodName()
	{
		string ret = "";
		if (periodMethod == Period::GAUSS_DIFF)ret = "GAUSS_DIFF";
		if (periodMethod == Period::OPTIMIZE)ret = "OPTIMIZE";
		if (periodMethod == Period::PRE_SET)ret = "PRE_SET";

		return ret;
	}
#pragma endregion
#pragma endregion

#pragma region TileLocalMultiScaleFilterFourier
	TileLocalMultiScaleFilterFourier::TileLocalMultiScaleFilterFourier()
	{
		msf = new LocalMultiScaleFilterFourier[threadMax];
		for (int i = 0; i < threadMax; i++)
			msf[i].setIsParallel(false);
	}

	TileLocalMultiScaleFilterFourier::~TileLocalMultiScaleFilterFourier()
	{
		delete[] msf;
	}


	void TileLocalMultiScaleFilterFourier::setComputeScheduleMethod(int schedule, bool useTable0, bool useTableLevel)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setComputeScheduleMethod(schedule, useTable0, useTableLevel);
	}

	string TileLocalMultiScaleFilterFourier::getComputeScheduleName()
	{
		return msf[0].getComputeScheduleName();
	}

	void TileLocalMultiScaleFilterFourier::setAdaptive(const bool flag, const Size div, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap)
	{
		if (flag)
		{
			vector<Mat> g{ adaptiveSigmaMap, adaptiveBoostMap };
			initGuide(div, g);
		}
		else
		{
			unsetUseGuide();
		}
	}

	void TileLocalMultiScaleFilterFourier::setPeriodMethod(const int method)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setPeriodMethod((LocalMultiScaleFilterFourier::Period)method);

	}

	void TileLocalMultiScaleFilterFourier::setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod scaleSpaceMethod)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setRangeDescopeMethod(scaleSpaceMethod);
	}

	void TileLocalMultiScaleFilterFourier::process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex)
	{
		if (isUseGuide)
		{
			msf[threadIndex].setAdaptive(true, guideTile[0][imageIndex], guideTile[1][imageIndex], level);
		}
		else
		{
			Mat a;
			msf[threadIndex].setAdaptive(false, a, a, 0);
		}

		msf[threadIndex].filter(src, dst, order, sigma_range, sigma_space, boost, level, scaleSpaceMethod);
	}

	void TileLocalMultiScaleFilterFourier::filter(const Size div, const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const MultiScaleFilter::ScaleSpace scaleSpaceMethod)
	{
		this->order = order;
		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;

		this->scaleSpaceMethod = scaleSpaceMethod;
		const int lowr = 3 * msf[0].getGaussianRadius(sigma_space);
		const int r_pad0 = lowr * (int)pow(2, level - 1);
		invoker(div, src, dest, r_pad0);
	}
#pragma endregion
}