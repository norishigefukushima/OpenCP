#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

namespace cp
{
	//A LOCAL STATISTICS PRIOR
	void bilateralFilterLocalStatisticsPrior(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, const float sigma_range, const float sigma_space, const float delta, std::vector<cv::Mat>& smooth)
	{
		CV_Assert(src.size() == 3);
		CV_Assert(src[0].depth() == CV_32F);

		if (dest.size() != 3)
		{
			dest.resize(3);
		}
		dest[0].create(src[0].size(), CV_32F);
		dest[1].create(src[1].size(), CV_32F);
		dest[2].create(src[2].size(), CV_32F);

		if (smooth.size() != 3)smooth.resize(3);
		if (smooth[0].empty())
		{
			smooth[0].create(src[0].size(), CV_32F);
			smooth[1].create(src[1].size(), CV_32F);
			smooth[2].create(src[2].size(), CV_32F);
			int D = 2 * (int)ceil(sigma_space * 3.f) + 1;
			cv::GaussianBlur(src[0], smooth[0], cv::Size(D, D), sigma_space);
			cv::GaussianBlur(src[1], smooth[1], cv::Size(D, D), sigma_space);
			cv::GaussianBlur(src[2], smooth[2], cv::Size(D, D), sigma_space);
		}

		const float* s0 = src[0].ptr<float>();
		const float* s1 = src[1].ptr<float>();
		const float* s2 = src[2].ptr<float>();
		float* m0 = smooth[0].ptr<float>();
		float* m1 = smooth[1].ptr<float>();
		float* m2 = smooth[2].ptr<float>();
		float* d0 = dest[0].ptr<float>();
		float* d1 = dest[1].ptr<float>();
		float* d2 = dest[2].ptr<float>();

		const float sqrt2_sr_divpi = float((sqrt(2.0) * sigma_range) / sqrt(CV_PI));
		const float sqrt2_sr_inv = float(1.0 / (sqrt(2.0) * sigma_range));
		const float eps2 = delta * sqrt2_sr_inv;
		const float exp2 = exp(-eps2 * eps2);
		const float erf2 = erf(eps2);
		const int simdsize = src[0].size().area() / 8;

		__m256 mexp2 = _mm256_set1_ps(exp2);
		__m256 merf2 = _mm256_set1_ps(erf2);

		__m256 mflt_epsilon = _mm256_set1_ps(+FLT_EPSILON);
		__m256 msqrt2_sr_inv = _mm256_set1_ps(sqrt2_sr_inv);
		__m256 msqrt2_sr_divpi = _mm256_set1_ps(sqrt2_sr_divpi);
		__m256 mdelta = _mm256_set1_ps(delta);
		__m256 mm2f = _mm256_set1_ps(2.f);
		__m256 mm1f = _mm256_set1_ps(-1.f);
		for (int i = 0; i < simdsize; i++)
		{
			const __m256 ms0 = _mm256_load_ps(s0);
			const __m256 ms1 = _mm256_load_ps(s1);
			const __m256 ms2 = _mm256_load_ps(s2);
			const __m256 mdiffb = _mm256_sub_ps(ms0, _mm256_load_ps(m0));
			const __m256 mdiffg = _mm256_sub_ps(ms1, _mm256_load_ps(m1));
			const __m256 mdiffr = _mm256_sub_ps(ms2, _mm256_load_ps(m2));
			__m256 mdiff = _mm256_add_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(mdiffr, mdiffr, _mm256_fmadd_ps(mdiffg, mdiffg, _mm256_mul_ps(mdiffb, mdiffb)))), mflt_epsilon);
			__m256 meps1 = _mm256_mul_ps(_mm256_fmadd_ps(mm2f, mdiff, mdelta), msqrt2_sr_inv);
			__m256 mcoeff = _mm256_div_ps(_mm256_sub_ps(_mm256_exp_ps(_mm256_mul_ps(mm1f, _mm256_mul_ps(meps1, meps1))), mexp2),
				_mm256_add_ps(_mm256_erf_ps(meps1), merf2));
			//const float coeff = (exp(-eps1 * eps1) - exp2) / (erf(eps1) + erf2);
			__m256 ma = _mm256_div_ps(_mm256_mul_ps(mcoeff, msqrt2_sr_divpi), mdiff);
			_mm256_store_ps(d0, _mm256_fmadd_ps(ma, mdiffb, ms0));
			_mm256_store_ps(d1, _mm256_fmadd_ps(ma, mdiffg, ms1));
			_mm256_store_ps(d2, _mm256_fmadd_ps(ma, mdiffr, ms2));
			s0 += 8; s1 += 8; s2 += 8;
			m0 += 8; m1 += 8; m2 += 8;
			d0 += 8; d1 += 8; d2 += 8;
		}
		/*for (int i = 0; i < src[0].size().area(); i++)
		{
			const float  diffb = (s0[i] - m0[i]);
			const float  diffg = (s1[i] - m1[i]);
			const float  diffr = (s2[i] - m2[i]);
			const float diff = sqrt(diffb * diffb + diffg * diffg + diffr * diffr) + FLT_EPSILON;
			const float eps1 = (2.f * diff+delta) * sqrt2_sr_inv;
			const float coeff = (exp(-eps1 * eps1) - exp2) / (erf(eps1) + erf2);

			d0[i] = s0[i] + coeff * sqrt2_sr_divpi / diff * diffb;
			d1[i] = s1[i] + coeff * sqrt2_sr_divpi / diff * diffg;
			d2[i] = s2[i] + coeff * sqrt2_sr_divpi / diff * diffr;
		}*/
	}

	static void bilateralFilterLocalStatisticsPrior32FC3(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float delta)
	{
		CV_Assert(src.channels() == 3);
		CV_Assert(src.depth() == CV_32F);

		dest.create(src.size(), src.type());

		cv::Mat smooth;
		int D = 2 * (int)ceil(sigma_space * 3.f) + 1;
		cv::GaussianBlur(src, smooth, cv::Size(D, D), sigma_space);
		const float* s = src.ptr<float>();
		float* sm = smooth.ptr<float>();
		float* dst = dest.ptr<float>();

		const float sqrt2_sr_divpi = float(sqrt(2.0 / CV_PI) * sigma_range);
		const float sqrt2_sr_inv = float(1.0 / (sqrt(2.0) * sigma_range));
		const float eps1 = delta * sqrt2_sr_inv;
		const float exp1 = exp(-eps1 * eps1);
		const float erf1 = erf(eps1);
		for (int i = 0; i < src.size().area(); i++)
		{
			const float diffb = sm[3 * i + 0] - s[3 * i + 0];
			const float diffg = sm[3 * i + 1] - s[3 * i + 1];
			const float diffr = sm[3 * i + 2] - s[3 * i + 2];
			const float diff = sqrt(diffb * diffb + diffg * diffg + diffr * diffr);
			const float eps2 = (delta + 2.f * diff) * sqrt2_sr_inv;
			const float coeff = (exp1 - exp(-eps2 * eps2)) / (erf1 + erf(eps2) + FLT_EPSILON);

			dst[3 * i + 0] = s[3 * i + 0] + coeff * sqrt2_sr_divpi / diff * diffb;
			dst[3 * i + 1] = s[3 * i + 1] + coeff * sqrt2_sr_divpi / diff * diffg;
			dst[3 * i + 2] = s[3 * i + 2] + coeff * sqrt2_sr_divpi / diff * diffr;
		}
	}

	static void bilateralFilterLocalStatisticsPrior32FC1(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float delta)
	{
		CV_Assert(src.channels() == 1);
		CV_Assert(src.depth() == CV_32F);

		dest.create(src.size(), src.type());

		cv::Mat smooth;
		int D = 2 * (int)ceil(sigma_space * 3.f) + 1;
		cv::GaussianBlur(src, smooth, cv::Size(D, D), sigma_space);
		const float* s = src.ptr<float>();
		float* m = smooth.ptr<float>();
		float* d = dest.ptr<float>();

		const float sqrt2_sr_divpi = float(sqrt(2.0 / CV_PI) * sigma_range);
		const float sqrt2_sr_inv = float(1.0 / (sqrt(2.0) * sigma_range));
		const float eps2 = delta * sqrt2_sr_inv;
		const float exp2 = exp(-eps2 * eps2);
		const float erf2 = erf(eps2);
		for (int i = 0; i < src.size().area(); i++)
		{
			const float  diffb = s[i] - m[i];
			const float diff = abs(diffb) + FLT_EPSILON;
			const float eps2 = (delta + 2.f * diff) * sqrt2_sr_inv;
			const float coeff = (exp(-eps2 * eps2) - exp2) / (erf(eps2) + erf2);

			d[i] = s[i] + coeff * sqrt2_sr_divpi / diff * diffb;
		}
	}

	void bilateralFilterLocalStatisticsPrior(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float delta)
	{
		if (src.type() == CV_32FC3)
		{
			bilateralFilterLocalStatisticsPrior32FC3(src, dest, sigma_range, sigma_space, delta);
		}
		else if (src.type() == CV_32FC1)
		{
			bilateralFilterLocalStatisticsPrior32FC1(src, dest, sigma_range, sigma_space, delta);
		}
		else
		{
			std::cout << "no support this type" << std::endl;
		}
	}

	void bilateralFilterLocalStatisticsPriorInternal(const std::vector<cv::Mat>& src, const cv::Mat& vecW, std::vector<cv::Mat>& split_inter, const float sigma_range, const float sigma_space, const float delta, cv::Mat& mask, BFLSPSchedule schedule, float* lut)
	{
		CV_Assert(src.size() == 3);
		CV_Assert(src[0].depth() == CV_32F);

		const float* s0 = src[0].ptr<float>();
		const float* s1 = src[1].ptr<float>();
		const float* s2 = src[2].ptr<float>();
		const float* w_ = vecW.ptr<float>();
		const float* mask_ptr = mask.ptr<float>();
		float* i0 = split_inter[0].ptr<float>();
		float* i1 = split_inter[1].ptr<float>();
		float* i2 = split_inter[2].ptr<float>();

		const float sqrt2_sr_divpi = float((sqrt(2.0) * sigma_range) / sqrt(CV_PI));
		const float sqrt2_sr_inv = float(1.0 / (sqrt(2.0) * sigma_range));
		const float eps2 = delta * sqrt2_sr_inv;
		const float exp2 = exp(-eps2 * eps2);
		const float erf2 = erf(eps2);
		const int simdsize8 = src[0].size().area() / 8;
		if (schedule == BFLSPSchedule::Compute)
		{
			const bool isSIMD = true;

			if (isSIMD)
			{
				if constexpr (true)//delta = 0.f
				{
					const __m256 mexp2 = _mm256_set1_ps(exp2);
					const __m256 merf2 = _mm256_set1_ps(erf2 / sqrt2_sr_divpi);
					const __m256 mflt_epsilon = _mm256_set1_ps(+FLT_EPSILON);
					const __m256 msqrt2_sr_inv2 = _mm256_set1_ps(sqrt2_sr_inv * 2.f);
					const __m256 msqrt2_sr_divpi = _mm256_set1_ps(1.f / sqrt2_sr_divpi);
					const __m256 mm1f = _mm256_set1_ps(-1.f);
					for (int i = 0; i < simdsize8; i++)
					{
						const __m256 mmask = _mm256_load_ps(mask_ptr);
						if (_mm256_movemask_ps(mmask) == 0)
						{
							s0 += 8; s1 += 8; s2 += 8;
							i0 += 8; i1 += 8; i2 += 8;
							w_ += 8;
							mask_ptr += 8;
							continue;
						}

						const __m256 ms0 = _mm256_load_ps(s0);
						const __m256 ms1 = _mm256_load_ps(s1);
						const __m256 ms2 = _mm256_load_ps(s2);
						const __m256 mi0 = _mm256_load_ps(i0);
						const __m256 mi1 = _mm256_load_ps(i1);
						const __m256 mi2 = _mm256_load_ps(i2);
						const __m256 mw_ = _mm256_load_ps(w_);

						const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), mw_);
						const __m256 mdiffb = _mm256_fnmadd_ps(norm, mi0, ms0);
						const __m256 mdiffg = _mm256_fnmadd_ps(norm, mi1, ms1);
						const __m256 mdiffr = _mm256_fnmadd_ps(norm, mi2, ms2);
						const __m256 mdiff = _mm256_add_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(mdiffr, mdiffr, _mm256_fmadd_ps(mdiffg, mdiffg, _mm256_mul_ps(mdiffb, mdiffb)))), mflt_epsilon);

						const __m256 meps1 = _mm256_mul_ps(mdiff, msqrt2_sr_inv2);
						const __m256 mcoeff = _mm256_sub_ps(_mm256_exp_ps(_mm256_mul_ps(mm1f, _mm256_mul_ps(meps1, meps1))), mexp2);
						const __m256 ma = _mm256_div_ps(mcoeff, _mm256_mul_ps(mdiff, _mm256_fmadd_ps(msqrt2_sr_divpi, _mm256_erf_ps(meps1), merf2)));

						_mm256_store_ps(i0, _mm256_blendv_ps(mi0, _mm256_mul_ps(mw_, _mm256_fmadd_ps(ma, mdiffb, ms0)), mmask));
						_mm256_store_ps(i1, _mm256_blendv_ps(mi1, _mm256_mul_ps(mw_, _mm256_fmadd_ps(ma, mdiffg, ms1)), mmask));
						_mm256_store_ps(i2, _mm256_blendv_ps(mi2, _mm256_mul_ps(mw_, _mm256_fmadd_ps(ma, mdiffr, ms2)), mmask));

						s0 += 8; s1 += 8; s2 += 8;
						i0 += 8; i1 += 8; i2 += 8;
						w_ += 8;
						mask_ptr += 8;
					}
				}
				else
				{
					const __m256 mexp2 = _mm256_set1_ps(exp2);
					const __m256 merf2 = _mm256_set1_ps(erf2);
					const __m256 mflt_epsilon = _mm256_set1_ps(+FLT_EPSILON);
					const __m256 msqrt2_sr_inv = _mm256_set1_ps(sqrt2_sr_inv);
					const __m256 msqrt2_sr_divpi = _mm256_set1_ps(sqrt2_sr_divpi);
					const __m256 mdelta = _mm256_set1_ps(delta);
					const __m256 mm2f = _mm256_set1_ps(2.f);
					const __m256 mm1f = _mm256_set1_ps(-1.f);
					for (int i = 0; i < simdsize8; i++)
					{
						const __m256 mmask = _mm256_load_ps(mask_ptr);
						if (_mm256_movemask_ps(mmask) == 0)
						{
							s0 += 8; s1 += 8; s2 += 8;
							i0 += 8; i1 += 8; i2 += 8;
							w_ += 8;
							mask_ptr += 8;
							continue;
						}

						const __m256 ms0 = _mm256_load_ps(s0);
						const __m256 ms1 = _mm256_load_ps(s1);
						const __m256 ms2 = _mm256_load_ps(s2);
						const __m256 mi0 = _mm256_load_ps(i0);
						const __m256 mi1 = _mm256_load_ps(i1);
						const __m256 mi2 = _mm256_load_ps(i2);
						const __m256 mvecw = _mm256_load_ps(w_);
						const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), mvecw);
						const __m256 mdiffb = _mm256_fnmadd_ps(norm, mi0, ms0);
						const __m256 mdiffg = _mm256_fnmadd_ps(norm, mi1, ms1);
						const __m256 mdiffr = _mm256_fnmadd_ps(norm, mi2, ms2);
						const __m256 mdiff = _mm256_add_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(mdiffr, mdiffr, _mm256_fmadd_ps(mdiffg, mdiffg, _mm256_mul_ps(mdiffb, mdiffb)))), mflt_epsilon);
						const __m256 meps1 = _mm256_mul_ps(_mm256_fmadd_ps(mm2f, mdiff, mdelta), msqrt2_sr_inv);
						const __m256 mcoeff = _mm256_div_ps(
							_mm256_sub_ps(_mm256_exp_ps(_mm256_mul_ps(mm1f, _mm256_mul_ps(meps1, meps1))), mexp2),
							_mm256_add_ps(_mm256_erf_ps(meps1), merf2));
						//const float coeff = (exp(-eps1 * eps1) - exp2) / (erf(eps1) + erf2);
						const __m256 ma = _mm256_div_ps(_mm256_mul_ps(mcoeff, msqrt2_sr_divpi), mdiff);

						_mm256_store_ps(i0, _mm256_blendv_ps(mi0, _mm256_mul_ps(mvecw, _mm256_fmadd_ps(ma, mdiffb, ms0)), mmask));
						_mm256_store_ps(i1, _mm256_blendv_ps(mi1, _mm256_mul_ps(mvecw, _mm256_fmadd_ps(ma, mdiffg, ms1)), mmask));
						_mm256_store_ps(i2, _mm256_blendv_ps(mi2, _mm256_mul_ps(mvecw, _mm256_fmadd_ps(ma, mdiffr, ms2)), mmask));

						s0 += 8; s1 += 8; s2 += 8;
						i0 += 8; i1 += 8; i2 += 8;
						w_ += 8;
						mask_ptr += 8;
					}
				}
			}
			else
			{
				for (int i = 0; i < src[0].size().area(); i++)
				{
					if (mask_ptr[i] == 0)continue;

					const float norm = 1.f / (w_[i] + FLT_EPSILON);
					const float diffb = (s0[i] - norm * i0[i]);
					const float diffg = (s1[i] - norm * i1[i]);
					const float diffr = (s2[i] - norm * i2[i]);
					const float diff = sqrt(diffb * diffb + diffg * diffg + diffr * diffr);

					const float eps1 = (2.f * diff + delta) * sqrt2_sr_inv;
					const float numer = (exp(-eps1 * eps1) - exp2) * sqrt2_sr_divpi;
					const float denom = (erf(eps1) + erf2) * diff;
					const float coeff = numer / (denom + FLT_EPSILON);
					i0[i] = w_[i] * (s0[i] + coeff * diffb);
					i1[i] = w_[i] * (s1[i] + coeff * diffg);
					i2[i] = w_[i] * (s2[i] + coeff * diffr);
				}
			}
		}
		else
		{
			//not work
			const __m256 mexp2 = _mm256_set1_ps(exp2);
			const __m256 merf2 = _mm256_set1_ps(erf2);
			const __m256 mflt_epsilon = _mm256_set1_ps(+FLT_EPSILON);
			const __m256 msqrt2_sr_inv = _mm256_set1_ps(sqrt2_sr_inv);
			const __m256 msqrt2_sr_divpi = _mm256_set1_ps(sqrt2_sr_divpi);
			const __m256 mdelta = _mm256_set1_ps(delta);
			const __m256 mm2f = _mm256_set1_ps(2.f);
			const __m256 mm1f = _mm256_set1_ps(-1.f);

			for (int i = 0; i < simdsize8; i++)
			{
				const __m256 ms0 = _mm256_load_ps(s0);
				const __m256 ms1 = _mm256_load_ps(s1);
				const __m256 ms2 = _mm256_load_ps(s2);
				const __m256 mi0 = _mm256_load_ps(i0);
				const __m256 mi1 = _mm256_load_ps(i1);
				const __m256 mi2 = _mm256_load_ps(i2);
				const __m256 mvecw = _mm256_load_ps(w_);

				const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), mvecw);
				const __m256 mdiffb = _mm256_fnmadd_ps(norm, mi0, ms0);
				const __m256 mdiffg = _mm256_fnmadd_ps(norm, mi1, ms1);
				const __m256 mdiffr = _mm256_fnmadd_ps(norm, mi2, ms2);
				const __m256 mdiff = _mm256_add_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(mdiffr, mdiffr, _mm256_fmadd_ps(mdiffg, mdiffg, _mm256_mul_ps(mdiffb, mdiffb)))), mflt_epsilon);

				const __m256 mcoeff = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_min_ps(_mm256_mul_ps(_mm256_set1_ps(10.f), mdiff), _mm256_set1_ps(4430.f))), 4);

				const __m256 mmask = _mm256_load_ps(mask_ptr);
				_mm256_store_ps(i0, _mm256_blendv_ps(mi0, _mm256_mul_ps(mvecw, _mm256_fmadd_ps(mcoeff, mdiffb, ms0)), mmask));
				_mm256_store_ps(i1, _mm256_blendv_ps(mi1, _mm256_mul_ps(mvecw, _mm256_fmadd_ps(mcoeff, mdiffg, ms1)), mmask));
				_mm256_store_ps(i2, _mm256_blendv_ps(mi2, _mm256_mul_ps(mvecw, _mm256_fmadd_ps(mcoeff, mdiffr, ms2)), mmask));

				s0 += 8; s1 += 8; s2 += 8;
				i0 += 8; i1 += 8; i2 += 8;
				w_ += 8;
				mask_ptr += 8;
			}
		}
	}
}