#include "stdafx.h"

#define USE_FMA_VYV
//optimization float AVX FMA
//optimization double AVX FMA

using namespace std;
using namespace cv;

namespace cp
{
	//functions for coefficients of VYV
	int getInvertMatrix(double* inv_A, double* A, const int order)
	{
		int ret = 0;

		double* col_j_ptr, * col_k_ptr, * inv_col_k_ptr;
		double temp;

		double c[VYV_ORDER_MAX];
		double d[VYV_ORDER_MAX];

		col_k_ptr = A;
		for (int k = 0; k < order - 1; ++k, col_k_ptr += order)
		{
			double scale = 0.0;
			for (int i = k; i < order; ++i)
			{
				if ((temp = fabs(col_k_ptr[i])) > scale)
				{
					scale = temp;
				}
			}

			if (scale == 0.0)
				return ret; // Singular matrix

			for (int i = k; i < order; ++i)
			{
				col_k_ptr[i] /= scale;
			}

			double sum = 0.0;
			for (int i = k; i < order; ++i)
			{
				sum += col_k_ptr[i] * col_k_ptr[i];
			}

			temp = (col_k_ptr[k] >= 0.0) ? sqrt(sum) : -sqrt(sum);
			col_k_ptr[k] += temp;
			c[k] = temp * col_k_ptr[k];
			d[k] = -scale * temp;

			col_j_ptr = col_k_ptr + order;
			for (int j = k + 1; j < order; ++j, col_j_ptr += order)
			{
				double scale = 0.0;
				for (int i = k; i < order; ++i)
				{
					scale += col_k_ptr[i] * col_j_ptr[i];
				}

				scale /= c[k];

				for (int i = k; i < order; ++i)
				{
					col_j_ptr[i] -= scale * col_k_ptr[i];
				}
			}
		}

		d[order - 1] = col_k_ptr[order - 1];

		if (d[order - 1] == 0.0)
		{
			return ret; // Singular matrix
		}

		inv_col_k_ptr = inv_A;
		for (int k = 0; k < order; ++k, inv_col_k_ptr += order)
		{
			for (int i = 0; i < order; ++i)
			{
				inv_col_k_ptr[i] = -A[k] * A[i] / c[0];
			}

			inv_col_k_ptr[k] += 1.0;

			col_j_ptr = A + order;
			for (int j = 1; j < order - 1; ++j, col_j_ptr += order)
			{
				double scale = 0.0;
				for (int i = j; i < order; ++i)
				{
					scale += col_j_ptr[i] * inv_col_k_ptr[i];
				}

				scale /= c[j];

				for (int i = j; i < order; ++i)
				{
					inv_col_k_ptr[i] -= scale * col_j_ptr[i];
				}
			}

			inv_col_k_ptr[order - 1] /= d[order - 1];

			for (int i = order - 2; i >= 0; --i)
			{
				double sum = 0.0;
				col_j_ptr = A + order * (i + 1);
				for (int j = i + 1; j < order; ++j, col_j_ptr += order)
				{
					sum += col_j_ptr[i] * inv_col_k_ptr[j];
				}

				inv_col_k_ptr[i] = (inv_col_k_ptr[i] - sum) / d[i];
			}
		}

		ret = 1;
		return ret;
	}

	void expandPoleProduct(const complex<double>* poles, const int order, double* dest)
	{
		assert(order <= VYV_ORDER_MAX);

		complex<double> denominator[VYV_ORDER_MAX + 1];

		denominator[0] = poles[0];
		denominator[1] = complex<double>(-1.0, 0.0);

		for (int i = 1; i < order; ++i)
		{
			denominator[i + 1] = -denominator[i];

			for (int j = i; j > 0; --j)
			{
				denominator[j] = denominator[j] * poles[i] - denominator[j - 1];
			}

			denominator[0] = denominator[0] * poles[i];
		}

		for (int i = 1; i <= order; ++i)
		{
			dest[i] = (denominator[i] / denominator[0]).real();
		}

		dest[0] = 1.0;
		for (int i = 1; i <= order; ++i)
		{
			dest[0] += dest[i];
		}
	}

	inline double getVariance(const complex<double>* poles, const int order, const double q)
	{
		complex<double> sum = { 0.0, 0.0 };

		for (int k = 0; k < order; ++k)
		{
			const complex<double> z = pow(poles[k], 1 / q);
			sum += z / pow(z - 1.0, 2.0);
		}

		return 2.0 * sum.real();
	}

	inline double getVariancedifferential(const complex<double>* poles, const int order, const double q)
	{
		complex<double> sum = { 0.0, 0.0 };
		for (int k = 0; k < order; ++k)
		{
			const complex<double> z = pow(poles[k], 1 / q);
			sum += z * log(z) * (z + 1.0) / pow(z - 1.0, 3);
		}

		return (2.0 / q) * sum.real();
	}

	inline double optimizeQ(const complex<double>* poles, const int order, const double sigma, const double q_init)
	{
		const double sigma2 = sigma * sigma;

		double q = q_init;
		for (int i = 0; i < VYV_NUM_NEWTON_ITERATIONS; ++i)
		{
			q -= (getVariance(poles, order, q) - sigma2) / getVariancedifferential(poles, order, q);
		}
		return q;
	}

#pragma region VYV_Naive
	template<typename Type>
	GaussianFilterVYV_Naive<Type>::GaussianFilterVYV_Naive(cv::Size imgSize, Type sigma, int order)
		: SpatialFilterBase(imgSize, cp::typeToCVDepth<Type>())
	{
		this->gf_order = order;
		this->sigma = sigma;
		truncate_r = (int)ceil(4.0 * sigma);
		const int matrixSize = order * order;

		h = new Type[truncate_r + order];

		//optimized unscaled pole locations
		const complex<double> poles0[VYV_ORDER_MAX - VYV_ORDER_MIN + 1][5] =
		{
			{ { 1.41650, 1.00829 },{ 1.41650, -1.00829 },{ 1.86543, 0.00000 } },
			{ { 1.13228, 1.28114 },{ 1.13228, -1.28114 },{ 1.78534, 0.46763 },{ 1.78534, -0.46763 } },
			{ { 0.86430, 1.45389 },{ 0.86430, -1.45389 },{ 1.61433, 0.83134 },{ 1.61433, -0.83134 },{ 1.87504,	0 } }
		};
		complex<double> poles[VYV_ORDER_MAX];
		double filter64F[VYV_ORDER_MAX + 1];
		double A[VYV_ORDER_MAX * VYV_ORDER_MAX], invA[VYV_ORDER_MAX * VYV_ORDER_MAX];

		const double q = optimizeQ(poles0[order - VYV_ORDER_MIN], order, sigma, sigma / (Type)2.0);

		for (int i = 0; i < order; ++i)
		{
			poles[i] = pow(poles0[order - VYV_ORDER_MIN][i], 1 / q);
		}

		expandPoleProduct(poles, order, filter64F);

		//matrix for inverse boundary processing
		for (int i = 0; i < matrixSize; ++i)
		{
			A[i] = 0.0;
		}

		for (int i = 0; i < order; ++i)
		{
			A[i + order * i] = 1.0;
			for (int j = 1; j <= order; ++j)
			{
				A[i + order * ((i + j < order) ? i + j : 2 * order - (i + j) - 1)] += filter64F[j];
			}
		}

		getInvertMatrix(invA, A, order);

		for (int i = 0; i < matrixSize; ++i)
		{
			invA[i] *= filter64F[0];
		}

		b = (Type)filter64F[0];
		for (int i = 1; i <= order; ++i)
		{
			a[i] = (Type)filter64F[i];
		}
		a[0] = 0.0;

		for (int i = 0; i < matrixSize; ++i)
		{
			M[i] = (Type)invA[i];
		}

		for (int i = 0; i < truncate_r + order; ++i)
		{
			h[i] = (i <= 0) ? b : (Type)0.0;

			for (int j = 1; j <= order && j <= i; ++j)
			{
				h[i] -= a[j] * h[i - j];
			}
		}
	}

	template<typename Type>
	GaussianFilterVYV_Naive<Type>::~GaussianFilterVYV_Naive()
	{
		delete[] h;
		h = nullptr;
	}

	template<typename Type>
	void GaussianFilterVYV_Naive<Type>::horizontalbody(const cv::Mat& src, cv::Mat& dest)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		Type accum[VYV_ORDER_MAX];

		//forward processing
		for (int y = 0; y < height; ++y)
		{
			//boundary processing
			const Type* srcPtr = src.ptr<Type>(y);
			Type* dstPtr = dest.ptr<Type>(y);

			for (int j = 0; j < gf_order; ++j)
			{
				accum[j] = (Type)0.0;
				for (int i = -j; i < truncate_r; ++i)
				{
					accum[j] += h[j + i] * srcPtr[ref_lborder(-i, borderType)];
				}
			}
			for (int j = 0; j < gf_order; ++j)
			{
				dstPtr[j] = accum[j];
			}

			//IIR filtering
			switch (gf_order)
			{
			case 3:
				for (int x = 3; x < width; ++x)
				{
					dstPtr[x] = b * srcPtr[x]
						- a[1] * dstPtr[x - 1]
						- a[2] * dstPtr[x - 2]
						- a[3] * dstPtr[x - 3];
				}
				break;
			case 4:
				for (int x = 4; x < width; ++x)
				{
					dstPtr[x] = b * srcPtr[x]
						- a[1] * dstPtr[x - 1]
						- a[2] * dstPtr[x - 2]
						- a[3] * dstPtr[x - 3]
						- a[4] * dstPtr[x - 4];
				}
				break;
			case 5:
				for (int x = 5; x < width; ++x)
				{
					dstPtr[x] = b * srcPtr[x]
						- a[1] * dstPtr[x - 1]
						- a[2] * dstPtr[x - 2]
						- a[3] * dstPtr[x - 3]
						- a[4] * dstPtr[x - 4]
						- a[5] * dstPtr[x - 5];
				}
				break;
			}
		}

		//backward processing
		for (int y = height - 1; y >= 0; --y)
		{
			//boundary processing
			Type* dstPtr = dest.ptr<Type>(y);

			Type q[6];
			for (int j = 0; j < gf_order; ++j)
			{
				q[j] = dstPtr[width - (gf_order - j)];
			}

			for (int j = 0; j < gf_order; ++j)
			{
				Type accum = (Type)0.0;
				for (int i = 0; i < gf_order; ++i)
				{
					accum += M[j + gf_order * i] * q[i];
				}

				dstPtr[width - (gf_order - j)] = accum;
			}

			//IIR filtering
			switch (gf_order)
			{
			case 3:
				for (int x = width - 4; x >= 0; --x)
				{
					dstPtr[x] = b * dstPtr[x]
						- a[1] * dstPtr[x + 1]
						- a[2] * dstPtr[x + 2]
						- a[3] * dstPtr[x + 3];
				}
				break;
			case 4:
				for (int x = width - 5; x >= 0; --x)
				{
					dstPtr[x] = b * dstPtr[x]
						- a[1] * dstPtr[x + 1]
						- a[2] * dstPtr[x + 2]
						- a[3] * dstPtr[x + 3]
						- a[4] * dstPtr[x + 4];
				}
				break;
			case 5:
				for (int x = width - 6; x >= 0; --x)
				{
					dstPtr[x] = b * dstPtr[x]
						- a[1] * dstPtr[x + 1]
						- a[2] * dstPtr[x + 2]
						- a[3] * dstPtr[x + 3]
						- a[4] * dstPtr[x + 4]
						- a[5] * dstPtr[x + 5];
				}
				break;
			}
		}
	}

	template<typename Type>
	void GaussianFilterVYV_Naive<Type>::verticalbody(cv::Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		Type accum[VYV_ORDER_MAX];

		int offset[VYV_ORDER_MAX + 1];
		for (int i = 0; i <= gf_order; ++i)
		{
			offset[i] = i * width;
		}

		//forward processing

		//boundary processing
		Type* imgPtr = img.ptr<Type>(0);
		for (int x = 0; x < width; ++x)
		{
			for (int j = 0; j < gf_order; ++j)
			{
				accum[j] = 0;
				for (int i = -j; i < truncate_r; ++i)
				{
					accum[j] += h[j + i] * imgPtr[ref_tborder(-i, width, borderType) + x];
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				imgPtr[x + offset[i]] = accum[i];
			}
		}

		//IIR filtering
		switch (gf_order)
		{
		case 3:
			for (int y = 3; y < height; ++y)
			{
				imgPtr = img.ptr<Type>(y);
				for (int x = 0; x < width; ++x)
				{
					imgPtr[x] = b * imgPtr[x]
						- a[1] * imgPtr[x - offset[1]]
						- a[2] * imgPtr[x - offset[2]]
						- a[3] * imgPtr[x - offset[3]];
				}
			}
			break;
		case 4:
			for (int y = 4; y < height; ++y)
			{
				imgPtr = img.ptr<Type>(y);
				for (int x = 0; x < width; ++x)
				{
					imgPtr[x] = b * imgPtr[x]
						- a[1] * imgPtr[x - offset[1]]
						- a[2] * imgPtr[x - offset[2]]
						- a[3] * imgPtr[x - offset[3]]
						- a[4] * imgPtr[x - offset[4]];
				}
			}
			break;
		case 5:
			for (int y = 5; y < height; ++y)
			{
				imgPtr = img.ptr<Type>(y);
				for (int x = 0; x < width; ++x)
				{
					imgPtr[x] = b * imgPtr[x]
						- a[1] * imgPtr[x - offset[1]]
						- a[2] * imgPtr[x - offset[2]]
						- a[3] * imgPtr[x - offset[3]]
						- a[4] * imgPtr[x - offset[4]]
						- a[5] * imgPtr[x - offset[5]];
				}
			}
			break;
		}

		//backward processing

		//boundary processing
		imgPtr = img.ptr<Type>(0);
		for (int x = width - 1; x >= 0; --x)
		{
			for (int j = 0; j < gf_order; ++j)
			{
				accum[j] = 0;
				for (int i = 0; i < gf_order; ++i)
				{
					accum[j] += M[j + gf_order * i] * imgPtr[width * (height - gf_order + i) + x];
				}
			}
			for (int j = 0; j < gf_order; ++j)
			{
				imgPtr[width * (height - gf_order + j) + x] = accum[j];
			}
		}

		//IIR filtering
		switch (gf_order)
		{
		case 3:
			for (int y = height - 4; y >= 0; --y)
			{
				imgPtr = img.ptr<Type>(y);
				for (int x = width - 1; x >= 0; --x)
				{
					imgPtr[x] = b * imgPtr[x]
						- a[1] * imgPtr[x + offset[1]]
						- a[2] * imgPtr[x + offset[2]]
						- a[3] * imgPtr[x + offset[3]];
				}
			}
			break;
		case 4:
			for (int y = height - 5; y >= 0; --y)
			{
				imgPtr = img.ptr<Type>(y);
				for (int x = width - 1; x >= 0; --x)
				{
					imgPtr[x] = b * imgPtr[x]
						- a[1] * imgPtr[x + offset[1]]
						- a[2] * imgPtr[x + offset[2]]
						- a[3] * imgPtr[x + offset[3]]
						- a[4] * imgPtr[x + offset[4]];
				}
			}
			break;
		case 5:
			for (int y = height - 6; y >= 0; --y)
			{
				imgPtr = img.ptr<Type>(y);
				for (int x = width - 1; x >= 0; --x)
				{
					imgPtr[x] = b * imgPtr[x]
						- a[1] * imgPtr[x + offset[1]]
						- a[2] * imgPtr[x + offset[2]]
						- a[3] * imgPtr[x + offset[3]]
						- a[4] * imgPtr[x + offset[4]]
						- a[5] * imgPtr[x + offset[5]];
				}
			}
			break;
		}
	}

	template<class Type>
	void GaussianFilterVYV_Naive<Type>::body(const cv::Mat& _src, cv::Mat& dst, const int borderTYpe)
	{
		this->borderType = borderType;

		Mat src;
		if (_src.depth() == depth)
			src = _src;
		else
			_src.convertTo(src, depth);

		if (dst.size() != imgSize || dst.depth() != depth)
			dst.create(imgSize, depth);

		horizontalbody(src, dst);
		verticalbody(dst);
	}

	template class GaussianFilterVYV_Naive<float>;
	template class GaussianFilterVYV_Naive<double>;

#pragma endregion

	//for float AVX
#pragma region VYV_32F_AVX
	void GaussianFilterVYV_AVX_32F::allocBuffer()
	{
		truncate_r = (int)ceil(4.0 * sigma);
		const int matrixSize = gf_order * gf_order;

		delete[] h;
		h = new float[truncate_r + gf_order];

		//optimized unscaled pole locations.
		const complex<double> poles0[VYV_ORDER_MAX - VYV_ORDER_MIN + 1][5] =
		{
			{ { 1.41650, 1.00829 },{ 1.41650, -1.00829 },{ 1.86543, 0.00000 } },
			{ { 1.13228, 1.28114 },{ 1.13228, -1.28114 },{ 1.78534, 0.46763 },{ 1.78534, -0.46763 } },
			{ { 0.86430, 1.45389 },{ 0.86430, -1.45389 },{ 1.61433, 0.83134 },{ 1.61433, -0.83134 },{ 1.87504,	0 } }
		};
		complex<double> poles[VYV_ORDER_MAX];
		double filter64F[VYV_ORDER_MAX + 1];
		cv::AutoBuffer<double> A(VYV_ORDER_MAX * VYV_ORDER_MAX);
		cv::AutoBuffer<double> invA(VYV_ORDER_MAX * VYV_ORDER_MAX);

		const double q = optimizeQ(poles0[gf_order - VYV_ORDER_MIN], gf_order, sigma, sigma / 2.0);

		for (int i = 0; i < gf_order; ++i)
		{
			poles[i] = pow(poles0[gf_order - VYV_ORDER_MIN][i], 1.0 / q);
		}

		expandPoleProduct(poles, gf_order, filter64F);

		//matrix for inverse boundary processing
		for (int i = 0; i < matrixSize; ++i)
		{
			A[i] = 0.0;
		}

		for (int i = 0; i < gf_order; ++i)
		{
			A[i + gf_order * i] = 1.0;
			for (int j = 1; j <= gf_order; ++j)
			{
				A[i + gf_order * ((i + j < gf_order) ? i + j : 2 * gf_order - (i + j) - 1)] += filter64F[j];
			}
		}
		getInvertMatrix(invA, A, gf_order);

		for (int i = 0; i < matrixSize; ++i)
		{
			invA[i] *= filter64F[0];
		}

		coeff[0] = (float)filter64F[0];
		for (int i = 1; i <= gf_order; ++i)
		{
			coeff[i] = (float)filter64F[i];
		}

		for (int i = 0; i < matrixSize; ++i)
		{
			M[i] = invA[i];
		}

		for (int i = 0; i < truncate_r + gf_order; ++i)
		{
			h[i] = (i <= 0) ? coeff[0] : 0.f;

			for (int j = 1; j <= gf_order && j <= i; ++j)
			{
				h[i] -= coeff[j] * h[i - j];
			}
		}
	}

	GaussianFilterVYV_AVX_32F::GaussianFilterVYV_AVX_32F(cv::Size imgSize, float sigma, int order)
		: SpatialFilterBase(imgSize, CV_32F)
	{
		this->gf_order = order;
		this->sigma = sigma;
		allocBuffer();
	}

	GaussianFilterVYV_AVX_32F::GaussianFilterVYV_AVX_32F(const int dest_depth)
	{
		this->dest_depth = dest_depth;
		this->depth = CV_32F;
	}

	GaussianFilterVYV_AVX_32F::~GaussianFilterVYV_AVX_32F()
	{
		delete[] h;
	}


	void GaussianFilterVYV_AVX_32F::horizontalFilterVLoadGatherTransposeStore(const cv::Mat& _src, cv::Mat& _dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256 input;
		__m256d inputD;
		__m256 patch[8];
		__m256 patch_t[8];//required!
		__m256d boundDlo[VYV_ORDER_MAX];
		__m256d boundDhi[VYV_ORDER_MAX];

		const __m256i mm_offset = _mm256_set_epi32(7 * width, 6 * width, 5 * width, 4 * width, 3 * width, 2 * width, width, 0);

		//forward direction
		for (int y = 0; y < height; y += 8)
		{
			//boundary processing
			const float* src = _src.ptr<float>(y);
			float* dst = _dst.ptr<float>(y);

			for (int k = 0; k < gf_order; ++k)
			{
				patch[k] = _mm256_setzero_ps();
				for (int x = -k; x < truncate_r; ++x)
				{				
					float* s = (float*)(src + ref_lborder(-x, borderType));
					input = _mm256_i32gather_ps(s, mm_offset, sizeof(float));
#ifdef USE_FMA_VYV
					patch[k] = _mm256_fmadd_ps(_mm256_set1_ps(h[k + x]), input, patch[k]);
#else
					patch[j] = _mm256_add_ps(patch[j], _mm256_mul_ps(_mm256_set1_ps(h[j + i]), input));
#endif
				}
			}

			switch (gf_order)
			{
			case 3:
			{
				//initial 8 row
				src += gf_order;
				for (int x = gf_order; x < 8; ++x)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[x] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[x - 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[x - 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[x - 1],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3]));
#endif
					++src;
				}
				_mm256_transpose8_ps(patch,patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[5]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[1],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7]));
#endif
					++src;

					for (int i = 3; i < 8; ++i)
					{
						input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3]));
#endif
						++src;
					}
					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst += 8;
				}
				break;
			}
			case 4:
			{
				//initial 8 row
				src += gf_order;
				for (int i = gf_order; i < 8; ++i)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4]));
#endif
					++src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[4]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[5]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[6]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[2],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7]));
#endif
					++src;

					for (int i = 4; i < 8; ++i)
					{
						input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4]));
#endif
						++src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst += 8;
				}
				break;
			}
			case 5:
			{
				//itinial 8 row
				src += gf_order;
				for (int i = gf_order; i < 8; ++i)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i - 5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i - 5]));
#endif
					++src;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[5],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[3]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[4]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[5]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[2],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[6]));
#endif
					++src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[4] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[3],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[7]));
#endif
					++src;

					for (int i = 5; i < 8; ++i)
					{
						input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i - 5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
											_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
												_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i - 5]));
#endif
						++src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst += 8;
				}
				break;
			}
			}
		}

		//backward direction
		for (int y = height - 8; y >= 0; y -= 8)
		{
			//boundary processing
			const float* src = _dst.ptr<float>(y);
			float* dst = _dst.ptr<float>(y) + width - 8;

			//boundary processing in double precision for stability
			for (int i = 0; i < gf_order; ++i)
			{
				boundDlo[i] = _mm256_setzero_pd();

				for (int j = 0; j < gf_order; ++j)
				{
					int refx = width - gf_order + j;
					inputD = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
					boundDlo[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), inputD, boundDlo[i]);
#else
					boundDlo[i] = _mm256_add_pd(boundDlo[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), inputD));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				boundDhi[i] = _mm256_setzero_pd();

				for (int j = 0; j < gf_order; ++j)
				{
					int refx = width - gf_order + j;
					inputD = _mm256_set_pd(src[7 * width + refx], src[6 * width + refx], src[5 * width + refx], src[4 * width + refx]);
#ifdef USE_FMA_VYV
					boundDhi[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), inputD, boundDhi[i]);
#else
					boundDhi[i] = _mm256_add_pd(boundDhi[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), inputD));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m128*)& patch[8 - gf_order + i] = _mm256_cvtpd_ps(boundDlo[i]);
				*(((__m128*) & patch[8 - gf_order + i]) + 1) = _mm256_cvtpd_ps(boundDhi[i]);
			}

			switch (gf_order)
			{
			case 3:
			{
				//last 8 row
				src += width - 4;
				for (int i = 4; i >= 0; --i)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3]));
#endif
					--src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[7] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[2]));
#endif
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[6] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1]));
#endif
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[5] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[6],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0]));
#endif
					--src;

					for (int i = 4; i >= 0; --i)
					{
						input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3]));
#endif
						--src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst -= 8;
				}
				break;
			}
			case 4:
			{
				//last 8 row
				src += width - 5;
				for (int i = 3; i >= 0; --i)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4]));
#endif
					--src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[7] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[3]));
#endif
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[6] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[2]));
#endif
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[5] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[6],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[1]));
#endif
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[4] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[5],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0]));
#endif
					--src;

					for (int i = 3; i >= 0; --i)
					{
						input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4]));
#endif
						--src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst -= 8;
				}
				break;
			}
			case 5:
			{
				//last 8 row
				src += width - 6;
				for (int i = 2; i >= 0; --i)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i + 5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i + 5]));
#endif
					--src;

				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[7] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[4]));
#endif
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[6] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[3]));
#endif					
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[5] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[6],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[2]));
#endif					
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[4] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[5],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[1]));
#endif
					--src;

					input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[5],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[4],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[0]));
#endif
					--src;

					for (int i = 2; i >= 0; --i)
					{
						input = _mm256_i32gather_ps(src, mm_offset, sizeof(float));
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i + 5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
											_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
												_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i + 5]));
#endif
						--src;
					}
					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst -= 8;
				}
				break;
			}
			default:
				break;
			}
		}
	}

	void GaussianFilterVYV_AVX_32F::horizontalFilterVLoadSetTransposeStore(const cv::Mat& _src, cv::Mat& _dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256 input;
		__m256d inputD;
		__m256 patch[8];
		__m256 patch_t[8];
		__m256d boundDlo[VYV_ORDER_MAX];
		__m256d boundDhi[VYV_ORDER_MAX];

		//forward direction
		for (int y = 0; y < height; y += 8)
		{
			//boundary processing
			const float* src = _src.ptr<float>(y);
			float* dst = _dst.ptr<float>(y);
			for (int j = 0; j < gf_order; ++j)
			{
				patch[j] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
					float* s = (float*)(src + ref_lborder(-i, borderType));
					input = _mm256_set_ps(s[7 * width], s[6 * width], s[5 * width], s[4 * width], s[3 * width], s[2 * width], s[width], s[0]);
#ifdef USE_FMA_VYV
					patch[j] = _mm256_fmadd_ps(_mm256_set1_ps(h[j + i]), input, patch[j]);
#else
					patch[j] = _mm256_add_ps(patch[j], _mm256_mul_ps(_mm256_set1_ps(h[j + i]), input));
#endif
				}
			}

			switch (gf_order)
			{
			case 3:
			{
				//initial 8 row
				src += gf_order;
				for (int i = gf_order; i < 8; ++i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3]));
#endif
					++src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[5]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[1],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7]));
#endif
					++src;

					for (int i = 3; i < 8; ++i)
					{
						input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3]));
#endif
						++src;
					}
					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst += 8;
				}
				break;
			}
			case 4:
			{
				//initial 8 row
				src += gf_order;
				for (int i = gf_order; i < 8; ++i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4]));
#endif
					++src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[4]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[5]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[6]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[2],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7]));
#endif
					++src;

					for (int i = 4; i < 8; ++i)
					{
						input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4]));
#endif
						++src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst += 8;
				}
				break;
			}
			case 5:
			{
				//itinial 8 row
				src += gf_order;
				for (int i = gf_order; i < 8; ++i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i - 5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i - 5]));
#endif
					++src;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[5],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[3]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[4]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[5]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[2],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[6]));
#endif
					++src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[4] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[3],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[7]));
#endif
					++src;

					for (int i = 5; i < 8; ++i)
					{
						input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i - 5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
											_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
												_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i - 5]));
#endif
						++src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst += 8;
				}
				break;
			}
			}
		}

		//backward direction
		for (int y = height - 8; y >= 0; y -= 8)
		{
			//boundary processing
			const float* src = _dst.ptr<float>(y);
			float* dst = _dst.ptr<float>(y) + width - 8;

			//boundary processing in double precision for stability
			for (int i = 0; i < gf_order; ++i)
			{
				boundDlo[i] = _mm256_setzero_pd();

				for (int j = 0; j < gf_order; ++j)
				{
					int refx = width - gf_order + j;
					inputD = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
					boundDlo[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), inputD, boundDlo[i]);
#else
					boundDlo[i] = _mm256_add_pd(boundDlo[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), inputD));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				boundDhi[i] = _mm256_setzero_pd();

				for (int j = 0; j < gf_order; ++j)
				{
					int refx = width - gf_order + j;
					inputD = _mm256_set_pd(src[7 * width + refx], src[6 * width + refx], src[5 * width + refx], src[4 * width + refx]);
#ifdef USE_FMA_VYV
					boundDhi[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), inputD, boundDhi[i]);
#else
					boundDhi[i] = _mm256_add_pd(boundDhi[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), inputD));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m128*)& patch[8 - gf_order + i] = _mm256_cvtpd_ps(boundDlo[i]);
				*(((__m128*) & patch[8 - gf_order + i]) + 1) = _mm256_cvtpd_ps(boundDhi[i]);
			}

			switch (gf_order)
			{
			case 3:
			{
				//last 8 row
				src += width - 4;
				for (int i = 4; i >= 0; --i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3]));
#endif
					--src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[7] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[2]));
#endif
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[6] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1]));
#endif
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[5] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[6],
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0]));
#endif
					--src;

					for (int i = 4; i >= 0; --i)
					{
						input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3]));
#endif
						--src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst -= 8;
				}
				break;
			}
			case 4:
			{
				//last 8 row
				src += width - 5;
				for (int i = 3; i >= 0; --i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4]));
#endif
					--src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[7] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[3]));
#endif
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[6] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[2]));
#endif
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[5] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[6],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[1]));
#endif
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[4] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[5],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0]));
#endif
					--src;

					for (int i = 3; i >= 0; --i)
					{
						input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input)))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4]));
#endif
						--src;
					}

					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst -= 8;
				}
				break;
			}
			case 5:
			{
				//last 8 row
				src += width - 6;
				for (int i = 2; i >= 0; --i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i + 5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i + 5]));
#endif
					--src;

				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[7] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[4]));
#endif
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[6] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[3]));
#endif					
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[5] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[6],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[2]));
#endif					
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[4] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[5],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[1]));
#endif
					--src;

					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[5],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[4],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[5])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[0]));
#endif
					--src;

					for (int i = 2; i >= 0; --i)
					{
						input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
						patch[i] =
#ifdef USE_FMA_VYV
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i + 5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
											_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
												_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
							_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
								_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
								_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4])),
								_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i + 5]));
#endif
						--src;
					}
					_mm256_transpose8_ps(patch, patch_t);
					_mm256_storepatch_ps(dst, patch_t, width);
					dst -= 8;
				}
				break;
			}
			default:
				break;
			}
		}
	}

	void GaussianFilterVYV_AVX_32F::horizontalFilterVLoadSetTransposeStoreOrder5(const cv::Mat& _src, cv::Mat& _dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		int i, j, x, y;
		__m256 input;
		__m256d inputD;
		__m256 patch[8];
		__m256 patch_t[8];
		__m256d boundDlo[VYV_ORDER_MAX];
		__m256d boundDhi[VYV_ORDER_MAX];

		//forward direction
		for (y = 0; y < height; y += 8)
		{
			//boundary processing
			const float* src = _src.ptr<float>(y);
			float* dst = _dst.ptr<float>(y);
			for (j = 0; j < 5; ++j)
			{
				patch[j] = _mm256_setzero_ps();
				for (i = -j; i < truncate_r; ++i)
				{
					float* s = (float*)(src + ref_lborder(-i, borderType));
					input = _mm256_set_ps(s[7 * width], s[6 * width], s[5 * width], s[4 * width], s[3 * width], s[2 * width], s[width], s[0]);

#ifdef USE_FMA_VYV
					patch[j] = _mm256_fmadd_ps(_mm256_set1_ps(h[j + i]), input, patch[j]);
#else
					patch[j] = _mm256_add_ps(patch[j], _mm256_mul_ps(_mm256_set1_ps(h[j + i]), input));
#endif
				}
			}

			//itinial 8 row
			src += 5;
			for (i = 5; i < 8; ++i)
			{
				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[i] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i - 5],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i - 5]));
#endif
				++src;
			}

			_mm256_transpose8_ps(patch, patch_t);
			_mm256_storepatch_ps(dst, patch_t, width);
			dst += 8;

			//IIR filtering
			for (x = 8; x < width; x += 8)
			{
				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[0] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[3],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[5],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[5])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[4])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[3]));
#endif
				++src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[1] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[4],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[5])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[4]));
#endif
				++src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[2] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[5],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[6],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[6])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[5]));
#endif
				++src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[3] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[6],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[2],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[2])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[6]));
#endif
				++src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[4] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[7],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[3],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[3])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[2])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[7]));
#endif
				++src;

				for (i = 5; i < 8; ++i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i - 5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i - 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i - 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i - 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i - 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i - 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i - 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i - 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i - 4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i - 5]));
#endif
					++src;
				}

				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst += 8;
			}

		}

		//backward direction
		for (y = height - 8; y >= 0; y -= 8)
		{
			//boundary processing
			const float* src = _dst.ptr<float>(y);
			float* dst = _dst.ptr<float>(y) + width - 8;

			//boundary processing in double precision for stability
			for (i = 0; i < 5; ++i)
			{
				boundDlo[i] = _mm256_setzero_pd();

				for (j = 0; j < 5; ++j)
				{
					int refx = width - 5 + j;
					inputD = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
					boundDlo[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + 5 * j]), inputD, boundDlo[i]);
#else
					boundDlo[i] = _mm256_add_pd(boundDlo[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), inputD));
#endif

				}
			}
			for (i = 0; i < 5; ++i)
			{
				boundDhi[i] = _mm256_setzero_pd();

				for (j = 0; j < 5; ++j)
				{
					int refx = width - 5 + j;
					inputD = _mm256_set_pd(src[7 * width + refx], src[6 * width + refx], src[5 * width + refx], src[4 * width + refx]);
#ifdef USE_FMA_VYV
					boundDhi[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + 5 * j]), inputD, boundDhi[i]);
#else
					boundDhi[i] = _mm256_add_pd(boundDhi[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), inputD));
#endif
				}
			}
			for (i = 0; i < 5; ++i)
			{
				*(__m128*)& patch[8 - 5 + i] = _mm256_cvtpd_ps(boundDlo[i]);
				*(((__m128*) & patch[8 - 5 + i]) + 1) = _mm256_cvtpd_ps(boundDhi[i]);
			}

			//last 8 row
			src += width - 6;
			for (i = 2; i >= 0; --i)
			{
				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[i] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i + 5],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i + 5]));
#endif
				--src;

			}
			_mm256_transpose8_ps(patch, patch_t);
			_mm256_storepatch_ps(dst, patch_t, width);
			dst -= 8;

			//IIR filtering
			for (x = width - 16; x >= 0; x -= 8)
			{
				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[7] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[4],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[3],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[2],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[1],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[0],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[2])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[3])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[4]));
#endif

				--src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[6] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[3],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[2],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[1],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[0],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[7],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[2])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[3]));
#endif					
				--src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[5] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[2],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[1],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[0],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[7],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[6],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[6])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[1])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[2]));
#endif					
				--src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[4] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[1],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[0],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[7],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[6],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[5],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[5])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[6])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[0])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[1]));
#endif
				--src;

				input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
				patch[3] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[0],
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[7],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[6],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[5],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[4],
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
					_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
						_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
						_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[4])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[5])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[6])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[7])),
						_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[0]));
#endif
				--src;

				for (i = 2; i >= 0; --i)
				{
					input = _mm256_set_ps(src[7 * width], src[6 * width], src[5 * width], src[4 * width], src[3 * width], src[2 * width], src[width], src[0]);
					patch[i] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), patch[i + 5],
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), patch[i + 4],
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), patch[i + 3],
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), patch[i + 2],
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), patch[i + 1],
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), input),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), patch[i + 1])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), patch[i + 2])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), patch[i + 3])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), patch[i + 4])),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), patch[i + 5]));
#endif
					--src;
				}
				_mm256_transpose8_ps(patch, patch_t);
				_mm256_storepatch_ps(dst, patch_t, width);
				dst -= 8;
			}
		}
	}

	void GaussianFilterVYV_AVX_32F::verticalFilter(cv::Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256 accum[VYV_ORDER_MAX];
		int offset[VYV_ORDER_MAX + 1];
		__m256d boundDlo[VYV_ORDER_MAX];
		__m256d boundDhi[VYV_ORDER_MAX];

		for (int i = 0; i <= gf_order; ++i)
		{
			offset[i] = i * width;
		}

		const float* img_ptr = img.ptr<float>();
		//boundary processing
		for (int x = 0; x < width; x += 8)
		{
			for (int j = 0; j < gf_order; ++j)
			{
				accum[j] = _mm256_setzero_ps();
				for (int i = -j; i < truncate_r; ++i)
				{
#ifdef USE_FMA_VYV
					accum[j] = _mm256_fmadd_ps(_mm256_set1_ps(h[j + i]), *(__m256*) & img_ptr[ref_tborder(-i, width, borderType) + x], accum[j]);
#else
					accum[j] = _mm256_add_ps(accum[j], _mm256_mul_ps(_mm256_set1_ps(h[j + i]), *(__m256*) & src_ptr[UREF(-i) + x]));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m256*)& img_ptr[x + offset[i]] = accum[i];
			}
		}

		//forward direction
		switch (gf_order)
		{
		case 3:
		{
			for (int y = 3; y < height; ++y)
			{
				img_ptr = img.ptr<float>(y);
				for (int x = 0; x < width; x += 8)
				{
					*(__m256*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(img_ptr + x - offset[3]),
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(img_ptr + x - offset[2]),
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(img_ptr + x - offset[1]),
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(img_ptr + x)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(src_ptr + x)),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(dst_ptr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(dst_ptr + x - offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(dst_ptr + x - offset[3])));
#endif
				}
			}
			break;
		}
		case 4:
		{
			for (int y = 4; y < height; ++y)
			{
				img_ptr = img.ptr<float>(y);
				for (int x = 0; x < width; x += 8)
				{
					*(__m256*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(img_ptr + x - offset[4]),
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(img_ptr + x - offset[3]),
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(img_ptr + x - offset[2]),
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(img_ptr + x - offset[1]),
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(img_ptr + x))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(src_ptr + x)),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(dst_ptr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(dst_ptr + x - offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(dst_ptr + x - offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(dst_ptr + x - offset[4])));
#endif
				}
			}
			break;
		}
		case 5:
		{
			for (int y = 5; y < height; ++y)
			{
				img_ptr = img.ptr<float>(y);
				for (int x = 0; x < width; x += 8)
				{
					*(__m256*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), *(__m256*)(img_ptr + x - offset[5]),
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(img_ptr + x - offset[4]),
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(img_ptr + x - offset[3]),
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(img_ptr + x - offset[2]),
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(img_ptr + x - offset[1]),
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(img_ptr + x)))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(src_ptr + x)),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(dst_ptr + x - offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(dst_ptr + x - offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(dst_ptr + x - offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(dst_ptr + x - offset[4]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), *(__m256*)(dst_ptr + x - offset[5])));
#endif
				}
			}
			break;
		}
		}

		//backward direction
		for (int x = width - 4; x >= 0; x -= 4)
		{
			//boundary processing in double precision for stability
			for (int i = 0; i < gf_order; ++i)
			{
				boundDhi[i] = _mm256_setzero_pd();
				for (int j = 0; j < gf_order; ++j)
				{
#ifdef USE_FMA_VYV
					boundDhi[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), _mm256_cvtps_pd(*(__m128*)(img_ptr - offset[gf_order - j - 1] + x)), boundDhi[i]);
#else
					boundDhi[i] = _mm256_add_pd(boundDhi[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), _mm256_cvtps_pd(*(__m128*)(dst_ptr - offset[order - j - 1] + x))));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m128*)(img_ptr - offset[gf_order - i - 1] + x) = _mm256_cvtpd_ps(boundDhi[i]);
			}
		}

		switch (gf_order)
		{
		case 3:
		{
			for (int y = height - 4; y >= 0; --y)
			{
				img_ptr = img.ptr<float>(y);
				for (int x = width - 8; x >= 0; x -= 8)
				{
					*(__m256*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(img_ptr + x + offset[3]),
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(img_ptr + x + offset[2]),
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(img_ptr + x + offset[1]),
									_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(img_ptr + x)))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(dst_ptr + x)),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(dst_ptr + x + offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(dst_ptr + x + offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(dst_ptr + x + offset[3])));
#endif
				}
			}
			break;
		}
		case 4:
		{
			for (int y = height - 5; y >= 0; --y)
			{
				img_ptr = img.ptr<float>(y);
				for (int x = width - 8; x >= 0; x -= 8)
				{
					*(__m256*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(img_ptr + x + offset[4]),
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(img_ptr + x + offset[3]),
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(img_ptr + x + offset[2]),
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(img_ptr + x + offset[1]),
										_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(img_ptr + x))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(dst_ptr + x)),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(dst_ptr + x + offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(dst_ptr + x + offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(dst_ptr + x + offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(dst_ptr + x + offset[4])));
#endif					
				}
			}
			break;
		}
		case 5:
		{
			for (int y = height - 6; y >= 0; --y)
			{
				img_ptr = img.ptr<float>(y);
				for (int x = width - 8; x >= 0; x -= 8)
				{
					*(__m256*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_ps(_mm256_set1_ps(coeff[5]), *(__m256*)(img_ptr + x + offset[5]),
							_mm256_fnmadd_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(img_ptr + x + offset[4]),
								_mm256_fnmadd_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(img_ptr + x + offset[3]),
									_mm256_fnmadd_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(img_ptr + x + offset[2]),
										_mm256_fnmadd_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(img_ptr + x + offset[1]),
											_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(img_ptr + x)))))));
#else
						_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
							_mm256_mul_ps(_mm256_set1_ps(coeff[0]), *(__m256*)(dst_ptr + x)),
							_mm256_mul_ps(_mm256_set1_ps(coeff[1]), *(__m256*)(dst_ptr + x + offset[1]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[2]), *(__m256*)(dst_ptr + x + offset[2]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[3]), *(__m256*)(dst_ptr + x + offset[3]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[4]), *(__m256*)(dst_ptr + x + offset[4]))),
							_mm256_mul_ps(_mm256_set1_ps(coeff[5]), *(__m256*)(dst_ptr + x + offset[5])));
#endif
				}
			}
			break;
		}
		}
	}

	void GaussianFilterVYV_AVX_32F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		this->borderType = borderType;

		CV_Assert(src.cols % 8 == 0);
		CV_Assert(src.rows % 8 == 0);
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

		const bool isInplace = src.data == dst.data;

		if (isInplace)
		{
			if (src.depth() == CV_32F)
			{
				src.copyTo(inter);
				horizontalFilterVLoadGatherTransposeStore(inter, dst);
				verticalFilter(dst);
			}
			else
			{
				src.convertTo(inter, CV_32F);
				inter2.create(inter.size(), CV_32F);
				horizontalFilterVLoadGatherTransposeStore(inter, inter2);
				verticalFilter(inter2);
				inter2.convertTo(dst, dest_depth);
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				if (dest_depth == CV_32F)
				{
					dst.create(imgSize, CV_32F);
					horizontalFilterVLoadGatherTransposeStore(src, dst);
					//horizontalFilterVLoadSetTransposeStore(_src, dst);
					verticalFilter(dst);
				}
				else
				{
					inter.create(imgSize, CV_32F);
					horizontalFilterVLoadGatherTransposeStore(src, inter);
					//horizontalFilterVLoadSetTransposeStore(_src, dst);
					verticalFilter(inter);
					inter.convertTo(dst, dest_depth);
				}
			}
			else
			{
				src.convertTo(inter, CV_32F);
				if (dest_depth == CV_32F)
				{
					dst.create(imgSize, CV_32F);
					horizontalFilterVLoadGatherTransposeStore(inter, dst);
					//horizontalFilterVLoadSetTransposeStore(_src, dst);
					verticalFilter(dst);
				}
				else
				{
					inter2.create(imgSize, CV_32F);
					horizontalFilterVLoadGatherTransposeStore(inter, inter2);
					//horizontalFilterVLoadSetTransposeStore(_src, dst);
					verticalFilter(inter2);
					inter2.convertTo(dst, dest_depth);
				}
			}
		}
	}

	void GaussianFilterVYV_AVX_32F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		if (this->sigma != sigma || this->gf_order != order || imgSize != src.size())
		{
			this->sigma = sigma;
			this->gf_order = order;
			this->imgSize = src.size();
			allocBuffer();
		}

		body(src, dst, borderType);
	}


#pragma endregion

#pragma region VYV_64F_AVX

	void GaussianFilterVYV_AVX_64F::allocBuffer()
	{
		truncate_r = (int)ceil(4.0 * sigma);
		const int matrixSize = gf_order * gf_order;

		h = new double[truncate_r + gf_order];

		// Optimized unscaled pole locations.
		const complex<double> poles0[VYV_ORDER_MAX - VYV_ORDER_MIN + 1][5] =
		{
			{ { 1.41650, 1.00829 },{ 1.41650, -1.00829 },{ 1.86543, 0.00000 } },
			{ { 1.13228, 1.28114 },{ 1.13228, -1.28114 },{ 1.78534, 0.46763 },{ 1.78534, -0.46763 } },
			{ { 0.86430, 1.45389 },{ 0.86430, -1.45389 },{ 1.61433, 0.83134 },{ 1.61433, -0.83134 },{ 1.87504,	0 } }
		};
		complex<double> poles[VYV_ORDER_MAX];
		double filter64F[VYV_ORDER_MAX + 1];
		double A[VYV_ORDER_MAX * VYV_ORDER_MAX], invA[VYV_ORDER_MAX * VYV_ORDER_MAX];

		const double q = optimizeQ(poles0[gf_order - VYV_ORDER_MIN], gf_order, sigma, sigma / 2.0);

		for (int i = 0; i < gf_order; ++i)
		{
			poles[i] = pow(poles0[gf_order - VYV_ORDER_MIN][i], 1.0 / q);
		}

		expandPoleProduct(poles, gf_order, filter64F);

		//matrix for inverse boundary processing
		for (int i = 0; i < matrixSize; ++i)
		{
			A[i] = 0.0;
		}

		for (int i = 0; i < gf_order; ++i)
		{
			A[i + gf_order * i] = 1.0;
			for (int j = 1; j <= gf_order; ++j)
			{
				A[i + gf_order * ((i + j < gf_order) ? i + j : 2 * gf_order - (i + j) - 1)] += filter64F[j];
			}
		}
		getInvertMatrix(invA, A, gf_order);

		for (int i = 0; i < matrixSize; ++i)
		{
			invA[i] *= filter64F[0];
		}

		coeff[0] = filter64F[0];
		for (int i = 1; i <= gf_order; ++i)
		{
			coeff[i] = filter64F[i];
		}

		for (int i = 0; i < matrixSize; ++i)
		{
			M[i] = invA[i];
		}

		for (int i = 0; i < truncate_r + gf_order; ++i)
		{
			h[i] = (i <= 0) ? coeff[0] : 0.0;

			for (int j = 1; j <= gf_order && j <= i; ++j)
			{
				h[i] -= coeff[j] * h[i - j];
			}
		}
	}

	GaussianFilterVYV_AVX_64F::GaussianFilterVYV_AVX_64F(cv::Size imgSize, double sigma, int order)
		: SpatialFilterBase(imgSize, CV_64F)
	{
		this->gf_order = order;
		this->sigma = sigma;
		allocBuffer();
	}

	GaussianFilterVYV_AVX_64F::GaussianFilterVYV_AVX_64F(const int dest_depth)
	{
		this->dest_depth = dest_depth;
		this->depth = CV_64F;
	}

	GaussianFilterVYV_AVX_64F::~GaussianFilterVYV_AVX_64F()
	{
		delete[] h;
	}


	void GaussianFilterVYV_AVX_64F::horizontalbody(const cv::Mat& src_, cv::Mat& dest)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256d input;
		__m256d patch[4];
		__m256d patch_t[4];

		//forward direction
		for (int y = 0; y < height; y += 4)
		{
			const double* src = src_.ptr<double>(y);
			double* dst = dest.ptr<double>(y);
			switch (gf_order)
			{
			case 3:
			{
				//boundary processing
				for (int j = 0; j < gf_order; ++j)
				{
					patch[j] = _mm256_setzero_pd();
					for (int i = -j; i < truncate_r; ++i)
					{
						int refx = ref_lborder(-i, borderType);
						input = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
						patch[j] = _mm256_fmadd_pd(_mm256_set1_pd(h[j + i]), input, patch[j]);
#else
						patch[j] = _mm256_add_pd(patch[j], _mm256_mul_pd(_mm256_set1_pd(h[j + i]), input));
#endif	
					}
				}
				src += gf_order;
				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[3] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[0],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
								_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0]));
#endif	

				++src;
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst += 4;

				//IIR filtering
				for (int x = 4; x < width; x += 4)
				{
					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[3],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[1]));
#endif						
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[0],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[2]));
#endif						
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3]));
#endif						
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0]));
#endif						
					++src;

					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst += 4;
				}
				break;
			}
			case 4:
			{
				//boundary processing
				for (int j = 0; j < gf_order; ++j)
				{
					patch[j] = _mm256_setzero_pd();
					for (int i = -j; i < truncate_r; ++i)
					{
						int refx = ref_lborder(-i, borderType);
						input = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);

#ifdef USE_FMA_VYV
						patch[j] = _mm256_fmadd_pd(_mm256_set1_pd(h[j + i]), input, patch[j]);
#else
						patch[j] = _mm256_add_pd(patch[j], _mm256_mul_pd(_mm256_set1_pd(h[j + i]), input));
#endif	
					}
				}
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst += 4;

				//IIR filtering
				src = src_.ptr<double>(y) + gf_order;
				for (int x = 4; x < width; x += 4)
				{

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[3],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[0]));
#endif	
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[0],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[1]));
#endif	
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[2]));
#endif	
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[3]));
#endif	
					++src;

					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst += 4;
				}
				break;
			}
			case 5:
			{
				//boundary processing
				__m256d extpatch[5];
				for (int j = 0; j < gf_order; ++j)
				{
					extpatch[j] = _mm256_setzero_pd();
					for (int i = -j; i < truncate_r; ++i)
					{
						int refx = ref_lborder(-i, borderType);;
						input = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
						extpatch[j] = _mm256_fmadd_pd(_mm256_set1_pd(h[j + i]), input, extpatch[j]);
#else
						extpatch[j] = _mm256_add_pd(extpatch[j], _mm256_mul_pd(_mm256_set1_pd(h[j + i]), input));
#endif					
					}
				}
				_mm256_transpose4_pd(extpatch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst += 4;

				//itinial 4 row
				patch[0] = extpatch[4];
				src += gf_order;
				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[1] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[0],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[0],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[0])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[3])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[0]));
#endif
				++src;

				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[2] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[1],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[0])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[3])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[1]));
#endif
				++src;

				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[3] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[2],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[3])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[2]));
#endif
				++src;

				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst += 4;

				//IIR filtering
				for (int x = 8; x < width; x += 8)
				{
					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[3],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[3]));
#endif
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[3],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), extpatch[0],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[0]));
#endif
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[0],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), extpatch[1],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[1]));
#endif
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[1],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), extpatch[2],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[2]));
#endif
					++src;

					_mm256_transpose4_pd(extpatch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst += 4;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[2],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), extpatch[3],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[3]));
#endif
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[3],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[0],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[0]));
#endif

					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[0],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[1]));
#endif
					++src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[2]));
#endif
					++src;

					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst += 4;
				}
				break;
			}
			}
		}

		//backward direction
		for (int y = height - 4; y >= 0; y -= 4)
		{
			const double* src = dest.ptr<double>(y);
			double* dst = dest.ptr<double>(y) + width - 4;
			switch (gf_order)
			{
			case 3:
			{
				//boundary processing
				for (int i = 0; i < gf_order; ++i)
				{
					patch[4 - gf_order + i] = _mm256_setzero_pd();

					for (int j = 0; j < gf_order; ++j)
					{
						int refx = width - gf_order + j;
						input = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
						patch[4 - gf_order + i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), input, patch[4 - gf_order + i]);
#else
						patch[4 - order + i] = _mm256_add_pd(patch[4 - order + i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), input));
#endif	
					}
				}

				//last 4 row
				src += width - 4;
				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[0] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
								_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3]));
#endif	
				--src;

				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst -= 4;

				//IIR filtering
				for (int x = width - 8; x >= 0; x -= 4)
				{
					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[0],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[2]));
#endif	
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[3],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[1]));
#endif	
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0]));
#endif	
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))));
#else

						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3]));
#endif	
					--src;

					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst -= 4;
				}
				break;
			}
			case 4:
			{
				//boundary processing
				for (int i = 0; i < gf_order; ++i)
				{
					patch[4 - gf_order + i] = _mm256_setzero_pd();

					for (int j = 0; j < gf_order; ++j)
					{
						int refx = width - gf_order + j;
						input = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
						patch[4 - gf_order + i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), input, patch[4 - gf_order + i]);
#else
						patch[4 - order + i] = _mm256_add_pd(patch[4 - order + i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), input));
#endif
					}
				}
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst -= 4;

				src += width - 5;
				//IIR filtering
				for (int x = width - 8; x >= 0; x -= 4)
				{
					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[0],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[3]));
#endif	
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[3],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[2]));
#endif	
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[1]));
#endif	
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[0]));
#endif	
					--src;

					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst -= 4;
				}
				break;
			}
			case 5:
			{
				//boundary processing
				__m256d extpatch[5];
				for (int i = 0; i < gf_order; ++i)
				{
					extpatch[5 - gf_order + i] = _mm256_setzero_pd();

					for (int j = 0; j < gf_order; ++j)
					{
						int refx = width - gf_order + j;
						input = _mm256_set_pd(src[3 * width + refx], src[2 * width + refx], src[width + refx], src[refx]);
#ifdef USE_FMA_VYV
						extpatch[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), input, extpatch[i]);
#else
						extpatch[i] = _mm256_add_pd(extpatch[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), input));
#endif
					}
				}
				_mm256_transpose4_pd(&extpatch[1], patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst -= 4;

				//last 8 row
				patch[3] = extpatch[0];
				extpatch[0] = extpatch[1];
				extpatch[1] = extpatch[2];
				extpatch[2] = extpatch[3];
				extpatch[3] = extpatch[4];
				src += width - 6;

				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[2] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[3],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[3],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[3])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[0])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[3]));
#endif
				--src;

				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[1] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[2],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[3])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[0])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[2]));
#endif
				--src;

				input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
				patch[0] =
#ifdef USE_FMA_VYV
					_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[1],
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
					_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
						_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
						_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[0])),
						_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[1]));
#endif
				--src;
				_mm256_transpose4_pd(patch, patch_t);
				_mm256_storeupatch_pd(dst, patch_t, width);
				dst -= 4;

				//IIR filtering
				for (int x = width - 16; x >= 0; x -= 8)
				{
					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[1],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[0],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[0]));
#endif
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), patch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[0],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), extpatch[3],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[3]));
#endif
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[1] =
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[2]));
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					extpatch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), patch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), patch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[2],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), extpatch[1],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), patch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[1]));
#endif
					--src;

					_mm256_transpose4_pd(extpatch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst -= 4;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[3] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), patch[0],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[3],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[2],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[1],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), extpatch[0],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), patch[0]));
#endif
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[2] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[3],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[2],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[1],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), extpatch[0],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[3],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[3]));
#endif
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[1] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[2],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[1],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), extpatch[0],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[3],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[2],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[2]));
#endif
					--src;

					input = _mm256_set_pd(src[3 * width], src[2 * width], src[width], src[0]);
					patch[0] =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), extpatch[1],
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), extpatch[0],
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), patch[3],
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), patch[2],
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), patch[1],
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), input),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), patch[1])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), patch[2])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), patch[3])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), extpatch[0])),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), extpatch[1]));
#endif
					--src;

					_mm256_transpose4_pd(patch, patch_t);
					_mm256_storeupatch_pd(dst, patch_t, width);
					dst -= 4;
				}
				break;
			}
			default:
				break;
			}
		}
	}

	void GaussianFilterVYV_AVX_64F::verticalbody(cv::Mat& img)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		__m256d accum[VYV_ORDER_MAX];
		int offset[VYV_ORDER_MAX + 1];

		for (int i = 0; i <= gf_order; ++i)
		{
			offset[i] = i * width;
		}

		//boundary processing
		double* img_ptr = img.ptr<double>(0);
		for (int x = 0; x < width; x += 4)
		{
			for (int j = 0; j < gf_order; ++j)
			{
				accum[j] = _mm256_setzero_pd();
				for (int i = -j; i < truncate_r; ++i)
				{
#ifdef USE_FMA_VYV
					accum[j] = _mm256_fmadd_pd(_mm256_set1_pd(h[j + i]), *(__m256d*) & img_ptr[ref_tborder(-i, width, borderType) + x], accum[j]);
#else
					accum[j] = _mm256_add_pd(accum[j], _mm256_mul_pd(_mm256_set1_pd(h[j + i]), *(__m256d*) & img_ptr[UREF(-i) + x]));
#endif
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m256d*)& img_ptr[x + offset[i]] = accum[i];
			}
		}

		//forward direction
		switch (gf_order)
		{
		case 3:
		{
			for (int y = 3; y < height; ++y)
			{
				img_ptr = img.ptr<double>(y);
				for (int x = 0; x < width; x += 4)
				{
					*(__m256d*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x - offset[3]),
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x - offset[2]),
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x - offset[1]),
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x - offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x - offset[3])));
#endif
				}
			}
			break;
		}
		case 4:
		{
			for (int y = 4; y < height; ++y)
			{
				img_ptr = img.ptr<double>(y);
				for (int x = 0; x < width; x += 4)
				{
					*(__m256d*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x - offset[4]),
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x - offset[3]),
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x - offset[2]),
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x - offset[1]),
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x - offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x - offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x - offset[4])));
#endif
				}
			}
			break;
		}
		case 5:
		{
			for (int y = 5; y < height; ++y)
			{
				img_ptr = img.ptr<double>(y);
				for (int x = 0; x < width; x += 4)
				{
					*(__m256d*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), *(__m256d*)(img_ptr + x - offset[5]),
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x - offset[4]),
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x - offset[3]),
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x - offset[2]),
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x - offset[1]),
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x - offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x - offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x - offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x - offset[4]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), *(__m256d*)(img_ptr + x - offset[5])));
#endif						
				}
			}
			break;
		}
		}

		//backward direction
		img_ptr = img.ptr<double>(height - 1);
		for (int x = width - 4; x >= 0; x -= 4)
		{
			//boundary processing
			for (int i = 0; i < gf_order; ++i)
			{
				accum[i] = _mm256_setzero_pd();
				for (int j = 0; j < gf_order; ++j)
				{
#ifdef USE_FMA_VYV
					accum[i] = _mm256_fmadd_pd(_mm256_set1_pd(M[i + gf_order * j]), *(__m256d*)(img_ptr - offset[gf_order - j - 1] + x), accum[i]);
#else
					accum[i] = _mm256_add_pd(accum[i], _mm256_mul_pd(_mm256_set1_pd(M[i + order * j]), *(__m256d*)(img_ptr - offset[order - j - 1] + x)));
#endif				
				}
			}
			for (int i = 0; i < gf_order; ++i)
			{
				*(__m256d*)(img_ptr - offset[gf_order - i - 1] + x) = accum[i];
			}
		}

		switch (gf_order)
		{
		case 3:
		{
			for (int y = height - 4; y >= 0; --y)
			{
				img_ptr = img.ptr<double>(y);
				for (int x = width - 4; x >= 0; x -= 4)
				{
					*(__m256d*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x + offset[3]),
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x + offset[2]),
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x + offset[1]),
									_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x + offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x + offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x + offset[3])));
#endif
				}
			}
			break;
		}
		case 4:
		{
			for (int y = height - 5; y >= 0; --y)
			{
				img_ptr = img.ptr<double>(y);
				for (int x = width - 4; x >= 0; x -= 4)
				{
					*(__m256d*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x + offset[4]),
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x + offset[3]),
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x + offset[2]),
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x + offset[1]),
										_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x))))));
#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)),
								_mm256_mul_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x + offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x + offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x + offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x + offset[4])));
#endif
				}
			}
			break;
		}
		case 5:
		{
			for (int y = height - 6; y >= 0; --y)
			{
				img_ptr = img.ptr<double>(y);
				for (int x = width - 4; x >= 0; x -= 4)
				{
					*(__m256d*)(img_ptr + x) =
#ifdef USE_FMA_VYV
						_mm256_fnmadd_pd(_mm256_set1_pd(coeff[5]), *(__m256d*)(img_ptr + x + offset[5]),
							_mm256_fnmadd_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x + offset[4]),
								_mm256_fnmadd_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x + offset[3]),
									_mm256_fnmadd_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x + offset[2]),
										_mm256_fnmadd_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x + offset[1]),
											_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)))))));

#else
						_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(
							_mm256_mul_pd(_mm256_set1_pd(coeff[0]), *(__m256d*)(img_ptr + x)),
							_mm256_mul_pd(_mm256_set1_pd(coeff[1]), *(__m256d*)(img_ptr + x + offset[1]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[2]), *(__m256d*)(img_ptr + x + offset[2]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[3]), *(__m256d*)(img_ptr + x + offset[3]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[4]), *(__m256d*)(img_ptr + x + offset[4]))),
							_mm256_mul_pd(_mm256_set1_pd(coeff[5]), *(__m256d*)(img_ptr + x + offset[5])));
#endif
				}
			}
			break;
		}
		}
	}

	void GaussianFilterVYV_AVX_64F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		this->borderType = borderType;

		CV_Assert(src.cols % 4 == 0);
		CV_Assert(src.rows % 4 == 0);
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);

		const bool isInplace = src.data == dst.data;

		if (isInplace)
		{
			if (src.depth() == CV_64F)
			{
				src.copyTo(inter);
				horizontalbody(inter, dst);
				verticalbody(dst);
			}
			else
			{
				src.convertTo(inter, CV_64F);
				inter2.create(inter.size(), CV_64F);
				horizontalbody(inter, inter2);
				verticalbody(inter2);
				inter2.convertTo(dst, dest_depth);
			}
		}
		else
		{
			if (src.depth() == CV_64F)
			{
				if (dest_depth == CV_64F)
				{
					dst.create(imgSize, CV_64F);
					horizontalbody(src, dst);
					verticalbody(dst);
				}
				else
				{
					inter.create(imgSize, CV_64F);
					horizontalbody(src, inter);
					verticalbody(inter);
					inter.convertTo(dst, dest_depth);
				}
			}
			else
			{
				src.convertTo(inter, CV_64F);
				if (dest_depth == CV_64F)
				{
					dst.create(imgSize, CV_64F);
					horizontalbody(inter, dst);
					verticalbody(dst);
				}
				else
				{
					inter2.create(imgSize, CV_64F);
					horizontalbody(inter, inter2);
					verticalbody(inter2);
					inter2.convertTo(dst, dest_depth);
				}
			}
		}
	}

	void GaussianFilterVYV_AVX_64F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		if (this->sigma != sigma || this->gf_order != order || imgSize != src.size())
		{
			this->sigma = sigma;
			this->gf_order = cliped_order(order, SpatialFilterAlgorithm::IIR_VYV);
			this->imgSize = src.size();
			allocBuffer();
		}

		body(src, dst, borderType);
	}
#pragma endregion
}