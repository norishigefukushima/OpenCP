#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//baseline 

	void dualBilateralWeightMapBase_32f(const Mat& src, const Mat& guide, Mat& dst, int d,
		double sigma_color, double sigma_guide_color, double sigma_space, int borderType, bool isLaplace)
	{
		if (d == 0){ src.copyTo(dst); return; }
		Size size = src.size();
		if (dst.empty())dst = Mat::zeros(src.size(), CV_32F);
		//CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		//	src.type() == dst.type() && src.size() == dst.size() &&
		//	src.data != dst.data );

		if (sigma_color <= 0.0)
			sigma_color = 1.0;
		if (sigma_space <= 0.0)
			sigma_space = 1.0;


		double gauss_color_coeff = (isLaplace) ? -1.0 / (sigma_color) : -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = (isLaplace) ? -1.0 / (sigma_guide_color) : -0.5 / (sigma_guide_color*sigma_guide_color);//trilateral(1)

		//do not suppert Laplace distribution
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		if (guide.empty())src.copyTo(guide);
		const int cn = src.channels();
		const int cnj = guide.channels();

		int radius;
		if (d <= 0)
			radius = cvRound(sigma_space*1.5);
		else
			radius = d / 2;
		radius = MAX(radius, 1);
		d = radius * 2 + 1;

		Mat jim;
		Mat sim;
		copyMakeBorder(guide, jim, radius, radius, radius, radius, borderType);
		copyMakeBorder(src, sim, radius, radius, radius, radius, borderType);

		vector<float> _color_weight(cn * 256);//trilateral(2)
		vector<float> _color_guide_weight(cnj * 256);//trilateral(3)
		vector<float> _space_weight(d*d);
		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];//trilateral(4)
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs_jnt(d*d);
		vector<int> _space_ofs_src(d*d);
		int* space_ofs_jnt = &_space_ofs_jnt[0];
		int* space_ofs_src = &_space_ofs_src[0];

		// initialize color-related bilateral filter coefficients
		for (int i = 0; i < 256 * cn; i++)//trilateral(5)
		{
			color_weight[i] = (isLaplace) ? (float)std::exp(i*gauss_color_coeff) : (float)std::exp(i*i*gauss_color_coeff);
		}
		for (int i = 0; i < 256 * cnj; i++)//trilateral(6)
		{
			color_guide_weight[i] = (isLaplace) ? (float)std::exp(i*gauss_guide_color_coeff) : (float)std::exp(i*i*gauss_guide_color_coeff);
		}

		int maxk = 0;
		// initialize space-related bilateral filter coefficients
		for (int i = -radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radius) continue;
				space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
				space_ofs_jnt[maxk] = (int)(i*jim.cols*cnj + j*cnj);
				space_ofs_src[maxk++] = (int)(i*sim.cols*cn + j*cn);
			}
		}

		for (int i = 0; i < size.height; i++)
		{
			const float* jptr = jim.ptr<float>(i + radius) + radius*cnj;
			const float* sptr = sim.ptr<float>(i + radius) + radius*cn;
			float* dptr = dst.ptr<float>(i);

			if (cn == 1 && cnj == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					float wsum = 0.f;
					const float val0 = jptr[j];
					const float vals0 = sptr[j];//trilateral(7)

					for (int k = 0; k < maxk; k++)
					{
						float val = jptr[j + space_ofs_src[k]];
						float vals = sptr[j + space_ofs_src[k]];

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(vals - vals0))]
							* color_guide_weight[cvRound(std::abs(val - val0))];//trilateral(8)
						wsum += w;
					}
					dptr[j] = wsum;
				}
			}
			else if (cn == 3 && cnj == 3)
			{
				for (int j = 0, l = 0; l < size.width; j += 3, l++)
				{
					float wsum = 0.f;
					float bs0 = sptr[j], gs0 = sptr[j + 1], rs0 = sptr[j + 2];//trilateral(9)
					float b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const float* jptr_k = jptr + j + space_ofs_jnt[k];
						const float* sptr_k = sptr + j + space_ofs_src[k];

						const float b = jptr_k[0], g = jptr_k[1], r = jptr_k[2];
						const float bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(bs - bs0) + std::abs(gs - gs0) + std::abs(rs - rs0))]
							* color_guide_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];//trilateral(11)
						wsum += w;
					}
					dptr[l] = wsum;
				}
			}
			else if (cn == 1 && cnj == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float wsum = 0.f;
					const float vs0 = sptr[l];//trilateral(9)
					const float b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						//					const float* sptr_k = sptr + l + space_ofs_src[k];
						const float* jptr_k = jptr + j + space_ofs_jnt[k];

						const float val = *(sptr + l + space_ofs_src[k]);
						float b = jptr_k[0], g = jptr_k[1], r = jptr_k[2];

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(val - vs0))]
							* color_guide_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];
						wsum += w;
					}
					dptr[l] = wsum;
				}
			}
			else if (cn == 3 && cnj == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float wsum = 0.f;
					const float bs0 = sptr[j], gs0 = sptr[j + 1], rs0 = sptr[j + 2];
					const float val0 = jptr[l];

					for (int k = 0; k < maxk; k++)
					{
						float val = jptr[l + space_ofs_jnt[k]];
						const float* sptr_k = sptr + j + space_ofs_src[k];
						const float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(b - bs0) + std::abs(g - gs0) + std::abs(r - rs0))]
							* color_guide_weight[cvRound(std::abs(val - val0))];
						wsum += w;
					}
					dptr[l] = wsum;
				}
			}
		}
	}

	void dualBilateralWeightMapBase(InputArray src_, InputArray guide_, OutputArray dst_, int d,
		double sigma_color, double sigma_guide_color, double sigma_space, int borderType, bool isLaplace)
	{
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();
		if (dst_.empty()) dst_.create(src_.size(), CV_32F);
		Mat dst = dst_.getMat();

		if (src.type() == CV_MAKE_TYPE(CV_32F, src.channels()))
		{
			dualBilateralWeightMapBase_32f(src, guide, dst, d, sigma_color, sigma_guide_color, sigma_space, borderType, isLaplace);
		}
		else
		{
			Mat ss, gg;
			src.convertTo(ss, CV_32F);
			guide.convertTo(gg, CV_32F);
			dualBilateralWeightMapBase_32f(ss, gg, dst, d, sigma_color, sigma_guide_color, sigma_space, borderType, isLaplace);
		}
	}

	void dualBilateralFilterBase_32f(const Mat& src, const Mat& guide, Mat& dst, int d,
		double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		if (d == 0){ src.copyTo(dst); return; }
		Size size = src.size();
		if (dst.empty())dst = Mat::zeros(src.size(), src.type());
		//CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		//	src.type() == dst.type() && src.size() == dst.size() &&
		//	src.data != dst.data );

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);//trilateral(1)
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		if (guide.empty())src.copyTo(guide);
		const int cn = src.channels();
		const int cnj = guide.channels();

		int radius;
		if (d <= 0)
			radius = cvRound(sigma_space*1.5);
		else
			radius = d / 2;
		radius = MAX(radius, 1);
		d = radius * 2 + 1;

		Mat jim;
		Mat sim;
		copyMakeBorder(guide, jim, radius, radius, radius, radius, borderType);
		copyMakeBorder(src, sim, radius, radius, radius, radius, borderType);

		vector<float> _color_weight(cn * 256);//trilateral(2)
		vector<float> _color_guide_weight(cnj * 256);//trilateral(3)
		vector<float> _space_weight(d*d);
		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];//trilateral(4)
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs_jnt(d*d);
		vector<int> _space_ofs_src(d*d);
		int* space_ofs_jnt = &_space_ofs_jnt[0];
		int* space_ofs_src = &_space_ofs_src[0];

		// initialize color-related bilateral filter coefficients
		for (int i = 0; i < 256 * cn; i++)//trilateral(5)
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);


		for (int i = 0; i < 256 * cnj; i++)//trilateral(6)
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);

		int maxk = 0;
		// initialize space-related bilateral filter coefficients
		for (int i = -radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radius)
					continue;
				space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
				space_ofs_jnt[maxk] = (int)(i*jim.cols*cnj + j*cnj);
				space_ofs_src[maxk++] = (int)(i*sim.cols*cn + j*cn);
			}
		}

		for (int i = 0; i < size.height; i++)
		{
			const float* jptr = jim.ptr<float>(i + radius) + radius*cnj;
			const float* sptr = sim.ptr<float>(i + radius) + radius*cn;
			float* dptr = dst.ptr<float>(i);

			if (cn == 1 && cnj == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					float sum = 0.f, wsum = 0.f;
					const float val0 = jptr[j];
					const float vals0 = sptr[j];//trilateral(7)

					for (int k = 0; k < maxk; k++)
					{
						float val = jptr[j + space_ofs_src[k]];
						float vals = sptr[j + space_ofs_src[k]];

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(vals - vals0))]
							* color_guide_weight[cvRound(std::abs(val - val0))];//trilateral(8)

						sum += vals*w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
			else if (cn == 3 && cnj == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f, wsum = 0.0f;
					float bs0 = sptr[j], gs0 = sptr[j + 1], rs0 = sptr[j + 2];//trilateral(9)
					float b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const float* jptr_k = jptr + j + space_ofs_jnt[k];
						const float* sptr_k = sptr + j + space_ofs_src[k];

						const float b = jptr_k[0], g = jptr_k[1], r = jptr_k[2];
						const float bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(bs - bs0) + std::abs(gs - gs0) + std::abs(rs - rs0))]
							* color_guide_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];//trilateral(11)
						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (sum_b / wsum);
					dptr[j + 1] = (sum_g / wsum);
					dptr[j + 2] = (sum_r / wsum);
				}
			}
			else if (cn == 1 && cnj == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum_b = 0, wsum = 0;
					const float vs0 = sptr[l];//trilateral(9)
					const float b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const float* jptr_k = jptr + j + space_ofs_jnt[k];

						const float val = *(sptr + l + space_ofs_src[k]);
						float b = jptr_k[0], g = jptr_k[1], r = jptr_k[2];

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(val - vs0))]
							* color_guide_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];
						sum_b += val*w;
						wsum += w;
					}
					dptr[l] = sum_b / wsum;
				}
			}
			else if (cn == 3 && cnj == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;

					const float bs0 = sptr[j], gs0 = sptr[j + 1], rs0 = sptr[j + 2];
					const float val0 = jptr[l];

					for (int k = 0; k < maxk; k++)
					{
						float val = jptr[l + space_ofs_jnt[k]];
						const float* sptr_k = sptr + j + space_ofs_src[k];
						const float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

						float w = space_weight[k]
							* color_weight[cvRound(std::abs(b - bs0) + std::abs(g - gs0) + std::abs(r - rs0))]
							* color_guide_weight[cvRound(std::abs(val - val0))];
						sum_b += b*w; sum_g += g*w; sum_r += r*w;
						wsum += w;
					}
					dptr[j] = sum_b / wsum;
					dptr[j + 1] = sum_g / wsum;
					dptr[j + 2] = sum_r / wsum;
				}
			}
		}
	}

	void dualBilateralFilterBase_8u(const Mat& src, const Mat& guide, Mat& dst, int d,
		const double sigma_color, const double sigma_guide_color, const double sigma_space, int borderType)
	{
		if (d == 0){ src.copyTo(dst); return; }
		Size size = src.size();
		if (dst.empty())dst = Mat::zeros(src.size(), src.type());
		//CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		//	src.type() == dst.type() && src.size() == dst.size() &&
		//	src.data != dst.data );

		const double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		const double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);//trilateral(1)
		const double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		if (guide.empty())src.copyTo(guide);
		const int cn = src.channels();
		const int cnj = guide.channels();

		int radius = d / 2;
		d = radius * 2 + 1;

		Mat jim;
		Mat sim;
		copyMakeBorder(guide, jim, radius, radius, radius, radius, borderType);
		copyMakeBorder(src, sim, radius, radius, radius, radius, borderType);

		vector<float> _color_weight(cn * 256);//trilateral(2)
		vector<float> _color_guide_weight(cnj * 256);//trilateral(3)
		vector<float> _space_weight(d*d);
		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];//trilateral(4)
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs_jnt(d*d);
		vector<int> _space_ofs_src(d*d);
		int* space_ofs_jnt = &_space_ofs_jnt[0];
		int* space_ofs_src = &_space_ofs_src[0];

		// initialize color-related bilateral filter coefficients
		for (int i = 0; i < 256 * cn; i++)//trilateral(5)
		{
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		}
		for (int i = 0; i < 256 * cnj; i++)//trilateral(6)
		{
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);
		}

		int maxk = 0;
		// initialize space-related bilateral filter coefficients
		for (int i = -radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radius)
					continue;
				space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);

				space_ofs_jnt[maxk] = (int)(i*jim.step + j*cnj);
				space_ofs_src[maxk++] = (int)(i*sim.step + j*cn);
			}
		}

		//#pragma omp parallel for
		for (int i = 0; i < size.height; i++)
		{
			const uchar* jptr = jim.data + (i + radius)*jim.step + radius*cnj;
			const uchar* sptr = sim.data + (i + radius)*sim.step + radius*cn;
			uchar* dptr = dst.data + i*dst.step;

			if (cn == 1 && cnj == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					float sum = 0.f, wsum = 0.f;

					const int val0 = jptr[j];
					const int vals0 = sptr[j];//trilateral(7)

					for (int k = 0; k < maxk; k++)
					{
						const int val = jptr[j + space_ofs_src[k]];
						const int vals = sptr[j + space_ofs_src[k]];
						float w = space_weight[k]
							* color_weight[std::abs(vals - vals0)]
							* color_guide_weight[std::abs(val - val0)];//trilateral(7)
						sum += vals*w;
						wsum += w;
					}
					// overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum / wsum);
				}
			}
			else if (cn == 3 && cnj == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;

					int bg0 = jptr[j], gg0 = jptr[j + 1], rg0 = jptr[j + 2];
					int bs0 = sptr[j], gs0 = sptr[j + 1], rs0 = sptr[j + 2];//trilateral(9)

					for (int k = 0; k < maxk; k++)
					{
						const uchar* jptr_k = jptr + j + space_ofs_jnt[k];
						const uchar* sptr_k = sptr + j + space_ofs_src[k];
						const int b = jptr_k[0], g = jptr_k[1], r = jptr_k[2];
						const int bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)
						float w = space_weight[k];
						w *= color_weight[abs(bs - bs0) + abs(gs - gs0) + abs(rs - rs0)];
						w *= color_guide_weight[abs(b - bg0) + abs(g - gg0) + abs(r - rg0)];

						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (uchar)cvRound(sum_b / wsum); gs0 = dptr[j + 1] = (uchar)cvRound(sum_g / wsum); dptr[j + 2] = (uchar)cvRound(sum_r / wsum);
				}
			}
			else if (cn == 1 && cnj == 3)
			{
				for (int j = 0, l = 0; l < size.width; j += 3, l++)
				{
					float sum_b = 0.f, wsum = 0.f;
					const int sv0 = sptr[l];//trilateral(9)
					const int jb0 = jptr[j], jg0 = jptr[j + 1], jr0 = jptr[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const int val = *(sptr + l + space_ofs_src[k]);
						const uchar* jptr_k = jptr + j + space_ofs_jnt[k];
						const int b = jptr_k[0], g = jptr_k[1], r = jptr_k[2];

						float w = space_weight[k]
							* color_weight[std::abs(val - sv0)]
							* color_guide_weight[std::abs(b - jb0) + std::abs(g - jg0) + std::abs(r - jr0)];
						sum_b += val*w;
						wsum += w;
					}
					dptr[l] = (uchar)cvRound(sum_b / wsum);
				}
			}
			else if (cn == 3 && cnj == 1)
			{
				for (int j = 0, l = 0; l < size.width; j += 3, l++)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;
					const int bs0 = sptr[j], gs0 = sptr[j + 1], rs0 = sptr[j + 2];
					const int vg0 = jptr[l];

					for (int k = 0; k < maxk; k++)
					{
						const uchar* sptr_k = sptr + j + space_ofs_src[k];
						const int bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];
						int vg = jptr[l + space_ofs_jnt[k]];

						float w = space_weight[k];
						w *= color_weight[abs(bs - bs0) + abs(gs - gs0) + abs(rs - rs0)];
						w *= color_guide_weight[abs(vg - vg0)];

						sum_b += bs*w;
						sum_g += gs*w;
						sum_r += rs*w;
						wsum += w;
					}
					// overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum_b / wsum);
					dptr[j + 1] = (uchar)cvRound(sum_g / wsum);
					dptr[j + 2] = (uchar)cvRound(sum_r / wsum);
				}
			}
		}
	}

	void dualBilateralFilterBase(const Mat& src, const Mat& guide, Mat& dst, int d,
		double sigma_color, double sigma_guide_color, double sigma_space, int method, int borderType)
	{
		if (dst.empty())dst.create(src.size(), src.type());
		if (method == FILTER_CIRCLE)
		{
			if (src.type() == CV_MAKE_TYPE(CV_8U, src.channels()))
			{
				dualBilateralFilterBase_8u(src, guide, dst, d, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
			else if (src.type() == CV_MAKE_TYPE(CV_32F, src.channels()))
			{
				dualBilateralFilterBase_32f(src, guide, dst, d, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			std::cout << "SEPARABLE is not support" << std::endl;
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//weighted trilatateral 32f!
	class WeightedTrilateralFilter_32f_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		WeightedTrilateralFilter_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _weightMap, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_w_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float *_guide_color_weight) :
			temp(&_temp), weightMap(&_weightMap), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_w_ofs(_space_w_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;

			const int cn = (temp->rows - 2 * radiusV) / dest->rows;
			const int cng = (guide->rows - 2 * radiusV) / dest->rows;
			const Size size = dest->size();

			const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* gptr = (float*)guide->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);//wmap!

				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int gstep = guide->cols;
				const int wstep = weightMap->cols;//wmap!

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep, wptr += wstep)//wmap!
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];
							int* wofs = &space_w_ofs[0];//wmap!

							float* spw = space_weight;

							const float* sptrj = sptr + j;
							const float* gptrj = gptr + j;
							const float* wptrj = wptr + j;//wmap!

							const __m128 val0j = _mm_load_ps((gptrj));
							const __m128 val0s = _mm_load_ps((sptrj));//tri

							__m128 wval1 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++, wofs++)//wmap!
							{
								__m128 sref = _mm_loadu_ps((gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0j, sref), *(const __m128*)v32f_absmask)));

								sref = _mm_loadu_ps((sptrj + *ofs));//tri
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0s, sref), *(const __m128*)v32f_absmask)));//tri

								const __m128 _sw = _mm_set1_ps(*spw);//space weight
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//tri
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));

								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0]));//wmap!

								sref = _mm_mul_ps(_w, sref);//値と重み全体との積
								tval1 = _mm_add_ps(tval1, sref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							_mm_stream_ps((dptr + j), tval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float val0j = gptr[j];
						const float val0s = sptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float valj = gptr[j + space_guide_ofs[k]];
							float vals = sptr[j + space_ofs[k]];
							float w = space_weight[k]
								* color_weight[cvRound(std::abs(vals - val0s))]
								* guide_color_weight[cvRound(std::abs(valj - val0j))];
							sum += vals*w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			else if (cn == 1 && cng == 3)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* gptrr = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* gptrg = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* gptrb = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);//wmap!

				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;
				const int wstep = weightMap->cols;//wmap!

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptr += sstep, dptr += dstep, wptr += wstep)//wmap!
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];
							int* wofs = &space_w_ofs[0];//wmap!

							float* spw = space_weight;

							const float* sptrj = sptr + j;
							const float* gptrrj = gptrr + j;
							const float* gptrgj = gptrg + j;
							const float* gptrbj = gptrb + j;
							const float* wptrj = wptr + j;//wmap!

							const __m128 val0s = _mm_load_ps((sptrj));
							const __m128 bval0j = _mm_load_ps((gptrbj));
							const __m128 gval0j = _mm_load_ps((gptrgj));
							const __m128 rval0j = _mm_load_ps((gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++, wofs++)//wmap!
							{
								const __m128 bref = _mm_loadu_ps((gptrbj + *gofs));
								const __m128 gref = _mm_loadu_ps((gptrgj + *gofs));
								const __m128 rref = _mm_loadu_ps((gptrrj + *gofs));

								_mm_store_si128((__m128i*)gbuf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval0j, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval0j, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval0j, bref), *(const __m128*)v32f_absmask)
									)
									));
								__m128 vref = _mm_loadu_ps((sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0s, vref), *(const __m128*)v32f_absmask)));

								__m128 _w = _mm_set1_ps(*spw);//space weight
								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//tri
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0]));//wmap!

								vref = _mm_mul_ps(_w, vref);//値と重み全体との積
								tval1 = _mm_add_ps(tval1, vref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							_mm_stream_ps((dptr + j), tval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* sptrj = sptr + j;
						const float* gptrrj = gptrr + j;
						const float* gptrgj = gptrg + j;
						const float* gptrbj = gptrb + j;

						const float v0s = sptrj[0];
						const float r0j = gptrrj[0];
						const float g0j = gptrgj[0];
						const float b0j = gptrbj[0];

						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							const float r = gptrrj[space_guide_ofs[k]], g = gptrgj[space_guide_ofs[k]], b = gptrbj[space_guide_ofs[k]];
							float w = space_weight[k]
								* color_weight[cvRound(std::abs(sptrj[space_ofs[k]] - v0s))]
								* guide_color_weight[cvRound(std::abs(b - b0j) + std::abs(g - g0j) + std::abs(r - r0j))];
							sum += sptrj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				assert(cng == 3);//カラー処理
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				float* sptrr = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrb = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* gptrr = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* gptrg = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* gptrb = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);//wmap!

				float* dptr = dest->ptr<float>(range.start);

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;
				const int wstep = weightMap->cols;//wmap!

				const int dstep = 3 * dest->cols;

				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep, wptr += wstep)//wmap!
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];
							int* wofs = &space_w_ofs[0];//wmap!

							float* spw = space_weight;

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;
							const float* gptrrj = gptrr + j;
							const float* gptrgj = gptrg + j;
							const float* gptrbj = gptrb + j;
							const float* wptrj = wptr + j;//wmap!

							const __m128 bval0s = _mm_load_ps((sptrbj));
							const __m128 gval0s = _mm_load_ps((sptrgj));
							const __m128 rval0s = _mm_load_ps((sptrrj));
							const __m128 bval0j = _mm_load_ps((gptrbj));
							const __m128 gval0j = _mm_load_ps((gptrgj));
							const __m128 rval0j = _mm_load_ps((gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 rval1 = _mm_setzero_ps();
							__m128 gval1 = _mm_setzero_ps();
							__m128 bval1 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++, wofs++)//wmap!
							{
								__m128 bref = _mm_loadu_ps((gptrbj + *gofs));
								__m128 gref = _mm_loadu_ps((gptrgj + *gofs));
								__m128 rref = _mm_loadu_ps((gptrrj + *gofs));

								_mm_store_si128((__m128i*)gbuf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rref, rval0j), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gref, gval0j), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bref, bval0j), *(const __m128*)v32f_absmask)
									)
									));

								bref = _mm_loadu_ps((sptrbj + *ofs));
								gref = _mm_loadu_ps((sptrgj + *ofs));
								rref = _mm_loadu_ps((sptrrj + *ofs));

								_mm_store_si128((__m128i*)buf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rref, rval0s), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gref, gval0s), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bref, bval0s), *(const __m128*)v32f_absmask)
									)
									));

								__m128 _w = _mm_set1_ps(*spw);//space weight
								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//tri
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0]));//wmap!

								rref = _mm_mul_ps(_w, rref);//値と重み全体との積
								gref = _mm_mul_ps(_w, gref);//値と重み全体との積
								bref = _mm_mul_ps(_w, bref);//値と重み全体との積

								rval1 = _mm_add_ps(rval1, rref);
								gval1 = _mm_add_ps(gval1, gref);
								bval1 = _mm_add_ps(bval1, bref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							rval1 = _mm_div_ps(rval1, wval1);
							gval1 = _mm_div_ps(gval1, wval1);
							bval1 = _mm_div_ps(bval1, wval1);

							__m128 a = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));
							float* dptrc = dptr + 3 * j;
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
						const float* gptrrj = gptrr + j;
						const float* gptrgj = gptrg + j;
						const float* gptrbj = gptrb + j;

						const float r0j = gptrrj[0];
						const float g0j = gptrgj[0];
						const float b0j = gptrbj[0];
						const float r0s = sptrrj[0];
						const float g0s = sptrgj[0];
						const float b0s = sptrbj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							const float r = sptrrj[space_guide_ofs[k]], g = sptrgj[space_guide_ofs[k]], b = sptrbj[space_guide_ofs[k]];
							const float rj = gptrrj[space_guide_ofs[k]], gj = gptrgj[space_guide_ofs[k]], bj = gptrbj[space_guide_ofs[k]];
							float w = space_weight[k]
								* color_weight[cvRound(std::abs(b - b0s) + std::abs(g - g0s) + std::abs(r - r0s))]
								* guide_color_weight[cvRound(std::abs(bj - b0j) + std::abs(gj - g0j) + std::abs(rj - r0j))];;
							sum_b += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_r += sptrbj[space_ofs[k]] * w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[3 * j] = sum_b*wsum;
						dptr[3 * j + 1] = sum_g*wsum;
						dptr[3 * j + 2] = sum_r*wsum;
					}
				}

			}
			else if (cn == 3 && cng == 1)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				float* sptrr = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrb = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* gptr = (float*)guide->ptr<float>(radiusV + range.start) + 4 * (radiusH / 4 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);//wmap!

				float* dptr = dest->ptr<float>(range.start);

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;
				const int wstep = weightMap->cols;//wmap!

				const int dstep = 3 * dest->cols;

				for (i = range.start; i != range.end; i++, gptr += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep, wptr += wstep)//wmap!
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];
							int* wofs = &space_w_ofs[0];//wmap!

							float* spw = space_weight;

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;
							const float* gptrj = gptr + j;
							const float* wptrj = wptr + j;//wmap!

							const __m128 bval0s = _mm_load_ps((sptrbj));
							const __m128 gval0s = _mm_load_ps((sptrgj));
							const __m128 rval0s = _mm_load_ps((sptrrj));
							const __m128 val0j = _mm_load_ps((gptrj));

							//重みと平滑化後の画素４ブロックづつ
							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++, wofs++)//wmap!
							{
								__m128 sref = _mm_loadu_ps((gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0j, sref), *(const __m128*)v32f_absmask)));

								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								_mm_store_si128((__m128i*)buf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rref, rval0s), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gref, gval0s), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bref, bval0s), *(const __m128*)v32f_absmask)
									)
									));

								const __m128 _sw = _mm_set1_ps(*spw);//space weight
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0]));//wmap!
								bref = _mm_mul_ps(_w, bref);//値と重み全体との積
								gref = _mm_mul_ps(_w, gref);//値と重み全体との積
								rref = _mm_mul_ps(_w, rref);//値と重み全体との積

								bval1 = _mm_add_ps(bval1, bref);
								gval1 = _mm_add_ps(gval1, gref);
								rval1 = _mm_add_ps(rval1, rref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							rval1 = _mm_div_ps(rval1, wval1);
							gval1 = _mm_div_ps(gval1, wval1);
							bval1 = _mm_div_ps(bval1, wval1);

							__m128 a = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));
							float* dptrc = dptr + 3 * j;
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
						const float* gptrj = gptr + j;

						const float v0j = gptrj[0];
						const float r0s = sptrrj[0];
						const float g0s = sptrgj[0];
						const float b0s = sptrbj[0];
						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							const float r = sptrrj[space_guide_ofs[k]], g = sptrgj[space_guide_ofs[k]], b = sptrbj[space_guide_ofs[k]];
							const float vj = gptrj[space_guide_ofs[k]];
							float w = space_weight[k]
								* color_weight[cvRound(std::abs(b - b0s) + std::abs(g - g0s) + std::abs(r - r0s))]
								* guide_color_weight[cvRound(std::abs(vj - v0j))];
							sum_b += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_r += sptrbj[space_ofs[k]] * w;
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
		const Mat *weightMap;
		Mat *dest;
		const Mat* guide;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs, *space_w_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};

	class WeightedTrilateralFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		WeightedTrilateralFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _weightMap, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_w_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float * _guide_color_weight) :
			temp(&_temp), weightMap(&_weightMap), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_w_ofs(_space_w_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = (temp->rows - 2 * radiusV) / dest->rows;
			int cng = (guide->rows - 2 * radiusV) / dest->rows;
			Size size = dest->size();

			//imshow("wwww",weightMap);waitKey();
#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				const int sstep = temp->cols;
				const int gstep = guide->cols;
				const int wstep = weightMap->cols;

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep, wptr += wstep)
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
							int* wofs = &space_w_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrj = gptr + j;
							const float* wptrj = wptr + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();
							const __m128i zero = _mm_setzero_si128();
							__m128i m1, m2;
							__m128 _valF, _w;
							for (k = 0; k < maxk; k++, ofs++, wofs++, gofs++, spw++)
							{
								//cout<<k<<":"<<ofs[0]<<","<<wofs[0]<<","<<gofs[0]<<endl;
								__m128i sref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(gval, sref), _mm_subs_epu8(sref, gval)));//guide weight

								sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, sref), _mm_subs_epu8(sref, sval)));

								m1 = _mm_unpacklo_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);
								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));

								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0]));

								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0] + 4));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);

								m1 = _mm_unpackhi_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0] + 8));
								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 12));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
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
						const uchar val0 = gptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int gval = gptr[j + space_guide_ofs[k]];
							int val = sptr[j + space_ofs[k]];
							float w = wptr[j + space_w_ofs[k]] * space_weight[k] * color_weight[std::abs(gval - val0)];
							sum += val*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 1 && cng == 3)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];


				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;
				const int wstep = weightMap->cols;

				const int dstep = dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptr += sstep, dptr += dstep, wptr += wstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						//for(; j < 0; j+=16)//16画素づつ処理
						for (; j < size.width; j += 16)//16画素づつ処理
						{
							__m128i m1, m2, n1, n2;
							__m128 _w, _valF;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];
							int* wofs = &space_w_ofs[0];

							float* spw = space_weight;

							const float* wptrj = wptr + j;
							const uchar* sptrj = sptr + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

							const __m128i bval = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i rval = _mm_load_si128((__m128i*)(gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, gofs++, wofs++, spw++)
							{
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

								_mm_store_si128((__m128i*)(gbuf + 8), m2);
								_mm_store_si128((__m128i*)gbuf, m1);

								const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, vref), _mm_subs_epu8(vref, sval)));

								m1 = _mm_unpacklo_epi8(vref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptrj + wofs[0]));
								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 4));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);


								m1 = _mm_unpackhi_epi8(vref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 8));
								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 12));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);//値と重み全体との積
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
						const uchar* sptrj = sptr + j;
						const uchar* gptrrj = gptrr + j;
						const uchar* gptrgj = gptrg + j;
						const uchar* gptrbj = gptrb + j;

						int r0 = gptrrj[0];
						int g0 = gptrgj[0];
						int b0 = gptrbj[0];

						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = gptrrj[space_guide_ofs[k]], g = gptrgj[space_guide_ofs[k]], b = gptrbj[space_guide_ofs[k]];
							float w = wptr[j + space_w_ofs[k]] * space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							sum += sptrj[space_ofs[k]] * w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;
				const int wstep = weightMap->cols;

				const int dstep = 3 * dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep, wptr += wstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						//for(; j < 0; j+=16)//16画素づつ処理
						for (; j < size.width; j += 16)//16画素づつ処理
						{
							__m128 _w, _valr, _valg, _valb;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];
							int* wofs = &space_w_ofs[0];

							float* spw = space_weight;

							//						const float* wptrj = wptr+j;
							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i gb = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gg = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i gr = _mm_load_si128((__m128i*)(gptrrj));
							const __m128i sb = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i sg = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i sr = _mm_load_si128((__m128i*)(sptrrj));


							//重みと平滑化後の画素４ブロックづつ
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

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, gofs++, wofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrbj + *gofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(gptrgj + *gofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(gptrrj + *gofs));

								__m128i r1 = _mm_add_epi8(_mm_subs_epu8(gr, rref), _mm_subs_epu8(rref, gr));
								__m128i r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gg, gref), _mm_subs_epu8(gref, gg));
								__m128i g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								__m128i b1 = _mm_add_epi8(_mm_subs_epu8(gb, bref), _mm_subs_epu8(bref, gb));
								__m128i b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(gbuf + 8), r2);
								_mm_store_si128((__m128i*)gbuf, r1);

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								r1 = _mm_add_epi8(_mm_subs_epu8(sr, rref), _mm_subs_epu8(rref, sr));
								r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								g1 = _mm_add_epi8(_mm_subs_epu8(sg, gref), _mm_subs_epu8(gref, sg));
								g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								b1 = _mm_add_epi8(_mm_subs_epu8(sb, bref), _mm_subs_epu8(bref, sb));
								b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(buf + 8), r2);
								_mm_store_si128((__m128i*)buf, r1);

								r1 = _mm_unpacklo_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								g1 = _mm_unpacklo_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								b1 = _mm_unpacklo_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0]));

								_valr = _mm_cvtepi32_ps(r1);//slide!
								_valg = _mm_cvtepi32_ps(g1);//slide!
								_valb = _mm_cvtepi32_ps(b1);//slide!

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

								rval1 = _mm_add_ps(rval1, _valr);
								gval1 = _mm_add_ps(gval1, _valg);
								bval1 = _mm_add_ps(bval1, _valb);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 4));
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

								rval2 = _mm_add_ps(rval2, _valr);
								gval2 = _mm_add_ps(gval2, _valg);
								bval2 = _mm_add_ps(bval2, _valb);
								wval2 = _mm_add_ps(wval2, _w);

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 8));
								_valr = _mm_cvtepi32_ps(r1);
								_valg = _mm_cvtepi32_ps(g1);
								_valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

								wval3 = _mm_add_ps(wval3, _w);
								rval3 = _mm_add_ps(rval3, _valr);
								gval3 = _mm_add_ps(gval3, _valg);
								bval3 = _mm_add_ps(bval3, _valb);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 12));
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

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
							uchar* dptrc = dptr + 3 * j;
							const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
							const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
							const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
							const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
							const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

							a = _mm_shuffle_epi8(a, mask1);
							b = _mm_shuffle_epi8(b, mask2);
							c = _mm_shuffle_epi8(c, mask3);
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
						const uchar* gptrrj = gptrr + j;
						const uchar* gptrgj = gptrg + j;
						const uchar* gptrbj = gptrb + j;

						int r0 = sptrrj[0];
						int g0 = sptrgj[0];
						int b0 = sptrbj[0];
						int gr0 = gptrrj[0];
						int gg0 = gptrgj[0];
						int gb0 = gptrbj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							int gr = gptrrj[space_guide_ofs[k]], gg = gptrgj[space_guide_ofs[k]], gb = gptrbj[space_guide_ofs[k]];
							float w = wptr[j + space_w_ofs[k]] * space_weight[k]
								* color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)]
								* guide_color_weight[std::abs(gb - gb0) + std::abs(gg - gg0) + std::abs(gr - gr0)];

							sum_b += r*w;
							sum_g += g*w;
							sum_r += b*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U

						wsum = 1.f / wsum;
						dptr[3 * j] = (uchar)cvRound(sum_b*wsum);
						dptr[3 * j + 1] = (uchar)cvRound(sum_g*wsum);
						dptr[3 * j + 2] = (uchar)cvRound(sum_r*wsum);
					}
				}
			}
			else if (cn == 3 && cng == 1)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;
				const int wstep = weightMap->cols;

				const int dstep = 3 * dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(radiusV + range.start) + 16 * (radiusH / 16 + 1);
				float* wptr = (float*)weightMap->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, gptr += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep, wptr += wstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						//for(; j < 0; j+=16)//16画素づつ処理
						for (; j < size.width; j += 16)//16画素づつ処理
						{
							__m128i m1, m2, n1, n2;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];
							int* wofs = &space_w_ofs[0];

							float* spw = space_weight;

							//						const float* wptrj = wptr+j;
							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrj = gptr + j;

							const __m128i bval = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i gval = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i rval = _mm_load_si128((__m128i*)(sptrrj));
							const __m128i val0j = _mm_load_si128((__m128i*)(gptrj));

							//重みと平滑化後の画素４ブロックづつ
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

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, gofs++, wofs++, spw++)
							{
								__m128i jref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(val0j, jref), _mm_subs_epu8(jref, val0j)));

								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

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

								__m128i r1 = _mm_unpacklo_epi8(rref, zero);
								__m128i r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								__m128i g1 = _mm_unpacklo_epi8(gref, zero);
								__m128i g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								__m128i b1 = _mm_unpacklo_epi8(bref, zero);
								__m128i b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0]));

								__m128 _valr = _mm_cvtepi32_ps(r1);//slide!
								__m128 _valg = _mm_cvtepi32_ps(g1);//slide!
								__m128 _valb = _mm_cvtepi32_ps(b1);//slide!

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

								rval1 = _mm_add_ps(rval1, _valr);
								gval1 = _mm_add_ps(gval1, _valg);
								bval1 = _mm_add_ps(bval1, _valb);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));

								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 4));
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

								rval2 = _mm_add_ps(rval2, _valr);
								gval2 = _mm_add_ps(gval2, _valg);
								bval2 = _mm_add_ps(bval2, _valb);
								wval2 = _mm_add_ps(wval2, _w);

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 8));
								_valr = _mm_cvtepi32_ps(r1);
								_valg = _mm_cvtepi32_ps(g1);
								_valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

								wval3 = _mm_add_ps(wval3, _w);
								rval3 = _mm_add_ps(rval3, _valr);
								gval3 = _mm_add_ps(gval3, _valg);
								bval3 = _mm_add_ps(bval3, _valb);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								_w = _mm_mul_ps(_w, _mm_loadu_ps(wptr + j + wofs[0] + 12));
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);//値と重み全体との積
								_valg = _mm_mul_ps(_w, _valg);//値と重み全体との積
								_valb = _mm_mul_ps(_w, _valb);//値と重み全体との積

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
							uchar* dptrc = dptr + 3 * j;
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
						const uchar* gptrj = gptr + j;

						int r0 = gptrj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = gptrj[space_guide_ofs[k]];
							float w = wptr[space_w_ofs[k]] * space_weight[k] * color_weight[std::abs(r - r0)];
							sum_b += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_r += sptrbj[space_ofs[k]] * w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U

						wsum = 1.f / wsum;
						int b0 = cvRound(sum_b*wsum);
						int g0 = cvRound(sum_g*wsum);
						r0 = cvRound(sum_r*wsum);
						dptr[3 * j] = (uchar)b0; dptr[3 * j + 1] = (uchar)g0; dptr[3 * j + 2] = (uchar)r0;
					}
				}
			}
		}
	private:
		const Mat *temp;
		const Mat *weightMap;
		Mat *dest;
		const Mat* guide;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs, *space_w_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};


	class DualBilateralFilter_32f_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		DualBilateralFilter_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float *_guide_color_weight) :
			temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = (temp->rows - 2 * radiusV) / dest->rows;
			int cng = (guide->rows - 2 * radiusV) / dest->rows;
			Size size = dest->size();

			static int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* gptr = (float*)guide->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* dptr = dest->ptr<float>(range.start);
				const int sstep = temp->cols;
				const int gstep = guide->cols;
				const int dstep = dest->cols;
				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const float* sptrj = sptr + j;
							const float* gptrj = gptr + j;

							const __m128 jval = _mm_load_ps((gptrj));
							const __m128 sval = _mm_load_ps((sptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128 jref = _mm_loadu_ps((gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(jval, jref), *(const __m128*)v32f_absmask)));

								__m128 vref = _mm_loadu_ps((sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval, vref), *(const __m128*)v32f_absmask)));

								const __m128 _sw = _mm_set1_ps(*spw);
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								vref = _mm_mul_ps(_w, vref);
								tval1 = _mm_add_ps(tval1, vref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							_mm_stream_ps((dptr + j), tval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float val0 = gptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float gval = gptr[j + space_guide_ofs[k]];
							float val = sptr[j + space_ofs[k]];
							float w = space_weight[k] * color_weight[cvRound(std::abs(gval - val0))];
							sum += val*w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			else if (cn == 1 && cng == 3)
			{
				assert(cng == 3);
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;
				const int dstep = dest->cols;

				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* gptrr = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* gptrg = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* gptrb = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);
				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptr += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const float* sptrj = sptr + j;
							const float* gptrrj = gptrr + j;
							const float* gptrgj = gptrg + j;
							const float* gptrbj = gptrb + j;

							const __m128 sval = _mm_load_ps((sptrj));

							const __m128 bvalj = _mm_load_ps((gptrbj));
							const __m128 gvalj = _mm_load_ps((gptrgj));
							const __m128 rvalj = _mm_load_ps((gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								const __m128 bref = _mm_loadu_ps((gptrbj + *gofs));
								const __m128 gref = _mm_loadu_ps((gptrgj + *gofs));
								const __m128 rref = _mm_loadu_ps((gptrrj + *gofs));

								_mm_store_si128((__m128i*)gbuf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rvalj, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gvalj, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bvalj, bref), *(const __m128*)v32f_absmask)
									)
									));
								__m128 vref = _mm_loadu_ps((sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval, vref), *(const __m128*)v32f_absmask)));
								const __m128 _sw = _mm_set1_ps(*spw);

								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								vref = _mm_mul_ps(_w, vref);
								tval1 = _mm_add_ps(tval1, vref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							tval1 = _mm_div_ps(tval1, wval1);
							_mm_stream_ps((dptr + j), tval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* sptrj = sptr + j;
						const float* gptrrj = gptrr + j;
						const float* gptrgj = gptrg + j;
						const float* gptrbj = gptrb + j;

						const float r0 = gptrrj[0];
						const float g0 = gptrgj[0];
						const float b0 = gptrbj[0];

						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							const float r = gptrrj[space_guide_ofs[k]], g = gptrgj[space_guide_ofs[k]], b = gptrbj[space_guide_ofs[k]];
							float w = space_weight[k] * color_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];
							sum += sptrj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;
				const int dstep = 3 * dest->cols;

				float* sptrr = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrb = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* gptrr = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* gptrg = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* gptrb = (float*)guide->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* dptr = dest->ptr<float>(range.start);
				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;
							const float* gptrrj = gptrr + j;
							const float* gptrgj = gptrg + j;
							const float* gptrbj = gptrb + j;

							const __m128 bvals = _mm_load_ps((sptrbj));
							const __m128 gvals = _mm_load_ps((sptrgj));
							const __m128 rvals = _mm_load_ps((sptrrj));
							const __m128 bvalj = _mm_load_ps((gptrbj));
							const __m128 gvalj = _mm_load_ps((gptrgj));
							const __m128 rvalj = _mm_load_ps((gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 rval1 = _mm_setzero_ps();
							__m128 gval1 = _mm_setzero_ps();
							__m128 bval1 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128 bref = _mm_loadu_ps((gptrbj + *gofs));
								__m128 gref = _mm_loadu_ps((gptrgj + *gofs));
								__m128 rref = _mm_loadu_ps((gptrrj + *gofs));

								_mm_store_si128((__m128i*)gbuf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rvalj, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gvalj, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bvalj, bref), *(const __m128*)v32f_absmask)
									)
									));

								bref = _mm_loadu_ps((sptrbj + *ofs));
								gref = _mm_loadu_ps((sptrgj + *ofs));
								rref = _mm_loadu_ps((sptrrj + *ofs));
								_mm_store_si128((__m128i*)buf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rvals, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gvals, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bvals, bref), *(const __m128*)v32f_absmask)
									)
									));


								__m128 _w = _mm_set1_ps(*spw);
								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
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

							__m128 a = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));
							float* dptrc = dptr + 3 * j;
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
						const float* gptrrj = gptrr + j;
						const float* gptrgj = gptrg + j;
						const float* gptrbj = gptrb + j;

						const float sr0 = sptrrj[0];
						const float sg0 = sptrgj[0];
						const float sb0 = sptrbj[0];
						const float r0 = gptrrj[0];
						const float g0 = gptrgj[0];
						const float b0 = gptrbj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							const float sr = sptrrj[space_ofs[k]], sg = sptrgj[space_ofs[k]], sb = sptrbj[space_ofs[k]];
							const float r = gptrrj[space_guide_ofs[k]], g = gptrgj[space_guide_ofs[k]], b = gptrbj[space_guide_ofs[k]];
							float w = space_weight[k];
							w *= color_weight[cvRound(std::abs(sb - sb0) + std::abs(sg - sg0) + std::abs(sr - sr0))];
							w *= guide_color_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];
							sum_b += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_r += sptrbj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[3 * j] = sum_b / wsum;
						dptr[3 * j + 1] = sum_g / wsum;
						dptr[3 * j + 2] = sum_r / wsum;
					}
				}

			}
			else if (cn == 3 && cng == 1)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;
				const int dstep = 3 * dest->cols;

				float* sptrr = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrb = (float*)temp->ptr<float>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* gptr = (float*)guide->ptr<float>(radiusV + range.start) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, gptr += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;
							const float* gptrj = gptr + j;

							const __m128 bval0s = _mm_load_ps((sptrbj));
							const __m128 gval0s = _mm_load_ps((sptrgj));
							const __m128 rval0s = _mm_load_ps((sptrrj));
							const __m128  val0j = _mm_load_ps((gptrj));

							__m128 wval1 = _mm_set1_ps(0.0f);
							__m128 rval1 = _mm_set1_ps(0.0f);
							__m128 gval1 = _mm_set1_ps(0.0f);
							__m128 bval1 = _mm_set1_ps(0.0f);

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128 ref = _mm_loadu_ps((gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0j, ref), *(const __m128*)v32f_absmask)));

								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								_mm_store_si128((__m128i*)buf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval0s, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval0s, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval0s, bref), *(const __m128*)v32f_absmask)
									)
									));

								__m128 _w = _mm_set1_ps(*spw);
								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								bref = _mm_mul_ps(_w, bref);
								gref = _mm_mul_ps(_w, gref);
								rref = _mm_mul_ps(_w, rref);

								bval1 = _mm_add_ps(bval1, bref);
								gval1 = _mm_add_ps(gval1, gref);
								rval1 = _mm_add_ps(rval1, rref);
								wval1 = _mm_add_ps(wval1, _w);
							}
							rval1 = _mm_div_ps(rval1, wval1);
							gval1 = _mm_div_ps(gval1, wval1);
							bval1 = _mm_div_ps(bval1, wval1);

							__m128 a = _mm_shuffle_ps(bval1, bval1, _MM_SHUFFLE(3, 0, 1, 2));
							__m128 b = _mm_shuffle_ps(rval1, rval1, _MM_SHUFFLE(1, 2, 3, 0));
							__m128 c = _mm_shuffle_ps(gval1, gval1, _MM_SHUFFLE(2, 3, 0, 1));
							float* dptrc = dptr + 3 * j;
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
						const float* gptrj = gptr + j;

						const float r0 = gptrj[0];
						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							const float r = gptrj[space_guide_ofs[k]];
							float w = space_weight[k] * color_weight[cvRound(std::abs(r - r0))];
							sum_b += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_r += sptrbj[space_ofs[k]] * w;
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
		const Mat* guide;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};

	class DualBilateralFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		DualBilateralFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float *_guide_color_weight) :
			temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{

			int i, j, k;
			int cn = (temp->rows - 2 * radiusV) / dest->rows;
			int cng = (guide->rows - 2 * radiusV) / dest->rows;

			Size size = dest->size();
#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				const int sstep = temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrj = gptr + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrj));
							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(gval, sref), _mm_subs_epu8(sref, gval)));//guide weight

								sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, sref), _mm_subs_epu8(sref, sval)));


								__m128i m1 = _mm_unpacklo_epi8(sref, zero);
								__m128i m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);

								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								__m128 _valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);

								m1 = _mm_unpackhi_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								_valF = _mm_cvtepi32_ps(m2);
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
						const uchar sv0 = sptr[j];
						const uchar gv0 = gptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int gval = gptr[j + space_guide_ofs[k]];
							int sval = sptr[j + space_ofs[k]];
							float w = space_weight[k]
								* color_weight[std::abs(sval - sv0)]
								* guide_color_weight[std::abs(gval - gv0)];
							sum += sval*w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 1 && cng == 3)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;

				const int dstep = dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptr += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							__m128i m1, m2, n1, n2;
							__m128 _valF, _w;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;

							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

							const __m128i bval = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i rval = _mm_load_si128((__m128i*)(gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
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

								_mm_store_si128((__m128i*)(gbuf + 8), m2);
								_mm_store_si128((__m128i*)gbuf, m1);

								const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, vref), _mm_subs_epu8(vref, sval)));

								m1 = _mm_unpacklo_epi8(vref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));

								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);


								m1 = _mm_unpackhi_epi8(vref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								tval3 = _mm_add_ps(tval3, _valF);
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval4 = _mm_add_ps(tval4, _valF);
								wval4 = _mm_add_ps(wval4, _w);
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
						const uchar* sptrj = sptr + j;
						const uchar* gptrrj = gptrr + j;
						const uchar* gptrgj = gptrg + j;
						const uchar* gptrbj = gptrb + j;

						int r0 = gptrrj[0];
						int g0 = gptrgj[0];
						int b0 = gptrbj[0];

						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = gptrrj[space_guide_ofs[k]], g = gptrgj[space_guide_ofs[k]], b = gptrbj[space_guide_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							sum += sptrj[space_ofs[k]] * w;
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;
				const int dstep = 3 * dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, gptrr += gstep, gptrg += gstep, gptrb += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i gb = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gg = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i gr = _mm_load_si128((__m128i*)(gptrrj));
							const __m128i sb = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i sg = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i sr = _mm_load_si128((__m128i*)(sptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							__m128 rval1 = _mm_setzero_ps();
							__m128 rval2 = _mm_setzero_ps();
							__m128 rval3 = _mm_setzero_ps();
							__m128 rval4 = _mm_setzero_ps();

							__m128 gval1 = _mm_setzero_ps();
							__m128 gval2 = _mm_setzero_ps();
							__m128 gval3 = _mm_setzero_ps();
							__m128 gval4 = _mm_setzero_ps();

							__m128 bval1 = _mm_setzero_ps();
							__m128 bval2 = _mm_setzero_ps();
							__m128 bval3 = _mm_setzero_ps();
							__m128 bval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrbj + *gofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(gptrgj + *gofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(gptrrj + *gofs));

								__m128i r1 = _mm_add_epi8(_mm_subs_epu8(gr, rref), _mm_subs_epu8(rref, gr));
								__m128i r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gg, gref), _mm_subs_epu8(gref, gg));
								__m128i g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								__m128i b1 = _mm_add_epi8(_mm_subs_epu8(gb, bref), _mm_subs_epu8(bref, gb));
								__m128i b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(gbuf + 8), r2);
								_mm_store_si128((__m128i*)gbuf, r1);

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								r1 = _mm_add_epi8(_mm_subs_epu8(sr, rref), _mm_subs_epu8(rref, sr));
								r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								g1 = _mm_add_epi8(_mm_subs_epu8(sg, gref), _mm_subs_epu8(gref, sg));
								g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								b1 = _mm_add_epi8(_mm_subs_epu8(sb, bref), _mm_subs_epu8(bref, sb));
								b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(buf + 8), r2);
								_mm_store_si128((__m128i*)buf, r1);

								r1 = _mm_unpacklo_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								g1 = _mm_unpacklo_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								b1 = _mm_unpacklo_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));

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

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
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

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								_valr = _mm_cvtepi32_ps(r1);
								_valg = _mm_cvtepi32_ps(g1);
								_valb = _mm_cvtepi32_ps(b1);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval3 = _mm_add_ps(rval3, _valr);
								gval3 = _mm_add_ps(gval3, _valg);
								bval3 = _mm_add_ps(bval3, _valb);
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								_valr = _mm_cvtepi32_ps(r2);
								_valg = _mm_cvtepi32_ps(g2);
								_valb = _mm_cvtepi32_ps(b2);

								_valr = _mm_mul_ps(_w, _valr);
								_valg = _mm_mul_ps(_w, _valg);
								_valb = _mm_mul_ps(_w, _valb);

								rval4 = _mm_add_ps(rval4, _valr);
								gval4 = _mm_add_ps(gval4, _valg);
								bval4 = _mm_add_ps(bval4, _valb);
								wval4 = _mm_add_ps(wval4, _w);
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
							uchar* dptrc = dptr + 3 * j;
							const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
							const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
							const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
							const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
							const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

							a = _mm_shuffle_epi8(a, mask1);
							b = _mm_shuffle_epi8(b, mask2);
							c = _mm_shuffle_epi8(c, mask3);
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
						const uchar* gptrrj = gptrr + j;
						const uchar* gptrgj = gptrg + j;
						const uchar* gptrbj = gptrb + j;

						int sr0 = sptrrj[0];
						int sg0 = sptrgj[0];
						int sb0 = sptrbj[0];
						int gr0 = gptrrj[0];
						int gg0 = gptrgj[0];
						int gb0 = gptrbj[0];

						float sum_r = 0.f;
						float sum_b = 0.f;
						float sum_g = 0.f;
						float wsum = 0.f;
						for (k = 0; k < maxk; k++)
						{
							int gr = gptrrj[space_guide_ofs[k]];
							int gg = gptrgj[space_guide_ofs[k]];
							int gb = gptrbj[space_guide_ofs[k]];
							int sr = sptrrj[space_ofs[k]];
							int sg = sptrgj[space_ofs[k]];
							int sb = sptrbj[space_ofs[k]];
							float w = space_weight[k];
							w *= color_weight[abs(sb - sb0) + abs(sg - sg0) + abs(sr - sr0)];
							w *= guide_color_weight[abs(gb - gb0) + abs(gg - gg0) + abs(gr - gr0)];

							sum_r += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_b += sptrbj[space_ofs[k]] * w;
							wsum += w;
						}

						wsum = 1.f / wsum; sb0 = cvRound(sum_b*wsum); sg0 = cvRound(sum_g*wsum); sr0 = cvRound(sum_r*wsum);
						dptr[3 * j] = (uchar)sr0; dptr[3 * j + 1] = (uchar)sg0; dptr[3 * j + 2] = (uchar)sb0;
					}
				}
			}
			else if (cn == 3 && cng == 1)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;

				const int dstep = 3 * dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(radiusV + range.start) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, gptr += gstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							__m128i m1, m2, n1, n2;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrj = gptr + j;

							const __m128i bval = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i gval = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i rval = _mm_load_si128((__m128i*)(sptrrj));
							const __m128i ggval = _mm_load_si128((__m128i*)(gptrj));

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

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(ggval, bref), _mm_subs_epu8(bref, ggval)));

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

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

								__m128i r1 = _mm_unpacklo_epi8(rref, zero);
								__m128i r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);
								__m128i g1 = _mm_unpacklo_epi8(gref, zero);
								__m128i g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);
								__m128i b1 = _mm_unpacklo_epi8(bref, zero);
								__m128i b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
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

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));

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

								r1 = _mm_unpackhi_epi8(rref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								g1 = _mm_unpackhi_epi8(gref, zero);
								g2 = _mm_unpackhi_epi16(g1, zero);
								g1 = _mm_unpacklo_epi16(g1, zero);

								b1 = _mm_unpackhi_epi8(bref, zero);
								b2 = _mm_unpackhi_epi16(b1, zero);
								b1 = _mm_unpacklo_epi16(b1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));

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

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));

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
							uchar* dptrc = dptr + 3 * j;
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
						const uchar* gptrj = gptr + j;

						int r0 = sptrrj[0];
						int g0 = sptrgj[0];
						int b0 = sptrbj[0];

						int gv = gptrj[0];

						float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int gr = gptrj[space_guide_ofs[k]];

							float w = space_weight[k]
								* color_weight[std::abs(sptrrj[space_ofs[k]] - r0) + std::abs(sptrgj[space_ofs[k]] - g0) + std::abs(sptrbj[space_ofs[k]] - b0)]
								* guide_color_weight[std::abs(std::abs(gr - gv))];
							sum_b += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_r += sptrbj[space_ofs[k]] * w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[3 * j] = (uchar)cvRound(sum_b*wsum);
						dptr[3 * j + 1] = (uchar)cvRound(sum_g*wsum);
						dptr[3 * j + 2] = (uchar)cvRound(sum_r*wsum);
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		const Mat* guide;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};



	void weightedTrilateralFilter_32f(const Mat& src, Mat& weight, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		if (kernelSize.width <= 1 && kernelSize.height <= 1){ src.copyTo(dst); return; }

		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
			(guide.type() == CV_32FC1 || guide.type() == CV_32FC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		if (sigma_guide_color <= 0)
			sigma_guide_color = 1;
		if (sigma_color <= 0)
			sigma_color = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg, tempw;

		int dpad = (4 - src.cols % 4) % 4;
		int spad = dpad + (4 - (2 * radiusH) % 4) % 4;
		if (spad < 4) spad += 4;
		int lpad = 4 * (radiusH / 4 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);

			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);

			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);

			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}

		double minv, maxv;
		minMaxLoc(src, &minv, &maxv);
		const int color_range = cvRound(maxv - minv);
		vector<float> _color_weight(cn*color_range);
		minMaxLoc(guide, &minv, &maxv);
		const int color_range_guide = cvRound(maxv - minv);
		vector<float> _color_guide_weight(cng*color_range_guide);
		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];
		// initialize color-related bilateral filter coefficients
		for (i = 0; i < color_range*cn; i++)
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		for (i = 0; i < color_range_guide*cng; i++)
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);

		vector<float> _space_weight(kernelSize.area() + 1);

		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_w_ofs(kernelSize.area() + 1);
		vector<int> _space_guide_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_w_ofs = &_space_w_ofs[0];
		int* space_guide_ofs = &_space_guide_ofs[0];

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
				space_ofs[maxk] = (int)(i*temp.cols*cn + j);
				space_w_ofs[maxk] = (int)(i*tempw.cols + j);
				space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		WeightedTrilateralFilter_32f_InvokerSSE4 body(dest, temp, tempw, tempg, radiusH, radiusV, maxk, space_ofs, space_w_ofs, space_guide_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void weightedTrilateralFilter_8u(const Mat& src, Mat& weight, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		if (kernelSize.width <= 1 && kernelSize.height <= 1){ src.copyTo(dst); return; }

		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			(guide.type() == CV_8UC1 || guide.type() == CV_8UC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		if (sigma_guide_color <= 0)
			sigma_guide_color = 1;
		if (sigma_color <= 0)
			sigma_color = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);


		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg, tempw;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);

			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);

			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);

			copyMakeBorder(weight, tempw, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}

		vector<float> _color_weight(cn * 256);
		vector<float> _color_guide_weight(cng * 256);
		vector<float> _space_weight(kernelSize.area() + 1);
		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_w_ofs(kernelSize.area() + 1);
		vector<int> _space_guide_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_w_ofs = &_space_w_ofs[0];
		int* space_guide_ofs = &_space_guide_ofs[0];

		// initialize color-related bilateral filter coefficients

		for (i = 0; i < 256 * cn; i++)
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		for (i = 0; i < 256 * cng; i++)
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);

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
				space_ofs[maxk] = (int)(i*temp.cols*cn + j);
				space_w_ofs[maxk] = (int)(i*tempw.cols + j);
				space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		WeightedTrilateralFilter_8u_InvokerSSE4 body(dest, temp, tempw, tempg, radiusH, radiusV, maxk, space_ofs, space_w_ofs, space_guide_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}


	void dualBilateralFilter_32f(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, const double sigma_color, const double sigma_guide_color, const double sigma_space, const int borderType, const bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
			(guide.type() == CV_32FC1 || guide.type() == CV_32FC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg;

		int dpad = (4 - src.cols % 4) % 4;
		int spad = dpad + (4 - (2 * radiusH) % 4) % 4;
		if (spad < 4) spad += 4;
		int lpad = 4 * (radiusH / 4 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}

		double minv, maxv;
		minMaxLoc(src, &minv, &maxv);
		const int color_range = cvRound(maxv - minv) + 1;
		minMaxLoc(guide, &minv, &maxv);
		const int color_range_guide = cvRound(maxv - minv) + 1;

		vector<float> _color_weight(cn*color_range);
		vector<float> _color_guide_weight(cng*color_range_guide);

		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];

		for (i = 0; i < color_range*cn; i++)
		{
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		}
		for (i = 0; i < color_range_guide*cng; i++)
		{
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);
		}

		vector<float> _space_weight(kernelSize.area() + 1);
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_guide_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_guide_ofs = &_space_guide_ofs[0];

		// initialize space-related bilateral filter coefficients
		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				j = -radiusH;
				for (; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);

					space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
					space_ofs[maxk] = (int)(i*temp.cols*cn + j);
					space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				j = -radiusH;
				for (; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH)) continue;

					space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
					space_ofs[maxk] = (int)(i*temp.cols*cn + j);
					space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		DualBilateralFilter_32f_InvokerSSE4 body(dest, temp, tempg, radiusH, radiusV, maxk, space_ofs, space_guide_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}


	void dualBilateralFilter_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, const double sigma_color, const double sigma_guide_color, const double sigma_space, const int borderType, bool isRectangle)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			(guide.type() == CV_8UC1 || guide.type() == CV_8UC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);



		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}

		vector<float> _color_weight(cn * 256);
		vector<float> _color_guide_weight(cng * 256);
		vector<float> _space_weight(kernelSize.area() + 1);
		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_guide_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_guide_ofs = &_space_guide_ofs[0];

		// initialize color-related bilateral filter coefficients

		for (i = 0; i < 256 * cn; i++)
		{
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		}
		for (i = 0; i < 256 * cng; i++)
		{
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);
		}

		// initialize space-related bilateral filter coefficients
		if (isRectangle)
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				j = -radiusH;

				for (; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);

					space_ofs[maxk] = (int)(i*temp.cols*cn + j);
					space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
				}
			}
		}
		else
		{
			for (i = -radiusV, maxk = 0; i <= radiusV; i++)
			{
				j = -radiusH;

				for (; j <= radiusH; j++)
				{
					double r = std::sqrt((double)i*i + (double)j*j);
					if (r > max(radiusV, radiusH)) continue;

					space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);

					space_ofs[maxk] = (int)(i*temp.cols*cn + j);
					space_guide_ofs[maxk++] = (int)(i*tempg.cols*cng + j);
				}
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		DualBilateralFilter_8u_InvokerSSE4 body(dest, temp, tempg, radiusH, radiusV, maxk, space_ofs, space_guide_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void weightedTrilateralFilterSP_8u(const Mat& src, Mat& weightMap, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		weightedTrilateralFilter_8u(src, weightMap, guide, dst, Size(kernelSize.width, 1), sigma_color, sigma_guide_color, sigma_space, borderType);
		weightedTrilateralFilter_8u(dst, weightMap, guide, dst, Size(1, kernelSize.height), sigma_color, sigma_guide_color, sigma_space, borderType);
	}

	void weightedTrilateralFilterSP_32f(const Mat& src, Mat& weightMap, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		weightedTrilateralFilter_32f(src, weightMap, guide, dst, Size(kernelSize.width, 1), sigma_color, sigma_guide_color, sigma_space, borderType);
		weightedTrilateralFilter_32f(dst, weightMap, guide, dst, Size(1, kernelSize.height), sigma_color, sigma_guide_color, sigma_space, borderType);
	}

	void weightedTrilateralFilter(const Mat& src, Mat& weightMap, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int method, int borderType)
	{
		if (dst.empty())dst.create(src.size(), src.type());
		if (method == FILTER_CIRCLE)
		{
			if (src.type() == CV_MAKE_TYPE(CV_8U, src.channels()))
			{
				weightedTrilateralFilter_8u(src, weightMap, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
			else if (src.type() == CV_MAKE_TYPE(CV_32F, src.channels()))
			{
				weightedTrilateralFilter_32f(src, weightMap, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			if (src.type() == CV_MAKE_TYPE(CV_8U, src.channels()))
			{
				weightedTrilateralFilterSP_8u(src, weightMap, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
			else if (src.type() == CV_MAKE_TYPE(CV_32F, src.channels()))
			{
				weightedTrilateralFilterSP_32f(src, weightMap, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
		}
	}

	void weightedTrilateralFilter(const Mat& src, Mat& weightMap, const Mat& guide, Mat& dst, int d, double sigma_color, double sigma_guide_color, double sigma_space, int method, int borderType)
	{
		weightedTrilateralFilter(src, weightMap, guide, dst, Size(d, d), sigma_color, sigma_guide_color, sigma_space, method, borderType);
	}


	class DualBilateralWeightMap_32f_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		DualBilateralWeightMap_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float *_guide_color_weight) :
			temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = (temp->rows - 2 * radiusV) / dest->rows;
			int cng = (guide->rows - 2 * radiusV) / dest->rows;
			Size size = dest->size();

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
			const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
#endif
			if (cn == 1 && cng == 1)
			{
				int CV_DECL_ALIGNED(16) buf[4];
				int CV_DECL_ALIGNED(16) gbuf[4];

				float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
				float* gptr = (float*)guide->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							//int* gofs = &space_guide_ofs[0];//gofs =ofs

							float* spw = space_weight;

							const float* sptrj = sptr + j;
							const float* gptrj = gptr + j;

							const __m128 val0s = _mm_load_ps((sptrj));
							const __m128 val0j = _mm_load_ps((gptrj));

							//重みと平滑化後の画素４ブロックづつ
							__m128 wval1 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, spw++)
							{
								__m128 sref = _mm_loadu_ps((sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0s, sref), *(const __m128*)v32f_absmask)));

								sref = _mm_loadu_ps((gptrj + *ofs));
								_mm_store_si128((__m128i*)gbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0j, sref), *(const __m128*)v32f_absmask)));

								__m128 _w = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア

								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

							}
							_mm_stream_ps(dptr + j, wval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float val0 = sptr[j];
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)//あまりもの処理o
						{
							float val = sptr[j + space_ofs[k]];
							float w = space_weight[k] * color_weight[cvRound(std::abs(val - val0))];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				int CV_DECL_ALIGNED(16) buf[16];
				int CV_DECL_ALIGNED(16) gbuf[4];

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;
				const int dstep = dest->cols;

				float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* gptrr = (float*)guide->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* gptrg = (float*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* gptrb = (float*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);
				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							//int* gofs = &space_guide_ofs[0];//gofs is same as ofs (s and g have same color channel case)

							float* spw = space_weight;

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;
							const float* gptrrj = gptrr + j;
							const float* gptrgj = gptrg + j;
							const float* gptrbj = gptrb + j;

							const __m128 bval0s = _mm_load_ps((sptrbj));
							const __m128 gval0s = _mm_load_ps((sptrgj));
							const __m128 rval0s = _mm_load_ps((sptrrj));
							const __m128 bval0j = _mm_load_ps((gptrbj));
							const __m128 gval0j = _mm_load_ps((gptrgj));
							const __m128 rval0j = _mm_load_ps((gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							for (k = 0; k < maxk; k++, ofs++, spw++)
							{
								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								_mm_store_si128((__m128i*)buf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval0s, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval0s, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval0s, bref), *(const __m128*)v32f_absmask)
									)
									));

								bref = _mm_loadu_ps((gptrbj + *ofs));
								gref = _mm_loadu_ps((gptrgj + *ofs));
								rref = _mm_loadu_ps((gptrrj + *ofs));

								_mm_store_si128((__m128i*)gbuf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval0j, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval0j, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval0j, bref), *(const __m128*)v32f_absmask)
									)
									));

								__m128 _w = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;
						const float r0 = sptrrj[0];
						const float g0 = sptrgj[0];
						const float b0 = sptrbj[0];

						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = space_weight[k] * color_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}
			else if (cn == 1 && cng == 3)
			{
				int CV_DECL_ALIGNED(16) buf[16];
				int CV_DECL_ALIGNED(16) gbuf[4];

				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;
				const int dstep = dest->cols;

				float* sptr = (float*)temp->ptr(radiusV + range.start) + 4 * (radiusH / 4 + 1);
				float* gptrr = (float*)guide->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* gptrg = (float*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* gptrb = (float*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptr += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const float* sptrj = sptr + j;
							const float* gptrrj = gptrr + j;
							const float* gptrgj = gptrg + j;
							const float* gptrbj = gptrb + j;

							const __m128 val0s = _mm_load_ps((sptrj));
							const __m128 bval0j = _mm_load_ps((gptrbj));
							const __m128 gval0j = _mm_load_ps((gptrgj));
							const __m128 rval0j = _mm_load_ps((gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128 bref = _mm_loadu_ps((gptrbj + *gofs));
								__m128 gref = _mm_loadu_ps((gptrgj + *gofs));
								__m128 rref = _mm_loadu_ps((gptrrj + *gofs));

								_mm_store_si128((__m128i*)gbuf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval0j, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval0j, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval0j, bref), *(const __m128*)v32f_absmask)
									)
									));

								bref = _mm_loadu_ps((sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0s, bref), *(const __m128*)v32f_absmask)));


								__m128 _w = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						/*
						const float* sptrj = sptr+j;
						const float r0 = sptrrj[0];
						const float g0 = sptrgj[0];
						const float b0 = sptrbj[0];
						*/
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							//	float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							//	float w = space_weight[k]*color_weight[cvRound(std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0))];
							//	wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}
			else if (cn == 3 && cng == 1)
			{
				int CV_DECL_ALIGNED(16) buf[16];
				int CV_DECL_ALIGNED(16) gbuf[4];

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;
				const int dstep = dest->cols;

				float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
				float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
				float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);
				float* gptr = (float*)guide->ptr(radiusV + range.start) + 4 * (radiusH / 4 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptr += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 4)//4 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const float* sptrrj = sptrr + j;
							const float* sptrgj = sptrg + j;
							const float* sptrbj = sptrb + j;
							const float* gptrj = gptr + j;

							const __m128 val0j = _mm_load_ps((gptrj));
							const __m128 bval0s = _mm_load_ps((sptrbj));
							const __m128 gval0s = _mm_load_ps((sptrgj));
							const __m128 rval0s = _mm_load_ps((sptrrj));

							__m128 wval1 = _mm_setzero_ps();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
								__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
								__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

								_mm_store_si128((__m128i*)buf,
									_mm_cvtps_epi32(
									_mm_add_ps(
									_mm_add_ps(
									_mm_and_ps(_mm_sub_ps(rval0s, rref), *(const __m128*)v32f_absmask),
									_mm_and_ps(_mm_sub_ps(gval0s, gref), *(const __m128*)v32f_absmask)),
									_mm_and_ps(_mm_sub_ps(bval0s, bref), *(const __m128*)v32f_absmask)
									)
									));

								bref = _mm_loadu_ps((gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(val0j, bref), *(const __m128*)v32f_absmask)));


								__m128 _w = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_w, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						/*
						const float* sptrj = sptr+j;
						const float r0 = sptrrj[0];
						const float g0 = sptrgj[0];
						const float b0 = sptrbj[0];
						*/
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							//	float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							//	float w = space_weight[k]*color_weight[cvRound(std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0))];
							//	wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}
		}
	private:
		const Mat* guide;
		const Mat *temp;

		Mat *dest;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};

	// s = 1 j=3 support only 
	class TrilateralWeightMapXOR_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		TrilateralWeightMapXOR_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float *_guide_color_weight) :
			temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = (temp->rows - 2 * radiusV) / dest->rows;
			int cng = (guide->rows - 2 * radiusV) / dest->rows;

			Size size = dest->size();

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						//for(; j < 0; j+=16)//16画素づつ処理
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrj = gptr + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, sref), _mm_subs_epu8(sref, sval)));

								sref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(gval, sref), _mm_subs_epu8(sref, gval)));//guide weight

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア

								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar val0 = sptr[j];
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int val = sptr[j + space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(val - val0)];
							wsum += w;
						}
						dptr[j] = wsum;
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;

				const int dstep = dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16画素づつ処理
						{
							__m128 _w;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i gb = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gg = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i gr = _mm_load_si128((__m128i*)(gptrrj));
							const __m128i sb = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i sg = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i sr = _mm_load_si128((__m128i*)(sptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrbj + *gofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(gptrgj + *gofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(gptrrj + *gofs));

								__m128i r1 = _mm_add_epi8(_mm_subs_epu8(gr, rref), _mm_subs_epu8(rref, gr));
								__m128i r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gg, gref), _mm_subs_epu8(gref, gg));
								__m128i g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								__m128i b1 = _mm_add_epi8(_mm_subs_epu8(gb, bref), _mm_subs_epu8(bref, gb));
								__m128i b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(gbuf + 8), r2);
								_mm_store_si128((__m128i*)gbuf, r1);

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								r1 = _mm_add_epi8(_mm_subs_epu8(sr, rref), _mm_subs_epu8(rref, sr));
								r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								g1 = _mm_add_epi8(_mm_subs_epu8(sg, gref), _mm_subs_epu8(gref, sg));
								g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								b1 = _mm_add_epi8(_mm_subs_epu8(sb, bref), _mm_subs_epu8(bref, sb));
								b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(buf + 8), r2);
								_mm_store_si128((__m128i*)buf, r1);

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;
						const int r0 = sptrrj[0];
						const int g0 = sptrgj[0];
						const int b0 = sptrbj[0];

						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}

			else if (cn == 1 && cng == 3)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;

				const int dstep = dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptr += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16画素づつ処理
							//for(; j < 0; j+=16)//16画素づつ処理
						{
							__m128i m1, m2, n1, n2;
							__m128 _w;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

							const __m128i bval = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i rval = _mm_load_si128((__m128i*)(gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128 ones = _mm_set1_ps(1.f);
							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								//const __m128i bref = _mm_loadu_si128((__m128i*)(gptrbj+*gofs));
								//const __m128i gref = _mm_loadu_si128((__m128i*)(gptrgj+*gofs));
								//const __m128i rref = _mm_loadu_si128((__m128i*)(gptrrj+*gofs));

								//m1 = _mm_add_epi8(_mm_subs_epu8(rval,rref),_mm_subs_epu8(rref,rval));
								//m2 = _mm_unpackhi_epi8(m1,zero);
								//m1 = _mm_unpacklo_epi8(m1,zero);

								//n1 = _mm_add_epi8(_mm_subs_epu8(gval,gref),_mm_subs_epu8(gref,gval));
								//n2 = _mm_unpackhi_epi8(n1,zero);
								//n1 = _mm_unpacklo_epi8(n1,zero);

								//m1 = _mm_add_epi16(m1,n1);
								//m2 = _mm_add_epi16(m2,n2);

								//n1 = _mm_add_epi8(_mm_subs_epu8(bval,bref),_mm_subs_epu8(bref,bval));
								//n2 = _mm_unpackhi_epi8(n1,zero);
								//n1 = _mm_unpacklo_epi8(n1,zero);

								//m1 = _mm_add_epi16(m1,n1);
								//m2 = _mm_add_epi16(m2,n2);

								//_mm_store_si128((__m128i*)(gbuf+8),m2);
								//_mm_store_si128((__m128i*)gbuf,m1);

								//const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
								//_mm_store_si128((__m128i*)buf,_mm_add_epi8(_mm_subs_epu8(sval,vref),_mm_subs_epu8(vref,sval)));

								//const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]],guide_color_weight[gbuf[2]],guide_color_weight[gbuf[1]],guide_color_weight[gbuf[0]]));
								//wval1 = _mm_add_ps(wval1,_w);

								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]],guide_color_weight[gbuf[6]],guide_color_weight[gbuf[5]],guide_color_weight[gbuf[4]]));
								//wval2 = _mm_add_ps(wval2,_w);

								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]],guide_color_weight[gbuf[10]],guide_color_weight[gbuf[9]],guide_color_weight[gbuf[8]]));
								//wval3 = _mm_add_ps(wval3,_w);

								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]],guide_color_weight[gbuf[14]],guide_color_weight[gbuf[13]],guide_color_weight[gbuf[12]]));
								//wval4 = _mm_add_ps(wval4,_w);

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

								_mm_store_si128((__m128i*)(gbuf + 8), m2);
								_mm_store_si128((__m128i*)gbuf, m1);

								const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, vref), _mm_subs_epu8(vref, sval)));

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア

								__m128 cw = _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]);
								__m128 dw = _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]);
								_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones, cw), dw), _mm_mul_ps(_mm_sub_ps(ones, dw), cw));
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval1 = _mm_add_ps(wval1, _w);

								cw = _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]);
								dw = _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]);
								_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones, cw), dw), _mm_mul_ps(_mm_sub_ps(ones, dw), cw));
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval2 = _mm_add_ps(wval2, _w);

								cw = _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]);
								dw = _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]);
								_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones, cw), dw), _mm_mul_ps(_mm_sub_ps(ones, dw), cw));
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval3 = _mm_add_ps(wval3, _w);


								cw = _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]);
								dw = _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]);
								_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones, cw), dw), _mm_mul_ps(_mm_sub_ps(ones, dw), cw));
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						/*const uchar* sptrj = sptr+j;
						const int r0 = sptrj[0];


						float wsum=0.0f;
						for(k=0 ; k < maxk; k++ )
						{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0)];
						wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;*/
					}
				}
			}
			else if (cn == 3 && cng == 1)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(radiusV + range.start) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptr += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//4 pixel unit
						{
							__m128i m1, m2, n1, n2;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrj = gptr + j;

							const __m128i bval0s = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i gval0s = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i rval0s = _mm_load_si128((__m128i*)(sptrrj));
							const __m128i val0j = _mm_load_si128((__m128i*)(gptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(val0j, bref), _mm_subs_epu8(bref, val0j)));

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								m1 = _mm_add_epi8(_mm_subs_epu8(rval0s, rref), _mm_subs_epu8(rref, rval0s));
								m2 = _mm_unpackhi_epi8(m1, zero);
								m1 = _mm_unpacklo_epi8(m1, zero);

								n1 = _mm_add_epi8(_mm_subs_epu8(gval0s, gref), _mm_subs_epu8(gref, gval0s));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								n1 = _mm_add_epi8(_mm_subs_epu8(bval0s, bref), _mm_subs_epu8(bref, bval0s));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								_mm_store_si128((__m128i*)(buf + 8), m2);
								_mm_store_si128((__m128i*)buf, m1);

								const __m128 _sw = _mm_set1_ps(*spw);
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;
						const int r0 = sptrrj[0];
						const int g0 = sptrgj[0];
						const int b0 = sptrbj[0];

						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		const Mat* guide;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};

	// s = 1 j=3 support only 
	class TrilateralWeightMapSGB_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		TrilateralWeightMapSGB_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float *_guide_color_weight) :
			temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = (temp->rows - 2 * radiusV) / dest->rows;
			int cng = (guide->rows - 2 * radiusV) / dest->rows;

			Size size = dest->size();

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrj = gptr + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, sref), _mm_subs_epu8(sref, sval)));

								sref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(gval, sref), _mm_subs_epu8(sref, gval)));//guide weight

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア

								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar val0 = sptr[j];
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int val = sptr[j + space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(val - val0)];
							wsum += w;
						}
						dptr[j] = wsum;
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;

				const int dstep = dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16画素づつ処理
						{
							__m128 _w;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i gb = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gg = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i gr = _mm_load_si128((__m128i*)(gptrrj));
							const __m128i sb = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i sg = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i sr = _mm_load_si128((__m128i*)(sptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrbj + *gofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(gptrgj + *gofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(gptrrj + *gofs));

								__m128i r1 = _mm_add_epi8(_mm_subs_epu8(gr, rref), _mm_subs_epu8(rref, gr));
								__m128i r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gg, gref), _mm_subs_epu8(gref, gg));
								__m128i g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								__m128i b1 = _mm_add_epi8(_mm_subs_epu8(gb, bref), _mm_subs_epu8(bref, gb));
								__m128i b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(gbuf + 8), r2);
								_mm_store_si128((__m128i*)gbuf, r1);

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								r1 = _mm_add_epi8(_mm_subs_epu8(sr, rref), _mm_subs_epu8(rref, sr));
								r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								g1 = _mm_add_epi8(_mm_subs_epu8(sg, gref), _mm_subs_epu8(gref, sg));
								g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								b1 = _mm_add_epi8(_mm_subs_epu8(sb, bref), _mm_subs_epu8(bref, sb));
								b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(buf + 8), r2);
								_mm_store_si128((__m128i*)buf, r1);

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;
						const int r0 = sptrrj[0];
						const int g0 = sptrgj[0];
						const int b0 = sptrbj[0];

						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}

			else if (cn == 1 && cng == 3)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;

				const int dstep = dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptr += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16画素づつ処理
							//for(; j < 0; j+=16)//16画素づつ処理
						{
							__m128i m1, m2, n1, n2;
							__m128 _w;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

							const __m128i bval = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i rval = _mm_load_si128((__m128i*)(gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128 ones = _mm_set1_ps(1.f);
							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								//const __m128i bref = _mm_loadu_si128((__m128i*)(gptrbj+*gofs));
								//const __m128i gref = _mm_loadu_si128((__m128i*)(gptrgj+*gofs));
								//const __m128i rref = _mm_loadu_si128((__m128i*)(gptrrj+*gofs));

								//m1 = _mm_add_epi8(_mm_subs_epu8(rval,rref),_mm_subs_epu8(rref,rval));
								//m2 = _mm_unpackhi_epi8(m1,zero);
								//m1 = _mm_unpacklo_epi8(m1,zero);

								//n1 = _mm_add_epi8(_mm_subs_epu8(gval,gref),_mm_subs_epu8(gref,gval));
								//n2 = _mm_unpackhi_epi8(n1,zero);
								//n1 = _mm_unpacklo_epi8(n1,zero);

								//m1 = _mm_add_epi16(m1,n1);
								//m2 = _mm_add_epi16(m2,n2);

								//n1 = _mm_add_epi8(_mm_subs_epu8(bval,bref),_mm_subs_epu8(bref,bval));
								//n2 = _mm_unpackhi_epi8(n1,zero);
								//n1 = _mm_unpacklo_epi8(n1,zero);

								//m1 = _mm_add_epi16(m1,n1);
								//m2 = _mm_add_epi16(m2,n2);

								//_mm_store_si128((__m128i*)(gbuf+8),m2);
								//_mm_store_si128((__m128i*)gbuf,m1);

								//const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
								//_mm_store_si128((__m128i*)buf,_mm_add_epi8(_mm_subs_epu8(sval,vref),_mm_subs_epu8(vref,sval)));

								//const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]],guide_color_weight[gbuf[2]],guide_color_weight[gbuf[1]],guide_color_weight[gbuf[0]]));
								//wval1 = _mm_add_ps(wval1,_w);

								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]],guide_color_weight[gbuf[6]],guide_color_weight[gbuf[5]],guide_color_weight[gbuf[4]]));
								//wval2 = _mm_add_ps(wval2,_w);

								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]],guide_color_weight[gbuf[10]],guide_color_weight[gbuf[9]],guide_color_weight[gbuf[8]]));
								//wval3 = _mm_add_ps(wval3,_w);

								//_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								//_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]],guide_color_weight[gbuf[14]],guide_color_weight[gbuf[13]],guide_color_weight[gbuf[12]]));
								//wval4 = _mm_add_ps(wval4,_w);

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

								_mm_store_si128((__m128i*)(gbuf + 8), m2);
								_mm_store_si128((__m128i*)gbuf, m1);

								const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, vref), _mm_subs_epu8(vref, sval)));

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア

								__m128 cw = _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]);
								__m128 dw = _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]);
								//_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones,cw), dw), _mm_mul_ps(_mm_sub_ps(ones,dw), cw));
								_w = _mm_mul_ps(_mm_sub_ps(ones, cw), dw);
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval1 = _mm_add_ps(wval1, _w);

								cw = _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]);
								dw = _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]);
								//_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones,cw), dw), _mm_mul_ps(_mm_sub_ps(ones,dw), cw));
								_w = _mm_mul_ps(_mm_sub_ps(ones, cw), dw);
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval2 = _mm_add_ps(wval2, _w);

								cw = _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]);
								dw = _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]);
								//_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones,cw), dw), _mm_mul_ps(_mm_sub_ps(ones,dw), cw));
								_w = _mm_mul_ps(_mm_sub_ps(ones, cw), dw);
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval3 = _mm_add_ps(wval3, _w);


								cw = _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]);
								dw = _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]);
								//_w = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(ones,cw), dw), _mm_mul_ps(_mm_sub_ps(ones,dw), cw));
								_w = _mm_mul_ps(_mm_sub_ps(ones, cw), dw);
								_w = _mm_sub_ps(ones, _w);
								_w = _mm_mul_ps(_sw, _w);
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						/*const uchar* sptrj = sptr+j;
						const int r0 = sptrj[0];


						float wsum=0.0f;
						for(k=0 ; k < maxk; k++ )
						{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0)];
						wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;*/
					}
				}
			}
			else if (cn == 3 && cng == 1)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(radiusV + range.start) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptr += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//4 pixel unit
						{
							__m128i m1, m2, n1, n2;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrj = gptr + j;

							const __m128i bval0s = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i gval0s = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i rval0s = _mm_load_si128((__m128i*)(sptrrj));
							const __m128i val0j = _mm_load_si128((__m128i*)(gptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(val0j, bref), _mm_subs_epu8(bref, val0j)));

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								m1 = _mm_add_epi8(_mm_subs_epu8(rval0s, rref), _mm_subs_epu8(rref, rval0s));
								m2 = _mm_unpackhi_epi8(m1, zero);
								m1 = _mm_unpacklo_epi8(m1, zero);

								n1 = _mm_add_epi8(_mm_subs_epu8(gval0s, gref), _mm_subs_epu8(gref, gval0s));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								n1 = _mm_add_epi8(_mm_subs_epu8(bval0s, bref), _mm_subs_epu8(bref, bval0s));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								_mm_store_si128((__m128i*)(buf + 8), m2);
								_mm_store_si128((__m128i*)buf, m1);

								const __m128 _sw = _mm_set1_ps(*spw);
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;
						const int r0 = sptrrj[0];
						const int g0 = sptrgj[0];
						const int b0 = sptrbj[0];

						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		const Mat* guide;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};

	class TrilateralWeightMap_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		TrilateralWeightMap_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, float *_guide_color_weight) :
			temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_guide_ofs(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), guide_color_weight(_guide_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{
			int i, j, k;
			int cn = (temp->rows - 2 * radiusV) / dest->rows;
			int cng = (guide->rows - 2 * radiusV) / dest->rows;

			Size size = dest->size();

#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng == 1)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				float* dptr = dest->ptr<float>(range.start);

				const int sstep = temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, gptr += gstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						//for(; j < 0; j+=16)//16画素づつ処理
						for (; j < size.width; j += 16)//16 pixel unit
						{
							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrj = gptr + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, sref), _mm_subs_epu8(sref, sval)));

								sref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(gval, sref), _mm_subs_epu8(sref, gval)));//guide weight

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア

								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar val0 = sptr[j];
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int val = sptr[j + space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(val - val0)];
							wsum += w;
						}
						dptr[j] = wsum;
					}
				}
			}
			else if (cn == 3 && cng == 3)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = 3 * guide->cols;

				const int dstep = dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//16画素づつ処理
						{
							__m128 _w;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i gb = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gg = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i gr = _mm_load_si128((__m128i*)(gptrrj));
							const __m128i sb = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i sg = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i sr = _mm_load_si128((__m128i*)(sptrrj));


							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrbj + *gofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(gptrgj + *gofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(gptrrj + *gofs));

								__m128i r1 = _mm_add_epi8(_mm_subs_epu8(gr, rref), _mm_subs_epu8(rref, gr));
								__m128i r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gg, gref), _mm_subs_epu8(gref, gg));
								__m128i g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								__m128i b1 = _mm_add_epi8(_mm_subs_epu8(gb, bref), _mm_subs_epu8(bref, gb));
								__m128i b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(gbuf + 8), r2);
								_mm_store_si128((__m128i*)gbuf, r1);

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								r1 = _mm_add_epi8(_mm_subs_epu8(sr, rref), _mm_subs_epu8(rref, sr));
								r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								g1 = _mm_add_epi8(_mm_subs_epu8(sg, gref), _mm_subs_epu8(gref, sg));
								g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								b1 = _mm_add_epi8(_mm_subs_epu8(sb, bref), _mm_subs_epu8(bref, sb));
								b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(buf + 8), r2);
								_mm_store_si128((__m128i*)buf, r1);

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;
						const int r0 = sptrrj[0];
						const int g0 = sptrgj[0];
						const int b0 = sptrbj[0];

						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}

			else if (cn == 1 && cng == 3)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = temp->cols;
				const int gstep = 3 * guide->cols;

				const int dstep = dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* gptrr = (uchar*)guide->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* gptrg = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* gptrb = (uchar*)guide->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptr += sstep, gptrr += gstep, gptrg += gstep, gptrb += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							__m128i m1, m2, n1, n2;
							__m128 _w;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* gptrrj = gptrr + j;
							const uchar* gptrgj = gptrg + j;
							const uchar* gptrbj = gptrb + j;

							const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

							const __m128i bval = _mm_load_si128((__m128i*)(gptrbj));
							const __m128i gval = _mm_load_si128((__m128i*)(gptrgj));
							const __m128i rval = _mm_load_si128((__m128i*)(gptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
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

								_mm_store_si128((__m128i*)(gbuf + 8), m2);
								_mm_store_si128((__m128i*)gbuf, m1);

								const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval, vref), _mm_subs_epu8(vref, sval)));

								const __m128 _sw = _mm_set1_ps(*spw);//位置のexp重みをレジスタにストア
								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));//メモリ上の絶対値差をexpを表すLUTに入れてそれをレジスタにストア（色重み）
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						/*const uchar* sptrj = sptr+j;
						const int r0 = sptrj[0];


						float wsum=0.0f;
						for(k=0 ; k < maxk; k++ )
						{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0)];
						wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;*/
					}
				}
			}
			else if (cn == 3 && cng == 1)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int gstep = guide->cols;

				const int dstep = dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* gptr = (uchar*)guide->ptr(radiusV + range.start) + 16 * (radiusH / 16 + 1);

				float* dptr = dest->ptr<float>(range.start);

				for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, gptr += gstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)//4 pixel unit
						{
							__m128i m1, m2, n1, n2;

							int* ofs = &space_ofs[0];
							int* gofs = &space_guide_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* gptrj = gptr + j;

							const __m128i bval0s = _mm_load_si128((__m128i*)(sptrbj));
							const __m128i gval0s = _mm_load_si128((__m128i*)(sptrgj));
							const __m128i rval0s = _mm_load_si128((__m128i*)(sptrrj));
							const __m128i val0j = _mm_load_si128((__m128i*)(gptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, gofs++, spw++)
							{
								__m128i bref = _mm_loadu_si128((__m128i*)(gptrj + *gofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(val0j, bref), _mm_subs_epu8(bref, val0j)));

								bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

								m1 = _mm_add_epi8(_mm_subs_epu8(rval0s, rref), _mm_subs_epu8(rref, rval0s));
								m2 = _mm_unpackhi_epi8(m1, zero);
								m1 = _mm_unpacklo_epi8(m1, zero);

								n1 = _mm_add_epi8(_mm_subs_epu8(gval0s, gref), _mm_subs_epu8(gref, gval0s));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								n1 = _mm_add_epi8(_mm_subs_epu8(bval0s, bref), _mm_subs_epu8(bref, bval0s));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								_mm_store_si128((__m128i*)(buf + 8), m2);
								_mm_store_si128((__m128i*)buf, m1);

								const __m128 _sw = _mm_set1_ps(*spw);
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[3]], color_weight[buf[2]], color_weight[buf[1]], color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[3]], guide_color_weight[gbuf[2]], guide_color_weight[gbuf[1]], guide_color_weight[gbuf[0]]));
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[7]], color_weight[buf[6]], color_weight[buf[5]], color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[7]], guide_color_weight[gbuf[6]], guide_color_weight[gbuf[5]], guide_color_weight[gbuf[4]]));
								wval2 = _mm_add_ps(wval2, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[11]], color_weight[buf[10]], color_weight[buf[9]], color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[11]], guide_color_weight[gbuf[10]], guide_color_weight[gbuf[9]], guide_color_weight[gbuf[8]]));
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(color_weight[buf[15]], color_weight[buf[14]], color_weight[buf[13]], color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide_color_weight[gbuf[15]], guide_color_weight[gbuf[14]], guide_color_weight[gbuf[13]], guide_color_weight[gbuf[12]]));
								wval4 = _mm_add_ps(wval4, _w);
							}
							_mm_stream_ps(dptr + j, wval1);
							_mm_stream_ps(dptr + j + 4, wval2);
							_mm_stream_ps(dptr + j + 8, wval3);
							_mm_stream_ps(dptr + j + 12, wval4);
						}
					}
#endif
					for (; j < size.width; j++)
					{
						const uchar* sptrrj = sptrr + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrbj = sptrb + j;
						const int r0 = sptrrj[0];
						const int g0 = sptrgj[0];
						const int b0 = sptrbj[0];

						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
							float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							wsum += w;
						}
						//overflow is not possible here => there is no need to use CV_CAST_8U
						dptr[j] = wsum;
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		const Mat* guide;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide_ofs;
		float *space_weight, *color_weight, *guide_color_weight;
	};

	void dualBilateralWeightMap_32f(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
			src.size() == dst.size() &&
			src.data != dst.data);

		if (sigma_guide_color <= 0)
			sigma_guide_color = 1;
		if (sigma_color <= 0)
			sigma_color = 1;
		if (sigma_space <= 0)
			sigma_space = 1;


		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;
		Mat temp, tempg;

		int dpad = (4 - src.cols % 4) % 4;
		int spad = dpad + (4 - (2 * radiusH) % 4) % 4;
		if (spad < 4) spad += 4;
		int lpad = 4 * (radiusH / 4 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
	
		double minv, maxv;
		minMaxLoc(src, &minv, &maxv);
		const int color_range = cvRound(maxv - minv);
		vector<float> _color_weight(cn*color_range);
		minMaxLoc(guide, &minv, &maxv);
		const int color_range_guide = cvRound(maxv - minv);
		vector<float> _color_guide_weight(cng*color_range_guide);
		float* color_weight = &_color_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];
		// initialize color-related bilateral filter coefficients
		
		for (i = 0; i < color_range*cn; i++)
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		for (i = 0; i < color_range_guide*cng; i++)
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);

		vector<float> _space_weight(kernelSize.area() + 1);
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_g_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_g_ofs = &_space_g_ofs[0];
		
		// initialize space-related bilateral filter coefficients
		for (i = -radiusV, maxk = 0; i <= radiusV; i++)
		{
			for (j = -radiusH; j <= radiusH; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > max(radiusV, radiusH)) continue;
				space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
				space_g_ofs[maxk] = (int)(i*tempg.cols*cng + j);
				space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
			}
		}
		
		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), CV_32F);
		DualBilateralWeightMap_32f_InvokerSSE4 body(dest, temp, tempg, radiusH, radiusV, maxk, space_ofs, space_g_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void dualBilateralWeightMapXOR_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			src.size() == dst.size() &&
			src.data != dst.data);

		if (sigma_guide_color <= 0)
			sigma_guide_color = 1;
		if (sigma_color <= 0)
			sigma_color = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}

		vector<float> _color_weight(cn * 256);
		vector<float> _color_guide_weight(cng * 256);
		vector<float> _space_weight(kernelSize.area() + 1);
		float* color_weight = &_color_weight[0];
		float* space_weight = &_space_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_g_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_g_ofs = &_space_g_ofs[0];

		// initialize color-related bilateral filter coefficients

		for (i = 0; i < 256 * cn; i++)
		{
			int v = (int)max(i - sigma_color, 0.0);
			color_weight[i] = (float)max(1.0 - 1.0 / (sigma_color*sigma_color)*v*v, 0.0);//1.0- (float)std::exp(i*i*gauss_color_coeff);
		}
		for (i = 0; i < 256 * cng; i++)
		{
			int v = max(i - 2, 0);
			//color_guide_weight[i] = (float)std::exp(v*v*gauss_guide_color_coeff);
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);
		}

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
				space_g_ofs[maxk] = (int)(i*tempg.cols*cng + j);
				space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), CV_32F);
		TrilateralWeightMapXOR_8u_InvokerSSE4 body(dest, temp, tempg, radiusH, radiusV, maxk, space_ofs, space_g_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void dualBilateralWeightMapSGB_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			src.size() == dst.size() &&
			src.data != dst.data);

		if (sigma_guide_color <= 0)
			sigma_guide_color = 1;
		if (sigma_color <= 0)
			sigma_color = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}

		vector<float> _color_weight(cn * 256);
		vector<float> _color_guide_weight(cng * 256);
		vector<float> _space_weight(kernelSize.area() + 1);
		float* color_weight = &_color_weight[0];
		float* space_weight = &_space_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_g_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_g_ofs = &_space_g_ofs[0];

		// initialize color-related bilateral filter coefficients

		for (i = 0; i < 256 * cn; i++)
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		for (i = 0; i < 256 * cng; i++)
		{
			int v = max(i - 2, 0);
			color_guide_weight[i] = (float)std::exp(v*v*gauss_guide_color_coeff);
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);
		}

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
				space_g_ofs[maxk] = (int)(i*tempg.cols*cng + j);
				space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), CV_32F);
		TrilateralWeightMapSGB_8u_InvokerSSE4 body(dest, temp, tempg, radiusH, radiusV, maxk, space_ofs, space_g_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void dualBilateralWeightMap_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		int cn = src.channels();
		int cng = guide.channels();
		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			src.size() == dst.size() &&
			src.data != dst.data);

		if (sigma_guide_color <= 0)
			sigma_guide_color = 1;
		if (sigma_color <= 0)
			sigma_color = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
		double gauss_guide_color_coeff = -0.5 / (sigma_guide_color*sigma_guide_color);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;

		if (cn == 1 && cng == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg);
		}
		else if (cn == 3 && cng == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);

			copyMakeBorder(guide, tempg, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}

		vector<float> _color_weight(cn * 256);
		vector<float> _color_guide_weight(cng * 256);
		vector<float> _space_weight(kernelSize.area() + 1);
		float* color_weight = &_color_weight[0];
		float* space_weight = &_space_weight[0];
		float* color_guide_weight = &_color_guide_weight[0];

		vector<int> _space_ofs(kernelSize.area() + 1);
		vector<int> _space_g_ofs(kernelSize.area() + 1);
		int* space_ofs = &_space_ofs[0];
		int* space_g_ofs = &_space_g_ofs[0];

		// initialize color-related bilateral filter coefficients

		for (i = 0; i < 256 * cn; i++)
			color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
		for (i = 0; i < 256 * cng; i++)
		{
			int v = max(i - 2, 0);
			color_guide_weight[i] = (float)std::exp(v*v*gauss_guide_color_coeff);
			color_guide_weight[i] = (float)std::exp(i*i*gauss_guide_color_coeff);
		}

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
				space_g_ofs[maxk] = (int)(i*tempg.cols*cng + j);
				space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), CV_32F);
		TrilateralWeightMap_8u_InvokerSSE4 body(dest, temp, tempg, radiusH, radiusV, maxk, space_ofs, space_g_ofs, space_weight, color_weight, color_guide_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void dualBilateralWeightMapSP_32f(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		dualBilateralWeightMap_32f(src, guide, dst, Size(kernelSize.width, 1), sigma_color, sigma_guide_color, sigma_space, borderType);
		dualBilateralWeightMap_32f(dst, guide, dst, Size(1, kernelSize.height), sigma_color, sigma_guide_color, sigma_space, borderType);
	}

	void dualBilateralWeightMapSP_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		dualBilateralWeightMap_8u(src, guide, dst, Size(kernelSize.width, 1), sigma_color, sigma_guide_color, sigma_space, borderType);
		dualBilateralWeightMap_8u(dst, guide, dst, Size(1, kernelSize.height), sigma_color, sigma_guide_color, sigma_space, borderType);
	}

	void dualBilateralWeightMap(InputArray src_, InputArray guide_, OutputArray dest, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int method, int borderType)
	{
		if (dest.empty() || dest.depth() != CV_32F || src_.size() != dest.size()) dest.create(src_.size(), CV_32F);
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();
		Mat dst = dest.getMat();

		if (method == FILTER_CIRCLE || method == FILTER_DEFAULT)
		{
			if (src.depth() == CV_8U)
			{
				dualBilateralWeightMap_8u(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
				//	dualBilateralWeightMapBase(src, guide, dst, kernelSize.width, sigma_color, sigma_guide_color, sigma_space, borderType);

			}
			else if (src.depth() == CV_32F)
			{
				dualBilateralWeightMap_32f(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			if (src.depth() == CV_8U)
			{
				dualBilateralWeightMapSP_8u(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
			else if (src.depth() == CV_32F)
			{
				dualBilateralWeightMapSP_32f(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
		}
		else
		{
			cout << "not suported "<<endl;
		}
	}

	void dualBilateralWeightMapXOR(InputArray src_, InputArray guide_, OutputArray dest, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int method, int borderType)
	{
		if (dest.empty())dest.create(src_.size(), CV_32F);
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();
		Mat dst = dest.getMat();

		if (method == FILTER_CIRCLE)
		{
			if (src.depth() == CV_8U)
			{
				dualBilateralWeightMapXOR_8u(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
			else if (src.depth() == CV_32F)
			{
				//dualBilateralWeightMap_32f(src,guide,dst,kernelSize,sigma_color,sigma_guide_color, sigma_space,borderType);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			if (src.depth() == CV_8U)
			{
				//dualBilateralWeightMapSP_8u(src,guide,dst,kernelSize,sigma_color,sigma_guide_color, sigma_space,borderType);
			}
			else if (src.depth() == CV_32F)
			{
				//dualBilateralWeightMapSP_32f(src,guide,dst,kernelSize,sigma_color,sigma_guide_color, sigma_space,borderType);
			}
		}
	}
	/*
	void dualBilateralWeightMapSGB(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color,double sigma_space,int method, int borderType)
	{
	if(dst.empty())dst.create(src.size(),CV_32F);
	if(method==BILATERAL_NORMAL)
	{
	if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
	{
	dualBilateralWeightMapSGB_8u(src,guide,dst,kernelSize,sigma_color,sigma_guide_color, sigma_space,borderType);
	}
	else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
	{
	//dualBilateralWeightMap_32f(src,guide,dst,kernelSize,sigma_color,sigma_guide_color, sigma_space,borderType);
	}
	}
	else if(method==BILATERAL_SEPARABLE)
	{
	if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
	{
	//dualBilateralWeightMapSP_8u(src,guide,dst,kernelSize,sigma_color,sigma_guide_color, sigma_space,borderType);
	}
	else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
	{
	//dualBilateralWeightMapSP_32f(src,guide,dst,kernelSize,sigma_color,sigma_guide_color, sigma_space,borderType);
	}
	}
	}
	*/

	void dualBilateralFilterSP_32f(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		dualBilateralFilter_32f(src, guide, dst, Size(kernelSize.width, 1), sigma_color, sigma_guide_color, sigma_space, borderType, true);
		dualBilateralFilter_32f(dst, guide, dst, Size(1, kernelSize.height), sigma_color, sigma_guide_color, sigma_space, borderType, true);
	}

	void dualBilateralFilterSP_8u(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int borderType)
	{
		dualBilateralFilter_8u(src, guide, dst, Size(kernelSize.width, 1), sigma_color, sigma_guide_color, sigma_space, borderType, true);
		dualBilateralFilter_8u(dst, guide, dst, Size(1, kernelSize.height), sigma_color, sigma_guide_color, sigma_space, borderType, true);
	}

	void dualBilateralFilter(InputArray src_, InputArray guide_, OutputArray dest, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int method, int borderType)
	{
		if (dest.empty())dest.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();
		Mat dst = dest.getMat();

		if (method == FILTER_CIRCLE || method == FILTER_DEFAULT)
		{
			if (src.depth() == CV_8U)
			{
				dualBilateralFilter_8u(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType, false);
			}
			else if (src.depth() == CV_32F)
			{
				dualBilateralFilter_32f(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType, false);
			}
		}
		else if (method == FILTER_RECTANGLE)
		{
			if (src.depth() == CV_8U)
			{
				dualBilateralFilter_8u(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType, true);
			}
			else if (src.depth() == CV_32F)
			{
				dualBilateralFilter_32f(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType, true);
			}
		}
		else if (method == FILTER_SEPARABLE)
		{
			if (src.depth() == CV_8U)
			{
				dualBilateralFilterSP_8u(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
			else if (src.depth() == CV_32F)
			{
				dualBilateralFilterSP_32f(src, guide, dst, kernelSize, sigma_color, sigma_guide_color, sigma_space, borderType);
			}
		}
		else if (method == FILTER_SLOWEST)
		{
			dualBilateralFilterBase(src, guide, dst, kernelSize.width, sigma_color, sigma_guide_color, sigma_space, FILTER_CIRCLE, borderType);
		}
	}

	void dualBilateralFilter(InputArray src, InputArray guide, OutputArray dest, int D, double sigma_color, double sigma_guide_color, double sigma_space, int method, int borderType)
	{
		dualBilateralFilter(src, guide, dest, Size(D, D), sigma_color, sigma_guide_color, sigma_space, method, borderType);
	}

	void separableDualBilateralFilter(InputArray src_, InputArray guide_, OutputArray dest, Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1, double alpha2, int sp_kernel_type, int borderType)
	{
		if (dest.empty())dest.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat guide = guide_.getMat();
		Mat dst = dest.getMat();

		if (sp_kernel_type == DUAL_KERNEL_HV)
		{
			dualBilateralFilter(src, guide, dst, Size(ksize.width, 1), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);

			//jointDualBilateralFilter(src, src, guide, dst, ksize, sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_DEFAULT);
			jointDualBilateralFilter(dst, src, guide, dst, Size(ksize.width, ksize.height), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);
		}
		else if (sp_kernel_type == DUAL_KERNEL_VH)
		{
			dualBilateralFilter(src, guide, dst, Size(1, ksize.height), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);
			jointDualBilateralFilter(dst, src, guide, dst, Size(ksize.width, 1), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);
		}
		else if (sp_kernel_type == DUAL_KERNEL_HVVH)
		{
			dualBilateralFilter(src, guide, dst, Size(ksize.width, 1), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);
			jointDualBilateralFilter(dst, src, guide, dst, Size(1, ksize.height), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);
			Mat dst2;
			dualBilateralFilter(src, guide, dst2, Size(1, ksize.height), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);
			jointDualBilateralFilter(dst2, src, guide, dst2, Size(ksize.width, 1), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);

			alphaBlend(dst, dst2, 0.5, dst);
		}
		/*else if (method==DUAL_KERNEL_CROSS)
		{
		bilateralFilter_direction_8u(src, dst, kernelSize, sigma_color, sigma_space, borderType, 1, true);
		jointBilateralFilter_direction_8u(dst, src, dst, kernelSize, sigma_color*alpha, sigma_space, borderType, -1, true);
		}
		else if (method==DUAL_KERNEL_CROSSCROSS)
		{
		bilateralFilter_direction_8u(src, dst, kernelSize, sigma_color, sigma_space, borderType, 1, true);
		jointBilateralFilter_direction_8u(dst, src, dst, kernelSize, sigma_color*alpha, sigma_space, borderType, -1,true);

		Mat dst2(src.size(),src.type());
		bilateralFilter_direction_8u(src, dst2, kernelSize, sigma_color, sigma_space, borderType, -1, true);
		jointBilateralFilter_direction_8u(dst2, src, dst2, kernelSize, sigma_color*alpha, sigma_space, borderType, 1, true);

		alphaBlend(dst,dst2,0.5,dst);
		}*/
	}

	void separableDualBilateralFilter(InputArray src, InputArray guide, OutputArray dest, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1, double alpha2, int sp_kernel_type, int borderType)
	{
		separableDualBilateralFilter(src, guide, dest, Size(D, D), sigma_color, sigma_guide_color, sigma_space, alpha1, alpha2, sp_kernel_type, borderType);
	}
}