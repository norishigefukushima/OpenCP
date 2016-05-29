#include "opencp.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//baseline 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void jointDualBilateralFilterBase_8u(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, Size ksize,
		double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int borderType, bool isRectangle)
	{
		if (ksize.area() == 1){ src.copyTo(dst); return; }
		Size size = src.size();
		if (dst.empty())dst = Mat::zeros(src.size(), src.type());

		double gauss_guide_color_coeff1 = -0.5 / (sigma_guide_color1*sigma_guide_color1);
		double gauss_guide_color_coeff2 = -0.5 / (sigma_guide_color2*sigma_guide_color2);//trilateral(1)
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		const int cn = src.channels();
		const int cnj1 = guide1.channels();
		const int cnj2 = guide2.channels();

		int radiusH = ksize.width / 2;
		int radiusV = ksize.height / 2;

		Mat jim1, jim2;
		Mat sim;
		copyMakeBorder(guide1, jim1, radiusV, radiusV, radiusH, radiusH, borderType);
		copyMakeBorder(guide2, jim2, radiusV, radiusV, radiusH, radiusH, borderType);
		copyMakeBorder(src, sim, radiusV, radiusV, radiusH, radiusH, borderType);

		vector<float> _color_guide_weight1(cnj1 * 256);//trilateral(3)
		vector<float> _color_guide_weight2(cnj2 * 256);//trilateral(3)
		vector<float> _space_weight(ksize.area());
		float* color_guide_weight1 = &_color_guide_weight1[0];//trilateral(4)
		float* color_guide_weight2 = &_color_guide_weight2[0];//trilateral(4)
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs_jnt1(ksize.area());
		vector<int> _space_ofs_jnt2(ksize.area());
		vector<int> _space_ofs_src(ksize.area());
		int* space_ofs_jnt1 = &_space_ofs_jnt1[0];
		int* space_ofs_jnt2 = &_space_ofs_jnt2[0];
		int* space_ofs_src = &_space_ofs_src[0];

		// initialize color-related bilateral filter coefficients

		for (int i = 0; i < 256 * cnj1; i++)//trilateral(6)
			color_guide_weight1[i] = (float)std::exp(i*i*gauss_guide_color_coeff1);
		
		for (int i = 0; i < 256 * cnj2; i++)//trilateral(6)
			color_guide_weight2[i] = (float)std::exp(i*i*gauss_guide_color_coeff2);

		int maxk = 0;
		// initialize space-related bilateral filter coefficients

		int radius = max(radiusH, radiusV);
		for (int i = -radiusV; i <= radiusV; i++)
		{
			for (int j = -radiusH; j <= radiusH; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radius && !isRectangle) continue;

				space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
				space_ofs_jnt1[maxk] = (int)(i*jim1.cols*cnj1 + j*cnj1);
				space_ofs_jnt2[maxk] = (int)(i*jim2.cols*cnj2 + j*cnj2);
				space_ofs_src[maxk++] = (int)(i*sim.cols*cn + j*cn);
			}
		}

		//#pragma omp parallel for
		for (int i = 0; i < size.height; i++)
		{
			const uchar* jptr1 = jim1.ptr<uchar>(i + radiusV) + radiusH*cnj1;
			const uchar* jptr2 = jim2.ptr<uchar>(i + radiusV) + radiusH*cnj2;
			const uchar* sptr = sim.ptr<uchar>(i + radiusV) + radiusH*cn;
			uchar* dptr = dst.ptr<uchar>(i);

			if (cn == 1 && cnj1 == 1 && cnj2 == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					float sum = 0.f, wsum = 0.f;
					const uchar val1_0 = jptr1[j];
					const uchar val2_0 = jptr2[j];

					for (int k = 0; k < maxk; k++)
					{
						uchar val = sptr[j + space_ofs_src[k]];
						uchar val1 = jptr1[j + space_ofs_src[k]];
						uchar val2 = jptr2[j + space_ofs_src[k]];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(val1_0 - val1))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];//trilateral(8)

						sum += val*w;
						wsum += w;
					}
					dptr[j] = (uchar)cvRound(sum / wsum);
				}
			}
			else if (cn == 1 && cnj1 == 1 && cnj2 == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum = 0.f, wsum = 0.f;
					const uchar jv1_0 = jptr1[l];
					const uchar b2_0 = jptr2[j + 0];
					const uchar g2_0 = jptr2[j + 1];
					const uchar r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const uchar val = sptr[l + space_ofs_src[k]];

						const uchar jv1 = jptr1[l + space_ofs_jnt1[k]];
						const uchar  b2 = jptr2[j + space_ofs_jnt2[k] + 0];
						const uchar  g2 = jptr2[j + space_ofs_jnt2[k] + 1];
						const uchar  r2 = jptr2[j + space_ofs_jnt2[k] + 2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(abs(jv1_0 - jv1))]
							* color_guide_weight2[cvRound(abs(b2 - b2_0) + abs(g2 - g2_0) + abs(r2 - r2_0))];

						sum += val*w;
						wsum += w;
					}
					dptr[l] = (uchar)cvRound(sum / wsum);
				}
			}
			else if (cn == 1 && cnj1 == 3 && cnj2 == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum = 0.f, wsum = 0.f;
					const uchar val2_0 = jptr2[l];
					uchar b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const uchar val = sptr[l + space_ofs_src[k]];
						const uchar val2 = jptr2[l + space_ofs_src[k]];
						const uchar* jptr1_k = jptr1 + j + space_ofs_jnt1[k];
						const uchar b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];

						sum += val*w;
						wsum += w;
					}
					dptr[l] = (uchar)cvRound(sum / wsum);
				}
			}
			else if (cn == 1 && cnj1 == 3 && cnj2 == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum_b = 0.f, wsum = 0.f;
					uchar b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];
					uchar b2_0 = jptr2[j], g2_0 = jptr2[j + 1], r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const uchar* jptr1_k = jptr1 + j + space_ofs_jnt1[k];
						const uchar* jptr2_k = jptr2 + j + space_ofs_jnt1[k];
						const uchar b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];
						const uchar b2 = jptr2_k[0], g2 = jptr2_k[1], r2 = jptr2_k[2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(b2 - b2_0) + std::abs(g2 - g2_0) + std::abs(r2 - r2_0))];//trilateral(11)

						const uchar val = *(sptr + l + space_ofs_src[k]);
						sum_b += val*w;
						wsum += w;
					}
					dptr[l] = cvRound(sum_b / wsum);
				}
			}
			else if (cn == 3 && cnj1 == 1 && cnj2 == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;

					const uchar val1_0 = jptr1[l];
					const uchar val2_0 = jptr2[l];

					for (int k = 0; k < maxk; k++)
					{
						const uchar* sptr_k = sptr + j + space_ofs_src[k];
						uchar val1 = jptr1[l + space_ofs_jnt1[k]];
						uchar val2 = jptr2[l + space_ofs_jnt1[k]];

						const uchar b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(val1_0 - val1))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];//trilateral(8)
						sum_b += b*w; sum_g += g*w; sum_r += r*w;
						wsum += w;
					}
					dptr[j] = (uchar)cvRound(sum_b / wsum);
					dptr[j + 1] = (uchar)cvRound(sum_g / wsum);
					dptr[j + 2] = (uchar)cvRound(sum_r / wsum);
				}
			}
			else if (cn == 3 && cnj1 == 1 && cnj2 == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; l++, j += 3)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;

					const uchar val1_0 = jptr1[l];
					uchar b2_0 = jptr2[j], g2_0 = jptr2[j + 1], r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						uchar val1 = jptr1[l + space_ofs_jnt1[k]];
						const uchar* jptr2_k = jptr2 + j + space_ofs_src[k];
						const uchar* sptr_k = sptr + j + space_ofs_src[k];

						const uchar b2 = jptr2_k[0], g2 = jptr2_k[1], r2 = jptr2_k[2];
						const uchar bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(val1_0 - val1))]
							* color_guide_weight2[cvRound(std::abs(b2 - b2_0) + std::abs(g2 - g2_0) + std::abs(r2 - r2_0))];//trilateral(11)
						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (uchar)cvRound(sum_b / wsum);
					dptr[j + 1] = (uchar)cvRound(sum_g / wsum);
					dptr[j + 2] = (uchar)cvRound(sum_r / wsum);
				}
			}
			else if (cn == 3 && cnj1 == 3 && cnj2 == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; l++, j += 3)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;

					const uchar val2_0 = jptr2[l];
					float b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						uchar val2 = jptr2[l + space_ofs_jnt2[k]];
						const uchar* jptr1_k = jptr1 + j + space_ofs_src[k];
						const uchar* sptr_k = sptr + j + space_ofs_src[k];

						const uchar b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];
						const uchar bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];
						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (uchar)cvRound(sum_b / wsum);
					dptr[j + 1] = (uchar)cvRound(sum_g / wsum);
					dptr[j + 2] = (uchar)cvRound(sum_r / wsum);
				}
			}
			else if (cn == 3 && cnj1 == 3 && cnj2 == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;

					uchar b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];
					uchar b2_0 = jptr2[j], g2_0 = jptr2[j + 1], r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const uchar* jptr1_k = jptr1 + j + space_ofs_src[k];
						const uchar* jptr2_k = jptr2 + j + space_ofs_src[k];
						const uchar* sptr_k = sptr + j + space_ofs_src[k];

						const uchar b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];
						const uchar b2 = jptr2_k[0], g2 = jptr2_k[1], r2 = jptr2_k[2];
						const uchar bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(b2 - b2_0) + std::abs(g2 - g2_0) + std::abs(r2 - r2_0))];//trilateral(11)
						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (uchar)cvRound(sum_b / wsum);
					dptr[j + 1] = (uchar)cvRound(sum_g / wsum);
					dptr[j + 2] = (uchar)cvRound(sum_r / wsum);
				}
			}
		}
	}

	void jointDualBilateralFilterBase_32f(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, Size ksize,
		double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int borderType, bool isRectangle)
	{
		if (ksize.area() == 1){ src.copyTo(dst); return; }
		Size size = src.size();
		if (dst.empty())dst = Mat::zeros(src.size(), src.type());

		double gauss_guide_color_coeff1 = -0.5 / (sigma_guide_color1*sigma_guide_color1);
		double gauss_guide_color_coeff2 = -0.5 / (sigma_guide_color2*sigma_guide_color2);//trilateral(1)
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		const int cn = src.channels();
		const int cnj1 = guide1.channels();
		const int cnj2 = guide2.channels();

		int radiusH = ksize.width / 2;
		int radiusV = ksize.height / 2;

		Mat jim1, jim2;
		Mat sim;
		copyMakeBorder(guide1, jim1, radiusV, radiusV, radiusH, radiusH, borderType);
		copyMakeBorder(guide2, jim2, radiusV, radiusV, radiusH, radiusH, borderType);
		copyMakeBorder(src, sim, radiusV, radiusV, radiusH, radiusH, borderType);

		vector<float> _color_guide_weight1(cnj1 * 256);//trilateral(3)
		vector<float> _color_guide_weight2(cnj2 * 256);//trilateral(3)
		vector<float> _space_weight(ksize.area());
		float* color_guide_weight1 = &_color_guide_weight1[0];//trilateral(4)
		float* color_guide_weight2 = &_color_guide_weight2[0];//trilateral(4)
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs_jnt1(ksize.area());
		vector<int> _space_ofs_jnt2(ksize.area());
		vector<int> _space_ofs_src(ksize.area());
		int* space_ofs_jnt1 = &_space_ofs_jnt1[0];
		int* space_ofs_jnt2 = &_space_ofs_jnt2[0];
		int* space_ofs_src = &_space_ofs_src[0];

		// initialize color-related bilateral filter coefficients

		for (int i = 0; i < 256 * cnj1; i++)//trilateral(6)
			color_guide_weight1[i] = (float)std::exp(i*i*gauss_guide_color_coeff1);

		for (int i = 0; i < 256 * cnj2; i++)//trilateral(6)
			color_guide_weight2[i] = (float)std::exp(i*i*gauss_guide_color_coeff2);

		int maxk = 0;
		// initialize space-related bilateral filter coefficients

		int radius = max(radiusH, radiusV);
		for (int i = -radiusV; i <= radiusV; i++)
		{
			for (int j = -radiusH; j <= radiusH; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > radius && !isRectangle) continue;

				space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
				space_ofs_jnt1[maxk] = (int)(i*jim1.cols*cnj1 + j*cnj1);
				space_ofs_jnt2[maxk] = (int)(i*jim2.cols*cnj2 + j*cnj2);
				space_ofs_src[maxk++] = (int)(i*sim.cols*cn + j*cn);
			}
		}

		//#pragma omp parallel for
		for (int i = 0; i < size.height; i++)
		{
			const float* jptr1 = jim1.ptr<float>(i + radiusV) + radiusH*cnj1;
			const float* jptr2 = jim2.ptr<float>(i + radiusV) + radiusH*cnj2;
			const float* sptr = sim.ptr<float>(i + radiusV) + radiusH*cn;
			float* dptr = dst.ptr<float>(i);

			if (cn == 1 && cnj1 == 1 && cnj2 == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					float sum = 0.f, wsum = 0.f;
					const float val1_0 = jptr1[j];
					const float val2_0 = jptr2[j];

					for (int k = 0; k < maxk; k++)
					{
						float val = sptr[j + space_ofs_src[k]];
						float val1 = jptr1[j + space_ofs_src[k]];
						float val2 = jptr2[j + space_ofs_src[k]];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(val1_0 - val1))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];//trilateral(8)

						sum += val*w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
			else if (cn == 1 && cnj1 == 1 && cnj2 == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum = 0.f, wsum = 0.f;
					const float jv1_0 = jptr1[l];
					const float b2_0 = jptr2[j + 0];
					const float g2_0 = jptr2[j + 1];
					const float r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const float val = sptr[l + space_ofs_src[k]];

						const float jv1 = jptr1[l + space_ofs_jnt1[k]];
						const float  b2 = jptr2[j + space_ofs_jnt2[k] + 0];
						const float  g2 = jptr2[j + space_ofs_jnt2[k] + 1];
						const float  r2 = jptr2[j + space_ofs_jnt2[k] + 2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(abs(jv1_0 - jv1))]
							* color_guide_weight2[cvRound(abs(b2 - b2_0) + abs(g2 - g2_0) + abs(r2 - r2_0))];

						sum += val*w;
						wsum += w;
					}
					dptr[l] = sum / wsum;
				}
			}
			else if (cn == 1 && cnj1 == 3 && cnj2 == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum = 0.f, wsum = 0.f;
					const float val2_0 = jptr2[l];
					float b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const float val = sptr[l + space_ofs_src[k]];
						const float val2 = jptr2[l + space_ofs_src[k]];
						const float* jptr1_k = jptr1 + j + space_ofs_jnt1[k];
						const float b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];

						sum += val*w;
						wsum += w;
					}
					dptr[l] = sum / wsum;
				}
			}
			else if (cn == 1 && cnj1 == 3 && cnj2 == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum_b = 0, wsum = 0;
					float b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];
					float b2_0 = jptr2[j], g2_0 = jptr2[j + 1], r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const float* jptr1_k = jptr1 + j + space_ofs_jnt1[k];
						const float* jptr2_k = jptr2 + j + space_ofs_jnt1[k];
						const float b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];
						const float b2 = jptr2_k[0], g2 = jptr2_k[1], r2 = jptr2_k[2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(b2 - b2_0) + std::abs(g2 - g2_0) + std::abs(r2 - r2_0))];//trilateral(11)

						const float val = *(sptr + l + space_ofs_src[k]);
						sum_b += val*w;
						wsum += w;
					}
					wsum = 1.f / wsum;
					dptr[l] = sum_b*wsum;
				}
			}
			else if (cn == 3 && cnj1 == 1 && cnj2 == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;

					const float val1_0 = jptr1[l];
					const float val2_0 = jptr2[l];

					for (int k = 0; k < maxk; k++)
					{
						const float* sptr_k = sptr + j + space_ofs_src[k];
						float val1 = jptr1[l + space_ofs_jnt1[k]];
						float val2 = jptr2[l + space_ofs_jnt1[k]];

						const float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(val1_0 - val1))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];//trilateral(8)
						sum_b += b*w; sum_g += g*w; sum_r += r*w;
						wsum += w;
					}
					dptr[j] = sum_b / wsum;
					dptr[j + 1] = sum_g / wsum;
					dptr[j + 2] = sum_r / wsum;
				}
			}
			else if (cn == 3 && cnj1 == 1 && cnj2 == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; l++, j += 3)
				{
					float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f, wsum = 0.0f;

					const float val1_0 = jptr1[l];
					float b2_0 = jptr2[j], g2_0 = jptr2[j + 1], r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						float val1 = jptr1[l + space_ofs_jnt1[k]];
						const float* jptr2_k = jptr2 + j + space_ofs_src[k];
						const float* sptr_k = sptr + j + space_ofs_src[k];

						const float b2 = jptr2_k[0], g2 = jptr2_k[1], r2 = jptr2_k[2];
						const float bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(val1_0 - val1))]
							* color_guide_weight2[cvRound(std::abs(b2 - b2_0) + std::abs(g2 - g2_0) + std::abs(r2 - r2_0))];//trilateral(11)
						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (sum_b / wsum);
					dptr[j + 1] = (sum_g / wsum);
					dptr[j + 2] = (sum_r / wsum);
				}
			}
			else if (cn == 3 && cnj1 == 3 && cnj2 == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; l++, j += 3)
				{
					float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f, wsum = 0.0f;

					const float val2_0 = jptr2[l];
					float b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						float val2 = jptr2[l + space_ofs_jnt2[k]];
						const float* jptr1_k = jptr1 + j + space_ofs_src[k];
						const float* sptr_k = sptr + j + space_ofs_src[k];

						const float b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];
						const float bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(val2_0 - val2))];
						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (sum_b / wsum);
					dptr[j + 1] = (sum_g / wsum);
					dptr[j + 2] = (sum_r / wsum);
				}
			}
			else if (cn == 3 && cnj1 == 3 && cnj2 == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f, wsum = 0.0f;

					float b1_0 = jptr1[j], g1_0 = jptr1[j + 1], r1_0 = jptr1[j + 2];
					float b2_0 = jptr2[j], g2_0 = jptr2[j + 1], r2_0 = jptr2[j + 2];

					for (int k = 0; k < maxk; k++)
					{
						const float* jptr1_k = jptr1 + j + space_ofs_src[k];
						const float* jptr2_k = jptr2 + j + space_ofs_src[k];
						const float* sptr_k = sptr + j + space_ofs_src[k];

						const float b1 = jptr1_k[0], g1 = jptr1_k[1], r1 = jptr1_k[2];
						const float b2 = jptr2_k[0], g2 = jptr2_k[1], r2 = jptr2_k[2];
						const float bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];//trilateral(10)

						float w = space_weight[k]
							* color_guide_weight1[cvRound(std::abs(b1 - b1_0) + std::abs(g1 - g1_0) + std::abs(r1 - r1_0))]
							* color_guide_weight2[cvRound(std::abs(b2 - b2_0) + std::abs(g2 - g2_0) + std::abs(r2 - r2_0))];//trilateral(11)
						sum_b += bs*w; sum_g += gs*w; sum_r += rs*w;
						wsum += w;
					}
					dptr[j] = (sum_b / wsum);
					dptr[j + 1] = (sum_g / wsum);
					dptr[j + 2] = (sum_r / wsum);
				}
			}
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class JointDualBilateralFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
	{
	public:
		JointDualBilateralFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _guide1, const Mat& _guide2, int _radiusH, int _radiusV, int _maxk,
			int* _space_ofs, int* _space_guide1_ofs, int* _space_guide2_ofs, float *_space_weight, float *_guide1_color_weight, float *_guide2_color_weight) :
			temp(&_temp), dest(&_dest), guide1(&_guide1), guide2(&_guide2), radiusH(_radiusH), radiusV(_radiusV),
			maxk(_maxk), space_ofs(_space_ofs), space_guide1_ofs(_space_guide1_ofs), space_guide2_ofs(_space_guide2_ofs), space_weight(_space_weight), guide1_color_weight(_guide1_color_weight), guide2_color_weight(_guide2_color_weight)
		{
		}

		virtual void operator() (const Range& range) const
		{

			int i, j, k;
			const int cn = (temp->rows - 2 * radiusV) / dest->rows;
			const int cng1 = (guide1->rows - 2 * radiusV) / dest->rows;
			const int cng2 = (guide2->rows - 2 * radiusV) / dest->rows;
			Size size = dest->size();
#if CV_SSE4_1
			bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
			if (cn == 1 && cng1 == 1 && cng2 == 1)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				uchar* g1ptr = (uchar*)guide1->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* g2ptr = (uchar*)guide2->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				const int sstep = temp->cols;
				const int g1step = guide1->cols;
				const int g2step = guide2->cols;

				const int dstep = dest->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep, g1ptr += g1step, g2ptr += g2step)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];
							int* g1ofs = &space_guide1_ofs[0];
							int* g2ofs = &space_guide2_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* g1ptrj = g1ptr + j;
							const uchar* g2ptrj = g2ptr + j;

							const __m128i g1val = _mm_load_si128((__m128i*)(g1ptrj));
							const __m128i g2val = _mm_load_si128((__m128i*)(g2ptrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, g1ofs++, g2ofs++, spw++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(g1ptrj + *g1ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(g1val, sref), _mm_subs_epu8(sref, g1val)));

								sref = _mm_loadu_si128((__m128i*)(g2ptrj + *g2ofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(g2val, sref), _mm_subs_epu8(sref, g2val)));


								sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								__m128i m1 = _mm_unpacklo_epi8(sref, zero);
								__m128i m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);

								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[3]], guide1_color_weight[buf[2]], guide1_color_weight[buf[1]], guide1_color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[3]], guide2_color_weight[gbuf[2]], guide2_color_weight[gbuf[1]], guide2_color_weight[gbuf[0]]));
								__m128 _valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[7]], guide1_color_weight[buf[6]], guide1_color_weight[buf[5]], guide1_color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[7]], guide2_color_weight[gbuf[6]], guide2_color_weight[gbuf[5]], guide2_color_weight[gbuf[4]]));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);

								m1 = _mm_unpackhi_epi8(sref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[11]], guide1_color_weight[buf[10]], guide1_color_weight[buf[9]], guide1_color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[11]], guide2_color_weight[gbuf[10]], guide2_color_weight[gbuf[9]], guide2_color_weight[gbuf[8]]));
								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								wval3 = _mm_add_ps(wval3, _w);
								tval3 = _mm_add_ps(tval3, _valF);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[15]], guide1_color_weight[buf[14]], guide1_color_weight[buf[13]], guide1_color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[15]], guide2_color_weight[gbuf[14]], guide2_color_weight[gbuf[13]], guide2_color_weight[gbuf[12]]));
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
						const uchar g1v0 = g1ptr[j];
						const uchar g2v0 = g2ptr[j];
						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int g1val = g1ptr[j + space_guide1_ofs[k]];
							int g2val = g2ptr[j + space_guide2_ofs[k]];
							int sval = sptr[j + space_ofs[k]];
							float w = space_weight[k]
								* guide1_color_weight[std::abs(g1val - g1v0)]
								* guide2_color_weight[std::abs(g2val - g2v0)];
							sum += sval*w;
							wsum += w;
						}
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 1 && cng1 == 1 && cng2 == 3)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = temp->cols;
				const int g1step = guide1->cols;
				const int g2step = 3 * guide2->cols;

				const int dstep = dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				uchar* g1ptr = (uchar*)guide1->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrr = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrg = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrb = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				for (i = range.start; i != range.end; i++, g1ptr += g1step, g2ptrr += g2step, g2ptrg += g2step, g2ptrb += g2step, sptr += sstep, dptr += dstep)
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
							int* g1ofs = &space_guide1_ofs[0];
							int* g2ofs = &space_guide2_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* g1ptrj = g1ptr + j;
							const uchar* g2ptrrj = g2ptrr + j;
							const uchar* g2ptrgj = g2ptrg + j;
							const uchar* g2ptrbj = g2ptrb + j;

							const __m128i g1val = _mm_load_si128((__m128i*)(g1ptrj));
							const __m128i g2bval = _mm_load_si128((__m128i*)(g2ptrbj));
							const __m128i g2gval = _mm_load_si128((__m128i*)(g2ptrgj));
							const __m128i g2rval = _mm_load_si128((__m128i*)(g2ptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();
							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();

							for (k = 0; k < maxk; k++, ofs++, g2ofs++, g1ofs++, spw++)
							{
								const __m128i g2bref = _mm_loadu_si128((__m128i*)(g2ptrbj + *g2ofs));
								const __m128i g2gref = _mm_loadu_si128((__m128i*)(g2ptrgj + *g2ofs));
								const __m128i g2rref = _mm_loadu_si128((__m128i*)(g2ptrrj + *g2ofs));

								m1 = _mm_add_epi8(_mm_subs_epu8(g2rval, g2rref), _mm_subs_epu8(g2rref, g2rval));
								m2 = _mm_unpackhi_epi8(m1, zero);
								m1 = _mm_unpacklo_epi8(m1, zero);

								n1 = _mm_add_epi8(_mm_subs_epu8(g2gval, g2gref), _mm_subs_epu8(g2gref, g2gval));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								n1 = _mm_add_epi8(_mm_subs_epu8(g2bval, g2bref), _mm_subs_epu8(g2bref, g2bval));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								_mm_store_si128((__m128i*)(gbuf + 8), m2);
								_mm_store_si128((__m128i*)gbuf, m1);

								__m128i g1ref = _mm_loadu_si128((__m128i*)(g1ptrj + *g1ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(g1val, g1ref), _mm_subs_epu8(g1ref, g1val)));

								const __m128i vref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
								m1 = _mm_unpacklo_epi8(vref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[3]], guide1_color_weight[buf[2]], guide1_color_weight[buf[1]], guide1_color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[3]], guide2_color_weight[gbuf[2]], guide2_color_weight[gbuf[1]], guide2_color_weight[gbuf[0]]));

								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								tval1 = _mm_add_ps(tval1, _valF);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[7]], guide1_color_weight[buf[6]], guide1_color_weight[buf[5]], guide1_color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[7]], guide2_color_weight[gbuf[6]], guide2_color_weight[gbuf[5]], guide2_color_weight[gbuf[4]]));
								_valF = _mm_cvtepi32_ps(m2);
								_valF = _mm_mul_ps(_w, _valF);
								tval2 = _mm_add_ps(tval2, _valF);
								wval2 = _mm_add_ps(wval2, _w);


								m1 = _mm_unpackhi_epi8(vref, zero);
								m2 = _mm_unpackhi_epi16(m1, zero);
								m1 = _mm_unpacklo_epi16(m1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[11]], guide1_color_weight[buf[10]], guide1_color_weight[buf[9]], guide1_color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[11]], guide2_color_weight[gbuf[10]], guide2_color_weight[gbuf[9]], guide2_color_weight[gbuf[8]]));
								_valF = _mm_cvtepi32_ps(m1);
								_valF = _mm_mul_ps(_w, _valF);
								tval3 = _mm_add_ps(tval3, _valF);
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[15]], guide1_color_weight[buf[14]], guide1_color_weight[buf[13]], guide1_color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[15]], guide2_color_weight[gbuf[14]], guide2_color_weight[gbuf[13]], guide2_color_weight[gbuf[12]]));
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
						const uchar* gptrj = g1ptr + j;
						const uchar* gptrrj = g2ptrr + j;
						const uchar* gptrgj = g2ptrg + j;
						const uchar* gptrbj = g2ptrb + j;

						int g1 = gptrj[0];
						int r0 = gptrrj[0];
						int g0 = gptrgj[0];
						int b0 = gptrbj[0];

						float sum = 0.0f;
						float wsum = 0.0f;
						for (k = 0; k < maxk; k++)
						{
							int r = gptrrj[space_guide2_ofs[k]], g = gptrgj[space_guide2_ofs[k]], b = gptrbj[space_guide2_ofs[k]];
							float w = space_weight[k] * guide1_color_weight[std::abs(gptrj[space_guide1_ofs[k]] - g1)] * guide2_color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							sum += sptrj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 1 && cng1 == 3 && cng2 == 3)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int  sstep = temp->cols;
				const int g1step = 3 * guide1->cols;
				const int g2step = 3 * guide2->cols;
				const int dstep = 3 * dest->cols;

				uchar* sptr = (uchar*)temp->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				uchar* g1ptrr = (uchar*)guide1->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* g1ptrg = (uchar*)guide1->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* g1ptrb = (uchar*)guide1->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrr = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrg = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrb = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				for (i = range.start; i != range.end; i++, g1ptrr += g1step, g1ptrg += g1step, g1ptrb += g1step, g2ptrr += g2step, g2ptrg += g2step, g2ptrb += g2step, sptr += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];
							int* g1ofs = &space_guide1_ofs[0];
							int* g2ofs = &space_guide2_ofs[0];

							float* spw = space_weight;

							const uchar* sptrj = sptr + j;
							const uchar* g1ptrrj = g1ptrr + j;
							const uchar* g1ptrgj = g1ptrg + j;
							const uchar* g1ptrbj = g1ptrb + j;
							const uchar* g2ptrrj = g2ptrr + j;
							const uchar* g2ptrgj = g2ptrg + j;
							const uchar* g2ptrbj = g2ptrb + j;

							const __m128i g1b = _mm_load_si128((__m128i*)(g1ptrbj));
							const __m128i g1g = _mm_load_si128((__m128i*)(g1ptrgj));
							const __m128i g1r = _mm_load_si128((__m128i*)(g1ptrrj));
							const __m128i g2b = _mm_load_si128((__m128i*)(g2ptrbj));
							const __m128i g2g = _mm_load_si128((__m128i*)(g2ptrgj));
							const __m128i g2r = _mm_load_si128((__m128i*)(g2ptrrj));

							__m128 wval1 = _mm_setzero_ps();
							__m128 wval2 = _mm_setzero_ps();
							__m128 wval3 = _mm_setzero_ps();
							__m128 wval4 = _mm_setzero_ps();

							__m128 tval1 = _mm_setzero_ps();
							__m128 tval2 = _mm_setzero_ps();
							__m128 tval3 = _mm_setzero_ps();
							__m128 tval4 = _mm_setzero_ps();

							const __m128i zero = _mm_setzero_si128();
							for (k = 0; k < maxk; k++, ofs++, g1ofs++, g2ofs++, spw++)
							{
								__m128i g2bref = _mm_loadu_si128((__m128i*)(g2ptrbj + *g2ofs));
								__m128i g2gref = _mm_loadu_si128((__m128i*)(g2ptrgj + *g2ofs));
								__m128i g2rref = _mm_loadu_si128((__m128i*)(g2ptrrj + *g2ofs));

								__m128i r1 = _mm_add_epi8(_mm_subs_epu8(g2r, g2rref), _mm_subs_epu8(g2rref, g2r));
								__m128i r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								__m128i g1 = _mm_add_epi8(_mm_subs_epu8(g2g, g2gref), _mm_subs_epu8(g2gref, g2g));
								__m128i g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								__m128i b1 = _mm_add_epi8(_mm_subs_epu8(g2b, g2bref), _mm_subs_epu8(g2bref, g2b));
								__m128i b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(gbuf + 8), r2);
								_mm_store_si128((__m128i*)gbuf, r1);

								__m128i g1bref = _mm_loadu_si128((__m128i*)(g1ptrbj + *g1ofs));
								__m128i g1gref = _mm_loadu_si128((__m128i*)(g1ptrgj + *g1ofs));
								__m128i g1rref = _mm_loadu_si128((__m128i*)(g1ptrrj + *g1ofs));

								r1 = _mm_add_epi8(_mm_subs_epu8(g1r, g1rref), _mm_subs_epu8(g1rref, g1r));
								r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								g1 = _mm_add_epi8(_mm_subs_epu8(g1g, g1gref), _mm_subs_epu8(g1gref, g1g));
								g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								b1 = _mm_add_epi8(_mm_subs_epu8(g1b, g1bref), _mm_subs_epu8(g1bref, g1b));
								b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(buf + 8), r2);
								_mm_store_si128((__m128i*)buf, r1);

								__m128i vref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));

								r1 = _mm_unpacklo_epi8(vref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								const __m128 _sw = _mm_set1_ps(*spw);
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[3]], guide1_color_weight[buf[2]], guide1_color_weight[buf[1]], guide1_color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[3]], guide2_color_weight[gbuf[2]], guide2_color_weight[gbuf[1]], guide2_color_weight[gbuf[0]]));

								__m128 _valr = _mm_cvtepi32_ps(r1);

								_valr = _mm_mul_ps(_w, _valr);
								tval1 = _mm_add_ps(tval1, _valr);
								wval1 = _mm_add_ps(wval1, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[7]], guide1_color_weight[buf[6]], guide1_color_weight[buf[5]], guide1_color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[7]], guide2_color_weight[gbuf[6]], guide2_color_weight[gbuf[5]], guide2_color_weight[gbuf[4]]));

								_valr = _mm_cvtepi32_ps(r2);
								_valr = _mm_mul_ps(_w, _valr);

								tval2 = _mm_add_ps(tval2, _valr);
								wval2 = _mm_add_ps(wval2, _w);

								r1 = _mm_unpackhi_epi8(vref, zero);
								r2 = _mm_unpackhi_epi16(r1, zero);
								r1 = _mm_unpacklo_epi16(r1, zero);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[11]], guide1_color_weight[buf[10]], guide1_color_weight[buf[9]], guide1_color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[11]], guide2_color_weight[gbuf[10]], guide2_color_weight[gbuf[9]], guide2_color_weight[gbuf[8]]));

								_valr = _mm_cvtepi32_ps(r1);
								_valr = _mm_mul_ps(_w, _valr);
								tval3 = _mm_add_ps(tval3, _valr);
								wval3 = _mm_add_ps(wval3, _w);

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[15]], guide1_color_weight[buf[14]], guide1_color_weight[buf[13]], guide1_color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[15]], guide2_color_weight[gbuf[14]], guide2_color_weight[gbuf[13]], guide2_color_weight[gbuf[12]]));

								_valr = _mm_cvtepi32_ps(r2);
								_valr = _mm_mul_ps(_w, _valr);

								tval4 = _mm_add_ps(tval4, _valr);
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
						const uchar* g1ptrrj = g1ptrr + j;
						const uchar* g1ptrgj = g1ptrg + j;
						const uchar* g1ptrbj = g1ptrb + j;
						const uchar* g2ptrrj = g2ptrr + j;
						const uchar* g2ptrgj = g2ptrg + j;
						const uchar* g2ptrbj = g2ptrb + j;

						int g1r0 = g1ptrrj[0];
						int g1g0 = g1ptrgj[0];
						int g1b0 = g1ptrbj[0];
						int g2r0 = g2ptrrj[0];
						int g2g0 = g2ptrgj[0];
						int g2b0 = g2ptrbj[0];

						float sum = 0.f;
						float wsum = 0.f;
						for (k = 0; k < maxk; k++)
						{
							int g1r = g1ptrrj[space_guide1_ofs[k]];
							int g1g = g1ptrgj[space_guide1_ofs[k]];
							int g1b = g1ptrbj[space_guide1_ofs[k]];
							int g2r = g2ptrrj[space_guide2_ofs[k]];
							int g2g = g2ptrgj[space_guide2_ofs[k]];
							int g2b = g2ptrbj[space_guide2_ofs[k]];
							float w = space_weight[k];
							w *= guide1_color_weight[abs(g1b - g1b0) + abs(g1g - g1g0) + abs(g1r - g1r0)];
							w *= guide2_color_weight[abs(g2b - g2b0) + abs(g2g - g2g0) + abs(g2r - g2r0)];

							sum += sptrj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[j] = (uchar)cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 3 && cng1 == 1 && cng2 == 1)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				uchar CV_DECL_ALIGNED(16) gbuf[16];

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* g1ptr = (uchar*)guide1->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* g2ptr = (uchar*)guide2->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				const int sstep = 3 * temp->cols;
				const int dstep = 3 * dest->cols;

				const int g1step = guide1->cols;
				const int g2step = guide2->cols;

				for (i = range.start; i != range.end; i++, dptr += dstep, sptrr += sstep, sptrg += sstep, sptrb += sstep, g1ptr += g1step, g2ptr += g2step)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];
							int* g1ofs = &space_guide1_ofs[0];
							int* g2ofs = &space_guide2_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* g1ptrj = g1ptr + j;
							const uchar* g2ptrj = g2ptr + j;

							const __m128i g1val = _mm_load_si128((__m128i*)(g1ptrj));
							const __m128i g2val = _mm_load_si128((__m128i*)(g2ptrj));

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

							for (k = 0; k < maxk; k++, ofs++, g1ofs++, g2ofs++, spw++)
							{
								__m128i sref = _mm_loadu_si128((__m128i*)(g1ptrj + *g1ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(g1val, sref), _mm_subs_epu8(sref, g1val)));

								sref = _mm_loadu_si128((__m128i*)(g2ptrj + *g2ofs));
								_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(g2val, sref), _mm_subs_epu8(sref, g2val)));

								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

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

								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[3]], guide1_color_weight[buf[2]], guide1_color_weight[buf[1]], guide1_color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[3]], guide2_color_weight[gbuf[2]], guide2_color_weight[gbuf[1]], guide2_color_weight[gbuf[0]]));

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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[7]], guide1_color_weight[buf[6]], guide1_color_weight[buf[5]], guide1_color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[7]], guide2_color_weight[gbuf[6]], guide2_color_weight[gbuf[5]], guide2_color_weight[gbuf[4]]));

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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[11]], guide1_color_weight[buf[10]], guide1_color_weight[buf[9]], guide1_color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[11]], guide2_color_weight[gbuf[10]], guide2_color_weight[gbuf[9]], guide2_color_weight[gbuf[8]]));
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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[15]], guide1_color_weight[buf[14]], guide1_color_weight[buf[13]], guide1_color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[15]], guide2_color_weight[gbuf[14]], guide2_color_weight[gbuf[13]], guide2_color_weight[gbuf[12]]));
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
						const uchar sr0 = sptrr[j];
						const uchar sg0 = sptrg[j];
						const uchar sb0 = sptrb[j];

						const uchar g1v0 = g1ptr[j];
						const uchar g2v0 = g2ptr[j];
						float sum_r = 0.f;
						float sum_g = 0.f;
						float sum_b = 0.f;
						float wsum = 0.f;
						for (k = 0; k < maxk; k++)
						{
							int g1val = g1ptr[j + space_guide1_ofs[k]];
							int g2val = g2ptr[j + space_guide2_ofs[k]];

							float w = space_weight[k]
								* guide1_color_weight[std::abs(g1val - g1v0)]
								* guide2_color_weight[std::abs(g2val - g2v0)];
							sum_r += sptrr[j + space_ofs[k]] * w;
							sum_g += sptrg[j + space_ofs[k]] * w;
							sum_b += sptrb[j + space_ofs[k]] * w;
							wsum += w;
						}
						dptr[3 * j] = (uchar)cvRound(sum_r / wsum);
						dptr[3 * j + 1] = (uchar)cvRound(sum_g / wsum);
						dptr[3 * j + 2] = (uchar)cvRound(sum_b / wsum);
					}
				}
			}
			else if (cn == 3 && cng1 == 1 && cng2 == 3)
			{
				uchar CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int sstep = 3 * temp->cols;
				const int dstep = 3 * dest->cols;

				const int g1step = guide1->cols;
				const int g2step = 3 * guide2->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* dptr = dest->ptr(range.start);

				uchar* g1ptr = (uchar*)guide1->ptr(range.start + radiusV) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrr = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrg = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrb = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				for (i = range.start; i != range.end; i++, g1ptr += g1step, g2ptrr += g2step, g2ptrg += g2step, g2ptrb += g2step, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							__m128i m1, m2, n1, n2;

							int* ofs = &space_ofs[0];
							int* g1ofs = &space_guide1_ofs[0];
							int* g2ofs = &space_guide2_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* g1ptrj = g1ptr + j;
							const uchar* g2ptrrj = g2ptrr + j;
							const uchar* g2ptrgj = g2ptrg + j;
							const uchar* g2ptrbj = g2ptrb + j;

							const __m128i g1val = _mm_load_si128((__m128i*)(g1ptrj));
							const __m128i g2bval = _mm_load_si128((__m128i*)(g2ptrbj));
							const __m128i g2gval = _mm_load_si128((__m128i*)(g2ptrgj));
							const __m128i g2rval = _mm_load_si128((__m128i*)(g2ptrrj));

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

							for (k = 0; k < maxk; k++, ofs++, g2ofs++, g1ofs++, spw++)
							{
								const __m128i g2bref = _mm_loadu_si128((__m128i*)(g2ptrbj + *g2ofs));
								const __m128i g2gref = _mm_loadu_si128((__m128i*)(g2ptrgj + *g2ofs));
								const __m128i g2rref = _mm_loadu_si128((__m128i*)(g2ptrrj + *g2ofs));

								m1 = _mm_add_epi8(_mm_subs_epu8(g2rval, g2rref), _mm_subs_epu8(g2rref, g2rval));
								m2 = _mm_unpackhi_epi8(m1, zero);
								m1 = _mm_unpacklo_epi8(m1, zero);

								n1 = _mm_add_epi8(_mm_subs_epu8(g2gval, g2gref), _mm_subs_epu8(g2gref, g2gval));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								n1 = _mm_add_epi8(_mm_subs_epu8(g2bval, g2bref), _mm_subs_epu8(g2bref, g2bval));
								n2 = _mm_unpackhi_epi8(n1, zero);
								n1 = _mm_unpacklo_epi8(n1, zero);

								m1 = _mm_add_epi16(m1, n1);
								m2 = _mm_add_epi16(m2, n2);

								_mm_store_si128((__m128i*)(gbuf + 8), m2);
								_mm_store_si128((__m128i*)gbuf, m1);

								__m128i g1ref = _mm_loadu_si128((__m128i*)(g1ptrj + *g1ofs));
								_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(g1val, g1ref), _mm_subs_epu8(g1ref, g1val)));

								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

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
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[3]], guide1_color_weight[buf[2]], guide1_color_weight[buf[1]], guide1_color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[3]], guide2_color_weight[gbuf[2]], guide2_color_weight[gbuf[1]], guide2_color_weight[gbuf[0]]));

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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[7]], guide1_color_weight[buf[6]], guide1_color_weight[buf[5]], guide1_color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[7]], guide2_color_weight[gbuf[6]], guide2_color_weight[gbuf[5]], guide2_color_weight[gbuf[4]]));
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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[11]], guide1_color_weight[buf[10]], guide1_color_weight[buf[9]], guide1_color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[11]], guide2_color_weight[gbuf[10]], guide2_color_weight[gbuf[9]], guide2_color_weight[gbuf[8]]));
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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[15]], guide1_color_weight[buf[14]], guide1_color_weight[buf[13]], guide1_color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[15]], guide2_color_weight[gbuf[14]], guide2_color_weight[gbuf[13]], guide2_color_weight[gbuf[12]]));
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
						int sr0 = sptrrj[0];
						int sg0 = sptrgj[0];
						int sb0 = sptrbj[0];

						const uchar* gptrj = g1ptr + j;
						const uchar* gptrrj = g2ptrr + j;
						const uchar* gptrgj = g2ptrg + j;
						const uchar* gptrbj = g2ptrb + j;
						int g1 = gptrj[0];
						int r0 = gptrrj[0];
						int g0 = gptrgj[0];
						int b0 = gptrbj[0];

						float sum_r = 0.f;
						float sum_b = 0.f;
						float sum_g = 0.f;
						float wsum = 0.f;
						for (k = 0; k < maxk; k++)
						{
							int r = gptrrj[space_guide2_ofs[k]], g = gptrgj[space_guide2_ofs[k]], b = gptrbj[space_guide2_ofs[k]];
							float w = space_weight[k] * guide1_color_weight[std::abs(gptrj[space_guide1_ofs[k]] - g1)] * guide2_color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
							sum_r += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_b += sptrbj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[3 * j] = (uchar)cvRound(sum_r / wsum);
						dptr[3 * j + 1] = (uchar)cvRound(sum_g / wsum);
						dptr[3 * j + 2] = (uchar)cvRound(sum_b / wsum);
					}
				}
			}
			else if (cn == 3 && cng1 == 3 && cng2 == 3)
			{
				short CV_DECL_ALIGNED(16) buf[16];
				short CV_DECL_ALIGNED(16) gbuf[16];

				const int  sstep = 3 * temp->cols;
				const int g1step = 3 * guide1->cols;
				const int g2step = 3 * guide2->cols;
				const int dstep = 3 * dest->cols;

				uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* g1ptrr = (uchar*)guide1->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* g1ptrg = (uchar*)guide1->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* g1ptrb = (uchar*)guide1->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrr = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrg = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
				uchar* g2ptrb = (uchar*)guide2->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

				uchar* dptr = dest->ptr(range.start);

				for (i = range.start; i != range.end; i++, g1ptrr += g1step, g1ptrg += g1step, g1ptrb += g1step, g2ptrr += g2step, g2ptrg += g2step, g2ptrb += g2step, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
				{
					j = 0;
#if CV_SSE4_1
					if (haveSSE4)
					{
						for (; j < size.width; j += 16)
						{
							int* ofs = &space_ofs[0];
							int* g1ofs = &space_guide1_ofs[0];
							int* g2ofs = &space_guide2_ofs[0];

							float* spw = space_weight;

							const uchar* sptrrj = sptrr + j;
							const uchar* sptrgj = sptrg + j;
							const uchar* sptrbj = sptrb + j;
							const uchar* g1ptrrj = g1ptrr + j;
							const uchar* g1ptrgj = g1ptrg + j;
							const uchar* g1ptrbj = g1ptrb + j;
							const uchar* g2ptrrj = g2ptrr + j;
							const uchar* g2ptrgj = g2ptrg + j;
							const uchar* g2ptrbj = g2ptrb + j;

							const __m128i g1b = _mm_load_si128((__m128i*)(g1ptrbj));
							const __m128i g1g = _mm_load_si128((__m128i*)(g1ptrgj));
							const __m128i g1r = _mm_load_si128((__m128i*)(g1ptrrj));
							const __m128i g2b = _mm_load_si128((__m128i*)(g2ptrbj));
							const __m128i g2g = _mm_load_si128((__m128i*)(g2ptrgj));
							const __m128i g2r = _mm_load_si128((__m128i*)(g2ptrrj));

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
							for (k = 0; k < maxk; k++, ofs++, g1ofs++, g2ofs++, spw++)
							{
								__m128i g2bref = _mm_loadu_si128((__m128i*)(g2ptrbj + *g2ofs));
								__m128i g2gref = _mm_loadu_si128((__m128i*)(g2ptrgj + *g2ofs));
								__m128i g2rref = _mm_loadu_si128((__m128i*)(g2ptrrj + *g2ofs));

								__m128i r1 = _mm_add_epi8(_mm_subs_epu8(g2r, g2rref), _mm_subs_epu8(g2rref, g2r));
								__m128i r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								__m128i g1 = _mm_add_epi8(_mm_subs_epu8(g2g, g2gref), _mm_subs_epu8(g2gref, g2g));
								__m128i g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								__m128i b1 = _mm_add_epi8(_mm_subs_epu8(g2b, g2bref), _mm_subs_epu8(g2bref, g2b));
								__m128i b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(gbuf + 8), r2);
								_mm_store_si128((__m128i*)gbuf, r1);

								__m128i g1bref = _mm_loadu_si128((__m128i*)(g1ptrbj + *g1ofs));
								__m128i g1gref = _mm_loadu_si128((__m128i*)(g1ptrgj + *g1ofs));
								__m128i g1rref = _mm_loadu_si128((__m128i*)(g1ptrrj + *g1ofs));

								r1 = _mm_add_epi8(_mm_subs_epu8(g1r, g1rref), _mm_subs_epu8(g1rref, g1r));
								r2 = _mm_unpackhi_epi8(r1, zero);
								r1 = _mm_unpacklo_epi8(r1, zero);

								g1 = _mm_add_epi8(_mm_subs_epu8(g1g, g1gref), _mm_subs_epu8(g1gref, g1g));
								g2 = _mm_unpackhi_epi8(g1, zero);
								g1 = _mm_unpacklo_epi8(g1, zero);

								r1 = _mm_add_epi16(r1, g1);
								r2 = _mm_add_epi16(r2, g2);

								b1 = _mm_add_epi8(_mm_subs_epu8(g1b, g1bref), _mm_subs_epu8(g1bref, g1b));
								b2 = _mm_unpackhi_epi8(b1, zero);
								b1 = _mm_unpacklo_epi8(b1, zero);

								r1 = _mm_add_epi16(r1, b1);
								r2 = _mm_add_epi16(r2, b2);

								_mm_store_si128((__m128i*)(buf + 8), r2);
								_mm_store_si128((__m128i*)buf, r1);

								__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
								__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
								__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

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
								__m128 _w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[3]], guide1_color_weight[buf[2]], guide1_color_weight[buf[1]], guide1_color_weight[buf[0]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[3]], guide2_color_weight[gbuf[2]], guide2_color_weight[gbuf[1]], guide2_color_weight[gbuf[0]]));

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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[7]], guide1_color_weight[buf[6]], guide1_color_weight[buf[5]], guide1_color_weight[buf[4]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[7]], guide2_color_weight[gbuf[6]], guide2_color_weight[gbuf[5]], guide2_color_weight[gbuf[4]]));
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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[11]], guide1_color_weight[buf[10]], guide1_color_weight[buf[9]], guide1_color_weight[buf[8]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[11]], guide2_color_weight[gbuf[10]], guide2_color_weight[gbuf[9]], guide2_color_weight[gbuf[8]]));
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

								_w = _mm_mul_ps(_sw, _mm_set_ps(guide1_color_weight[buf[15]], guide1_color_weight[buf[14]], guide1_color_weight[buf[13]], guide1_color_weight[buf[12]]));
								_w = _mm_mul_ps(_w, _mm_set_ps(guide2_color_weight[gbuf[15]], guide2_color_weight[gbuf[14]], guide2_color_weight[gbuf[13]], guide2_color_weight[gbuf[12]]));
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
						const uchar* g1ptrrj = g1ptrr + j;
						const uchar* g1ptrgj = g1ptrg + j;
						const uchar* g1ptrbj = g1ptrb + j;
						const uchar* g2ptrrj = g2ptrr + j;
						const uchar* g2ptrgj = g2ptrg + j;
						const uchar* g2ptrbj = g2ptrb + j;

						int sr0 = sptrrj[0];
						int sg0 = sptrgj[0];
						int sb0 = sptrbj[0];
						int g1r0 = g1ptrrj[0];
						int g1g0 = g1ptrgj[0];
						int g1b0 = g1ptrbj[0];
						int g2r0 = g2ptrrj[0];
						int g2g0 = g2ptrgj[0];
						int g2b0 = g2ptrbj[0];

						float sum_r = 0.f;
						float sum_b = 0.f;
						float sum_g = 0.f;
						float wsum = 0.f;
						for (k = 0; k < maxk; k++)
						{
							int g1r = g1ptrrj[space_guide1_ofs[k]];
							int g1g = g1ptrgj[space_guide1_ofs[k]];
							int g1b = g1ptrbj[space_guide1_ofs[k]];
							int g2r = g2ptrrj[space_guide2_ofs[k]];
							int g2g = g2ptrgj[space_guide2_ofs[k]];
							int g2b = g2ptrbj[space_guide2_ofs[k]];
							float w = space_weight[k];
							w *= guide1_color_weight[abs(g1b - g1b0) + abs(g1g - g1g0) + abs(g1r - g1r0)];
							w *= guide2_color_weight[abs(g2b - g2b0) + abs(g2g - g2g0) + abs(g2r - g2r0)];

							sum_r += sptrrj[space_ofs[k]] * w;
							sum_g += sptrgj[space_ofs[k]] * w;
							sum_b += sptrbj[space_ofs[k]] * w;
							wsum += w;
						}
						dptr[3 * j] = (uchar)cvRound(sum_r / wsum);
						dptr[3 * j + 1] = (uchar)cvRound(sum_g / wsum);
						dptr[3 * j + 2] = (uchar)cvRound(sum_b / wsum);
					}
				}
			}
		}
	private:
		const Mat *temp;
		Mat *dest;
		const Mat* guide1;
		const Mat* guide2;
		int radiusH, radiusV, maxk, *space_ofs, *space_guide1_ofs, *space_guide2_ofs;
		float *space_weight, *guide1_color_weight, *guide2_color_weight;
	};


	void jointDualBilateralFilter_8u(const Mat& src, const Mat& guide1_, const Mat& guide2_, Mat& dst, Size kernelSize, const double sigma_guide_color1_, const double sigma_guide_color2_, const double sigma_space, const int borderType, bool isRectangle)
	{
		double sigma_guide_color1;
		double sigma_guide_color2;
		Mat guide1, guide2;
		if (guide1_.channels() > guide2_.channels())
		{
			guide2 = guide1_;
			guide1 = guide2_;

			sigma_guide_color1 = sigma_guide_color2_;
			sigma_guide_color2 = sigma_guide_color1_;
		}
		else
		{
			guide2 = guide2_;
			guide1 = guide1_;

			sigma_guide_color1 = sigma_guide_color1_;
			sigma_guide_color2 = sigma_guide_color2_;
		}

		if (kernelSize.width == 0 || kernelSize.height == 0){ src.copyTo(dst); return; }
		int cn = src.channels();

		int cng1 = guide1.channels();
		int cng2 = guide2.channels();

		int i, j, maxk;
		Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
			(guide1.type() == CV_8UC1 || guide1.type() == CV_8UC3) &&
			(guide2.type() == CV_8UC1 || guide2.type() == CV_8UC3) &&
			src.type() == dst.type() && src.size() == dst.size());

		double gauss_guide1_color_coeff = -0.5 / (sigma_guide_color1*sigma_guide_color1);
		double gauss_guide2_color_coeff = -0.5 / (sigma_guide_color2*sigma_guide_color2);
		double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		int radiusH = kernelSize.width >> 1;
		int radiusV = kernelSize.height >> 1;

		Mat temp, tempg1, tempg2;

		int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		int rpad = spad - lpad;
		if (cn == 1 && cng1 == 1 && cng2 == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide1, tempg1, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide2, tempg2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 1 && cng1 == 1 && cng2 == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide1, tempg1, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide2, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg2);
		}
		else if (cn == 1 && cng1 == 3 && cng2 == 1)
		{
			std::cout << "NA" << std::endl;
		}
		else if (cn == 1 && cng1 == 3 && cng2 == 3)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			Mat temp2;
			copyMakeBorder(guide1, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg1);
			copyMakeBorder(guide2, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg2);
		}
		else if (cn == 3 && cng1 == 1 && cng2 == 1)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
			copyMakeBorder(guide1, tempg1, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide2, tempg2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3 && cng1 == 1 && cng2 == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
			copyMakeBorder(guide1, tempg1, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			copyMakeBorder(guide2, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg2);
		}
		else if (cn == 3 && cng1 == 3 && cng2 == 1)
		{
			std::cout << "NA" << std::endl;
		}
		else if (cn == 3 && cng1 == 3 && cng2 == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, temp);
			copyMakeBorder(guide1, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg1);
			copyMakeBorder(guide2, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleave(temp2, tempg2);
		}

		vector<float> _color_guide1_weight(cng1 * 256);
		vector<float> _color_guide2_weight(cng2 * 256);
		vector<float> _space_weight(kernelSize.area());
		float* color_guide1_weight = &_color_guide1_weight[0];
		float* color_guide2_weight = &_color_guide2_weight[0];
		float* space_weight = &_space_weight[0];

		vector<int> _space_ofs(kernelSize.area());
		vector<int> _space_guide1_ofs(kernelSize.area());
		vector<int> _space_guide2_ofs(kernelSize.area());
		int* space_ofs = &_space_ofs[0];
		int* space_guide1_ofs = &_space_guide1_ofs[0];
		int* space_guide2_ofs = &_space_guide2_ofs[0];

		// initialize color-related bilateral filter coefficients


		for (i = 0; i < 256 * cng1; i++)
		{
			color_guide1_weight[i] = (float)std::exp(i*i*gauss_guide1_color_coeff);
		}
		for (i = 0; i < 256 * cng2; i++)
		{
			color_guide2_weight[i] = (float)std::exp(i*i*gauss_guide2_color_coeff);
		}

		// initialize space-related bilateral filter coefficients
		for (i = -radiusV, maxk = 0; i <= radiusV; i++)
		{
			j = -radiusH;
			for (; j <= radiusH; j++)
			{
				double r = std::sqrt((double)i*i + (double)j*j);
				if (r > max(radiusV, radiusH) && !isRectangle) continue;

				space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);

				space_ofs[maxk] = (int)(i*temp.cols*cn + j);
				space_guide1_ofs[maxk] = (int)(i*tempg1.cols*cng1 + j);
				space_guide2_ofs[maxk++] = (int)(i*tempg2.cols*cng2 + j);
			}
		}

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		JointDualBilateralFilter_8u_InvokerSSE4 body(dest, temp, tempg1, tempg2, radiusH, radiusV, maxk, space_ofs, space_guide1_ofs, space_guide2_ofs, space_weight, color_guide1_weight, color_guide2_weight);
		parallel_for_(Range(0, size.height), body);
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void jointDualBilateralFilter(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, Size ksize, double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int method, int borderType)
	{
		if (dst.empty()) dst.create(src.size(), src.type());

		if (method == FILTER_CIRCLE || method == FILTER_DEFAULT)
		{
			if (src.depth() == CV_8U)
			{
				jointDualBilateralFilter_8u(src, guide1, guide2, dst, ksize, sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, false);
			}
			else if (src.depth() == CV_32F)
			{
				jointDualBilateralFilterBase_32f(src, guide1, guide2, dst, ksize, sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, false);
			}
		}
		else if (method == FILTER_RECTANGLE)
		{
			if (src.depth() == CV_8U)
			{
				jointDualBilateralFilter_8u(src, guide1, guide2, dst, ksize, sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, true);
			}
			else if (src.depth() == CV_32F)
			{
				jointDualBilateralFilterBase_32f(src, guide1, guide2, dst, ksize, sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, true);
			}
		}
		else if (method == FILTER_SLOWEST)
		{
			if (src.depth() == CV_8U)
			{
				jointDualBilateralFilterBase_8u(src, guide1, guide2, dst, ksize, sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, false);
			}
			else if (src.depth() == CV_32F)
			{
				jointDualBilateralFilterBase_32f(src, guide1, guide2, dst, ksize, sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, false);
			}
			else
			{
				Mat guidef1, guidef2, srcf, destf;
				src.convertTo(srcf, CV_32F);
				guide1.convertTo(guidef1, CV_32F);
				guide2.convertTo(guide2, CV_32F);

				jointDualBilateralFilterBase_32f(srcf, guidef1, guidef2, destf, ksize, sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, false);
				destf.convertTo(dst, src.depth());
			}
		}
	}

	void jointDualBilateralFilter(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, int d, double sigma_guide_color1, double sigma_guide_color2, double sigma_space, int method, int borderType)
	{
		jointDualBilateralFilter(src, guide1, guide2, dst, Size(d, d), sigma_guide_color1, sigma_guide_color2, sigma_space, borderType, method);
	}

	void separableJointDualBilateralFilter(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, Size ksize, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1, double alpha2, int method, int borderType)
	{
		if (method == DUAL_KERNEL_HV)
		{
			jointDualBilateralFilter(src, guide1, guide2, dst, Size(ksize.width, 1), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);
			jointDualBilateralFilter(dst, guide1, guide2, dst, Size(1, ksize.height), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);
		}
		else if (method == DUAL_KERNEL_VH)
		{
			jointDualBilateralFilter(src, guide1, guide2, dst, Size(1, ksize.height), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);
			jointDualBilateralFilter(dst, guide1, guide2, dst, Size(ksize.width, 1), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);
		}
		else if (method == DUAL_KERNEL_HVVH)
		{
			jointDualBilateralFilter(src, guide1, guide2, dst, Size(ksize.width, 1), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);
			jointDualBilateralFilter(dst, guide1, guide2, dst, Size(1, ksize.height), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);
			Mat dst2;
			jointDualBilateralFilter(src, guide1, guide2, dst2, Size(1, ksize.height), sigma_color, sigma_guide_color, sigma_space, FILTER_RECTANGLE, borderType);
			jointDualBilateralFilter(dst2, guide1, guide2, dst2, Size(ksize.width, 1), sigma_color*alpha1, sigma_guide_color*alpha2, sigma_space, FILTER_RECTANGLE, borderType);

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

	void separableJointDualBilateralFilter(const Mat& src, const Mat& guide1, const Mat& guide2, Mat& dst, int D, double sigma_color, double sigma_guide_color, double sigma_space, double alpha1, double alpha2, int method, int borderType)
	{
		separableJointDualBilateralFilter(src, guide1, guide2, dst, Size(D, D), sigma_color, sigma_guide_color, sigma_space, alpha1, alpha2, method, borderType);
	}
}