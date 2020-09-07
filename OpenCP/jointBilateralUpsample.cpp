#include "jointBilateralUpsample.hpp"
#include "minmaxfilter.hpp"
#include "shiftImage.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	template <class srcType>
	static void nnUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols);
		const int dh = dest.rows / (src.rows);

		Mat sim; copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;
			srcType* s = sim.ptr<srcType>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType ltd = s[i];
				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						d[m + k] = ltd;
					}
				}
			}
		}
	}

	void nnUpsample(InputArray src_, OutputArray dest_)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) nnUpsample_<uchar>(src, dest);
		else if (src.depth() == CV_16S) nnUpsample_<short>(src, dest);
		else if (src.depth() == CV_16U) nnUpsample_<ushort>(src, dest);
		else if (src.depth() == CV_32S) nnUpsample_<int>(src, dest);
		else if (src.depth() == CV_32F) nnUpsample_<float>(src, dest);
		else if (src.depth() == CV_64F) nnUpsample_<double>(src, dest);
	}

	inline int linearinterpolate_(int lt, int rt, int lb, int rb, double a, double b)
	{
		return (int)((b*a*lt + b*(1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb) + 0.5);
	}

	template <class srcType>
	inline double linearinterpolate_(srcType lt, srcType rt, srcType lb, srcType rb, double a, double b)
	{
		return (b*a*lt + b*(1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb);
	}

	template <class srcType>
	static void linearUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols - 1);
		const int dh = dest.rows / (src.rows - 1);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		Mat sim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, BORDER_REPLICATE);
		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;
			srcType* s = sim.ptr<srcType>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType ltd = s[i];
				const srcType rtd = s[i + 1];
				const srcType lbd = s[i + sim.cols];
				const srcType rbd = s[i + 1 + sim.cols];
				for (int l = 0; l < dh; l++)
				{
					double beta = 1.0 - (double)l / dh;
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;
						d[m + k] = saturate_cast<srcType> (linearinterpolate_<srcType>(ltd, rtd, lbd, rbd, alpha, beta));
					}
				}
			}
		}
	}

	void linearUpsample(InputArray src_, OutputArray dest_)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) linearUpsample_<uchar>(src, dest);
		else if (src.depth() == CV_16S) linearUpsample_<short>(src, dest);
		else if (src.depth() == CV_16U) linearUpsample_<ushort>(src, dest);
		else if (src.depth() == CV_32S) linearUpsample_<int>(src, dest);
		else if (src.depth() == CV_32F) linearUpsample_<float>(src, dest);
		else if (src.depth() == CV_64F) linearUpsample_<double>(src, dest);
	}


	inline double cubicfunc(double x, double a = -1.0)
	{
		double X = abs(x);
		if (X <= 1)
			return ((a + 2.0)*x*x*x - (a + 3.0)*x*x + 1.0);
		else if (X <= 2)
			return (a*x*x*x - 5.0*a*x*x + 8.0*a*x - 4.0*a);
		else
			return 0.0;
	}

	template <class srcType>
	static void cubicUpsample_(Mat& src, Mat& dest, double a)
	{
		const int dw = dest.cols / (src.cols - 1);
		const int dh = dest.rows / (src.rows - 1);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		vector<vector<double>> weight(dh*dw);
		for (int i = 0; i < weight.size(); i++)weight[i].resize(16);

		int idx = 0;

		for (int l = 0; l < dh; l++)
		{
			const double y = (double)l / (double)dh;
			for (int k = 0; k < dw; k++)
			{
				const double x = (double)k / (double)dw;

				weight[idx][0] = cubicfunc(1.0 + x, a)*cubicfunc(1.0 + y, a);
				weight[idx][1] = cubicfunc(0.0 + x, a)*cubicfunc(1.0 + y, a);
				weight[idx][2] = cubicfunc(1.0 - x, a)*cubicfunc(1.0 + y, a);
				weight[idx][3] = cubicfunc(2.0 - x, a)*cubicfunc(1.0 + y, a);

				weight[idx][4] = cubicfunc(1.0 + x, a)*cubicfunc(0.0 + y, a);
				weight[idx][5] = cubicfunc(0.0 + x, a)*cubicfunc(0.0 + y, a);
				weight[idx][6] = cubicfunc(1.0 - x, a)*cubicfunc(0.0 + y, a);
				weight[idx][7] = cubicfunc(2.0 - x, a)*cubicfunc(0.0 + y, a);

				weight[idx][8] = cubicfunc(1.0 + x, a)*cubicfunc(1.0 - y, a);
				weight[idx][9] = cubicfunc(0.0 + x, a)*cubicfunc(1.0 - y, a);
				weight[idx][10] = cubicfunc(1.0 - x, a)*cubicfunc(1.0 - y, a);
				weight[idx][11] = cubicfunc(2.0 - x, a)*cubicfunc(1.0 - y, a);

				weight[idx][12] = cubicfunc(1.0 + x, a)*cubicfunc(2.0 - y, a);
				weight[idx][13] = cubicfunc(0.0 + x, a)*cubicfunc(2.0 - y, a);
				weight[idx][14] = cubicfunc(1.0 - x, a)*cubicfunc(2.0 - y, a);
				weight[idx][15] = cubicfunc(2.0 - x, a)*cubicfunc(2.0 - y, a);

				double wsum = 0.0;
				for (int i = 0; i < 16; i++)wsum += weight[idx][i];
				for (int i = 0; i < 16; i++)weight[idx][i] /= wsum;

				idx++;
			}
		}

		Mat sim;
		copyMakeBorder(src, sim, 1, 2, 1, 2, BORDER_REPLICATE);
		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;
			srcType* s = sim.ptr<srcType>(j + 1) + 1;

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType v00 = s[i - 1 - sim.cols];
				const srcType v01 = s[i - 0 - sim.cols];
				const srcType v02 = s[i + 1 - sim.cols];
				const srcType v03 = s[i + 2 - sim.cols];
				const srcType v10 = s[i - 1];
				const srcType v11 = s[i - 0];
				const srcType v12 = s[i + 1];
				const srcType v13 = s[i + 2];
				const srcType v20 = s[i - 1 + sim.cols];
				const srcType v21 = s[i - 0 + sim.cols];
				const srcType v22 = s[i + 1 + sim.cols];
				const srcType v23 = s[i + 2 + sim.cols];
				const srcType v30 = s[i - 1 + 2 * sim.cols];
				const srcType v31 = s[i - 0 + 2 * sim.cols];
				const srcType v32 = s[i + 1 + 2 * sim.cols];
				const srcType v33 = s[i + 2 + 2 * sim.cols];

				int idx = 0;
				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						
						d[m + k] = saturate_cast<srcType>(
							weight[idx][0] * v00 + weight[idx][1] * v01 + weight[idx][2] * v02 + weight[idx][3] * v03
							+ weight[idx][4] * v10 + weight[idx][5] * v11 + weight[idx][6] * v12 + weight[idx][7] * v13
							+ weight[idx][8] * v20 + weight[idx][9] * v21 + weight[idx][10] * v22 + weight[idx][11] * v23
							+ weight[idx][12] * v30 + weight[idx][13] * v31 + weight[idx][14] * v32 + weight[idx][15] * v33
							);
						idx++;
					}
				}
			}
		}
	}

	void cubicUpsample(InputArray src_, OutputArray dest_, double a)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) cubicUpsample_<uchar>(src, dest, a);
		else if (src.depth() == CV_16S) cubicUpsample_<short>(src, dest, a);
		else if (src.depth() == CV_16U) cubicUpsample_<ushort>(src, dest, a);
		else if (src.depth() == CV_32S) cubicUpsample_<int>(src, dest, a);
		else if (src.depth() == CV_32F) cubicUpsample_<float>(src, dest, a);
		else if (src.depth() == CV_64F) cubicUpsample_<double>(src, dest, a);
	}

	void setUpsampleMask(InputArray src, OutputArray dst)
	{
		Mat dest = dst.getMat();
		if (dest.empty())
		{
			cout << "please alloc dest Mat" << endl;
			return;
		}
		dest.setTo(0);
		const int dw = dest.cols / (src.size().width);
		const int dh = dest.rows / (src.size().height);

		for (int j = 0; j < src.size().height; j++)
		{
			int n = j*dh;
			uchar* d = dest.ptr<uchar>(n);
			for (int i = 0, m = 0; i < src.size().width; i++, m += dw)
			{
				d[m] = 255;
			}
		}
	}

	template<class srcType>
	inline srcType weightedinterpolation_(srcType lt, srcType rt, srcType lb, srcType rb, double wlt, double wrt, double wlb, double  wrb)
	{
		return (srcType)((wlt*lt + wrt*rt + wlb*lb + wrb*rb) / (wlt + wrt + wlb + wrb) + 0.5);
	}

	template<class srcType>
	inline srcType weightedlinearinterpolate_(srcType lt, srcType rt, srcType lb, srcType rb, double wlt, double wrt, double wlb, double  wrb, double a, double b)
	{
		return (srcType)((b*a*wlt*lt + b*(1.0 - a)*wrt*rt + (1.0 - b)*a*wlb*lb + (1.0 - b)*(1.0 - a)*wrb*rb) / (b*a*wlt + b*(1.0 - a)*wrt + (1.0 - b)*a*wlb + (1.0 - b)*(1.0 - a)*wrb) + 0.5);
		//return (int)((b*a*lt + b*(1.0-a)*rt + (1.0-b)*a*lb + (1.0-b)*(1.0-a)*rb)+0.5);
	}

	//NAFDU:
	template<class srcType>
	static void noiseAwareFilterDepthUpsample_(Mat& src, Mat& joint, Mat& dest, double sigma_c, double sigma_d, double sigma_s, double eps, double tau)
	{
		Mat maxf;
		Mat minf;
		maxFilter(src, maxf, Size(3, 3));
		minFilter(src, minf, Size(3, 3));
		Mat diff = (maxf - minf);
		Mat ups;

		resize(src, ups, joint.size(), 0, 0, INTER_LINEAR);
		warpShift(ups, ups, -4, -4);

		if (dest.empty())dest.create(joint.size(), src.type());
		Mat sim, jim, eim, uim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		copyMakeBorder(ups, uim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut_sig[256];
		for (int i = 0; i < 256; i++)
		{
			lut_sig[i] = 1.0 / (1.0 + (double)std::exp(-eps*(i - tau)));
		}

		double lut_d[256];
		double gauss_d_coeff = -0.5 / (sigma_d*sigma_d);
		for (int i = 0; i < 256; i++)
		{
			lut_d[i] = (double)std::exp(i*i*gauss_d_coeff);
		}

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}
		vector<double> lut_(dw*dh);
		double* lut_s = &lut_[0];
		if (sigma_s <= 0.0)
		{
			for (int i = 0; i < dw*dh; i++)
			{
				lut_s[i] = 1.0;
			}
		}
		else
		{
			double gauss_s_coeff = -0.5 / (sigma_s*sigma_s);
			for (int i = 0; i < dw*dh; i++)
			{
				lut_s[i] = (double)std::exp(i*i*gauss_s_coeff);
			}
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			srcType* s = sim.ptr<srcType>(j);
			srcType* minmaxdiff = diff.ptr<srcType>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];


				const srcType ltd = s[i];
				const srcType rtd = s[i + 1];
				const srcType lbd = s[i + sim.cols];
				const srcType rbd = s[i + 1 + sim.cols];

				double a = lut_sig[(int)minmaxdiff[i]];
				double ia = 1.0 - a;

				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					uchar* jnt = jim.ptr<uchar>(n + l);
					srcType* ud = uim.ptr<srcType>(n + l);
					//double beta = 1.0-(double)l/dh;

					for (int k = 0; k < dw; k++)
					{
						//double alpha = 1.0-(double)k/dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];
						const int z = (int)ud[(m + k)];

						double wlt = lut_s[k + l] * (a*lut[abs(ltr - r) + abs(ltg - g) + abs(ltb - b)] + ia*lut_d[abs(z - (int)ltd)]);
						double wrt = lut_s[dw - k + l] * (a*lut[abs(rtr - r) + abs(rtg - g) + abs(rtb - b)] + ia*lut_d[abs(z - (int)rtd)]);
						double wlb = lut_s[k + dh - l] * (a*lut[abs(lbr - r) + abs(lbg - g) + abs(lbb - b)] + ia*lut_d[abs(z - (int)lbd)]);
						double wrb = lut_s[dw - k + dh - l] * (a*lut[abs(rbr - r) + abs(rbg - g) + abs(rbb - b)] + ia*lut_d[abs(z - (int)rbd)]);


						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = weightedinterpolation_<srcType>(ltd, rtd, lbd, rbd, wlt, wrt, wlb, wrb);
					}
				}
			}
		}
	}

	void noiseAwareFilterDepthUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c, double sigma_d, double sigma_s, double eps, double tau)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U) noiseAwareFilterDepthUpsample_<uchar>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_16S) noiseAwareFilterDepthUpsample_<short>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_16U) noiseAwareFilterDepthUpsample_<ushort>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_32S) noiseAwareFilterDepthUpsample_<int>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_32F) noiseAwareFilterDepthUpsample_<float>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_64F) noiseAwareFilterDepthUpsample_<double>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
	}

	template<class srcType>
	static void jointBilateralLinearUpsample_(Mat& src, const Mat& joint, Mat& dest, double sigma_c)
	{
		if (dest.empty())dest.create(joint.size(), src.type());
		Mat sim, jim, eim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			srcType* s = sim.ptr<srcType>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];

				const srcType ltd = s[i];
				const srcType rtd = s[i + 1];
				const srcType lbd = s[i + sim.cols];
				const srcType rbd = s[i + 1 + sim.cols];

				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					uchar* jnt = jim.ptr(n + l);
					double beta = 1.0 - (double)l / dh;

					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];

						int minE = abs(ltr - r) + abs(ltg - g) + abs(ltb - b);
						//T mind=ltd;
						//int minp=0;

						double wlt = lut[minE];

						int e = abs(rtr - r) + abs(rtg - g) + abs(rtb - b);
						double wrt = lut[e];

						if (minE > e)
						{
							minE = e;
							//mind = rtd;
							//minp=1;
						}

						e = abs(lbr - r) + abs(lbg - g) + abs(lbb - b);
						double wlb = lut[e];

						if (minE > e)
						{
							minE = e;
							//mind = lbd;
							//minp=2;
						}

						e = abs(rbr - r) + abs(rbg - g) + abs(rbb - b);
						double wrb = lut[e];

						if (minE > e)
						{
							minE = e;
							//mind = rbd;
							//minp=3;
						}

						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = weightedlinearinterpolate_<srcType>(ltd, rtd, lbd, rbd, wlt, wrt, wlb, wrb, alpha, beta);
					}
				}
			}
		}
	}

	void jointBilateralLinearUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U)
			jointBilateralLinearUpsample_<uchar>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_16S)
			jointBilateralLinearUpsample_<short>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_16U)
			jointBilateralLinearUpsample_<ushort>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_32F)
			jointBilateralLinearUpsample_<float>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_64F)
			jointBilateralLinearUpsample_<double>(src, joint, dest, sigma_c);
	}


	template<class srcType>
	static void jointBilateralUpsample_(Mat& src, Mat& joint, Mat& dest, double sigma_c, double sigma_s)
	{
		if (dest.empty())dest.create(joint.size(), src.type());
		Mat sim, jim, eim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}
		vector<double> lut_(dw*dh);
		double* lut2 = &lut_[0];
		if (sigma_s <= 0.0)
		{
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = 1.0;
			}
		}
		else
		{
			double gauss_s_coeff = -0.5 / (sigma_s*sigma_s);
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = (double)std::exp(i*i*gauss_s_coeff);
			}
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			srcType* s = sim.ptr<srcType>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];

				const srcType ltd = s[i];
				const srcType rtd = s[i + 1];
				const srcType lbd = s[i + sim.cols];
				const srcType rbd = s[i + 1 + sim.cols];

				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					uchar* jnt = jim.ptr(n + l);
					//double beta = 1.0-(double)l/dh;

					for (int k = 0; k < dw; k++)
					{
						//double alpha = 1.0-(double)k/dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];



						double wlt = lut2[k + l] * lut[abs(ltr - r) + abs(ltg - g) + abs(ltb - b)];
						double wrt = lut2[dw - k + l] * lut[abs(rtr - r) + abs(rtg - g) + abs(rtb - b)];
						double wlb = lut2[k + dh - l] * lut[abs(lbr - r) + abs(lbg - g) + abs(lbb - b)];
						double wrb = lut2[dw - k + dh - l] * lut[abs(rbr - r) + abs(rbg - g) + abs(rbb - b)];



						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = weightedinterpolation_<srcType>(ltd, rtd, lbd, rbd, wlt, wrt, wlb, wrb);
					}
				}
			}
		}
	}

	void jointBilateralUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c, double sigma_s)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U)
			jointBilateralUpsample_<uchar>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16S)
			jointBilateralUpsample_<short>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16U)
			jointBilateralUpsample_<ushort>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_32F)
			jointBilateralUpsample_<float>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_64F)
			jointBilateralUpsample_<double>(src, joint, dest, sigma_c, sigma_s);
	}


	template<class srcType>
	static void jointBilateralNNUpsample_(Mat& src, Mat& joint, Mat& dest, double sigma_c, double sigma_s)
	{
		Mat sim, jim, eim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}
		vector<double> lut_(dw*dh);
		double* lut2 = &lut_[0];
		if (sigma_s <= 0.0)
		{
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = 1.0;
			}
		}
		else
		{
			double gauss_s_coeff = -0.5 / (sigma_s*sigma_s);
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = (double)std::exp(i*i*gauss_s_coeff);
			}
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			srcType* s = sim.ptr<srcType>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];

				const srcType ltd = s[i];
				const srcType rtd = s[i + 1];
				const srcType lbd = s[i + sim.cols];
				const srcType rbd = s[i + 1 + sim.cols];

				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					uchar* jnt = jim.ptr(n + l);
					double beta = 1.0 - (double)l / dh;

					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];

						int minE = abs(ltr - r) + abs(ltg - g) + abs(ltb - b);
						srcType mind = ltd;
						//int minp=0;

						double wlt = lut2[k + l] * lut[minE];
						double maxE = wlt;

						int e = abs(rtr - r) + abs(rtg - g) + abs(rtb - b);
						double wrt = lut2[dw - k + l] * lut[e];

						if (maxE < wrt)
						{
							maxE = wrt;
							mind = rtd;
							//minp=1;
						}

						e = abs(lbr - r) + abs(lbg - g) + abs(lbb - b);
						double wlb = lut2[k + dh - l] * lut[e];

						if (maxE < wlb)
						{
							maxE = wlb;
							mind = lbd;
							//minp=2;
						}

						e = abs(rbr - r) + abs(rbg - g) + abs(rbb - b);
						double wrb = lut2[dw - k + dh - l] * lut[e];

						if (maxE < wrb)
						{
							maxE = wrb;
							mind = rbd;
							//minp=3;
						}

						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = mind;//weightedlinearinterpolate_<T>(ltd,rtd,lbd,rbd,wlt,wrt,wlb,wrb,alpha,beta);
					}
				}
			}
		}
	}

	void jointBilateralNNUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c, double sigma_s)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U)
			jointBilateralNNUpsample_<uchar>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16S)
			jointBilateralNNUpsample_<short>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16U)
			jointBilateralNNUpsample_<ushort>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_32F)
			jointBilateralNNUpsample_<float>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_64F)
			jointBilateralNNUpsample_<double>(src, joint, dest, sigma_c, sigma_s);
	}

}