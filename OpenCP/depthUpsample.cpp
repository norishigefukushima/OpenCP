#include "depthUpsample.hpp"
#include "minmaxfilter.hpp"
#include "shiftImage.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	template<class srcType>
	static inline srcType weightedinterpolation_(srcType lt, srcType rt, srcType lb, srcType rb, double wlt, double wrt, double wlb, double  wrb)
	{
		return (srcType)((wlt * lt + wrt * rt + wlb * lb + wrb * rb) / (wlt + wrt + wlb + wrb) + 0.5);
	}

	template<class srcType>
	static inline srcType weightedlinearinterpolate_(srcType lt, srcType rt, srcType lb, srcType rb, double wlt, double wrt, double wlb, double  wrb, double a, double b)
	{
		return (srcType)((b * a * wlt * lt + b * (1.0 - a) * wrt * rt + (1.0 - b) * a * wlb * lb + (1.0 - b) * (1.0 - a) * wrb * rb) / (b * a * wlt + b * (1.0 - a) * wrt + (1.0 - b) * a * wlb + (1.0 - b) * (1.0 - a) * wrb) + 0.5);
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
			lut_sig[i] = 1.0 / (1.0 + (double)std::exp(-eps * (i - tau)));
		}

		double lut_d[256];
		double gauss_d_coeff = -0.5 / (sigma_d * sigma_d);
		for (int i = 0; i < 256; i++)
		{
			lut_d[i] = (double)std::exp(i * i * gauss_d_coeff);
		}

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c * sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i * i * gauss_c_coeff);
		}
		vector<double> lut_(dw * dh);
		double* lut_s = &lut_[0];
		if (sigma_s <= 0.0)
		{
			for (int i = 0; i < dw * dh; i++)
			{
				lut_s[i] = 1.0;
			}
		}
		else
		{
			double gauss_s_coeff = -0.5 / (sigma_s * sigma_s);
			for (int i = 0; i < dw * dh; i++)
			{
				lut_s[i] = (double)std::exp(i * i * gauss_s_coeff);
			}
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j * dh;

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

				const uchar lbr = jnt_[3 * (m + jim.cols * dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols * dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols * dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols * dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols * dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols * dh) + 2];


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

						double wlt = lut_s[k + l] * (a * lut[abs(ltr - r) + abs(ltg - g) + abs(ltb - b)] + ia * lut_d[abs(z - (int)ltd)]);
						double wrt = lut_s[dw - k + l] * (a * lut[abs(rtr - r) + abs(rtg - g) + abs(rtb - b)] + ia * lut_d[abs(z - (int)rtd)]);
						double wlb = lut_s[k + dh - l] * (a * lut[abs(lbr - r) + abs(lbg - g) + abs(lbb - b)] + ia * lut_d[abs(z - (int)lbd)]);
						double wrb = lut_s[dw - k + dh - l] * (a * lut[abs(rbr - r) + abs(rbg - g) + abs(rbb - b)] + ia * lut_d[abs(z - (int)rbd)]);


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

	void noiseAwareFilterDepthUpsample(InputArray src_, InputArray joint_, OutputArray dst, const double sigma_c, const double sigma_d, const double sigma_s, const double eps, const double tau)
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
}