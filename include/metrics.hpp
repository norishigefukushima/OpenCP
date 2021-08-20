#pragma once

#include "common.hpp"

namespace cp
{
	enum PSNR_CHANNEL
	{
		PSNR_ALL,
		PSNR_Y,
		PSNR_B,
		PSNR_G,
		PSNR_R,

		PSNR_CHANNEL_SIZE
	};
	enum PSNR_PRECISION
	{
		PSNR_UP_CAST,
		PSNR_8U,
		PSNR_32F,
		PSNR_64F,
		PSNR_KAHAN_64F,

		PSNR_PRECISION_SIZE
	};

	std::string CP_EXPORT getPSNR_PRECISION(const int precision);

	std::string CP_EXPORT getPSNR_CHANNEL(const int channel);

	class CP_EXPORT PSNRMetrics
	{
		cv::Mat source;
		cv::Mat reference;

		cv::Mat crops;
		cv::Mat cropr;
		cv::Mat temp;
		std::vector<cv::Mat> vtemp;

		double MSE_8U(cv::Mat& src, cv::Mat& reference);
		double MSE_32F(cv::Mat& src, cv::Mat& reference);
		double MSE_64F(cv::Mat& src, cv::Mat& reference, bool isKahan = true);

		inline int getPrecisionUpCast(cv::InputArray src, cv::InputArray ref);
		void cvtImageForMSE64F(const cv::Mat& src, cv::Mat& dest, const int cmethod);
		void cvtImageForMSE32F(const cv::Mat& src, cv::Mat& dest, const int cmethod);
		void cvtImageForMSE8U(const cv::Mat& src, cv::Mat& dest, const int cmethod);
	public:
		/// <summary>
		/// compute MSE
		/// </summary>
		/// <param name="src">src image</param>
		/// <param name="ref">reference image</param>
		/// <param name="boundingBox">bonding box ignoring outside region. default is 0.</param>
		/// <param name="precision">computing precision. Default: PSNR_UP_CAST(0). Other: PSNR_8U(0),PSNR_32F(1), PSNR_64F(2), PSNR_KAHAN_64F(3)</param>
		/// <param name="compare_channel">computing channlel. Default: compute MSE all channele PSNR_ALL(0). Other: PSNR_Y(1), PSNR_B(2), PSNR_G(3), PSNR_R(4)</param>
		/// <returns>MSE value</returns>
		double getMSE(cv::InputArray src, cv::InputArray ref, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);

		/// <summary>
		/// compute PSNR
		/// </summary>
		/// <param name="src">src image</param>
		/// <param name="ref">reference image</param>
		/// <param name="boundingBox">bonding box ignoring outside region. default is 0.</param>
		/// <param name="precision">computing precision. Default: PSNR_UP_CAST(0). Other: PSNR_8U(0),PSNR_32F(1), PSNR_64F(2), PSNR_KAHAN_64F(3)</param>
		/// <param name="compare_channel">computing channlel. Default: compute MSE all channele and then logged PSNR_ALL(0). Other: PSNR_Y(1), PSNR_B(2), PSNR_G(3), PSNR_R(4)</param>
		/// <returns>PSNR value, 0: same, -1: NaN, -2: Inf</returns>
		double getPSNR(cv::InputArray src, cv::InputArray ref, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);

		/// <summary>
		/// same function of getPSNR() for short cut
		/// </summary>
		/// <param name="src">src image</param>
		/// <param name="ref">reference image</param>
		/// <param name="boundingBox">bonding box ignoring outside region. default is 0.</param>
		/// <param name="precision">computing precision. Default: PSNR_UP_CAST(0). Other: PSNR_8U(0),PSNR_32F(1), PSNR_64F(2), PSNR_KAHAN_64F(3)</param>
		/// <param name="compare_channel">computing channlel. Default: compute MSE all channele and then logged PSNR_ALL(0). Other: PSNR_Y(1), PSNR_B(2), PSNR_G(3), PSNR_R(4)</param>
		/// <returns>PSNR value, 0: same, -1: NaN, -2: Inf</returns>
		double operator()(cv::InputArray src, cv::InputArray ref, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);

		//set reference image for acceleration
		void setReference(cv::InputArray src, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);

		//using preset reference image for acceleration
		double getMSEPreset(cv::InputArray src, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);

		//using preset reference image for acceleration
		double getPSNRPreset(cv::InputArray src, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	};

	/*

	boundingBox: ignoring outside region. default is 0
	precision: computing precision, default PSNR_32F(1), other PSNR_8U(0),PSNR_64F(2), PSNR_KAHAN_64F(3)
	compare_channel: default compute MSE all channele and then logged PSNR_ALL(0), PSNR_Y(1), PSNR_B(2), PSNR_G(3), PSNR_R(4),
	0: same image, -1: NaN, -2: Inf, else PSNR.
	*/

	/// <summary>
	/// Wrapper function for class PSNRMetrics::getPSNR
	/// </summary>
	/// <param name="src">Src image</param>
	/// <param name="reference">Reference image</param>
	/// <param name="boundingBox">bonding box ignoring outside region. Default is 0.</param>
	/// <param name="precision">Computing precision. Default: PSNR_UP_CAST(0). Other: PSNR_8U(0),PSNR_32F(1), PSNR_64F(2), PSNR_KAHAN_64F(3)</param>
	/// <param name="compare_channel">Computing channlel. Default: compute MSE all channele and then logged PSNR_ALL(0). Other: PSNR_Y(1), PSNR_B(2), PSNR_G(3), PSNR_R(4)</param>
	/// <returns>PSNR value</returns>
	CP_EXPORT double getPSNR(cv::InputArray src, cv::InputArray reference, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);

	/// <summary>
	/// clip src and reference value with min/maxval an then compuite PSNR.
	/// </summary>
	/// <param name="src">Src image</param>
	/// <param name="reference">Reference image</param>
	/// <param name="minval">Minimum value for clip.</param>
	/// <param name="maxval">Maximum value for clip.</param>
	/// <param name="boundingBox">bonding box ignoring outside region. Default is 0.</param>
	/// <param name="precision">Computing precision. Default: PSNR_UP_CAST(0). Other: PSNR_8U(0),PSNR_32F(1), PSNR_64F(2), PSNR_KAHAN_64F(3)</param>
	/// <param name="compare_channel">Computing channlel. Default: compute MSE all channele and then logged PSNR_ALL(0). Other: PSNR_Y(1), PSNR_B(2), PSNR_G(3), PSNR_R(4)</param>
	/// <returns>PSNR value</returns>
	CP_EXPORT double getPSNRClip(cv::InputArray src, cv::InputArray reference, const double minval = 0.0, const double maxval = 255.0, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	CP_EXPORT double getMSE(cv::InputArray src1, cv::InputArray src2const, int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	CP_EXPORT double getMSE(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask);

	CP_EXPORT void localPSNRMap(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dest, const int r, const int compare_channel, const double psnr_infinity_value = 0.0);
	CP_EXPORT void guiLocalPSNRMap(cv::InputArray src1, cv::InputArray src2, const bool isWait = true, std::string wname = "AreaPSNR");



	//bad pixel ratio for stere matching evaluation
	CP_EXPORT double getInacceptableRatio(cv::InputArray src, cv::InputArray ref, const int threshold);

	CP_EXPORT double getEntropy(cv::InputArray src, cv::InputArray mask = cv::noArray());

	CP_EXPORT double getTotalVariation(cv::InputArray src);

	//CP_EXPORT double SSIM(cv::Mat& src, cv::Mat& ref, double sigma = 1.5);removed for update
}