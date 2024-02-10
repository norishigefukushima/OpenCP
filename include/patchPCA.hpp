#pragma once
#include <opencv2/core.hpp>

namespace cp
{
	enum class DRIM2COLType
	{
		FULL_SUB_FULL_32F,
		FULL_SUB_FULL_64F,
		FULL_SUB_HALF_32F,
		FULL_SUB_HALF_64F,
		FULL_SUB_REP_32F,
		FULL_SUB_REP_64F,

		MEAN_SUB_HALF_32F,
		NO_SUB_HALF_32F,
		CONST_SUB_HALF_32F,
		MEAN_SUB_HALF_64F,
		NO_SUB_HALF_64F,
		CONST_SUB_HALF_64F,

		MEAN_SUB_REP_32F,
		NO_SUB_REP_32F,
		CONST_SUB_REP_32F,
		MEAN_SUB_REP_64F,
		NO_SUB_REP_64F,
		CONST_SUB_REP_64F,

		MEAN_SUB_CONV_32F,
		NO_SUB_CONV_32F,
		CONST_SUB_CONV_32F,
		MEAN_SUB_CONV_64F,
		NO_SUB_CONV_64F,
		CONST_SUB_CONV_64F,

		MEAN_SUB_CONVF_32F,
		NO_SUB_CONVF_32F,
		CONST_SUB_CONVF_32F,
		MEAN_SUB_CONVF_64F,
		NO_SUB_CONVF_64F,
		CONST_SUB_CONVF_64F,

		MEAN_SUB_FFT_32F,
		NO_SUB_FFT_32F,
		CONST_SUB_FFT_32F,
		MEAN_SUB_FFT_64F,
		NO_SUB_FFT_64F,
		CONST_SUB_FFT_64F,

		MEAN_SUB_FFTF_32F,
		NO_SUB_FFTF_32F,
		CONST_SUB_FFTF_32F,
		MEAN_SUB_FFTF_64F,
		NO_SUB_FFTF_64F,
		CONST_SUB_FFTF_64F,

		TEST,

		OPENCV_PCA,
		OPENCV_COV,

		MEAN_SUB_SEPSVD,
		NO_SUB_SEPSVD,
		CONST_SUB_SEPSVD,

		MEAN_SUB_SEPCOVX,
		NO_SUB_SEPCOVX,
		CONST_SUB_SEPCOVX,

		MEAN_SUB_SEPCOVY,
		NO_SUB_SEPCOVY,
		CONST_SUB_SEPCOVY,

		MEAN_SUB_SEPCOVXXt,
		NO_SUB_SEPCOVXXt,
		CONST_SUB_SEPCOVXXt,


		SIZE
	};

	enum class DRIM2COLElementSkipElement
	{
		FULL,
		HALF,
		REP,
		CONV,
		CONVF,
		FFT,
		FFTF,
	};

	std::string getDRIM2COLName(const DRIM2COLType method);
	std::string getDRIM2COLElementSkipTypeName(DRIM2COLElementSkipElement method);

	class CalcPatchCovarMatrix
	{
	public:
		void computeCov(const std::vector<cv::Mat>& src, const int patch_rad, cv::Mat& cov, const DRIM2COLType method = DRIM2COLType::NO_SUB_HALF_32F, const int skip = 1, const bool isParallel = false);
		void computeSepCov(const std::vector<cv::Mat>& src, const int patch_rad, std::vector<cv::Mat>& cov, const DRIM2COLType method = DRIM2COLType::NO_SUB_SEPCOVX, const int skip = 1, const bool isParallel = false);
		void computeCov(const cv::Mat& src, const int patch_rad, cv::Mat& cov, const DRIM2COLType method = DRIM2COLType::NO_SUB_HALF_32F, const int skip = 1, const bool isParallel = false);
		void computeSepCov(const cv::Mat& src, const int patch_rad, std::vector<cv::Mat>& cov, const DRIM2COLType method = DRIM2COLType::NO_SUB_SEPCOVX, const int skip = 1, const bool isParallel = false);
		void setBorder(const int border);
		void setConstSub(const int const_sub);
		static int getNumElements(int partch_rad, int color_channel, const DRIM2COLType method);
	private:
		int border = cv::BORDER_DEFAULT;
		int patch_rad;
		int D;
		int color_channels;
		int dim;
		std::vector<cv::Mat> data;
		std::vector<cv::Mat> dataBorder;
		void getScanorder(int* scan, const int step, const int channels, const bool isReverse = true);
		void getScanorderBorder(int* scan, const int step, const int channels);

		void setCovHalf(cv::Mat& destCovariance, const std::vector<double>& covElem, const double normalSize);
		void setCovRep(const std::vector<double>& meanv, const std::vector<double>& varElem, cv::Mat& destCovariance, std::vector<double>& covElem, std::vector<std::vector<cv::Point>>& covset, const double normalSize);

		void naive(const std::vector<cv::Mat>& src, cv::Mat& cov, const int skip);
		void naive(const std::vector<cv::Mat>& src, cv::Mat& cov, cv::Mat& mask);

		enum class CenterMethod
		{
			FULL,
			MEAN,
			CONST,
			NO
		};
		CenterMethod getCenterMethod(const DRIM2COLType method);
		double const_sub = 127.5;
		//template<int color_channels, int dim>
		void simdOMPCovFullCenterFullElement32F(const std::vector<cv::Mat>& src, cv::Mat& cov, const int border);
		void simdOMPCovFullCenterHalfElement32F(const std::vector<cv::Mat>& src, cv::Mat& cov, const int border);
		void simdOMPCovFullCenterRepElement32F(const std::vector<cv::Mat>& src, cv::Mat& cov, const int border);

		void simdOMPCovFullCenterHalfElementTEST32F(const std::vector<cv::Mat>& src, cv::Mat& cov, const int border);

		template<int color_channels, int patch_rad>
		void simdOMPCov_RepCenterHalfElement32F(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f);
		void simdOMPCov_RepCenterHalfElement32FCn(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f);
		template<int color_channels, int dim>
		void simdOMPCov_RepCenterHalfElement64F(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CenterMethod method, const double constant_sub = 127.5);
		void simdOMPCov_RepCenterHalfElement64FCn(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CenterMethod method, const double constant_sub = 127.5);

		template<int color_channels, int patch_rad>
		void simdOMPCov_RepCenterRepElement32F(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f);
		void simdOMPCov_RepCenterRepElement32FCn(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f);

		template<int color_channels, int patch_rad>
		void simdOMPCov_RepCenterConvElement32F(const std::vector<cv::Mat>& src, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f, const int border = cv::BORDER_WRAP);
		void simdOMPCov_RepCenterConvElement32FCn(const std::vector<cv::Mat>& src, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f, const int border = cv::BORDER_WRAP);
		template<int color_channels, int patch_rad>
		void simdOMPCov_RepCenterConvFElement32F(const std::vector<cv::Mat>& src, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f, const int border = cv::BORDER_WRAP);
		void simdOMPCov_RepCenterConvFElement32FCn(const std::vector<cv::Mat>& src, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f, const int border = cv::BORDER_WRAP);

		void covFFT(const std::vector<cv::Mat>& src, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f);
		void covFFTFull(const std::vector<cv::Mat>& src, cv::Mat& cov, const CenterMethod method, const float constant_sub = 127.5f);

		//separable
		void computeSeparateCov(const std::vector<cv::Mat>& src, const int patch_rad, const int borderType, std::vector<cv::Mat>& covmat);
		void computeSeparateCovXXt(const std::vector<cv::Mat>& src, const int patch_rad, const int borderType, std::vector<cv::Mat>& covmat);

#pragma region under_debug_funcions
		void simd_32FSkipCn(const std::vector<cv::Mat>& src, cv::Mat& cov, const int skip);
#pragma endregion
	};

	void imshowDRIM2COLEigenVec(std::string wname, cv::Mat& evec, const int channels);

	void DRIM2COLEigenVec(const cv::Mat& src, const cv::Mat& evec, cv::Mat& dest, const int r, const int channels, const int border = cv::BORDER_WRAP, const bool isParallel = false);
	void DRIM2COLEigenVecCn(const std::vector<cv::Mat>& src, const cv::Mat& evec, std::vector<cv::Mat>& dest, const int r, const int border);
	void DRIM2COLSepEigenVecCn(const std::vector<cv::Mat>& src, const cv::Mat& evecH, const cv::Mat& evecV, std::vector<cv::Mat>& dest, const int r, const int border);

	CP_EXPORT void DRIM2COL(const cv::Mat& src, cv::Mat& dst, const int neighborhood_r, const int dest_channels, const int border = cv::BORDER_WRAP, const int method = 0, const bool isParallel = false, const double const_sub = 127.5);
	CP_EXPORT void DRIM2COL(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const int neighborhood_r, const int dest_channels, const int border = cv::BORDER_WRAP, const int method = 0, const bool isParallel = false, const double const_sub = 127.5);

	CP_EXPORT void DRIM2COLTile(const cv::Mat& src, cv::Mat& dest, const int neighborhood_r, const int dest_channels, const int border, const int method, const cv::Size div);

	void reprojectDRIM2COL(const cv::Mat& src, cv::Mat& dest, const int neighborhood_r, const int dest_channels, const int border, const DRIM2COLType type, const bool isParallel, const double const_sub);
	void reprojectDRIM2COL(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, const int neighborhood_r, const int dest_channels, const int border, const DRIM2COLType type, const bool isParallel, const double const_sub);
	void reprojectIM2COL(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, const int neighborhood_r, const int border);
	//void reprojectNeighborhoodEigenVec(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, const cv::Mat& evec, const cv::Mat& evecInv, const int r, const int border);

	double getPSNRReprojectDRIM2COL(const cv::Mat& src, const int patch_radius, const int dest_channels, const int border, const DRIM2COLType type, const int bb, const bool isNormalizeDimension, const bool isParallel, const double const_sub);
	double getPSNRReprojectDRIM2COL(const std::vector<cv::Mat>& src, const int patch_radius, const int dest_channels, const int border, const DRIM2COLType type, const int bb, const bool isNormalizeDimension, const bool isParallel, const double const_sub);
	//double getPSNRReprojectDRIM2COLEigenVec(const std::vector<cv::Mat>& src, const cv::Mat& evec, const cv::Mat& evecInv, const int r, const int border, const int bb, const bool isNormalizeDimension);
	double getPSNRReprojectDRIM2COLEigenVec(const std::vector<cv::Mat>& src, const cv::Mat& evec, const int r, const int border, const int bb, const bool isNormalizeDimension);
	double getPSNRReprojectDRIM2COLSepEigenVec(const std::vector<cv::Mat>& vsrc, const cv::Mat& evecH, const cv::Mat& evecV, const int patch_radius, const int borderType, const int bb, const bool isNormalizeDimension);

	void patchPCADenoise(const cv::Mat& src, cv::Mat& dest, const int neighborhood_r, const int dest_channels, const float th, const int border, const int method, const bool isParallel);
	void patchPCADenoise(const std::vector<cv::Mat>& vsrc, std::vector<cv::Mat>& dest, const int neighborhood_r, const float th, const int dest_channels, const int border, const int method, const bool isParallel);

	double GrassmannDistance(const cv::Mat& src, const cv::Mat& ref);
	void computeCovRepresentive(const cv::Mat& src, const int patch_rad, cv::Mat& cov);
	void computeCovRelativePosition(const int r, std::vector<std::vector<int>>& RP_list);
	void separateEvecSVD(const cv::Mat& evec, const int iscolor, cv::Mat& dstEvec, cv::Mat& dstEvecH, cv::Mat& dstEvecV, int order = 1);
	void separateEvecSVD(const cv::Mat& evec, const int iscolor, const int dchannels, cv::Mat& dstEvec, cv::Mat& dstEvecH, cv::Mat& dstEvecV, int order = 1);
	void calcEvecfromSepCov(const cv::Mat& XXt, const cv::Mat& XtX, const int dchannel, cv::Mat& dstEvec, cv::Mat& dstEvecH, cv::Mat& dstEvecV);
	void colorPCA(const cv::Mat& src, cv::Mat& dst, cv::Mat& transmat);

	double getPSNRVector(std::vector<cv::Mat>& src, std::vector<cv::Mat>& ref);
}