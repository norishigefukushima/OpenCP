#pragma once
#include "common.hpp"
#include "dithering.hpp"

namespace cp
{
	enum
	{
		IMAGE_TEXTURNESS_FLOYD_STEINBERG,
		IMAGE_TEXTURENESS_OSTRO,
		IMAGE_TEXTURENESS_SIERRA2,
		IMAGE_TEXTURENESS_SIERRA3,
		IMAGE_TEXTURENESS_JARVIS,
		IMAGE_TEXTURENESS_STUCKI,
		IMAGE_TEXTURENESS_BURKES,
		IMAGE_TEXTURENESS_STEAVENSON,
		IMAGE_DEPTH,
		IMAGE_FLAT_FLOYD_STEINBERG,
		IMAGE_FLAT_OSTRO,
		IMAGE_FLAT_SIERRA2,
		IMAGE_FLAT_SIERRA3,
		IMAGE_FLAT_JARVIS,
		IMAGE_FLAT_STUCKI,
		IMAGE_FLAT_BURKES,
		IMAGE_FLAT_STEAVENSON,
		IMAGE_AREA_MONTECARLO,
		IMAGE_BLUENOISE,
		IMAGE_TEXTURENESS_AFTER_BF,

		NUMBER_OF_IMAGE_SAMPLINGMETHODS,
	};

	CP_EXPORT std::string getImageSamplingMethodName(const int sample_method);


	//maskoption
	enum DITHER_POSTPROCESS
	{
		NO_POSTPROCESS,
		FlipBottomCopy,
		FlipTopCopy,
		RANDOM_ROTATION,
	};

	CP_EXPORT std::string getDitheringPostProcessName(const int method);

	class CP_EXPORT IntensityRemappedDither
	{
		float scale = 0.f;
		float sampling_ratio;
		const int n;
		const int dither_method;
		const int dither_scanorder;
		const int dither_postprocess;

		float compute_s(const cv::Mat& src);
		int body(const cv::Mat& src, cv::Mat& dest);
		cv::Mat imagebuff;
	public:
		void remap(const cv::Mat& src, cv::Mat& dest);
		IntensityRemappedDither(cv::Size image_size, const float sampling_ratio, const int dither_method = cp::DITHER_METHOD::OSTROMOUKHOW, const int dither_scanorder = cp::DITHER_SCANORDER::MEANDERING, const int dither_postprocess = DITHER_POSTPROCESS::RANDOM_ROTATION);
		int generate(const cv::Mat& src, cv::Mat& destMask, const float ratio);
	};

	CP_EXPORT int generateSamplingMaskRemappedDitherWeight(const cv::Mat& weight, cv::Mat& dest, const float sampling_ratio, const int dithering_method, const int dithering_order, const float bin_ratio = 0.1f, const int maskOption = FlipBottomCopy);

	CP_EXPORT void generateSamplingMaskRemappedDitherFlat(cv::RNG& rng, cv::Mat& mask, int& sample_num, const float sampling_ratio, int dithering_method, const bool isCircle);
	CP_EXPORT void generateSamplingMaskRemappedDitherGaussian(cv::RNG& rng, cv::Mat& mask, int& sample_num, const float sampling_ratio, int dithering_method, int dithering_order, const float sigma);
	CP_EXPORT void generateSamplingMaskRemappedTextureness(const cv::Mat& src, cv::Mat& mask, int& sample_num, float sampling_ratio, int ditheringMethod, float bin_ratio = 0.1f);
	CP_EXPORT void generateSamplingMaskRemappedDitherDepthSigma(const cv::Mat& depthmap, cv::Mat& mask, const int inforcus_disp, const float sigma_base, const float inc_sigma, int& sample_num, float sampling_ratio);

	CP_EXPORT void createImportanceMapFlatBlueNoise(cv::Mat& dest, int& sample_num, float sampling_ratio);
	//void createSamplingOffset(cv::Mat& importanceMap, int*& importance_ofs, int*& importance_ofs_store, int& sample_num, const int r, const int src_step, bool is_AVX_padding);

	//packed
	CP_EXPORT void generateSamplingMaskRemappedDitherTexturenessPackedAoS(cv::Mat& src, cv::Mat& dest, const float sampling_ratio, int ditheringMethod = 1);
	CP_EXPORT void generateSamplingMaskRemappedDitherTexturenessPackedSoA(std::vector<cv::Mat>& src, cv::Mat& dest, const float sampling_ratio, const bool isUseAverage, int ditheringMethod = 1);
	CP_EXPORT void generateSamplingMaskRemappedDitherTexturenessPackedSoA(std::vector<cv::Mat>& src, std::vector<cv::Mat>& guide, cv::Mat& dest, const float sampling_ratio, int ditheringMethod = 1);
}