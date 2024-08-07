#pragma once
#include <opencp.hpp>

static inline double getPyramidSigma(double sigma, double level)
{
	if (level == 0) return FLT_EPSILON;
	/*double ret = sqrt(2.0);

	for (int l = 0; l < level - 1; l++)
	{
		double v = pow(2.0, l + 1);
		ret += v;
	}
	return ret * sigma;*/

	double ret = 0.0;
	for (int l = 1; l <= level; l++)
	{
		double v = pow(2, l - 1);
		ret = ret + v * v;
	}
	return sigma * sqrt(ret);

}

enum WindowType
{
	GAUSS,
	LAPLACE,
	S_TONE,
	HAT,
	SMOOTH_HAT
};


static inline float getGaussWeight(int x, int y, float sigma)
{
	float ret;
	ret = (float)exp((x * x + y * y) / (-2.0 * sigma * sigma));
	return ret;
}

static inline float getSToneWeight(float x, float remap_sigma, float beta, float alpha)
{
	if (abs(x) <= remap_sigma)
	{
		return cp::sign(x) * remap_sigma * pow((abs(x) / remap_sigma), alpha);
		//return sign(x) * remap_sigma * sqrt(abs(x) / remap_sigma);
	}
	else
		return cp::sign(x) * (beta * (abs(x) - remap_sigma) + remap_sigma);
}

template<typename T>
static inline T getSToneCurve(T i, T g, T remap_sigma, T beta, T alpha)
{
	if (abs(i - g) <= remap_sigma)
	{
		//std::cout << "power : " << std::pow((abs(i - g) / remap_sigma), alpha) << std::endl;
		return g + cp::sign(i - g) * remap_sigma * (T)pow((abs(i - g) / remap_sigma), alpha);
		//return g + sign(i - g) * remap_sigma * sqrt(abs(i - g) / remap_sigma);
	}
	else
		return g + cp::sign(i - g) * (beta * (abs(i - g) - remap_sigma) + remap_sigma);
}



static inline float getSmoothingHat(float x, float t, float sigma, int r)
{
	float re = 0.f;
	for (int i = -r; i <= r; i++)
	{
		re += (x - t + i) * std::max(0.f, 1.f - abs((float)((x + i - t) / sigma)));
	}
	return re / (2 * r + 1);
}

static inline double getSmoothingHat(double x, double t, double sigma, int r)
{
	double re = 0.0;
	for (int i = -r; i <= r; i++)
	{
		re += (x - t + i) * std::max(0.0, 1.0 - abs((double)((x + i - t) / sigma)));
	}
	return re / (2 * r + 1);
}

static inline float getremapCoefficient(float i, float g, const int window_type, float sigma, float alpha, float beta, float remap_sigma, float boost)
{
	float ret = 0.f;
	switch (window_type)
	{
	case GAUSS:
		ret = i - float(boost * (i - g) * -exp((i - g) * (i - g) / (-2.0 * sigma * sigma)));
		break;
	case S_TONE:
		ret = getSToneCurve(i, g, remap_sigma, beta, alpha);
		break;
	case HAT:
		ret = i - (i - g) * -std::max(0.f, 1.f - abs((float)(i - g) / sigma));
		break;
	case SMOOTH_HAT:
		ret = i + getSmoothingHat(i, g, sigma, 10);
	}
	return ret;
}

namespace cp
{
	class CP_EXPORT ComputePyramidSize
	{
		const bool isLastConv;
		const int size;
		const int r;
		const int level;
		int minsize = 0;
		std::vector<int> borderSizeL;
		std::vector<int> borderSizeR;
		std::vector<int> imageSize;
		std::vector<int> borderSizeActive;
		std::vector<int> borderSizeRequire;
		std::vector<int> borderOffset;

	public:
		ComputePyramidSize(int size, int r, int level, bool isLastConv);
		void computeSize(int size, int r, int level, bool isLastConv);
		cv::Mat vizPyramid(int bs = 16);
		void print();
		void get(const int level, int& borderSizeL, int& borderSizeR, int& imageSize, int& borderSizeActive, int& borderSizeRequire, int& borderOffset);
		int getBorderL(int level);
		int getBorderR(int level);
		int getBorderOffset(int level);
		int getBorderSizeRequire(int level);
	};

	//abstract class for each multi-scale filter
	class CP_EXPORT MultiScaleFilter
	{
	public:
		enum AdaptiveMethod
		{
			FIX,
			ADAPTIVE,
		};
		enum ScaleSpace
		{
			Pyramid,
			DoG,
			ContrastPyramid,
			CoG,
			DoGSep
		};
		enum RangeDescopeMethod
		{
			FULL,
			MINMAX,
			LOCAL,
		};
		enum PyramidComputeMethod
		{
			IgnoreBoundary,
			Fast,
			Full,
			OpenCV
		};

		void setAdaptive(const bool adaptiveMethod, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap, const int level);
		std::string getAdaptiveName();
		std::string getScaleSpaceName();

		void setPyramidComputeMethod(const PyramidComputeMethod scaleSpaceMethod);
		std::string getPyramidComputeName();

		void setRangeDescopeMethod(RangeDescopeMethod scaleSpaceMethod);
		std::string getRangeDescopeMethod();
		cv::Size getLayerSize(const int level);

		bool isDoGPyramidApprox = false;
		void setDoGPyramidApprox(bool flag);
		void setIsCompute(bool flag);
		int getGaussianRadius(const float sigma);

		void showImageStack(std::string wname);
		static void showPyramid(std::string wname, std::vector<cv::Mat>& pyramid, float scale = 1.f, bool isShowLevel = true);
		void drawRemap(bool isWait = true, const cv::Size size = cv::Size(512, 512));

		const float nsigma = 3.f;//3.f;
		void buildGaussianPyramid(const cv::Mat& src, std::vector<cv::Mat>& GaussianPyramid, const int level, const float sigma);
		void buildGaussianLaplacianPyramid(const cv::Mat& src, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& LaplacianPyramid, const int level, const float sigma);
		void buildLaplacianPyramid(const cv::Mat& src, std::vector<cv::Mat>& LaplacianPyramid, const int level, const float sigma);
		void buildContrastPyramid(const cv::Mat& src, std::vector<cv::Mat>& LaplacianPyramid, const int level, const float sigma);

		template<int D, int d, int d2> void buildLaplacianPyramid(const cv::Mat& src, std::vector<cv::Mat>& LaplacianPyramid, const int level, const float sigma, float* linebuff);
		//using precomputed Gaussian pyramid
		void buildLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int level, const float sigma);
		//L0+...resize(Ln-2+resize(Ln-1+resize(Ln)))
		void collapseLaplacianPyramid(std::vector<cv::Mat>& LaplacianPyramid, cv::Mat& dest, const int depth);
		void collapseContrastPyramid(std::vector<cv::Mat>& LaplacianPyramid, cv::Mat& dest, const int depth);

		void buildGaussianStack(const cv::Mat& src, std::vector<cv::Mat>& GaussianStack, const float sigma_s, const int level);
		void buildDoGStack(const cv::Mat& src, std::vector<cv::Mat>& ImageStack, const float sigma_s, const int level);
		void buildCoGStack(const cv::Mat& src, std::vector<cv::Mat>& ImageStack, const float sigma_s, const int level);
		void buildDoGSeparableStack(const cv::Mat& src, std::vector<cv::Mat>& ImageStack, const float sigma_s, const int level);
		void collapseDoGStack(std::vector<cv::Mat>& ImageStack, cv::Mat& dest, const int depth);
		void collapseCoGStack(std::vector<cv::Mat>& ImageStack, cv::Mat& dest, const int depth);
	protected:
		bool isCompute = true;
		std::vector<cv::Size> layerSize;
		void allocSpaceWeight(const float sigma);
		void freeSpaceWeight();
		float evenratio = 0.f;//set in generateWeight
		float oddratio = 0.f;//set in generateWeight
		float* GaussWeight = nullptr;
		RangeDescopeMethod rangeDescopeMethod = RangeDescopeMethod::MINMAX;
		int radius = 0;//set in allocSpaceWeight
		PyramidComputeMethod pyramidComputeMethod = IgnoreBoundary;

		const int threadMax = omp_get_max_threads();

		ScaleSpace scalespaceMethod = ScaleSpace::Pyramid;
		const int borderType = cv::BORDER_DEFAULT;
		float sigma_range = 0.f;
		float sigma_space = 0.f;
		float boost = 1.f;
		int level = 0;
		AdaptiveMethod adaptiveMethod = AdaptiveMethod::FIX;
		std::vector<cv::Mat> adaptiveSigmaMap;
		std::vector<cv::Mat> adaptiveBoostMap;
		std::vector<cv::Mat> adaptiveSigmaBorder;//level+1
		std::vector<cv::Mat> adaptiveBoostBorder;//level+1
		float Salpha = 1.f;
		float Sbeta = 1.f;
		const int windowType = 0;

		std::vector<cv::Mat> ImageStack;
		float intensityMin = 0.f;
		float intensityMax = 255.f;
		float intensityRange = 255.f;
		const int rangeMax = 256;
		std::vector<float> rangeTable;

		inline float getGaussianRangeWeight(const float v, const float sigma_range, const float boost);
		void initRangeTable(const float sigma, const float boost);
		void remap(const cv::Mat& src, cv::Mat& dest, const float g, const float sigma_range, const float boost);
		void remapAdaptive(const cv::Mat& src, cv::Mat& dest, const float g, const cv::Mat& sigma_range, const cv::Mat& boost);

		void rangeDescope(const cv::Mat& src);

		float* generateGaussianWeight(int r, const float sigma, float& evenratio, float& oddratio);
		void GaussDownFull(const cv::Mat& src, cv::Mat& dest, const float sigma, const int borderType);
		void GaussDown(const cv::Mat& src, cv::Mat& dest);
		template<int D> void GaussDown(const cv::Mat& src, cv::Mat& dest, float* linebuff);//linebuffsize = src.cols+2*radius
		void GaussDownIgnoreBoundary(const cv::Mat& src, cv::Mat& dest);
		template<int D> void GaussDownIgnoreBoundary(const cv::Mat& src, cv::Mat& dest, float* linebuff);//linebuffsize = src.cols

		void GaussUpFull(const cv::Mat& src, cv::Mat& dest, const float sigma, const int borderType);
		void GaussUp(const cv::Mat& src, cv::Mat& dest);
		void GaussUpIgnoreBoundary(const cv::Mat& src, cv::Mat& dest);

		//dest = addsubsrc +/- srcup
		template<bool isAdd> void GaussUpAddFull(const cv::Mat& src, const cv::Mat& addsubsrc, cv::Mat& dest, const float sigma, const int borderType);
		//dest = addsubsrc +/- srcup
		template<bool isAdd> void GaussUpAdd(const cv::Mat& src, const cv::Mat& addsubsrc, cv::Mat& dest);
		template<bool isAdd, int D, int D2> void GaussUpAdd(const cv::Mat& src, const cv::Mat& addsubsrc, cv::Mat& dest);
		//dest = addsubsrc +/- srcup
		template<bool isAdd> void GaussUpAddIgnoreBoundary(const cv::Mat& src, const cv::Mat& addsubsrc, cv::Mat& dest);
		template<bool isAdd, int D2> void GaussUpAddIgnoreBoundary(const cv::Mat& src, const cv::Mat& addsubsrc, cv::Mat& dest, float* linee, float* lineo);//linebuffsize = src.cols

		void body(const cv::Mat& src, cv::Mat& dest);
		void gray(const cv::Mat& src, cv::Mat& dest);
		void body(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest);
		void gray(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest);

		virtual void pyramid(const cv::Mat& src, cv::Mat& dest) = 0;
		virtual void dog(const cv::Mat& src, cv::Mat& dest) = 0;
		virtual void contrastpyramid(const cv::Mat& src, cv::Mat& dest) {};
		virtual void cog(const cv::Mat& src, cv::Mat& dest) {};
		virtual void dogsep(const cv::Mat& src, cv::Mat& dest) {};

		virtual void pyramid(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest) {};
		virtual void dog(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest) {};
		virtual void contrastpyramid(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest) {};
		virtual void cog(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest) {};
		virtual void dogsep(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest) {};
	};

	//classical multi-scale filtering (no edge-preserving filter)
	class CP_EXPORT MultiScaleGaussianFilter : public MultiScaleFilter
	{
	public:
		void filter(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);
	protected:
		void pyramid(const cv::Mat& src, cv::Mat& dest) override;
		void contrastpyramid(const cv::Mat& src, cv::Mat& dest) override;
		void dog(const cv::Mat& src, cv::Mat& dest) override;
		void cog(const cv::Mat& src, cv::Mat& dest) override;
		void dogsep(const cv::Mat& src, cv::Mat& dest) override;
	};

	//multi-scale filtering with bilateral filtering-based pyramid
	class CP_EXPORT MultiScaleBilateralFilter : public MultiScaleFilter
	{
	public:
		void filter(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);
	protected:
		void pyramid(const cv::Mat& src, cv::Mat& dest) override;
		void dog(const cv::Mat& src, cv::Mat& dest) override;
		void cog(const cv::Mat& src, cv::Mat& dest) override;
		void buildDoBFStack(const cv::Mat& src, std::vector<cv::Mat>& DoBFStack, const float sigma_r, const float sigma_s, const int level);
		void buildCoBFStack(const cv::Mat& src, std::vector<cv::Mat>& DoBFStack, const float sigma_r, const float sigma_s, const int level);
	};

	//multi-scale filtering with guided image filtering-based pyramid
	class CP_EXPORT MultiScaleGuidedImageFilter : public MultiScaleFilter
	{
	public:
		void filter(const cv::Mat& src, cv::Mat& dest, const float eps, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);
	protected:
		void pyramid(const cv::Mat& src, cv::Mat& dest)override;
		void dog(const cv::Mat& src, cv::Mat& dest)override;
		void cog(const cv::Mat& src, cv::Mat& dest)override;
		//for compatible eps = sqrt(sigma_r/1.5), sigma_gif = sqrt(2)*sigma_bf, radius_gif = radius_bf*2
		void buildDoGIFStack(const cv::Mat& src, std::vector<cv::Mat>& DoBFStack, const float eps, const float sigma_s, const int level);
		void buildCoGIFStack(const cv::Mat& src, std::vector<cv::Mat>& DoBFStack, const float eps, const float sigma_s, const int level);
	};


	//build scale-space per image (very slow)
	class CP_EXPORT LocalMultiScaleFilterFull : public MultiScaleFilter
	{
	public:
		void filter(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);
		void jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);
	protected:
		void pyramid(const cv::Mat& src, cv::Mat& dest) override;
		void dog(const cv::Mat& src, cv::Mat& dest) override;
		void dog(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest) override;

		void setDoGKernel(float* weight, int* index, const int index_step, cv::Size ksize, const float sigma1, const float sigma2);
		float getDoGCoeffLnNoremap(cv::Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight);
		float getDoGCoeffLn(const cv::Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight);
		float getDoGCoeffLn(const cv::Mat& src, const cv::Mat& guide, const float g, const float h, const int y, const int x, const int size, int* index, float* weight);
	};

	//build scale-space per block (same accuracy as full, pyramid: LLF naive implementation, dog: same as full)
	class CP_EXPORT LocalMultiScaleFilter : public MultiScaleFilter
	{
	public:
		void filter(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);
	protected:
		bool isCompute = true;
		cv::Mat border;
		std::vector<cv::Mat> LaplacianPyramid;
		//Local Laplacian Filter(Paris2011)
		void pyramid(const cv::Mat& src, cv::Mat& dest)override;
		// DoG
		void dog(const cv::Mat& src, cv::Mat& dest)override;
		void dog(const cv::Mat& src, cv::Mat& guide, cv::Mat& dest);
		void setDoGKernel(float* weight, int* index, const int index_step, cv::Size ksize1, cv::Size ksize2, const float sigma1, const float sigma2);
		float getRemapDoGConv(const cv::Mat& src, const float g, const int y, const int x, const int size, int* index, float* weight, bool isCompute);
	};

	//reference implementation: not optimized
	class CP_EXPORT FastLLFReference : public MultiScaleFilter
	{
	public:
		//interpolationMethod: {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC}
		void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid, const int interpolationMethod = cv::INTER_LINEAR);
	protected:
		std::vector<cv::Mat> GaussianPyramid;
		std::vector<std::vector<cv::Mat>> LaplacianPyramidOrder;
		std::vector<cv::Mat> LaplacianPyramid;

		cv::Mat border;
		int order = 0;

		float getTau(const int k);

		////fast Local Laplacian Filter(2014)
		void pyramid(const cv::Mat& src, cv::Mat& dest) override;
		void dog(const cv::Mat& src, cv::Mat& dest) override { ; };

		//last level is not blended; thus, inplace operation for input Gaussian Pyramid is required.
		void blendLaplacianNearest(const std::vector<std::vector<cv::Mat>>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order);

		inline void getLinearIndex(float v, int& index_l, int& index_h, float& alpha, const int order, const float intensityMin, const float intensityMax)
		{
			const float intensityRange = intensityMax - intensityMin;
			const float delta = intensityRange / (order - 1);
			const int i = (int)(v / delta);
			/*cout << step << endl;
			cout << i * step << ":" << v << endl;*/
			//cout << "check:" << (v - (i * step)) / step << endl;

			if (i < 0)
			{
				//cout << "-sign" << endl;
				index_l = 0;
				index_h = 0;
				alpha = 1.f;
			}
			else if (i + 1 > order - 1)
			{
				//cout << "-sign" << endl;
				index_l = order - 1;
				index_h = order - 1;
				alpha = 0.f;
			}
			else
			{
				index_l = i;
				index_h = i + 1;
				alpha = 1.f - (v - (i * delta)) / (delta);
				//const float vv = (i * step) * alpha + ((i + 1) * step) * (1.0 - alpha);
				//print_debug5(v, i * step, step, alpha,vv);
			}
		}
		//do not handle last level
		void blendLaplacianLinear(const std::vector<std::vector<cv::Mat>>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order);
	};

	class CP_EXPORT LocalMultiScaleFilterInterpolation : public MultiScaleFilter
	{
	public:
		//interpolationMethod: {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC}
		void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid, const int interpolationMethod = cv::INTER_LINEAR);
		void setCubicAlpha(const float alpha);
		void setIsParallel(const bool flag);
		void setComputeScheduleMethod(const bool useTable);
		std::string getComputeScheduleName();
	private:
		bool isUseTable = true;
		float cubicAlpha = -0.5f;

		std::vector<cv::Mat> remapIm;
		std::vector<cv::Mat> GaussianStack;
		std::vector<std::vector<cv::Mat>> LayerStack;

		cv::Mat border;
		int interpolation_method = 0;
		int order = 0;
		bool isParallel = true;
		float getTau(const int k);
		int tableSize = 0;
		float* integerSampleTable = nullptr;
		void initRangeTableInteger(const float sigma, const float boost);
		////fast Local Laplacian Filter(2014)
		void pyramidParallel(const cv::Mat& src, cv::Mat& dest);
		void pyramidSerial(const cv::Mat& src, cv::Mat& dest);
		void pyramid(const cv::Mat& src, cv::Mat& dest) override;
		void dog(const cv::Mat& src, cv::Mat& dest) override;
		void cog(const cv::Mat& src, cv::Mat& dest) override;
		void dogsep(const cv::Mat& src, cv::Mat& dest) override;

		template<bool isUseTable>
		void remapGaussDownIgnoreBoundary(const cv::Mat& src, cv::Mat& remapIm, cv::Mat& dest, const float g, const float sigma_range, const float boost);
		template<bool isUseTable, int D>
		void remapGaussDownIgnoreBoundary(const cv::Mat& src, cv::Mat& remapIm, cv::Mat& dest, const float g, const float sigma_range, const float boost);

		void remapAdaptiveGaussDownIgnoreBoundary(const cv::Mat& src, cv::Mat& remapIm, cv::Mat& dest, const float g, const cv::Mat& sigma_range, const cv::Mat& boost);

		template<bool isInit, int interpolation>
		void GaussUpSubProductSumIgnoreBoundary(const cv::Mat& src, const cv::Mat& addsubsrc, const cv::Mat& GaussianPyramid, cv::Mat& dest, const float g);

		template<bool isInit, int interpolation, int D2>
		void GaussUpSubProductSumIgnoreBoundary(const cv::Mat& src, const cv::Mat& addsubsrc, const cv::Mat& GaussianPyramid, cv::Mat& dest, const float g);

		//for parallel and serial test
		void buildRemapLaplacianPyramidEachOrder(const cv::Mat& src, std::vector<cv::Mat>& destPyramid, const int level, const float sigma, const float g, const float sigma_range, const float boost);

		inline void getLinearIndex(float v, int& index_l, int& index_h, float& alpha, const int order, const float intensityMin, const float intensityMax)
		{
			const float intensityRange = intensityMax - intensityMin;
			const float delta = intensityRange / (order - 1);
			const int i = (int)(v / delta);
			/*cout << step << endl;
			cout << i * step << ":" << v << endl;*/
			//cout << "check:" << (v - (i * step)) / step << endl;

			if (i < 0)
			{
				//cout << "-sign" << endl;
				index_l = 0;
				index_h = 0;
				alpha = 1.f;
			}
			else if (i + 1 > order - 1)
			{
				//cout << "-sign" << endl;
				index_l = order - 1;
				index_h = order - 1;
				alpha = 0.f;
			}
			else
			{
				index_l = i;
				index_h = i + 1;
				alpha = 1.f - (v - (i * delta)) / (delta);
				//const float vv = (i * step) * alpha + ((i + 1) * step) * (1.0 - alpha);
				//print_debug5(v, i * step, step, alpha,vv);
			}
		}
		inline float getCubicCoeff(const float xn, const float a = -0.5f)
		{
			const float d = abs(xn);

			float ret = 0.f;
			if (d < 1.f) ret = (a + 2.f) * d * d * d - (a + 3.f) * d * d + 1.f;
			else if (d < 2.f) ret = a * d * d * d - 5.f * a * d * d + 8.f * a * d - 4.f * a;

			return ret;
		}
		inline float getCubicInterpolation(const float v, const int order, const float** lptr, const int idx, const float cubicAlpha, const float intensityMin, const float intensityMax)
		{
			const float intensityRange = intensityMax - intensityMin;
			const float vv = std::max(intensityMin, std::min(intensityMax, v));
			const float idelta = (order - 1) / intensityRange;
			const int i = (int)(vv * idelta);

			float ret = 0.f;
			if (i == 0 || i + 1 >= order - 1)//linear interpolation
			{

				int l, h;
				float alpha;
				getLinearIndex(vv, l, h, alpha, order, intensityMin, intensityMax);
				ret = alpha * lptr[l][idx] + (1.f - alpha) * lptr[h][idx];
				/*const float istep = vv * idelta - i;
				ret = getCubicCoeff(istep + 1.f, cubicAlpha) * lptr[i - 0][idx]
					+ getCubicCoeff(istep + 0.f, cubicAlpha) * lptr[i + 0][idx]
					+ getCubicCoeff(istep - 1.f, cubicAlpha) * lptr[i + 1][idx]
					+ getCubicCoeff(istep - 2.f, cubicAlpha) * lptr[i + 1][idx];*/
			}
			else
			{
				const float istep = vv * idelta - i;
				ret = getCubicCoeff(istep + 1.f, cubicAlpha) * lptr[i - 1][idx]
					+ getCubicCoeff(istep + 0.f, cubicAlpha) * lptr[i + 0][idx]
					+ getCubicCoeff(istep - 1.f, cubicAlpha) * lptr[i + 1][idx]
					+ getCubicCoeff(istep - 2.f, cubicAlpha) * lptr[i + 2][idx];
			}
			return ret;
		}
		//do not handle lastlevel
		void blendDetailStack(const std::vector<std::vector<cv::Mat>>& detailStack, const std::vector<cv::Mat>& approxStack, std::vector<cv::Mat>& destStack, const int order, const int interpolationMethod);

		//for serial
		template<bool isInit>
		void buildRemapLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& LaplacianPyramid, std::vector<cv::Mat>& destPyramid, const int level, const float sigma, const float g, const float sigma_range, const float boost);
		template<bool isInit>
		void buildRemapAdaptiveLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& LaplacianPyramid, std::vector<cv::Mat>& destPyramid, const int level, const float sigma, const float g, const cv::Mat& sigma_range, const cv::Mat& boost);
		//for serial test
		template<int interpolation>
		void productSumLaplacianPyramid(const std::vector<cv::Mat>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order, const float g);
	};

	//reference implementation: not optimized
	class CP_EXPORT LocalMultiScaleFilterFourierReference : public MultiScaleFilter
	{
	public:
		void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);

	private:
		const bool isSIMD = false;
		int order = 0;
		float T = 0.f;
		std::vector<float> alpha, beta;
		std::vector<float> omega;//(CV_2PI*(k + 1)/T)

		std::vector<cv::Mat> FourierPyramidSin; //level+1
		std::vector<cv::Mat> FourierPyramidCos; //level+1
		std::vector<cv::Mat> LaplacianPyramid; //level+1
		std::vector<cv::Mat> GaussianPyramid; //level+1
		void initRangeFourier(const int order, const float sigma_range, const float boost);

		void remapCos(const cv::Mat& src, cv::Mat& dest, const float omega);
		void remapSin(const cv::Mat& src, cv::Mat& dest, const float omega);
		void productSumPyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const float sigma, const float boost);
		void productSumAdaptivePyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const cv::Mat& sigma, const cv::Mat& boost);

		void pyramid(const cv::Mat& src, cv::Mat& dest)override;
		void dog(const cv::Mat& src, cv::Mat& dest) override { ; };
	};

	class CP_EXPORT LocalMultiScaleFilterFourier : public MultiScaleFilter
	{
	public:
		~LocalMultiScaleFilterFourier();
		float KernelError = 0.f;

		void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost = 1.f, const int level = 2, const ScaleSpace scaleSpaceMethod = ScaleSpace::Pyramid);

		enum Period
		{
			GAUSS_DIFF,
			OPTIMIZE,
			PRE_SET
		};

		enum
		{
			MergeFourier,
			SplitFourier
		};

		void setIsPlot(const bool flag);
		void setIsParallel(const bool flag);
		void setPeriodMethod(Period scaleSpaceMethod);
		void setComputeScheduleMethod(int schedule = MergeFourier, bool useTable0 = true, bool useTableLevel = false);
		std::string getComputeScheduleName();

		std::string getPeriodName();
	private:
		int kernel_plotting_t = 128;
		int kernel_plotting_amp = 0;
		bool isPlotted = false;//for destroy window flag
		template<typename Type>
		void kernelPlot(const int window_type, const int order, const int R, const double boost, const double sigma_range, float Salpha, float Sbeta, float Ssigma, const int Imin, const int Imax, const int Irange, const Type T,
			Type* sinTable, Type* cosTable, std::vector<Type>& alpha, std::vector<Type>& beta, int windowType, const std::string wname = "plt f(x)", const cv::Size windowSize = cv::Size(512, 512));

		bool isParallel = true;
		bool isPlot = false;

		int preorder = 0;
		float presigma_range = 0.f;
		float predetail_param = 0.f;
		float preIntensityMin = 0.f;
		float preIntensityRange = 255.f;
		Period preperiodMethod = GAUSS_DIFF;

		int order = 0;
		float T = 0.f;
		//int PeriodMethod = OPTIMIZE;
		Period periodMethod = GAUSS_DIFF;
		std::vector<float> alpha, beta;
		std::vector<float> omega;//(CV_2PI*(k + 1)/T)

		int computeScheduleFourier = MergeFourier;//MergeSINCOS
		//const int computeScheduleFourier = SplitSINCOS;//SplitSINCOS
		bool isUseFourierTable0 = true;
		bool isUseFourierTableLevel = false;

		const int FourierTableSize = 256;
		float* sinTable = nullptr;//FourierTableSize*order initialized in initRangeFourier
		float* cosTable = nullptr;//FourierTableSize*order initialized in initRangeFourier
		cv::Mat src8u;//used for isUseSplatTable case

		std::vector<std::vector<cv::Mat>> FourierStackSin; //[k][l] max(order,threadMax) x (level + 1)
		std::vector<std::vector<cv::Mat>> FourierStackCos; //[k][l] max(order,threadMax) x (level + 1)
		std::vector<cv::Mat> DetailStack;//level+1
		std::vector<std::vector<cv::Mat>> destEachOrder;//[k][l] max(order,threadMax) x level
		//dest: T, alpha, omega, (optional: sin-cos table)
		void initRangeFourier(const int order, const float sigma_range, const float boost);
		void allocImageBuffer(const int order, const int level);

		// fastest
		//unroll cos sin
		template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseProductTable, int D, int D2>
		void buildLaplacianFourierPyramidIgnoreBoundary(const std::vector<cv::Mat>& GaussianPyramid, const cv::Mat& src8u, std::vector<cv::Mat>& destPyramid, const int k, const int level, std::vector<cv::Mat>& destSplatPyramidCos, std::vector<cv::Mat>& destSplatPyramidSin);
		template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseProductTable>
		void buildLaplacianFourierPyramidIgnoreBoundary(const std::vector<cv::Mat>& GaussianPyramid, const cv::Mat& src8u, std::vector<cv::Mat>& destPyramid, const int k, const int level, std::vector<cv::Mat>& destSplatPyramidCos, std::vector<cv::Mat>& destSplatPyramidSin);
		//split cos sin
		template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel, int D, int D2>
		void buildLaplacianCosPyramidIgnoreBoundary(const std::vector<cv::Mat>& GaussianPyramid, const cv::Mat& src8u, std::vector<cv::Mat>& destPyramid, const int k, const int level, std::vector<cv::Mat>& destSplatPyramidCos);
		template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel>
		void buildLaplacianCosPyramidIgnoreBoundary(const std::vector<cv::Mat>& GaussianPyramid, const cv::Mat& src8u, std::vector<cv::Mat>& destPyramid, const int k, const int level, std::vector<cv::Mat>& destSplatPyramidCos);
		template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel, int D, int D2>
		void buildLaplacianSinPyramidIgnoreBoundary(const std::vector<cv::Mat>& GaussianPyramid, const cv::Mat& src8u, std::vector<cv::Mat>& destPyramid, const int k, const int level, std::vector<cv::Mat>& destSplatPyramidSin);
		template<bool isInit, bool adaptiveMethod, bool isUseFourierTable0, bool isUseFourierTableLevel>
		void buildLaplacianSinPyramidIgnoreBoundary(const std::vector<cv::Mat>& GaussianPyramid, const cv::Mat& src8u, std::vector<cv::Mat>& destPyramid, const int k, const int level, std::vector<cv::Mat>& destSplatPyramidSin);

		//without last level summing
		void sumPyramid(const std::vector<std::vector<cv::Mat>>& orderPyramid, std::vector<cv::Mat>& destPyramid, const int order, const int level, std::vector<bool>& used);

		void pyramid(const cv::Mat& src, cv::Mat& dest) override;
		void pyramidParallel(const cv::Mat& src, cv::Mat& dest);
		void pyramidSerial(const cv::Mat& src, cv::Mat& dest);
		void dog(const cv::Mat& src, cv::Mat& dest) override;
		void cog(const cv::Mat& src, cv::Mat& dest) override;

#pragma region DoG
		void remapCosSin(const cv::Mat& src, int k, cv::Mat& destCos, cv::Mat& destSin, bool isCompute);
		//makeDoG and then product sum
		void productSummingTrig(const std::vector <cv::Mat>& src, std::vector <cv::Mat>& dest, float sigma_range, bool isCompute);
#pragma endregion
	};


	class CP_EXPORT TileLocalMultiScaleFilterInterpolation :public cp::TileParallelBody
	{
		LocalMultiScaleFilterInterpolation* msf;
		int order = 0;
		float sigma_range = 0.f;
		float sigma_space = 0.f;
		float boost = 0.f;
		int level = 0;
		int interpolation = cv::INTER_LINEAR;
		MultiScaleFilter::ScaleSpace scaleSpaceMethod = MultiScaleFilter::ScaleSpace::Pyramid;
	public:
		TileLocalMultiScaleFilterInterpolation();
		~TileLocalMultiScaleFilterInterpolation();

		void setComputeScheduleMethod(const bool useTable);
		std::string getComputeScheduleName();
		void setAdaptive(const bool flag, const cv::Size div, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap);
		void setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod scaleSpaceMethod);
		void setCubicAlpha(const float alpha);
		void process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex);
		void filter(const cv::Size div, const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const MultiScaleFilter::ScaleSpace scaleSpaceMethod, int interpolation);
	};

	class CP_EXPORT TileLocalMultiScaleFilterFourier :public cp::TileParallelBody
	{
		LocalMultiScaleFilterFourier* msf;
		int order = 0;
		float sigma_range = 0.f;
		float sigma_space = 0.f;
		float boost = 0.f;
		int level = 0;
		MultiScaleFilter::ScaleSpace scaleSpaceMethod = MultiScaleFilter::ScaleSpace::Pyramid;
	public:
		TileLocalMultiScaleFilterFourier();
		~TileLocalMultiScaleFilterFourier();

		void setComputeScheduleMethod(int schedule, bool useTable0, bool useTableLevel);
		std::string getComputeScheduleName();
		void setAdaptive(const bool flag, const cv::Size div, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap);
		void setPeriodMethod(int method);

		void setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod scaleSpaceMethod);
		void process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex);
		void filter(const cv::Size div, const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const MultiScaleFilter::ScaleSpace scaleSpaceMethod);
	};

	//alias: LocalMultiScaleFilterFull
	CP_EXPORT void localLaplacianFilterFull(cv::InputArray src, cv::OutputArray dest, const float sigma_range, const float sigma_space, const float boost, const int level);
	//alias: LocalMultiScaleFilter with pyramid
	CP_EXPORT void localLaplacianFilter(cv::InputArray src, cv::OutputArray dest, const float sigma_range, const float sigma_space, const float boost, const int level);
	//alias: LocalMultiScaleFilter with DoG
	CP_EXPORT void localDoGFilter(cv::InputArray src, cv::OutputArray dest, const float sigma_range, const float sigma_space, const float boost, const int level);
	//alias: LocalMultiScaleFilterInterpolation with pyramid
	CP_EXPORT void fastLocalLaplacianFilter(cv::InputArray src, cv::OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level);
	//alias: LocalMultiScaleFilterInterpolation with DoG
	CP_EXPORT void fastLocalDoGFilter(cv::InputArray src, cv::OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level);
	//alias: LocalMultiScaleFilterFourier with pyramid
	CP_EXPORT void FourierLocalLaplacianFilter(cv::InputArray src, cv::OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level);
	//alias: LocalMultiScaleFilterFourier with DoG
	CP_EXPORT void FourierLocalDoGFilter(cv::InputArray src, cv::OutputArray dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level);
}