#pragma once
#include <omp.h>
#include <opencp.hpp>
#include <spatialfilter/SpatialFilter.hpp>

namespace cp
{
	inline int rate2scale(const double rate)
	{
		int ret = 1;
		const int scalemax = 10;
		for (int scale = 2; scale < scalemax; scale++)
		{
			if (rate > 1.0 / (scale * scale))
			{
				ret = scale - 1;
				break;
			}
		}
		return ret;
	}

	enum class ClusterMethod
	{
		random_sample,
		K_means,
		K_means_pp,
		K_means_fast,
		K_means_pp_fast,
		K_means_mspp_fast,
		KGaussInvMeansPPFast,

		mediancut_median,
		mediancut_max,
		mediancut_min,

		//debug
		quantize_wan,
		kmeans_wan,
		quantize_wu,
		kmeans_wu,
		quantize_neural,
		kmeans_neural,

		quantize_DIV,
		kmeans_DIV,
		quantize_PNN,
		kmeans_PNN,
		quantize_SPA,
		kmeans_SPA,
		quantize_EAS,
		kmeans_EAS,

		initial_centers,
		X_means,
		Size
	};
	inline std::string getClusterMethodName(ClusterMethod method)
	{
		std::string ret = "";
		switch (method)
		{
		case ClusterMethod::random_sample:			ret = "random_sample"; break;
		case ClusterMethod::K_means:				ret = "Kmeans"; break;
		case ClusterMethod::K_means_pp:				ret = "Kmeans_pp";	break;
		case ClusterMethod::K_means_fast:			ret = "Kmeans_fast"; break;
		case ClusterMethod::K_means_pp_fast:		ret = "Kmeans_pp_fast"; break;
		case ClusterMethod::K_means_mspp_fast:		ret = "K_means_mspp_fast"; break;
		case ClusterMethod::KGaussInvMeansPPFast:	ret = "KGaussInvMeansPPFast"; break;

		case ClusterMethod::mediancut_median:		ret = "mediancut_median"; break;
		case ClusterMethod::mediancut_max:			ret = "mediancut_max"; break;
		case ClusterMethod::mediancut_min:			ret = "mediancut_min"; break;


		case ClusterMethod::quantize_wan:			ret = "quantize_wan"; break;
		case ClusterMethod::kmeans_wan:				ret = "kmeans_wan"; break;
		case ClusterMethod::quantize_wu:			ret = "quantize_wu"; break;
		case ClusterMethod::kmeans_wu:				ret = "kmeans_wu"; break;
		case ClusterMethod::quantize_neural:		ret = "quantize_neural"; break;
		case ClusterMethod::kmeans_neural:			ret = "kmeans_neural"; break;

		case ClusterMethod::quantize_DIV:			ret = "quantize_DIV"; break;
		case ClusterMethod::kmeans_DIV:				ret = "kmeans_DIV"; break;
		case ClusterMethod::quantize_PNN:			ret = "quantize_PNN"; break;
		case ClusterMethod::kmeans_PNN:				ret = "kmeans_PNN"; break;
		case ClusterMethod::quantize_SPA:			ret = "quantize_SPA"; break;
		case ClusterMethod::kmeans_SPA:				ret = "kmeans_SPA"; break;
		case ClusterMethod::quantize_EAS:			ret = "quantize_EAS"; break;
		case ClusterMethod::kmeans_EAS:				ret = "kmeans_EAS"; break;

		case ClusterMethod::initial_centers:		ret = "initial_centers"; break;
		case ClusterMethod::X_means:				ret = "X_means"; break;
		default:
			break;
		}
		return ret;
	}

	typedef enum DownsampleMethod
	{
		NEAREST,
		AREA,
		RANDOM,
		DITHER_UNIFORM,
		GRADIENT_MAX,
		DITHER_COLORNORMAL,
		DITHER_GRADIENT_MAX,
		DITHER_DOG,
		CVNEAREST,
		CVLINEAR,
		CVCUBIC,
		CVAREA,
		CVLANCZOS,
		IMPORTANCE_MAP2,

		DownsampleMethodSize
	} DownsampleMethod;

	inline std::string getDownsampleMethodName(const int method)
	{
		std::string ret = "no supported";
		switch (method)
		{
		case NEAREST:             ret = "NEAREST"; break;
		case AREA:				  ret = "AREA"; break;
		case RANDOM:			  ret = "RANDOM"; break;
		case DITHER_UNIFORM:	  ret = "DITHER_UNIFORM"; break;
		case GRADIENT_MAX:		  ret = "GRADIENT_MAX"; break;
		case DITHER_COLORNORMAL:  ret = "DITHER_COLORNORMAL"; break;
		case DITHER_GRADIENT_MAX: ret = "DITHER_GRADIENT_MAX"; break;
		case DITHER_DOG:		  ret = "DITHER_DOG"; break;
		case CVNEAREST:			  ret = "cv::NEAREST"; break;
		case CVLINEAR:			  ret = "cv::LINEAR"; break;
		case CVCUBIC:			  ret = "cv::CUBIC"; break;
		case CVAREA:			  ret = "cv::AREA"; break;
		case CVLANCZOS:			  ret = "cv::LANCZOS4"; break;
		case IMPORTANCE_MAP2:	  ret = "IMPORTANCE_MAP2"; break;

		default:
			break;
		}
		return ret;
	}




	inline int getNumGF(int k)
	{
		return 4 * k;
	}

	class CP_EXPORT ClusteringHDKFBase
	{
	protected:
		cv::Mat input_image8u, input_image32f, labels, centers,
			reshaped_image8u, reshaped_image32f;
		cp::KMeans kmcluster;

		cv::Mat guide_image32f;

		bool isJoint = false;

		int threadMax = omp_get_max_threads();

		cv::Size img_size;
		int channels = 3;//src channels
		int guide_channels = 3;//guide channels
		int depth = CV_32F;

		int radius = 0;//only supported for sliding DCT
		double sigma_space = 0.0;
		double sigma_range = 0.0;
		int spatial_order = 0;
		int border = cv::BORDER_REPLICATE;

		ClusterMethod cm = ClusterMethod::K_means;
		int K = 0; // Cluster num
		int iterations = 10;

		bool isDownsampleClustering = false;
		int downsampleRate = 0;
		int downsampleMethod = cv::INTER_NEAREST;

		int concat_offset = 0;
		int pca_r = 0;
		float kmeans_ratio = 0;

		std::vector<cv::Ptr<cp::SpatialFilterBase>> GF;
		void clustering();
		void downsampleForClustering();

		virtual void filtering(const cv::Mat& src, cv::Mat& dst) = 0;
		//virtual void allocImageBuffer() = 0;
		virtual void body(const cv::Mat& src, cv::Mat& dst, const cv::Mat& guide) = 0;
	public:
		ClusteringHDKFBase();
		virtual ~ClusteringHDKFBase();
		void setGaussianFilterRadius(const int r);
		void setGaussianFilter(const double sigma_space, const cp::SpatialFilterAlgorithm gf_method, const int gf_order);

		void setParameter(cv::Size img_size, double sigma_space, double sigma_range, ClusterMethod cm,
			int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			bool isDownsampleClustering = false, int downsampleRate = 2, int downsampleMethod = cv::INTER_NEAREST);

		void setK(int k);
		int getK();

		void setConcat_offset(int concat_offset);
		void setPca_r(int pca_r);
		void setKmeans_ratio(float kmeans_ratio);
		void setNumIterations(int iterations) { this->iterations = iterations; }

		virtual void filtering(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			bool isDownsampleClustering = false, int downsampleRate = 2, int downsampleMethod = cv::INTER_NEAREST) = 0;

		virtual void jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			bool isDownsampleClustering = false, int downsampleRate = 2, int downsampleMethod = cv::INTER_NEAREST) = 0;
	};

	class TexturenessSampling
	{
		cv::RNG rng;
		cv::Mat histogram;

		cv::Mat importance;
		cv::Mat src_32f;
		cv::Mat gray;
		cv::Mat sampleMask;
		cv::Mat weight;
		bool isUseWeight = false;
		//Ptr<cp::SpatialFilterBase> gf = cp::createSpatialFilter(cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, CV_32F, cp::SpatialKernel::GAUSSIAN);

		float edgeDiffusionSigma = 0.7f;
		void gradientMax(const cv::Mat& src, cv::Mat& dest, const bool isAccumurate);

		void gradientMaxDiffuse3x3(const cv::Mat& src, cv::Mat& dest, const bool isAccumurate, const float sigma);

		//return maxval;
		template<int ch>
		float gradientMaxDiffuse3x3(const std::vector<cv::Mat>& src, cv::Mat& dest, const bool isAccumurate, const float sigma, const cv::Rect roi);
		float computeTexturenessGradientMax(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Rect roi);
		float computeTexturenessGradientMax(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi);

		float getmax(const cv::Mat& src);

		void normalize_max1(cv::Mat& srcdest);

		float computeTexturenessDoG(const std::vector<cv::Mat>& src, cv::Mat& dest, const float sigma);
		float computeTexturenessDoG(const cv::Mat& src, cv::Mat& dest, const float sigma);



		void computeTexturenessBilateral(const std::vector<cv::Mat>& src, cv::Mat& dest, const float sigmaSpace, const float sr = 70.f);

		void mask2samples(const int sample_num, const cv::Mat& sampleMask, const std::vector<cv::Mat>& guide, cv::Mat& dest, cv::Rect roi);

		void remap(cv::Mat& srcdest, const float scale, const float rangemax);
		float computeScaleHistogram(cv::Mat& src, const float sampling_ratio, const int numBins = 100, const float rangemin = 0.f, const float rangemax = 1.f);

		//source [0:1]
		//_MM_XORSHIFT32_AVX2 mrand;
		int randDither(const cv::Mat& src, cv::Mat& dest);

		template<int channels>
		int remapRandomDitherSamples(const int sample_num, const float scale, const float rangemax, const cv::Mat& prob, const std::vector<cv::Mat>& guide, cv::Mat& dest, cv::Rect roi);
		template<int channels>
		int remapFloydSteinbergDitherSamples(const int sample_num, const float scale, const float rangemax, cv::Mat& prob, const std::vector<cv::Mat>& guide, cv::Mat& dest, cv::Rect roi);
		template<int channels>
		int remapFloydSteinbergDitherSamplesWeight(const int sample_num, const float scale, const float rangemax, cv::Mat& prob, const std::vector<cv::Mat>& guide, cv::Mat& dest, cv::Mat& wmap, cv::Rect roi);

		int dither(cv::Mat& inportance, cv::Mat& dest, const int ditheringMethod);

	public:
		TexturenessSampling();
		TexturenessSampling(const float edgeDiffusionSigma);

		void generateDoG(const std::vector<cv::Mat>& src, cv::Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const float sigma);
		void generateDoG(const cv::Mat& src, cv::Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const float sigma);

		void generateGradientMax(const std::vector<cv::Mat>& src, cv::Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage);
		void generateGradientMax(const cv::Mat& src, cv::Mat& dest, int& sample_num, const float sampling_ratio, const int ditheringMethod);


		int generateUniformSamples(const std::vector<cv::Mat>& image, cv::Mat& samples, const float sampling_ratio, const int ditheringMethod, const cv::Rect roi);
		int generateColorNormalizedSamples(const std::vector<cv::Mat>& image, cv::Mat& samples, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const cv::Rect roi);
		int generateGradientMaxSamples(const std::vector<cv::Mat>& image, cv::Mat& samples, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const cv::Rect roi);
		int generateGradientMaxSamplesWeight(const std::vector<cv::Mat>& image, cv::Mat& samples, cv::Mat& weight, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const cv::Rect roi);
		int generateDoGSamples(const std::vector<cv::Mat>& image, cv::Mat& samples, const float sigma, const float sampling_ratio, const int ditheringMethod, const bool isUseAverage, const cv::Rect roi);

		void setEdgeDiffusionSigma(const float sigma);
		void setUseWeight(const bool flag);
	};

	class CP_EXPORT ClusteringHDKFSingleBase
	{
		cv::Mat wmap;
		TexturenessSampling ts;

		cv::Mat labels, guide_image8u,
			reshaped_image8u, reshaped_image32f;
	protected:
		bool isTestClustering;
		cv::Mat projectionMatrix;//for PCA
		cv::Mat eigenValue;//for PCA

		const int num_timerblock_max = 10;
		std::vector<cp::Timer> timer;
		int downSampleImage = 1;
		std::vector<cv::Mat> downsampleSRC;
		cv::Mat guide_image32f;

		cv::Mat sampling_mask;//dithering result
		bool isJoint = false;
		int statePCA = 0;//0: noPCA, 1: srcPCA, 2: jointPCA
		std::vector<cv::Mat> vsrc;//input image
		std::vector<cv::Mat> vguide;//guide image
		std::vector<cv::Mat> vsrcRes;//input image resize
		std::vector<cv::Mat> vguideRes;//guide image resize
		std::vector<cv::Mat> NumerDenomRes;//numer denom for resize upsample
		cp::KMeans kmcluster;
		int boundaryLength = 0;
		int borderType = cv::BORDER_DEFAULT;

		cv::Size img_size;
		int channels = 3;//src channels
		int guide_channels = 3;//guide channels
		int depth = CV_32F;

		int radius = 0;//only supported for sliding DCT
		double sigma_space = 0.0;
		double sigma_range = 0.0;
		int spatial_order = 0;

		int kmeanspp_trials = 3;
		int kmeansrepp_trials = 3;
		double kmeans_sigma = 25.0;
		double kmeans_signal_max = 255.0;

		// for kmeans
		int num_sample_max = 0;
		ClusterMethod cm = ClusterMethod::K_means_pp_fast;
		int clusterRefineMethod = 0;
		int K = 0; // Cluster num
		int attempts = 1;
		int iterations = 20;
		cv::Mat mu;//[guide_channelsxK]clustering sampling points

		bool isCropBoundaryClustering = false;
		double sampleRate = 0.25;
		int downsampleClusteringMethod = cv::INTER_NEAREST;
		int downsampleImageMethod = cv::INTER_AREA;

		int concat_offset = 0;
		int pca_r = 0;
		float kmeans_ratio = 0;

		int patchPCAMethod = 0;

		cv::Ptr<cp::SpatialFilterBase> GF;

		void downsampleImage(const std::vector<cv::Mat>& vsrc, std::vector<cv::Mat>& vsrcRes, const std::vector<cv::Mat>& vguide, std::vector<cv::Mat>& vguideRes, const int downsampleImageMethod = cv::INTER_AREA);

		//swich downsampleForClustering/mergeForClustering
		void clustering();
		void downsampleForClustering(std::vector<cv::Mat>& src, cv::Mat& dest, const bool isCropBoundary);
		void downsampleForClusteringWith8U(std::vector<cv::Mat>& src, cv::Mat& dest, cv::Mat& image8u, const bool isCropBoundary);
		void mergeForClustering(std::vector<cv::Mat>& src, cv::Mat& dest, const bool isCropBoundary);

		virtual void body(const std::vector<cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide) = 0;//bgr split data in ->bgr merged data out
		//bool isUseFmath = false;//false
		bool isUseFmath = true;//true

	public:
		ClusteringHDKFSingleBase();
		virtual ~ClusteringHDKFSingleBase();
		void setGaussianFilterRadius(const int r);
		void setGaussianFilter(const double sigma_space, const cp::SpatialFilterAlgorithm method, const int gf_order);
		void setBoundaryLength(const int length);
		void setNumIterations(const int iterations) { this->iterations = iterations; }

		void setKMeansPPTrials(const int num) { this->kmeanspp_trials = num; }
		void setKMeansREPPTrials(const int num) { this->kmeansrepp_trials = num; }
		void setKMeansAttempts(const int attempts) { this->attempts = attempts; }
		void setKMeansSigma(const double sigma) { this->kmeans_sigma = sigma; }
		void setKMeansSignalMax(const double signal_max) { this->kmeans_signal_max = signal_max; }

		void setClusterRefine(const int method) { this->clusterRefineMethod = method; };
		void setDownsampleImageSize(const int val) { this->downSampleImage = val; }
		void setParameter(cv::Size img_size, double sigma_space, double sigma_range, ClusterMethod cm,
			int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST, int boundarylength = 0, int borderType = cv::BORDER_DEFAULT);
		void setSampleRate(const double rate);
		void setConcat_offset(int concat_offset);
		void setPca_r(int pca_r);
		void setKmeans_ratio(float kmeans_ratio);
		void setCropClustering(bool isCropClustering);
		void setPatchPCAMethod(int method);

		void setTestClustering(bool flag);

		void filter(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, const cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			const double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST, int border = cv::BORDER_DEFAULT);
		void jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, const double sigma_space, const double sigma_range,
			const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth,
			const double downsampleRate = 0.25, const int downsampleMethod = cv::INTER_NEAREST, int border = cv::BORDER_DEFAULT);

		void filter(const std::vector<cv::Mat>& src, cv::Mat& dst, const double sigma_space, const double sigma_range,
			const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth,
			const double downsampleRate = 0.25, const int downsampleMethod = cv::INTER_NEAREST, const int boundaryLength = 0, int border = cv::BORDER_DEFAULT);
		void PCAfilter(const std::vector<cv::Mat>& src, const int pca_channels, cv::Mat& dst, const double sigma_space, const double sigma_range,
			const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth,
			const double downsampleRate = 0.25, const int downsampleMethod = cv::INTER_NEAREST, const int boundaryLength = 0, int border = cv::BORDER_DEFAULT);
		void jointfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, cv::Mat& dst, const double sigma_space, const double sigma_range,
			const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth,
			const double downsampleRate = 0.25, const int downsampleMethod = cv::INTER_NEAREST, const int boundaryLength = 0, int border = cv::BORDER_DEFAULT);
		void jointPCAfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, const int pca_channels, cv::Mat& dst, const double sigma_space, const double sigma_range,
			const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth,
			const double downsampleRate = 0.25, const int downsampleMethod = cv::INTER_NEAREST, const int boundaryLength = 0, int border = cv::BORDER_DEFAULT);

		void nlmfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, cv::Mat& dst, const double sigma_space, const double sigma_range, const int patch_r, const int reduced_dim,
			const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth,
			const double downsampleRate = 0.25, const int downsampleMethod = cv::INTER_NEAREST, const int boundaryLength = 0, int border = cv::BORDER_DEFAULT);

		cv::Mat getSamplingPoints();
		cv::Mat cloneEigenValue();//clone eigenValue
		void printRapTime();
	};

	//clerstering, constant-time, color BF
	enum class ConstantTimeHDGF
	{
		Interpolation,
		Nystrom,
		SoftAssignment,
		Interpolation2,
		Interpolation3,
	};
	inline std::string getclusteringHDKFMethodName(int method)
	{
		std::string ret = "";
		if (method == (int)ConstantTimeHDGF::Interpolation) ret = "Interpolation";
		if (method == (int)ConstantTimeHDGF::Interpolation2) ret = "Interpolation2";
		if (method == (int)ConstantTimeHDGF::Interpolation3) ret = "Interpolation3";
		if (method == (int)ConstantTimeHDGF::Nystrom) ret = "Nystrom";
		if (method == (int)ConstantTimeHDGF::SoftAssignment) ret = "Soft";
		return ret;
	}

	CP_EXPORT cv::Ptr<ClusteringHDKFBase> createClusteringHDKF(ConstantTimeHDGF method);

	CP_EXPORT cv::Ptr<ClusteringHDKFSingleBase> createClusteringHDKFSingle(ConstantTimeHDGF method);

	class CP_EXPORT ClusteringHDKF_Nystrom : public ClusteringHDKFBase
	{
	private:
		cv::Mat A, V, D, denom;
		std::vector<cv::Mat> vecW, split_image, split_numer, inter_denom;
		std::vector<std::vector<cv::Mat>> inter_numer;
		float coef = 0.f;
	protected:
		void init(const cv::Mat& src, cv::Mat& dst);
		void calcVecW();
		void calcA();
		void mul_add_gaussian();
		void summing();
		void divide();
		void xmeans_init(const cv::Mat& src, cv::Mat& dst);
		void body(const cv::Mat& src, cv::Mat& dst, const cv::Mat& guide);
	public:
		//ConstantTimeCBF_Nystrom() {};
		//~ConstantTimeCBF_Nystrom();
		cv::Mat get_centers();
		void set_labels(const cv::Mat& labels);
		void set_centers(const cv::Mat& centers);
		void filtering(const cv::Mat& src, cv::Mat& dst);
		void filtering(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			bool isDownsampleClustering = false, int downsampleRate = 2, int downsampleMethod = cv::INTER_NEAREST);
		void jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			bool isDownsampleClustering = false, int downsampleRate = 2, int downsampleMethod = cv::INTER_NEAREST);
	};

	class CP_EXPORT ClusteringHDKF_NystromSingle : public ClusteringHDKFSingleBase
	{
	private:
		cv::Mat A;//K x K: approximated small matrix
		cv::Mat eigenvecA, lambdaA;
		cv::Mat denom;
		std::vector<cv::Mat> numer;
		std::vector<cv::Mat> B;//image size x K: projected matrix

		//internal mul_add_gaussian()
		std::vector<cv::Mat> Uf;//splatted image
		cv::Mat U;//image_size
		cv::Mat U_Gaussian;//image_size

		void alloc(cv::Mat& dst);
		void computeAandEVD(const cv::Mat& mu, cv::Mat& lambdaA, cv::Mat& eigenvecA);//O(GC*K*K/2)

		template<int use_fmath, int channel>
		void computeB(const std::vector<cv::Mat>& guide);//O(GC*N*K)
		template<int use_fmath>
		void computeBCn(const std::vector<cv::Mat>& guide);


		template<int channels>
		void split_blur_merge();
		void split_blur_merge();
		void normalize(cv::Mat& dst);
		template<int channels>
		void normalize(cv::Mat& dst);
		void body(const std::vector<cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide) override;

	public:
	};

	class CP_EXPORT ClusteringHDKF_SoftAssignment : public ClusteringHDKFBase
	{
	private:
		cv::Mat W2_sum;
		std::vector<cv::Mat> split_image, vecW, W2, split_inter2;
		std::vector<std::vector<cv::Mat>> split_inter;
		float coef;
		float lambda = 0.5;

	protected:
		void init(const cv::Mat& src, cv::Mat& dst);
		void calcAlpha();
		void mul_add_gaussian();
		void xmeans_init(const cv::Mat& src, cv::Mat& dst);

		void body(const cv::Mat& src, cv::Mat& dst, const cv::Mat& guide);
	public:
		cv::Mat get_centers();
		void set_labels(const cv::Mat& labels);
		void set_centers(const cv::Mat& centers);
		void filtering(const cv::Mat& src, cv::Mat& dst);
		void filtering(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			bool isDownsampleClustering = false, int downsampleRate = 2, int downsampleMethod = cv::INTER_NEAREST);
		void jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			bool isDownsampleClustering = false, int downsampleRate = 2, int downsampleMethod = cv::INTER_NEAREST);
	};

	class CP_EXPORT ClusteringHDKF_SoftAssignmentSingle : public ClusteringHDKFSingleBase
	{
	private:
		float lambda = 0.5f;

		cv::Mat alphaSum;
		std::vector<cv::Mat> vecW, alpha, numer;
		std::vector<cv::Mat> split_inter;

		void alloc(cv::Mat& dst);
		template<int use_fmath>
		void computeWandAlpha(const std::vector<cv::Mat>& guide, const std::vector<cv::Mat>& guideRes);

		void split_blur(const int k);
		template<int flag>
		void merge(cv::Mat& dst, const int k);

		void body(const std::vector <cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide) override;

	public:
		void setLambdaInterpolation(const float lambda_);
	};

	class CP_EXPORT ClusteringHDKF_InterpolationSingle : public ClusteringHDKFSingleBase
	{
	private:
		bool isUseLocalStatisticsPrior = false;
		bool isUseLocalMu = true;//default true;
		//bool isUseLocalMu = false;//default false;
		float delta = 0.f;

		std::vector<float> lut_bflsp;
		std::vector<cv::Mat> lsp;
		cv::Mat denom;
		std::vector<cv::Mat> vecW, alpha, numer;
		std::vector<cv::Mat> split_inter;
		cv::Mat blendLSPMask;
		cv::Mat index;
		void alloc(cv::Mat& dst);

		void setMergingNumerDenomMat(std::vector<cv::Mat>& dest, const int k, const int upsampleSize);
		template<int use_fmath>
		void computeW(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& vecW);//valid only (src==guide case)
		template<int use_fmath>
		void computeWandAlpha(const std::vector<cv::Mat>& guide);

		void computeIndex(const std::vector<cv::Mat>& guide, const std::vector<cv::Mat>& guideRes);
		template<int use_fmath>
		void computeAlpha(const std::vector<cv::Mat>& guide, const int k);

		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingMu(std::vector<cv::Mat>& src, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingNLMMu(std::vector<cv::Mat>& src, const int k);

		template<int use_fmath, const bool isInit, int channels, int guide_channels>
		void mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlpha(const std::vector<cv::Mat>& guide, const int k);
		void mergePreComputedAlpha(const int k, const bool isInit);

		void merge(const int k, const bool isInit);


		template<int channels>
		void split_blur(const int k, const bool isUseFmath, const bool isUseLSP);
		void split_blur(const int k, const bool isUseFmath, const bool isUseLSP);

		void normalize(cv::Mat& dst);

		void body(const std::vector <cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide) override;

		bool isWRedunductLoadDecomposition = true;//true
		const bool isUsePrecomputedWforeachK = false;//false
	public:
		void setDeltaLocalStatisticsPrior(const float delta);//LocalStatisticsPrior
		void setIsUseLocalMu(const bool flag);
		void setIsUseLocalStatisticsPrior(const bool flag);
	};

	class CP_EXPORT ClusteringHDKF_Interpolation2Single : public ClusteringHDKFSingleBase
	{
	private:
		bool isUseLocalStatisticsPrior = false;
		bool isUseLocalMu = true;//default true;
		//bool isUseLocalMu = false;//default false;
		float delta = 0.f;

		std::vector<float> lut_bflsp;
		std::vector<cv::Mat> lsp;
		cv::Mat denom;
		std::vector<cv::Mat> vecW, alpha, numer;
		std::vector<cv::Mat> split_inter;
		cv::Mat blendLSPMask;
		cv::Mat index;
		std::vector<cv::Mat> wmap;//temp
		void alloc(cv::Mat& dst);

		void mergeNumerDenomMat(std::vector<cv::Mat>& dest, const int k, const int upsampleSize);
		template<int use_fmath>
		void computeW(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& vecW);//valid only (src==guide case)
		template<int use_fmath>
		void computeWandAlpha(const std::vector<cv::Mat>& guide);

		void computeIndex(const std::vector<cv::Mat>& guide, const std::vector<cv::Mat>& guideRes);
		template<int use_fmath>
		void computeAlpha(const std::vector<cv::Mat>& guide, const int k);

		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingMu(std::vector<cv::Mat>& src, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingNLMMu(std::vector<cv::Mat>& src, const int k);

		template<int use_fmath, const bool isInit, int channels, int guide_channels>
		void mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlpha(const std::vector<cv::Mat>& guide, const int k);
		void mergePreComputedAlpha(const int k, const bool isInit);

		void merge(const int k, const bool isInit);


		template<int channels>
		void split_blur(const int k, const bool isUseFmath, const bool isUseLSP);
		void split_blur(const int k, const bool isUseFmath, const bool isUseLSP);

		void normalize(cv::Mat& dst);

		void body(const std::vector <cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide) override;

		bool isWRedunductLoadDecomposition = true;//true
		const bool isUsePrecomputedWforeachK = false;//false
	public:
		void setDeltaLocalStatisticsPrior(const float delta);//LocalStatisticsPrior
		void setIsUseLocalMu(const bool flag);
		void setIsUseLocalStatisticsPrior(const bool flag);
	};

	class CP_EXPORT ClusteringHDKF_Interpolation3Single : public ClusteringHDKFSingleBase
	{
	private:
		bool isUseLocalStatisticsPrior = false;
		bool isUseLocalMu = true;//default true;
		//bool isUseLocalMu = false;//default false;
		float delta = 0.f;

		std::vector<float> lut_bflsp;
		std::vector<cv::Mat> lsp;
		cv::Mat denom;
		std::vector<cv::Mat> vecW, alpha, numer;
		std::vector<cv::Mat> split_inter;
		cv::Mat blendLSPMask;
		cv::Mat index;
		std::vector<cv::Mat> wmap;//temp
		std::vector<cv::Mat> vmap;//temp
		void alloc(cv::Mat& dst);

		void mergeNumerDenomMat(std::vector<cv::Mat>& dest, const int k, const int upsampleSize);
		template<int use_fmath>
		void computeW(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& vecW);//valid only (src==guide case)
		template<int use_fmath>
		void computeWandAlpha(const std::vector<cv::Mat>& guide);

		void computeIndex(const std::vector<cv::Mat>& guide, const std::vector<cv::Mat>& guideRes);
		template<int use_fmath>
		void computeAlpha(const std::vector<cv::Mat>& guide, const int k);

		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingMu(std::vector<cv::Mat>& src, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRefineAlpha(std::vector<cv::Mat>& src, std::vector<cv::Mat>& refine, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingNLMMu(std::vector<cv::Mat>& src, const int k);

		template<int use_fmath, const bool isInit, int channels, int guide_channels>
		void mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k);
		template<int use_fmath, const bool isInit>
		void mergeRecomputeAlpha(const std::vector<cv::Mat>& guide, const int k);
		void mergePreComputedAlpha(const int k, const bool isInit);

		void merge(const int k, const bool isInit);


		template<int channels>
		void split_blur(const int k, const bool isUseFmath, const bool isUseLSP);
		void split_blur(const int k, const bool isUseFmath, const bool isUseLSP);

		void normalize(cv::Mat& dst);

		void body(const std::vector <cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide) override;

		bool isWRedunductLoadDecomposition = true;//true
		const bool isUsePrecomputedWforeachK = false;//false
	public:
		void setDeltaLocalStatisticsPrior(const float delta);//LocalStatisticsPrior
		void setIsUseLocalMu(const bool flag);
		void setIsUseLocalStatisticsPrior(const bool flag);
	};


	class CP_EXPORT TileClusteringHDKF
	{
	private:
		const int thread_max;
		cv::Size div;
		cv::Size divImageSize;
		cv::Size tileSize;
		int boundaryLength = 0;
		std::vector<cv::Mat> split_src, split_dst, subImageOutput;
		std::vector<std::vector<cv::Mat>> subImageInput;
		std::vector<std::vector<cv::Mat>> subImageGuide;
		std::vector<cv::Ptr<ClusteringHDKFSingleBase>> scbf;

		std::vector<cv::Mat> srcSplit;
		std::vector<cv::Mat> guideSplit;

		int channels = 3;
		int guide_channels = 3;
		int attempts = 1;

		bool isDebug = true;
		std::vector<cv::Mat> mu;//for debug
		cp::ConsoleImage ci;
		ConstantTimeHDGF method;

		//for stats
		std::vector<cv::Mat> eigenVectors;
		int guiTestIndex = -1;
	public:
		TileClusteringHDKF(cv::Size div, ConstantTimeHDGF method);
		~TileClusteringHDKF();

		void setBoundaryLength(int length);
		void setDownsampleImageSize(const int val);
		void setKMeansAttempts(const int attempts);
		void setNumIterations(const int iterations);

		void setKMeansPPTrials(const int num);
		void setKMeansREPPTrials(const int num);
		void setKMeansSigma(const double sigma);
		void setKMeansSignalMax(const double signal_max);

		void setClusterRefine(const int method);
		void setConcat_offset(int concat_offset);
		void setPca_r(int pca_r);
		void setKmeans_ratio(float kmeans_ratio);
		void setCropClustering(bool isCropClustering);
		void setPatchPCAMethod(int method);
		void setGUITestIndex(int index);

		//for Interpoation
		void setLambdaInterpolation(const float lambda_);
		void setDeltaLocalStatisticsPrior(const float delta);
		void setIsUseLocalMu(const bool flag);
		void setIsUseLocalStatisticsPrior(const bool flag);

		void setSampleRate(const float rate);

		void filter(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST
			, double truncateBoundary = 3.0, const int borderType = cv::BORDER_DEFAULT);

		void PCAfilter(const cv::Mat& src, const int reduced_dim, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST
			, double truncateBoundary = 3.0, const int borderType = cv::BORDER_DEFAULT);

		void jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST
			, double truncateBoundary = 3.0, const int borderType = cv::BORDER_DEFAULT);

		void jointPCAfilter(const cv::Mat& src, const cv::Mat& guide, const int reduced_dim, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST
			, double truncateBoundary = 3.0, const int borderType = cv::BORDER_DEFAULT);

		void jointPCAfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, const int reduced_dim, cv::Mat& dst, double sigma_space, double sigma_range,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST
			, double truncateBoundary = 3.0, const int borderType = cv::BORDER_DEFAULT);

		void nlmfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int patch_r, const int reduced_dim,
			ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
			double downsampleRate = 0.25, int downsampleMethod = cv::INTER_NEAREST
			, double truncateBoundary = 3.0, const int borderType = cv::BORDER_DEFAULT);

		cv::Size getTileSize();
		void getTileInfo();
		void getEigenValueInfo();
	};

	//method 0: Wan, 1: Wu, 2: NN
	void quantization(const cv::Mat& input_image, int K, cv::Mat& centers, cv::Mat& labels, const int method);
	void nQunat(const cv::Mat& input_image, const int K, cv::Mat& centers, cv::Mat& labels, const ClusterMethod cm);

	//Local Uniform Distribution
	void bilateralFilterLocalStatisticsPrior(const cv::Mat& src, cv::Mat& dest, const float sigma_range, const float sigma_space, const float delta);
	//void bilateralFilterLocalStatisticsPrior(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, const float sigma_range, const float sigma_space, const float delta, std::vector<cv::Mat>& smooth = std::vector<cv::Mat>());
	void bilateralFilterLocalStatisticsPrior(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest, const float sigma_range, const float sigma_space, const float delta, std::vector<cv::Mat>& smooth);
	enum class BFLSPSchedule
	{
		Compute,
		LUT,
		LUTSQRT,
	};
	void bilateralFilterLocalStatisticsPriorInternal(const std::vector<cv::Mat>& src, const cv::Mat& vecW, std::vector<cv::Mat>& split_inter, const float sigma_range, const float sigma_space, const float delta, cv::Mat& mask, BFLSPSchedule schedule, float* lut = nullptr);
}