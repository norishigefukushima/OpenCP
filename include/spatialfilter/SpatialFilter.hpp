#pragma once
#include <opencp.hpp>
//VYV
#define VYV_NUM_NEWTON_ITERATIONS 6
#define VYV_ORDER_MAX 5
#define VYV_ORDER_MIN 3
#define VYV_VALID_ORDER(K)  (VYV_ORDER_MIN <= (K) && (K) <= VYV_ORDER_MAX)

//Deriche
#define DERICHE_ORDER_MIN 2
#define DERICHE_ORDER_MAX 4
#define DERICHE_VALID_ORDER(K)  (DERICHE_ORDER_MIN <= (K) && (K) <= DERICHE_ORDER_MAX)

class CP_EXPORT TileDivision
{
	std::vector<cv::Point> pt;//left top point
	std::vector<cv::Size> tileSize;
	std::vector<int> threadnum;
	cv::Size div;
	cv::Size imgSize;
	int width_step = 0;
	int height_step = 0;

	void update_pt();

public:

	//div.width * y + x;
	cv::Rect getROI(const int x, int y);
	cv::Rect getROI(const int index);

	TileDivision(cv::Size imgSize, cv::Size div);

	bool compute(const int width_step_, const int height_step_);

	void draw(cv::Mat& src, cv::Mat& dst);
	void show(std::string wname);
};

namespace cp
{
	//CP_EXPORT void GaussianFilterDDAcc(cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_space, const int borderType, const bool isRectangle);
	//CP_EXPORT void GaussianFilterDD(cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_space, const int borderType, const bool isRectangle);

	// filtering method
#define TEST_GF_METHOD
#ifdef TEST_GF_METHOD
	enum class SpatialFilterAlgorithm
	{
		IIR_AM,
		IIR_VYV,
		IIR_DERICHE,
		SlidingDCT1_AVX,
		SlidingDCT3_AVX,
		SlidingDCT5_AVX,
		SlidingDCT7_AVX,
		FIR_OPENCV,
		FIR_SEPARABLE,//under debug
		FIR_Sep2D_OPENCV,//call sepFilter2D
		FIR_OPENCV2D,//call filter2D
		DCT_AVX,
		FIR_KAHAN,
		BOX,
		SlidingDCT1_CONV,
		SlidingDCT1_64_AVX,
		SlidingDCT3_16_AVX,
		SlidingDCT3_VXY,
		SlidingDCT3_CONV,
		SlidingDCT3_DEBUG,
		SlidingDCT3_64_AVX,
		SlidingDCT5_16_AVX,
		SlidingDCT5_VXY,
		SlidingDCT5_CONV,
		SlidingDCT5_DEBUG,
		SlidingDCT5_64_AVX,
#ifdef CP_AVX_512
		SlidingDCT5_AVX512,
#endif
		SlidingDCT7_VXY,
		SlidingDCT7_CONV,
		SlidingDCT7_64_AVX,
		FIR_OPENCV_64F,
		SIZE,

		//some wrapper function does not support as follows
		IIR_VYV_NAIVE,
		IIR_DERICHE_NAIVE,
		IIR_AM_NAIVE,
	};
#else
	enum class SpatialFilterAlgorithm
	{
		IIR_AM,
		IIR_VYV,
		IIR_DERICHE,

		BOX,

		SlidingDCT1_AVX,
		SlidingDCT1_CONV,
		SlidingDCT1_64_AVX,
		SlidingDCT3_16_AVX,
		SlidingDCT3_AVX,
		SlidingDCT3_VXY,
		SlidingDCT3_CONV,
		SlidingDCT3_DEBUG,
		SlidingDCT3_64_AVX,
		SlidingDCT5_16_AVX,
		SlidingDCT5_AVX,
		SlidingDCT5_VXY,
		SlidingDCT5_CONV,
		SlidingDCT5_DEBUG,
		SlidingDCT5_64_AVX,
#ifdef CP_AVX_512
		SlidingDCT5_AVX512,
#endif
		SlidingDCT7_AVX,
		SlidingDCT7_VXY,
		SlidingDCT7_CONV,
		SlidingDCT7_64_AVX,
		DCT_AVX,
		FIR_OPENCV,
		FIR_Sep2D_OPENCV,
		FIR_OPENCV_64F,
		FIR_KAHAN,

		SIZE,
		FIR_SEPARABLE,//under debug
		//some wrapper function does not support as follows
		IIR_VYV_NAIVE,
		IIR_DERICHE_NAIVE,
		IIR_AM_NAIVE,
	};
#endif

	CP_EXPORT std::string getAlgorithmName(SpatialFilterAlgorithm method);
	CP_EXPORT std::string getAlgorithmNameShort(SpatialFilterAlgorithm method);

	inline int cliped_order(const int order, SpatialFilterAlgorithm m)
	{
		int cliped = order;

		switch (m)
		{
		case SpatialFilterAlgorithm::IIR_VYV_NAIVE:
		case SpatialFilterAlgorithm::IIR_VYV:
			cliped = std::max(3, std::min(5, order));
			break;
		case SpatialFilterAlgorithm::IIR_DERICHE_NAIVE:
		case SpatialFilterAlgorithm::IIR_DERICHE:
			cliped = std::max(2, std::min(4, order));
			break;
		case SpatialFilterAlgorithm::IIR_AM_NAIVE:
		case SpatialFilterAlgorithm::IIR_AM:
			cliped = std::max(1, order);
			break;
		case SpatialFilterAlgorithm::SlidingDCT1_AVX:
		case SpatialFilterAlgorithm::SlidingDCT1_64_AVX:
		case SpatialFilterAlgorithm::SlidingDCT3_AVX:
		case SpatialFilterAlgorithm::SlidingDCT3_16_AVX:
		case SpatialFilterAlgorithm::SlidingDCT3_64_AVX:
		case SpatialFilterAlgorithm::SlidingDCT5_16_AVX:
		case SpatialFilterAlgorithm::SlidingDCT5_AVX:
		case SpatialFilterAlgorithm::SlidingDCT5_64_AVX:
		case SpatialFilterAlgorithm::SlidingDCT7_AVX:
		case SpatialFilterAlgorithm::SlidingDCT7_64_AVX:
			cliped = std::max(1, order);
			break;
		default:
			break;
		}
		return cliped;
	}

	//base class for Smoothing filter
	enum class SpatialKernel
	{
		GAUSSIAN,
		HAT,

		SIZE,
	};

	class CP_EXPORT SpatialFilterBase
	{
	protected:
		//for peeling outer loop
		int top = 0;
		int bottom = 0;
		int left = 0;
		int right = 0;

		int depth = CV_32F;
		cp::SpatialFilterAlgorithm algorithm = SpatialFilterAlgorithm::SIZE;
		int gf_order = 0;
		double sigma = 0.0;
		cv::Size imgSize;
		int radius = 0;
		bool isUseFixRadius = false;
		
	public:
		void computeRadius(const int radius);

		void setIsInner(const int top, const int bottom, const int left, const int right);

		cp::SpatialFilterAlgorithm getAlgorithmType();
		int getOrder();
		double getSigma();
		cv::Size getSize();
		
		void setFixRadius(const int radius);
		void unsetFixRadius();
		int getRadius();
		virtual int getRadius(const double sigma, const int order) { return (int)ceil(sigma * order); };

		SpatialFilterBase(cv::Size imgSize, int depth);
		SpatialFilterBase() { ; };
		virtual ~SpatialFilterBase() { ; };
		virtual void body(const cv::Mat& src, cv::Mat& dst, const int borderType) = 0;
		virtual void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) { ; };

		void* operator new(size_t i)
		{
			return _mm_malloc(i, 32);
		}
		void operator delete(void* p)
		{
			_mm_free(p);
		}
	};

#pragma region FIR
	class CP_EXPORT GaussianFilterFIROpenCV : public SpatialFilterBase
	{
	private:
		cv::Mat internalBuff;
		int dest_depth = -1;
		int d = 0;

	public:
		GaussianFilterFIROpenCV(cv::Size img_size, double sigma, int trunc, int depth);
		GaussianFilterFIROpenCV(const int internal_depth, const bool isCompute32F);
		~GaussianFilterFIROpenCV();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT)override;
	};

	class CP_EXPORT GaussianFilterFIROpenCV2D : public SpatialFilterBase
	{
	private:
		cv::Mat internalBuff;
		int dest_depth = -1;
		int d = 0;

	public:
		GaussianFilterFIROpenCV2D(cv::Size img_size, double sigma, int trunc, int depth);
		GaussianFilterFIROpenCV2D(const int internal_depth, const bool isCompute32F);
		~GaussianFilterFIROpenCV2D();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT)override;
	};

	class CP_EXPORT GaussianFilterFIRSep2DOpenCV : public SpatialFilterBase
	{
	private:
		cv::Mat internalBuff;
		int dest_depth = -1;
		int d = 0;

	public:
		GaussianFilterFIRSep2DOpenCV(cv::Size img_size, double sigma, int trunc, int depth);
		GaussianFilterFIRSep2DOpenCV(const int internal_depth, const bool isCompute32F);
		~GaussianFilterFIRSep2DOpenCV();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT)override;
	};

	class CP_EXPORT GaussianFilterFIRKahan : public SpatialFilterBase
	{
	private:
		int dest_depth = -1;
		cv::Mat internalBuff;
		cv::Mat src64;

		int d = 0;

	public:
		GaussianFilterFIRKahan(cv::Size img_size, double sigmaS, int trunc, int depth);
		GaussianFilterFIRKahan(const int dest_depth);
		~GaussianFilterFIRKahan();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int border = cv::BORDER_DEFAULT)override;
	};

	class CP_EXPORT GaussianFilterSeparableFIR :public SpatialFilterBase
	{
		int d = 0;

		cv::Mat buffer;
		cv::Mat bufferImageBorder;
		cv::Mat bufferImageBorder2;
		std::vector<cv::Mat> bufferLineCols;
		std::vector<cv::Mat> bufferLineRows;
		std::vector<cv::Mat> bufferTile;
		std::vector<cv::Mat> bufferTile2;
		std::vector<cv::Mat> bufferTileLine;
		std::vector<cv::Mat> bufferSubImage;

		int border_s(const int val);
		int border_e(const int val, const int maxval);
		int constVal;

		float* gauss32F = nullptr;
		double* gauss64F = nullptr;
		void createGaussianTable32F(const int r, const float sigma);
		void createGaussianTable64F(const int r, const double sigma);
		void filter2DFIR(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border);
		void filter2DFIR2(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border);

		void filterHVLine(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVLineBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVLineBVP(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVLineHBVPB(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImage(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBHD(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBV(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBHBV(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBHDBV(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBVP(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBHBVP(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBHDBVP(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageTrB(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVImageBHBTr(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterVHLine(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHLineBVP(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHLineBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHLineBVPBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHImage(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHImageBV(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHImageBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHImageBVBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHImageBVBHChacheConflict(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHImageBVP(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHImageBVPBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterHVILine(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVILineB(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVIImage(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterVHILine(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHILineB(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHIImage(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHIImageBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHIImageBVBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterVHILineBufferOverRun(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);//for test


		void filterHVTileImage(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVTileImageBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVTileImageBHTr(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVTileImageTr(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVTileImageT2(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterVHTileImage(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHTileImageBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHTileImageBV(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterVHITileLineBH(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHITileImageBV(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVITileLine(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterHVBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedBorder32F(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedBorderSort32F(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedBorder64F(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedBorderSort64F(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedVPBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHDelayedBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVIBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHIBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterVHIBlockBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterHVTileBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHTileBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHITileBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVITileBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		//not used
		void filterHVNonRasterBorder(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);

		void filterHVBorderSingle(cv::Mat& srcBorder, cv::Size tileSize, cv::Mat& dest, cv::Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHBorderSingle(cv::Mat& srcBorder, cv::Size tileSize, cv::Mat& dest, cv::Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVIBorderSingle(cv::Mat& srcBorder, cv::Size tileSize, cv::Mat& dest, cv::Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHIBorderSingle(cv::Mat& srcBorder, cv::Size tileSize, cv::Mat& dest, cv::Point tileIndex, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedBorderSingle(cv::Mat& srcBorder, cv::Size srcSize, cv::Mat& dest, cv::Point destTop, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterHVDelayedVPBorderSingle(cv::Mat& srcBorder, cv::Size srcSize, cv::Mat& dest, cv::Point destTop, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHDelayedBorderSingle(cv::Mat& srcBorder, cv::Size srcSize, cv::Mat& dest, cv::Point destTop, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterVHVPBorderSingle(cv::Mat& srcBorder, cv::Size srcSize, cv::Mat& dest, cv::Point destTop, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		void filterCopyBorderSingle(cv::Mat& srcBorder, cv::Size srcSize, cv::Mat& dest, cv::Point destTop, const int r, const float sigma, int border, int opt, bool useAllocBuffer);//for debug

		void filterTileSubImage(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer, int method);
		void filterTileSubImageInplace(cv::Mat& srcdest, const int r, const float sigma, int border, int opt, bool useAllocBuffer, int method);

		void filterHVDelayedBorderLJ(cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int border, int opt, bool useAllocBuffer);
		int num_threads;
	public:
		enum
		{
			FIR2D_Border,
			FIR2D2_Border,

			HV_Line,
			HV_LineBH,
			HV_LineBVP,
			HV_LineBHBVP,
			HV_Image,
			HV_ImageBH,
			HV_ImageBHD,
			HV_ImageBV,
			HV_ImageBHBV,
			HV_ImageBHDBV,
			HV_ImageBVP,
			HV_ImageBHBVP,
			HV_ImageBHDBVP,
			HV_ImageBTr,
			HV_ImageBHBTr,

			VH_Line,
			VH_LineBH,
			VH_LineBVP,
			VH_LineBVPBH,
			VH_Image,
			VH_ImageBV,
			VH_ImageBH,
			VH_ImageBVBH,
			VH_ImageBVBHCC,
			VH_ImageBVP,
			VH_ImageBVPBH,

			HVI_Line,
			HVI_LineB,
			HVI_Image,//not important. almost same as HVI line

			VHI_Line,
			VHI_LineBH,
			VHI_Image,
			VHI_ImageBH,
			VHI_ImageBVBH,
			VHIO_Line,//for debug

			HV_T_Image,
			HV_T_ImageBH,
			HV_T_ImageBHTr,
			HV_T_ImageTr,
			HV_T_ImageT2,
			VH_T_Image,
			VH_T_ImageBH,
			VH_T_ImageBV,
			VHI_T_LineBH,
			VHI_T_ImageBV,
			HVI_T_Line,

			HV_Border,
			HVN_Border,//non raster scan
			HV_BorderD,
			HV_BorderDVP,

			VH_Border,
			VH_BorderD,

			HVI_Border,
			VHI_Border,
			VHI_BorderB,

			HV_T_Border,
			VH_T_Border,
			VHI_T_Border,
			HVI_T_Border,


			HV_T_Sub,
			HV_T_SubD,
			HV_T_SubDVP,
			VH_T_Sub,
			VH_T_SubD,
			VH_T_SubVP,
			HVI_T_Sub,
			VHI_T_Sub,
		};
		enum
		{
			VECTOR_WITHOUT = 0,
			VECTOR_AVX = 1,
			VECTOR_SSE = 2,//not implemented
		};
		bool useParallelBorder;
		int schedule = 0;
		int vectorization = VECTOR_AVX;
		GaussianFilterSeparableFIR(cv::Size imgSize, double sigmaS, int trunc, int depth);
		GaussianFilterSeparableFIR(const int schedule, const int depth);
		~GaussianFilterSeparableFIR();

		int numTiles;
		int numTilesPerThread;

		cv::Size tileDiv;
		void setTileDiv(const int tileDivX, const int tileDivY);
		std::vector<cv::Point> tileIndex;
		void createTileIndex(const int tileSizeX, const int tileSizeY);

		void filter(const cv::Mat& src, cv::Mat& dest, const int r, const float sigma, int method, int border = cv::BORDER_REFLECT, int vectorization = VECTOR_AVX, bool useAllocBuffer = true);
		void filter(const cv::Mat& src, cv::Mat& dest, const double sigma, const int order, const int border = cv::BORDER_DEFAULT) override;
		void body(const cv::Mat& src, cv::Mat& dst, int borderType)override;
	};

#pragma endregion

#pragma region IIR
	// a product of causal and anti-causal systems
	template<class Type>
	class GaussianFilterVYV_Naive : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		int truncate_r;

		Type a[VYV_ORDER_MAX + 1];
		Type b;
		Type* h = nullptr;
		Type M[VYV_ORDER_MAX * VYV_ORDER_MAX];

		void horizontalbody(const cv::Mat& src, cv::Mat& dest);
		void verticalbody(cv::Mat& img);

	public:
		GaussianFilterVYV_Naive(cv::Size imgSize, Type sigmaS, int order);
		virtual ~GaussianFilterVYV_Naive();

		void body(const cv::Mat& src, cv::Mat& dst, int borderType)override;
	};

	class CP_EXPORT GaussianFilterVYV_AVX_32F : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		cv::Mat inter;
		cv::Mat inter2;
		//	int dest_depth = -1;
		int dest_depth = CV_32F;

		int truncate_r;

		float coeff[VYV_ORDER_MAX + 1];
		float* h = nullptr;
		double M[VYV_ORDER_MAX * VYV_ORDER_MAX];

		void horizontalFilterVLoadGatherTransposeStore(const cv::Mat& src, cv::Mat& dst);
		void horizontalFilterVLoadSetTransposeStore(const cv::Mat& src, cv::Mat& dst);
		void horizontalFilterVLoadSetTransposeStoreOrder5(const cv::Mat& src, cv::Mat& dst);
		void verticalFilter(cv::Mat& src);

		void allocBuffer();

	public:
		GaussianFilterVYV_AVX_32F(cv::Size imgSize, float sigmaS, int order);
		GaussianFilterVYV_AVX_32F(const int dest_depth);
		virtual ~GaussianFilterVYV_AVX_32F();

		void body(const cv::Mat& src, cv::Mat& dst, int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	class CP_EXPORT GaussianFilterVYV_AVX_64F : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		cv::Mat inter;
		cv::Mat inter2;
		int dest_depth = -1;

		int truncate_r;

		double coeff[VYV_ORDER_MAX + 1];
		double* h = nullptr;
		double M[VYV_ORDER_MAX * VYV_ORDER_MAX];

		void horizontalbody(const cv::Mat& src, cv::Mat& dest);
		void verticalbody(cv::Mat& img);

		void allocBuffer();

	public:
		GaussianFilterVYV_AVX_64F(cv::Size imgSize, double sigmaS, int order);
		GaussianFilterVYV_AVX_64F(const int dest_depth);
		virtual ~GaussianFilterVYV_AVX_64F();

		void body(const cv::Mat& src, cv::Mat& dst, int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	//a sum of causal and anti-causal systems
	template<class Type>
	class GaussianFilterDERICHE : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		int order;
		int truncate_r;
		//T tol;

		Type sigma;

		Type a[DERICHE_ORDER_MAX + 1];
		Type fb[DERICHE_ORDER_MAX];
		Type bb[DERICHE_ORDER_MAX + 1];

		Type* fh;
		Type* bh;

		cv::Mat inter;
		cv::Mat buf;

		void horizontalFilter(const cv::Mat& src, cv::Mat& dst);
		void verticalFilter(const cv::Mat& src, cv::Mat& dst);
	public:
		GaussianFilterDERICHE(cv::Size img_size, Type sigma_space, int order);
		~GaussianFilterDERICHE();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
	};

	class CP_EXPORT GaussianFilterDERICHE_AVX_32F : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		cv::Mat inter;
		cv::Mat inter2;
		//	int dest_depth = -1;
		int dest_depth = CV_32F;

		int truncate_r;

		float a[DERICHE_ORDER_MAX + 1];
		float fb[DERICHE_ORDER_MAX];//forward
		float bb[DERICHE_ORDER_MAX + 1];//backward

		float* fh = nullptr;
		float* bh = nullptr;

		__m256** buf;

		void horizontalFilterVLoadSetTransposeStore(const cv::Mat& src, cv::Mat& dst);
		void horizontalFilterVLoadGatherTransposeStore(const cv::Mat& src, cv::Mat& dst);
		void verticalFiler(const cv::Mat& src, cv::Mat& dst);

		void allocBuffer();

	public:
		GaussianFilterDERICHE_AVX_32F(cv::Size img_size, float sigma_space, int order);
		GaussianFilterDERICHE_AVX_32F(const int dest_depth);
		virtual ~GaussianFilterDERICHE_AVX_32F();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType) override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	class CP_EXPORT GaussianFilterDERICHE_AVX_64F : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		cv::Mat inter;
		cv::Mat inter2;
		int dest_depth = -1;

		int truncate_r;

		double a[DERICHE_ORDER_MAX + 1];
		double fb[DERICHE_ORDER_MAX];
		double bb[DERICHE_ORDER_MAX + 1];

		double* fh = nullptr;
		double* bh = nullptr;

		__m256d** buf = nullptr;

		void horizontalFilter(const cv::Mat& src, cv::Mat& dst);
		void verticalFilter(const cv::Mat& src, cv::Mat& dst);

		void allocBuffer();

	public:
		GaussianFilterDERICHE_AVX_64F(cv::Size imgSize, double sigmaS, int order);
		GaussianFilterDERICHE_AVX_64F(const int dest_depth);
		virtual ~GaussianFilterDERICHE_AVX_64F();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	//AM
	template<class Type>
	class GaussianFilterAM_Naive : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		int order;
		int r_init;
		Type tol;

		Type sigma;

		Type nu;
		Type scale;

		Type* h;

		void horizontalbody(cv::Mat& img);
		void verticalbody(cv::Mat& img);
	public:
		GaussianFilterAM_Naive(cv::Size img_size, Type sigma_space, int order);
		virtual ~GaussianFilterAM_Naive();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
	};

	class CP_EXPORT GaussianFilterAM_AVX_32F : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		cv::Mat inter;
		int dest_depth = -1;

		int r_init;
		const float tol = float(1.0e-6);

		float nu;
		float scale;
		float norm;

		float* h = nullptr;

		void horizontalFilterVLoadGatherTransposeStore(cv::Mat& img);
		void horizontalFilterVLoadSetTransposeStore(cv::Mat& img);
		void verticalFilter(cv::Mat& img);

		void allocBuffer();

	public:
		GaussianFilterAM_AVX_32F(cv::Size imgSize, float sigmaS, int order);
		GaussianFilterAM_AVX_32F(const int dest_depth);
		virtual ~GaussianFilterAM_AVX_32F();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	class CP_EXPORT GaussianFilterAM_AVX_64F : public SpatialFilterBase
	{
	private:
		int borderType = cv::BORDER_DEFAULT;
		cv::Mat inter;
		int dest_depth = -1;

		int r_init;
		double tol = 1.0e-6;

		double nu;
		double scale;
		__m256d* mm_h = nullptr;
		void horizontalFilterVLoadSetTransposeStore(cv::Mat& img);//not implemented
		void horizontalFilterVloadGatherTranposeStore(cv::Mat& img);
		void verticalFilter(cv::Mat& img);

		void allocBuffer();

	public:
		GaussianFilterAM_AVX_64F(cv::Size imgSize, double sigmaS, int order);
		GaussianFilterAM_AVX_64F(const int dest_depth);
		virtual ~GaussianFilterAM_AVX_64F();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

#pragma endregion

#pragma region SlidingDCT
	void setGaussKernelHalf(cv::Mat& dest, const int R, const double sigma, bool isNormalize);
	void generateCosKernel(double* dest, double& totalInv, const int dctType, const double* Gk, const int radius, const int order);

	CP_EXPORT bool  optimizeSpectrum(const double sigma, const int K, const int R, const int dcttype, double* destSpect, const int M = 0);
	//K+1 loop
	CP_EXPORT void computeSpectrumGaussianClosedForm(const double sigma, const int K, const int R, const int dcttype, double* destSpect);
	//spect: K+1 size
	CP_EXPORT int argminR_BruteForce_DCT(const double sigma, const int K, const int dcttype, const double* spect, const bool isOptimize, const bool isGoldenSelectionSearch = true);
	CP_EXPORT int argminR_ContinuousForm_DCT(const double sigma, const int K, const int dcttype, const bool isGoldenSelectionSearch = true);

	/// <summary>
	/// plotting DCT kernel on cp::plot
	/// </summary>
	/// <param name="wname">window name</param>
	/// <param name="isWait">waiting GUI loop (defalt=true)</param>
	/// <param name="GCn">DCT coefficients</param>
	/// <param name="radius">radius</param>
	/// <param name="order">order</param>
	/// <param name="G0">0-th coefficient</param>
	void plotDCTKernel(std::string wname, const bool isWait, const float* GCn, const int radius, const int order, const float G0, const double simga);
	void plotDCTKernel(std::string wname, const bool isWait, const double* GCn, const int radius, const int order, const double G0, const double sigma);

	enum class DCT_COEFFICIENTS
	{
		FULL_SEARCH_OPT,
		FULL_SEARCH_NOOPT,

		SIZE
	};

	enum class SLIDING_DCT_SCHEDULE
	{
		DEFAULT,
		INNER_LOW_PRECISION,
		V_XY_LOOP,
		CONVOLUTION,

		DEBUG,

		SIZE
	};

	//Sliding DCT 1 with AVX (32F)
	class CP_EXPORT SpatialFilterSlidingDCT1_AVX_32F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		float G0 = 0.f;
		float* GCn = nullptr;//G_k * C_n_k
		float* shift = nullptr;//0: 2*C_1, 1: G*C_R
		float* Gk_dct1 = nullptr;
		__m256* buffVFilter = nullptr;
		__m256* fn_hfilter = nullptr;

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);

		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);//naive DCT convolution

		//32F
		template<int order>
		void horizontalFilteringInnerXK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order>
		void horizontalFilteringInnerXK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT1_AVX_32F(cv::Size imgSize, float sigmaS, int order);
		SpatialFilterSlidingDCT1_AVX_32F(const DCT_COEFFICIENTS method, const int dest_depth = -1, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT1_AVX_32F();

		int getRadius(const double sigma, const int order) override;	
		virtual void body(const cv::Mat& src, cv::Mat& dst, const int borderType) override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	//Sliding DCT 1 with AVX (64F)
	class CP_EXPORT SpatialFilterSlidingDCT1_AVX_64F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		bool isHBuff32F = false;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		double G0 = 0.0;
		double* GCn = nullptr;//G_k * C_n_k
		double* shift = nullptr;//0: 2*C_1, 1: G*C_R
		double* Gk_dct1 = nullptr;
		__m256d* buffVFilter = nullptr;
		__m256d* fn_hfilter = nullptr;

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);

		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);//naive DCT convolution

		template<int order>
		void horizontalFilteringInnerXK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType);

		template<int order>
		void horizontalFilteringInnerXK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT1_AVX_64F(cv::Size imgSize, double sigmaS, int order, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT);
		SpatialFilterSlidingDCT1_AVX_64F(const DCT_COEFFICIENTS method, const int dest_depth, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT1_AVX_64F();

		int getRadius(const double sigma, const int order) override;
		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};


	//Sliding DCT 3 with AVX (32F)
	class CP_EXPORT SpatialFilterSlidingDCT3_AVX_32F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		float* GCn = nullptr;//G_k * C_n_k
		float* shift = nullptr;//0: 2*C_1, 1: G*C_R
		__m256* buffVFilter = nullptr;//size: width*(2*order+2))
		__m256* fn_hfilter = nullptr;//size: width*8

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);

		//Naive DCT Convolution: O(KR)
		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);//naive DCT convolution

		//for 32F
		template<int order>
		void horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType);

		//vfilter Y-X loop
		template<int order, typename destT>
		void verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		//vfilter X-Y loop (tend to cause cache slashing)
		template<int order, typename destT>
		void verticalFilteringInnerXYK_XYLoop(const cv::Mat& src, cv::Mat& dst, const int borderType);
		//for 16F
		template<int order>
		void horizontalFilteringInnerXKdest16F(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYKsrc16F(const cv::Mat& src, cv::Mat& dst, const int borderType);

		__m256* computeZConvVFilter(const float* srcPtr, const int y, const int x, const int width, const int height, const int order, float* GCn, const int radius, const float G0, const int borderType);

		//for debug plot GkFk
		void verticalFilteringInnerXYKn_32F_Debug(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT3_AVX_32F(cv::Size imgSize, float sigmaS, int order);
		SpatialFilterSlidingDCT3_AVX_32F(const DCT_COEFFICIENTS method, const int dest_depth = -1, const SLIDING_DCT_SCHEDULE isInterBuff16F = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT3_AVX_32F();

		int getRadius(const double sigma, const int order) override;	
		virtual void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	//Sliding DCT 3 with AVX (64F)
	class CP_EXPORT SpatialFilterSlidingDCT3_AVX_64F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		bool isHBuff32F = false;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		double* GCn = nullptr;//G_k * C_n_k
		double* shift = nullptr;//0: 2*C_1, 1: G*C_R
		__m256d* buffVFilter = nullptr;
		__m256d* fn_hfilter = nullptr;
		__m128* buffHFilter32F = nullptr;

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);
		/*
		//horizontal transposed buffer(32F), output(32F)
		template<int>
		void horizontalFilteringInnerXLoadStore32FK(const cv::Mat& src, cv::Mat& dst);
		//horizontal transposed buffer(64F), output(32F)
		template<int>
		void horizontalFilteringInnerXStore32FK(const cv::Mat& src, cv::Mat& dst);
		template<int>
		void verticalFilteringInnerXYStore32FK(const cv::Mat& src, cv::Mat& dst);
		template<int>
		void verticalFilteringInnerXYLoadStore32FK(const cv::Mat& src, cv::Mat& dst);
		*/

		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);//naive DCT convolution

		//64F
		template<int>
		void horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int, typename destT>
		void verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT3_AVX_64F(cv::Size imgSize, double sigmaS, int order, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT);
		SpatialFilterSlidingDCT3_AVX_64F(const DCT_COEFFICIENTS method, const int dest_depth = -1, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT3_AVX_64F();

		int getRadius(const double sigma, const int order) override;
		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	//Sliding DCT 5 with AVX (32F)
	class CP_EXPORT SpatialFilterSlidingDCT5_AVX_32F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		float G0 = 0.f;
		float* GCn = nullptr;//G_k * C_n_k
		float* shift = nullptr;//0: 2*C_1, 1: G*C_R
		__m256* buffVFilter = nullptr;//size: width*(2*order+1)
		__m256* fn_hfilter = nullptr;//size: width*8

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);

		//Naive DCT Convolution: O(KR)
		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);

		//for 32F
		//hfilter (Default)
		template<int order>
		void horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		//non template order: hfilter (Default)
		void horizontalFilteringInnerXKn(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);

		//vfilter Y-X loop (Default)
		template<int order, typename destT>
		void verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYKReuseAll(const cv::Mat& src, cv::Mat& dst, int borderType);
		//non template order: vfilter Y-X loop (Default)
		template<typename destT>
		void verticalFilteringInnerXYKn(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);

		//vfilter X-Y loop (tend to cause cache slashing)
		template<int order, typename destT>
		void verticalFilteringInnerXYK_XYLoop(const cv::Mat& src, cv::Mat& dst, const int borderType);

		void verticalFilteringInnerXYK_XYLoop_Debug(const cv::Mat& src, cv::Mat& dst, int order, const int borderType);

		//for 16F
		template<int order>
		void horizontalFilteringInnerXKdest16F(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYKsrc16F(const cv::Mat& src, cv::Mat& dst, const int borderType);

		//for debug plot GkFk		
		__m256* computeZConvHFilter(const float* srcPtr, const int y, const int x, const int width, const int height, const int order, float* GCn, const int radius, const float G0, const int borderType);
		void horizontalFilteringInnerXKn_32F_Debug(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);
		__m256* computeZConvVFilter(const float* srcPtr, const int y, const int x, const int width, const int height, const int order, float* GCn, const int radius, const float G0, const int borderType);
		void verticalFilteringInnerXYKn_32F_Debug(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT5_AVX_32F(cv::Size imgSize, float sigmaS, int order);
		SpatialFilterSlidingDCT5_AVX_32F(const DCT_COEFFICIENTS method, const int dest_depth = -1, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT5_AVX_32F();

		int getRadius(const double sigma, const int order) override;
		virtual void body(const cv::Mat& src, cv::Mat& dst, const int borderType) override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	//Sliding DCT 5 with AVX512 (32F)
#ifdef CP_AVX_512
	class GaussianFilterSlidingDCT5_AVX512_32F : public GaussianFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;

		float norm;

		float* table = nullptr;
		float* shift = nullptr;

		__m512* buffVFilter = nullptr;
		__m512* buffHFilter = nullptr;

		__m512 patch[16];

		virtual void copyPatchHorizontalbody(cv::Mat& src, const int y);

		void horizontalFilteringInnerXK1(const cv::Mat& src, cv::Mat& dst);
		void verticalFilteringInnerXYK1(const cv::Mat& src, cv::Mat& dst);

		void horizontalFilteringInnerXK2(const cv::Mat& src, cv::Mat& dst);
		void verticalFilteringInnerXYK2(const cv::Mat& src, cv::Mat& dst);

		void horizontalFilteringInnerXK3(const cv::Mat& src, cv::Mat& dst);
		void verticalFilteringInnerXYK3(const cv::Mat& src, cv::Mat& dst);

		template<int order>
		void horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst);
		template<int order>
		void verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst);

		void horizontalFilteringInnerXKn(const cv::Mat& src, cv::Mat& dst);
		void verticalFilteringInnerXYKn(const cv::Mat& src, cv::Mat& dst);

	public:
		GaussianFilterSlidingDCT5_AVX512_32F(cv::Size imgSize, float sigmaS, int order);
		~GaussianFilterSlidingDCT5_AVX512_32F();

		void setRadius(const int rad, const bool isOptimize);
		void body(const cv::Mat& src, cv::Mat& dst);
	};
#endif
	//Sliding DCT 5 with AVX (64F)
	class CP_EXPORT SpatialFilterSlidingDCT5_AVX_64F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		double G0 = 0.0;
		double* GCn = nullptr;//G_k * C_n_k
		double* shift = nullptr;//0: 2*C_1, 1: G*C_R
		__m256d* buffVFilter = nullptr;
		__m256d* fn_hfilter = nullptr;

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);

		//naive DCT convolution
		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);

		template<int order>
		void horizontalFilteringInnerXStore32FK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order>
		void verticalFilteringInnerXYLoadStore32FK(const cv::Mat& src, cv::Mat& dst, const int borderType);

		//64F
		template<int order>
		void horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT5_AVX_64F(cv::Size imgSize, double sigmaS, int order, const bool isBuff32F = false);
		SpatialFilterSlidingDCT5_AVX_64F(const DCT_COEFFICIENTS method, const int dest_depth = -1, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT5_AVX_64F();

		int getRadius(const double sigma, const int order) override;
		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};


	//Sliding DCT 7 with AVX (32F)
	class CP_EXPORT SpatialFilterSlidingDCT7_AVX_32F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		float* GCn = nullptr;//G_k * C_n_k
		float* shift = nullptr;//0: 2*C_1, 1: G*C_R
		__m256* buffVFilter = nullptr;//size: width*2*(order+1)
		__m256* fn_hfilter = nullptr;//size: width*8

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);

		//Naive DCT Convolution: O(KR)
		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);//naive DCT convolution

		//for 32F
		template<int order>
		void horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		//vfilter Y-X loop
		template<int order, typename destT>
		void verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		//vfilter X-Y loop (tend to cause cache slashing)
		template<int order, typename destT>
		void verticalFilteringInnerXYK_XYLoop(const cv::Mat& src, cv::Mat& dst, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT7_AVX_32F(cv::Size imgSize, float sigmaS, int order);
		SpatialFilterSlidingDCT7_AVX_32F(const DCT_COEFFICIENTS method, const int dest_depth = -1, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT7_AVX_32F();

		int getRadius(const double sigma, const int order) override;	
		virtual void body(const cv::Mat& src, cv::Mat& dst, const int borderType) override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

	class CP_EXPORT SpatialFilterSlidingDCT7_AVX_64F : public SpatialFilterBase
	{
	protected:
		int dest_depth = -1;
		cv::Mat inter;
		bool isHBuff32F = false;
		SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		DCT_COEFFICIENTS dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;

		double* GCn = nullptr;//G_k * C_n_k
		double* shift = nullptr;//0: 2*C_1, 1: G*C_R
		__m256d* buffVFilter = nullptr;
		__m256d* fn_hfilter = nullptr;

		virtual void interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad = 0);

		void horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType);//naive DCT convolution

		//64F
		template<int order>
		void horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType);
		template<int order, typename destT>
		void verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType);

		void allocBuffer();
		void computeRadius(const int rad, const bool isOptimize);
	public:
		SpatialFilterSlidingDCT7_AVX_64F(cv::Size imgSize, double sigmaS, int order, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT);
		SpatialFilterSlidingDCT7_AVX_64F(const DCT_COEFFICIENTS method, const int dest_depth = -1, const SLIDING_DCT_SCHEDULE schedule = SLIDING_DCT_SCHEDULE::DEFAULT, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);
		~SpatialFilterSlidingDCT7_AVX_64F();

		int getRadius(const double sigma, const int order) override;	
		void body(const cv::Mat& src, cv::Mat& dst, const int borderType) override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};

#pragma endregion

#pragma region box
	class CP_EXPORT SpatialFilterBox : public SpatialFilterBase
	{
	protected:
		cv::Mat inter;
		int dest_depth = -1;

		cv::Ptr<cp::BoxFilterBase> box = nullptr;
		cp::BoxFilterMethod boxfilter_type = cp::BoxFilterMethod::OPENCV;
		//cp::BoxFilterMethod boxfilter_type = cp::BoxFilterMethod::SEPARABLE_VHI_AVX;

	public:
		SpatialFilterBox(const cp::BoxFilterMethod boxfilter_type, const int dest_depth);
		virtual void body(const cv::Mat& src, cv::Mat& dst, int borderType) override;
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT) override;
	};
#pragma endregion

#pragma region DCT
	class CP_EXPORT SpatialFilterDCT_AVX_32F : public SpatialFilterBase
	{
	private:
		cv::Mat inter;
		cv::Mat frec;
		int dest_depth = -1;

		float lutsigma = 0.f;
		int lutsizex = 0;
		int lutsizey = 0;
		float* LUTx = nullptr;
		float* LUTy = nullptr;
		float denormalclip = -sqrt(87.3365f);
		void allocLUT();
		void setGaussianKernel(const bool isForcedUpdate);
	public:
		SpatialFilterDCT_AVX_32F(cv::Size img_size, float sigma_space, int order);
		SpatialFilterDCT_AVX_32F(const int dest_depth);
		~SpatialFilterDCT_AVX_32F();

		void body(const cv::Mat& src, cv::Mat& dst, int borderType)override;
		//order is no meaning, border is always BORDER_REFLECT due to DCTII transform
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int border = cv::BORDER_REFLECT)override;
	};

	class CP_EXPORT SpatialFilterDCT_AVX_64F : public SpatialFilterBase
	{
	private:
		cv::Mat inter;
		cv::Mat frec;
		int dest_depth = -1;

		double lutsigma = 0.0;
		int lutsizex = 0;
		int lutsizey = 0;
		double* LUTx = nullptr;
		double* LUTy = nullptr;
		double denormalclip = -sqrt(87.3365);
		void allocLUT();

	public:
		SpatialFilterDCT_AVX_64F(cv::Size img_size, double sigma_space, int order);
		SpatialFilterDCT_AVX_64F(const int dest_depth);
		~SpatialFilterDCT_AVX_64F();

		void body(const cv::Mat& src, cv::Mat& dst, const int borderType)override;
		//order is no meaning, border is always BORDER_REFLECT due to DCT-II transform
		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int border = cv::BORDER_REFLECT)override;
	};
#pragma endregion

	//option 0: DCT_COEFFICIENTS::FULL_SEARCH_OPT, 1: DCT_COEFFICIENTS::FULL_SEARCH_NOOPT;
	CP_EXPORT cv::Ptr<cp::SpatialFilterBase> createSpatialFilter(const cp::SpatialFilterAlgorithm method, const int dest_depth, const SpatialKernel skernel, const int option = 0);

	//implement class
	//SpatialFilter(const cp::SpatialFilterAlgorithm method, const int depth, const SpatialKernel skernel = SpatialKernel::GAUSSIAN, const int option = 0)
	class CP_EXPORT SpatialFilter
	{
	protected:
		cv::Ptr<SpatialFilterBase> gauss = nullptr;
	public:
		SpatialFilter(const cp::SpatialFilterAlgorithm method, const int depth, const SpatialKernel skernel = SpatialKernel::GAUSSIAN, const int option = 0);

		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT);

		cp::SpatialFilterAlgorithm getMethodType();
		int getOrder();
		double getSigma();
		cv::Size getSize();
		int getRadius();
		void setFixRadius(const int r);
		void setIsInner(const int top, const int bottom, const int left, const int right);
	};

	class CP_EXPORT SpatialFilterTile
	{
		int thread_max = 0;
		cv::Size div;
		int depth = CV_32F;
		cv::Size tileSize;
		std::vector<cv::Mat> srcTile;
		std::vector<cv::Mat> dstTile;
		std::vector<cv::Ptr<cp::SpatialFilterBase>> gauss;
		void init(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const cv::Size div, const SpatialKernel spatial_kernel);
	public:
		SpatialFilterTile(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const cv::Size div, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);

		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType, const float truncateBoundary);
		void filterDoG(const cv::Mat& src, cv::Mat& dst, const double sigma1, const double sigma2, const int order, const int borderType, const float truncateBoundary);

		cv::Size getTileSize();
		void printParameter();
	};

	class CP_EXPORT SpatialFilterDoGTile
	{
		int thread_max = 0;
		cv::Size div;
		int depth = CV_32F;
		cv::Size tileSize;
		std::vector<cv::Mat> srcTile;
		std::vector<cv::Mat> srcTile2;
		std::vector<cv::Mat> dstTile;
		std::vector<cv::Ptr<cp::SpatialFilterBase>> gauss;
		std::vector<cv::Ptr<cp::SpatialFilterBase>> gauss2;
		cv::Mat buff;
		void init(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const cv::Size div, const SpatialKernel spatial_kernel);
	public:
		SpatialFilterDoGTile(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const cv::Size div, const SpatialKernel skernel = SpatialKernel::GAUSSIAN);

		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma1, const double sigma2, const int order, const int borderType, const float truncateBoundary);

		cv::Size getTileSize();
		void printParameter();
	};

	inline int get_xend_slidingdct(int start, int loopsize, int imgwidth, int step)
	{
		const int ret = get_loop_end(start, loopsize + start, step);
		return (ret <= imgwidth) ? ret : ret - step;
	}

	inline int get_hfilterdct_ystart(int height, int top, int bottom, int radius, const int simdUnrollSize)
	{
		int ystart = top - radius - 1;
		if (top - radius - 1 < 0)
		{
			ystart = 0;
		}
		return ystart;
	}

	inline int get_hfilterdct_yend(int height, int top, int bottom, int radius, const int simdUnrollSize)
	{
		int ylength = height - (top + bottom) + 2 * radius + 3;
		int ystart = top - radius - 1;
		int yend = height - bottom + radius + 2;
		if (top - radius - 1 < 0)
		{
			ystart = 0;
			ylength -= (radius + 1);
		}
		if (radius + 2 - bottom > 0)
		{
			ylength -= (radius + 2);
			yend = ystart + get_simd_ceil(ylength, simdUnrollSize);
		}
		return yend;
	}
}