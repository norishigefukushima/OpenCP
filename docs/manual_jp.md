Reference Manual
================
Documents in Japanese for Lab's members  
（現在，現状研究室内部専用：以下，研究室内向けメッセージ）  
マニュアルは，みんなで更新していってください．  
更新するのを期待して待っても，ほぼ更新されない更新頻度です．  
（バグを発見したときについでに更新するくらい）  

[Todo] がついているものはドキュメントの追記が必要．

# core
## [Todo] arithmetic.hpp
画像の点の演算

```cpp
CP_EXPORT void pow_fmath(const float a, const cv::Mat& src, cv::Mat& dest);
CP_EXPORT void pow_fmath(const cv::Mat& src, const float a, cv::Mat& dest);
CP_EXPORT void pow_fmath(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dest);
CP_EXPORT void compareRange(cv::InputArray src, cv::OutputArray destMask, const double validMin, const double validMax);
CP_EXPORT void setTypeMaxValue(cv::InputOutputArray src);
CP_EXPORT void setTypeMinValue(cv::InputOutputArray src);

	//a*x+b
CP_EXPORT void fmadd(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);
	//a*x-b
CP_EXPORT void fmsub(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);
	//-a*x+b
CP_EXPORT void fnmadd(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);
	//-a*x-b
CP_EXPORT void fnmsub(cv::Mat& a, cv::Mat& x, cv::Mat& b, cv::Mat& dest);

	//dest=src>>shift, lostbit=src-dest<<shift
CP_EXPORT void bitshiftRight(cv::InputArray src, cv::OutputArray dest, cv::OutputArray lostbit, const int shift);
	//src>>shift
CP_EXPORT void bitshiftRight(cv::InputArray src, cv::OutputArray dest, const int shift);

CP_EXPORT double average(const cv::Mat& src, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0, const bool isNormalize = true);
CP_EXPORT void average_variance(const cv::Mat& src, double& ave, double& var, const int left = 0, const int right = 0, const int top = 0, const int bottom = 0, const bool isNormalize = true);
```

## [Todo] bitconvert.hpp
`src.convertTo`に頼らない型変換．

```cpp
CP_EXPORT void cvt32F8U(const cv::Mat& src, cv::Mat& dest);
CP_EXPORT void cvt64F8U(const cv::Mat& src, cv::Mat& dest);

CP_EXPORT void cvt8U32F(const cv::Mat& src, cv::Mat& dest, const float amp);
CP_EXPORT void cvt8U32F(const cv::Mat& src, cv::Mat& dest);
	
CP_EXPORT void cvt32F16F(cv::Mat& srcdst);
```
## [Todo] bitconvertDD.hpp
doubledouble型の型変換．こっちはconvertToがそもそもない．

```cpp
CP_EXPORT void cvtMattoDD(const cv::Mat& src, doubledouble* dest);
CP_EXPORT void cvtDDtoMat(const doubledouble* src, cv::Mat& dest);
CP_EXPORT void cvtDDtoMat(const doubledouble* src, const cv::Size size, cv::Mat& dest, const int depth);
```

## checkSameImage.hpp
ランダムサンプルで点を選び，2つの画像が同一画像かどうか高速に確認する関数群．
* [class CheckSameImage](checkSameImage.md "class CheckSameImage")
* [checkSameImage](checkSameImage.md "checkSameImage")

## concat.hpp
画像の連結・分解の関数群
* [concat/concatMerge](concat_jp.md "#concat/concatMerge")
* [concatSplit](concat_jp.md "#concatSplit")
* [concatExtract](concat_jp.md "#")

## consoleImage.hpp
printf等のconsole出力を画像として出力するためのクラスConsoleImage．

* [class ConsoleImage](ConsoleImage_jp.md "#class ConsoleImage")

## copyMakeBorder.hpp
cv::copyMakeBorderを高速化した関数群
* [copyMakeBorderReplicate](copyMakeBorder_jp.md "#copyMakeBorderReplicate")
* [splitCopyMakeBorder](copyMakeBorder_jp.md "#splitCopyMakeBorder")

## count.hpp
画像の画素の属性をカウントする関数群
 * [countNaN](count_jp.md "#countNaN")
 * [countInf](count_jp.md "#countInf")
 * [countDenormalizedNumber](count_jp.md "#countDenormalizedNumber")
 * [countDenormalizedNumberRatio](count_jp.md "#countDenormalizedNumberRatio")

## crop.hpp
画像の切り抜き（クロップ）用関数群
 * [cropZoom](crop_jp.md "#cropZoom")
 * [cropZoomWithBoundingBox](crop_jp.md "#cropZoomWithBoundingBox")
 * [cropZoomWithSrcMarkAndBoundingBox](crop_jp.md "#cropZoomWithSrcMarkAndBoundingBox")
 * [cropZoom](crop_jp.md "#cropCenter")
 * [guiCropZoom](crop_jp.md "#guiCropZoom")

## [Todo] csv.hpp
CSVを走査するクラス

```cpp
class CP_EXPORT CSV
	{
		FILE* fp = NULL;
		bool isTop;
		long fileSize;
		std::string filename;
		bool isCommaSeparator = true;
	public:
		std::vector<double> argMin;
		std::vector<double> argMax;
		std::vector<std::vector<double>> data;
		std::vector<bool> filter;
		int width;
		void setSeparator(bool flag);//true:","comma, false, " "space
		void findMinMax(int result_index, bool isUseFilter, double minValue, double maxValue);
		void initFilter();
		void filterClear();
		void makeFilter(int index, double val, double emax = 0.00000001);
		void readHeader();
		void readData();

		void init(std::string name, bool isWrite, bool isClear);
		CSV();
		CSV(std::string name, bool isWrite = true, bool isClear = true);
		~CSV();
		void write(std::string v);
		void write(double v);
		void write(int v);
		void end();
	};

	void CP_EXPORT writeCSV(std::string name, cv::InputArray src);
```

## [Todo] draw.hpp

描画関数
```cpp
	CP_EXPORT void triangle(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void triangleinv(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void diamond(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void pentagon(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1);
	CP_EXPORT void drawPlus(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawTimes(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_typee = 8, int shift = 0);
	CP_EXPORT void drawAsterisk(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);
	CP_EXPORT void drawGrid(cv::InputOutputArray src, cv::Point crossCenter = cv::Point(0, 0), cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);//when crossCenter = Point(0, 0), draw grid line crossing center point
	CP_EXPORT void drawGridMulti(cv::InputOutputArray src, cv::Size division = cv::Size(2, 2), cv::Scalar color = COLOR_WHITE, int thickness = 1, int line_type = 8, int shift = 0);
	
	//copyMakeBorder without boundary expansion: erase step pixels and then copyMakeBorder with step
	CP_EXPORT void eraseBoundary(const cv::Mat& src, cv::Mat& dest, const int step, const int border = cv::BORDER_DEFAULT);
```

## [Todo] fftinfo.hpp
画像をFFTして表示する．

```cpp
CP_EXPORT void imshowFFT(std::string wname, cv::InputArray src, const float amp = 0.f);
```

## [Todo] highguiex.hpp
Qtで表示するテキストのサイズを取得する．

```cpp
	CP_EXPORT cv::Size getTextSizeQt(std::string message, std::string font, const int fontSize);
	CP_EXPORT cv::Mat getTextImageQt(std::string message, std::string font, const int fontSize, cv::Scalar text_color = cv::Scalar::all(0), cv::Scalar background_color = cv::Scalar(255, 255, 255, 0), bool isItalic = false);
```

## histogram.hpp
ヒストグラムの描画関数群
 * [drawHistogramImage](histogram_jp.md "#drawHistogramImage")
 * [drawHistogramImageGray](histogram_jp.md "#drawHistogramImageGray")
 * [drawAccumulateHistogramImage](histogram_jp.md "#drawAccumulateHistogramImage")
 * [drawAccumulateHistogramImageGray](histogram_jp.md "#drawAccumulateHistogramImageGray")
 * [guiLocalDiffHistogram](histogram_jp.md "#guiLocalDiffHistogram")

## [Todo] imagediff.hpp
画像のdiffを取る

```cpp
	CP_EXPORT void diffshow(std::string wname, cv::InputArray src, cv::InputArray ref, const double scale = 1.0);
	CP_EXPORT void guiDiff(cv::InputArray src, cv::InputArray ref, const bool isWait = true, std::string wname = "gui::diff");
	CP_EXPORT void guiCompareDiff(cv::InputArray before, cv::InputArray after, cv::InputArray ref, std::string name_before = "before", std::string name_after = "after", std::string wname = "gui::compared_iff");
	CP_EXPORT void guiAbsDiffCompareGE(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareLE(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareEQ(const cv::Mat& src1, const cv::Mat& src2);
	CP_EXPORT void guiAbsDiffCompareNE(const cv::Mat& src1, const cv::Mat& src2);
```

## imshowExtension.hpp
imshowの拡張．
* [imshowNormalize](imshowExtension_jp.md "#imshowNormalize")
* [imshowScale](imshowExtension_jp.md "#imshowScale")
* [imshowResize](imshowExtension_jp.md "#imshowResize")
* [imshowCountDown](imshowExtension_jp.md "#imshowCountDown")

## kmeans.hpp
* [class KMeans](kmeans_jp.md "#class KMeans")
* [kmeans](kmeans_jp.md "#kmeans")

## [Todo] maskoperation.hpp
矩形のマスクを作る．

```cpp
	CP_EXPORT void addBoxMask(cv::Mat& mask, const int boundx, const int boundy);//overwrite box mask
	CP_EXPORT cv::Mat createBoxMask(const cv::Size size, const int boundx, const int boundy);//create box mask
	CP_EXPORT void setBoxMask(cv::Mat& mask, const int boundx, const int boundy);//clear mask and then set box mask
```

## matinfo.hpp
Matの中身の情報を表示するデバッグ関数．  
`print_matinfo`はマクロ展開することで，引数も表示する．  
* [showMatInfo](matinfo_jp.md "#showMatInfo")
* [print_matinfo](matinfo_jp.md "#print_matinfo")
* [print_matinfo_detail](matinfo_jp.md "#print_matinfo_detail")

## noise.hpp
ノイズを付与する関数
* [addNoise](noise_jp.md "#addNoise")
* [addJPEGNoise](noise_jp.md "#addJPEGNoise")

## [Todo] plot.hpp
プロット関数

```cpp
class CP_EXPORT Plot
class CP_EXPORT GNUPlot
class CP_EXPORT Plot2D
class CP_EXPORT RGBHistogram
```

## [Todo] randomizedQueue.hpp
乱択Queueのクラス

```cpp
class CP_EXPORT RandomizedQueue
```

## stat.hpp
統計情報を計算するクラス

* [class Stat](Stat_jp.md "#class Stat")

## [Todo] stencil.hpp

何かに統合したほうがいいかも

```cpp
	CP_EXPORT void mergeFromGrid(std::vector<cv::Mat>& src, cv::Size beforeSize, cv::Mat& dest, cv::Size grid, int borderRadius);
	CP_EXPORT void splitToGrid(const cv::Mat& src, std::vector<cv::Mat>& dest, cv::Size grid, int borderRadius);

	CP_EXPORT void mergeHorizon(const std::vector<cv::Mat>& src, cv::Mat& dest);
	CP_EXPORT void splitHorizon(const cv::Mat& src, std::vector<cv::Mat>& dest, int num);
```

## [Todo] tiling.hpp

タイリング用．いろいろデバッグ途中

```cpp
	//get online image size
	CP_EXPORT cv::Size getTileAlignSize(const cv::Size src, const cv::Size div_size, const int r, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1);
	CP_EXPORT cv::Size getTileSize(const cv::Size src, const cv::Size div_size, const int r);

	//create a divided sub image
	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	CP_EXPORT void cropSplitTile(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType= cv::BORDER_DEFAULT);
	CP_EXPORT void cropSplitTileAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	//set a divided sub image to a large image
	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest,      const cv::Rect roi, const int top, const int left);
	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest,      const cv::Rect roi, const int r);
	CP_EXPORT void pasteTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int left_multiple = 1, const int top_multiple = 1);

	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int top, const int left);
	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r);
	CP_EXPORT void pasteTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int left_multiple = 1, const int top_multiple = 1);

	//split an image to sub images in std::vector 
	CP_EXPORT void divideTiles(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void divideTilesAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int left_multiple = 1, const int top_multiple = 1);

	//merge subimages in std::vector to an image
	CP_EXPORT void conquerTiles(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r);
	CP_EXPORT void conquerTilesAlign(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r, const int left_multiple = 1, const int top_multiple = 1);
```

## timer.hpp
タイマーのクラス

* [class Timer](Timer_jp.md "#class Timer")
* [class DestinationTimePrediction](Timer_jp.md "#class DestinationTimePrediction")

```cpp
class CP_EXPORT Timer
class CP_EXPORT DestinationTimePrediction
```
## updateCheck.hpp
パラメータの更新確認クラス．
* [class UpdateCheck](UpdateCheck_jp.md "#class UpdateCheck")

## [Todo] video.hpp
動画を表示するだけ．

```cpp
CP_EXPORT void guiVideoShow(std::string wname);
```

## VideoSubtitle.hpp
* [class VideoSubtitle](VideoSubtitle_jp.md "#class VideoSubtitle")

## [Todo] yuvio.hpp
YUVファイルの読み書き．

```cpp
class CP_EXPORT YUVReader
	{
		FILE* fp;
		int framemax;
		char* buff;
		bool isloop;

		int yuvSize;

	public:
		int width;
		int height;
		int imageSize;
		int imageCSize;
		int frameCount;

		void init(std::string name, cv::Size size, int frame_max);
		YUVReader(std::string name, cv::Size size, int frame_max);
		YUVReader();
		~YUVReader();

		void readNext(cv::Mat& dest);
		bool read(cv::Mat& dest, int frame);
	};

	CP_EXPORT void readYUVGray(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void readYUV2BGR(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void writeYUVBGR(std::string fname, cv::InputArray src);
	CP_EXPORT void writeYUVGray(std::string fname, cv::InputArray src);
	CP_EXPORT void readY16(std::string fname, cv::OutputArray dest, cv::Size size, int frame);
	CP_EXPORT void writeYUV(cv::Mat& InputArray, std::string name, int mode = 1);
```


# imgprog
## blend.hpp
2枚の画像の合成関数群
* [alphaBlend](blend_jp.md "#alphaBlend")
* [alphaBlendFixedPoint](blend_jp.md "#alphaBlendFixedPoint")
* [guiAlphaBlend](blend_jp.md "#guiAlphaBlend")
* [dissolveSlideBlend](blend_jp.md "#dissolveSlideBlend")
* [guiDissolveSlideBlend](blend_jp.md "#guiDissolveSlideBlend")

## [Todo] color.hpp

```cpp

```

## contrast.hpp
トーンカーブによるコントラスト強調の関数群
* [convert](contrast_jp.md "#convert")
* [cenvertCentering](contrast_jp.md "#cenvertCentering")
* [contrastSToneExp](contrast_jp.md "#contrastSToneExp")
* [contrastGamma](contrast_jp.md "#contrastGamma")
* [quantization](contrast_jp.md "#quantization")
* [guiContrast](contrast_jp.md "#guiContrast")

## detailEnhancement.hpp
詳細強調の関数
* [detailEnhancementBox](detailEnhancement_jp.md "#detailEnhancementBox")
* [detailEnhancementGauss](detailEnhancement_jp.md "#detailEnhancementGauss")
* [detailEnhancementBilateral](detailEnhancement_jp.md "#detailEnhancementBilateral")
* [detailEnhancementGuided](detailEnhancement_jp.md "#detailEnhancementGuided")

## [Todo] diffPixel.hpp

```cpp

```

## [Todo] hazeRemove.hpp

```cpp

```

## [Todo] iterativeBackProjection.hpp

```cpp

```

## metrics.hpp
画質評価関数群
 * [getPSNR](metrics_jp.md "#getPSNR")
 * [getPSNR_PRECISION](metrics_jp.md "#getPSNR_PRECISION")
 * [getPSNR_CHANNEL](metrics_jp.md "#getPSNR_CHANNEL")
 * [class PSNRMetrics](metrics_jp.md "#class PSNRMetrics")
 * [localPSNRMap](metrics_jp.md "#localPSNRMap")
 * [guiLocalPSNRMap](metrics_jp.md "#guiLocalPSNRMap")
 * [getMSE](metrics_jp.md "#getMSE")
 * [getInacceptableRatio](metrics_jp.md "#getInacceptableRatio")
 * [getEntropy](metrics_jp.md "#getEntropy")
 * [getTotalVariation](metrics_jp.md "#getTotalVariation")
 * [isSameMat](metrics_jp.md "#isSameMat")
 * SSIM：関数修正中

## [Todo] metricsDD.hpp
double-double精度

```cpp

```

## [Todo] ppmx.hpp
ppmファイルの読み書き

```cpp

```

## [Todo] shiftimage.hpp

```cpp

```

## [Todo] shiftimage.hpp

```cpp

```
## [Todo] speckle.hpp

```cpp

```

# filter
## guidedFilter.hpp
ガイデットフィルタの関数
* [guidedImageFilter](guidedFilter_jp.md "#guidedImageFilter")
* [class GuidedImageFilter](guidedFilter_jp.md "#class GuidedImageFilter")
* [class GuidedImageFilterTiling](guidedFilter_jp.md "#GuidedImageFilterTiling")

## weightedHistogramFilter.hpp
weighted histogram filterの関数
* [weightedHistogramFilter](weightedHistogramFilter_jp.md "#weightedHistogramFilter")
* [weightedWeightedHistogramFilter](weightedHistogramFilter_jp.md "#weightedWeightedHistogramFilter")
* [weightedModeFilter/weightedModeFilter](weightedHistogramFilter_jp.md "#weightedModeFilter/weightedMedianFilter")
* [weightedWeightedModeFilter/weightedWeightedModeFilter](weightedHistogramFilter_jp.md "#weightedWeightedModeFilter/weightedWeightedMedianFilter")

# stereo
## StereoBase.hpp
ステレオマッチングの関数
* [class StereoBase](StereoBase.md)
## StereoEval.hpp
ベンチマーク用の評価関数
* [calcBadPixel](StereoEval.md "#calcBadPixel")
* [createDisparityALLMask](StereoEval.md "#createDisparityALLMask")
* [createDisparityNonOcclusionMask](StereoEval.md "#createDisparityNonOcclusionMask")
* [class calcBadPixel](StereoEval.md "#class StereoEval")

# memo
Optimization
* Naive：ほとんど最適化されていない素のC++で書かれたコード．
* OpenCV：OpenCVの最適化された処理を可能な限り使った
* SSE：SSEを使って最適化（昔作った関数で更新がされていない．）
* AVX：AVX/AVX2を使って最適化
* AVX512:現在はコメントアウト
* full optimized:最高レベルの最適化
* single:シングルスレッドの挙動のみ
* parallel：OpenCVのparallel_forを使ったマルチコア実装．
* openmp：OpenMPのparallel_for．コンパイル時コード生成になるため，最大パフォーマンスのためにはライブラリをコンパイルしなおす必要あり．
* コメントなし：NaiveかOpenCVのどちらかと思われる．

# Todo
優先すべきドキュメントは下記のデバッグと開発に頻繁に使用する関数群

* diff
* Plot
* inlineSIMDFunctions.hpp
* inlineMathFunctions.hpp
* inlineCVFunctions.hpp
* onelineCVFunctions.hpp
* draw
