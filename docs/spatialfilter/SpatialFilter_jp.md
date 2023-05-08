SpatialFilter.hpp
=================
線形空間フィルタの効率的な複数の実装を集めたクラスです．
現在は，グレイスケール専用です．

# class SpatialFilter

```cpp
	class CP_EXPORT SpatialFilter
	{
	protected:
		cv::Ptr<SpatialFilterBase> gauss = nullptr;
	public:
		SpatialFilter(const cp::SpatialFilterAlgorithm method, const int dest_depth, const SpatialKernel skernel = SpatialKernel::GAUSSIAN, const int dct_option = 0);

		void filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType = cv::BORDER_DEFAULT);

		cp::SpatialFilterAlgorithm getAlgorithmType();
		int getOrder();
		double getSigma();
		cv::Size getSize();
		int getRadius();

		void setFixRadius(const int r);
		void setIsInner(const int top, const int bottom, const int left, const int right);
	};
```

```
enum class SpatialFilterAlgorithm
	{
		IIR_AM,
		IIR_VYV,
		IIR_DERICHE,
		SlidingDCT1_AVX,
		SlidingDCT3_AVX,
		SlidingDCT5_AVX,
		SlidingDCT7_AVX,
		FIR_OPENCV_GAUSSIAN,
		FIR_OPENCV_GAUSSIAN64F,
		FIR_OPENCV_SEP2D,//call sepFilter2D
		FIR_OPENCV_FILTER2D,//call filter2D
		FIR_KAHAN,
		DCTFULL_OPENCV,
		FIR_SEPARABLE,//under debug
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
		SIZE,

		//some wrapper function does not support as follows
		IIR_VYV_NAIVE,
		IIR_DERICHE_NAIVE,
		IIR_AM_NAIVE,
	};
```

## Usage
コンストラクタで，アルゴリズムと出力のdepthタイプを指定して，`filter`メソッドで畳み込みを実行します．
選択可能なアルゴリズムは，`SpatialFilterAlgorithm`で指定可能です．
現在は`IIR_AM`から`BOX`まではテスト関数で検証済みです．それ以外は，引数の指定の仕方によっては落ちます（2023/5/9）．
また，`SpatialKernel`で畳み込みのカーネル重みを設定できるようにしていますが基本的にはまだ`SpatialKernel::GAUSSIAN`専用です．（2023/5/9）．
`filter`メソッドの引数は，入出力画像，ガウス重みのsigma, 近似次数か畳み込み半径クリップを指定するorder，境界領域オプションのborderTypeが指定できます．
	
いくつか実装してあるgetterで内部のパラメータが取得できます．
setterに相当するものは，filterメソッドでほぼ指定可能です．
ただし，半径rを強制的にセットする`setFixRadius`メソッドと，畳み込む画素のROIを設定する`setIsInner`があります．
filter関数には，畳み込み半径を指定するオプションがありません．これは，いくつかのアルゴリズムは自動的に畳み込み半径を決定するからです．
また，`setIsInner`で処理する画素の有効範囲領域を指定可能ですがいくつかの関数でしか動作せず，バグはたくさん残っています．コードの中身を把握してない状態では指定しないように（2023/5/9）．

以下，アルゴリズムの説明です．
* IIR_AM
	* SIMD実装された下記論文のIIRフィルタでガウシアンフィルタを実行します．
	* L.AlvarezandL.Mazorra,“Signal and image restoration using shock filters and an isotropic diffusion,” SIAM Journal on Numerical Analysis, vol.31, no.2, pp.590–605, 1994.
* IIR_VYV
	* SIMD実装された下記論文のIIRフィルタでガウシアンフィルタを実行します．近似次数orderは3,4,5のみ実装済みです．IIRの中では最も精度速度のバランスが良いです．
	* L. J. vanVliet, I. T. Young, and P. W. Verbeek, “Recursive gaussian derivative filters,” in Proceegings of International Conference on Pattern Recognition (ICPR), 1998.
* IIR_DERICHE
	* SIMD実装されたIIRガウシアンフィルタの古典である下記実装を実行します．近似次数orderは2,3,4のみ実装済みです．
	* R. Deriche, “Fast algorithms for low-level vision,” IEEE Transactions on Pattern Analysis Machine Intelligence, vol. 12, pp. 78–87, 1990.
* SlidingDCT1_AVX
	* SIMD実装された再帰型FIRフィルタのSlidingDCTのDCT-type I実装を実行します．効率はSlidingDCTの中で最も低いです．
	* T. Otsuka, N. Fukushima, Y. Maeda, K. Sugimoto, and S. Kamata, "Optimization of Sliding-DCT based Gaussian Filtering for Hardware Accelerator," in Proc. International Conference on Visual Communications and Image Processing (VCIP), Dec. 2020.
* SlidingDCT3_AVX
	* SlidingDCTのDCT-type III実装を実行します．次数が3以上ある時は効率はSlidingDCTの中で最も高いです．
* SlidingDCT5_AVX
	* SlidingDCTのDCT-type V実装を実行します．次数が2以下ある時は効率はSlidingDCTの中で最も高いです．
* SlidingDCT7_AVX
	* SlidingDCTのDCT-type VII実装を実行します．速度と精度のバランスが取れていますが，IIIとV以外を使う利点は今のところありません．
* FIR_OPENCV_GAUSSIAN,
	* OpenCVのガウシアンフィルタ関数を呼び出します．畳み込み半径rはorderｘsigmaで指定されます．通常はorder=3です．
* FIR_OPENCV_GAUSSIAN64F,
	* OpenCVのガウシアンフィルタ関数を強制的にdoubleで計算するように呼び出します．
* FIR_OPENCV_SEP2D,//call sepFilter2D
	* OpenCVのsepFilter2Dのカーネルにガウシアンカーネルを設定して呼び出します．
* FIR_OPENCV_FILTER2D,//call filter2D
	* OpenCVのfilter2Dのカーネルにガウシアンカーネルを設定して呼び出します．
* FIR_KAHAN
	* 畳み込みをKahanの総和アルゴリズムで行い，高精度な畳み込みを実現します．
* DCTFULL_OPENCV
	* 画像全体（フルサイズ）でDCT変換を行った後にガウス関数の重みを乗算して，逆DCT変換をして戻します．
* FIR_SEPARABLE,//under debug
	* sepFilter2Dの高速実装です．いくつも実装がありますが，デバッグ中でかなり不安定です．
* BOX
	* OpenCVのボックスフィルタを呼び出します．

## Test function
`testSpatialFilter.cpp`にサンプルコードがあります．以下コードの概要です．  
1. SpatialFilterクラスにアルゴリズムとデプス指定してインスタンスを生成します．
2. filterメソッドで関数呼び出します．
3. 呼び出しの1回目とそれ以降で速度が異なり，2回目以降は高速化しているのでその時間計測を分けています．
4. 精度は，PSNRで検証し，正解を半径9σのKahanのアルゴリズムを用いたdouble精度で畳み込んだものを正解としています．
	* なぜこのように設定したかはそのうち論文を書くので参考文献に追加します．（2023/5/9）．
5. borderの指定はFIR畳み込みは基本的に何でもよいが，DCTの場合は，DCT変換のアルゴリズムの制約でBORDER_REFLECTでないと境界の精度が低下します．

```cpp
void testSpatialFilter(Mat& src)
{
	string wname = "testSpatialFilter";
	ConsoleImage ci;
	namedWindow(wname);
	int algo = 10; createTrackbar("algorithm", wname, &algo, (int)SpatialFilterAlgorithm::BOX);	//(int)SpatialFilterAlgorithm::SIZE - 1
	int sigma10 = 20;  createTrackbar("sigma*0.1", wname, &sigma10, 200);
	int order = 5;  createTrackbar("order", wname, &order, 20);
	int type = 1;  createTrackbar("type", wname, &type, 2);
	int borderType = cv::BORDER_REFLECT; createTrackbar("border", wname, &borderType, 4);
	int key = 0;
	Mat show;
	Timer tfirst("", TIME_MSEC, false);
	Timer t("", TIME_MSEC, false);
	UpdateCheck ucs(sigma10, borderType);
	UpdateCheck uc(algo, order, type);
	Mat ref;

	while (key != 'q')
	{
		const bool isRefUpdate = ucs.isUpdate(sigma10, borderType);
		if (isRefUpdate)
		{
			SpatialFilter reff(SpatialFilterAlgorithm::FIR_KAHAN, CV_64F, SpatialKernel::GAUSSIAN, 0);
			reff.filter(src, ref, sigma10 * 0.1, 9, borderType);
		}
		if (uc.isUpdate(algo, order, type) || isRefUpdate)
		{
			tfirst.clearStat();
			t.clearStat();
		}

		SpatialFilterAlgorithm algorithm = SpatialFilterAlgorithm(algo);
		int desttype = (type == 0) ? CV_8U : (type == 1) ? CV_32F : CV_64F;
		
		SpatialFilter sf(algorithm, desttype, SpatialKernel::GAUSSIAN, 0);//const DCT_COEFFICIENTS dct_coeff = (option == 0) ? DCT_COEFFICIENTS::FULL_SEARCH_OPT : DCT_COEFFICIENTS::FULL_SEARCH_NOOPT;
		
		tfirst.start();
		sf.filter(src, show, sigma10 * 0.1, order, borderType);
		tfirst.pushLapTime();
		t.start();
		sf.filter(src, show, sigma10 * 0.1, order, borderType);
		t.pushLapTime();

		imshowScale(wname, show);

		ci("%d: sigma = %f", t.getStatSize(), sigma10 * 0.1);
		ci("algorith: " + cp::getAlgorithmName(algorithm));
		ci("order:  %d", cp::clipOrder(order, algorithm));
		ci("type: " + cp::getDepthName(desttype));
		ci("border: " + cp::getBorderName(borderType));
		ci("TIME: %f ms (1st)", tfirst.getLapTimeMedian());
		ci("TIME: %f ms (2nd)", t.getLapTimeMedian());
		ci("PSNR: %f dB", getPSNR(ref, show));
		ci.show();
		key = waitKey(1);
	}
}
```
