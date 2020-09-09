StereoBase.hpp
================

このクラスのテスト関数は以下にあります．
```cpp
void testStereoBase();
```

また，OpenCVのステレオマッチング関数を呼び出す類似のテスト関数は以下です．
```cpp
void testCVStereoBM();
void testCVStereoSGBM();
```

# class StereoBMBase
処理の本体は下記メソッドです．  
呼び出される主要な関数を下記に示します．
```cpp
void StereoBase::matching(Mat& leftim, Mat& rightim, Mat& destDisparityMap)
{
	computeGuideImageForAggregation(leftim);
	computePrefilter(leftim, rightim);
	computePixelMatchingCost(d, DSI[i]);
	if (isFeedback)addCostIterativeFeedback(DSI[i], d, destDisparityMap, feedbackFunction, feedbackClip, feedbackAmp);
	computeCostAggregation(DSI[i], DSI[i], guide);
	computeWTA(DSI, dest);//最小コストの値を取る
	//postprocessings
	//...
}
```
## 1. ガイド画像の処理とプレフィルタ
```cpp
//leftimをグレイスケールに変換したり，多少のフィルタを掛けたりする
computeGuideImageForAggregation(leftim);

//* カラー画像の場合RGBにスプリット．
//* X方向のSobelフィルタ min(2*preFilterCap, Sobel(x)．+preFilterCap)
//* CENSUS変換するなどのマッチングに必要な画像を生成する．
prefilter(leftim, rightim);
```

## 2. マッチングコスト計算
`computePixelMatchingCost(d, DSI[i])`内で，以下のオプションが選択可能．
出力は`uchar (CV_8U)`．
```cpp
enum Cost
{
	SD,
	SDColor,
	SDEdge,
	SDEdgeColor,
	SDEdgeBlend,
	SDEdgeBlendColor,
	AD,
	ADColor,
	ADEdge,
	ADEdgeColor,
	ADEdgeBlend,
	ADEdgeBlendColor,
	BT,
	BTColor,
	BTEdge,
	BTEdgeColor,
	BTEdgeBlend,
	BTEdgeBlendColor,
	BTFull,
	BTFullColor,
	BTFullEdge,
	BTFullEdgeColor,
	BTFullEdgeBlend,
	BTFullEdgeBlendColor,
	CENSUS3x3,
	CENSUS3x3Color,
	CENSUS5x5,
	CENSUS5x5Color,
	CENSUS7x5,
	CENSUS7x5Color,
	CENSUS9x1,
	CENSUS9x1Color,
	//Pixel_Matching_SAD_TextureBlend,
	//Pixel_Matching_BT_TextureBlend,

	Pixel_Matching_Method_Size
};
//上記のgetter
std::string getCostMethodName(const PixelMatching method);
//上記のsetter
void setCostMethod(const PixelMatching method);
```

**例**（L,R：左右の画像，EL，ER：左右のエッジ画像）

* AD: 絶対値誤差（偶数番号に格納）  
	* min(|L-R|, pixelMatchErrorCap)
* ADEdge: エッジ画像の絶対値誤差（奇数番号に格納）  
	* min(|EL-ER|, pixelMatchErrorCap)
* ADEdgeBlend: 上記二つのコストをαブレンド
	* a*min(|L-R|, pixelMatchErrorCap)+(1-a)*min(|EL-ER|, pixelMatchErrorCap)
* SD:二乗誤差（エッジ，ブレンドの場合も宇組む）
	* min((L-R)^2, pixelMatchErrorCap)
* BT: Birchfield and Tomasi
	* サブピクセルに強いコスト（左画像のみ）
* BTFull: 左右どちらにもBTを適用したもの
	* なぜか精度が下がる
* CENSUS: CENSUS変換．
	* `uchar`で収まる8画素のセットと`int`で収まるいくつかのセットを用意

カラーの場合は，RGB情報をどのように扱うかが選択可能
```cpp
enum ColorDistance
{
	ADD,
	AVG,
	MIN,
	MAX,
	ColorDistance_Size
};
//上記のgetter
std::string getCostColorDistanceName(ColorDistance method);
//上記のsetter
void setCostColorDistance(const ColorDistance method);
```
例えば，ADColorは色の絶対値誤差の関数であり，その色を下記オプションで計算可能．
* ADD：RGBの総和
* AVG：RGBの平均値
* MIN：最小値
* MAX：最大値

また，このコスト計算はフィードバックオプションを追加することが可能．
すでに推定済みの視差画像を使ってその値でコスト関数を弱く拘束する．
```cpp
if (isFeedback)addCostIterativeFeedback(DSI[i], d, destDisparityMap, feedbackFunction, feedbackClip, feedbackAmp);
//functionType: 0:EXP, 1:L1, 2:L2
inline float distance_functionEXP(float diff, float clip)
{
	return 1.f - exp(diff * diff / (-2.f * clip * clip));
}
inline float distance_functionL1(float diff, float clip)
{
	return min(abs(diff), clip);
}
inline float distance_functionL2(float diff, float clip)
{
	return min(diff * diff, clip * clip);
}
```
詳しくは，以下の論文の参照のこと．  
[T. Matsuo, S. Fujita, N. Fukushima, and Y. Ishibashi, "Efficient edge-awareness propagation via single-map filtering for edge-preserving stereo matching," in Proc. IS&T/SPIE Electronic Imaging, Three-Dimensional Image Processing, Measurement, and Applications, 9393-27, Feb. 2015.](https://fukushima.web.nitech.ac.jp/paper/2015_spie_ei_matsuo.pdf)

### 備考
CENSUSはブレンドがない．（作ってもよいかも．ただしハミング距離がAVX512からしかSIMD命令がない．）

## 3. コストアグリゲーション
`computeCostAggregation(DSI[i], DSI[i], guide)`では以下のアグリゲーションが可能．
```cpp
enum Aggregation
{
	Box,//ブロックマッチング
	BoxShiftable,//シフタブルブロックマッチング
	Gaussian,//ガウシアン窓のマッチング
	GaussianShiftable,//ガウシアン窓のシフタブルマッチング
	Guided,//ガイデットフィルタによるマッチング．プレフィルタで生成したguide画像を使用
	CrossBasedBox,//クロスベースのボックスフィルタによるアグリゲーション．プレフィルタで生成したguide画像を使用．
	Bilateral,//バイラテラルフィルタによるマッチング．プレフィルタで生成したguide画像を使用

	Aggregation_Method_Size
};
//上記のgetter
std::string getAggregationMethodName(const Aggregation method);
//上記のsetter
void setAggregationMethod(const Aggregation method);
```

## 4. 最適化・
`computeWTA(DSI, dest)`でアグリゲーションしたコストの最小値をとることができる．
なお，SGM関数である`computeOptimizeScanline()`を事前に呼べばSGMになるはずだが，デバッグが不十分．

## 5. ポストフィルタ
下記の順序でポストフィルタを実行する．

### 1. ユニークネスフィルタ
エラーの最最小と2番目のエラーの差を見て曖昧性の高い物をはじくフィルタ．OpenCVと同じ実装．
```cpp
uniquenessFilter(minCostMap, dest);
```
### 2. サブピクセル補間
まず，コスト関数をみて双曲線補間もしくは線形当てはめで補間して出力する．  
出力の視差は`short`型なため，出力は16倍した整数で出力．
```cpp
enum SUBPIXEL
{
	NONE,//無し
	QUAD,//双曲線
	LINEAR,　//線形
	SUBPIXEL_METHOD_SIZE
};
//上記のgetter
std::string getSubpixelInterpolationMethodName(const SUBPIXEmethod);
//上記のsetter
void setSubpixelInterpolationMethodName(const SUBPIXEL method);
//呼び出し関数
subpixelInterpolation(dest, subpixMethod);//サブピクセル補間
```
### 3. レンジフィルタ平滑化によるノイズ除去とサブピクセル補間
次に，バイナリレンジフィルタ（εフィルタ）で視差画像を平滑化．
これは，バイラテラルフィルタの空間重み無し（ボックスフィルタ），レンジ重みがガウス関数ではなく，二値の閾値処理の関数．
```cpp
binalyWeightedRangeFilter(dest, dest, subboxWindowR, subboxRange);
```
### 4. LRチェック
LRチェックによる，整合が取れない視差値の除去を行う．左右の視差を計算せずに1枚の視差画像だけから推定する．
`WITH_MINCOST`のほうのDSIをWTAしたときのMINCOSTを使うが精度が高いのでそちらがデフォルト．速度はほぼ変わらないのでこちらを使えばよい．

```cpp
enum class LRCHECK
{
	NONE,
	WITH_MINCOST,
	WITHOUT_MINCOST,
	LRCHECK_SIZE
};
//上記のgetter
std::string getLRCheckMethodName(const LRCHECK method);
//上記のsetter
void setLRCheckMethod(const LRCHECK method);

fastLRCheck(minCostMap, dest);//1枚画像のLRチェック
fastLRCheck(dest);//minCostを使わないバージョン
```
### 5. LRチェックの修正
左サイドのバウンダリは値が怪しいため，無視するためのオプション．
有効範囲外は0クリアする．
使用はどちらでもよい．

### 6. 最小コストフィルタ
視差値が１以上は慣れている時に，隣り合うピクセルの最小コストが小さい視差値で置き換える．

DPなど以外はほとんど役に立たないはず．
デフォルトは呼んでいない．
```cpp
minCostFilter(minCostMap, dest);
```

### 7. スペックルフィルタ
スペックルを除去するフィルタ．OpenCVの実装と同じ．
```cpp
filterSpeckles(dest, 0, speckleWindowSize, speckleRange, specklebuffer);
```


### 8. ホールフィリング
ユニークネス，LRチェック，スペックルフィルタで無効化された視差値を周囲の画素値から補間します．
現在は，何もしない`NONE`かスキャンライン上を操作し，最小の値を持つ視差値で補間する`NEAREST_MIN_SCANLINE`が有効です．
これは，主にオクルージョン領域を埋めるために使われるフィルタです．

それ以外は動作未確認です．
基本的には，バウンダリを画像のエッジに合わせて動かすものですが，調整されていません．

```cpp
enum HOLE_FILL
		{
			NONE,
			NEAREST_MIN_SCANLINE,
			METHOD2,
			METHOD3,
			METHOD4,

			FILL_OCCLUSION_SIZE
		};
		std::string getHollFillingMethodName(const HOLE_FILL method);
		void setHoleFiillingMethodName(const HOLE_FILL method);
		double getValidRatio();//get valid pixel ratio before hole filling
```
### 9. リファインメント
視差画像のリファインメントフィルタを実行します．
ガイデットフィルタ，ジョイントバイラテラルフィルタ＋ジョイントニアエストフィルタもしくはWeighted Mode Filterによるリファインメントを行います．
重み付きのリファインメントは，ガウシアンフィルタとの差分で重みを計算します．
```cpp
enum class REFINEMENT
		{
			NONE,
			GIF_JNF,//guided image filter + joint nearest filter
			WGIF_GAUSS_JNF,//weighted guided image filter + joint nearest filter
			JBF_JNF,//joint bilateral filter + joint nearest filter
			WJBF_GAUSS_JNF,//weighted joint bilateral filter + joint nearest filter
			WMF,//weighted mode filter

			REFINEMENT_SIZE
		};
		std::string getRefinementMethodName(const REFINEMENT method);
		//param<0: unchange parameter
		void setRefinementMethod(const REFINEMENT refinementMethod, const int refinementR = -1, const float refinementSigmaRange = -1.f, const float refinementSigmaSpace = -1.f, const int jointNearestR = -1);
```

詳細は以下を参照してください．
* [WJBF+JNF] T. Matsuo, N. Fukushima, and Y. Ishibashi, "Weighted joint bilateral filter with slope depth compensation filter for depth map refinement," in Proc. International Conference on Computer Vision Theory and Applications (VISAPP), Feb. 2013.
* [WMF] D. Min, J. Lu, and M. N. Do. "Depth video enhancement based on weighted mode filtering." IEEE Transactions on Image Processing 21(3), pp. 1176-1190, 2011.


# StereoBMSimple::gui
下記メソッドでGUIによるパラメータ調整が可能．
`Ctrl+p`でパラメータ調整バーが呼び出せる．

```cpp
void StereoBMSimple::gui(Mat& leftim, Mat& rightim, Mat& dest, StereoEval& eval)
```

## 出力のイメージ．
視差画像，情報画面，パラメータ調整バー，リファインメント時の重み画像，プロファイル曲線，サブピクセル補間のヒストグラム，コスト関数が表示されます．
情報画面に書いてある(1)や(i)などは対応するキーボードショートカットです．

<img src="./docimg/outputimage_stereobase.png" width="800px">

# Reference

* BT
	* S. Birchfield and C. Tomasi. A pixel dissimilarity measure
that is insensitive to image sampling. TPAMI, 20(4):401–
406, 1998.
	* [Birchfield–Tomasi dissimilarity](https://en.wikipedia.org/wiki/Birchfield%E2%80%93Tomasi_dissimilarity)
* Pixel cost function survey
	* https://www.cs.middlebury.edu/~schar/papers/evalCosts_cvpr07.pdf
* [Cross-Based Local Stereo Matching Using Orthogonal Integral Images](https://ieeexplore.ieee.org/document/4811952?denied=)
