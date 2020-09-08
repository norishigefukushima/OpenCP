StereoBase.hpp
================

# class StereoBMSimple
処理の本体はこれ．呼び出す主要な関数は下記．
```cpp
void StereoBase::matching(Mat& leftim, Mat& rightim, Mat& destDisparityMap)
{
	computeGuideImageForAggregation(leftim);
	prefilter(leftim, rightim);
	getPixelMatchingCost(d, DSI[i]);
	if (isFeedback)addCostIterativeFeedback(DSI[i], d, destDisparityMap, feedbackFunction, feedbackClip, feedbackAmp);
	getCostAggregation(DSI[i], DSI[i], guide);
	getWTA(DSI, dest);//最小コストの値を取る
	//postprocessings
	//...
}
```
## 0. ガイド画像の処理とプレフィルタ
```cpp
//leftimをグレイスケールに変換したり，多少のフィルタを掛けたりする
computeGuideImageForAggregation(leftim);

//* カラー画像の場合RGBにスプリット．
//* X方向のSobelフィルタ min(2*preFilterCap, Sobel(x)．+preFilterCap)
//* CENSUS変換するなどのマッチングに必要な画像を生成する．
prefilter(leftim, rightim);
```

## 1. マッチングコスト計算
`getPixelMatchingCost(d, DSI[i])`内で，以下のオプションが選択可能．
出力は`uchar (CV_8U)`．
```cpp
enum PixelMatching
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

## 2. コストアグリゲーション
`getCostAggregation(DSI[i], DSI[i], guide)`では以下のアグリゲーションが可能．
```cpp
enum Aggregation
{
	Box,//ブロックマッチング
	BoxShiftable,//シフタブルブロックマッチング
	Gaussian,//ガウシアン窓のマッチング
	GaussShiftable,//ガウシアン窓のシフタブルマッチング
	Guided,//ガイデットフィルタによるマッチング．プレフィルタで生成したguide画像を使用
	CrossBasedBox,//クロスベースのボックスフィルタによるアグリゲーション．プレフィルタで生成したguide画像を使用．
	Bilateral,//バイラテラルフィルタによるマッチング．プレフィルタで生成したguide画像を使用

	Aggregation_Method_Size
};
```
## 3. 最適化・
`getWTA(DSI, dest)`でアグリゲーションしたコストの最小値をとることができる．
なお，SGM関数である`getOptScanline()`を事前に呼べばSGMになるはずだが，デバッグが不十分．

## 4. ポストフィルタ
```cpp
uniquenessFilter(minCostMap, dest);//エラーの最最小と2番目のエラーの差を見て曖昧性の高い物をはじく
subpixelInterpolation(dest, subpixMethod);//サブピクセル補間
case SUBPIXEL_NONE:
case SUBPIXEL_QUAD: //双曲線
case SUBPIXEL_LINEAR://線形

binalyWeightedRangeFilter(dest, dest, subboxWindowR, subboxRange);//レンジフィルタによるサブピクセル補間
fastLRCheck(minCostMap, dest);//1枚画像のLRチェック
minCostFilter(minCostMap, dest);//最小コストによるフィルタ：DPなど以外は役に立たないはず．
filterSpeckles(dest, 0, speckleWindowSize, speckleRange, specklebuffer);//スペックルを除去するフィルタ
```

Todo: joint nearest filterなどを突っ込む

# StereoBMSimple::gui
下記メソッドでGUIによるパラメータ調整が可能．
`Ctrl+p`でパラメータ調整バーが呼び出せる．

```cpp
void StereoBMSimple::gui(Mat& leftim, Mat& rightim, Mat& dest, StereoEval& eval)
```

## 出力のイメージ．
デプスマップ，パラメータ調整バー，状態のコンソール，サブピクセル補間のヒストグラム．
コンソールに書いてある(1)や(i)などは対応するキーボードショートカット．

<img src="docimg/outputimage_stereobase.png" width="800px">

# Reference

* BT
	* S. Birchfield and C. Tomasi. A pixel dissimilarity measure
that is insensitive to image sampling. TPAMI, 20(4):401–
406, 1998.
	* [Birchfield–Tomasi dissimilarity](https://en.wikipedia.org/wiki/Birchfield%E2%80%93Tomasi_dissimilarity)
* Pixel cost function survey
	* https://www.cs.middlebury.edu/~schar/papers/evalCosts_cvpr07.pdf
* [Cross-Based Local Stereo Matching Using Orthogonal Integral Images](https://ieeexplore.ieee.org/document/4811952?denied=)
