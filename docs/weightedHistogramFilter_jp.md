weightedHistogramFilter.hpp
================
Weighted Mode FilterやWeighted Median Filterなどの，ヒストグラムベースのエッジ保存平滑化フィルタ．

テスト関数は以下の2種類
```cpp
//disparity map test
void testWeightedHistogramFilterDisparity()
//RGB/Gray image test
void testWeightedHistogramFilter(Mat& src_, Mat& guide_)
```

# weightedHistogramFilter
```cpp
void weightedHistogramFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
```
## USage
画像中のローカルヒストグラムを操作するエッジ保存平滑化フィルタです．
下記オプションで指定することで，
* weightedModeFilter (WMF)
* weightedMedianFilter (WMF)

として動きます．  
また，ヒストグラムにデータを加算する際にオリジナルの実装のガウス関数に加えて，双曲線や，ハットカーネル（Linear），インパルス応答を加えています．
`WHF_HISTOGRAM_WEIGHT_FUNCTION`を指定してください．

重み取得は，元の`BILATERAL`重みに加えて，ただの`BOX`や`GAUSSIAN`も追加しています．
`WHF_OPERATION`を指定することで，これらのMODEとMEDIANを計算可能です．

また，最後の`mask`指定で，マスクがではない値を持つ所だけ計算することが可能です．
このフィルタを重いため，必要なところだけ計算する場合に使います．
maskは`uchar`で指定してください．

```cpp
enum class WHF_HISTOGRAM_WEIGHT_FUNCTION
{
	IMPULSE,
	LINEAR,
	QUADRIC,
	GAUSSIAN,//original paper
	SIZE
};
enum WHF_OPERATION
{
	BOX_MODE,
	GAUSSIAN_MODE,
	BILATERAL_MODE,//original paper
	BOX_MEDIAN,
	GAUSSIAN_MEDIAN,
	BILATERAL_MEDIAN,
	SIZE
};
std::string getWHFHistogramWeightName(const WHF_HISTOGRAM_WEIGHT_FUNCTION method);
std::string getWHFOperationName(const WHF_OPERATION method);
```

## Optimization
* AVX (histogram operation)
* Schalar (weight computing)
* OpenMP

ヒストグラムに入力する重み計算（バイラテラルフィルタの重みを計算する場所）が，まだベクトル化されていません．
（うまく実装をしないと，キャッシュ効率が下がり性能向上が見込めないため）

# weightedModeFilter/weightedMedianFilter
WMFとしてコールするためのラッパー関数．

# weightedWeightedHistogramFilter
```cpp
void weightedWeightedHistogramFilter(cv::InputArray src, cv::InputArray weight, cv::InputArray guide, cv::OutputArray dst, const int r, const double sigmaColor, const double sigmaSpace, const double sigmaHistogram, const WHF_HISTOGRAM_WEIGHT_FUNCTION weightFunctionType, const WHF_OPERATION method, const int borderType = cv::BORDER_DEFAULT, cv::InputArray mask = cv::noArray());
```
## Uasge
重み付きの重み付きヒストグラムフィルタ．
第2引数に`weight`が追加された物です．
この`weight`は`float`の値をとり，重要な場所に大きな値，重要ではない場所には小さな値を入れることで，重み付きのフィルタとして働きます．
また，不必要な画素の位置では，`0`を入力することで，補間として働きます．

## Optimization
* AVX (histogram operation)
* Schalar (weight computing)
* OpenMP

# weightedWeightedModeFilter/weightedWeightedMedianFilter
WMFにweightを付けたものです．

# Reference
* D. Min, J. Lu, and M. N. Do. "Depth video enhancement based on weighted mode filtering." IEEE Transactions on Image Processing 21(3), pp. 1176-1190, 2011.

* A. Ishikawa, N. Fukushima, and H. Tajima, "Halide Implementation of Weighted Median Filter," in Proc. International Workshop on Advanced Image Technology (IWAIT), Jan. 2020.