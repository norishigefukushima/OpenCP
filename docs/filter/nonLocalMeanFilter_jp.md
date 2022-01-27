nonLocalMeanFilter.hpp
======================

ノンローカルミーンフィルタ Non-local means (NLM) filterとその拡張関数群

テスト関数は，guiDenoiseTest.cppの下記．
```cpp
void guiDenoiseTest(Mat& src)
```

# nonLocalMeansFilter
```cpp
void nonLocalMeansFilter(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);
void nonLocalMeansFilter(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);
```
## Usage
ノンローカルミーンフィルタを適用します．
2つの関数は，パッチサイズ（非局所パッチのサイズ）と畳み込みカーネルサイズの指定の仕方の違いで，cv::Sizeで指定したほうは，縦横サイズを変えられ長方形にできますが，intの指定は必ず正方形になります．

```cpp
cv::InputArray src,  //入力画像
cv::OutputArray dest, //出力画像
cv::Size patchWindowSize, //パッチサイズ
cv::Size kernelWindowSize, //カーネルサイズ
double sigma, //レンジ重みのσ
double powexp = 2 //レンジ重みの形状．デフォルトは2でガウス
int patchNorm = 2 //パッチ間距離のノルム
int borderType = cv::BORDER_REPLICATE//境界条件
```

重み関数は下記で計算されます．
powexp=2でガウス，powexp=１でラプラス，infinity(0で代用)でボックスになります．
デノイジング時は，ある程度ボックスに近いほうが高性能になります．

```math
w_r[i] = \exp(-\frac{(|i/\sigma|)^{\rm{powexp}}}{\rm{powexp}})
```

Githubはtex math表記ができないので，実装

```cpp
w_r[i] = exp(-pow(abs(i/σ),powexp)/powexp)
```

元論文の重み関数は，σに代わってhを使って下記で定義されてますが，この実装はexpのn乗関数として拡張してしています．
powexp=2のガウスとして定義した場合，元論文よりもパラメータを半分に設定しないと同じ出力は得られません．


```cpp
w_r[i] = exp(-i*i/(h*h))
```

また，元論文は，σを使って分散の中央付近を平らにする処理の実装も記述してあります．

```cpp
w_r[i] = exp(-max(i*i-σ*σ,0)/(h*h)) 
```

これは，概ね重みの0付近を1に強制的にセットして平らにするするための処理で，OpenCPの実装の場合powexpの次数を増やすことで対応できます．
元論文の実装は重み関数が微分不可能な関数になり，この実装は微分可能な関数になります．

## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

# jointNonLocalMeansFilter
```cpp
void jointNonLocalMeansFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);
void jointNonLocalMeansFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);
```

## USage
重みをguide画像から計算するNLMの拡張です．
第2引数にguideを追加で取る以外，`nonLocalMeansFilter`とパラメータの指定は同じです．

## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

guideとsourceを同じアドレスを持つ画像に設定しても，ガイド無しの実装のほうが速く動きます．

# patchBilateralFilter
```cpp
void patchBilateralFilter(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);
void patchBilateralFilter(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);
```

## USage
NLMに空間重みを追加したNLMの拡張です．
つまり，バイラテラルフィルタのレンジ重みをパッチで取ることに相当します．
空間重みもレンジ重みと同様にべき乗のパラメータ設定を取ることができます．
`powexp_space`を0に設定したらsigma_spaceが十分な大きささえあればNLMと動作は完全に一致します．


## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

guideとsourceを同じアドレスを持つ画像に設定しても，ガイド無しの実装のほうが速く動きます．

# jointPatchBilateralFilter
```cpp
void jointPatchBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);
void jointPatchBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);
```
## USage
重みをguide画像から計算するPBFの拡張です．
第2引数にguideを追加で取る以外，`patchBilateralFilter`とパラメータの指定は同じです．


## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

guideとsourceを同じアドレスを持つ画像に設定しても，ガイド無しの実装のほうが速く動きます．

# nonLocalMeansFilterSeparable
```cpp
void nonLocalMeansFilterSeparable(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method= SEPARABLE_METHOD::SWITCH_VH, const double alpha=0.8, const int borderType = cv::BORDER_DEFAULT);
void nonLocalMeansFilterSeparable(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);
```
## USage
セパラブルフィルタとしてnonLocalMeansFilterを近似します．
引数は，近似なしフィルタのborder指定の直前に以下の二つの追加されます．
```cpp
SEPARABLE_METHOD method= SEPARABLE_METHOD::SWICH_VH, 
double alpha = 0.8,
```

SEPARABLE_METHODは，セパラブルフィルタの動作を定義したもので，Switching dual kernel (SDK)か通常のセパラブルを指定できます．
SDKのほうが高精度です．
詳しくは，[getSeparableMethodName](./separableFilterCore_jp.md "#getSeparableMethodName")を参照してください．

パラメータの`alpha`はSDKのパラメータで，2つに分かれたフィルタの2回目のセパラブルフィルタのsigma_rangeを小さくすることで，近似精度を上げます．

## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

なお，実装が最適ではなく，セパラブル化するときに非セパラブル関数を2回呼び出す実装をしています．
画像全体を2度スキャンし，2度fork-joinをするため，キャッシュ効率，並列化効率の観点から最適ではないため，実験で高速な実装が必要な時は再実装が必要です．


# jointNonLocalMeansFilterSeparable
```cpp
void jointNonLocalMeansFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);
void jointNonLocalMeansFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);
```
## USage
jointNonLocalMeansFilterをセパラブルで近似します．
パラメータの指定は，nonLocalMeansFilterのセパラブル化と同じです．
詳細は，[nonLocalMeansFilter]("#nonLocalMeansFilter")のドキュメントを参照してください．


なお，jointフィルタの場合は，alpha=1.0とした場合，DIRECTとSWITCHは同じ動作になります．

## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

guideとsourceを同じアドレスを持つ画像に設定しても，ガイド無しの実装のほうが速く動きます．

なお，実装が最適ではなく，セパラブル化するときに非セパラブル関数を2回呼び出す実装をしています．
画像全体を2度スキャンし，2度fork-joinをするため，キャッシュ効率，並列化効率の観点から最適ではないため，実験で高速な実装が必要な時は再実装が必要です．

# patchBilateralFilterSeparable
```cpp
void patchBilateralFilterSeparable(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);
void patchBilateralFilterSeparable(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);
```
## USage
セパラブルフィルタとしてpatchBilateralFilterを近似します．
パラメータの指定は，nonLocalMeansFilterのセパラブル化と同じです．
詳細は，nonLocalMeansFilterのドキュメントを参照してください．


## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

なお，実装が最適ではなく，セパラブル化するときに非セパラブル関数を2回呼び出す実装をしています．
画像全体を2度スキャンし，2度fork-joinをするため，キャッシュ効率，並列化効率の観点から最適ではないため，実験で高速な実装が必要な時は再実装が必要です．


# jointPatchBilateralFilterSeparable
```cpp
void jointPatchBilateralFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);
void jointPatchBilateralFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);
```
## USage
jointPatchBilateralFilterSeparableをセパラブルで近似します．
パラメータの指定は，nonLocalMeansFilterのセパラブル化と同じです．
詳細は，nonLocalMeansFilterのドキュメントを参照してください．

なお，jointフィルタは，alpha=1.0とした場合，DIRECTも，SWITCHも動作は同じになります．

## Optimization
* AVX
* OpenCV parallel framework

uchar入力の場合，L1距離の計測が優位に速く動きます．
floatの場合は大差がありません．

guideとsourceを同じアドレスを持つ画像に設定しても，ガイド無しの実装のほうが速く動きます．

なお，実装が最適ではなく，セパラブル化するときに非セパラブル関数を2回呼び出す実装をしています．
画像全体を2度スキャンし，2度fork-joinをするため，キャッシュ効率，並列化効率の観点から最適ではないため，実験で高速な実装が必要な時は再実装が必要です．


# Reference

* Original Non local means filtering (NLM)
	* Antoni Buades, Bartomeu Coll, and Jean-Michel Morel, "A non-local algorithm for image denoising," in Proc. Computer Vision and Pattern Recognition (CVPR), 2005.
	* [Non-Local Means Denoising (IPOL)](http://www.ipol.im/pub/art/2011/bcm_nlm/)
* Separable NLM (switching dual kernel: SDK)
	* N. Fukushima, S. Fujita, and Y. Ishibashi, "Switching dual kernels for separable edge-preserving filtering," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1588-1592, Apr. 2015. 
	* [pdf](https://fukushima.web.nitech.ac.jp/paper/2015_icassp_fukushima.pdf)
	* [IEEE xplore](https://ieeexplore.ieee.org/document/7178238?arnumber=7178238)
	* [old code](https://github.com/norishigefukushima/Separable-Edge-Preserving-Filter)