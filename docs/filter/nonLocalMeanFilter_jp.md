nonLocalMeanFilter.hpp
======================

ノンローカルミーンフィルタ Non-local means (NLM) filter

テスト関数は，guiDenoiseTest.cppの下記．
```cpp
void guiDenoiseTest(Mat& src)
```

# nonLocalMeansFilter
```cpp
void nonLocalMeansFilter(const cv::Mat& src, cv::Mat& dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int borderType = cv::BORDER_REPLICATE);
void nonLocalMeansFilter(const cv::Mat& src, cv::Mat& dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int borderType = cv::BORDER_REPLICATE);
```
## USage
ノンローカルミーンフィルタを適用します．2つの関数は，パッチサイズ（非局所パッチのサイズ）と畳み込みカーネルサイズの指定の仕方の違いで，cv::Sizeで指定したほうは縦横サイズを変えられます．

```cpp
const cv::Mat& src,  //入力画像
cv::Mat& dest, //出力画像
const cv::Size patchWindowSize, //パッチサイズ
const cv::Size kernelWindowSize, //カーネルサイズ
const double sigma, //レンジ重みのσ
const double powexp //レンジ重みの形状．
const int borderType = cv::BORDER_REPLICATE//境界条件
```

重み関数は下記で計算されます．
powexp=2でガウス，powexp=１でラプラス，infinityでボックスになります．
デノイジング時は，ある程度ボックスに近いほうが高性能になります．

```math
w_r[i] = \frac{-|i/\sigma|^{powexp}}{powexp}
```

元論文の重み関数は，σに代わってhを使って下記で定義されてますが，この実装はexpのn乗関数として拡張してしています．
powexp=2のガウスとして定義した場合，元論文よりもパラメータを半分に設定しないと同じ出力は得られません．


```
exp(x^2/h^2)
```

また，元論文は，σを使って分散の中央付近を平らにする処理の実装も記述してあります．

```
exp(max(x^2-σ^2,0)/h^2)
```

これは，概ね重みの0付近を1に強制的にセットして平らにするするための処理で，OpenCPの実装の場合powexpの次数を増やすことで対応できます．
元論文の実装は重み関数が微分不可能な関数になり，この実装は微分可能な関数になります．

## Optimization
* AVX (histogram operation)
* OpenCV parallel framework

# nonLocalMeansFilterL1PatchDistance
```cpp
void nonLocalMeansFilterL1PatchDistance(const cv::Mat& src, cv::Mat& dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int borderType = cv::BORDER_REPLICATE);
void nonLocalMeansFilterL1PatchDistance(const cv::Mat& src, cv::Mat& dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int borderType = cv::BORDER_REPLICATE);
```
## USage
ノンローカルミーンフィルタを適用します．
ただし，パッチ間距離をL1距離で測って高速に演算します．
何もついていないのはL2距離です．

他のパラメータは通常の物と同じです．

## Optimization
* SSE：まだ最適化していなが，AVXのL2よりも速いです．
* OpenCV parallel framework


# epsillonFilterL1PatchDistance
```cpp
//powexp=infinity in nonLocalMeansFilter
	CP_EXPORT void epsillonFilterL1PatchDistance(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, int borderType = cv::BORDER_REPLICATE);
```
修正前．
L1距離でカーネル重みをボックスにした場合の高効率実装．

# separableNonLocalMeansFilterL1PatchDistance
```cpp
CP_EXPORT void separableNonLocalMeansFilterL1PatchDistance(cv::Mat& src, cv::Mat& dest, cv::Size templeteWindowSize, cv::Size searchWindowSize, double h, double sigma = -1.0, double alpha = 1.0, int method = DUAL_KERNEL_HV, int borderType = cv::BORDER_REPLICATE);
```
修正前．
L1距離で強制的にセパラブルにNLMを書けた実装．

# Reference

* Original NLM: Antoni Buades, Bartomeu Coll, and Jean-Michel Morel, "A non-local algorithm for image denoising," in Proc. Computer Vision and Pattern Recognition (CVPR), 2005.
	* [Non-Local Means Denoising (IPOL)](http://www.ipol.im/pub/art/2011/bcm_nlm/)
* Separable NLM: N. Fukushima, S. Fujita, and Y. Ishibashi, "Switching dual kernels for separable edge-preserving filtering," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1588-1592, Apr. 2015.