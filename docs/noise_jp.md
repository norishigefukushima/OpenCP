noise.hpp
===================
ノイズ付与する関数

# addNoise
ガウシアンノイズ，ソルトアンドペッパーノイズ（ごま塩ノイズ）を付与します．

```cpp
	CP_EXPORT void addNoise(cv::InputArray src, cv::OutputArray dest, const double sigma, const double solt_papper_ratio = 0.0, const uint64 seed = 0);
	CP_EXPORT void addJPEGNoise(cv::InputArray src, cv::OutputArray dest, const int quality);
```

## Usage
* sigmaが0平均のガウシアンノイズの標準偏差
* solt_papper_ratioがソルトアンドペッパーノイズの混入率
* これが0の場合（デフォルト） ，ソルトアンドペッパーノイズの計算は省略される．
* 最後の引数でランダムシードを指定可能（再現実験用）
# addJPEGNoise
JPEGノイズを付与します．

```cpp
	CP_EXPORT void addNoise(cv::InputArray src, cv::OutputArray dest, const double sigma, const double solt_papper_ratio = 0.0, const uint64 seed = 0);
	CP_EXPORT void addJPEGNoise(cv::InputArray src, cv::OutputArray dest, const int quality);
```
* qualityでJPEGの劣化具合を指定可能
