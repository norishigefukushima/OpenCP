imshowExtension.hpp
===================
`imshow`を拡張して，使いやすくした関数群です．

# imshowNormalize
```cpp
void imshowNormalize(std::string wname, cv::InputArray src, const int norm_type = cv::NORM_MINMAX);
```
## Usage
imshowをする前にノーマライズをして8Uにキャストします．  
floatだと画素値が見えないときや，正規化したいときに使います．  
デフォルトは，最大値と最小値を0-255にマップするように`NORM_MINMAX`を使います．  
最大値だけでスケールするには（最小値をシフトしないには），`NORM_INF`を使用してください．  


# imshowScale
```cpp
void imshowScale(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);
```
## Usage
imshowをする前にスケーリング（ax+b）をして8Uにキャストします．  
内部で`convertTo(dest, CV_8U)`を呼んでからimshowしています．
floatだと画素値が見えないときに困るときによく使います．  
Normalizeと違って，デフォルトでは値を変更しないため，ただ`CV_32F`を`CV_8U`で表示したいだけの場合はこちらを使います．  

# imshowScaleAbs
```cpp
void imshowScale(std::string name, cv::InputArray src, const double alpha = 1.0, const double beta = 0.0);
```
## Usage
imshowScaleと違って引数srcをabsしてからimshowします．


# imshowResize
```cpp
void imshowResize(std::string name, cv::InputArray src, const cv::Size dsize, const double fx = 0.0, const double fy = 0.0, const int interpolation = cv::INTER_NEAREST, bool isCast8U = true);
```
## Usage
imshowをする前にリサイズします．  
画素値を維持するために，デフォルトはNearestNeighborでリサイズします．  
また，最後のオプションで強制的に8Uにキャストできます．デフォルトはオンです．

# imshowCountDown
```cpp
void imshowCountDown(std::string wname, cv::InputArray src, const int waitTime = 1000, cv::Scalar color = cv::Scalar::all(0), const int pointSize = 128, std::string fontName = "Consolas");
```
## Usage
imshowをするとともに，カウントダウンをします．  
デモンストレーションプログラム用に使用します．

