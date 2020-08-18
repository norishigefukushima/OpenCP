contrast.hpp
==========
コントラスト変換をする関数群です．

# convert
```cpp
cv::Mat convert(cv::Mat& src, const int depth, const double alpha = 1.0, const double beta = 0.0);
```
## Usage
`convert`関数の戻り値を`cv::Mat`に変えただけの関数です．
下記のような型のキャストや，何倍するかなどを指定するのを1行で書きたい場合に使います．
```cpp
Mat a = convert(b, CV_8U, 5);
```
下記のように書けることの代用です．
```
Mat src = a.clone()
```

# convert
```cpp
cv::Mat cenvertCentering(cv::InputArray src, int depth, double a = 1.0, double b = 127.5);
```
## Usage
`convert`と同じ用途ですが，`convert`は`ax+b`を行い，こちらは，`a(x-b)`を行います．

# contrastSToneExp
```cpp
void contrastSToneExp(cv::InputArray src, cv::OutputArray dest, const double sigma = 30.0, const double a = 1.0, const double b = 127.5);
```
## Usage
exp関数によるSトーンカーブでコントラスト変換をします．
`x- a*gauss(x-b, sigma)(x-b)`で変換します．
`gauss(x-b, sigma)=((x-b)*(x-b)/(-2*sigma*sigma))`です．

# contrastGamma

```cpp
void contrastGamma(cv::InputArray src, cv::OutputArray dest, const double gamma)
```
## Usage
ガンマ変換をします．

# quantization
```cpp
void quantization(cv::InputArray src, cv::OutputArray dest, const int num_levels)
```
## Usage
量子化によるポスタリゼーションを行います．

# guiContrast
```cpp
cv::Mat guiContrast(InputArray src_, string wname)
```
## Usage
guiで各種コントラスト変換を行います．  
また，トーンカーブも表示します．  
戻り値は，表示画像です．

static変数で内部のパラメータは保持されています．

* `ijkl`でパラメータ`a``b`を変えられます．
* `q`で終了
* `b`でbの値を0と127でフリップします．
* `?`でヘルプです．

