onelineCVFunctions.hpp
======================
下記の頻出の変換を1行で書けるようにした関数群
* convert
* cvtColor
* copyMakeBorder
* split

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