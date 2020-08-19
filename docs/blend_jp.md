blend.hpp
================
2枚の画像のブレンド関数群．
guiBlendTest.cppにテスト関数がある．
```cpp
void testAlphaBlend(Mat& src1, Mat& src2);//マスク無
void testAlphaBlendMask(Mat& src1, Mat& src2);//マスク有
```

# alphaBlend
```cpp
void alphaBlend(cv::InputArray src1, cv::InputArray src2, const double alpha, cv::OutputArray dest);
void alphaBlend(cv::InputArray src1, cv::InputArray src2, cv::InputArray alpha, cv::OutputArray dest);
```
## Usage
src1とsrc2をアルファブレンドします．  
第3引数が定数値の場合と，Matなどのアルファマスクを入力する場合をサポートします．  
片方がグレイの場合はカラーに変換してブレンドします．  
OpenCVの同様の関数はaddWeightedですが，余計な処理を消しているためaddWeightedよりも高速です．  
また，addWeightedはマスクでブレンドすることはできません．  

なお，アルファマスクが8Uの時は，0-255の値を0-1だと思ってブレンドします．
## Optimization
* AVX

# alphaBlendFixedPoint
```cpp
void alphaBlendFixedPoint(cv::InputArray src1, cv::InputArray src2, const int alpha/*0-255*/, cv::OutputArray dest);
void alphaBlendFixedPoint(cv::InputArray src1, cv::InputArray src2, cv::InputArray alpha, cv::OutputArray dest);
```
## Usage
固定小数点で計算することで高速化されたalphaBlend関数です．
固定値のアルファの値は，double alpha(0.0-1.0)ではなくてint alpha(0-255)であることに注意すること．
アルファマスクは8Uのマスクしか取れない．
また，入力画像も8Uしかとることができない．
これは，浮動小数点を入力する場合は，もともと整数にキャストして整数演算するよりもそのまま浮動小数点演算したほうが高速なためである．

# guiAlphaBlend
```cpp
cv::Mat guiAlphaBlend(cv::InputArray src1, cv::InputArray src2, bool isShowImageStats = false, std::string wname = "alphaBlend");
```
## Usage
定数値のアルファブレンド関数をGUI内でコールする関数です．  
二つの入力を比較するために使います．  
デバッグで頻繁に使います．  
isShowImageStatsをtrueにすることで，入力画像の配列サイズや型，統計情報がどのようになっているかを関数実行前に実行します．  

alphaBlendのカラーの不一致への対応に加えて，入力画像の一方が8U一方が32Fの場合に３２Fにアップキャストしたのちに8Uとして表示します．  
また，戻り値で出力画像を返します．

**キーボードショートカット**

* `f`: アルファ値のフリップ
* `i`: showMatInfoのコール
* `p`: PSNRとMSEの計測
* `v`: ビデオキャプチャの開始・終了トグル
* `?`：ヘルプの表示
* `q`: 終了

# dissolveSlideBlend
```cpp
void dissolveSlideBlend(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dest, const double ratio = 0.5, const double slant_ratio = 0.4, const int direction = 0, cv::Scalar line_color = cv::Scalar::all(255), const int line_thickness = 2);
```
## Usage
2枚の画像をセパレータで分けて1つにマージします．  
例えば，下記のように関数をよぶと，1枚の画像の左半分を`img1`，右半分を`img2`として表示し，分割する境界には白いラインが入ります．  
```
dissolveSlideBlend(img1, img2, 0.5, 0.5)
```

`ratio`で分割位置を0.0から1.0で表します（0.5が中央）．  
`slant_ratio`で分割の傾き位置を0.0から1.0で表します（0.5が傾き無し）．  
`direction`が0で縦分割，1で横分割．  
`line_color`は，線分の色．  
`line_thickness`は線の太さで，0の場合は線無しになる．  

## Optimization
* OpenCV

## guiDissolveSlideBlend
```cpp
cv::Mat guiDissolveSlideBlend(cv::InputArray src1, cv::InputArray src2, std::string wname = "dissolveSlideBlend");
```
分割表示の関数をGUI内でコールする関数です．  
論文等の図で，分割表示する図を作成するために使います．  
また，分割のパラメータはstatic変数として保持されているため，前回設定したパラメータは保持されます．  
戻り値で出力画像を返します．

細かな仕様は`?`キーでヘルプがでるのでそれを見てください．  






