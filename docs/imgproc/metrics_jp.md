Metrics.hpp
================

# getPSNR
```cpp
double getPSNR(cv::InputArray src, 
	cv::InputArray reference, 
	const int boundingBox = 0, //bounding box to ignore boundary
	const int precision = PSNR_UP_CAST, 
	const int compare_channel = PSNR_ALL);
```
## Usage
[PSNR (Peak signal-to-noise ratio)](https://ja.wikipedia.org/wiki/%E3%83%94%E3%83%BC%E3%82%AF%E4%BF%A1%E5%8F%B7%E5%AF%BE%E9%9B%91%E9%9F%B3%E6%AF%94)を計測します．  
OpenCVの関数`cv::PSNR`とは異なり，入力の型同士が異なっても計測できます．  
また画像の劣化が例外的になりやすい外周を`boundingBox`で指定可能です．  
また，`cv::PSNR`よりも若干高速です．  
この関数は`class PSNRMetrics`のラッパーであり，何度も使用する場合はクラスを使ったほうがより高速に動作します．

`precision` で計測のビット深度を指定できます．  
デフォルトは`PSNR_UP_CAST`であり，srcかreferenceのより高い精度の値が使用されます．  
詳細は[getPSNR_PRECISION](#getPSNR_PRECISION)を参照してください．

`compare_channel` カラーの場合のPSNRの色を指定できます．  
デフォルトは`PSNR_ALL`であり，全色のMSEの合計値からPSNRを計測します．  
詳細は[getPSNR_CHANNEL](#getPSNR_CHANNEL)を参照してください．

なお，例外は以下の値が戻ります．
* 完全に一致する場合は0
* MSEがNaNの場合 -1
* MSEがInfの場合 -2

## Optimization
* AVX/single
* Any depth/color

## example
```cpp
Mat src = imread("lenna.png");
Mat ref; addNoise(src, ref, 50);
cout << getPSNR(src, ref) << endl;
cout << getPSNR(src, ref, 20) << endl;//using bounding box
cout << getPSNR(src, ref, 20, PSNR_8U) << endl;//using down cast to 8U
cout << getPSNR(src, ref, 20, PSNR_8U, PSNR_Y);//getPSNR for Y channel with YUV convertion
```

Test function in OpenCP
```cpp
void testPSNR(Mat& src);
```
# getPSNRClip
```cpp
double getPSNRClip(cv::InputArray src, 
	cv::InputArray reference, 
	double minval,
	double maxval,
	const int boundingBox = 0, //bounding box to ignore boundary
	const int precision = PSNR_UP_CAST, 
	const int compare_channel = PSNR_ALL);
```
## Usage
getPSNR関数にsrcとreferenceの値に制限値を付けるclip関数付きの実装です．
入出力の値が0-255の値に収まらない時に収まったとみなして計算するなどの用途に使います．

# getPSNR_PRECISION
```cpp
enum PSNR_PRECISION
	{
		PSNR_UP_CAST,
		PSNR_8U,
		PSNR_32F,
		PSNR_64F,
		PSNR_KAHAN_64F,

		PSNR_PRECISION_SIZE
	};
```
```cpp
string getPSNR_PRECISION(const int precision)
```

## Usage
PSNRやMSEを計算するときに，画素値をダウンキャストするためのオプションのenumです．  
ダウンキャストすることで，実際に使用するビット深度におけるPSNRを計測できます．  
`PSNR_UP_CAST`は入力，参照画素のうちデプスの優先順位(`uchar<short<int<float<double`)が高いほうに計算精度がキャストされます．  
`PSNR_8U/32F/64F`はそれぞれ`uchar`,`float`,`double`精度にキャストして計算します．  
`PSNR_KAHAN_64F`は，`double`精度の計算をKahanの精度補償アルゴリズムを用いて計算します．  
Kahanの精度補償が必要になるほどの精度はほとんどの場合でありません．

また，`getPSNR_PRECISION`でenumの名前をstring型で取得できます．


# getPSNR_CHANNEL
```cpp
enum PSNR_CHANNEL
	{
		PSNR_ALL,
		PSNR_Y,
		PSNR_B,
		PSNR_G,
		PSNR_R,

		PSNR_CHANNEL_SIZE
	};
```
```cpp
string getPSNR_CHANNEL(const int channel);
```
PSNRやMSEを計算するときに，カラー画像をどのように処理するかのオプションのenumです．  
デフォルトは，`PSNR_ALL`で，すべての画素をMSEとして計算に含めて出力します．  
`PSNR_Y`は内部でYUV変換してYの値でPSNRを測定します．  
ITU-R BT.601 / ITU-R BT.709 (1250/50/2:1)もしくは，PAL, SECAMのYの値です．  
`PSNR_B/G/R`は各チャネルのどれかでPSNRを計測します．

また，`getPSNR_CHANNEL`でenumの名前をstring型で取得できます．

なお，BGRそれぞれのPSNRを平均するオプションはありません．  
同一画像の多チャンネルのPSNRを平均する，MSEを平均するといった処理は，多くの場合で正しくありません．

# getMSE
```cpp
double getMSE(cv::InputArray src1, cv::InputArray src2const, int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
```
```cpp
double getMSE(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask);
```
`getPSNR`関数において，PSNRではなくMSEを返します．  
挙動としては，`getPSNR`関数がMSE求めたのちに，`inlineMathFunctions.hpp`内の`MSEtoPSNR`インライン関数を呼び出しています．

なお，マスク付きのMSE関数は特殊化されており，マスク無しの関数と挙動が違います．  
入力を`double`でキャストしたのち，必ずPSNR_ALLオプションで出力します．  
マスクは，計測しない画素を0にしてください．

# class PSNRMetrics
```cpp
class PSNRMetrics
{
public:
	double getMSE(cv::InputArray src, cv::InputArray ref, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	double getPSNR(cv::InputArray src, cv::InputArray ref, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	double operator()(cv::InputArray src, cv::InputArray ref, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	void setReference(cv::InputArray src, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	double getMSEPreset(cv::InputArray src, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
	double getPSNRPreset(cv::InputArray src, const int boundingBox = 0, const int precision = PSNR_UP_CAST, const int compare_channel = PSNR_ALL);
};
```
## Usage
`getPSNR`,`getMSE`のラップ元です．  
何度もPSNRを計測する場合は，こちらのクラスを使ったほうが，バッファを使いまわすため速く動作します．  
このクラス内の`getMSE`,`getPSNR`メソッドの挙動は，関数の場合と同様です．  
また`operator()`により，`getPSNR`を省略できます．  

もし，参照画像が固定の場合は，`setReference`で比較元をセットしたのちに，`getMSEPreset`や`getPSNRPreset`で入力画像だけ入れることで若干の高速化をすることが可能です．  
その場合は，`boundingBox`, `precision`, `compare_channel`の各オプションを，比較元と，入力で必ず一致させる必要があります．  
よく間違えるので注意してください．

# localPSNRMap
```cpp
void localPSNRMap(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dest, const int r, const int compare_channel, const double psnr_infinity_value = 0.0);
```
## Usage
局所ウィンドウのPSNRを計算し，画像として出力します．  
各画素の周辺のPSNRを観測するために使用します．  

`r`が局所ウィンドウの半径，compare_channelは`getPSNR`関数と同じ意味です．  
内部ですべてdoubleにアップキャストしたのちにPSNRを計算するため，`precision`のオプションはありません．
最後の引数でPSNRが無限大の時の値を指定できます．デフォルトは0.0です．  

## Optimization
* OpenCV/single
* upcast double, any color

## example
```cpp
Mat src = imread("lenna.png");
Mat ref; addNoise(src, ref, 50);
Mat dest;
localPSNRMap(src, ref, dest, 10, PSNR_ALL);
```

# guiLocalPSNRMap
```cpp
void guiLocalPSNRMap(cv::InputArray src1, cv::InputArray src2, const bool isWait = true, std::string wname = "AreaPSNR");
```
## Usage
GUI付きで`localPSNRMap`を試します．  
ヘルプはキーワードの?を押してください．

# getInacceptableRatio
```cpp
double getInacceptableRatio(cv::InputArray src, const cv::Mat& ref, const int threshold);
```
## Usage
閾値`threshold`以上の絶対値差を持つ画素数の割合を返します．  
0~100の値が出力されます．  
ステレオマッチングの精度評価で用いる，Bad Pixel指標と同じ計算式です．

## Optimization
* グレイ画像のみ対応です．カラー画像は強制的にグレイスケールに変換されます．
* 型は任意の型で動作します．ただし，内部のcvtColorがdoubleで動作しないため，doubleの入力は受け付けません．
* 高速化はされていません．

# getEntropy
```cpp
double getEntropy(cv::InputArray src, cv::InputArray mask = cv::noArray());
```
## Usage
入力画像のエントロピーが出力されます．  
また，計算する領域のマスクを追加することが可能です．
## Optimization
* 8Uか16Sのみ対応しています．
* 高速ははされていません．

# getTotalVariation
```cpp
double getTotalVariation(cv::InputArray src);
```
## Usage
入力画像のトータルバリエーションが出力されます．  
## Optimization
* 高速ははされていません．