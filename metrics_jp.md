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
PSNRを計測します．  
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

## Status
* AVX/single
* Any depth/color

## example
```cpp
void testPSNR(Mat& src);
```

## Test function in OpenCP
```cpp
void testPSNR(Mat& src);
```

# getPSNR_PRECISION

```cpp
getPSNR_PRECISION(const int precision)
```

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
## Usage
return the enum name of PSNR_PRECISION.


# getPSNR_CHANNEL
```cpp
std::string getPSNR_CHANNEL(const int channel);
```
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

