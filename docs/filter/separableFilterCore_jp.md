separableFilterCore.hpp
=======================

2Dフィルタをセパラブルフィルタとして実装する場合のコア関数群です．

# string getSeparableMethodName(SEPARABLE_METHOD method)
```cpp
string getSeparableMethodName(SEPARABLE_METHOD method)
std::string getSeparableMethodName(int method)
```

## Usage
SEPARABLE_METHODの名前をstringで返します．

SEPARABLE_METHODはセパラブルフィルタによる近似方法を指定するenumで，下記から選択できます．

* Switching dual kernel (SDK) を縦横にかける実装
* Switching dual kernel (SDK) を横縦にかける実装
* 通常セパラブルを縦横にかける実装
* 通常セパラブルを横縦にかける実装

```cpp
enum class SEPARABLE_METHOD
{
	SWITCH_VH, //Switching dual kernel (SDK) vertical-filtering then horizontal-filtering
	SWITCH_HV, //Switching dual kernel (SDK) horizontal-filtering then vertical-filtering
	DIRECT_VH, //Usual separable vertical-filtering then horizontal-filtering
	DIRECT_HV  //Usual separable horizontal-filtering then vertical-filtering
};
```

# Reference

* Switching dual kernel: SDK
	* N. Fukushima, S. Fujita, and Y. Ishibashi, "Switching dual kernels for separable edge-preserving filtering," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1588-1592, Apr. 2015. 
	* [pdf](https://fukushima.web.nitech.ac.jp/paper/2015_icassp_fukushima.pdf)
	* [IEEE xplore](https://ieeexplore.ieee.org/document/7178238?arnumber=7178238)
	* [old code](https://github.com/norishigefukushima/Separable-Edge-Preserving-Filter)